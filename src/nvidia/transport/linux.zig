//! Linux Transport: the kernel-RM backend. Talks to the NVIDIA open kernel
//! module via the user-accessible /dev/nvidiactl + /dev/nvidia* ioctl/mmap path
//! (NO root). Holds the control fd, the detected driver version/ABI, and the
//! monotonic RM handle counter. The OS-primitive RM operations (alloc/free,
//! control, map/unmap, device bring-up) live here; the higher-level RM ops stay
//! on Client and compose these.

const std = @import("std");
const ioctl = @import("../ioctl.zig");
const sdk = @import("../sdk.zig");
const version = @import("../version.zig");
const types = @import("types.zig");

const Error = types.Error;
const Device = types.Device;
const Memory = types.Memory;
const Mapping = types.Mapping;
const GpuMapping = types.GpuMapping;
const Channel = types.Channel;
const control_device = types.control_device;
const owner_tag = types.owner_tag;

pub const Transport = struct {
    fd: std.posix.fd_t,
    /// Driver version, detected at open time.
    driver_version: version.Version,
    /// ABI generation selected from driver_version; switch on this for
    /// version-specific RM layouts/commands.
    abi: version.Abi,
    /// Monotonic source of unique RM object handles within this client.
    next_handle: sdk.NvHandle = 0xc1d00000,

    /// Open /dev/nvidiactl and detect the driver version (so version-specific
    /// ABI can be selected before any further RM calls).
    pub fn open() Error!Transport {
        const rc = std.os.linux.open(control_device, .{ .ACCMODE = .RDWR }, 0);
        switch (std.os.linux.errno(rc)) {
            .SUCCESS => {},
            else => return error.OpenFailed,
        }
        const fd: std.posix.fd_t = @intCast(rc);
        errdefer _ = std.os.linux.close(fd);

        const params = try checkVersionFd(fd, sdk.RmApiVersion.CMD_QUERY, "");
        const end = std.mem.indexOfScalar(u8, &params.version_string, 0) orelse params.version_string.len;
        const v = version.Version.parse(params.version_string[0..end]) orelse return error.BadVersion;
        return .{ .fd = fd, .driver_version = v, .abi = version.Abi.fromVersion(v) };
    }

    pub fn deinit(self: Transport) void {
        _ = std.os.linux.close(self.fd);
    }

    pub fn newHandle(self: *Transport) sdk.NvHandle {
        self.next_handle += 1;
        return self.next_handle;
    }

    /// NV_ESC_CHECK_VERSION_STR. CMD_QUERY makes the kernel fill in its own
    /// version string; CMD_STRICT/RELAXED check the provided one (reply tells
    /// whether it was RECOGNIZED).
    pub fn checkVersion(self: Transport, cmd: sdk.NvU32, ver: []const u8) Error!sdk.RmApiVersion {
        return checkVersionFd(self.fd, cmd, ver);
    }

    /// NV_ESC_RM_ALLOC (NVOS21): allocate `h_class` under `h_parent` in client
    /// `h_root`, with `h_new` as the requested handle. `params`/`params_size` is
    /// the class-specific alloc param block (null/0 for classes that take none).
    ///
    /// Returns the ACTUAL handle: `h_object_new` is in/out - for the root client
    /// the RM assigns its own handle and writes it back, so the caller must use
    /// the returned value (not the hint) as the parent for child objects.
    pub fn rmAlloc(
        self: *Transport,
        h_root: sdk.NvHandle,
        h_parent: sdk.NvHandle,
        h_new: sdk.NvHandle,
        h_class: sdk.NvV32,
        params: ?*const anyopaque,
        params_size: sdk.NvU32,
    ) Error!sdk.NvHandle {
        var p = sdk.Os21Params{
            .h_root = h_root,
            .h_object_parent = h_parent,
            .h_object_new = h_new,
            .h_class = h_class,
            .p_alloc_parms = if (params) |pp| @intCast(@intFromPtr(pp)) else 0,
            .params_size = params_size,
            .status = 0,
        };
        const req = ioctl.iowr(ioctl.NV_ESC_RM_ALLOC, sdk.Os21Params);
        // RM reclaims freed objects (channels, VA mappings, memory) ASYNCHRONOUSLY, so
        // under rapid alloc/free churn a fresh alloc can transiently fail with
        // NV_ERR_INSUFFICIENT_RESOURCES (0x36) while the kernel is still draining recently
        // freed ones. Retry a bounded handful of times with a short backoff: a genuine
        // exhaustion still fails (after ~25 ms total), but a transient shortage clears as
        // reclamation catches up. This keeps a long-running app (and the GPU test suite)
        // from hard-failing on churn instead of a momentary, self-correcting condition.
        const insufficient_resources: sdk.NvU32 = 0x36;
        var attempt: u32 = 0;
        while (true) : (attempt += 1) {
            // Reset the in/out fields each attempt: a failed ioctl may have written back
            // h_object_new (and status), so a retry must restart from the requested handle.
            p.status = 0;
            p.h_object_new = h_new;
            const rc = std.os.linux.ioctl(self.fd, req, @intFromPtr(&p));
            switch (std.os.linux.errno(rc)) {
                .SUCCESS => {},
                else => return error.IoctlFailed,
            }
            if (p.status == 0) return p.h_object_new;
            if (p.status != insufficient_resources or attempt >= 8) return error.RmAllocFailed;
            var ts = std.os.linux.timespec{ .sec = 0, .nsec = @intCast(@as(u64, 200_000) << @intCast(@min(attempt, 7))) }; // 0.2ms .. ~25ms
            _ = std.os.linux.nanosleep(&ts, &ts);
        }
    }

    /// NV_ESC_RM_FREE: free an RM object.
    pub fn rmFree(self: *Transport, h_root: sdk.NvHandle, h_parent: sdk.NvHandle, h_object: sdk.NvHandle) void {
        var p = sdk.Os00Params{ .h_root = h_root, .h_object_parent = h_parent, .h_object_old = h_object, .status = 0 };
        const req = ioctl.iowr(ioctl.NV_ESC_RM_FREE, sdk.Os00Params);
        _ = std.os.linux.ioctl(self.fd, req, @intFromPtr(&p));
    }

    /// NV_ESC_CARD_INFO: enumerate the GPUs the driver knows about.
    fn cardInfo(self: *Transport) Error![sdk.NV_MAX_DEVICES]sdk.CardInfo {
        var cards = [_]sdk.CardInfo{std.mem.zeroes(sdk.CardInfo)} ** sdk.NV_MAX_DEVICES;
        const req = std.os.linux.IOCTL.IOWR(ioctl.NV_IOCTL_MAGIC, @intCast(ioctl.NV_ESC_CARD_INFO), [sdk.NV_MAX_DEVICES]sdk.CardInfo);
        const rc = std.os.linux.ioctl(self.fd, req, @intFromPtr(&cards));
        switch (std.os.linux.errno(rc)) {
            .SUCCESS => {},
            else => return error.IoctlFailed,
        }
        return cards;
    }

    /// NV_ESC_ATTACH_GPUS_TO_FD: attach a GPU (by gpu_id) to the control fd. The
    /// kernel reads size/4 ids; a trailing 0 terminates the list.
    fn attachGpu(self: *Transport, gpu_id: sdk.NvU32) Error!void {
        var ids = [_]sdk.NvU32{ gpu_id, 0 };
        const req = std.os.linux.IOCTL.IOWR(ioctl.NV_IOCTL_MAGIC, @intCast(ioctl.NV_ESC_ATTACH_GPUS_TO_FD), [2]sdk.NvU32);
        const rc = std.os.linux.ioctl(self.fd, req, @intFromPtr(&ids));
        switch (std.os.linux.errno(rc)) {
            .SUCCESS => {},
            else => return error.IoctlFailed,
        }
    }

    /// Open /dev/nvidia{minor} and NV_ESC_REGISTER_FD it against the control fd.
    /// Required before NV01_DEVICE_0 alloc is permitted (else INSUFFICIENT_PERMISSIONS).
    fn openGpuNode(self: *Transport, minor: sdk.NvU32) Error!std.posix.fd_t {
        var buf: [32]u8 = undefined;
        const path = std.fmt.bufPrintZ(&buf, "/dev/nvidia{d}", .{minor}) catch return error.OpenFailed;
        const rc = std.os.linux.open(path.ptr, .{ .ACCMODE = .RDWR }, 0);
        switch (std.os.linux.errno(rc)) {
            .SUCCESS => {},
            else => return error.OpenFailed,
        }
        const node_fd: std.posix.fd_t = @intCast(rc);
        var reg = sdk.RegisterFd{ .ctl_fd = self.fd };
        const req = std.os.linux.IOCTL.IOWR(ioctl.NV_IOCTL_MAGIC, @intCast(ioctl.NV_ESC_REGISTER_FD), sdk.RegisterFd);
        const reg_rc = std.os.linux.ioctl(node_fd, req, @intFromPtr(&reg));
        switch (std.os.linux.errno(reg_rc)) {
            .SUCCESS => {},
            else => {
                _ = std.os.linux.close(node_fd);
                return error.IoctlFailed;
            },
        }
        return node_fd;
    }

    /// Bring up the `index`-th GPU end to end: register its node, attach it,
    /// then allocate the root client, device, and subdevice. The full sequence
    /// required by the open kernel modules.
    pub fn allocDevice(self: *Transport, index: u32) Error!Device {
        // Find the index-th valid GPU from CARD_INFO.
        const cards = try self.cardInfo();
        var card: ?sdk.CardInfo = null;
        var seen: u32 = 0;
        for (cards) |entry| {
            if (entry.valid == 0) continue;
            if (seen == index) {
                card = entry;
                break;
            }
            seen += 1;
        }
        const gpu = card orelse return error.NoDevice; // no GPU at this index

        // Register the per-GPU node and attach the GPU to the control fd.
        const node_fd = try self.openGpuNode(gpu.minor_number);
        errdefer _ = std.os.linux.close(node_fd);
        try self.attachGpu(gpu.gpu_id);

        // Root client (NV01_ROOT, no params). RM assigns + returns the handle.
        const client = try self.rmAlloc(0, 0, self.newHandle(), sdk.NV01_ROOT, null, 0);
        errdefer self.rmFree(0, 0, client);

        // Device under the client.
        var dp = sdk.Nv0080AllocParameters{
            .device_id = index,
            .h_client_share = sdk.NV01_NULL_OBJECT,
            .h_target_client = 0,
            .h_target_device = 0,
            .flags = 0,
            .va_space_size = 0,
            .va_start_internal = 0,
            .va_limit_internal = 0,
            .va_mode = 0,
        };
        const device = try self.rmAlloc(client, client, self.newHandle(), sdk.NV01_DEVICE_0, &dp, @sizeOf(sdk.Nv0080AllocParameters));

        // Subdevice 0 under the device.
        var sp = sdk.Nv2080AllocParameters{ .sub_device_id = 0 };
        const subdevice = try self.rmAlloc(client, device, self.newHandle(), sdk.NV20_SUBDEVICE_0, &sp, @sizeOf(sdk.Nv2080AllocParameters));

        return .{ .node_fd = node_fd, .minor = gpu.minor_number, .gpu_id = gpu.gpu_id, .client = client, .device = device, .subdevice = subdevice };
    }

    /// Free a Device's RM objects (subdevice, device, client) and close its node fd.
    pub fn freeDevice(self: *Transport, dev: Device) void {
        self.rmFree(dev.client, dev.device, dev.subdevice);
        self.rmFree(dev.client, dev.client, dev.device);
        self.rmFree(0, 0, dev.client);
        _ = std.os.linux.close(dev.node_fd);
    }

    /// Open a fresh fd dedicated to one mmap context. System memory maps via a
    /// new control-node fd; VRAM via a new device-node fd registered to the
    /// control fd. (The RM keeps ONE mmap context per fd, so each concurrent
    /// mapping needs its own.)
    fn openMappingFd(self: *Transport, dev: Device, location: Memory.Location) Error!std.posix.fd_t {
        switch (location) {
            .system, .system_wc => {
                const rc = std.os.linux.open(control_device, .{ .ACCMODE = .RDWR }, 0);
                switch (std.os.linux.errno(rc)) {
                    .SUCCESS => {},
                    else => return error.OpenFailed,
                }
                return @intCast(rc);
            },
            .vram => return self.openGpuNode(dev.minor),
        }
    }

    /// CPU-map a memory object: NV_ESC_RM_MAP_MEMORY sets up an mmap context on a
    /// dedicated fd, then mmap returns the pointer. System memory maps via the
    /// control node, VRAM via the device node. Supports many concurrent mappings.
    pub fn mapMemory(self: *Transport, dev: Device, mem: Memory) Error!Mapping {
        const fd = try self.openMappingFd(dev, mem.location);
        errdefer _ = std.os.linux.close(fd);

        var w = sdk.Nvos33WithFd{
            .params = .{
                .h_client = dev.client,
                .h_device = dev.subdevice,
                .h_memory = mem.handle,
                .offset = 0,
                .length = mem.size,
                .p_linear_address = 0,
                .status = 0,
                .flags = 0, // NVOS33_FLAGS_ACCESS_READ_WRITE
            },
            .fd = fd,
        };
        const req = std.os.linux.IOCTL.IOWR(ioctl.NV_IOCTL_MAGIC, @intCast(ioctl.NV_ESC_RM_MAP_MEMORY), sdk.Nvos33WithFd);
        const rc = std.os.linux.ioctl(self.fd, req, @intFromPtr(&w));
        switch (std.os.linux.errno(rc)) {
            .SUCCESS => {},
            else => return error.IoctlFailed,
        }
        if (w.params.status != 0) return error.MapFailed;

        const m = std.os.linux.mmap(null, mem.size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
        switch (std.os.linux.errno(m)) {
            .SUCCESS => {},
            else => return error.MapFailed,
        }
        const ptr: [*]u8 = @ptrFromInt(m);
        return .{ .bytes = ptr[0..mem.size], .fd = fd };
    }

    /// Unmap a mapping returned by mapMemory (and close its dedicated fd).
    pub fn unmapMemory(self: *Transport, mapping: Mapping) void {
        _ = self;
        _ = std.os.linux.munmap(@ptrCast(mapping.bytes.ptr), mapping.bytes.len);
        _ = std.os.linux.close(mapping.fd);
    }

    /// NV_ESC_RM_CONTROL (NVOS54): invoke control `cmd` on `h_object` with the
    /// command-specific `params` block (in/out).
    pub fn control(self: *Transport, dev: Device, h_object: sdk.NvHandle, cmd: sdk.NvU32, params: ?*anyopaque, params_size: sdk.NvU32) Error!void {
        var p = sdk.Os54Params{
            .h_client = dev.client,
            .h_object = h_object,
            .cmd = cmd,
            .flags = 0,
            .params = if (params) |pp| @intCast(@intFromPtr(pp)) else 0,
            .params_size = params_size,
            .status = 0,
        };
        const req = ioctl.iowr(ioctl.NV_ESC_RM_CONTROL, sdk.Os54Params);
        const rc = std.os.linux.ioctl(self.fd, req, @intFromPtr(&p));
        switch (std.os.linux.errno(rc)) {
            .SUCCESS => {},
            else => return error.IoctlFailed,
        }
        if (p.status != 0) return error.ControlFailed;
    }

    /// Bind a physical memory object into `vaspace` at GPU virtual address
    /// `gpu_va`, so the GPU can address it (e.g. a channel's GPFIFO/pushbuffer).
    /// Carves an NV01_MEMORY_VIRTUAL range then RM_MAP_MEMORY_DMA with a fixed
    /// offset (dmaOffset is an input for a virtual hDma; VA 0 is reserved).
    pub fn mapToGpu(self: *Transport, dev: Device, vaspace: sdk.NvHandle, mem: Memory, gpu_va: u64) Error!GpuMapping {
        return self.mapToGpuPaged(dev, vaspace, mem, gpu_va, sdk.dma_flags.PAGE_SIZE_DEFAULT);
    }

    /// mapToGpu with an explicit page size. A BLOCK_LINEAR surface (a ZETA depth
    /// buffer) must be mapped with BIG GPU pages - the block-linear PTE kind is only
    /// valid on big pages, so a small-page (4 KB) mapping faults the depth draw with
    /// an Xid 69 Class Error (ErrorCode 0x9c). `page_size_flag` is an NVOS46 PAGE_SIZE_*.
    pub fn mapToGpuPaged(self: *Transport, dev: Device, vaspace: sdk.NvHandle, mem: Memory, gpu_va: u64, page_size_flag: sdk.NvU32) Error!GpuMapping {
        // Reserve [gpu_va, gpu_va+size) in the VA space.
        var vp = sdk.VirtMemAllocParams{ .offset = gpu_va, .limit = gpu_va + mem.size - 1, .h_vaspace = vaspace };
        const virtual = try self.rmAlloc(dev.client, dev.device, self.newHandle(), sdk.NV01_MEMORY_VIRTUAL, &vp, @sizeOf(sdk.VirtMemAllocParams));
        errdefer self.rmFree(dev.client, dev.device, virtual);

        var p = sdk.Os46Params{
            .h_client = dev.client,
            .h_device = dev.device,
            .h_dma = virtual,
            .h_memory = mem.handle,
            .offset = 0,
            .length = mem.size,
            .flags = sdk.dma_flags.ACCESS_READ_WRITE | sdk.dma_flags.DMA_OFFSET_FIXED | page_size_flag,
            .flags2 = 0,
            .kind_override = 0,
            .dma_offset = gpu_va,
            .status = 0,
        };
        const req = ioctl.iowr(ioctl.NV_ESC_RM_MAP_MEMORY_DMA, sdk.Os46Params);
        const rc = std.os.linux.ioctl(self.fd, req, @intFromPtr(&p));
        switch (std.os.linux.errno(rc)) {
            .SUCCESS => {},
            else => return error.IoctlFailed,
        }
        if (p.status != 0) return error.MapFailed;
        return .{ .virtual = virtual, .gpu_va = p.dma_offset, .size = mem.size };
    }

    /// Release a GPU-VA mapping (frees the NV01_MEMORY_VIRTUAL object, which
    /// unmaps the physical memory). The physical Memory is freed separately.
    pub fn unmapFromGpu(self: *Transport, dev: Device, mapping: GpuMapping) void {
        self.rmFree(dev.client, dev.device, mapping.virtual);
    }
};

fn checkVersionFd(fd: std.posix.fd_t, cmd: sdk.NvU32, ver: []const u8) Error!sdk.RmApiVersion {
    var p = sdk.RmApiVersion{
        .cmd = cmd,
        .reply = 0,
        .version_string = [_]u8{0} ** sdk.RmApiVersion.STRING_LENGTH,
    };
    const n = @min(ver.len, sdk.RmApiVersion.STRING_LENGTH - 1);
    @memcpy(p.version_string[0..n], ver[0..n]);
    const req = ioctl.iowr(ioctl.NV_ESC_CHECK_VERSION_STR, sdk.RmApiVersion);
    const rc = std.os.linux.ioctl(fd, req, @intFromPtr(&p));
    switch (std.os.linux.errno(rc)) {
        .SUCCESS => {},
        else => return error.IoctlFailed,
    }
    return p;
}
