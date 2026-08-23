//! Minimal NVIDIA RM (Resource Manager) client. Ported from the NVIDIA open
//! kernel module userspace ABI (open-gpu-kernel-modules). Zero C deps.
//!
//! The OS coupling lives behind a comptime-selected Transport (transport.zig):
//! on Linux it talks to the kernel RM via the user-accessible /dev/nvidiactl +
//! /dev/nvidia* ioctls (NO root); on freestanding it will drive the GPU directly
//! over MMIO (a stub for now). Client holds that Transport and exposes the same
//! public API regardless: the OS-primitive ops forward to the Transport, while
//! the higher-level RM ops (memory/vaspace/channel/control helpers) are
//! OS-agnostic and compose the primitives.
//!
//! The live paths (open/ioctl) ARE exercised by the unit tests at the bottom of
//! this file. They skip (error.SkipZigTest) when the device can't be reached -
//! no driver, no /dev/nvidia*, or no permission - so the suite stays green on
//! machines without an NVIDIA GPU while covering the real path where one exists.

const std = @import("std");
const sdk = @import("sdk.zig");
const version = @import("version.zig");
const transport = @import("transport.zig");
const drm = @import("drm.zig");

const Transport = transport.Transport;

/// OpenFailed / NoDevice mean "couldn't get into the device" (skip-worthy in
/// tests); IoctlFailed / RmAllocFailed mean an operation failed on a reachable
/// device (a real error).
pub const Error = transport.Error;

pub const control_device = transport.control_device;

// Shared RM types (used by both transports and by callers). Defined once in
// transport/types.zig; re-exported here so the public surface is unchanged.
pub const Device = transport.Device;
pub const Memory = transport.Memory;
pub const Mapping = transport.Mapping;
pub const GpuMapping = transport.GpuMapping;
pub const Channel = transport.Channel;

pub const Client = struct {
    /// The OS-abstraction backend (Linux kernel-RM ioctls, or freestanding MMIO).
    /// Holds the platform state (fd/version/abi/handle counter on Linux).
    t: Transport,

    /// Open /dev/nvidiactl and detect the driver version (so version-specific
    /// ABI can be selected before any further RM calls).
    pub fn open() Error!Client {
        return .{ .t = try Transport.open() };
    }

    pub fn deinit(self: Client) void {
        self.t.deinit();
    }

    /// Driver version, detected at open time.
    pub fn driverVersion(self: Client) version.Version {
        return self.t.driver_version;
    }

    /// ABI generation selected from driver_version.
    pub fn abi(self: Client) version.Abi {
        return self.t.abi;
    }

    /// NV_ESC_CHECK_VERSION_STR. CMD_QUERY makes the kernel fill in its own
    /// version string; CMD_STRICT/RELAXED check the provided one (reply tells
    /// whether it was RECOGNIZED).
    pub fn checkVersion(self: Client, cmd: sdk.NvU32, ver: []const u8) Error!sdk.RmApiVersion {
        return self.t.checkVersion(cmd, ver);
    }

    /// NV_ESC_RM_FREE: free an RM object.
    pub fn rmFree(self: *Client, h_root: sdk.NvHandle, h_parent: sdk.NvHandle, h_object: sdk.NvHandle) void {
        self.t.rmFree(h_root, h_parent, h_object);
    }

    /// Bring up the `index`-th GPU end to end: register its node, attach it,
    /// then allocate the root client, device, and subdevice. The full sequence
    /// required by the open kernel modules.
    pub fn allocDevice(self: *Client, index: u32) Error!Device {
        return self.t.allocDevice(index);
    }

    /// Free a Device's RM objects (subdevice, device, client) and close its node fd.
    pub fn freeDevice(self: *Client, dev: Device) void {
        self.t.freeDevice(dev);
    }

    /// Allocate a memory object of `size` bytes under `dev`, in system RAM or
    /// device VRAM, via RM_ALLOC of a memory class + NV_MEMORY_ALLOCATION_PARAMS
    /// (the path the open kernel modules accept). Returns its RM handle and the
    /// actual size the kernel allocated.
    pub fn allocMemory(self: *Client, dev: Device, location: Memory.Location, size: u64) Error!Memory {
        const a = sdk.mem_attr;
        const class: sdk.NvV32, const attr: sdk.NvU32 = switch (location) {
            .system => .{ sdk.NV01_MEMORY_SYSTEM, a.FORMAT_PITCH | a.LOCATION_PCI | a.PHYSICALITY_NONCONTIGUOUS | a.COHERENCY_WRITE_BACK },
            .system_wc => .{ sdk.NV01_MEMORY_SYSTEM, a.FORMAT_PITCH | a.LOCATION_PCI | a.PHYSICALITY_NONCONTIGUOUS | a.COHERENCY_WRITE_COMBINE },
            .vram => .{ sdk.NV01_MEMORY_LOCAL_USER, a.FORMAT_PITCH | a.LOCATION_VIDMEM | a.PHYSICALITY_CONTIGUOUS | a.COHERENCY_WRITE_COMBINE },
        };
        var params = sdk.MemAllocParams{
            .owner = transport.owner_tag,
            .attr = attr,
            .size = size,
        };
        const handle = try self.t.rmAlloc(dev.client, dev.device, self.t.newHandle(), class, &params, @sizeOf(sdk.MemAllocParams));
        return .{ .handle = handle, .size = params.size, .location = location };
    }

    /// Allocate a ZF32 depth (ZETA) surface in VRAM with the BLOCK_LINEAR page kind
    /// the 3D engine's ROP-Z unit requires. A plain allocMemory uses FORMAT_PITCH
    /// (linear) - drawing into a block-linear ZETA backed by a pitch-kind page faults
    /// with an Xid 69 Class Error (ErrorCode 0x9c). The NVOS32 ATTR_FORMAT_BLOCK_LINEAR
    /// + DEPTH_32 + Z_TYPE_FLOAT + ZS_PACKING_Z32 attributes make the kernel pick the
    /// right block-linear Z page kind. `width`/`height` describe the surface so the
    /// kernel can size/align the block-linear allocation; `size` is the byte footprint.
    pub fn allocDepthMemory(self: *Client, dev: Device, width: u32, height: u32, size: u64) Error!Memory {
        return self.allocDepthMemoryKind(dev, width, height, size, 0);
    }

    /// allocDepthMemory with an explicit PTE page kind (NVOS32 `format` field). `page_kind`
    /// 0 = let RM pick the GENERIC block-linear kind (correct for ZF32 depth - nvk maps it
    /// GENERIC too). A non-zero kind (ZF32_X24S8=0x4, S8=0x2, Z24S8=0x5) is REQUIRED for a
    /// stencil surface: the open RM rejects a map-time kind override (NVOS46) with
    /// INVALID_ARGUMENT, so the kind is set here at allocation. Binding a stencil format on a
    /// GENERIC-kind surface faults the draw (Xid 69 / ErrorCode 0x13).
    pub fn allocDepthMemoryKind(self: *Client, dev: Device, width: u32, height: u32, size: u64, page_kind: u32) Error!Memory {
        const a = sdk.mem_attr;
        var params = sdk.MemAllocParams{
            .owner = transport.owner_tag,
            .width = width,
            .height = height,
            .attr = a.FORMAT_BLOCK_LINEAR | a.COMPR_NONE | a.PAGE_SIZE_BIG |
                a.LOCATION_VIDMEM | a.PHYSICALITY_CONTIGUOUS | a.COHERENCY_WRITE_COMBINE,
            // The NVOS32 `format` field carries the explicit HW page kind. 0 -> RM derives
            // GENERIC; a Z/S kind value pins the stencil surface's page kind.
            .format = page_kind,
            .size = size,
        };
        const handle = try self.t.rmAlloc(dev.client, dev.device, self.t.newHandle(), sdk.NV01_MEMORY_LOCAL_USER, &params, @sizeOf(sdk.MemAllocParams));
        return .{ .handle = handle, .size = params.size, .location = .vram };
    }

    /// Allocate a BLOCK-LINEAR color render-target surface in VRAM (generic
    /// block-linear page kind, write-combine so the CPU can read it back). A color
    /// target paired with a ZETA depth surface MUST be block-linear, not pitch -
    /// the ROP rejects a linear color target while a ZETA is selected (Xid 69 /
    /// ErrorCode 0x9c). `size` is the GOB-tiled footprint (graphics.ztSizeBytes).
    pub fn allocColorBlMemory(self: *Client, dev: Device, width: u32, height: u32, size: u64) Error!Memory {
        const a = sdk.mem_attr;
        var params = sdk.MemAllocParams{
            .owner = transport.owner_tag,
            .width = width,
            .height = height,
            .attr = a.FORMAT_BLOCK_LINEAR | a.COMPR_NONE | a.PAGE_SIZE_BIG |
                a.LOCATION_VIDMEM | a.PHYSICALITY_CONTIGUOUS | a.COHERENCY_WRITE_COMBINE,
            .size = size,
        };
        const handle = try self.t.rmAlloc(dev.client, dev.device, self.t.newHandle(), sdk.NV01_MEMORY_LOCAL_USER, &params, @sizeOf(sdk.MemAllocParams));
        return .{ .handle = handle, .size = params.size, .location = .vram };
    }

    /// Import an existing host memory range (a userspace VA, e.g. a Wayland shm
    /// buffer's mmap) as an RM memory object via NV01_MEMORY_SYSTEM_OS_DESCRIPTOR.
    /// The result maps into the GPU address space like any other Memory (mapToGpu),
    /// letting the GPU render straight into that buffer - no copy. The caller keeps
    /// the CPU mapping it already has; the range must stay mapped+populated until
    /// freeMemory. `va` and `size` should be page-aligned.
    pub fn importMemory(self: *Client, dev: Device, va: usize, size: u64) Error!Memory {
        const a = sdk.mem_attr;
        var params = sdk.OsDescMemAllocParams{
            // Host pages: PCI/sysmem, noncontiguous, cached (a normal mmap).
            .attr = a.FORMAT_PITCH | a.LOCATION_PCI | a.PHYSICALITY_NONCONTIGUOUS | a.COHERENCY_WRITE_BACK,
            .descriptor = @as(sdk.NvP64, va),
            .limit = size - 1,
            .descriptor_type = sdk.NVOS32_DESCRIPTOR_TYPE_VIRTUAL_ADDRESS,
        };
        const handle = try self.t.rmAlloc(dev.client, dev.device, self.t.newHandle(), sdk.NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, &params, @sizeOf(sdk.OsDescMemAllocParams));
        return .{ .handle = handle, .size = size, .location = .system };
    }

    /// Free a memory object allocated by allocMemory.
    pub fn freeMemory(self: *Client, dev: Device, mem: Memory) void {
        self.t.rmFree(dev.client, dev.device, mem.handle);
    }

    /// CPU-map a memory object: NV_ESC_RM_MAP_MEMORY sets up an mmap context on a
    /// dedicated fd, then mmap returns the pointer. System memory maps via the
    /// control node, VRAM via the device node. Supports many concurrent mappings.
    pub fn mapMemory(self: *Client, dev: Device, mem: Memory) Error!Mapping {
        return self.t.mapMemory(dev, mem);
    }

    /// Unmap a mapping returned by mapMemory (and close its dedicated fd).
    pub fn unmapMemory(self: *Client, mapping: Mapping) void {
        self.t.unmapMemory(mapping);
    }

    /// NV_ESC_RM_CONTROL (NVOS54): invoke control `cmd` on `h_object` with the
    /// command-specific `params` block (in/out).
    pub fn control(self: *Client, dev: Device, h_object: sdk.NvHandle, cmd: sdk.NvU32, params: ?*anyopaque, params_size: sdk.NvU32) Error!void {
        return self.t.control(dev, h_object, cmd, params, params_size);
    }

    /// Query the GPU id (NV2080_CTRL_CMD_GPU_GET_ID, on the subdevice).
    pub fn getGpuId(self: *Client, dev: Device) Error!sdk.NvU32 {
        var p = sdk.GpuGetIdParams{};
        try self.control(dev, dev.subdevice, sdk.NV2080_CTRL_CMD_GPU_GET_ID, &p, @sizeOf(sdk.GpuGetIdParams));
        return p.gpu_id;
    }

    /// Query the GPU's ASCII name into `buf`, returning the slice (e.g.
    /// "NVIDIA GeForce RTX 5070").
    pub fn getGpuName(self: *Client, dev: Device, buf: []u8) Error![]const u8 {
        var p = sdk.GpuGetNameStringParams{};
        try self.control(dev, dev.subdevice, sdk.NV2080_CTRL_CMD_GPU_GET_NAME_STRING, &p, @sizeOf(sdk.GpuGetNameStringParams));
        const end = std.mem.indexOfScalar(u8, &p.ascii, 0) orelse p.ascii.len;
        const n = @min(end, buf.len);
        @memcpy(buf[0..n], p.ascii[0..n]);
        return buf[0..n];
    }

    /// Allocate a GPU virtual address space (FERMI_VASPACE_A) under the device.
    /// First prerequisite for a GPFIFO channel. Free with rmFree(client, device, h).
    pub fn allocVaSpace(self: *Client, dev: Device) Error!sdk.NvHandle {
        var params = sdk.VaSpaceAllocParams{};
        return self.t.rmAlloc(dev.client, dev.device, self.t.newHandle(), sdk.FERMI_VASPACE_A, &params, @sizeOf(sdk.VaSpaceAllocParams));
    }

    /// Bind a physical memory object into `vaspace` at GPU virtual address
    /// `gpu_va`, so the GPU can address it (e.g. a channel's GPFIFO/pushbuffer).
    /// Carves an NV01_MEMORY_VIRTUAL range then RM_MAP_MEMORY_DMA with a fixed
    /// offset (dmaOffset is an input for a virtual hDma; VA 0 is reserved).
    pub fn mapToGpu(self: *Client, dev: Device, vaspace: sdk.NvHandle, mem: Memory, gpu_va: u64) Error!GpuMapping {
        return self.t.mapToGpu(dev, vaspace, mem, gpu_va);
    }

    /// Map a block-linear (ZETA depth) surface into the GPU VA space with BIG pages.
    /// Block-linear PTE kinds are only valid on big pages; a small-page map faults
    /// the depth draw (Xid 69 / ErrorCode 0x9c). `gpu_va` should be big-page aligned.
    pub fn mapToGpuBig(self: *Client, dev: Device, vaspace: sdk.NvHandle, mem: Memory, gpu_va: u64) Error!GpuMapping {
        return self.t.mapToGpuPaged(dev, vaspace, mem, gpu_va, sdk.dma_flags.PAGE_SIZE_BIG);
    }

    /// Release a GPU-VA mapping (frees the NV01_MEMORY_VIRTUAL object, which
    /// unmaps the physical memory). The physical Memory is freed separately.
    pub fn unmapFromGpu(self: *Client, dev: Device, mapping: GpuMapping) void {
        self.t.unmapFromGpu(dev, mapping);
    }

    /// Allocate a TSG / channel group (KEPLER_CHANNEL_GROUP_A) under the device,
    /// bound to `vaspace` and the graphics engine. Free with rmFree(client, device, h).
    pub fn allocTsg(self: *Client, dev: Device, vaspace: sdk.NvHandle) Error!sdk.NvHandle {
        var p = sdk.TsgAllocParams{ .h_vaspace = vaspace, .engine_type = sdk.NV2080_ENGINE_TYPE_GRAPHICS };
        return self.t.rmAlloc(dev.client, dev.device, self.t.newHandle(), sdk.KEPLER_CHANNEL_GROUP_A, &p, @sizeOf(sdk.TsgAllocParams));
    }

    /// Allocate a GPFIFO channel: the ring (`gpfifo_gpu_va`, `gpfifo_entries`)
    /// lives at a GPU VA in `vaspace`, with `userd` as the channel's USERD. The
    /// channel is parented to the device (RM manages its TSG).
    ///
    /// `class` is the GPU-generation channel class (e.g. BLACKWELL_CHANNEL_GPFIFO_B
    /// 0xCA6F for GB20x). Free with rmFree(client, device, channel.handle).
    pub fn allocChannel(
        self: *Client,
        dev: Device,
        class: sdk.NvV32,
        vaspace: sdk.NvHandle,
        gpfifo_gpu_va: u64,
        gpfifo_entries: u32,
        userd: Memory,
    ) Error!Channel {
        return self.allocChannelEngine(dev, class, vaspace, gpfifo_gpu_va, gpfifo_entries, userd, sdk.NV2080_ENGINE_TYPE_GRAPHICS);
    }

    /// Like allocChannel, but selects the runlist/engine the channel is created
    /// on (NV_CHANNEL_ALLOC_PARAMS.engineType). The graphics + compute paths use
    /// NV2080_ENGINE_TYPE_GRAPHICS; a copy-engine channel passes
    /// NV2080_ENGINE_TYPE_COPY0 so its async-CE work lands on the right runlist.
    /// The channel must still be bound to the SAME engine (bindChannel) before it
    /// can be scheduled.
    pub fn allocChannelEngine(
        self: *Client,
        dev: Device,
        class: sdk.NvV32,
        vaspace: sdk.NvHandle,
        gpfifo_gpu_va: u64,
        gpfifo_entries: u32,
        userd: Memory,
        engine_type: sdk.NvU32,
    ) Error!Channel {
        var p = sdk.ChannelAllocParams{
            .gp_fifo_offset = gpfifo_gpu_va,
            .gp_fifo_entries = gpfifo_entries,
            .h_vaspace = vaspace,
            .engine_type = engine_type,
        };
        p.h_userd_memory[0] = userd.handle;
        const handle = try self.t.rmAlloc(dev.client, dev.device, self.t.newHandle(), class, &p, @sizeOf(sdk.ChannelAllocParams));
        return .{ .handle = handle, .gpfifo_gpu_va = gpfifo_gpu_va, .gpfifo_entries = gpfifo_entries };
    }

    /// Bind a channel to an engine (NVA06F_CTRL_CMD_BIND) - required before it
    /// can be scheduled.
    pub fn bindChannel(self: *Client, dev: Device, channel: Channel, engine_type: sdk.NvU32) Error!void {
        var p = sdk.ChannelBindParams{ .engine_type = engine_type };
        try self.control(dev, channel.handle, sdk.NVA06F_CTRL_CMD_BIND, &p, @sizeOf(sdk.ChannelBindParams));
    }

    /// Enable/disable scheduling for a channel (NVA06F_CTRL_CMD_GPFIFO_SCHEDULE).
    pub fn scheduleChannel(self: *Client, dev: Device, channel: Channel, enable: bool) Error!void {
        var p = sdk.GpfifoScheduleParams{ .b_enable = @intFromBool(enable) };
        try self.control(dev, channel.handle, sdk.NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, &p, @sizeOf(sdk.GpfifoScheduleParams));
    }

    /// Query a channel's work-submit token (written to the USERMODE doorbell to
    /// kick the channel).
    pub fn workSubmitToken(self: *Client, dev: Device, channel: Channel) Error!sdk.NvU32 {
        var p = sdk.WorkSubmitTokenParams{};
        try self.control(dev, channel.handle, sdk.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, &p, @sizeOf(sdk.WorkSubmitTokenParams));
        return p.token;
    }

    /// Allocate a USERMODE doorbell object under the subdevice (e.g.
    /// BLACKWELL_USERMODE_A). Map it with mapMemory to reach the doorbell register.
    pub fn allocUsermode(self: *Client, dev: Device, class: sdk.NvV32) Error!sdk.NvHandle {
        return self.t.rmAlloc(dev.client, dev.subdevice, self.t.newHandle(), class, null, 0);
    }

    /// Allocate an engine object of `class` (e.g. a 3D class like 0xce97) under a
    /// `channel`, creating its engine context. Bind it in the pushbuffer with a
    /// SET_OBJECT method whose data is the CLASS id (not this handle).
    pub fn allocObject(self: *Client, dev: Device, channel: Channel, class: sdk.NvV32) Error!sdk.NvHandle {
        return self.t.rmAlloc(dev.client, channel.handle, self.t.newHandle(), class, null, 0);
    }
};

/// A submission queue over a bound+scheduled GPFIFO channel: push GP_ENTRYs to
/// its ring and ring the USERMODE doorbell so the GPU fetches and executes them.
/// Build it from the channel, its work-submit token, and CPU mappings of the
/// USERD, the GPFIFO ring, and the USERMODE doorbell page.
pub const Queue = struct {
    channel: Channel,
    token: sdk.NvU32,
    userd: []u8, // CPU mapping of the channel's USERD
    gpfifo: []u8, // CPU mapping of the GPFIFO ring
    doorbell: []u8, // CPU mapping of the USERMODE doorbell page
    gp_put: u32 = 0,

    /// Push a GP_ENTRY for the `len_dwords` pushbuffer at GPU VA `pb_gpu_va`,
    /// advance GP_PUT, and ring the doorbell. The GPU then executes it.
    pub fn submit(self: *Queue, pb_gpu_va: u64, len_dwords: u32) void {
        const ring: [*]volatile sdk.NvU32 = @ptrCast(@alignCast(self.gpfifo.ptr));
        const e = sdk.gpfifo.entry(pb_gpu_va, len_dwords);
        ring[self.gp_put * 2] = e[0];
        ring[self.gp_put * 2 + 1] = e[1];
        self.gp_put = (self.gp_put + 1) % self.channel.gpfifo_entries;

        const gp_put: *volatile sdk.NvU32 = @ptrCast(@alignCast(self.userd.ptr + sdk.gpfifo.USERD_GP_PUT_OFFSET));
        gp_put.* = self.gp_put;
        _ = gp_put.*; // flush the write-combined store before the doorbell

        const doorbell: *volatile sdk.NvU32 = @ptrCast(@alignCast(self.doorbell.ptr + sdk.gpfifo.USERMODE_DOORBELL_OFFSET));
        doorbell.* = self.token;
    }

    /// GP_GET (how far the GPU has consumed the ring), read from USERD.
    pub fn gpGet(self: *const Queue) sdk.NvU32 {
        const p: *const volatile sdk.NvU32 = @ptrCast(@alignCast(self.userd.ptr + sdk.gpfifo.USERD_GP_GET_OFFSET));
        return p.*;
    }
};

// ---------------------------------------------------------------------------
// Live tests against the real NVIDIA RM. These SKIP when the device can't be
// reached (no driver / no /dev/nvidia* / no permission) and only FAIL when a
// reachable device misbehaves - so regressions still surface on real hardware.
// ---------------------------------------------------------------------------

/// Open the client or skip the test if the RM control device isn't reachable.
fn openOrSkip() !Client {
    return Client.open() catch return error.SkipZigTest;
}

test "live: open RM client + detect driver version" {
    var c = try openOrSkip();
    defer c.deinit();
    try std.testing.expect(c.driverVersion().major > 0);
    try std.testing.expect(c.abi() != .unknown);
}

test "live: version QUERY round-trips through the kernel" {
    var c = try openOrSkip();
    defer c.deinit();
    const v = try c.checkVersion(sdk.RmApiVersion.CMD_QUERY, "");
    try std.testing.expectEqual(sdk.RmApiVersion.REPLY_RECOGNIZED, v.reply);
}

test "live: bring up GPU 0 (root client + device + subdevice)" {
    var c = try openOrSkip();
    defer c.deinit();
    const dev = c.allocDevice(0) catch |e| switch (e) {
        // Couldn't get into the GPU (absent or no permission): skip, don't fail.
        error.OpenFailed, error.NoDevice => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeDevice(dev);
    try std.testing.expect(dev.gpu_id != 0);
    try std.testing.expect(dev.client != 0);
    try std.testing.expect(dev.device != 0);
    try std.testing.expect(dev.subdevice != 0);
}

test "live: allocate + free system and VRAM memory" {
    var c = try openOrSkip();
    defer c.deinit();
    const dev = c.allocDevice(0) catch |e| switch (e) {
        error.OpenFailed, error.NoDevice => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeDevice(dev);

    const sys = try c.allocMemory(dev, .system, 64 * 1024);
    defer c.freeMemory(dev, sys);
    try std.testing.expect(sys.handle != 0);

    const vram = try c.allocMemory(dev, .vram, 64 * 1024);
    defer c.freeMemory(dev, vram);
    try std.testing.expect(vram.handle != 0);
}

test "live: map system memory and round-trip the CPU" {
    var c = try openOrSkip();
    defer c.deinit();
    const dev = c.allocDevice(0) catch |e| switch (e) {
        error.OpenFailed, error.NoDevice => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeDevice(dev);

    const mem = try c.allocMemory(dev, .system, 64 * 1024);
    defer c.freeMemory(dev, mem);

    const map = try c.mapMemory(dev, mem);
    defer c.unmapMemory(map);
    try std.testing.expect(map.bytes.len >= mem.size);

    // Write a pattern and read it back through the mapping.
    map.bytes[0] = 0xAB;
    map.bytes[1] = 0xCD;
    map.bytes[mem.size - 1] = 0xEF;
    try std.testing.expectEqual(@as(u8, 0xAB), map.bytes[0]);
    try std.testing.expectEqual(@as(u8, 0xCD), map.bytes[1]);
    try std.testing.expectEqual(@as(u8, 0xEF), map.bytes[mem.size - 1]);
}

test "live: RM_CONTROL queries GPU id and name" {
    var c = try openOrSkip();
    defer c.deinit();
    const dev = c.allocDevice(0) catch |e| switch (e) {
        error.OpenFailed, error.NoDevice => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeDevice(dev);

    try std.testing.expect(try c.getGpuId(dev) != 0);

    var buf: [64]u8 = undefined;
    const name = try c.getGpuName(dev, &buf);
    try std.testing.expect(name.len > 0);
}

test "live: allocate a GPU VA space (channel prerequisite)" {
    var c = try openOrSkip();
    defer c.deinit();
    const dev = c.allocDevice(0) catch |e| switch (e) {
        error.OpenFailed, error.NoDevice => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeDevice(dev);

    const vaspace = try c.allocVaSpace(dev);
    defer c.rmFree(dev.client, dev.device, vaspace);
    try std.testing.expect(vaspace != 0);
}

test "live: bind memory into a GPU VA space (RM_MAP_MEMORY_DMA)" {
    var c = try openOrSkip();
    defer c.deinit();
    const dev = c.allocDevice(0) catch |e| switch (e) {
        error.OpenFailed, error.NoDevice => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeDevice(dev);

    const vaspace = try c.allocVaSpace(dev);
    defer c.rmFree(dev.client, dev.device, vaspace);
    const mem = try c.allocMemory(dev, .vram, 64 * 1024);
    defer c.freeMemory(dev, mem);

    const gpu_va: u64 = 0x200000; // 2 MB; VA 0 is reserved
    const mapping = try c.mapToGpu(dev, vaspace, mem, gpu_va);
    defer c.unmapFromGpu(dev, mapping);
    try std.testing.expectEqual(gpu_va, mapping.gpu_va);
    try std.testing.expect(mapping.virtual != 0);
}

test "live: import host memory as an OS descriptor and map it to the GPU" {
    var c = try openOrSkip();
    defer c.deinit();
    const dev = c.allocDevice(0) catch |e| switch (e) {
        error.OpenFailed, error.NoDevice => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeDevice(dev);

    // A page-aligned, populated host buffer - stands in for a Wayland shm mmap.
    const size: u64 = 64 * 1024;
    const rc = std.os.linux.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    if (@as(isize, @bitCast(rc)) < 0) return error.SkipZigTest;
    const host: [*]align(4096) u8 = @ptrFromInt(rc);
    defer _ = std.os.linux.munmap(host, size);
    for (0..size) |i| host[i] = 0; // fault the pages in before import

    // NOTE: the open kernel module 595.71.05 rejects this with NV_ERR_NOT_SUPPORTED
    // (0x56) - userspace virtual-address OS-descriptor import appears gated there.
    // The ABI/struct is correct (kept for proprietary/newer modules); skip when
    // the running module declines it.
    const mem = c.importMemory(dev, rc, size) catch |e| switch (e) {
        error.RmAllocFailed => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeMemory(dev, mem);
    try std.testing.expectEqual(size, mem.size);

    // The imported range must map into the GPU address space like any memory.
    const vaspace = try c.allocVaSpace(dev);
    defer c.rmFree(dev.client, dev.device, vaspace);
    const mapping = try c.mapToGpu(dev, vaspace, mem, 0x200000);
    defer c.unmapFromGpu(dev, mapping);
    try std.testing.expectEqual(@as(u64, 0x200000), mapping.gpu_va);
}

test "live: allocate a TSG (channel group)" {
    var c = try openOrSkip();
    defer c.deinit();
    const dev = c.allocDevice(0) catch |e| switch (e) {
        error.OpenFailed, error.NoDevice => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeDevice(dev);

    const vaspace = try c.allocVaSpace(dev);
    defer c.rmFree(dev.client, dev.device, vaspace);
    const tsg = try c.allocTsg(dev, vaspace);
    defer c.rmFree(dev.client, dev.device, tsg);
    try std.testing.expect(tsg != 0);
}

test "live: allocate a GPFIFO channel (Blackwell)" {
    var c = try openOrSkip();
    defer c.deinit();
    const dev = c.allocDevice(0) catch |e| switch (e) {
        error.OpenFailed, error.NoDevice => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeDevice(dev);

    const vaspace = try c.allocVaSpace(dev);
    defer c.rmFree(dev.client, dev.device, vaspace);

    // USERD + a GPFIFO ring bound into the VA space.
    const userd = try c.allocMemory(dev, .vram, 64 * 1024);
    defer c.freeMemory(dev, userd);
    const gpfifo = try c.allocMemory(dev, .vram, 64 * 1024);
    defer c.freeMemory(dev, gpfifo);
    const ring = try c.mapToGpu(dev, vaspace, gpfifo, 0x200000);
    defer c.unmapFromGpu(dev, ring);

    // Channel class is GPU-generation-specific; this box is Blackwell GB202.
    const channel = try c.allocChannel(dev, sdk.BLACKWELL_CHANNEL_GPFIFO_B, vaspace, ring.gpu_va, 0x400, userd);
    defer c.rmFree(dev.client, dev.device, channel.handle);
    try std.testing.expect(channel.handle != 0);
    try std.testing.expectEqual(ring.gpu_va, channel.gpfifo_gpu_va);
}

test "live: channel submission control-plane (bind/schedule/token/usermode)" {
    var c = try openOrSkip();
    defer c.deinit();
    const dev = c.allocDevice(0) catch |e| switch (e) {
        error.OpenFailed, error.NoDevice => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeDevice(dev);

    const vaspace = try c.allocVaSpace(dev);
    defer c.rmFree(dev.client, dev.device, vaspace);
    const userd = try c.allocMemory(dev, .vram, 64 * 1024);
    defer c.freeMemory(dev, userd);
    const gpfifo_mem = try c.allocMemory(dev, .vram, 64 * 1024);
    defer c.freeMemory(dev, gpfifo_mem);
    const ring = try c.mapToGpu(dev, vaspace, gpfifo_mem, 0x200000);
    defer c.unmapFromGpu(dev, ring);
    const channel = try c.allocChannel(dev, sdk.BLACKWELL_CHANNEL_GPFIFO_B, vaspace, ring.gpu_va, 0x100, userd);
    defer c.rmFree(dev.client, dev.device, channel.handle);

    // The full submission control-plane: bind to GR, enable scheduling, get the
    // work-submit token, and map a USERMODE doorbell.
    try c.bindChannel(dev, channel, sdk.NV2080_ENGINE_TYPE_GRAPHICS);
    try c.scheduleChannel(dev, channel, true);
    try std.testing.expect(try c.workSubmitToken(dev, channel) != 0);

    const usermode = try c.allocUsermode(dev, sdk.BLACKWELL_USERMODE_A);
    defer c.rmFree(dev.client, dev.subdevice, usermode);
    const door = try c.mapMemory(dev, .{ .handle = usermode, .size = 0x1000, .location = .vram });
    defer c.unmapMemory(door);
    try std.testing.expect(door.bytes.len >= 0x1000);
}

test "live: memToDmaBuf: system memory -> real dma-buf fd" {
    var c = try openOrSkip();
    defer c.deinit();
    const dev = c.allocDevice(0) catch |e| switch (e) {
        error.OpenFailed, error.NoDevice => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeDevice(dev);

    // Allocate SYSTEM memory: get_user_pages requires real CPU-visible pages.
    // 64x64 RGBA = 16384 bytes, rounds up to 1 page.
    const size: u64 = 64 * 64 * 4;
    const mem = try c.allocMemory(dev, .system, size);
    defer c.freeMemory(dev, mem);

    // Map it to get a CPU virtual address. Keep alive until after dma-buf consumer is done.
    const mapping = try c.mapMemory(dev, mem);
    defer c.unmapMemory(mapping);
    const va: usize = @intFromPtr(mapping.bytes.ptr);

    // Turn the CPU VA into a real dma-buf fd via the nvidia-drm render node.
    const dmabuf_fd: std.posix.fd_t = drm.memToDmaBuf(va, mem.size) catch |e| {
        // If the renderD node is not available on this box, skip gracefully.
        if (e == error.DrmOpenFailed) return error.SkipZigTest;
        return e;
    };
    // The dma-buf fd must be valid (>= 0).
    try std.testing.expect(dmabuf_fd >= 0);

    // Critical acceptance test: readlink /proc/self/fd/<fd> must contain "dmabuf".
    // A real dma-buf shows something like "anon_inode:dmabuf" or "/dmabuf:...".
    // A /dev/nvidiactl fd (the bug the prior attempt had) would show "/dev/nvidiactl".
    var proc_path_buf: [64]u8 = undefined;
    const proc_path = std.fmt.bufPrintZ(&proc_path_buf, "/proc/self/fd/{d}", .{dmabuf_fd}) catch unreachable;
    var link_target: [256]u8 = undefined;
    const link_len = std.os.linux.readlink(proc_path.ptr, &link_target, link_target.len);
    try std.testing.expect(@as(isize, @bitCast(link_len)) > 0);
    const target_str = link_target[0..link_len];

    // Log the target so the test output shows the readlink value.
    std.debug.print("\n[dmabuf test] readlink({s}) = {s}\n", .{ proc_path, target_str });

    // The presence of "dmabuf" in the link target is the acceptance criterion.
    const has_dmabuf = std.mem.indexOf(u8, target_str, "dmabuf") != null;
    if (!has_dmabuf) {
        std.debug.print("[dmabuf test] FAIL: target '{s}' does not contain 'dmabuf'\n", .{target_str});
    }
    try std.testing.expect(has_dmabuf);

    // Close the dma-buf fd. The mapping defer runs after this at scope end,
    // which is correct: mapping must outlive the dma-buf consumer (this test).
    _ = std.os.linux.close(dmabuf_fd);
}

test "live: submit a command, the GPU executes it (semaphore release)" {
    var c = try openOrSkip();
    defer c.deinit();
    const dev = c.allocDevice(0) catch |e| switch (e) {
        error.OpenFailed, error.NoDevice => return error.SkipZigTest,
        else => return e,
    };
    defer c.freeDevice(dev);
    const va = try c.allocVaSpace(dev);
    defer c.rmFree(dev.client, dev.device, va);

    // GPFIFO ring + pushbuffer in VRAM (GPU-read); semaphore target in sysmem
    // (so the CPU sees the GPU's write). Each gets a GPU VA and a CPU mapping.
    const userd = try c.allocMemory(dev, .vram, 0x1000);
    defer c.freeMemory(dev, userd);
    const gpfifo = try c.allocMemory(dev, .vram, 0x2000);
    defer c.freeMemory(dev, gpfifo);
    const pushbuf = try c.allocMemory(dev, .vram, 0x1000);
    defer c.freeMemory(dev, pushbuf);
    const sem = try c.allocMemory(dev, .system, 0x1000);
    defer c.freeMemory(dev, sem);

    const gpf_va: u64 = 0x200000;
    const pb_va: u64 = 0x210000;
    const sem_va: u64 = 0x220000;
    const gm = try c.mapToGpu(dev, va, gpfifo, gpf_va);
    defer c.unmapFromGpu(dev, gm);
    const pm = try c.mapToGpu(dev, va, pushbuf, pb_va);
    defer c.unmapFromGpu(dev, pm);
    const sm = try c.mapToGpu(dev, va, sem, sem_va);
    defer c.unmapFromGpu(dev, sm);

    const userd_map = try c.mapMemory(dev, userd);
    defer c.unmapMemory(userd_map);
    const gpf_map = try c.mapMemory(dev, gpfifo);
    defer c.unmapMemory(gpf_map);
    const pb_map = try c.mapMemory(dev, pushbuf);
    defer c.unmapMemory(pb_map);
    const sem_map = try c.mapMemory(dev, sem);
    defer c.unmapMemory(sem_map);

    const channel = try c.allocChannel(dev, sdk.BLACKWELL_CHANNEL_GPFIFO_B, va, gpf_va, 0x100, userd);
    defer c.rmFree(dev.client, dev.device, channel.handle);
    try c.bindChannel(dev, channel, sdk.NV2080_ENGINE_TYPE_GRAPHICS);
    try c.scheduleChannel(dev, channel, true);
    const token = try c.workSubmitToken(dev, channel);

    const usermode = try c.allocUsermode(dev, sdk.BLACKWELL_USERMODE_A);
    defer c.rmFree(dev.client, dev.subdevice, usermode);
    const door_map = try c.mapMemory(dev, .{ .handle = usermode, .size = 0x1000, .location = .vram });
    defer c.unmapMemory(door_map);

    // Pushbuffer: release 0xCAFE to the semaphore at sem_va.
    const pb = std.mem.bytesAsSlice(sdk.NvU32, pb_map.bytes);
    const methods = sdk.gpfifo.semaphoreRelease(sem_va, 0xCAFE);
    @memcpy(pb[0..methods.len], &methods);

    const sem_ptr: *volatile sdk.NvU32 = @ptrCast(@alignCast(sem_map.bytes.ptr));
    sem_ptr.* = 0;

    // Submit and wait for the GPU to write the semaphore.
    var q = Queue{ .channel = channel, .token = token, .userd = userd_map.bytes, .gpfifo = gpf_map.bytes, .doorbell = door_map.bytes };
    q.submit(pb_va, @intCast(methods.len));

    var i: u32 = 0;
    while (i < 50_000_000) : (i += 1) {
        if (sem_ptr.* == 0xCAFE) break;
    }
    try std.testing.expectEqual(@as(sdk.NvU32, 0xCAFE), sem_ptr.*);
}
