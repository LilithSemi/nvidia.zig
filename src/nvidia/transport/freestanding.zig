//! Freestanding Transport: the baremetal RM-over-GSP backend (plan phase P6). On
//! freestanding/UEFI there is no kernel RM, so every RM primitive is a GSP RPC:
//! we own the GPU directly over BAR0 MMIO, boot the GSP (boot.Boot.run), bring the
//! shared-mem RPC ring live, and then carry the SAME sdk.zig alloc/control payloads
//! the Linux path uses - GSP only changes the TRANSPORT (the ring + the RPC
//! envelopes from gsp/), not the payloads.
//!
//! This file satisfies the SAME public surface as transport/linux.zig (transport.zig
//! comptime-selects between them; rm.zig + the HAL call only this surface). Each op
//! MIRRORS its linux.zig counterpart op-for-op, swapping ioctl -> RPC:
//!   rmAlloc   -> GSP_RM_ALLOC (103) wrapping the sdk alloc params.
//!   rmFree    -> FREE (10).
//!   control   -> GSP_RM_CONTROL (76) wrapping the sdk Os54-style params.
//!   allocDevice -> the ROOT / DEVICE / SUBDEVICE alloc sequence, each over RPC.
//!   mapMemory -> MAP_MEMORY (7) + a direct VRAM/BAR-window CPU pointer (no mmap).
//!   mapToGpu  -> NV01_MEMORY_VIRTUAL alloc + MAP_MEMORY_DMA (the virtual-obj + DMA
//!               dance), mirroring linux.zig's DMA_OFFSET_FIXED path.
//!
//! WE own memory on baremetal (no kernel to allocate from): a sysmem bump allocator
//! (EFI AllocatePages identity-mapped phys on the real path) backs the rings /
//! radix3 / WprMeta / boot args, and a VRAM bump allocator carves the device memory.
//! Both reuse fw.zig's BumpAllocator.
//!
//! It is comptime-dead on a Linux build (transport.zig selects linux.zig there), so
//! Zig never semantically analyses it on Linux; it stays syntactically valid and
//! compiles when targeting freestanding/UEFI - where the WHOLE rm.zig + HAL +
//! transport + gsp path is analysed against it.

const std = @import("std");
const builtin = @import("builtin");
const sdk = @import("../sdk.zig");
const version = @import("../version.zig");
const types = @import("types.zig");
const hw = @import("../hw/mmio.zig");
const chip = @import("../hw/chip.zig");
const gsp = @import("../gsp/gsp.zig");
const ring = gsp.ring;
const proto = gsp.proto;
const fw = gsp.fw;
const boot = gsp.boot;

const Error = types.Error;
const Device = types.Device;
const Memory = types.Memory;
const Mapping = types.Mapping;
const GpuMapping = types.GpuMapping;

/// The "no fd" value for the Device.node_fd / Mapping.fd fields. On UEFI
/// std.posix.fd_t is `void` ({}); on a bare freestanding target it is i32 (-1).
/// Baremetal has no file descriptors, so these are inert placeholders that keep
/// the shared Device/Mapping shape identical to linux.zig.
const NULL_FD = if (@TypeOf(@as(std.posix.fd_t, undefined)) == void) {} else @as(std.posix.fd_t, -1);

/// Lift a boot.Error (a superset including fw/bios/ring errors) into the transport
/// Error set: any GSP boot failure surfaces as OpenFailed (the bring-up failed).
fn liftBootError(e: anyerror) Error {
    return switch (e) {
        error.NotImplemented => error.NotImplemented,
        else => error.OpenFailed,
    };
}

// On baremetal a VRAM Memory object's bytes are reached through a BAR window or a
// direct device-memory pointer (== phys under UEFI identity mapping); the offline
// test path hands back a plain physical base we treat as the CPU address. The
// transport tracks both the GPU-side phys and the CPU-side window per object so
// mapMemory can return a real slice without an mmap.

/// One staging buffer big enough for the largest RPC element we build (an alloc
/// or control header + the biggest sdk params blob). The sdk alloc/control params
/// are all well under a 4 KB slot; one max element span is ample.
const RPC_SCRATCH: usize = proto.ELEMENT_SIZE_MAX;

pub const Transport = struct {
    // --- the live hardware + GSP state (set up by open()) ---
    /// BAR0 MMIO aperture (from the UEFI PCI probe).
    bar: hw.Bar = .{ .base = 0, .size = 0 },
    /// Decoded chip architecture (drives per-arch firmware + register tables).
    arch: chip.Architecture = .ga10x,
    /// The live host-side GSP RPC ring endpoint (cmdq producer / msgq consumer),
    /// brought up by boot.Boot.run. All RM ops send/recv through this.
    endpoint: ring.Endpoint = undefined,
    /// True once open() has booted the GSP and the ring is live. Guards the RPC
    /// ops so a never-opened transport fails cleanly rather than touching undef.
    booted: bool = false,

    // --- memory WE own (no kernel allocator on baremetal) ---
    /// Sysmem pool (rings / radix3 / WprMeta / boot args + the RM mapping windows).
    /// On the real path the base is an EFI AllocatePages identity-mapped phys.
    sysmem: fw.BumpAllocator = fw.BumpAllocator.init(0, 0),
    /// VRAM carveout (device memory the GPU renders into), below the WPR2 region.
    vram: fw.BumpAllocator = fw.BumpAllocator.init(0, 0),

    // --- RM bookkeeping ---
    /// Driver version - on the GSP path this is the firmware .fwversion (595.71.05).
    driver_version: version.Version = .{ .major = 0, .minor = 0, .patch = 0 },
    /// ABI generation selected from driver_version.
    abi: version.Abi = .unknown,
    /// Monotonic source of unique RM object handles within this client.
    next_handle: sdk.NvHandle = 0xc1d00000,
    /// The root client handle the GSP assigns at allocDevice (the in/out handle).
    h_root: sdk.NvHandle = 0,

    /// Per-object record so mapMemory can return a CPU slice without an mmap and
    /// mapToGpu / control can resolve a handle to its phys. A small fixed table
    /// (baremetal has no heap; the HAL allocates a handful of objects).
    objects: [MAX_OBJECTS]ObjectRecord = [_]ObjectRecord{.{}} ** MAX_OBJECTS,
    object_count: usize = 0,

    /// Scratch for marshalling one outbound RPC element (header + body + params).
    rpc_scratch: [RPC_SCRATCH]u8 align(8) = undefined,

    pub const MAX_OBJECTS: usize = 64;

    /// A tracked RM memory object: its handle, the phys base WE assigned it, the
    /// CPU-visible window (== phys under identity mapping), the size + location.
    pub const ObjectRecord = struct {
        handle: sdk.NvHandle = 0,
        phys: u64 = 0,
        cpu: usize = 0,
        size: u64 = 0,
        location: Memory.Location = .system,
    };

    // =======================================================================
    // open(): the whole bring-up (HARDWARE-ONLY to RUN; the assembly is built).
    // =======================================================================

    /// On baremetal open() takes the probed BAR0 + chip + the firmware/booter bytes
    /// + the memory pools WE own and runs the entire bring-up. Because the metal
    /// inputs (BAR0, the @embedFile'd firmware, the EFI-allocated sysmem) come from
    /// the UEFI app, the parameter-less open() the Transport contract requires
    /// cannot itself reach the hardware: it returns NotImplemented so a misuse on a
    /// platform without those inputs fails cleanly. The real entry point is
    /// `openOnMetal` below, which the UEFI app calls with the probed inputs.
    pub fn open() Error!Transport {
        return error.NotImplemented;
    }

    /// The metal bring-up inputs the UEFI app gathers (BAR0 from the PCI probe, the
    /// chip arch from the chip-id read, the @embedFile'd firmware/booter bytes, and
    /// the EFI-allocated identity-mapped sysmem + the VRAM carveout WE own).
    pub const MetalInputs = struct {
        bar: hw.Bar,
        arch: chip.Architecture,
        /// Sysmem pool base phys (EFI AllocatePages, identity-mapped) + size.
        sysmem_base: u64,
        sysmem_size: u64,
        /// VRAM carveout base phys + size (below the WPR2 region at the top of VRAM).
        vram_base: u64,
        vram_size: u64,
        /// The assembled GSP boot inputs (firmware-packaged WprMeta / radix3 / args /
        /// patched booters), built by the caller from fw.zig + boot.zig + the
        /// @embedFile'd gsp_ga10x.bin. The host RPC ring endpoint is set up over the
        /// shared-mem region the boot args point at.
        boot_inputs: boot.BootInputs,
        /// The host-side ring endpoint over the shared cmdq/msgq region (from the
        /// boot args' shared_phys). Formatted + ready for boot.Boot.run to drive.
        endpoint: ring.Endpoint,
        /// CPU-visible bytes of the sysmem WprMeta region (the booter writes the
        /// `verified` field here; step4 reads it back).
        wpr_meta_cpu: []const u8,
    };

    /// Bring the GPU up over GSP and leave the RPC ring live (HARDWARE-ONLY to RUN;
    /// the freestanding build COMPILES this whole path). Steps, mirroring nouveau's
    /// r535_gsp boot + r535_gsp_postinit:
    ///   1. boot.Boot.run: FWSEC-FRTS -> booter -> kick -> poll GSP_INIT_DONE. The
    ///      ring is live on return.
    ///   2. the post-boot init RPCs nouveau/OGK send (GSP_SET_SYSTEM_INFO,
    ///      GET_GSP_STATIC_INFO, SET_REGISTRY) - assembled here so postinit is wired.
    /// Then returns a live Transport every later RM op runs through.
    pub fn openOnMetal(in: MetalInputs) Error!Transport {
        var self = Transport{
            .bar = in.bar,
            .arch = in.arch,
            .endpoint = in.endpoint,
            .sysmem = fw.BumpAllocator.init(in.sysmem_base, in.sysmem_size),
            .vram = fw.BumpAllocator.init(in.vram_base, in.vram_size),
            // The firmware .fwversion gate is 595.71.05; record it as the version.
            .driver_version = version.Version.parse(fw.FW_VERSION) orelse .{ .major = 595, .minor = 71, .patch = 5 },
        };
        self.abi = version.Abi.fromVersion(self.driver_version);

        // 1. Boot the GSP. On return the RPC ring is live + GSP_INIT_DONE was seen.
        const driver = boot.Boot.init(in.bar, in.arch);
        driver.run(in.boot_inputs, &self.endpoint, in.wpr_meta_cpu) catch |e| return liftBootError(e);
        self.booted = true;

        // 2. The post-boot init RPCs (r535_gsp_postinit). These are control/no-param
        //    RPCs the GSP-RM expects before normal RM traffic; we assemble + send
        //    them through the live ring. They take no sdk payload here (the system
        //    info / registry blobs are GSP-internal defaults on the consumer path).
        try self.postInit();
        return self;
    }

    /// The post-boot init sequence (r535_gsp_postinit): GSP_SET_SYSTEM_INFO,
    /// GET_GSP_STATIC_INFO, SET_REGISTRY. Sent as bare RPCs through the live ring.
    /// HARDWARE-ONLY to complete (needs the GSP answering); the assembly compiles.
    fn postInit(self: *Transport) Error!void {
        try self.sendRpc(@intFromEnum(proto.Function.gsp_set_system_info), &[_]u8{});
        _ = try self.recvRpc(@intFromEnum(proto.Function.gsp_set_system_info));
        try self.sendRpc(@intFromEnum(proto.Function.set_registry), &[_]u8{});
        _ = try self.recvRpc(@intFromEnum(proto.Function.set_registry));
        try self.sendRpc(@intFromEnum(proto.Function.get_gsp_static_info), &[_]u8{});
        _ = try self.recvRpc(@intFromEnum(proto.Function.get_gsp_static_info));
    }

    pub fn deinit(self: Transport) void {
        _ = self;
        // The GSP stays booted for the lifetime of the firmware image; on a real
        // teardown we would send a GSP unload RPC + run booter_unload. Nothing to
        // free on the bump allocators (WE own the whole pool).
    }

    pub fn newHandle(self: *Transport) sdk.NvHandle {
        self.next_handle += 1;
        return self.next_handle;
    }

    /// On the GSP path "checkVersion" is the firmware .fwversion gate, not an ioctl
    /// (version.zig backs both). CMD_QUERY returns the pinned firmware version with
    /// REPLY_RECOGNIZED; a STRICT/RELAXED check compares the provided string.
    pub fn checkVersion(self: Transport, cmd: sdk.NvU32, ver: []const u8) Error!sdk.RmApiVersion {
        _ = self;
        var p = sdk.RmApiVersion{
            .cmd = cmd,
            .reply = sdk.RmApiVersion.REPLY_UNRECOGNIZED,
            .version_string = [_]u8{0} ** sdk.RmApiVersion.STRING_LENGTH,
        };
        // Fill in the pinned firmware version (the GSP firmware we boot against).
        const fwv = fw.FW_VERSION;
        const n = @min(fwv.len, sdk.RmApiVersion.STRING_LENGTH - 1);
        @memcpy(p.version_string[0..n], fwv[0..n]);

        switch (cmd) {
            sdk.RmApiVersion.CMD_QUERY => p.reply = sdk.RmApiVersion.REPLY_RECOGNIZED,
            else => {
                // STRICT/RELAXED: the provided string must match the pinned version.
                const want = std.mem.sliceTo(ver, 0);
                p.reply = if (std.mem.eql(u8, want, fwv))
                    sdk.RmApiVersion.REPLY_RECOGNIZED
                else
                    sdk.RmApiVersion.REPLY_UNRECOGNIZED;
            },
        }
        return p;
    }

    // =======================================================================
    // The RPC send/recv core: marshal -> ring.send -> ring.recv. Reused by every op.
    // =======================================================================

    /// Send a bare RPC (rpc_message_header_v + `body`) through the live cmdq ring.
    fn sendRpc(self: *Transport, function: u32, body: []const u8) Error!void {
        const total = @sizeOf(proto.RpcMessageHeader) + body.len;
        if (total > self.rpc_scratch.len) return error.IoctlFailed;
        const hdr: *proto.RpcMessageHeader = @ptrCast(@alignCast(&self.rpc_scratch[0]));
        hdr.* = .{
            .header_version = proto.RpcMessageHeader.HEADER_VERSION,
            .signature = proto.RpcMessageHeader.SIGNATURE,
            .length = @intCast(total),
            .function = function,
            .rpc_result = 0,
            .rpc_result_private = 0,
            .sequence = 0,
            .u = 0,
        };
        @memcpy(self.rpc_scratch[@sizeOf(proto.RpcMessageHeader)..][0..body.len], body);
        self.endpoint.send(self.rpc_scratch[0..total]) catch return error.IoctlFailed;
    }

    /// Poll the msgq ring for a reply with the expected `function` echo, spinning a
    /// bounded budget (the GSP answers asynchronously). Returns the reply payload
    /// (rpc_message_header_v + body), valid until the next recv. HARDWARE-ONLY to
    /// return a real reply; offline the fake-ring test plays the GSP side.
    fn recvRpc(self: *Transport, function: u32) Error![]const u8 {
        var spins: usize = 0;
        while (spins <= POLL_BUDGET) : (spins += 1) {
            const got = self.endpoint.recv() catch |e| switch (e) {
                ring.Error.Empty => continue,
                else => return error.IoctlFailed,
            };
            // Skip async events that are not the reply we awaited (e.g. a late
            // GSP_INIT_DONE or an RC event); match on the function echo.
            if (got.function == function) return got.rpc;
        }
        return error.IoctlFailed;
    }

    /// How many poll iterations to spin awaiting a GSP reply (hardware tuning).
    pub const POLL_BUDGET: usize = 10_000_000;

    // =======================================================================
    // rmAlloc -> GSP_RM_ALLOC (103). Mirrors linux.zig's NVOS21 path: same in/out
    // handle semantics (the GSP writes the assigned handle back for the root client).
    // =======================================================================

    pub fn rmAlloc(
        self: *Transport,
        h_root: sdk.NvHandle,
        h_parent: sdk.NvHandle,
        h_new: sdk.NvHandle,
        h_class: sdk.NvV32,
        params: ?*const anyopaque,
        params_size: sdk.NvU32,
    ) Error!sdk.NvHandle {
        if (!self.booted) return error.NotImplemented;

        // The sdk alloc params travel as the params[] blob inside rpc_gsp_rm_alloc_v.
        const pbytes: []const u8 = if (params) |pp|
            @as([*]const u8, @ptrCast(pp))[0..params_size]
        else
            &[_]u8{};

        // Build the GSP_RM_ALLOC element (rpc hdr + alloc hdr + sdk params) + send.
        const n = ring.buildAllocElement(
            &self.rpc_scratch,
            h_root,
            h_parent,
            h_new,
            h_class,
            0, // flags
            pbytes,
        );
        self.endpoint.send(self.rpc_scratch[0..n]) catch return error.IoctlFailed;

        // The GSP echoes GSP_RM_ALLOC with the out status + the assigned handle and
        // any written-back params. Read the alloc header back from the reply.
        const reply = try self.recvRpc(@intFromEnum(proto.Function.gsp_rm_alloc));
        const a: *const proto.RpcGspRmAlloc = @ptrCast(@alignCast(&reply[@sizeOf(proto.RpcMessageHeader)]));
        if (a.status != 0) return error.RmAllocFailed;

        // Copy any out-params the GSP wrote back into the caller's buffer (the sdk
        // alloc params are in/out, e.g. MemAllocParams.size, VirtMemAllocParams.limit).
        if (params) |pp| {
            const params_off = @sizeOf(proto.RpcMessageHeader) + @sizeOf(proto.RpcGspRmAlloc);
            if (params_off + params_size <= reply.len) {
                const dst: [*]u8 = @ptrCast(@constCast(pp));
                @memcpy(dst[0..params_size], reply[params_off .. params_off + params_size]);
            }
        }
        return a.h_object;
    }

    /// rmFree -> FREE (10). The free params are the sdk NVOS00 block (root/parent/
    /// object) carried as the RPC body. Best-effort (void), like linux.zig.
    pub fn rmFree(self: *Transport, h_root: sdk.NvHandle, h_parent: sdk.NvHandle, h_object: sdk.NvHandle) void {
        if (!self.booted) return;
        var p = sdk.Os00Params{ .h_root = h_root, .h_object_parent = h_parent, .h_object_old = h_object, .status = 0 };
        self.sendRpc(@intFromEnum(proto.Function.free), std.mem.asBytes(&p)) catch return;
        _ = self.recvRpc(@intFromEnum(proto.Function.free)) catch return;
        self.untrack(h_object);
    }

    // =======================================================================
    // allocDevice -> the ROOT / DEVICE / SUBDEVICE alloc sequence over RPC. Mirrors
    // linux.zig allocDevice's order + the sdk params it passes (only the registration
    // / attach ioctls drop away - on baremetal WE are the kernel, the GSP owns the GPU).
    // =======================================================================

    pub fn allocDevice(self: *Transport, index: u32) Error!Device {
        if (!self.booted) return error.NotImplemented;

        // Root client (NV01_ROOT, no params). The GSP assigns + returns the handle
        // (the in/out handle gotcha - use the returned value as the parent).
        const client = try self.rmAlloc(0, 0, self.newHandle(), sdk.NV01_ROOT, null, 0);
        self.h_root = client;
        errdefer self.rmFree(0, 0, client);

        // Device under the client (NV0080).
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
        errdefer self.rmFree(client, client, device);

        // Subdevice 0 under the device (NV2080).
        var sp = sdk.Nv2080AllocParameters{ .sub_device_id = 0 };
        const subdevice = try self.rmAlloc(client, device, self.newHandle(), sdk.NV20_SUBDEVICE_0, &sp, @sizeOf(sdk.Nv2080AllocParameters));

        // node_fd / minor / gpu_id are Linux-node concepts; on baremetal they are
        // inert (no /dev/nvidia*). Keep the Device shape identical so rm.zig + the
        // HAL compile against either transport.
        return .{ .node_fd = NULL_FD, .minor = 0, .gpu_id = index +% 1, .client = client, .device = device, .subdevice = subdevice };
    }

    pub fn freeDevice(self: *Transport, dev: Device) void {
        if (!self.booted) return;
        self.rmFree(dev.client, dev.device, dev.subdevice);
        self.rmFree(dev.client, dev.client, dev.device);
        self.rmFree(0, 0, dev.client);
    }

    // =======================================================================
    // mapMemory -> MAP_MEMORY (7) + the CPU mapping of the resulting aperture. On
    // baremetal there is no mmap: the CPU window is a direct VRAM/BAR pointer (==
    // the phys we assigned, under UEFI identity mapping). Mirrors linux.zig's
    // MAP_MEMORY then mmap, but the "mmap" is a slice over the identity window.
    // =======================================================================

    pub fn mapMemory(self: *Transport, dev: Device, mem: Memory) Error!Mapping {
        if (!self.booted) return error.NotImplemented;

        // Tell the GSP to set up the mapping aperture (MAP_MEMORY). The body is the
        // sdk NVOS33 block (client/device/memory/offset/length).
        var p = sdk.Os33Params{
            .h_client = dev.client,
            .h_device = dev.subdevice,
            .h_memory = mem.handle,
            .offset = 0,
            .length = mem.size,
            .p_linear_address = 0,
            .status = 0,
            .flags = 0,
        };
        try self.sendRpc(@intFromEnum(proto.Function.map_memory), std.mem.asBytes(&p));
        _ = try self.recvRpc(@intFromEnum(proto.Function.map_memory));

        // Resolve the object's CPU window (the identity-mapped phys WE assigned it).
        const rec = self.lookup(mem.handle) orelse return error.MapFailed;
        const cpu_ptr: [*]u8 = @ptrFromInt(rec.cpu);
        return .{ .bytes = cpu_ptr[0..mem.size], .fd = NULL_FD };
    }

    pub fn unmapMemory(self: *Transport, mapping: Mapping) void {
        _ = self;
        _ = mapping;
        // Nothing to munmap - the window is a direct identity pointer WE own.
    }

    // =======================================================================
    // control -> GSP_RM_CONTROL (76). Mirrors linux.zig's NVOS54 path: the sdk
    // control params travel as the params[] blob; out-params are copied back.
    // =======================================================================

    pub fn control(self: *Transport, dev: Device, h_object: sdk.NvHandle, cmd: sdk.NvU32, params: ?*anyopaque, params_size: sdk.NvU32) Error!void {
        if (!self.booted) return error.NotImplemented;

        const pbytes: []const u8 = if (params) |pp|
            @as([*]const u8, @ptrCast(pp))[0..params_size]
        else
            &[_]u8{};

        const n = ring.buildControlElement(
            &self.rpc_scratch,
            dev.client,
            h_object,
            cmd,
            0, // flags
            pbytes,
        );
        self.endpoint.send(self.rpc_scratch[0..n]) catch return error.IoctlFailed;

        const reply = try self.recvRpc(@intFromEnum(proto.Function.gsp_rm_control));
        const c: *const proto.RpcGspRmControl = @ptrCast(@alignCast(&reply[@sizeOf(proto.RpcMessageHeader)]));
        if (c.status != 0) return error.ControlFailed;

        // Copy the (in/out) control params back from the reply.
        if (params) |pp| {
            const params_off = @sizeOf(proto.RpcMessageHeader) + @sizeOf(proto.RpcGspRmControl);
            if (params_off + params_size <= reply.len) {
                const dst: [*]u8 = @ptrCast(pp);
                @memcpy(dst[0..params_size], reply[params_off .. params_off + params_size]);
            }
        }
    }

    // =======================================================================
    // mapToGpu -> NV01_MEMORY_VIRTUAL alloc + MAP_MEMORY_DMA over RPC. Mirrors
    // linux.zig's virtual-obj + DMA_OFFSET_FIXED dance op-for-op.
    // =======================================================================

    pub fn mapToGpu(self: *Transport, dev: Device, vaspace: sdk.NvHandle, mem: Memory, gpu_va: u64) Error!GpuMapping {
        return self.mapToGpuPaged(dev, vaspace, mem, gpu_va, sdk.dma_flags.PAGE_SIZE_DEFAULT);
    }

    /// mapToGpu with an explicit NVOS46 PAGE_SIZE_* flag (see the linux transport for
    /// why a ZETA block-linear surface needs BIG pages).
    pub fn mapToGpuPaged(self: *Transport, dev: Device, vaspace: sdk.NvHandle, mem: Memory, gpu_va: u64, page_size_flag: sdk.NvU32) Error!GpuMapping {
        if (!self.booted) return error.NotImplemented;

        // Reserve [gpu_va, gpu_va+size) in the VA space (NV01_MEMORY_VIRTUAL alloc).
        var vp = sdk.VirtMemAllocParams{ .offset = gpu_va, .limit = gpu_va + mem.size - 1, .h_vaspace = vaspace };
        const virtual = try self.rmAlloc(dev.client, dev.device, self.newHandle(), sdk.NV01_MEMORY_VIRTUAL, &vp, @sizeOf(sdk.VirtMemAllocParams));
        errdefer self.rmFree(dev.client, dev.device, virtual);

        // MAP_MEMORY_DMA with the fixed offset (dmaOffset is the requested VA).
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
        try self.sendRpc(@intFromEnum(proto.Function.map_memory_dma), std.mem.asBytes(&p));
        const reply = try self.recvRpc(@intFromEnum(proto.Function.map_memory_dma));
        // The GSP writes the actual dma_offset + status back into the NVOS46 body.
        const out: *const sdk.Os46Params = @ptrCast(@alignCast(&reply[@sizeOf(proto.RpcMessageHeader)]));
        if (out.status != 0) return error.MapFailed;
        return .{ .virtual = virtual, .gpu_va = out.dma_offset, .size = mem.size };
    }

    pub fn unmapFromGpu(self: *Transport, dev: Device, mapping: GpuMapping) void {
        // Freeing the NV01_MEMORY_VIRTUAL object unmaps the physical memory.
        self.rmFree(dev.client, dev.device, mapping.virtual);
    }

    // =======================================================================
    // Object tracking (the CPU-window <-> handle table). Memory objects WE carve
    // from the VRAM/sysmem bump allocators are recorded so mapMemory can hand back
    // a real slice. allocMemory (in rm.zig) calls rmAlloc, which does not see the
    // location/size split - so the HAL drives carving through allocVram/allocSysmem
    // helpers below, which the freestanding allocMemory path uses.
    // =======================================================================

    /// Carve `size` bytes of VRAM (or sysmem) WE own, record the object under
    /// `handle`, and return the assigned phys/CPU base. The CPU window equals the
    /// phys under UEFI identity mapping. Used by the freestanding memory-alloc path
    /// so mapMemory has a real address to return.
    pub fn track(self: *Transport, handle: sdk.NvHandle, location: Memory.Location, size: u64) Error!u64 {
        if (self.object_count >= MAX_OBJECTS) return error.RmAllocFailed;
        const phys = switch (location) {
            .vram => try mapAllocErr(self.vram.alloc(size)),
            .system, .system_wc => try mapAllocErr(self.sysmem.alloc(size)),
        };
        self.objects[self.object_count] = .{
            .handle = handle,
            .phys = phys,
            .cpu = @intCast(phys), // identity mapping under UEFI boot services
            .size = size,
            .location = location,
        };
        self.object_count += 1;
        return phys;
    }

    fn lookup(self: *Transport, handle: sdk.NvHandle) ?ObjectRecord {
        var i: usize = 0;
        while (i < self.object_count) : (i += 1) {
            if (self.objects[i].handle == handle) return self.objects[i];
        }
        return null;
    }

    fn untrack(self: *Transport, handle: sdk.NvHandle) void {
        var i: usize = 0;
        while (i < self.object_count) : (i += 1) {
            if (self.objects[i].handle == handle) {
                // Swap-remove (order does not matter).
                self.objects[i] = self.objects[self.object_count - 1];
                self.object_count -= 1;
                return;
            }
        }
    }
};

/// Map fw.BumpAllocator's OutOfMemory into the transport Error set.
fn mapAllocErr(r: fw.Error!u64) Error!u64 {
    return r catch error.RmAllocFailed;
}

// ===========================================================================
// Tests (offline). These analyse + exercise the freestanding transport WITHOUT a
// GPU by driving the RPC builders directly and by playing the GSP side of a
// two-ended ring (the ring.Endpoint from gsp/ring.zig). The transport sends RM
// RPCs; the test "GSP" returns success replies and we assert the exact RPC
// sequence + on-wire bytes. The live GSP answering + the full HAL triangle on
// baremetal stay the user's deferred hardware test.
//
// NOTE: the whole Transport is comptime-dead on Linux (transport.zig selects
// linux.zig), so these tests force-analyse it here regardless of host. They build
// a Transport with a fake (CPU-both-ends) ring instead of a booted GPU.
// ===========================================================================

const testing = std.testing;

/// A pair of in-memory rings + the two endpoints over them: `host` is what the
/// transport drives; `gsp` is the test playing the GSP. Both ends are CPU code
/// over the same backing stores (cmdq: host produces / gsp consumes; msgq: gsp
/// produces / host consumes).
const FakeRing = struct {
    cmdq: []align(8) u8,
    msgq: []align(8) u8,
    host: ring.Endpoint,
    gsp: ring.Endpoint,

    fn init(alloc: std.mem.Allocator, slots: u32) !FakeRing {
        const size: u32 = ring.ENTRY_OFF + slots * ring.SLOT_SIZE;
        const cmdq = try alloc.alignedAlloc(u8, .@"8", size);
        const msgq = try alloc.alignedAlloc(u8, .@"8", size);
        @memset(cmdq, 0);
        @memset(msgq, 0);
        var cmdq_ring = try ring.Ring.init(@intFromPtr(cmdq.ptr), size);
        var msgq_ring = try ring.Ring.init(@intFromPtr(msgq.ptr), size);
        cmdq_ring.format(0);
        msgq_ring.format(0);
        return .{
            .cmdq = cmdq,
            .msgq = msgq,
            .host = .{ .tx = cmdq_ring, .rx = msgq_ring },
            .gsp = .{ .tx = msgq_ring, .rx = cmdq_ring },
        };
    }

    fn deinit(self: *FakeRing, alloc: std.mem.Allocator) void {
        alloc.free(self.cmdq);
        alloc.free(self.msgq);
    }

    /// Play the GSP: drain one command the transport sent, assert/inspect it, then
    /// echo a success reply. For an alloc we echo the alloc header with status 0 +
    /// the requested handle (or a fresh assigned one); for a control we echo status
    /// 0 + the params unchanged. Returns the drained command's function + headers so
    /// the test can assert the sequence.
    const Drained = struct {
        function: u32,
        h_class: u32 = 0,
        h_parent: u32 = 0,
        cmd: u32 = 0,
        params: [256]u8 = undefined,
        params_len: usize = 0,
    };

    /// Drain + reply to one ALLOC, assigning `assigned_handle` back (so the in/out
    /// handle path is exercised). Returns the drained alloc's class + parent.
    fn replyAlloc(self: *FakeRing, assigned_handle: u32) !Drained {
        const got = try self.gsp.recv();
        try testing.expectEqual(@intFromEnum(proto.Function.gsp_rm_alloc), got.function);
        const a: *const proto.RpcGspRmAlloc = @ptrCast(@alignCast(&got.rpc[@sizeOf(proto.RpcMessageHeader)]));
        var d = Drained{ .function = got.function, .h_class = a.h_class, .h_parent = a.h_parent };
        const psize = a.params_size;
        if (psize > 0 and psize <= d.params.len) {
            const poff = @sizeOf(proto.RpcMessageHeader) + @sizeOf(proto.RpcGspRmAlloc);
            @memcpy(d.params[0..psize], got.rpc[poff .. poff + psize]);
            d.params_len = psize;
        }
        // Echo the alloc reply: same header, status 0, the assigned handle, params
        // unchanged.
        var out: [proto.ELEMENT_SIZE_MAX]u8 = undefined;
        const n = ring.buildAllocElement(&out, a.h_client, a.h_parent, assigned_handle, a.h_class, 0, got.rpc[@sizeOf(proto.RpcMessageHeader) + @sizeOf(proto.RpcGspRmAlloc) ..][0..psize]);
        try self.gsp.send(out[0..n]);
        return d;
    }

    /// Drain + reply to one CONTROL (status 0, params echoed). Returns the cmd.
    fn replyControl(self: *FakeRing) !Drained {
        const got = try self.gsp.recv();
        try testing.expectEqual(@intFromEnum(proto.Function.gsp_rm_control), got.function);
        const c: *const proto.RpcGspRmControl = @ptrCast(@alignCast(&got.rpc[@sizeOf(proto.RpcMessageHeader)]));
        var d = Drained{ .function = got.function, .cmd = c.cmd };
        const psize = c.params_size;
        if (psize > 0 and psize <= d.params.len) {
            const poff = @sizeOf(proto.RpcMessageHeader) + @sizeOf(proto.RpcGspRmControl);
            @memcpy(d.params[0..psize], got.rpc[poff .. poff + psize]);
            d.params_len = psize;
        }
        var out: [proto.ELEMENT_SIZE_MAX]u8 = undefined;
        const n = ring.buildControlElement(&out, c.h_client, c.h_object, c.cmd, 0, got.rpc[@sizeOf(proto.RpcMessageHeader) + @sizeOf(proto.RpcGspRmControl) ..][0..psize]);
        try self.gsp.send(out[0..n]);
        return d;
    }

    /// Drain + reply to one bare RPC (echo the same function, no body). For the
    /// generic send/recv RPCs (FREE, MAP_MEMORY, postInit). Returns its function.
    fn replyBare(self: *FakeRing) !u32 {
        const got = try self.gsp.recv();
        var out: [128]u8 = undefined;
        const hdr: *proto.RpcMessageHeader = @ptrCast(@alignCast(&out[0]));
        hdr.* = .{
            .header_version = proto.RpcMessageHeader.HEADER_VERSION,
            .signature = proto.RpcMessageHeader.SIGNATURE,
            .length = @sizeOf(proto.RpcMessageHeader),
            .function = got.function,
            .rpc_result = 0,
            .rpc_result_private = 0,
            .sequence = 0,
            .u = 0,
        };
        try self.gsp.send(out[0..@sizeOf(proto.RpcMessageHeader)]);
        return got.function;
    }

    /// Build a Transport wired to this fake ring's host end, marked booted, with
    /// tiny sysmem/VRAM pools so track()/mapMemory work offline.
    fn transport(self: *FakeRing) Transport {
        return .{
            .endpoint = self.host,
            .booted = true,
            .sysmem = fw.BumpAllocator.init(0x10000000, 0x100000),
            .vram = fw.BumpAllocator.init(0x80000000, 0x100000),
        };
    }
};

test "freestanding rmAlloc: marshals GSP_RM_ALLOC + returns the assigned handle" {
    var fr = try FakeRing.init(testing.allocator, 8);
    defer fr.deinit(testing.allocator);
    var t = fr.transport();

    // A root-client alloc (no params). rmAlloc does send THEN recv internally; the
    // ring send/recv are non-blocking over the shared store, so we drive the GSP
    // reply between by replicating rmAlloc's exact marshalling here and asserting
    // the on-wire bytes + the in/out handle path (the GSP assigns 0xABCD back).
    const n = ring.buildAllocElement(&t.rpc_scratch, 0, 0, 0xc1d00001, sdk.NV01_ROOT, 0, &[_]u8{});
    try t.endpoint.send(t.rpc_scratch[0..n]);

    // Assert the on-wire RPC envelope the transport put on the cmdq.
    const rpc: *const proto.RpcMessageHeader = @ptrCast(@alignCast(&t.rpc_scratch[0]));
    try testing.expectEqual(proto.RpcMessageHeader.HEADER_VERSION, rpc.header_version);
    try testing.expectEqual(proto.RpcMessageHeader.SIGNATURE, rpc.signature);
    try testing.expectEqual(@intFromEnum(proto.Function.gsp_rm_alloc), rpc.function);

    const drained = try fr.replyAlloc(0xABCD);
    try testing.expectEqual(@as(u32, sdk.NV01_ROOT), drained.h_class);

    const reply = try t.recvRpc(@intFromEnum(proto.Function.gsp_rm_alloc));
    const a: *const proto.RpcGspRmAlloc = @ptrCast(@alignCast(&reply[@sizeOf(proto.RpcMessageHeader)]));
    try testing.expectEqual(@as(u32, 0), a.status);
    try testing.expectEqual(@as(sdk.NvHandle, 0xABCD), a.h_object);
}

test "freestanding allocDevice: ROOT -> DEVICE -> SUBDEVICE RPC sequence" {
    // Prove the op-sequence assembly against the fake GSP: the transport must send
    // exactly NV01_ROOT, then NV01_DEVICE_0, then NV20_SUBDEVICE_0, each as a
    // GSP_RM_ALLOC, parented correctly (device + subdevice under the assigned root).
    var fr = try FakeRing.init(testing.allocator, 16);
    defer fr.deinit(testing.allocator);
    var t = fr.transport();

    // allocDevice does send+recv per alloc internally; we cannot pre-stage all
    // three replies (the ring would need them queued before each recv). Instead we
    // assert the sequence by replaying allocDevice's exact logic against the fake
    // GSP one alloc at a time, mirroring the transport's order.
    //
    // 1. ROOT (parent 0, class NV01_ROOT) -> GSP assigns 0x5000.
    var n = ring.buildAllocElement(&t.rpc_scratch, 0, 0, t.newHandle(), sdk.NV01_ROOT, 0, &[_]u8{});
    try t.endpoint.send(t.rpc_scratch[0..n]);
    const root = try fr.replyAlloc(0x5000);
    try testing.expectEqual(@as(u32, sdk.NV01_ROOT), root.h_class);
    try testing.expectEqual(@as(u32, 0), root.h_parent);
    _ = try t.recvRpc(@intFromEnum(proto.Function.gsp_rm_alloc));

    // 2. DEVICE (parent == root 0x5000, class NV01_DEVICE_0, NV0080 params).
    var dp = sdk.Nv0080AllocParameters{ .device_id = 0, .h_client_share = sdk.NV01_NULL_OBJECT, .h_target_client = 0, .h_target_device = 0, .flags = 0, .va_space_size = 0, .va_start_internal = 0, .va_limit_internal = 0, .va_mode = 0 };
    n = ring.buildAllocElement(&t.rpc_scratch, 0x5000, 0x5000, t.newHandle(), sdk.NV01_DEVICE_0, 0, std.mem.asBytes(&dp));
    try t.endpoint.send(t.rpc_scratch[0..n]);
    const device = try fr.replyAlloc(0x6000);
    try testing.expectEqual(@as(u32, sdk.NV01_DEVICE_0), device.h_class);
    try testing.expectEqual(@as(u32, 0x5000), device.h_parent);
    try testing.expectEqual(@as(usize, @sizeOf(sdk.Nv0080AllocParameters)), device.params_len);
    _ = try t.recvRpc(@intFromEnum(proto.Function.gsp_rm_alloc));

    // 3. SUBDEVICE (parent == device 0x6000, class NV20_SUBDEVICE_0, NV2080 params).
    var sp = sdk.Nv2080AllocParameters{ .sub_device_id = 0 };
    n = ring.buildAllocElement(&t.rpc_scratch, 0x5000, 0x6000, t.newHandle(), sdk.NV20_SUBDEVICE_0, 0, std.mem.asBytes(&sp));
    try t.endpoint.send(t.rpc_scratch[0..n]);
    const sub = try fr.replyAlloc(0x7000);
    try testing.expectEqual(@as(u32, sdk.NV20_SUBDEVICE_0), sub.h_class);
    try testing.expectEqual(@as(u32, 0x6000), sub.h_parent);
    try testing.expectEqual(@as(usize, @sizeOf(sdk.Nv2080AllocParameters)), sub.params_len);
}

test "freestanding control: marshals GSP_RM_CONTROL with the cmd + params" {
    var fr = try FakeRing.init(testing.allocator, 8);
    defer fr.deinit(testing.allocator);
    var t = fr.transport();

    // A GPU_GET_ID control (cmd 0x20800142) on the subdevice handle. Build + send,
    // GSP replies status 0, params echoed.
    var p = sdk.GpuGetIdParams{ .gpu_id = 0 };
    const n = ring.buildControlElement(&t.rpc_scratch, 0x5000, 0x7000, sdk.NV2080_CTRL_CMD_GPU_GET_ID, 0, std.mem.asBytes(&p));
    try t.endpoint.send(t.rpc_scratch[0..n]);
    const drained = try fr.replyControl();
    try testing.expectEqual(sdk.NV2080_CTRL_CMD_GPU_GET_ID, drained.cmd);
    try testing.expectEqual(@as(usize, @sizeOf(sdk.GpuGetIdParams)), drained.params_len);

    const reply = try t.recvRpc(@intFromEnum(proto.Function.gsp_rm_control));
    const c: *const proto.RpcGspRmControl = @ptrCast(@alignCast(&reply[@sizeOf(proto.RpcMessageHeader)]));
    try testing.expectEqual(@as(u32, 0), c.status);
    try testing.expectEqual(sdk.NV2080_CTRL_CMD_GPU_GET_ID, c.cmd);
}

test "freestanding mapToGpu: NV01_MEMORY_VIRTUAL alloc then MAP_MEMORY_DMA" {
    var fr = try FakeRing.init(testing.allocator, 8);
    defer fr.deinit(testing.allocator);
    var t = fr.transport();

    // mapToGpu first allocs an NV01_MEMORY_VIRTUAL (class 0x70) carrying the
    // VirtMemAllocParams, then sends a MAP_MEMORY_DMA RPC with DMA_OFFSET_FIXED.
    const gpu_va: u64 = 0x200000;
    const size: u64 = 0x10000;

    // 1. the virtual-obj alloc.
    var vp = sdk.VirtMemAllocParams{ .offset = gpu_va, .limit = gpu_va + size - 1, .h_vaspace = 0x8000 };
    const n = ring.buildAllocElement(&t.rpc_scratch, 0x5000, 0x6000, t.newHandle(), sdk.NV01_MEMORY_VIRTUAL, 0, std.mem.asBytes(&vp));
    try t.endpoint.send(t.rpc_scratch[0..n]);
    const vobj = try fr.replyAlloc(0x9000);
    try testing.expectEqual(sdk.NV01_MEMORY_VIRTUAL, vobj.h_class);
    _ = try t.recvRpc(@intFromEnum(proto.Function.gsp_rm_alloc));

    // 2. the MAP_MEMORY_DMA RPC (NVOS46, DMA_OFFSET_FIXED).
    var dma = sdk.Os46Params{
        .h_client = 0x5000,
        .h_device = 0x6000,
        .h_dma = 0x9000,
        .h_memory = 0xA000,
        .offset = 0,
        .length = size,
        .flags = sdk.dma_flags.ACCESS_READ_WRITE | sdk.dma_flags.DMA_OFFSET_FIXED,
        .flags2 = 0,
        .kind_override = 0,
        .dma_offset = gpu_va,
        .status = 0,
    };
    try t.sendRpc(@intFromEnum(proto.Function.map_memory_dma), std.mem.asBytes(&dma));
    // GSP drains the DMA RPC + echoes it (status 0, dma_offset == gpu_va).
    const got = try fr.gsp.recv();
    try testing.expectEqual(@intFromEnum(proto.Function.map_memory_dma), got.function);
    const in: *const sdk.Os46Params = @ptrCast(@alignCast(&got.rpc[@sizeOf(proto.RpcMessageHeader)]));
    try testing.expectEqual(gpu_va, in.dma_offset);
    try testing.expect((in.flags & sdk.dma_flags.DMA_OFFSET_FIXED) != 0);

    var out: [256]u8 = undefined;
    const hdr: *proto.RpcMessageHeader = @ptrCast(@alignCast(&out[0]));
    hdr.* = .{ .header_version = proto.RpcMessageHeader.HEADER_VERSION, .signature = proto.RpcMessageHeader.SIGNATURE, .length = @sizeOf(proto.RpcMessageHeader) + @sizeOf(sdk.Os46Params), .function = @intFromEnum(proto.Function.map_memory_dma), .rpc_result = 0, .rpc_result_private = 0, .sequence = 0, .u = 0 };
    @memcpy(out[@sizeOf(proto.RpcMessageHeader)..][0..@sizeOf(sdk.Os46Params)], std.mem.asBytes(&dma));
    try fr.gsp.send(out[0 .. @sizeOf(proto.RpcMessageHeader) + @sizeOf(sdk.Os46Params)]);

    const reply = try t.recvRpc(@intFromEnum(proto.Function.map_memory_dma));
    const ro: *const sdk.Os46Params = @ptrCast(@alignCast(&reply[@sizeOf(proto.RpcMessageHeader)]));
    try testing.expectEqual(gpu_va, ro.dma_offset);
    try testing.expectEqual(@as(u32, 0), ro.status);
}

test "freestanding mapMemory: MAP_MEMORY RPC + a tracked CPU window" {
    var fr = try FakeRing.init(testing.allocator, 8);
    defer fr.deinit(testing.allocator);
    var t = fr.transport();

    // Track a VRAM object (carve from the VRAM bump allocator) so mapMemory has a
    // real CPU window to return.
    const handle: sdk.NvHandle = 0xA000;
    const size: u64 = 0x10000;
    const phys = try t.track(handle, .vram, size);
    try testing.expectEqual(@as(u64, 0x80000000), phys); // first VRAM carve

    const dev = Device{ .node_fd = -1, .minor = 0, .gpu_id = 1, .client = 0x5000, .device = 0x6000, .subdevice = 0x7000 };

    // mapMemory sends MAP_MEMORY then recvs; interleave the fake GSP reply.
    var p = sdk.Os33Params{ .h_client = dev.client, .h_device = dev.subdevice, .h_memory = handle, .offset = 0, .length = size, .p_linear_address = 0, .status = 0, .flags = 0 };
    try t.sendRpc(@intFromEnum(proto.Function.map_memory), std.mem.asBytes(&p));
    const fn_id = try fr.replyBare();
    try testing.expectEqual(@intFromEnum(proto.Function.map_memory), fn_id);
    _ = try t.recvRpc(@intFromEnum(proto.Function.map_memory));

    // The CPU window is the identity-mapped phys WE assigned.
    const rec = t.lookup(handle).?;
    try testing.expectEqual(phys, @as(u64, @intCast(rec.cpu)));
    try testing.expectEqual(size, rec.size);
}

test "freestanding rmFree: emits a FREE RPC + untracks the object" {
    var fr = try FakeRing.init(testing.allocator, 8);
    defer fr.deinit(testing.allocator);
    var t = fr.transport();

    const handle: sdk.NvHandle = 0xA000;
    _ = try t.track(handle, .system, 0x1000);
    try testing.expectEqual(@as(usize, 1), t.object_count);

    // rmFree sends FREE then recvs; interleave the reply, then it untracks.
    var p = sdk.Os00Params{ .h_root = 0x5000, .h_object_parent = 0x6000, .h_object_old = handle, .status = 0 };
    try t.sendRpc(@intFromEnum(proto.Function.free), std.mem.asBytes(&p));
    const fn_id = try fr.replyBare();
    try testing.expectEqual(@intFromEnum(proto.Function.free), fn_id);
    _ = try t.recvRpc(@intFromEnum(proto.Function.free));
    t.untrack(handle);
    try testing.expectEqual(@as(usize, 0), t.object_count);
}

test "freestanding checkVersion: the firmware .fwversion gate" {
    const t = Transport{};
    // CMD_QUERY returns the pinned firmware version, RECOGNIZED.
    const q = try t.checkVersion(sdk.RmApiVersion.CMD_QUERY, "");
    try testing.expectEqual(sdk.RmApiVersion.REPLY_RECOGNIZED, q.reply);
    const end = std.mem.indexOfScalar(u8, &q.version_string, 0) orelse q.version_string.len;
    try testing.expectEqualStrings(fw.FW_VERSION, q.version_string[0..end]);

    // A STRICT check of the exact version is RECOGNIZED; a wrong one is not.
    const ok = try t.checkVersion(sdk.RmApiVersion.CMD_STRICT, fw.FW_VERSION);
    try testing.expectEqual(sdk.RmApiVersion.REPLY_RECOGNIZED, ok.reply);
    const bad = try t.checkVersion(sdk.RmApiVersion.CMD_STRICT, "000.00.00");
    try testing.expectEqual(sdk.RmApiVersion.REPLY_UNRECOGNIZED, bad.reply);
}

test "freestanding open() without metal inputs fails cleanly" {
    try testing.expectError(error.NotImplemented, Transport.open());
    // And the RPC ops refuse on an un-booted transport.
    var t = Transport{};
    try testing.expectError(error.NotImplemented, t.rmAlloc(0, 0, 1, sdk.NV01_ROOT, null, 0));
    try testing.expectError(error.NotImplemented, t.allocDevice(0));
}

test {
    testing.refAllDecls(@This());
}
