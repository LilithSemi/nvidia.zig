//! Shared, OS-agnostic types used by both the Client (rm.zig) and every
//! Transport backend (transport/linux.zig, transport/freestanding.zig). These
//! describe RM objects and CPU/GPU mappings independent of how the bytes are
//! actually moved to the kernel/hardware.

const std = @import("std");
const sdk = @import("../sdk.zig");

/// OpenFailed / NoDevice mean "couldn't get into the device" (skip-worthy in
/// tests); IoctlFailed / RmAllocFailed mean an operation failed on a reachable
/// device (a real error). NotImplemented is returned by the freestanding stub.
pub const Error = error{ OpenFailed, IoctlFailed, BadVersion, RmAllocFailed, NoDevice, MapFailed, ControlFailed, NotImplemented };

pub const control_device = "/dev/nvidiactl";

/// Memory owner tag ("PRSM"), used for RM allocation tracking/debug.
pub const owner_tag: sdk.NvU32 = 0x5052534d;

/// A brought-up GPU: the registered per-GPU node fd plus the RM object handles
/// for client/device/subdevice. Free with Client.freeDevice.
pub const Device = struct {
    /// /dev/nvidiaN, opened and registered to the control fd. Must stay open
    /// for the RM objects to remain valid.
    node_fd: std.posix.fd_t,
    minor: sdk.NvU32, // device-node minor (/dev/nvidia{minor})
    gpu_id: sdk.NvU32,
    client: sdk.NvHandle,
    device: sdk.NvHandle,
    subdevice: sdk.NvHandle,
};

/// A memory object allocated on a Device. `handle` is the RM object handle.
pub const Memory = struct {
    handle: sdk.NvHandle,
    size: u64,
    location: Location,

    /// system: cached host RAM. system_wc: write-combining host RAM - use for
    /// buffers the CPU writes and the GPU also writes (e.g. a framebuffer), since
    /// NVIDIA's GPU sysmem writes do not snoop the CPU cache and dirty cached CPU
    /// lines would otherwise shadow the GPU's writes. vram: device memory.
    pub const Location = enum { system, system_wc, vram };
};

/// A CPU mapping of a Memory object (with its dedicated mmap-context fd).
pub const Mapping = struct {
    bytes: []u8,
    fd: std.posix.fd_t,
};

/// A physical memory object bound into a GPU VA space at `gpu_va`.
pub const GpuMapping = struct {
    virtual: sdk.NvHandle, // the NV01_MEMORY_VIRTUAL object holding the binding
    gpu_va: u64,
    size: u64,
};

/// An allocated GPFIFO channel: its RM handle plus the GPU VA + size of its
/// GPFIFO ring. Submission (USERD doorbell + GPFIFO entries) builds on this.
pub const Channel = struct {
    handle: sdk.NvHandle,
    gpfifo_gpu_va: u64,
    gpfifo_entries: u32,
};
