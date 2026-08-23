//! Userspace NVIDIA RM (Resource Manager) bindings for the NVIDIA open kernel
//! modules, ported from open-gpu-kernel-modules. Abstract and zero C deps - a
//! consumer (Prism's nvidia driver) builds device/memory/channel layers on top.
//!
//! Status: the version handshake (open /dev/nvidiactl, NV_ESC_CHECK_VERSION_STR)
//! is implemented and verified live against the 595.71.05 driver. Device/client
//! allocation, memory, and command submission are the next layers.

const std = @import("std");

pub const ioctl = @import("nvidia/ioctl.zig");
pub const sdk = @import("nvidia/sdk.zig");
pub const version = @import("nvidia/version.zig");
pub const rm = @import("nvidia/rm.zig");
pub const threed = @import("nvidia/threed.zig");
pub const sass = @import("nvidia/sass.zig");
pub const compute = @import("nvidia/compute.zig");
pub const copy = @import("nvidia/copy.zig");
pub const graphics = @import("nvidia/graphics.zig");
pub const hw = @import("nvidia/hw.zig");
pub const gsp = @import("nvidia/gsp/gsp.zig");
/// The OS-abstraction seam. `transport.Transport` is the comptime-selected backend
/// (Linux kernel-RM ioctls, or the freestanding RM-over-GSP path) - exposed so the
/// UEFI app can assemble the freestanding Transport's metal bring-up (openOnMetal).
pub const transport = @import("nvidia/transport.zig");
/// nvidia-drm node helpers: GEM_IMPORT_USERSPACE_MEMORY + PRIME_HANDLE_TO_FD.
/// Use memToDmaBuf(va, size) to turn a CPU mapping into a real dma-buf fd.
pub const drm = @import("nvidia/drm.zig");

/// Turn a CPU virtual address (from mapMemory on a .system allocation) into a
/// real Linux dma-buf fd via the nvidia-drm render node.
pub const memToDmaBuf = drm.memToDmaBuf;

pub const Client = rm.Client;
pub const Device = rm.Device;
pub const Memory = rm.Memory;
pub const Mapping = rm.Mapping;
pub const GpuMapping = rm.GpuMapping;
pub const Channel = rm.Channel;
pub const Queue = rm.Queue;
pub const Version = version.Version;
pub const Abi = version.Abi;

test {
    std.testing.refAllDecls(@This());
}
