//! OS-abstraction seam for the NVIDIA RM. `Transport` is the set of OS-primitive
//! RM operations (alloc/free, control, map/unmap, device bring-up) that differ
//! between platforms; the higher-level RM ops on rm.Client compose these and are
//! OS-agnostic.
//!
//! Linux (the current, no-root path) talks to the kernel RM via /dev/nvidia*
//! ioctls. Freestanding/baremetal drives the GPU directly over MMIO + boots the
//! GSP (the RM-over-GSP path). The backend is selected at comptime by the target
//! OS, so only the matching one is analysed/compiled.
//!
//! The freestanding backend covers BOTH a bare `.freestanding` target AND a UEFI
//! (`.uefi`) target: a UEFI app has no kernel RM (and cannot even compile the
//! linux.zig std.os.linux syscalls), so the RM-over-GSP transport backs it. The
//! `zig build uefi` target reports os.tag == .uefi, so it must route here too.

const target_os = @import("builtin").target.os.tag;
const is_freestanding = target_os == .freestanding or target_os == .uefi;

pub const Transport = if (is_freestanding)
    @import("transport/freestanding.zig").Transport
else
    @import("transport/linux.zig").Transport;

// The freestanding transport is comptime-dead on a non-freestanding build (the
// selection above picks linux.zig there), so Zig would never analyse it - and its
// RM-over-GSP tests would never run. It is pure, host-agnostic logic (the GSP ring
// + RPC envelopes over plain memory, no syscalls), so we force its tests to be
// COLLECTED on every target via a test-only reference. This keeps the freestanding
// RM-over-GSP path covered by `zig build test` on Linux while the live build still
// links only the selected Transport.
test {
    _ = @import("transport/freestanding.zig");
}

// Shared, OS-agnostic types, re-exported so rm.zig (and callers) have one source.
pub const types = @import("transport/types.zig");
pub const Error = types.Error;
pub const Device = types.Device;
pub const Memory = types.Memory;
pub const Mapping = types.Mapping;
pub const GpuMapping = types.GpuMapping;
pub const Channel = types.Channel;
pub const control_device = types.control_device;
pub const owner_tag = types.owner_tag;
