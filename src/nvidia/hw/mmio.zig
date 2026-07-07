//! Direct GPU MMIO register access. This is the reusable, OS-agnostic piece the
//! freestanding/baremetal RM will build on: once BAR0 (the 16 MiB register
//! aperture) is mapped, the from-scratch RM reads and writes hardware registers
//! through here. Milestone 1 only READS - the UEFI probe maps BAR0 and decodes
//! the chip id without touching any register.
//!
//! Nothing here depends on UEFI, Linux, or the kernel RM; it is pure pointer
//! arithmetic over a base address, so transport/freestanding.zig can adopt it
//! unchanged when the direct path is wired in.

const std = @import("std");

/// A mapped GPU MMIO aperture (typically BAR0). `base` is the CPU-visible
/// virtual (== physical, under UEFI identity mapping) address of the aperture;
/// `size` is its byte length. All accesses are volatile - the compiler must not
/// elide or reorder them, since they hit real hardware.
pub const Bar = struct {
    base: usize,
    size: usize,

    pub fn init(base: usize, size: usize) Bar {
        return .{ .base = base, .size = size };
    }

    inline fn ptr(self: Bar, comptime T: type, off: u32) *volatile T {
        std.debug.assert(off + @sizeOf(T) <= self.size);
        return @ptrFromInt(self.base + off);
    }

    /// Read a 32-bit register at byte offset `off` from the aperture base.
    pub inline fn read32(self: Bar, off: u32) u32 {
        return self.ptr(u32, off).*;
    }

    /// Read a 16-bit register.
    pub inline fn read16(self: Bar, off: u32) u16 {
        return self.ptr(u16, off).*;
    }

    /// Read an 8-bit register.
    pub inline fn read8(self: Bar, off: u32) u8 {
        return self.ptr(u8, off).*;
    }

    /// Read a 64-bit register. NVIDIA registers are 32-bit; this is for the rare
    /// 64-bit aperture/scratch reads. On hardware the two 32-bit halves may not be
    /// atomic, so prefer two read32s when ordering matters; provided for convenience.
    pub inline fn read64(self: Bar, off: u32) u64 {
        return self.ptr(u64, off).*;
    }

    /// Write a 32-bit register. NOT used in milestone 1 (the probe is read-only);
    /// present so milestone 2 (devinit) can drive registers through the same Bar.
    pub inline fn write32(self: Bar, off: u32, value: u32) void {
        self.ptr(u32, off).* = value;
    }

    /// Write a 64-bit register.
    pub inline fn write64(self: Bar, off: u32, value: u64) void {
        self.ptr(u64, off).* = value;
    }

    /// Write a 16-bit register.
    pub inline fn write16(self: Bar, off: u32, value: u16) void {
        self.ptr(u16, off).* = value;
    }

    /// Write an 8-bit register.
    pub inline fn write8(self: Bar, off: u32, value: u8) void {
        self.ptr(u8, off).* = value;
    }

    /// Read-modify-write a 32-bit register: clear the bits in `mask`, then set the
    /// bits in `value` (which must already be confined to `mask`). This is the
    /// nvkm_falcon_mask primitive the FALCON sequences lean on.
    pub inline fn mask32(self: Bar, off: u32, m: u32, value: u32) void {
        const p = self.ptr(u32, off);
        p.* = (p.* & ~m) | value;
    }
};

test "bar offset math is identity over base" {
    var backing: [4096]u8 = undefined;
    const bar = Bar.init(@intFromPtr(&backing), backing.len);
    // Write through a raw pointer, read through the Bar - proves the addressing.
    const p: *volatile u32 = @ptrFromInt(bar.base + 0x40);
    p.* = 0xDEADBEEF;
    try std.testing.expectEqual(@as(u32, 0xDEADBEEF), bar.read32(0x40));
}
