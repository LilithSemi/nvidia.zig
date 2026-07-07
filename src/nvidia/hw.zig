//! Reusable, OS-agnostic GPU hardware-access layer: direct MMIO over a mapped
//! BAR plus chip identification off the boot registers. This is the foundation
//! the baremetal/freestanding RM (transport/freestanding.zig) grows into - it is
//! deliberately free of any UEFI, Linux, or kernel-RM dependency so the same code
//! serves the UEFI probe (milestone 1) and the direct RM path (milestone 4).

pub const mmio = @import("hw/mmio.zig");
pub const chip = @import("hw/chip.zig");

pub const Bar = mmio.Bar;
pub const ChipId = chip.ChipId;
pub const Architecture = chip.Architecture;

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
