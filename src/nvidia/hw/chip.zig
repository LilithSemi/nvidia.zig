//! NVIDIA GPU chip identification by direct register read, decoded from the boot
//! registers in the PMC (Master Control) block. This is the very first thing the
//! from-scratch RM must do over MMIO: read who the GPU is before devinit/GSP.
//!
//! Register layout matched to NVIDIA open-gpu-kernel-modules
//! (src/common/inc/swref/published/nv_ref.h):
//!
//!   NV_PMC_BOOT_0  (0x00000000)
//!     MINOR_REVISION   [3:0]
//!     MAJOR_REVISION   [7:4]
//!     ARCHITECTURE_1   [8]        MSB of the architecture (Hopper+ widened it)
//!     IMPLEMENTATION   [23:20]
//!     ARCHITECTURE_0   [28:24]    LSB(s) of the architecture
//!       architecture = (ARCHITECTURE_1 << 5) | ARCHITECTURE_0
//!
//!   NV_PMC_BOOT_42 (0x00000A00)   (Fermi+; the precise modern chip id)
//!     MINOR_EXTENDED_REVISION [11:8]
//!     MINOR_REVISION          [15:12]
//!     MAJOR_REVISION          [19:16]
//!     IMPLEMENTATION          [23:20]
//!     ARCHITECTURE            [29:24]
//!     CHIP_ID                 [29:20] (ARCHITECTURE..IMPLEMENTATION combined)
//!
//! Architecture enum values (ARCHITECTURE field, OGK NV_PMC_BOOT_42):
//!   TU10x=0x16 GA10x=0x17 GH100=0x18 AD10x=0x19 GB100=0x1A GB200=0x1B GR100=0x1C
//! The Blackwell consumer/RTX 50 series (GB20x, e.g. GB202) reports architecture
//! 0x1B. AMODEL (simulation) is 0x1F.

const std = @import("std");
const mmio = @import("mmio.zig");

pub const NV_PMC_BOOT_0: u32 = 0x00000000;
pub const NV_PMC_BOOT_42: u32 = 0x00000A00;

/// Decoded NVIDIA architecture family. Values are the raw ARCHITECTURE field.
pub const Architecture = enum(u8) {
    tu10x = 0x16, // Turing
    ga10x = 0x17, // Ampere
    gh100 = 0x18, // Hopper
    ad10x = 0x19, // Ada Lovelace
    gb100 = 0x1A, // Blackwell (datacenter, GB100)
    gb20x = 0x1B, // Blackwell (consumer GB20x / GB200 family)
    gr100 = 0x1C, // Rubin
    amodel = 0x1F, // simulation model
    _,

    /// Human-readable family name for the banner / log.
    pub fn name(self: Architecture) []const u8 {
        return switch (self) {
            .tu10x => "Turing (TU10x)",
            .ga10x => "Ampere (GA10x)",
            .gh100 => "Hopper (GH100)",
            .ad10x => "Ada Lovelace (AD10x)",
            .gb100 => "Blackwell (GB100)",
            .gb20x => "Blackwell (GB20x)",
            .gr100 => "Rubin (GR100)",
            .amodel => "AMODEL (simulation)",
            _ => "Unknown",
        };
    }

    pub fn isBlackwell(self: Architecture) bool {
        return self == .gb100 or self == .gb20x;
    }

    pub fn isAmpere(self: Architecture) bool {
        return self == .ga10x;
    }

    /// GSP-RM based families (Turing and later). The from-scratch baremetal RM
    /// must boot the GSP for ALL of these - Ampere (30-series) included - so the
    /// devinit + GSP + RM-over-GSP milestones target this whole set, with Ampere
    /// as the primary bring-up chip (best documented via nouveau / OGK).
    pub fn isGspBased(self: Architecture) bool {
        return switch (self) {
            .tu10x, .ga10x, .gh100, .ad10x, .gb100, .gb20x, .gr100 => true,
            else => false,
        };
    }
};

/// The decoded identity of a GPU, from a single read of each boot register.
pub const ChipId = struct {
    /// Raw NV_PMC_BOOT_0 value.
    boot0: u32,
    /// Raw NV_PMC_BOOT_42 value.
    boot42: u32,

    /// Architecture field from BOOT_0 = (ARCH_1 << 5) | ARCH_0.
    arch_boot0: u8,
    /// Architecture field from BOOT_42 [29:24] - the authoritative modern value.
    architecture: u8,
    /// IMPLEMENTATION [23:20] from BOOT_42 (the chip within the family).
    implementation: u8,
    /// CHIP_ID [29:20] from BOOT_42 (architecture..implementation combined).
    chip_id: u16,
    major_revision: u8,
    minor_revision: u8,

    pub fn arch(self: ChipId) Architecture {
        return @enumFromInt(self.architecture);
    }
};

/// BOOT_0 architecture = (ARCHITECTURE_1[8] << 5) | ARCHITECTURE_0[28:24].
pub fn archFromBoot0(boot0: u32) u8 {
    const arch0: u8 = @intCast((boot0 >> 24) & 0x1f);
    const arch1: u8 = @intCast((boot0 >> 8) & 0x1);
    return (arch1 << 5) | arch0;
}

/// Decode an already-read pair of boot registers (pure; unit-testable offline).
pub fn decode(boot0: u32, boot42: u32) ChipId {
    return .{
        .boot0 = boot0,
        .boot42 = boot42,
        .arch_boot0 = archFromBoot0(boot0),
        .architecture = @intCast((boot42 >> 24) & 0x3f),
        .implementation = @intCast((boot42 >> 20) & 0xf),
        .chip_id = @intCast((boot42 >> 20) & 0x3ff),
        .major_revision = @intCast((boot42 >> 16) & 0xf),
        .minor_revision = @intCast((boot42 >> 12) & 0xf),
    };
}

/// Read + decode the chip id straight off a mapped BAR0 aperture. Read-only.
/// This is what milestone 4's freestanding transport calls on the live GPU.
pub fn identify(bar0: mmio.Bar) ChipId {
    const boot0 = bar0.read32(NV_PMC_BOOT_0);
    const boot42 = bar0.read32(NV_PMC_BOOT_42);
    return decode(boot0, boot42);
}

test "decode a Blackwell GB20x boot register pair" {
    // Synthetic BOOT_42 for a GB20x: ARCH=0x1B, IMPL=0x2, rev fields set.
    // [29:24]=0x1B, [23:20]=0x2 -> CHIP_ID [29:20] = 0x1B2.
    const boot42: u32 = (0x1B << 24) | (0x2 << 20) | (0x1 << 16) | (0x0 << 12);
    // BOOT_0: ARCH_0[28:24]=0x1B, ARCH_1[8]=0, IMPL[23:20]=0x2, rev a1.
    const boot0: u32 = (0x1B << 24) | (0x2 << 20) | (0xa << 4) | 0x1;
    const id = decode(boot0, boot42);
    try std.testing.expectEqual(@as(u8, 0x1B), id.architecture);
    try std.testing.expectEqual(Architecture.gb20x, id.arch());
    try std.testing.expect(id.arch().isBlackwell());
    try std.testing.expectEqual(@as(u8, 0x2), id.implementation);
    try std.testing.expectEqual(@as(u16, 0x1B2), id.chip_id);
    try std.testing.expectEqual(@as(u8, 0x1B), id.arch_boot0);
    // major/minor revision come from BOOT_42 [19:16]/[15:12].
    try std.testing.expectEqual(@as(u8, 0x1), id.major_revision);
    try std.testing.expectEqual(@as(u8, 0x0), id.minor_revision);
}

test "decode an Ampere GA10x (30-series) boot register pair" {
    // GA104 (RTX 3070): ARCH=0x17, IMPL=0x4 -> CHIP_ID [29:20] = 0x174.
    const boot42: u32 = (0x17 << 24) | (0x4 << 20) | (0x1 << 16);
    const boot0: u32 = (0x17 << 24) | (0x4 << 20) | (0xa << 4) | 0x1;
    const id = decode(boot0, boot42);
    try std.testing.expectEqual(@as(u8, 0x17), id.architecture);
    try std.testing.expectEqual(Architecture.ga10x, id.arch());
    try std.testing.expect(id.arch().isAmpere());
    try std.testing.expect(id.arch().isGspBased());
    try std.testing.expect(!id.arch().isBlackwell());
    try std.testing.expectEqual(@as(u8, 0x4), id.implementation);
    try std.testing.expectEqual(@as(u16, 0x174), id.chip_id);
}

test "decode 20-series Turing and 40-series Ada boot registers" {
    // TU104 (RTX 2080): ARCH=0x16, IMPL=0x4 -> CHIP_ID 0x164.
    const tu = decode((0x16 << 24) | (0x4 << 20), (0x16 << 24) | (0x4 << 20));
    try std.testing.expectEqual(Architecture.tu10x, tu.arch());
    try std.testing.expect(tu.arch().isGspBased());
    try std.testing.expect(!tu.arch().isAmpere() and !tu.arch().isBlackwell());
    try std.testing.expectEqual(@as(u16, 0x164), tu.chip_id);
    // AD104 (RTX 4070): ARCH=0x19, IMPL=0x4 -> CHIP_ID 0x194.
    const ad = decode((0x19 << 24) | (0x4 << 20), (0x19 << 24) | (0x4 << 20));
    try std.testing.expectEqual(Architecture.ad10x, ad.arch());
    try std.testing.expect(ad.arch().isGspBased());
    try std.testing.expectEqual(@as(u16, 0x194), ad.chip_id);
}

test "archFromBoot0 folds in the high architecture bit" {
    // Hopper-style: ARCH_1 set (bit 8), ARCH_0 = 0x18 -> 0x38.
    const boot0: u32 = (0x18 << 24) | (1 << 8);
    try std.testing.expectEqual(@as(u8, 0x38), archFromBoot0(boot0));
}

test "identify reads the right offsets off a fake bar" {
    var backing = [_]u8{0} ** 0x1000;
    const bar = mmio.Bar.init(@intFromPtr(&backing), backing.len);
    const p0: *volatile u32 = @ptrFromInt(bar.base + NV_PMC_BOOT_0);
    const p42: *volatile u32 = @ptrFromInt(bar.base + NV_PMC_BOOT_42);
    p0.* = (0x1B << 24);
    p42.* = (0x1B << 24) | (0x4 << 20);
    const id = identify(bar);
    try std.testing.expectEqual(Architecture.gb20x, id.arch());
    try std.testing.expectEqual(@as(u8, 0x4), id.implementation);
}
