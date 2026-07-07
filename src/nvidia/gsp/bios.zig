//! VBIOS read + parse (plan phase P2): read the GPU ROM image, parse the BIT
//! table, extract the FWSEC microcode and patch it with the FRTS command so the
//! GSP falcon (run in legacy mode in P4) carves the WPR2/FRTS region at the top
//! of VRAM. Almost all of this is pure byte logic and is fully offline unit-
//! testable on an injected ROM buffer; only the live PROM/PRAMIN reads and the
//! FWSEC RUN on the falcon are hardware-only (deferred).
//!
//! References (matched field-for-field):
//!   - VBIOS shadow read paths: nouveau nvkm/subdev/bios/{shadow,shadowrom,
//!     shadowramin}.c. PROM aperture BAR0 + 0x300000 (1 MiB); PRAMIN fallback
//!     (GA100+) RAMIN ptr in scratch 0x820c04, window 0x001700, image at 0x700000.
//!   - BIT parse: nvkm/subdev/bios/{base,bit}.c. Scan for "\xff\xb8""BIT"
//!     (nvbios_findstr), header at bit_offset (+9 entry size, +10 count, +12
//!     entries), each bit_entry = id(+0,u8) version(+1,u8) length(+2,u16)
//!     offset(+4,u16). devinit table is BIT id 'I'; the FWSEC/PMU ucode table is
//!     reached via BIT id 'p' (PMU, nvkm/subdev/bios/pmu.c).
//!   - FWSEC extraction + FRTS patch: nvkm/subdev/gsp/fwsec.c. The PMU table
//!     lists falcon ucodes; the FWSEC ucode is application type 0x85. Its nvfw
//!     FALCON ucode descriptor gives the DMEM image layout; the application
//!     interface table inside DMEM holds the DMEMMAPPER application (ID 0x4),
//!     which we patch with nvfw_fwsec_frts_cmd (init_cmd FRTS 0x15, region type
//!     FB 0x2, addr>>12, size>>12).
//!
//! DEVINIT FINDING (the plan's riskiest P2 open question): we do NOT need to run
//! the VBIOS devinit-script interpreter ourselves on the GSP path. On Turing+
//! (all our targets), nouveau's GSP bring-up performs device init by running
//! FWSEC (FWSEC-SB / FWSEC-FRTS) on the GSP falcon and then handing the rest to
//! GSP-RM once it boots - nvkm_gsp_fwsec_sb + nvkm_gsp_fwsec_frts replace the old
//! nvbios_init() devinit-script execution that the pre-GSP (Maxwell/Pascal) path
//! used. The devinit BIT table ('I') is still located (so it can be pointed at if
//! ever needed), but its script is not interpreted by us: FWSEC carves FRTS/WPR2
//! and GSP-RM completes clock/MC/FB init. See nvkm/subdev/gsp/fwsec.c
//! (nvkm_gsp_fwsec_frts / nvkm_gsp_fwsec_sb) and gsp/tu102.c (the init order),
//! versus the legacy nvbios_init() path that GSP chips skip.
//!
//! Freestanding-safe: the PROM/PRAMIN reads go through hw.Bar (volatile MMIO), so
//! they compile on the UEFI/freestanding target; they only RUN on metal. No
//! std.fs / std.Io is pulled in - the parse/patch logic takes the ROM bytes as a
//! plain slice, so tests inject a synthetic (or, if reachable, a real) ROM buffer.

const std = @import("std");
const hw = @import("../hw/mmio.zig");

const readInt = std.mem.readInt;
const writeInt = std.mem.writeInt;

pub const Error = error{
    /// The ROM image had no valid "\x55\xAA" PCIR signature.
    NoPciRom,
    /// The "\xff\xb8""BIT" token was not found in the image.
    NoBitToken,
    /// A header/entry/offset ran past the end of the ROM buffer.
    Truncated,
    /// The requested BIT entry id was not present.
    BitEntryNotFound,
    /// The PMU table (BIT 'p') was missing or the wrong version/length.
    NoPmuTable,
    /// No FWSEC ucode (application type 0x85) in the PMU table.
    FwsecNotFound,
    /// The FWSEC nvfw FALCON ucode descriptor was an unsupported version.
    UnsupportedUcodeVersion,
    /// The DMEMMAPPER application (ID 0x4) was not in the appif table.
    DmemMapperNotFound,
    /// The provided output buffer was too small.
    BufferTooSmall,
};

// ===========================================================================
// 1. VBIOS SHADOW read (hardware side, freestanding-clean, deferred run)
// ===========================================================================

/// PROM aperture: a 1 MiB window into the ROM at BAR0 + 0x300000. (shadowrom.c)
pub const PROM_OFFSET: u32 = 0x300000;
pub const PROM_SIZE: u32 = 0x100000; // 1 MiB

/// PRAMIN fallback (GA100+). (shadowramin.c)
/// The RAMIN base pointer lives in a PBUS scratch register; the PRAMIN window is
/// programmed at 0x001700 and the ROM image is read out of the window at 0x700000.
pub const PRAMIN_SCRATCH_PTR: u32 = 0x820c04;
pub const PRAMIN_WINDOW: u32 = 0x001700;
pub const PRAMIN_IMAGE: u32 = 0x700000;
pub const PRAMIN_SIZE: u32 = 0x100000; // 1 MiB window

/// A ROM signature: every PCI option ROM image starts with 0x55 0xAA.
pub const PCIR_SIG0: u8 = 0x55;
pub const PCIR_SIG1: u8 = 0xAA;

/// The BIT token, preceded by 0xff 0xb8 (nvbios_findstr "\xff\xb8""BIT").
pub const BIT_TOKEN = [_]u8{ 0xff, 0xb8, 'B', 'I', 'T' };

/// Copy the VBIOS ROM image out of the PROM aperture into `out`, returning the
/// filled slice. HARDWARE-ONLY at runtime (the reads are volatile MMIO), but
/// compiles freestanding. Reads `out.len` (clamped to PROM_SIZE) bytes 32 bits at
/// a time - the PROM window is not byte-addressable on all chips, so we read u32s
/// and split. (shadowrom.c nvbios_rd32 over the PROM aperture.)
pub fn readVbiosProm(bar: hw.Bar, out: []u8) []u8 {
    return readVbiosWindow(bar, PROM_OFFSET, out);
}

/// Copy the VBIOS ROM image out of the PRAMIN window (GA100+ fallback). The
/// caller must already have pointed the PRAMIN window at the ROM (program
/// PRAMIN_WINDOW from the RAMIN scratch ptr). HARDWARE-ONLY at runtime.
pub fn readVbiosPramin(bar: hw.Bar, out: []u8) []u8 {
    return readVbiosWindow(bar, PRAMIN_IMAGE, out);
}

/// Read the RAMIN pointer used to position the PRAMIN window (GA100+). On metal
/// this scratch holds the ROM's RAMIN base; the caller shifts/writes it into the
/// PRAMIN window register before readVbiosPramin. Pure register read.
pub fn pramInPointer(bar: hw.Bar) u32 {
    return bar.read32(PRAMIN_SCRATCH_PTR);
}

fn readVbiosWindow(bar: hw.Bar, base: u32, out: []u8) []u8 {
    const n: u32 = @intCast(@min(out.len, @as(usize, PROM_SIZE)));
    var i: u32 = 0;
    // Read full 32-bit words.
    while (i + 4 <= n) : (i += 4) {
        writeInt(u32, out[i..][0..4], bar.read32(base + i), .little);
    }
    // Tail bytes (rare): read one final word and split.
    if (i < n) {
        var tail: [4]u8 = undefined;
        writeInt(u32, &tail, bar.read32(base + i), .little);
        var k: u32 = 0;
        while (i < n) : (i += 1) {
            out[i] = tail[k];
            k += 1;
        }
    }
    return out[0..n];
}

/// Validate a ROM image: it must begin with the 0x55 0xAA PCIR signature and
/// contain the "\xff\xb8""BIT" token. Returns the BIT-token offset on success.
/// This is the offline gate the parser runs first.
pub fn validate(rom: []const u8) Error!u32 {
    if (rom.len < 2 or rom[0] != PCIR_SIG0 or rom[1] != PCIR_SIG1) return Error.NoPciRom;
    return findBitToken(rom);
}

/// Linear scan for "\xff\xb8""BIT" (nvbios_findstr). Returns the offset of the
/// 0xff byte (the BIT header starts there, matching nouveau's bit_offset).
pub fn findBitToken(rom: []const u8) Error!u32 {
    if (rom.len < BIT_TOKEN.len) return Error.NoBitToken;
    var i: usize = 0;
    const last = rom.len - BIT_TOKEN.len;
    while (i <= last) : (i += 1) {
        if (std.mem.eql(u8, rom[i .. i + BIT_TOKEN.len], &BIT_TOKEN)) {
            return @intCast(i);
        }
    }
    return Error.NoBitToken;
}

// ===========================================================================
// 2. BIT parse (pure offline byte logic - the high-value testable part)
// ===========================================================================

/// A single BIT table entry. Layout (nvkm/subdev/bios/bit.c bit_entry):
///   id(+0,u8) version(+1,u8) length(+2,u16 LE) offset(+4,u16 LE)
/// `offset` is an absolute offset into the ROM image where the table lives.
pub const BitEntry = struct {
    id: u8,
    version: u8,
    length: u16,
    offset: u16,
};

/// A parsed BIT header. The header sits at the "\xff\xb8""BIT" token:
///   "\xff\xb8""BIT\0" then a header. nouveau reads:
///   entry size  @ bit_offset + 9  (u8)
///   entry count @ bit_offset + 10 (u8)
///   entries     @ bit_offset + 12
pub const Bit = struct {
    rom: []const u8,
    /// Offset of the 0xff byte (== nouveau bit_offset).
    bit_offset: u32,
    entry_size: u8,
    entry_count: u8,
    entries_offset: u32,

    pub const HDR_ENTRY_SIZE_OFF: u32 = 9;
    pub const HDR_ENTRY_COUNT_OFF: u32 = 10;
    pub const HDR_ENTRIES_OFF: u32 = 12;

    /// Well-known BIT entry ids.
    pub const ID_DEVINIT: u8 = 'I'; // devinit tables (script + the 'I' table)
    pub const ID_PMU: u8 = 'p'; // PMU/FALCON ucode table (carries FWSEC)

    /// Parse the BIT header located via `findBitToken`. Validates the header and
    /// the entry table fit inside the ROM.
    pub fn parse(rom: []const u8) Error!Bit {
        const bit_offset = try findBitToken(rom);
        return parseAt(rom, bit_offset);
    }

    /// Parse the BIT header at a known token offset (avoids re-scanning).
    pub fn parseAt(rom: []const u8, bit_offset: u32) Error!Bit {
        const count_end = bit_offset + HDR_ENTRY_COUNT_OFF + 1;
        if (count_end > rom.len) return Error.Truncated;
        const entry_size = rom[bit_offset + HDR_ENTRY_SIZE_OFF];
        const entry_count = rom[bit_offset + HDR_ENTRY_COUNT_OFF];
        const entries_offset = bit_offset + HDR_ENTRIES_OFF;
        // Entries must be at least 6 bytes (id+ver+len+off) and fit in the ROM.
        if (entry_size < 6) return Error.Truncated;
        const table_end = @as(u64, entries_offset) +
            @as(u64, entry_size) * @as(u64, entry_count);
        if (table_end > rom.len) return Error.Truncated;
        return .{
            .rom = rom,
            .bit_offset = bit_offset,
            .entry_size = entry_size,
            .entry_count = entry_count,
            .entries_offset = entries_offset,
        };
    }

    /// Read the i-th BIT entry (0 <= i < entry_count).
    pub fn entryAt(self: Bit, i: usize) Error!BitEntry {
        if (i >= self.entry_count) return Error.BitEntryNotFound;
        const base = self.entries_offset + @as(u32, @intCast(i)) * self.entry_size;
        if (base + 6 > self.rom.len) return Error.Truncated;
        return .{
            .id = self.rom[base + 0],
            .version = self.rom[base + 1],
            .length = readInt(u16, self.rom[base + 2 ..][0..2], .little),
            .offset = readInt(u16, self.rom[base + 4 ..][0..2], .little),
        };
    }

    /// Find the first BIT entry whose id matches. (nvkm bit_entry().)
    pub fn entry(self: Bit, id: u8) Error!BitEntry {
        var i: usize = 0;
        while (i < self.entry_count) : (i += 1) {
            const e = try self.entryAt(i);
            if (e.id == id) return e;
        }
        return Error.BitEntryNotFound;
    }

    /// Resolve the devinit ('I') table offset.
    pub fn devinitOffset(self: Bit) Error!u16 {
        return (try self.entry(ID_DEVINIT)).offset;
    }
};

// ===========================================================================
// 3. PMU table -> FWSEC ucode location (nvkm/subdev/bios/pmu.c)
// ===========================================================================

/// The PMU table is reached from BIT entry 'p' (version 2, length >= 4): the
/// entry's data holds a u32 pointing at the PMU table base. The table header:
///   version(+0,u8) header_size(+1,u8) entry_length(+2,u8) entry_count(+3,u8)
/// Each entry: type(+0,u8) ... data(+2,u32). FWSEC is type 0x85.
pub const PmuTable = struct {
    rom: []const u8,
    base: u32,
    version: u8,
    header_size: u8,
    entry_length: u8,
    entry_count: u8,

    pub const FWSEC_APP_TYPE: u8 = 0x85;

    pub const ENTRY_TYPE_OFF: u32 = 0x00;
    pub const ENTRY_DATA_OFF: u32 = 0x02;

    /// Locate + parse the PMU table from a parsed BIT.
    pub fn parse(bit: Bit) Error!PmuTable {
        const e = bit.entry(Bit.ID_PMU) catch return Error.NoPmuTable;
        if (e.version != 2 or e.length < 4) return Error.NoPmuTable;
        const rom = bit.rom;
        if (@as(u32, e.offset) + 4 > rom.len) return Error.NoPmuTable;
        const base = readInt(u32, rom[e.offset..][0..4], .little);
        if (base + 4 > rom.len) return Error.NoPmuTable;
        return .{
            .rom = rom,
            .base = base,
            .version = rom[base + 0],
            .header_size = rom[base + 1],
            .entry_length = rom[base + 2],
            .entry_count = rom[base + 3],
        };
    }

    pub fn entryOffset(self: PmuTable, i: usize) u32 {
        return self.base + self.header_size +
            @as(u32, @intCast(i)) * self.entry_length;
    }

    /// Read the i-th PMU entry's (type, data-pointer).
    pub fn entryAt(self: PmuTable, i: usize) Error!struct { type: u8, data: u32 } {
        if (i >= self.entry_count) return Error.FwsecNotFound;
        const off = self.entryOffset(i);
        if (off + 6 > self.rom.len) return Error.Truncated;
        return .{
            .type = self.rom[off + ENTRY_TYPE_OFF],
            .data = readInt(u32, self.rom[off + ENTRY_DATA_OFF ..][0..4], .little),
        };
    }

    /// Find the FWSEC ucode descriptor pointer (application type 0x85).
    pub fn fwsecDescOffset(self: PmuTable) Error!u32 {
        var i: usize = 0;
        while (i < self.entry_count) : (i += 1) {
            const e = try self.entryAt(i);
            if (e.type == FWSEC_APP_TYPE) return e.data;
        }
        return Error.FwsecNotFound;
    }
};

// ===========================================================================
// 4. nvfw FALCON ucode descriptor + application interface (extern, layout-asserted)
// ===========================================================================

/// nvfw FALCON ucode descriptor, V2 variant (nvfw_falcon_ucode_desc v2 in
/// nvkm/subdev/gsp/fwsec.c). The FWSEC type-0x85 ucode on Turing/Ampere uses v2;
/// the v3 layout (PKC-signed) appears on later parts. We support v2 (Hdr version
/// 2) which is what 595.71.05 GA10x ships. The header `Hdr` packs:
///   bit 0   = valid
///   bits 8..15  = version  (we require 2)
///   bits 16..31 = size
pub const NvfwFalconUcodeDescV2 = extern struct {
    Hdr: u32, // +0x00
    StoredSize: u32, // +0x04
    UncompressedSize: u32, // +0x08
    VirtualEntry: u32, // +0x0c
    InterfaceOffset: u32, // +0x10
    IMEMPhysBase: u32, // +0x14
    IMEMLoadSize: u32, // +0x18
    IMEMVirtBase: u32, // +0x1c
    IMEMSecBase: u32, // +0x20
    IMEMSecSize: u32, // +0x24
    DMEMOffset: u32, // +0x28
    DMEMPhysBase: u32, // +0x2c
    DMEMLoadSize: u32, // +0x30
    altIMEMLoadSize: u32, // +0x34
    altDMEMLoadSize: u32, // +0x38

    pub fn version(self: NvfwFalconUcodeDescV2) u8 {
        return @intCast((self.Hdr & 0x0000ff00) >> 8);
    }
};

/// Application interface header (union nvfw_falcon_appif_hdr v1). Lives at
/// DMEM image base + InterfaceOffset.
///   ver(+0,u8) hdr(+1,u8 = size of this header) len(+2,u8 = size of each
///   entry) cnt(+3,u8 = entry count)
pub const NvfwFalconAppifHdr = extern struct {
    ver: u8, // +0x00
    hdr: u8, // +0x01
    len: u8, // +0x02
    cnt: u8, // +0x03
};

/// Application interface entry (union nvfw_falcon_appif v1): id + dmem_base.
///   id(+0,u32) dmem_base(+4,u32 = offset into DMEM of this application's data)
pub const NvfwFalconAppif = extern struct {
    id: u32, // +0x00
    dmem_base: u32, // +0x04

    pub const ID_DMEMMAPPER: u32 = 0x4;
};

/// DMEMMAPPER application descriptor V3 (union nvfw_falcon_appif_dmemmapper v3).
/// `init_cmd` selects the FWSEC command; `cmd_in_buffer_offset` points (within
/// DMEM) at where we write the nvfw_fwsec_frts_cmd.
pub const NvfwFalconAppifDmemmapperV3 = extern struct {
    signature: u32, // +0x00
    version: u16, // +0x04
    size: u16, // +0x06
    cmd_in_buffer_offset: u32, // +0x08
    cmd_in_buffer_size: u32, // +0x0c
    cmd_out_buffer_offset: u32, // +0x10
    cmd_out_buffer_size: u32, // +0x14
    nvf_img_data_buffer_offset: u32, // +0x18
    nvf_img_data_buffer_size: u32, // +0x1c
    printf_buffer_hdr: u32, // +0x20
    ucode_build_time_stamp: u32, // +0x24
    ucode_signature: u32, // +0x28
    init_cmd: u32, // +0x2c
    ucode_feature: u32, // +0x30
    ucode_cmd_mask0: u32, // +0x34
    ucode_cmd_mask1: u32, // +0x38
    multi_tgt_tbl: u32, // +0x3c

    /// FWSEC command ids (init_cmd).
    pub const CMD_FRTS: u32 = 0x15;
    pub const CMD_SB: u32 = 0x19;
};

/// nvfw_fwsec_frts_cmd: written at DMEMMAPPER.cmd_in_buffer_offset. Two
/// sub-commands: read_vbios (flags=2) then the frts_region descriptor. Field
/// offsets (fwsec.c):
///   read_vbios:  ver(+0x00) hdr(+0x04) addr(+0x08,u64) size(+0x10) flags(+0x14)
///   frts_region: ver(+0x18) hdr(+0x1c) addr(+0x20) size(+0x24) type(+0x28)
pub const NvfwFwsecFrtsCmd = extern struct {
    read_vbios: ReadVbiosDesc, // +0x00, 0x18 bytes
    frts_region: FrtsRegionDesc, // +0x18, 0x14 bytes

    pub const ReadVbiosDesc = extern struct {
        ver: u32, // +0x00
        hdr: u32, // +0x04
        addr: u64, // +0x08
        size: u32, // +0x10
        flags: u32, // +0x14

        /// VBIOS read flags constant (fwsec.c hardcodes 2).
        pub const FLAGS_VBIOS: u32 = 2;
    };

    pub const FrtsRegionDesc = extern struct {
        ver: u32, // +0x00 (+0x18 absolute)
        hdr: u32, // +0x04
        addr: u32, // +0x08 (region_addr >> 12)
        size: u32, // +0x0c (region_size >> 12)
        type: u32, // +0x10

        /// FRTS region type: framebuffer (VRAM).
        pub const TYPE_FB: u32 = 0x2;
    };
};

comptime {
    // nvfw FALCON ucode + FWSEC/FRTS struct layout asserts vs the C reference.
    std.debug.assert(@sizeOf(NvfwFalconUcodeDescV2) == 0x3c);
    std.debug.assert(@offsetOf(NvfwFalconUcodeDescV2, "InterfaceOffset") == 0x10);
    std.debug.assert(@offsetOf(NvfwFalconUcodeDescV2, "DMEMOffset") == 0x28);
    std.debug.assert(@offsetOf(NvfwFalconUcodeDescV2, "DMEMPhysBase") == 0x2c);
    std.debug.assert(@offsetOf(NvfwFalconUcodeDescV2, "DMEMLoadSize") == 0x30);

    std.debug.assert(@sizeOf(NvfwFalconAppifHdr) == 4);
    std.debug.assert(@offsetOf(NvfwFalconAppifHdr, "cnt") == 3);

    std.debug.assert(@sizeOf(NvfwFalconAppif) == 8);
    std.debug.assert(@offsetOf(NvfwFalconAppif, "dmem_base") == 4);

    std.debug.assert(@offsetOf(NvfwFalconAppifDmemmapperV3, "cmd_in_buffer_offset") == 0x08);
    std.debug.assert(@offsetOf(NvfwFalconAppifDmemmapperV3, "init_cmd") == 0x2c);
    std.debug.assert(@sizeOf(NvfwFalconAppifDmemmapperV3) == 0x40);

    std.debug.assert(@offsetOf(NvfwFwsecFrtsCmd, "read_vbios") == 0x00);
    std.debug.assert(@offsetOf(NvfwFwsecFrtsCmd, "frts_region") == 0x18);
    std.debug.assert(@offsetOf(NvfwFwsecFrtsCmd.ReadVbiosDesc, "addr") == 0x08);
    std.debug.assert(@offsetOf(NvfwFwsecFrtsCmd.ReadVbiosDesc, "flags") == 0x14);
    std.debug.assert(@offsetOf(NvfwFwsecFrtsCmd.FrtsRegionDesc, "addr") == 0x08);
    std.debug.assert(@offsetOf(NvfwFwsecFrtsCmd.FrtsRegionDesc, "type") == 0x10);
}

// ===========================================================================
// 5. FWSEC extraction + FRTS patch (the core deliverable)
// ===========================================================================

/// A located FWSEC ucode: the descriptor (parsed from the ROM) plus the slices
/// of the IMEM/DMEM images. The DMEM slice is where the appif table + the
/// DMEMMAPPER live and where we write the FRTS command.
pub const Fwsec = struct {
    /// The FWSEC nvfw FALCON ucode descriptor (v2).
    desc: NvfwFalconUcodeDescV2,
    /// Offset (in the ROM) of the descriptor itself.
    desc_offset: u32,
    /// Offset (in the ROM) of the ucode payload (descriptor + StoredSize bytes
    /// follow; the IMEM/DMEM images are inside it). nouveau treats the payload as
    /// descriptor-relative; the DMEM image base inside it is `DMEMOffset`.
    payload_offset: u32,

    /// The byte offset (into the ROM) of the DMEM image: payload_offset +
    /// DMEMOffset. The appif table lives at dmem_image + InterfaceOffset.
    pub fn dmemImageOffset(self: Fwsec) u32 {
        return self.payload_offset + self.desc.DMEMOffset;
    }
};

/// Locate the FWSEC ucode (application type 0x85) in the ROM and parse its
/// descriptor. The DMEM/IMEM images are addressed relative to the descriptor.
pub fn findFwsec(rom: []const u8) Error!Fwsec {
    const bit = try Bit.parse(rom);
    const pmu = try PmuTable.parse(bit);
    const desc_off = try pmu.fwsecDescOffset();
    return findFwsecAt(rom, desc_off);
}

/// Parse the FWSEC descriptor at a known ROM offset. The payload (IMEM+DMEM
/// images) immediately follows the descriptor; nouveau's image offsets
/// (DMEMOffset, InterfaceOffset) are relative to the start of the payload, which
/// here is the descriptor offset (the descriptor is the head of the ucode blob).
pub fn findFwsecAt(rom: []const u8, desc_off: u32) Error!Fwsec {
    if (desc_off + @sizeOf(NvfwFalconUcodeDescV2) > rom.len) return Error.Truncated;
    const desc = readStruct(NvfwFalconUcodeDescV2, rom, desc_off);
    if (desc.version() != 2) return Error.UnsupportedUcodeVersion;
    return .{
        .desc = desc,
        .desc_offset = desc_off,
        .payload_offset = desc_off,
    };
}

/// Result of an FRTS patch: where the DMEMMAPPER and the FRTS command landed
/// (offsets into the produced image), for verification.
pub const PatchResult = struct {
    /// Offset (in the produced image) of the DMEMMAPPER descriptor.
    dmemmapper_offset: u32,
    /// Offset (in the produced image) of the nvfw_fwsec_frts_cmd we wrote.
    frts_cmd_offset: u32,
};

/// Produce a ready-to-run FWSEC image: copy the ROM's FWSEC ucode bytes into
/// `out`, find the DMEMMAPPER application (ID 0x4), set its init_cmd to FRTS
/// (0x15), and write the nvfw_fwsec_frts_cmd at its cmd_in_buffer_offset with the
/// FRTS region (type FB, addr>>12, size>>12). `frts_addr`/`frts_size` come from
/// fw.zig's Wpr2Layout (layout.frts_addr / layout.frts_size).
///
/// `out` must be at least `fwsec.payload_offset + StoredSize` bytes; the returned
/// slice is the produced image (the bytes the GSP falcon executes). The ROM
/// region outside the DMEMMAPPER/FRTS patch points is copied byte-for-byte.
pub fn extractAndPatchFrts(
    rom: []const u8,
    fwsec: Fwsec,
    frts_addr: u64,
    frts_size: u64,
    out: []u8,
) Error!PatchResult {
    // The image we run is the FWSEC ucode blob: descriptor + payload. We copy the
    // whole ROM region the ucode occupies (descriptor through end-of-DMEM) so the
    // IMEM/DMEM images are intact, then patch the DMEM in place.
    const image_len = fwsec.payload_offset -| fwsec.desc_offset + fwsecImageLen(fwsec);
    const total = fwsec.desc_offset + image_len;
    if (total > rom.len) return Error.Truncated;
    if (out.len < total) return Error.BufferTooSmall;
    @memcpy(out[0..total], rom[0..total]);

    // Locate the appif header inside the DMEM image.
    const dmem_img = fwsec.dmemImageOffset();
    const appif_hdr_off = dmem_img + fwsec.desc.InterfaceOffset;
    if (appif_hdr_off + @sizeOf(NvfwFalconAppifHdr) > total) return Error.Truncated;
    const hdr = readStruct(NvfwFalconAppifHdr, out, appif_hdr_off);

    // Walk the appif entries for the DMEMMAPPER (id 0x4).
    var dmemmapper_off: u32 = 0;
    var found = false;
    var i: u32 = 0;
    while (i < hdr.cnt) : (i += 1) {
        const e_off = appif_hdr_off + hdr.hdr + i * hdr.len;
        if (e_off + @sizeOf(NvfwFalconAppif) > total) return Error.Truncated;
        const e = readStruct(NvfwFalconAppif, out, e_off);
        if (e.id == NvfwFalconAppif.ID_DMEMMAPPER) {
            // dmem_base is an offset into the DMEM image.
            dmemmapper_off = dmem_img + e.dmem_base;
            found = true;
            break;
        }
    }
    if (!found) return Error.DmemMapperNotFound;
    if (dmemmapper_off + @sizeOf(NvfwFalconAppifDmemmapperV3) > total) return Error.Truncated;

    // Patch init_cmd = FRTS (0x15) in the DMEMMAPPER.
    const init_cmd_off = dmemmapper_off + @offsetOf(NvfwFalconAppifDmemmapperV3, "init_cmd");
    writeInt(u32, out[init_cmd_off..][0..4], NvfwFalconAppifDmemmapperV3.CMD_FRTS, .little);

    // The command buffer offset is relative to the DMEM image.
    const cmd_in_off = readInt(u32, out[dmemmapper_off + @offsetOf(NvfwFalconAppifDmemmapperV3, "cmd_in_buffer_offset") ..][0..4], .little);
    const frts_cmd_off = dmem_img + cmd_in_off;
    if (frts_cmd_off + @sizeOf(NvfwFwsecFrtsCmd) > total) return Error.Truncated;

    // Build + write the FRTS command.
    const cmd = NvfwFwsecFrtsCmd{
        .read_vbios = .{
            .ver = 1,
            .hdr = @sizeOf(NvfwFwsecFrtsCmd.ReadVbiosDesc),
            .addr = 0,
            .size = 0,
            .flags = NvfwFwsecFrtsCmd.ReadVbiosDesc.FLAGS_VBIOS,
        },
        .frts_region = .{
            .ver = 1,
            .hdr = @sizeOf(NvfwFwsecFrtsCmd.FrtsRegionDesc),
            .addr = @intCast(frts_addr >> 12),
            .size = @intCast(frts_size >> 12),
            .type = NvfwFwsecFrtsCmd.FrtsRegionDesc.TYPE_FB,
        },
    };
    writeStruct(NvfwFwsecFrtsCmd, out, frts_cmd_off, cmd);

    return .{
        .dmemmapper_offset = dmemmapper_off,
        .frts_cmd_offset = frts_cmd_off,
    };
}

/// The length of the FWSEC ucode image to copy: descriptor + DMEMOffset (where
/// the DMEM image sits) + DMEMLoadSize (DMEM image length). This bounds the bytes
/// the falcon needs. We take the max with the IMEM extent so both images are
/// included.
fn fwsecImageLen(fwsec: Fwsec) u32 {
    const d = fwsec.desc;
    const dmem_end = d.DMEMOffset + d.DMEMLoadSize;
    const imem_end = d.IMEMPhysBase + d.IMEMLoadSize;
    return @max(dmem_end, imem_end);
}

// ===========================================================================
// Small endian struct read/write helpers (LE, the VBIOS/nvfw byte order)
// ===========================================================================

fn readStruct(comptime T: type, buf: []const u8, off: u32) T {
    var v: T = undefined;
    const bytes = std.mem.asBytes(&v);
    @memcpy(bytes, buf[off .. off + @sizeOf(T)]);
    return v;
}

fn writeStruct(comptime T: type, buf: []u8, off: u32, v: T) void {
    const bytes = std.mem.asBytes(&v);
    @memcpy(buf[off .. off + @sizeOf(T)], bytes);
}

// ===========================================================================
// Unit tests (offline, on injected synthetic ROM buffers)
// ===========================================================================

const testing = std.testing;

/// Build a minimal-but-valid synthetic VBIOS image for the BIT + PMU + FWSEC
/// tests. Returns the bytes. Layout (all little-endian):
///   [0..2]   = 0x55 0xAA (PCIR)
///   bit token "\xff\xb8""BIT\0" at BIT_OFF
///   BIT header (entry size 6, count = entries.len) + entries
///   one 'I' devinit entry pointing at DEVINIT_OFF
///   one 'p' PMU entry (v2, len 4) pointing at a u32 == PMU_BASE
///   PMU table at PMU_BASE: ver/hdr/len/cnt + one entry type 0x85 -> FWSEC_DESC
///   FWSEC descriptor (v2) at FWSEC_DESC with DMEM image holding an appif table
///   + a DMEMMAPPER (id 0x4) + room for the FRTS command.
const SynthVbios = struct {
    buf: [4096]u8,

    const BIT_OFF: u32 = 0x40;
    const DEVINIT_OFF: u16 = 0x300;
    const PMU_BASE: u32 = 0x100;
    const FWSEC_DESC: u32 = 0x400;
    // Inside the FWSEC ucode blob (descriptor-relative).
    const DMEM_OFFSET: u32 = 0x80; // DMEMOffset
    const IFACE_OFFSET: u32 = 0x00; // InterfaceOffset (appif at dmem_img + 0)
    const APPIF_DMEM_BASE: u32 = 0x40; // DMEMMAPPER dmem_base within the DMEM image
    const CMD_IN_OFFSET: u32 = 0x100; // cmd_in_buffer_offset (within DMEM image)
    const DMEM_LOAD: u32 = 0x200;

    fn build() SynthVbios {
        var s: SynthVbios = .{ .buf = [_]u8{0} ** 4096 };
        const b = &s.buf;

        // PCIR signature.
        b[0] = PCIR_SIG0;
        b[1] = PCIR_SIG1;

        // BIT token.
        @memcpy(b[BIT_OFF .. BIT_OFF + BIT_TOKEN.len], &BIT_TOKEN);
        b[BIT_OFF + 5] = 0; // the trailing NUL of "BIT\0"
        // BIT header: entry size @ +9, count @ +10, entries @ +12.
        const entry_size: u8 = 6;
        b[BIT_OFF + 9] = entry_size;
        b[BIT_OFF + 10] = 2; // two entries: 'I' and 'p'
        const entries = BIT_OFF + 12;

        // entry 0: 'I' devinit -> DEVINIT_OFF
        b[entries + 0] = 'I';
        b[entries + 1] = 1;
        writeInt(u16, b[entries + 2 ..][0..2], 4, .little); // length
        writeInt(u16, b[entries + 4 ..][0..2], DEVINIT_OFF, .little);

        // entry 1: 'p' PMU (v2, len 4) -> data offset holds a u32 PMU_BASE.
        const e1 = entries + entry_size;
        b[e1 + 0] = 'p';
        b[e1 + 1] = 2; // version 2
        writeInt(u16, b[e1 + 2 ..][0..2], 4, .little); // length >= 4
        const pmu_ptr_off: u16 = 0x200; // where the u32 PMU_BASE lives
        writeInt(u16, b[e1 + 4 ..][0..2], pmu_ptr_off, .little);
        writeInt(u32, b[pmu_ptr_off..][0..4], PMU_BASE, .little);

        // PMU table @ PMU_BASE: ver, hdr, len, cnt, then one entry.
        b[PMU_BASE + 0] = 1; // version
        b[PMU_BASE + 1] = 4; // header size
        b[PMU_BASE + 2] = 6; // entry length
        b[PMU_BASE + 3] = 1; // one entry
        const pe = PMU_BASE + 4;
        b[pe + 0] = PmuTable.FWSEC_APP_TYPE; // 0x85
        writeInt(u32, b[pe + 2 ..][0..4], FWSEC_DESC, .little);

        // FWSEC descriptor (v2) @ FWSEC_DESC. Hdr: valid|version2|size.
        const Hdr: u32 = 0x1 | (@as(u32, 2) << 8) | (@as(u32, 0x100) << 16);
        writeInt(u32, b[FWSEC_DESC + 0x00 ..][0..4], Hdr, .little);
        writeInt(u32, b[FWSEC_DESC + 0x04 ..][0..4], DMEM_OFFSET + DMEM_LOAD, .little); // StoredSize
        writeInt(u32, b[FWSEC_DESC + 0x10 ..][0..4], IFACE_OFFSET, .little); // InterfaceOffset
        writeInt(u32, b[FWSEC_DESC + 0x18 ..][0..4], 0x10, .little); // IMEMLoadSize
        writeInt(u32, b[FWSEC_DESC + 0x28 ..][0..4], DMEM_OFFSET, .little); // DMEMOffset
        writeInt(u32, b[FWSEC_DESC + 0x30 ..][0..4], DMEM_LOAD, .little); // DMEMLoadSize

        // DMEM image base inside the blob = FWSEC_DESC + DMEM_OFFSET.
        const dmem_img = FWSEC_DESC + DMEM_OFFSET;
        // appif header @ dmem_img + IFACE_OFFSET: ver, hdr=4, len=8, cnt=1.
        const appif = dmem_img + IFACE_OFFSET;
        b[appif + 0] = 1; // ver
        b[appif + 1] = 4; // hdr size
        b[appif + 2] = 8; // entry len
        b[appif + 3] = 1; // one entry
        // appif entry 0: id=DMEMMAPPER(4), dmem_base=APPIF_DMEM_BASE.
        const ae = appif + 4;
        writeInt(u32, b[ae + 0 ..][0..4], NvfwFalconAppif.ID_DMEMMAPPER, .little);
        writeInt(u32, b[ae + 4 ..][0..4], APPIF_DMEM_BASE, .little);

        // DMEMMAPPER @ dmem_img + APPIF_DMEM_BASE. Set cmd_in_buffer_offset.
        const dm = dmem_img + APPIF_DMEM_BASE;
        writeInt(u32, b[dm + 0x08 ..][0..4], CMD_IN_OFFSET, .little); // cmd_in_buffer_offset
        // init_cmd starts as 0; the patch must set it to FRTS.

        return s;
    }

    fn dmemImg() u32 {
        return FWSEC_DESC + DMEM_OFFSET;
    }
    fn dmemmapper() u32 {
        return dmemImg() + APPIF_DMEM_BASE;
    }
    fn frtsCmd() u32 {
        return dmemImg() + CMD_IN_OFFSET;
    }
};

test "validate accepts a synthetic ROM and finds the BIT token" {
    var s = SynthVbios.build();
    const off = try validate(&s.buf);
    try testing.expectEqual(SynthVbios.BIT_OFF, off);
}

test "validate rejects a ROM without the PCIR signature" {
    var s = SynthVbios.build();
    s.buf[0] = 0x00;
    try testing.expectError(Error.NoPciRom, validate(&s.buf));
}

test "validate rejects a ROM without the BIT token" {
    var s = SynthVbios.build();
    // Stomp the BIT token bytes.
    s.buf[SynthVbios.BIT_OFF] = 0;
    s.buf[SynthVbios.BIT_OFF + 1] = 0;
    try testing.expectError(Error.NoBitToken, validate(&s.buf));
}

test "BIT parse iterates entries and resolves ids" {
    var s = SynthVbios.build();
    const bit = try Bit.parse(&s.buf);
    try testing.expectEqual(@as(u8, 6), bit.entry_size);
    try testing.expectEqual(@as(u8, 2), bit.entry_count);

    const i_entry = try bit.entry('I');
    try testing.expectEqual(@as(u8, 'I'), i_entry.id);
    try testing.expectEqual(SynthVbios.DEVINIT_OFF, i_entry.offset);
    try testing.expectEqual(SynthVbios.DEVINIT_OFF, try bit.devinitOffset());

    const p_entry = try bit.entry('p');
    try testing.expectEqual(@as(u8, 'p'), p_entry.id);
    try testing.expectEqual(@as(u8, 2), p_entry.version);
}

test "BIT parse reports not-found for a missing id" {
    var s = SynthVbios.build();
    const bit = try Bit.parse(&s.buf);
    try testing.expectError(Error.BitEntryNotFound, bit.entry('Z'));
}

test "BIT parse rejects a truncated header" {
    // A buffer that has the token at the very end leaves no room for the header.
    var buf: [SynthVbios.BIT_OFF + 6]u8 = [_]u8{0} ** (SynthVbios.BIT_OFF + 6);
    buf[0] = PCIR_SIG0;
    buf[1] = PCIR_SIG1;
    @memcpy(buf[SynthVbios.BIT_OFF .. SynthVbios.BIT_OFF + BIT_TOKEN.len], &BIT_TOKEN);
    try testing.expectError(Error.Truncated, Bit.parse(&buf));
}

test "PMU table parse locates the FWSEC descriptor" {
    var s = SynthVbios.build();
    const bit = try Bit.parse(&s.buf);
    const pmu = try PmuTable.parse(bit);
    try testing.expectEqual(@as(u32, SynthVbios.PMU_BASE), pmu.base);
    try testing.expectEqual(@as(u8, 1), pmu.entry_count);
    const desc_off = try pmu.fwsecDescOffset();
    try testing.expectEqual(SynthVbios.FWSEC_DESC, desc_off);
}

test "PMU table reports FwsecNotFound when no type-0x85 entry" {
    var s = SynthVbios.build();
    // Change the single PMU entry's type away from 0x85.
    s.buf[SynthVbios.PMU_BASE + 4] = 0x01;
    const bit = try Bit.parse(&s.buf);
    const pmu = try PmuTable.parse(bit);
    try testing.expectError(Error.FwsecNotFound, pmu.fwsecDescOffset());
}

test "findFwsec parses the v2 descriptor" {
    var s = SynthVbios.build();
    const fwsec = try findFwsec(&s.buf);
    try testing.expectEqual(@as(u8, 2), fwsec.desc.version());
    try testing.expectEqual(SynthVbios.FWSEC_DESC, fwsec.desc_offset);
    try testing.expectEqual(SynthVbios.dmemImg(), fwsec.dmemImageOffset());
}

test "extractAndPatchFrts writes the FRTS command at the right place" {
    var s = SynthVbios.build();
    const fwsec = try findFwsec(&s.buf);

    const frts_addr: u64 = 0x1_2345_6000; // page-aligned
    const frts_size: u64 = 0x10_0000; // 1 MiB

    var out: [4096]u8 = undefined;
    const res = try extractAndPatchFrts(&s.buf, fwsec, frts_addr, frts_size, &out);

    // The DMEMMAPPER + FRTS command landed where the synthetic layout put them.
    try testing.expectEqual(SynthVbios.dmemmapper(), res.dmemmapper_offset);
    try testing.expectEqual(SynthVbios.frtsCmd(), res.frts_cmd_offset);

    // init_cmd in the DMEMMAPPER is now FRTS (0x15).
    const init_cmd = readInt(u32, out[res.dmemmapper_offset + 0x2c ..][0..4], .little);
    try testing.expectEqual(NvfwFalconAppifDmemmapperV3.CMD_FRTS, init_cmd);

    // The written FRTS command: read_vbios then frts_region.
    const cmd = readStruct(NvfwFwsecFrtsCmd, &out, res.frts_cmd_offset);
    try testing.expectEqual(@as(u32, 1), cmd.read_vbios.ver);
    try testing.expectEqual(NvfwFwsecFrtsCmd.ReadVbiosDesc.FLAGS_VBIOS, cmd.read_vbios.flags);

    try testing.expectEqual(@as(u32, 1), cmd.frts_region.ver);
    try testing.expectEqual(@as(u32, @intCast(frts_addr >> 12)), cmd.frts_region.addr);
    try testing.expectEqual(@as(u32, @intCast(frts_size >> 12)), cmd.frts_region.size);
    try testing.expectEqual(NvfwFwsecFrtsCmd.FrtsRegionDesc.TYPE_FB, cmd.frts_region.type);
}

test "extractAndPatchFrts leaves the rest of the image untouched" {
    var s = SynthVbios.build();
    const fwsec = try findFwsec(&s.buf);

    var out: [4096]u8 = undefined;
    const res = try extractAndPatchFrts(&s.buf, fwsec, 0x1000_0000, 0x10_0000, &out);

    // Everything before the FWSEC descriptor is a byte-for-byte copy.
    try testing.expectEqualSlices(u8, s.buf[0..SynthVbios.FWSEC_DESC], out[0..SynthVbios.FWSEC_DESC]);

    // The PCIR + BIT token survived.
    try testing.expectEqual(PCIR_SIG0, out[0]);
    try testing.expectEqualSlices(u8, &BIT_TOKEN, out[SynthVbios.BIT_OFF .. SynthVbios.BIT_OFF + BIT_TOKEN.len]);

    // Only the init_cmd word + the FRTS-command region changed within the DMEM.
    // The DMEMMAPPER's cmd_in_buffer_offset field is preserved.
    const cib = readInt(u32, out[res.dmemmapper_offset + 0x08 ..][0..4], .little);
    try testing.expectEqual(SynthVbios.CMD_IN_OFFSET, cib);
}

test "extractAndPatchFrts errors when the DMEMMAPPER is absent" {
    var s = SynthVbios.build();
    // Change the appif entry id away from DMEMMAPPER (0x4).
    const ae = SynthVbios.dmemImg() + SynthVbios.IFACE_OFFSET + 4;
    writeInt(u32, s.buf[ae..][0..4], 0x99, .little);
    const fwsec = try findFwsec(&s.buf);
    var out: [4096]u8 = undefined;
    try testing.expectError(Error.DmemMapperNotFound, extractAndPatchFrts(&s.buf, fwsec, 0, 0x100000, &out));
}

test "extractAndPatchFrts errors on a too-small output buffer" {
    var s = SynthVbios.build();
    const fwsec = try findFwsec(&s.buf);
    var out: [16]u8 = undefined;
    try testing.expectError(Error.BufferTooSmall, extractAndPatchFrts(&s.buf, fwsec, 0, 0x100000, &out));
}

test "readVbiosProm copies words out of a fake PROM aperture" {
    // A fake BAR whose backing is a pretend ROM at PROM_OFFSET.
    var backing: [PROM_OFFSET + 64]u8 = [_]u8{0} ** (PROM_OFFSET + 64);
    backing[PROM_OFFSET + 0] = PCIR_SIG0;
    backing[PROM_OFFSET + 1] = PCIR_SIG1;
    writeInt(u32, backing[PROM_OFFSET + 4 ..][0..4], 0xDEADBEEF, .little);
    const bar = hw.Bar.init(@intFromPtr(&backing), backing.len);

    var out: [16]u8 = undefined;
    const img = readVbiosProm(bar, &out);
    try testing.expectEqual(@as(usize, 16), img.len);
    try testing.expectEqual(PCIR_SIG0, img[0]);
    try testing.expectEqual(PCIR_SIG1, img[1]);
    try testing.expectEqual(@as(u32, 0xDEADBEEF), readInt(u32, img[4..8], .little));
}

test {
    testing.refAllDecls(@This());
}
