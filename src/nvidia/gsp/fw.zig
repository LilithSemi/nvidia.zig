//! GSP firmware packaging (plan phase P3): the "cold" data structures the GSP
//! boot (P4) consumes. Almost all pure byte/arithmetic logic, so this whole file
//! is offline unit-testable - and it is validated against the REAL 595.71.05
//! gsp_ga10x.bin (an ELF container) read at test time.
//!
//! What this builds (each matched to a reference):
//!   - ELF parse: extract `.fwimage` (the GSP-RM image -> radix3 data),
//!     `.fwsignature_ga10x` (per-chip signature), `.fwversion` (must == "595.71.05").
//!     Ref: OGK kernel_gsp.c _kgspFwContainerGetSection (LibosElf64Header walk).
//!   - radix3: the 3-level page table over the GSP-RM image data pages.
//!     Ref: OGK kgspCreateRadix3_IMPL (4-entry working array, entriesLog2 = 9).
//!   - WPR2 layout: the top-down VRAM offset stack.
//!     Ref: nouveau tu102_gsp_oneinit + tu102_gsp_wpr_heap_size.
//!   - GspFwWprMeta fill: ref nouveau tu102_gsp_wpr_meta_init.
//!   - libos init args + GSP_ARGUMENTS_CACHED: ref nouveau r535_gsp_libos_init /
//!     r535_gsp_set_rmargs.
//!   - BumpAllocator: a page-aligned phys allocator over a region WE own (no
//!     kernel on baremetal). On hardware the sysmem region is EFI AllocatePages
//!     identity-mapped phys; in the test it is a buffer + a base phys address.
//!
//! Freestanding-safe: this file is pure byte logic and pulls in NO std.fs /
//! std.Io. The real-firmware file read used by the tests lives inside a
//! `!freestanding`-gated test body (the ELF-parse test), so the UEFI/freestanding
//! build never links the IO machinery. On baremetal the firmware bytes are
//! injected (see the @embedFile note at the bottom) - this module only ever
//! takes the bytes as a slice.

const std = @import("std");
const builtin = @import("builtin");
const proto = @import("proto.zig");
const sdk = @import("../sdk.zig");

const NvU8 = sdk.NvU8;
const NvU32 = sdk.NvU32;
const NvU64 = sdk.NvU64;

pub const Error = error{
    /// Not an ELF64 LE container (bad magic/class/endianness).
    NotElf,
    /// The requested section name was not present.
    SectionNotFound,
    /// A section header / offset ran past the end of the buffer.
    Truncated,
    /// The .fwversion did not match the pinned 595.71.05 string.
    VersionMismatch,
    /// The provided memory region was too small for the requested layout.
    OutOfMemory,
    /// The VRAM size given could not fit the WPR2 + heaps.
    VramTooSmall,
};

// ===========================================================================
// Constants (all confirmed against the references at 595.71.05 / nouveau)
// ===========================================================================

/// LIBOS_MEMORY_REGION_RADIX_PAGE_SIZE (libos_init_args.h). The radix3 page size.
pub const RADIX_PAGE_SIZE: u64 = 4096;
/// LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2.
pub const RADIX_PAGE_LOG2: u6 = 12;
/// entriesLog2 = RADIX_PAGE_LOG2 - 3 (8 bytes per PTE) -> 512 entries per page.
pub const RADIX_ENTRIES_LOG2: u6 = RADIX_PAGE_LOG2 - 3;
/// 1 << entriesLog2 == 512 PTEs per radix page.
pub const RADIX_ENTRIES_PER_PAGE: u64 = @as(u64, 1) << RADIX_ENTRIES_LOG2;

/// The pinned firmware version string this build is locked to (.fwversion).
/// OGK gate _kgspFwContainerVerifyVersion: .fwversion must equal NV_VERSION_STRING.
pub const FW_VERSION: []const u8 = "595.71.05";

// WPR2 layout constants (tu102_gsp_oneinit / tu102_gsp_wpr_heap_size, GA10x = libos3).
const MB: u64 = 1 << 20;
/// FRTS region size on Turing+/Ampere (GA100 0x170 uses 0; GA10x uses 0x100000).
pub const FRTS_SIZE: u64 = 0x100000;
/// The non-WPR heap size (tu102_gsp_oneinit: gsp->fb.heap.size = 0x100000).
pub const NON_WPR_HEAP_SIZE: u64 = 0x100000;

// GSP_FW_HEAP_PARAM_* (nouveau rm/r535/nvrm/gsp.h). GA10x => r535_wpr_libos3.
//   os_carveout_size = GSP_FW_HEAP_PARAM_OS_SIZE_LIBOS3       = 20 << 20
//   base_size        = GSP_FW_HEAP_PARAM_BASE_RM_SIZE_TU10X   =  8 << 20  (Turing..Ada)
//   heap_size_min    = GSP_FW_HEAP_SIZE_OVERRIDE_LIBOS3_BAREMETAL_MIN_MB = 84 MB
//   per-GB-FB        = GSP_FW_HEAP_PARAM_SIZE_PER_GB_FB       = 96 << 10  (all archs)
//   client-alloc     = GSP_FW_HEAP_PARAM_CLIENT_ALLOC_SIZE    = (48 << 10) * 2048
// NOTE: the r570 drop (matching the 595 firmware era) bumps os_carveout to
// 22 << 20 (GSP_FW_HEAP_PARAM_OS_SIZE_LIBOS3_BAREMETAL). We use the r535 libos3
// value (20 MB) as the documented baseline; the heap is then ALIGN'd and floored
// at heap_size_min (84 MB) anyway, so for typical GA10x FB sizes the min-floor
// dominates and the carveout delta is absorbed. FLAGGED: confirm 20 vs 22 MB on
// hardware via the `verified` writeback if the heap ends up undersized.
pub const HeapParams = struct {
    os_carveout_size: u64 = 20 * MB,
    base_size: u64 = 8 * MB,
    heap_size_min: u64 = 84 * MB,
    pub const SIZE_PER_GB_FB: u64 = 96 << 10;
    pub const CLIENT_ALLOC_SIZE: u64 = (48 << 10) * 2048;
};

fn alignUp(v: u64, a: u64) u64 {
    return (v + a - 1) & ~(a - 1);
}
fn alignDown(v: u64, a: u64) u64 {
    return v & ~(a - 1);
}
fn divRoundUp(n: u64, d: u64) u64 {
    return (n + d - 1) / d;
}

// ===========================================================================
// ELF64 parse (minimal, pure byte logic - matches OGK _kgspFwContainerGetSection)
// ===========================================================================

const ELF_MAGIC: u32 = 0x464C457F; // "\x7fELF" little-endian
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;

/// The ELF64 header fields we need (read by offset, no packed-struct alignment
/// assumptions about the input buffer).
pub const Elf = struct {
    bytes: []const u8,
    sh_off: u64,
    sh_entsize: u16,
    sh_num: u16,
    sh_strndx: u16,

    /// Validate the ELF64 LE identity + read the section-header table location.
    pub fn parse(bytes: []const u8) Error!Elf {
        if (bytes.len < 64) return Error.NotElf;
        if (rdU32(bytes, 0) != ELF_MAGIC) return Error.NotElf;
        if (bytes[4] != ELFCLASS64) return Error.NotElf;
        if (bytes[5] != ELFDATA2LSB) return Error.NotElf;
        const sh_off = rdU64(bytes, 0x28);
        const sh_entsize = @as(u16, @truncate(rdU16(bytes, 0x3a)));
        const sh_num = @as(u16, @truncate(rdU16(bytes, 0x3c)));
        const sh_strndx = @as(u16, @truncate(rdU16(bytes, 0x3e)));
        if (sh_entsize < 64) return Error.NotElf;
        // The whole section-header table must be in range.
        const table_len = @as(u64, sh_entsize) * @as(u64, sh_num);
        if (sh_off + table_len > bytes.len) return Error.Truncated;
        return .{
            .bytes = bytes,
            .sh_off = sh_off,
            .sh_entsize = sh_entsize,
            .sh_num = sh_num,
            .sh_strndx = sh_strndx,
        };
    }

    fn shdrOffset(self: Elf, idx: usize) u64 {
        return self.sh_off + @as(u64, self.sh_entsize) * @as(u64, idx);
    }

    fn shName(self: Elf, idx: usize) u32 {
        return rdU32(self.bytes, self.shdrOffset(idx) + 0x00);
    }
    fn shOffset(self: Elf, idx: usize) u64 {
        return rdU64(self.bytes, self.shdrOffset(idx) + 0x18);
    }
    fn shSize(self: Elf, idx: usize) u64 {
        return rdU64(self.bytes, self.shdrOffset(idx) + 0x20);
    }

    fn sectionNameStr(self: Elf, name_off: u32) Error![]const u8 {
        const strtab_off = self.shOffset(self.sh_strndx);
        const strtab_size = self.shSize(self.sh_strndx);
        const start = strtab_off + name_off;
        if (start >= self.bytes.len or name_off >= strtab_size) return Error.Truncated;
        var end = start;
        while (end < self.bytes.len and self.bytes[end] != 0) : (end += 1) {}
        return self.bytes[start..end];
    }

    /// Return the raw bytes of the named section (e.g. ".fwimage"). Returns
    /// SectionNotFound if absent, Truncated if its extent runs past the buffer.
    pub fn section(self: Elf, want: []const u8) Error![]const u8 {
        var i: usize = 0;
        while (i < self.sh_num) : (i += 1) {
            const nm = try self.sectionNameStr(self.shName(i));
            if (std.mem.eql(u8, nm, want)) {
                const off = self.shOffset(i);
                const sz = self.shSize(i);
                if (off + sz > self.bytes.len) return Error.Truncated;
                return self.bytes[off .. off + sz];
            }
        }
        return Error.SectionNotFound;
    }
};

/// The three sections the GSP boot needs out of the gsp_*.bin container.
pub const FirmwareSections = struct {
    /// `.fwimage` - the GSP-RM image (becomes the radix3 data).
    image: []const u8,
    /// `.fwsignature_<chip>` - the per-chip signature blob.
    signature: []const u8,
    /// `.fwversion` - the pinned version string (with trailing NUL trimmed).
    version: []const u8,
};

/// Extract the GSP-RM image + the per-chip signature + the version, and GATE on
/// the version matching FW_VERSION (OGK _kgspFwContainerVerifyVersion). The
/// signature section name is per-chip: `.fwsignature_ga10x` for GA10x/AD10x.
pub fn extractSections(elf: Elf, signature_section: []const u8) Error!FirmwareSections {
    const image = try elf.section(".fwimage");
    const signature = try elf.section(signature_section);
    const raw_ver = try elf.section(".fwversion");
    // Trim any trailing NUL(s).
    var vlen = raw_ver.len;
    while (vlen > 0 and raw_ver[vlen - 1] == 0) : (vlen -= 1) {}
    const version = raw_ver[0..vlen];
    if (!std.mem.eql(u8, version, FW_VERSION)) return Error.VersionMismatch;
    return .{ .image = image, .signature = signature, .version = version };
}

/// The signature section name for a given chip family. GA10x AND AD10x both
/// consume gsp_ga10x.bin but pick their own signature section.
pub fn signatureSectionFor(comptime chip: enum { ga10x, ad10x }) []const u8 {
    return switch (chip) {
        .ga10x => ".fwsignature_ga10x",
        .ad10x => ".fwsignature_ad10x",
    };
}

// ===========================================================================
// BumpAllocator - page-aligned phys allocator over a region WE own.
// ===========================================================================

/// A simple bump allocator over a contiguous physical region. On baremetal the
/// backing comes from EFI AllocatePages (identity-mapped, so phys == the base we
/// pass in); offline it is just a buffer + a chosen base phys. Every allocation
/// is RADIX_PAGE_SIZE-aligned (the GPU DMAs these), matching the kernel's
/// page-aligned DMA allocations.
pub const BumpAllocator = struct {
    base_phys: u64,
    size: u64,
    cursor: u64 = 0,

    pub fn init(base_phys: u64, size: u64) BumpAllocator {
        return .{ .base_phys = base_phys, .size = size };
    }

    /// Allocate `n` bytes, page-aligned, returning the phys address. Errors if
    /// the region is exhausted.
    pub fn alloc(self: *BumpAllocator, n: u64) Error!u64 {
        const start = alignUp(self.cursor, RADIX_PAGE_SIZE);
        const end = start + alignUp(n, RADIX_PAGE_SIZE);
        if (end > self.size) return Error.OutOfMemory;
        self.cursor = end;
        return self.base_phys + start;
    }

    /// Bytes consumed so far (page-aligned cursor).
    pub fn used(self: BumpAllocator) u64 {
        return self.cursor;
    }
};

// ===========================================================================
// radix3 builder (OGK kgspCreateRadix3_IMPL)
// ===========================================================================

/// The result of laying out a radix3 page table over `image_size` bytes of GSP
/// image data. Levels are named per the plan: lvl0 is the single-page root.
pub const Radix3 = struct {
    /// Phys address of the level-0 root page (-> wprMeta.sysmemAddrOfRadix3Elf).
    lvl0_phys: u64,
    /// Pages per level: [0]=root (==1), [1], [2]=PTEs over data, [3]=data pages.
    /// (OGK's 4-entry working array; data is radix3[3].)
    n_pages: [4]u64,
    /// Byte offsets of each level within the contiguous PT region (low to high:
    /// lvl0 at 0, then lvl1, lvl2, then the data at [3]).
    offsets: [4]u64,
    /// Total bytes of the page-table levels (excluding the data pages).
    pt_size: u64,
};

/// A phys-address provider for a data page index. On hardware this returns the
/// phys addr of image page `i`; offline the test supplies a fake mapping. This
/// is the "phys-address allocator callback" the plan asks for.
pub const PageAddrFn = *const fn (ctx: *anyopaque, page_index: u64) u64;

/// Compute the radix3 nPages/offsets for `image_size` bytes (no memory writes).
/// Matches OGK exactly: bottom level = ceil(size/4096); each parent =
/// ((child-1) >> 9) + 1; lvl0 MUST be exactly 1 page.
pub fn radix3Layout(image_size: u64) Error!Radix3 {
    if (image_size == 0) return Error.OutOfMemory;
    var n: [4]u64 = .{ 0, 0, 0, 0 };
    // Bottom (data) level.
    n[3] = divRoundUp(image_size, RADIX_PAGE_SIZE);
    // Parents, high index to low.
    var i: usize = 3;
    while (i > 0) : (i -= 1) {
        n[i - 1] = ((n[i] - 1) >> RADIX_ENTRIES_LOG2) + 1;
    }
    if (n[0] != 1) return Error.OutOfMemory; // image too large for a 3-level table
    // Offsets low to high (lvl0 at 0). The page-table region holds lvl0..lvl2;
    // the data sits at offsets[3] (right after the PT levels).
    var offsets: [4]u64 = .{ 0, 0, 0, 0 };
    var acc: u64 = 0;
    var j: usize = 1;
    while (j < 4) : (j += 1) {
        acc += n[j - 1];
        offsets[j] = acc << RADIX_PAGE_LOG2;
    }
    const pt_pages = n[0] + n[1] + n[2];
    return .{
        .lvl0_phys = 0, // filled by buildRadix3
        .n_pages = n,
        .offsets = offsets,
        .pt_size = pt_pages << RADIX_PAGE_LOG2,
    };
}

/// Lay out the 3 page-table levels in `pt_region` (a page-aligned phys region we
/// own, at least `pt_size` bytes) and fill the PTEs. `pt_region_cpu` is the
/// CPU-visible bytes of that region (so we can write the PTEs); `pt_region_phys`
/// is its phys base (so the PDEs can point at the lvl1/lvl2 pages by phys). The
/// data-page phys addresses come from `pageAddr` (the GSP image's pages). Returns
/// the filled Radix3 with lvl0_phys set.
///
/// Layout in the region: [lvl0 | lvl1 | lvl2] contiguously (offsets[0..2]).
/// - lvl0[0]   = phys of lvl1
/// - lvl1[k]   = phys of lvl2 page k
/// - lvl2[k]   = phys of data page k  (from pageAddr)
pub fn buildRadix3(
    image_size: u64,
    pt_region_cpu: []u8,
    pt_region_phys: u64,
    ctx: *anyopaque,
    pageAddr: PageAddrFn,
) Error!Radix3 {
    var r = try radix3Layout(image_size);
    if (pt_region_cpu.len < r.pt_size) return Error.OutOfMemory;
    if (pt_region_phys & (RADIX_PAGE_SIZE - 1) != 0) return Error.OutOfMemory;

    // Zero the page-table region first.
    @memset(pt_region_cpu[0..r.pt_size], 0);
    r.lvl0_phys = pt_region_phys;

    // Phys base of each level (contiguous within the region).
    const lvl1_phys = pt_region_phys + r.offsets[1];
    const lvl2_phys = pt_region_phys + r.offsets[2];

    // lvl0: a single PTE pointing at lvl1.
    writePte(pt_region_cpu, r.offsets[0], 0, lvl1_phys);

    // lvl1: one PTE per lvl2 page (phys-contiguous, so just stride by page size).
    var k: u64 = 0;
    while (k < r.n_pages[2]) : (k += 1) {
        writePte(pt_region_cpu, r.offsets[1], k, lvl2_phys + k * RADIX_PAGE_SIZE);
    }

    // lvl2: one PTE per data page, phys addr from the caller.
    var p: u64 = 0;
    while (p < r.n_pages[3]) : (p += 1) {
        writePte(pt_region_cpu, r.offsets[2], p, pageAddr(ctx, p));
    }

    return r;
}

fn writePte(region: []u8, level_offset: u64, entry: u64, phys: u64) void {
    const at = level_offset + entry * 8;
    std.mem.writeInt(u64, region[at..][0..8], phys, .little);
}

/// Read a PTE back (test helper).
pub fn readPte(region: []const u8, level_offset: u64, entry: u64) u64 {
    const at = level_offset + entry * 8;
    return std.mem.readInt(u64, region[at..][0..8], .little);
}

// ===========================================================================
// WPR2 layout (nouveau tu102_gsp_oneinit + tu102_gsp_wpr_heap_size)
// ===========================================================================

/// The full top-down VRAM layout the GSP boot needs. All values are absolute
/// VRAM byte offsets (addr) + sizes, computed downward from the top of usable
/// VRAM (bios.addr). Mirrors nouveau's gsp->fb.* fields.
pub const Wpr2Layout = struct {
    fb_size: u64,
    // The VGA/BIOS workspace at the very top; WPR2 builds DOWN from bios.addr.
    bios_addr: u64,
    vga_workspace_addr: u64,
    vga_workspace_size: u64,

    frts_addr: u64,
    frts_size: u64,
    boot_addr: u64, // bootloader image
    boot_size: u64,
    elf_addr: u64, // the GSP-RM ELF (radix3 data target in WPR2)
    elf_size: u64,
    heap_addr: u64, // WPR2 GSP-FW heap (LIBOS3)
    heap_size: u64,
    wpr2_addr: u64, // start of WPR2 (where GspFwWprMeta sits)
    wpr2_size: u64,
    wpr2_end: u64,
    non_wpr_heap_addr: u64,
    non_wpr_heap_size: u64,
};

/// GA10x WPR heap size (tu102_gsp_wpr_heap_size). fb_size is the VRAM size in
/// bytes (on hardware: reg 0x1183a4 << 20; here a parameter so it is testable).
pub fn wprHeapSize(fb_size: u64, hp: HeapParams) u64 {
    const fb_size_gb = divRoundUp(fb_size, 1 << 30);
    const heap = hp.os_carveout_size +
        hp.base_size +
        alignUp(HeapParams.SIZE_PER_GB_FB * fb_size_gb, MB) +
        alignUp(HeapParams.CLIENT_ALLOC_SIZE, MB);
    return @max(heap, hp.heap_size_min);
}

/// Compute the WPR2 top-down layout (tu102_gsp_oneinit). Inputs:
///   fb_size      - VRAM size in bytes (param so it is offline-testable).
///   bios_addr    - the top of usable VRAM the BIOS/VGA workspace ends at. On
///                  hardware this is gsp->fb.bios.addr (just below the VGA
///                  workspace at the very top of VRAM). For the offline test we
///                  pass a plausible value (e.g. fb_size minus the vga reserve).
///   vga_size     - the VGA workspace size (gsp->fb.bios.vga_workspace.size).
///   boot_size    - the bootloader image size (gsp->boot.fw.size).
///   elf_size     - the GSP-RM ELF size (gsp->fw.len == .fwimage size).
///   hp           - the heap params (GA10x defaults = libos3).
pub fn wpr2Layout(
    fb_size: u64,
    bios_addr: u64,
    vga_size: u64,
    boot_size: u64,
    elf_size: u64,
    hp: HeapParams,
) Error!Wpr2Layout {
    // FRTS: just below ALIGN_DOWN(bios.addr, 0x20000).
    const frts_size: u64 = FRTS_SIZE;
    const wpr2_end = alignDown(bios_addr, 0x20000);
    const frts_addr = wpr2_end - frts_size;

    // Bootloader image.
    const boot_addr = alignDown(frts_addr - boot_size, 0x1000);

    // GSP-RM ELF.
    const elf_addr = alignDown(boot_addr - elf_size, 0x10000);

    // WPR heap (LIBOS3). nouveau computes the size, floors the addr, then
    // RE-derives the size from the (aligned) addr so it exactly fills the gap.
    var heap_size = wprHeapSize(fb_size, hp);
    const heap_addr = alignDown(elf_addr - heap_size, 0x100000);
    heap_size = alignDown(elf_addr - heap_addr, 0x100000);

    // WPR2 start: one GspFwWprMeta below the heap, 1 MB aligned.
    const wpr2_addr = alignDown(heap_addr - @sizeOf(proto.GspFwWprMeta), 0x100000);
    const wpr2_size = frts_addr + frts_size - wpr2_addr;

    // Non-WPR heap: just below WPR2.
    const non_wpr_heap_size = NON_WPR_HEAP_SIZE;
    const non_wpr_heap_addr = wpr2_addr - non_wpr_heap_size;

    // Sanity: everything must fit inside VRAM and be ordered.
    if (wpr2_end > fb_size) return Error.VramTooSmall;
    if (non_wpr_heap_addr >= wpr2_addr) return Error.VramTooSmall;
    if (wpr2_addr >= heap_addr or heap_addr >= elf_addr) return Error.VramTooSmall;
    if (elf_addr >= boot_addr or boot_addr >= frts_addr) return Error.VramTooSmall;

    return .{
        .fb_size = fb_size,
        .bios_addr = bios_addr,
        .vga_workspace_addr = bios_addr,
        .vga_workspace_size = vga_size,
        .frts_addr = frts_addr,
        .frts_size = frts_size,
        .boot_addr = boot_addr,
        .boot_size = boot_size,
        .elf_addr = elf_addr,
        .elf_size = elf_size,
        .heap_addr = heap_addr,
        .heap_size = heap_size,
        .wpr2_addr = wpr2_addr,
        .wpr2_size = wpr2_size,
        .wpr2_end = wpr2_end,
        .non_wpr_heap_addr = non_wpr_heap_addr,
        .non_wpr_heap_size = non_wpr_heap_size,
    };
}

// ===========================================================================
// GspFwWprMeta fill (nouveau tu102_gsp_wpr_meta_init)
// ===========================================================================

/// The sysmem-side inputs needed to fill the meta (radix3 root + bootloader +
/// signature phys/size + the bootloader code/data/manifest offsets).
pub const MetaInputs = struct {
    radix3_lvl0_phys: u64,
    radix3_elf_size: u64, // == elf_size (the .fwimage size)
    bootloader_phys: u64,
    bootloader_size: u64,
    bootloader_code_offset: u64,
    bootloader_data_offset: u64,
    bootloader_manifest_offset: u64,
    signature_phys: u64,
    signature_size: u64,
};

/// Fill a GspFwWprMeta from the WPR2 layout + the sysmem inputs (radix3 / boot /
/// signature). Mirrors tu102_gsp_wpr_meta_init field-for-field. The Booter later
/// writes `verified` = VERIFIED on success.
pub fn fillWprMeta(layout: Wpr2Layout, in: MetaInputs) proto.GspFwWprMeta {
    var m = std.mem.zeroes(proto.GspFwWprMeta);
    m.magic = proto.GspFwWprMeta.MAGIC;
    m.revision = proto.GspFwWprMeta.REVISION;

    m.sysmem_addr_of_radix3_elf = in.radix3_lvl0_phys;
    m.size_of_radix3_elf = in.radix3_elf_size;
    m.sysmem_addr_of_bootloader = in.bootloader_phys;
    m.size_of_bootloader = in.bootloader_size;
    m.bootloader_code_offset = in.bootloader_code_offset;
    m.bootloader_data_offset = in.bootloader_data_offset;
    m.bootloader_manifest_offset = in.bootloader_manifest_offset;

    m.sysmem_addr_of_signature = in.signature_phys;
    m.size_of_signature = in.signature_size;

    // nouveau: gspFwRsvdStart == nonWprHeapOffset == fb.heap.addr (the non-WPR heap).
    m.gsp_fw_rsvd_start = layout.non_wpr_heap_addr;
    m.non_wpr_heap_offset = layout.non_wpr_heap_addr;
    m.non_wpr_heap_size = layout.non_wpr_heap_size;
    m.gsp_fw_wpr_start = layout.wpr2_addr;
    m.gsp_fw_heap_offset = layout.heap_addr;
    m.gsp_fw_heap_size = layout.heap_size;
    m.gsp_fw_offset = layout.elf_addr;
    m.boot_bin_offset = layout.boot_addr;
    m.frts_offset = layout.frts_addr;
    m.frts_size = layout.frts_size;
    m.gsp_fw_wpr_end = layout.wpr2_end;
    m.fb_size = layout.fb_size;
    m.vga_workspace_offset = layout.vga_workspace_addr;
    m.vga_workspace_size = layout.vga_workspace_size;
    m.boot_count = 0;
    m.partition_rpc_addr = 0;
    m.partition_rpc_request_offset = 0;
    m.partition_rpc_reply_offset = 0;
    m.verified = 0;
    return m;
}

// ===========================================================================
// libos init args + GSP_ARGUMENTS_CACHED (nouveau r535_gsp_libos_init / set_rmargs)
// ===========================================================================

/// The four libos boot regions. LOGINIT/LOGINTR/LOGRM are 64 KB log buffers;
/// RMARGS holds the GSP_ARGUMENTS_CACHED. Order matches nouveau's libos_id list.
pub const LIBOS_LOG_SIZE: u64 = 0x10000; // 64 KB
pub const LIBOS_REGION_COUNT: usize = 4;

/// The shared cmdq+msgq region geometry (one allocation, two 256 KB rings).
pub const SHARED_QUEUE_SIZE: u64 = 0x40000; // 256 KB per ring (RM page-table sized)

/// The built boot-args bundle: the 4 libos regions + the GSP_ARGUMENTS_CACHED +
/// the phys addrs of the shared queue region + the args region itself.
pub const BootArgs = struct {
    regions: [LIBOS_REGION_COUNT]proto.LibosMemoryRegionInitArgument,
    args: proto.GspArgumentsCached,
    /// Phys of the GSP_ARGUMENTS_CACHED region (== the RMARGS region pa).
    args_phys: u64,
    /// Phys of the shared cmdq+msgq region.
    shared_phys: u64,
    /// Phys of the radix3 root (also handed to the RMARGS libos region as RADIX3).
    radix3_lvl0_phys: u64,
};

/// Build the libos regions + GSP_ARGUMENTS_CACHED. Allocates from `sysmem` (the
/// bump allocator over the sysmem pool WE own). The radix3 root phys is passed in
/// (built earlier from the image); the RMARGS libos region is tagged RADIX3 and
/// points at it (r535_gsp_libos_init: the args region uses the radix3 mapping).
///
/// `page_table_entry_count` is the number of 4 KB pages backing the shared queue
/// region (MESSAGE_QUEUE_INIT_ARGUMENTS.pageTableEntryCount).
pub fn buildBootArgs(
    sysmem: *BumpAllocator,
    radix3_lvl0_phys: u64,
) Error!BootArgs {
    // The three log regions (contiguous, sysmem).
    const loginit_pa = try sysmem.alloc(LIBOS_LOG_SIZE);
    const logintr_pa = try sysmem.alloc(LIBOS_LOG_SIZE);
    const logrm_pa = try sysmem.alloc(LIBOS_LOG_SIZE);

    // The shared cmdq+msgq region: two 256 KB rings, one allocation.
    const shared_size = SHARED_QUEUE_SIZE * 2;
    const shared_pa = try sysmem.alloc(shared_size);

    // The RMARGS region holds the GSP_ARGUMENTS_CACHED.
    const args_pa = try sysmem.alloc(@sizeOf(proto.GspArgumentsCached));

    const L = proto.LibosMemoryRegionInitArgument;
    const sysmem_loc = @intFromEnum(proto.LibosMemoryRegionLoc.sysmem);
    const contig = @intFromEnum(proto.LibosMemoryRegionKind.contiguous);
    const radix3_kind = @intFromEnum(proto.LibosMemoryRegionKind.radix3);

    var regions: [LIBOS_REGION_COUNT]L = undefined;
    regions[0] = .{ .id8 = L.idFromTag("LOGINIT"), .pa = loginit_pa, .size = LIBOS_LOG_SIZE, .kind = contig, .loc = sysmem_loc };
    regions[1] = .{ .id8 = L.idFromTag("LOGINTR"), .pa = logintr_pa, .size = LIBOS_LOG_SIZE, .kind = contig, .loc = sysmem_loc };
    regions[2] = .{ .id8 = L.idFromTag("LOGRM"), .pa = logrm_pa, .size = LIBOS_LOG_SIZE, .kind = contig, .loc = sysmem_loc };
    // RMARGS is mapped through the radix3 root (the GSP-RM ELF mapping).
    regions[3] = .{ .id8 = L.idFromTag("RMARGS"), .pa = radix3_lvl0_phys, .size = RADIX_PAGE_SIZE, .kind = radix3_kind, .loc = sysmem_loc };

    // GSP_ARGUMENTS_CACHED: the message-queue init args point at the shared region.
    const pte_count: u32 = @intCast(divRoundUp(shared_size, RADIX_PAGE_SIZE));
    var args = std.mem.zeroes(proto.GspArgumentsCached);
    args.message_queue_init_arguments = .{
        .shared_mem_phys_addr = shared_pa,
        .page_table_entry_count = pte_count,
        .cmd_queue_offset = 0, // cmdq at the start of the shared region
        .stat_queue_offset = SHARED_QUEUE_SIZE, // msgq right after the cmdq
        .queue_element_hdr_size = proto.GspMsgQueueElement.HDR_SIZE,
        .queue_element_size_min = proto.ELEMENT_SIZE_MIN,
        .queue_element_size_max = proto.ELEMENT_SIZE_MAX,
        .queue_header_align = @as(u32, 1) << proto.QUEUE_HEADER_ALIGN_SHIFT,
        .queue_element_align = @as(u32, 1) << proto.QUEUE_ELEMENT_ALIGN_SHIFT,
    };
    args.gpu_instance = 0;
    args.b_dmem_stack = 1; // nouveau sets bDmemStack = 1 on the GSP-RM boot path

    return .{
        .regions = regions,
        .args = args,
        .args_phys = args_pa,
        .shared_phys = shared_pa,
        .radix3_lvl0_phys = radix3_lvl0_phys,
    };
}

// ===========================================================================
// Firmware byte injection (the @embedFile / ESP story for P4)
// ===========================================================================

// P4 (the boot phase) needs the gsp_ga10x.bin bytes. On the freestanding/UEFI
// build there is no filesystem at run-the-driver time, so the bytes must be
// either:
//   (A) @embedFile'd into the UEFI binary (the ~70 MB image becomes part of
//       BOOTX64.efi). SIMPLEST + self-contained. RECOMMENDED for P4 as the
//       default - keyed per-arch (gsp_ga10x.bin vs gsp_tu10x.bin) by chip.
//   (B) read from the ESP via EFI_SIMPLE_FILE_SYSTEM_PROTOCOL before
//       ExitBootServices. Keeps the binary small; needs a file-read path.
// This module NEVER reads the file itself - it only takes the bytes as a slice
// (see Elf.parse / extractSections), so it stays freestanding-clean. The
// embed/ESP choice is entirely in the P4 caller (and only compiled on UEFI).
//
// Sketch for P4 (do NOT enable here - it would pull 70 MB into the test binary):
//   pub const ga10x_firmware = @embedFile("gsp_ga10x.bin"); // option A
// then: const sections = try extractSections(try Elf.parse(ga10x_firmware),
//                                            signatureSectionFor(.ga10x));

/// The known 595.71.05 gsp_ga10x.bin path in the nix store (used by the offline
/// tests). If the environment differs, the test skips rather than fails.
pub const GA10X_STORE_PATH: []const u8 =
    "/nix/store/h6qir1zdzxhn9cbj4rndbb806lvp5w1i-nvidia-x11-aarch64-unknown-linux-gnu-595.71.05-firmware/lib/firmware/nvidia/595.71.05/gsp_ga10x.bin";

// ===========================================================================
// little-endian readers (no alignment assumptions about the input buffer)
// ===========================================================================

fn rdU16(b: []const u8, off: u64) u64 {
    const o: usize = @intCast(off);
    return std.mem.readInt(u16, b[o..][0..2], .little);
}
fn rdU32(b: []const u8, off: u64) u32 {
    const o: usize = @intCast(off);
    return std.mem.readInt(u32, b[o..][0..4], .little);
}
fn rdU64(b: []const u8, off: u64) u64 {
    const o: usize = @intCast(off);
    return std.mem.readInt(u64, b[o..][0..8], .little);
}

// ===========================================================================
// Tests (offline)
// ===========================================================================

const testing = std.testing;

test "radix3 layout: small image (single data page)" {
    const r = try radix3Layout(4096);
    try testing.expectEqual(@as(u64, 1), r.n_pages[3]); // 1 data page
    try testing.expectEqual(@as(u64, 1), r.n_pages[2]);
    try testing.expectEqual(@as(u64, 1), r.n_pages[1]);
    try testing.expectEqual(@as(u64, 1), r.n_pages[0]);
    // offsets low->high: lvl0=0, lvl1=4096, lvl2=8192, data=12288.
    try testing.expectEqual(@as(u64, 0), r.offsets[0]);
    try testing.expectEqual(@as(u64, 4096), r.offsets[1]);
    try testing.expectEqual(@as(u64, 8192), r.offsets[2]);
    try testing.expectEqual(@as(u64, 12288), r.offsets[3]);
    try testing.expectEqual(@as(u64, 3 * 4096), r.pt_size);
}

test "radix3 layout: ceil math + lvl0==1 invariant" {
    // 513 data pages -> lvl2 needs ceil(513/512) = 2 pages -> lvl1 = 1 -> lvl0 = 1.
    const size = 513 * RADIX_PAGE_SIZE;
    const r = try radix3Layout(size);
    try testing.expectEqual(@as(u64, 513), r.n_pages[3]);
    try testing.expectEqual(@as(u64, 2), r.n_pages[2]);
    try testing.expectEqual(@as(u64, 1), r.n_pages[1]);
    try testing.expectEqual(@as(u64, 1), r.n_pages[0]);
}

test "radix3 layout: crosses an lvl1 boundary (lvl1 > 1 page)" {
    // Need lvl2 > 512 pages so lvl1 needs > 1 page.
    // lvl2 pages = ceil(data/512); want that > 512 -> data > 512*512 = 262144.
    const data_pages: u64 = 262145; // -> lvl2 = ceil(262145/512) = 513 -> lvl1 = 2 -> lvl0 = 1
    const size = data_pages * RADIX_PAGE_SIZE;
    const r = try radix3Layout(size);
    try testing.expectEqual(@as(u64, 262145), r.n_pages[3]);
    try testing.expectEqual(@as(u64, 513), r.n_pages[2]);
    try testing.expectEqual(@as(u64, 2), r.n_pages[1]);
    try testing.expectEqual(@as(u64, 1), r.n_pages[0]);
}

const FakePages = struct {
    base: u64,
    fn addr(ctx: *anyopaque, i: u64) u64 {
        const self: *FakePages = @ptrCast(@alignCast(ctx));
        // Deliberately non-contiguous so the test proves each PTE is distinct.
        return self.base + i * 0x2000 + 0x10000000;
    }
};

test "radix3 build: PTEs point at the right phys addresses" {
    var fake = FakePages{ .base = 0 };
    const data_pages: u64 = 600; // -> lvl2 = 2 pages, lvl1 = 1, lvl0 = 1
    const size = data_pages * RADIX_PAGE_SIZE;
    const layout = try radix3Layout(size);
    const region = try testing.allocator.alloc(u8, @intCast(layout.pt_size));
    defer testing.allocator.free(region);
    const pt_phys: u64 = 0x40000000;
    const r = try buildRadix3(size, region, pt_phys, &fake, FakePages.addr);

    try testing.expectEqual(pt_phys, r.lvl0_phys);
    // lvl0[0] -> lvl1 phys.
    try testing.expectEqual(pt_phys + r.offsets[1], readPte(region, r.offsets[0], 0));
    // lvl1[k] -> lvl2 page k phys (contiguous in the region).
    const lvl2_phys = pt_phys + r.offsets[2];
    var k: u64 = 0;
    while (k < r.n_pages[2]) : (k += 1) {
        try testing.expectEqual(lvl2_phys + k * RADIX_PAGE_SIZE, readPte(region, r.offsets[1], k));
    }
    // lvl2[p] -> data page p phys (from the fake allocator).
    var p: u64 = 0;
    while (p < r.n_pages[3]) : (p += 1) {
        try testing.expectEqual(FakePages.addr(&fake, p), readPte(region, r.offsets[2], p));
    }
}

test "radix3 build: across an lvl1 boundary the lvl1 PTEs span 2 pages" {
    var fake = FakePages{ .base = 0x1000 };
    const data_pages: u64 = 262145;
    const size = data_pages * RADIX_PAGE_SIZE;
    const layout = try radix3Layout(size);
    const region = try testing.allocator.alloc(u8, @intCast(layout.pt_size));
    defer testing.allocator.free(region);
    const pt_phys: u64 = 0x80000000;
    const r = try buildRadix3(size, region, pt_phys, &fake, FakePages.addr);
    try testing.expectEqual(@as(u64, 2), r.n_pages[1]);
    // The 513th lvl1 PTE (index 512) lives in the 2nd lvl1 page; assert it is
    // populated (non-zero) and equal to lvl2 page 512's phys.
    const lvl2_phys = pt_phys + r.offsets[2];
    try testing.expectEqual(lvl2_phys + 512 * RADIX_PAGE_SIZE, readPte(region, r.offsets[1], 512));
}

test "BumpAllocator: page-aligned + exhaustion" {
    var a = BumpAllocator.init(0x100000000, 0x10000);
    const p0 = try a.alloc(1);
    try testing.expectEqual(@as(u64, 0x100000000), p0);
    const p1 = try a.alloc(4097); // 2 pages
    try testing.expectEqual(@as(u64, 0x100001000), p1);
    try testing.expectEqual(@as(u64, 0x100003000), a.base_phys + a.used());
    try testing.expectError(Error.OutOfMemory, a.alloc(0x10000));
}

test "WPR2 layout: ordered, page-aligned, fits, sizes correct" {
    // 8 GB VRAM (GA104/RTX 3070 typical).
    const fb_size: u64 = 8 * (1 << 30);
    const vga_size: u64 = 0x10000;
    const bios_addr: u64 = fb_size - 0x100000; // just below the very top
    const boot_size: u64 = 0x40000; // ~256 KB bootloader
    const elf_size: u64 = 72_853_408; // ~ the real .fwimage size
    const hp = HeapParams{};
    const L = try wpr2Layout(fb_size, bios_addr, vga_size, boot_size, elf_size, hp);

    // Top-down ordering (addr decreasing): non_wpr < wpr2 < heap < elf < boot < frts < end.
    try testing.expect(L.non_wpr_heap_addr < L.wpr2_addr);
    try testing.expect(L.wpr2_addr < L.heap_addr);
    try testing.expect(L.heap_addr < L.elf_addr);
    try testing.expect(L.elf_addr < L.boot_addr);
    try testing.expect(L.boot_addr < L.frts_addr);
    try testing.expect(L.frts_addr < L.wpr2_end);
    try testing.expect(L.wpr2_end <= fb_size);

    // Alignments.
    try testing.expectEqual(@as(u64, 0), L.wpr2_end & 0x1ffff); // 0x20000
    try testing.expectEqual(@as(u64, 0), L.boot_addr & 0xfff); // 0x1000
    try testing.expectEqual(@as(u64, 0), L.elf_addr & 0xffff); // 0x10000
    try testing.expectEqual(@as(u64, 0), L.heap_addr & 0xfffff); // 0x100000
    try testing.expectEqual(@as(u64, 0), L.wpr2_addr & 0xfffff);

    // Sizes.
    try testing.expectEqual(FRTS_SIZE, L.frts_size);
    try testing.expectEqual(NON_WPR_HEAP_SIZE, L.non_wpr_heap_size);
    try testing.expectEqual(elf_size, L.elf_size);
    // Heap is floored at heap_size_min (84 MB) - assert it is at least that.
    try testing.expect(L.heap_size >= hp.heap_size_min);
    // Non-overlap: each region's [addr, addr+size) is disjoint + adjacent-ish.
    try testing.expect(L.heap_addr + L.heap_size <= L.elf_addr);
    try testing.expect(L.elf_addr + L.elf_size <= L.boot_addr);
    try testing.expect(L.boot_addr + L.boot_size <= L.frts_addr);
    try testing.expect(L.frts_addr + L.frts_size == L.wpr2_end);
}

test "wprHeapSize: GA10x formula floors at heap_size_min" {
    const hp = HeapParams{};
    // 8 GB: per-GB = ALIGN(96KB*8, 1MB) = ALIGN(768KB,1MB) = 1MB; client = ALIGN(48KB*2048,1MB).
    const got = wprHeapSize(8 * (1 << 30), hp);
    const client = alignUp(HeapParams.CLIENT_ALLOC_SIZE, MB);
    const expect = @max(hp.os_carveout_size + hp.base_size + (1 * MB) + client, hp.heap_size_min);
    try testing.expectEqual(expect, got);
    try testing.expect(got >= hp.heap_size_min);
}

test "fillWprMeta: magic/revision + pointers match the layout" {
    const fb_size: u64 = 8 * (1 << 30);
    const L = try wpr2Layout(fb_size, fb_size - 0x100000, 0x10000, 0x40000, 72_853_408, HeapParams{});
    const meta = fillWprMeta(L, .{
        .radix3_lvl0_phys = 0xdead0000,
        .radix3_elf_size = L.elf_size,
        .bootloader_phys = 0xbeef0000,
        .bootloader_size = 0x40000,
        .bootloader_code_offset = 0x100,
        .bootloader_data_offset = 0x200,
        .bootloader_manifest_offset = 0x300,
        .signature_phys = 0xcafe0000,
        .signature_size = 0x1000,
    });
    try testing.expectEqual(proto.GspFwWprMeta.MAGIC, meta.magic);
    try testing.expectEqual(proto.GspFwWprMeta.REVISION, meta.revision);
    try testing.expectEqual(@as(u64, 0xdead0000), meta.sysmem_addr_of_radix3_elf);
    try testing.expectEqual(L.elf_size, meta.size_of_radix3_elf);
    try testing.expectEqual(@as(u64, 0xcafe0000), meta.sysmem_addr_of_signature);
    try testing.expectEqual(L.elf_addr, meta.gsp_fw_offset);
    try testing.expectEqual(L.boot_addr, meta.boot_bin_offset);
    try testing.expectEqual(L.frts_addr, meta.frts_offset);
    try testing.expectEqual(L.frts_size, meta.frts_size);
    try testing.expectEqual(L.wpr2_addr, meta.gsp_fw_wpr_start);
    try testing.expectEqual(L.wpr2_end, meta.gsp_fw_wpr_end);
    try testing.expectEqual(L.heap_addr, meta.gsp_fw_heap_offset);
    try testing.expectEqual(L.heap_size, meta.gsp_fw_heap_size);
    try testing.expectEqual(L.non_wpr_heap_addr, meta.non_wpr_heap_offset);
    try testing.expectEqual(L.fb_size, meta.fb_size);
    try testing.expectEqual(@as(u64, 0), meta.verified);
    try testing.expectEqual(@as(u64, 0), meta.boot_count);
}

test "buildBootArgs: regions + message queue init args" {
    // 4 MB sysmem pool.
    var pool: [1]u8 = undefined; // unused; allocator is phys-only
    _ = &pool;
    var sysmem = BumpAllocator.init(0x200000000, 4 * (1 << 20));
    const radix3_root: u64 = 0x123000;
    const ba = try buildBootArgs(&sysmem, radix3_root);

    // 3 log regions + RMARGS.
    try testing.expectEqual(@as(usize, 4), ba.regions.len);
    try testing.expectEqual(proto.LibosMemoryRegionInitArgument.idFromTag("LOGINIT"), ba.regions[0].id8);
    try testing.expectEqual(@as(u64, LIBOS_LOG_SIZE), ba.regions[0].size);
    try testing.expectEqual(@intFromEnum(proto.LibosMemoryRegionLoc.sysmem), ba.regions[0].loc);
    // RMARGS is RADIX3 + points at the radix3 root.
    try testing.expectEqual(@intFromEnum(proto.LibosMemoryRegionKind.radix3), ba.regions[3].kind);
    try testing.expectEqual(radix3_root, ba.regions[3].pa);

    // Message-queue init args.
    const mq = ba.args.message_queue_init_arguments;
    try testing.expectEqual(ba.shared_phys, mq.shared_mem_phys_addr);
    try testing.expectEqual(@as(u64, 0), mq.cmd_queue_offset);
    try testing.expectEqual(@as(u64, SHARED_QUEUE_SIZE), mq.stat_queue_offset);
    try testing.expectEqual(@as(u64, proto.GspMsgQueueElement.HDR_SIZE), mq.queue_element_hdr_size);
    try testing.expectEqual(proto.ELEMENT_SIZE_MIN, @as(u32, @intCast(mq.queue_element_size_min)));
    // pageTableEntryCount = ceil(2*256KB / 4KB) = 128.
    try testing.expectEqual(@as(u32, 128), mq.page_table_entry_count);
    try testing.expectEqual(@as(u8, 1), ba.args.b_dmem_stack);

    // The id8 packing matches *(u64*)"LOGRM\0\0\0".
    const expect_logrm: u64 = 'L' | (@as(u64, 'O') << 8) | (@as(u64, 'G') << 16) | (@as(u64, 'R') << 24) | (@as(u64, 'M') << 32);
    try testing.expectEqual(expect_logrm, ba.regions[2].id8);
}

test "ELF parse against the REAL gsp_ga10x.bin (595.71.05)" {
    if (builtin.target.os.tag == .freestanding) return error.SkipZigTest;
    const alloc = testing.allocator;
    const path = GA10X_STORE_PATH;

    // Zig 0.16 routes filesystem IO through an Io instance (the threaded backend
    // here). This stays in a !freestanding test only, so the UEFI build never
    // links std.Io.Threaded / std.fs.
    var threaded = std.Io.Threaded.init(alloc, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const cwd = std.Io.Dir.cwd();

    // If the file is not present in this environment, skip (do not fail).
    cwd.access(io, path, .{}) catch return error.SkipZigTest;

    // Read the whole container (~70 MB - a sub-second read) and parse it. We
    // only touch the headers + 3 small sections; the bulk .fwimage body is never
    // walked, just sized.
    const bytes = cwd.readFileAlloc(io, path, alloc, .limited(128 * 1024 * 1024)) catch return error.SkipZigTest;
    defer alloc.free(bytes);

    const elf = try Elf.parse(bytes);
    const sections = try extractSections(elf, signatureSectionFor(.ga10x));
    // .fwversion decodes to the pinned string.
    try testing.expectEqualStrings(FW_VERSION, sections.version);
    // .fwimage is the bulk (tens of MB) - assert plausible size.
    try testing.expect(sections.image.len > 16 * (1 << 20));
    try testing.expect(sections.image.len < 128 * (1 << 20));
    // .fwsignature_ga10x is one 4 KB blob.
    try testing.expectEqual(@as(usize, 0x1000), sections.signature.len);
}
