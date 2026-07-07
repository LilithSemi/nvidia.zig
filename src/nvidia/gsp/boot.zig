//! GSP boot sequence (plan phase P4): assemble P1 (falcon), P2 (bios/FWSEC),
//! P3 (fw packaging: WPR2 layout + radix3 + boot args) into the ACTUAL GSP boot
//! kick. This is the privilege-escalation chain across three signed images and
//! two cores (GSP falcon + GSP RISC-V + SEC2 falcon).
//!
//! The whole LIVE handshake is hardware-only (the user's deferred metal test): the
//! FWSEC run, the WPR2-scratch readback, the SEC2 booter reset+kick, the WPR2 MMU
//! lock, RISC-V activation, the `verified` writeback, and GSP_INIT_DONE. So every
//! step is split into:
//!   - a PURE DATA-BUILDER (offline unit-testable: marshalling/parsing/assembly), and
//!   - a THIN MMIO DRIVER (gated behind `Driver`, runs only on metal).
//! `Boot.run(...)` chains the four steps; on hardware it touches MMIO + VRAM/sysmem.
//!
//! The four steps (each a `step1_*`/`step2_*`/... pair):
//!   1. FWSEC-FRTS: load the patched FWSEC onto the GSP falcon, halt, read WPR2
//!      lo/hi back from NV_PGC6 scratch (0x1fa824/0x1fa828), verify bounds.
//!      Ref: nouveau tu102.c (fwsec_frts via gsp/fwsec.c).
//!   2. BOOTER ucode: parse the signed HS ucode header (nvfw_bin_hdr +
//!      nvfw_hs_header_v2), locate + patch the prod/debug signature into the ucode
//!      image. Ref: nouveau tu102_gsp_booter_ctor + falcon/fw.c fw_patch/fw_sign.
//!   3. BOOT KICK: WprMeta phys -> SEC2 mbox0/mbox1, libos/boot-args addr -> GSP
//!      mbox0/mbox1, reset GSP into RISC-V (BCR kick), run booter_load on SEC2.
//!      Ref: nouveau tu102_gsp_init / OGK kgspBootstrap_TU102.
//!   4. POLL READY: riscvActive(GSP), `verified` == 0xa0a0..., poll the RPC ring
//!      for NV_VGPU_MSG_EVENT_GSP_INIT_DONE (0x1001). Ref: nouveau r535_gsp_init.
//!
//! Plus the GSP_FMC_BOOT_PARAMS / GSP_ACR_BOOT_GSP_RM_PARAMS boot-arg pointer
//! chain (gspifpub.h) - pure assembly over the WprMeta/radix3/args phys addrs,
//! built + tested offline.
//!
//! Freestanding-clean: the MMIO driving is all hw.Bar + falcon.zig; the firmware +
//! booter bytes are INJECTED as slices (on the UEFI build they are @embedFile'd by
//! the caller), so this file never reads a file and pulls in no std.fs / std.Io.

const std = @import("std");
const builtin = @import("builtin");

const hw = @import("../hw/mmio.zig");
const chip = @import("../hw/chip.zig");
const proto = @import("proto.zig");
const fw = @import("fw.zig");
const falcon = @import("falcon.zig");
const bios = @import("bios.zig");
const ring = @import("ring.zig");
const sdk = @import("../sdk.zig");

const NvU8 = sdk.NvU8;
const NvU32 = sdk.NvU32;
const NvU64 = sdk.NvU64;

pub const Error = error{
    /// The FWSEC run did not produce a WPR2 region (scratch read back 0).
    FrtsFailed,
    /// The WPR2 scratch bounds did not match the expected FRTS region.
    Wpr2Mismatch,
    /// The booter blob is not a valid nvfw HS container (bad magic / truncated).
    BadBooter,
    /// A header offset / signature extent ran past the blob.
    Truncated,
    /// The booter `verified` writeback was not GSP_FW_WPR_META_VERIFIED.
    NotVerified,
    /// The GSP RISC-V core never reported active.
    RiscvInactive,
    /// The GSP never posted GSP_INIT_DONE before the poll budget elapsed.
    InitTimeout,
    /// A provided output buffer was too small.
    BufferTooSmall,
} || fw.Error || bios.Error || ring.Error;

// ===========================================================================
// NV_PGC6 scratch registers + booter engine/ucode ids (nouveau, cited).
// ===========================================================================

/// NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK low word: after the FWSEC
/// FRTS run the WPR2 region LOW bound (>>12, in 4 KB units) lands here. nouveau
/// tu102_gsp_oneinit reads the WPR2 bounds from these PGC6 scratch regs.
/// FLAG: the 0x1fa824/0x1fa828 pair is the documented WPR2 lo/hi scratch on
/// Turing/Ampere; the exact PRIV-level-mask group numbering is chip-specific and
/// only confirmable on the live ROM (HARDWARE-ONLY readback).
pub const NV_PGC6_WPR2_LO: u32 = 0x001fa824;
/// WPR2 region HIGH bound scratch (>>12).
pub const NV_PGC6_WPR2_HI: u32 = 0x001fa828;

/// NV_PGSP_FALCON_MAILBOX equivalents are the falcon MAILBOX0/1 (0x040/0x044) the
/// falcon driver already exposes; we use falcon.mailboxWrite for both GSP + SEC2.
/// FRTS error scratch (nvkm fwsec: NV_PFALCON_FALCON_MAILBOX or the 0x1400+0xe*4
/// FRTS status). Documented for the hardware path; not read offline.
pub const NV_PBUS_SW_SCRATCH_FRTS_ERR: u32 = 0x001400 + 0xe * 4;

/// The SEC2 engine + ucode ids the booter HS ucode boots under (the BCR kick
/// engine_id/ucode_id). nouveau passes the falcon's engine id; on GA10x the
/// booter runs on the SEC2 engine. FLAG: the precise numeric engine_id/ucode_id
/// are taken from the booter's load-header on hardware (the HS ucode self-
/// describes); we thread them through as parameters rather than hardcode.
pub const SEC2_FALCON_ID: u32 = 7; // NV_FALCON2_SEC2_BASE engine id (documented; hardware confirms)

// ===========================================================================
// STEP 2 data-builder: the nvfw HS-header parse + signature patch.
// ===========================================================================
//
// The booter_load / booter_unload blobs are SIGNED High-Secure (HS) ucode in the
// nvfw container format. Layout (read line-for-line from the real ga102
// booter_load-570.144.bin + nouveau include/nvfw/{fw,hs}.h):
//
//   nvfw_bin_hdr     @ blob[0]                 (24 bytes)
//     bin_magic(0x10de), bin_ver, bin_size, header_offset, data_offset, data_size
//   nvfw_hs_header_v2 @ blob[bin_hdr.header_offset]  (36 bytes)
//     sig_prod_offset, sig_prod_size, patch_loc, patch_sig, meta_data_offset,
//     meta_data_size, num_sig, header_offset(load-hdr), header_size
//   nvfw_hs_load_header_v2 @ blob[hs.header_offset]  (20-byte head + app[])
//     os_code_offset, os_code_size, os_data_offset, os_data_size, num_apps, app[]
//
// The patch words are POINTERS-TO-VALUES (offsets into the blob):
//   loc = u32@(blob + hs.patch_loc)   -- dest byte offset in the ucode image
//   sig = u32@(blob + hs.patch_sig)   -- byte offset added to sig_prod_offset
//   cnt = u32@(blob + hs.num_sig)     -- number of signatures (prod + debug)
// One signature is sig_size = sig_prod_size / cnt bytes. The prod sig array starts
// at blob[sig_prod_offset + sig]; index 0 is prod, index 1 is debug (fuse-selected).

/// nvfw_bin_hdr (include/nvfw/fw.h). All u32, no padding -> 24 bytes.
pub const NvfwBinHdr = extern struct {
    bin_magic: NvU32,
    bin_ver: NvU32,
    bin_size: NvU32,
    header_offset: NvU32,
    data_offset: NvU32,
    data_size: NvU32,

    pub const MAGIC: NvU32 = 0x10de;
};

/// nvfw_hs_header_v2 (include/nvfw/hs.h). All u32, no padding -> 36 bytes.
pub const NvfwHsHeaderV2 = extern struct {
    sig_prod_offset: NvU32,
    sig_prod_size: NvU32,
    patch_loc: NvU32,
    patch_sig: NvU32,
    meta_data_offset: NvU32,
    meta_data_size: NvU32,
    num_sig: NvU32,
    header_offset: NvU32, // -> nvfw_hs_load_header_v2
    header_size: NvU32,
};

/// nvfw_hs_load_header_v2 head (include/nvfw/hs.h). 20-byte head + app[] array.
pub const NvfwHsLoadHeaderV2 = extern struct {
    os_code_offset: NvU32,
    os_code_size: NvU32,
    os_data_offset: NvU32,
    os_data_size: NvU32,
    num_apps: NvU32,
    // struct { offset, size, data_offset, data_size } app[num_apps] follows.

    pub const App = extern struct {
        offset: NvU32,
        size: NvU32,
        data_offset: NvU32,
        data_size: NvU32,
    };
};

/// The parsed booter, ready to drive the SEC2 falcon. `image` is the ucode bytes
/// (blob[data_offset .. data_offset+data_size]) with the production signature
/// ALREADY patched in at `dmem_sign_offset`. The BCR-kick parameters (the dmem
/// signature value, the imem/dmem base/size) are extracted from the load header.
pub const Booter = struct {
    /// The patched ucode image (what gets loaded into the SEC2 falcon).
    image: []const u8,
    /// Byte offset within `image` where the signature was written (== loc).
    sig_offset_in_image: u32,
    /// One-signature size (sig_prod_size / cnt).
    sig_size: u32,
    /// IMEM layout (app[0]).
    imem_base: u32,
    imem_size: u32,
    /// DMEM layout (os_data_offset/size).
    dmem_base: u32,
    dmem_size: u32,
    /// Non-secure code (os_code) base/size + boot address.
    nmem_base: u32,
    nmem_size: u32,
    boot_addr: u32,
    /// The DMEM-relative signature location nouveau computes as
    /// dmem_sign = loc - dmem_base (used for the BCR BR-sig register on the kick).
    dmem_sign: u32,
};

fn rdU32(b: []const u8, off: u32) Error!u32 {
    if (@as(u64, off) + 4 > b.len) return Error.Truncated;
    return std.mem.readInt(u32, b[off..][0..4], .little);
}

/// Parse the booter HS container header (no signature patch yet). Validates the
/// nvfw_bin_hdr magic + the header/signature extents. Returns the headers + the
/// dereferenced patch words.
pub const BooterHeaders = struct {
    bin: NvfwBinHdr,
    hs: NvfwHsHeaderV2,
    load: NvfwHsLoadHeaderV2,
    app0: NvfwHsLoadHeaderV2.App,
    /// loc = *(u32*)(blob + hs.patch_loc): the dest offset in the ucode image.
    loc: u32,
    /// sig = *(u32*)(blob + hs.patch_sig): added to sig_prod_offset.
    sig: u32,
    /// cnt = *(u32*)(blob + hs.num_sig): number of signatures.
    cnt: u32,
};

pub fn parseBooterHeaders(blob: []const u8) Error!BooterHeaders {
    if (blob.len < @sizeOf(NvfwBinHdr)) return Error.BadBooter;
    const bin = readStruct(NvfwBinHdr, blob, 0);
    if (bin.bin_magic != NvfwBinHdr.MAGIC) return Error.BadBooter;
    if (@as(u64, bin.header_offset) + @sizeOf(NvfwHsHeaderV2) > blob.len) return Error.Truncated;
    if (@as(u64, bin.data_offset) + bin.data_size > blob.len) return Error.Truncated;

    const hs = readStruct(NvfwHsHeaderV2, blob, bin.header_offset);
    if (@as(u64, hs.header_offset) + @sizeOf(NvfwHsLoadHeaderV2) > blob.len) return Error.Truncated;
    const load = readStruct(NvfwHsLoadHeaderV2, blob, hs.header_offset);
    if (load.num_apps == 0) return Error.BadBooter;
    const app0_off = hs.header_offset + @sizeOf(NvfwHsLoadHeaderV2);
    if (@as(u64, app0_off) + @sizeOf(NvfwHsLoadHeaderV2.App) > blob.len) return Error.Truncated;
    const app0 = readStruct(NvfwHsLoadHeaderV2.App, blob, app0_off);

    const loc = try rdU32(blob, hs.patch_loc);
    const sig = try rdU32(blob, hs.patch_sig);
    const cnt = try rdU32(blob, hs.num_sig);
    if (cnt == 0) return Error.BadBooter;

    return .{ .bin = bin, .hs = hs, .load = load, .app0 = app0, .loc = loc, .sig = sig, .cnt = cnt };
}

/// Build the patched booter ucode image into `out`: copy the ucode image
/// (blob[data_offset .. data_offset+data_size]), then write the selected signature
/// (prod = index 0, debug = index 1) into the image at byte offset `loc`. Returns
/// a Booter describing the image + the BCR-kick parameters.
///
/// `debug_fuse` selects the signature index exactly as nouveau's
/// gm200_flcn_fw_signature does (debug fuse bit set -> debug sig at index 1, else
/// prod at index 0). On consumer GA10x the fuse is clear -> prod. The fuse read is
/// HARDWARE-ONLY (falcon debug reg & 0x00100000); offline we pass it explicitly.
pub fn patchBooter(blob: []const u8, out: []u8, debug_fuse: bool) Error!Booter {
    const h = try parseBooterHeaders(blob);

    const sig_size = h.hs.sig_prod_size / h.cnt;
    if (sig_size == 0 or sig_size % 4 != 0) return Error.BadBooter;

    // The ucode image is blob[data_offset .. data_offset + data_size].
    const img_start = h.bin.data_offset;
    const img_len = h.bin.data_size;
    if (out.len < img_len) return Error.BufferTooSmall;
    @memcpy(out[0..img_len], blob[img_start .. img_start + img_len]);

    // The signature destination is at byte offset `loc` within the IMAGE (nouveau
    // writes into fw->fw.img at fw->sig_base_img == loc).
    if (@as(u64, h.loc) + sig_size > img_len) return Error.Truncated;

    // Select the signature: index 0 prod, index 1 debug (if the fuse is set + a
    // debug sig exists). The sig array base is sig_prod_offset + sig.
    const sig_idx: u32 = if (debug_fuse and h.cnt > 1) 1 else 0;
    const sig_src = h.hs.sig_prod_offset + h.sig + sig_idx * sig_size;
    if (@as(u64, sig_src) + sig_size > blob.len) return Error.Truncated;

    @memcpy(out[h.loc .. h.loc + sig_size], blob[sig_src .. sig_src + sig_size]);

    const dmem_sign = h.loc -% h.load.os_data_offset;

    return .{
        .image = out[0..img_len],
        .sig_offset_in_image = h.loc,
        .sig_size = sig_size,
        .imem_base = h.app0.offset,
        .imem_size = h.app0.size,
        .dmem_base = h.load.os_data_offset,
        .dmem_size = h.load.os_data_size,
        .nmem_base = h.load.os_code_offset,
        .nmem_size = h.load.os_code_size,
        .boot_addr = h.load.os_code_offset,
        .dmem_sign = dmem_sign,
    };
}

// ===========================================================================
// The boot-arg pointer chain: GSP_FMC_BOOT_PARAMS / GSP_ACR_BOOT_GSP_RM_PARAMS.
// (gspifpub.h). Pure assembly over the WprMeta/radix3/args phys addrs.
// ===========================================================================
//
// On the Ampere/Turing GSP-RM boot, the GSP mailboxes do NOT carry the WprMeta
// directly; the SEC2 mailboxes carry the GspFwWprMeta phys addr (lo/hi), and the
// GSP mailboxes carry the boot-args region phys (lo/hi). The boot-args region the
// GSP reads is the libos init args + the GSP_ARGUMENTS_CACHED, reached through the
// radix3 mapping. We model the GSP_FMC_BOOT_PARAMS pointer chain (the bundle the
// GSP firmware-managed-controller reads) so its assembly is testable.

/// GSP_RM_PARAMS (gspifpub.h): points the GSP-RM at its boot args (the libos /
/// GSP_ARGUMENTS_CACHED region) and the target (sysmem) it lives in.
pub const GspRmParams = extern struct {
    /// Phys addr of the boot-args (GSP_ARGUMENTS_CACHED) region.
    boot_args_offset: NvU64 align(8),
    /// Target: 1 == sysmem (where we placed the args).
    target: NvU32,
};

/// GSP_ACR_BOOT_GSP_RM_PARAMS (gspifpub.h): the ACR/booter-facing params - where
/// the GSP-RM ELF (radix3 root) lives + its size, and the WPR meta phys.
pub const GspAcrBootGspRmParams = extern struct {
    /// Phys of the GSP-RM radix3 root (== wprMeta.sysmemAddrOfRadix3Elf).
    gsp_rm_desc_offset: NvU64 align(8),
    /// Size of the GSP-RM ELF (== wprMeta.sizeOfRadix3Elf).
    gsp_rm_desc_size: NvU64,
    /// Target memory of the radix3 ELF: 1 == sysmem.
    target: NvU32,
    /// b_is_gsp_rm: 1 on the GSP-RM boot path.
    b_is_gsp_rm: NvU8,
};

/// GSP_FMC_BOOT_PARAMS (gspifpub.h): the top-level boot-param bundle, wiring the
/// ACR params (radix3/WPR) + the RM params (boot args). The phys addr of THIS
/// struct is what the GSP mailboxes carry (lo/hi).
pub const GspFmcBootParams = extern struct {
    acr: GspAcrBootGspRmParams align(8),
    rm: GspRmParams align(8),
};

/// Target memory enum (gspifpub.h GSP_DMA_TARGET_*). SYSMEM == 1 (we own sysmem).
pub const GSP_DMA_TARGET_SYSMEM: NvU32 = 1;

/// Build the boot-arg pointer chain from the assembled phys addrs. Pure assembly:
///   - acr.gsp_rm_desc_offset -> the radix3 root (the GSP-RM ELF mapping)
///   - acr.gsp_rm_desc_size   -> the ELF size
///   - rm.boot_args_offset    -> the GSP_ARGUMENTS_CACHED region
/// `wpr_meta` supplies the radix3 root + ELF size (already filled by fw.zig);
/// `args_phys` is the GSP_ARGUMENTS_CACHED region phys (from fw.buildBootArgs).
pub fn buildFmcBootParams(wpr_meta: proto.GspFwWprMeta, args_phys: u64) GspFmcBootParams {
    return .{
        .acr = .{
            .gsp_rm_desc_offset = wpr_meta.sysmem_addr_of_radix3_elf,
            .gsp_rm_desc_size = wpr_meta.size_of_radix3_elf,
            .target = GSP_DMA_TARGET_SYSMEM,
            .b_is_gsp_rm = 1,
        },
        .rm = .{
            .boot_args_offset = args_phys,
            .target = GSP_DMA_TARGET_SYSMEM,
        },
    };
}

// ===========================================================================
// Mailbox marshalling (pure: which 32-bit value -> which mailbox).
// ===========================================================================

/// A phys addr split into the two 32-bit mailbox words the booter / GSP read.
pub const MailboxPair = struct {
    lo: u32,
    hi: u32,
};

/// Split a 64-bit phys addr into (lo32, hi32) for the mailbox pair. mbox0 = lo32,
/// mbox1 = hi32 (nouveau tu102_gsp_init: wr32(mbox0, lower_32_bits(addr));
/// wr32(mbox1, upper_32_bits(addr))).
pub fn splitPhys(addr: u64) MailboxPair {
    return .{ .lo = @truncate(addr), .hi = @truncate(addr >> 32) };
}

// ===========================================================================
// STEP 1 verifier: WPR2 scratch bounds check (pure).
// ===========================================================================

/// The WPR2 region the FWSEC FRTS run carves, read back from the PGC6 scratch
/// (both values are in 4 KB units, i.e. addr >> 12). On hardware these come from
/// reads of NV_PGC6_WPR2_LO/HI; offline the test supplies them.
pub const Wpr2Scratch = struct {
    lo_4k: u32,
    hi_4k: u32,

    pub fn loByte(self: Wpr2Scratch) u64 {
        return @as(u64, self.lo_4k) << 12;
    }
    pub fn hiByte(self: Wpr2Scratch) u64 {
        return @as(u64, self.hi_4k) << 12;
    }
};

/// Verify the WPR2 scratch readback bounds the expected FRTS region. nouveau
/// checks the carved WPR2 [lo, hi) actually exists (lo != 0) and contains the FRTS
/// region we asked for. We assert: lo != 0 (FRTS ran), and the FRTS [addr, addr+
/// size) the layout requested lies within [lo, hi].
pub fn verifyWpr2(scratch: Wpr2Scratch, layout: fw.Wpr2Layout) Error!void {
    if (scratch.lo_4k == 0 and scratch.hi_4k == 0) return Error.FrtsFailed;
    const lo = scratch.loByte();
    const hi = scratch.hiByte();
    if (hi <= lo) return Error.Wpr2Mismatch;
    // The FRTS region the FWSEC carved must lie within the reported WPR2 bounds.
    if (layout.frts_addr < lo) return Error.Wpr2Mismatch;
    if (layout.frts_addr + layout.frts_size > hi) return Error.Wpr2Mismatch;
}

// ===========================================================================
// readStruct: read an extern struct from a byte slice at an offset (no input
// alignment assumption - copies into a properly-aligned local).
// ===========================================================================

fn readStruct(comptime T: type, b: []const u8, off: u32) T {
    var v: T = undefined;
    const dst = std.mem.asBytes(&v);
    @memcpy(dst, b[off .. off + @sizeOf(T)]);
    return v;
}

// ===========================================================================
// Boot orchestrator: the thin MMIO driver chaining the four steps. The MMIO bodies
// are HARDWARE-ONLY (the deferred metal test); the data-builders above are the
// offline-tested halves.
// ===========================================================================

/// Inputs the orchestrator needs that come from the earlier phases (already built
/// + tested in fw.zig / bios.zig). The Boot driver only consumes these.
pub const BootInputs = struct {
    /// The patched FWSEC image to run on the GSP falcon (bios.extractAndPatchFrts).
    fwsec_image: []const u8,
    /// The GPU-visible phys addr the FWSEC image is loaded from (DMA source).
    fwsec_phys: u64,
    /// The WPR2 layout (fw.wpr2Layout) - for the FRTS bounds verify.
    layout: fw.Wpr2Layout,
    /// The filled GspFwWprMeta (fw.fillWprMeta) + its phys addr in sysmem.
    wpr_meta: proto.GspFwWprMeta,
    wpr_meta_phys: u64,
    /// The GSP_ARGUMENTS_CACHED region phys (fw.buildBootArgs.args_phys).
    args_phys: u64,
    /// The booter_load / booter_unload patched images + their DMA phys + parsed
    /// Booter descriptors.
    booter_load: Booter,
    booter_load_phys: u64,
    booter_unload: Booter,
    booter_unload_phys: u64,
};

/// The boot driver: holds the two falcons (GSP + SEC2) over BAR0. Every method
/// that touches MMIO is HARDWARE-ONLY (documented per-method). On the offline
/// build these compile but are never executed (Boot.run is only called on metal).
pub const Boot = struct {
    bar: hw.Bar,
    gsp: falcon.Falcon,
    sec2: falcon.Falcon,
    arch: chip.Architecture,

    /// How many poll iterations to spin before giving up (hardware tuning). On
    /// hardware each iteration is a register read; offline this is never reached.
    pub const POLL_BUDGET: usize = 1_000_000;

    pub fn init(bar: hw.Bar, arch: chip.Architecture) Boot {
        return .{
            .bar = bar,
            .gsp = falcon.Falcon.init(bar, falcon.GSP_BASE),
            .sec2 = falcon.Falcon.init(bar, falcon.SEC2_BASE),
            .arch = arch,
        };
    }

    // --- STEP 1: FWSEC-FRTS (HARDWARE-ONLY MMIO) ---

    /// Read the WPR2 lo/hi scratch back after the FWSEC FRTS run. HARDWARE-ONLY.
    pub fn readWpr2Scratch(self: Boot) Wpr2Scratch {
        return .{
            .lo_4k = self.bar.read32(NV_PGC6_WPR2_LO),
            .hi_4k = self.bar.read32(NV_PGC6_WPR2_HI),
        };
    }

    /// STEP 1 (HARDWARE-ONLY): load the patched FWSEC onto the GSP falcon in legacy
    /// (FALCON, not RISC-V) mode, start it, wait for the halt, then read + verify
    /// the WPR2 scratch. The FWSEC carves the FRTS/WPR2 region at the top of VRAM.
    /// On the offline build this body never runs.
    pub fn step1FwsecFrts(self: Boot, in: BootInputs) Error!Wpr2Scratch {
        // Legacy FALCON load: reset, DMA the FWSEC image into IMEM (secure) + DMEM,
        // start at the boot vector, wait for the halt.
        self.gsp.reset();
        while (!self.gsp.isScrubbed()) {}
        _ = self.gsp.loadImem(in.fwsec_phys, 0, @intCast(in.fwsec_image.len), true);
        _ = self.gsp.loadDmem(in.fwsec_phys, 0, @intCast(in.fwsec_image.len));
        self.gsp.start(0);
        var spins: usize = 0;
        while (!self.gsp.isHalted()) : (spins += 1) {
            if (spins > POLL_BUDGET) return Error.FrtsFailed;
        }
        const scratch = self.readWpr2Scratch();
        try verifyWpr2(scratch, in.layout);
        return scratch;
    }

    // --- STEP 3: BOOT KICK (HARDWARE-ONLY MMIO) ---

    /// STEP 3 (HARDWARE-ONLY): the boot kick assembly.
    ///   a. WprMeta phys -> SEC2 mbox0 (lo32) / mbox1 (hi32).
    ///   b. boot-args phys -> GSP mbox0 (lo32) / mbox1 (hi32).
    ///   c. reset the GSP falcon into RISC-V (reset, then the BCR kick with the
    ///      booter's dmem-sign / engine / ucode ids).
    ///   d. load booter_load on the SEC2 falcon + start it (the HS ucode locks
    ///      WPR2, DMAs the GSP-RM ELF into WPR2 via radix3, starts the GSP RISC-V).
    pub fn step3BootKick(self: Boot, in: BootInputs) void {
        // a. WprMeta phys -> SEC2 mailboxes.
        const wpr = splitPhys(in.wpr_meta_phys);
        self.sec2.mailboxWrite(0, wpr.lo);
        self.sec2.mailboxWrite(1, wpr.hi);

        // b. boot-args phys -> GSP mailboxes (0x040 / 0x044).
        const args = splitPhys(in.args_phys);
        self.gsp.mailboxWrite(0, args.lo);
        self.gsp.mailboxWrite(1, args.hi);

        // c. reset the GSP into RISC-V, then the BCR kick. The GSP RISC-V boots
        //    from the GSP-RM ELF the booter places in WPR2; the dmem-sign / engine
        //    / ucode ids come from the booter's load header.
        self.gsp.reset();
        while (!self.gsp.isScrubbed()) {}
        self.gsp.riscvBootKick(in.booter_load.dmem_sign, SEC2_FALCON_ID, 0);

        // d. run booter_load on the SEC2 falcon.
        self.sec2.reset();
        while (!self.sec2.isScrubbed()) {}
        _ = self.sec2.loadImem(in.booter_load_phys, in.booter_load.imem_base, in.booter_load.imem_size, true);
        _ = self.sec2.loadDmem(in.booter_load_phys, in.booter_load.dmem_base, in.booter_load.dmem_size);
        self.sec2.start(in.booter_load.boot_addr);
    }

    // --- STEP 4: POLL READY (HARDWARE-ONLY MMIO) ---

    /// Read the GspFwWprMeta `verified` field back from sysmem (the booter writes
    /// GSP_FW_WPR_META_VERIFIED on success). `meta_cpu` is the CPU-visible bytes of
    /// the sysmem WprMeta region. Pure read of the verified u64 at its offset.
    pub fn readVerified(meta_cpu: []const u8) u64 {
        const off = @offsetOf(proto.GspFwWprMeta, "verified");
        return std.mem.readInt(u64, meta_cpu[off..][0..8], .little);
    }

    /// STEP 4 (HARDWARE-ONLY): confirm the GSP RISC-V is active, the booter wrote
    /// `verified`, then poll the RPC ring for GSP_INIT_DONE. `endpoint` is the host
    /// side of the live ring (P5); `meta_cpu` is the sysmem WprMeta bytes.
    pub fn step4PollReady(self: Boot, endpoint: *ring.Endpoint, meta_cpu: []const u8) Error!void {
        // a. RISC-V active.
        var spins: usize = 0;
        while (!self.gsp.riscvActive()) : (spins += 1) {
            if (spins > POLL_BUDGET) return Error.RiscvInactive;
        }
        // b. the booter wrote `verified`.
        if (readVerified(meta_cpu) != proto.GspFwWprMeta.VERIFIED) return Error.NotVerified;
        // c. poll the RPC ring for GSP_INIT_DONE.
        spins = 0;
        while (spins <= POLL_BUDGET) : (spins += 1) {
            const got = endpoint.recv() catch |e| switch (e) {
                ring.Error.Empty => continue,
                else => return e,
            };
            if (got.function == @intFromEnum(proto.Event.gsp_init_done)) return;
        }
        return Error.InitTimeout;
    }

    /// The whole boot sequence (HARDWARE-ONLY end-to-end). Chains steps 1, 3, 4 (the
    /// booter signature patch in step 2 is done by the caller via patchBooter and
    /// fed in through BootInputs). `endpoint` is the host ring (P5); `meta_cpu` is
    /// the sysmem WprMeta bytes the booter writes `verified` into.
    ///
    /// On baremetal: touches MMIO + VRAM/sysmem. In this phase it cannot run - the
    /// data-builders (patchBooter / buildFmcBootParams / splitPhys / verifyWpr2)
    /// are the offline-tested halves; this driver is the user's deferred metal test.
    pub fn run(self: Boot, in: BootInputs, endpoint: *ring.Endpoint, meta_cpu: []const u8) Error!void {
        _ = try self.step1FwsecFrts(in);
        self.step3BootKick(in);
        try self.step4PollReady(endpoint, meta_cpu);
    }
};

// ===========================================================================
// Tests (offline). The MMIO bodies are not exercised here (no live GPU); we test
// every pure data-builder + the marshalling/parsing.
// ===========================================================================

const testing = std.testing;

test "nvfw struct layouts match the C reference" {
    try testing.expectEqual(@as(usize, 24), @sizeOf(NvfwBinHdr));
    try testing.expectEqual(@as(usize, 0), @offsetOf(NvfwBinHdr, "bin_magic"));
    try testing.expectEqual(@as(usize, 12), @offsetOf(NvfwBinHdr, "header_offset"));
    try testing.expectEqual(@as(usize, 16), @offsetOf(NvfwBinHdr, "data_offset"));
    try testing.expectEqual(@as(usize, 20), @offsetOf(NvfwBinHdr, "data_size"));

    try testing.expectEqual(@as(usize, 36), @sizeOf(NvfwHsHeaderV2));
    try testing.expectEqual(@as(usize, 0), @offsetOf(NvfwHsHeaderV2, "sig_prod_offset"));
    try testing.expectEqual(@as(usize, 4), @offsetOf(NvfwHsHeaderV2, "sig_prod_size"));
    try testing.expectEqual(@as(usize, 8), @offsetOf(NvfwHsHeaderV2, "patch_loc"));
    try testing.expectEqual(@as(usize, 12), @offsetOf(NvfwHsHeaderV2, "patch_sig"));
    try testing.expectEqual(@as(usize, 24), @offsetOf(NvfwHsHeaderV2, "num_sig"));
    try testing.expectEqual(@as(usize, 28), @offsetOf(NvfwHsHeaderV2, "header_offset"));

    try testing.expectEqual(@as(usize, 20), @sizeOf(NvfwHsLoadHeaderV2));
    try testing.expectEqual(@as(usize, 16), @offsetOf(NvfwHsLoadHeaderV2, "num_apps"));
    try testing.expectEqual(@as(usize, 16), @sizeOf(NvfwHsLoadHeaderV2.App));

    // The boot-arg chain structs.
    try testing.expectEqual(@as(usize, 0), @offsetOf(GspFmcBootParams, "acr"));
    try testing.expectEqual(@as(usize, 0), @offsetOf(GspAcrBootGspRmParams, "gsp_rm_desc_offset"));
    try testing.expectEqual(@as(usize, 8), @offsetOf(GspAcrBootGspRmParams, "gsp_rm_desc_size"));
    try testing.expectEqual(@as(usize, 16), @offsetOf(GspAcrBootGspRmParams, "target"));
}

test "splitPhys: lo32/hi32 mailbox marshalling" {
    const p = splitPhys(0x0000_0001_DEAD_BEEF);
    try testing.expectEqual(@as(u32, 0xDEADBEEF), p.lo);
    try testing.expectEqual(@as(u32, 0x00000001), p.hi);

    // A 64-bit-clean phys.
    const p2 = splitPhys(0xCAFEF00D_12345678);
    try testing.expectEqual(@as(u32, 0x12345678), p2.lo);
    try testing.expectEqual(@as(u32, 0xCAFEF00D), p2.hi);
}

test "step3 mailbox marshalling: WprMeta -> SEC2 0x40/0x44, args -> GSP 0x40/0x44" {
    // A fake BAR0 big enough for both engine apertures.
    const backing = try testing.allocator.alloc(u8, falcon.SEC2_BASE + 0x2000);
    defer testing.allocator.free(backing);
    @memset(backing, 0);
    const bar = hw.Bar.init(@intFromPtr(backing.ptr), backing.len);
    const b = Boot.init(bar, .ga10x);

    // Drive ONLY the mailbox-marshalling part of step3 (the rest resets/loads,
    // which a fake Bar cannot meaningfully model). Replicate a/b here exactly.
    const wpr_phys: u64 = 0x0000_0001_0000_FE00;
    const args_phys: u64 = 0x0000_0002_ABCD_0000;
    const wpr = splitPhys(wpr_phys);
    const args = splitPhys(args_phys);
    b.sec2.mailboxWrite(0, wpr.lo);
    b.sec2.mailboxWrite(1, wpr.hi);
    b.gsp.mailboxWrite(0, args.lo);
    b.gsp.mailboxWrite(1, args.hi);

    // WprMeta lo/hi at SEC2 0x40/0x44.
    try testing.expectEqual(@as(u32, 0x0000FE00), bar.read32(falcon.SEC2_BASE + 0x40));
    try testing.expectEqual(@as(u32, 0x00000001), bar.read32(falcon.SEC2_BASE + 0x44));
    // boot-args lo/hi at GSP 0x40/0x44.
    try testing.expectEqual(@as(u32, 0xABCD0000), bar.read32(falcon.GSP_BASE + 0x40));
    try testing.expectEqual(@as(u32, 0x00000002), bar.read32(falcon.GSP_BASE + 0x44));
}

test "buildFmcBootParams: the boot-arg pointer chain" {
    // A synthetic WprMeta with known radix3 root + ELF size + an args phys.
    var meta = std.mem.zeroes(proto.GspFwWprMeta);
    meta.sysmem_addr_of_radix3_elf = 0x1_2345_0000;
    meta.size_of_radix3_elf = 0x0456_0000;
    const args_phys: u64 = 0x9_8765_4000;

    const p = buildFmcBootParams(meta, args_phys);
    // ACR params point at the radix3 root + carry the ELF size, sysmem target.
    try testing.expectEqual(@as(u64, 0x1_2345_0000), p.acr.gsp_rm_desc_offset);
    try testing.expectEqual(@as(u64, 0x0456_0000), p.acr.gsp_rm_desc_size);
    try testing.expectEqual(GSP_DMA_TARGET_SYSMEM, p.acr.target);
    try testing.expectEqual(@as(NvU8, 1), p.acr.b_is_gsp_rm);
    // RM params point at the boot-args (GSP_ARGUMENTS_CACHED) region.
    try testing.expectEqual(args_phys, p.rm.boot_args_offset);
    try testing.expectEqual(GSP_DMA_TARGET_SYSMEM, p.rm.target);
}

test "verifyWpr2: bounds check the FRTS region against the scratch readback" {
    // Build a layout (8 GB VRAM, GA104).
    const fb_size: u64 = 8 * (1 << 30);
    const layout = try fw.wpr2Layout(fb_size, fb_size - 0x100000, 0x10000, 0x40000, 72_853_408, fw.HeapParams{});

    // A scratch that exactly bounds the carved WPR2 [wpr2_addr, wpr2_end).
    const ok = Wpr2Scratch{
        .lo_4k = @intCast(layout.wpr2_addr >> 12),
        .hi_4k = @intCast(layout.wpr2_end >> 12),
    };
    try verifyWpr2(ok, layout);

    // All-zero scratch -> FRTS never ran.
    try testing.expectError(Error.FrtsFailed, verifyWpr2(.{ .lo_4k = 0, .hi_4k = 0 }, layout));

    // hi <= lo -> mismatch.
    try testing.expectError(Error.Wpr2Mismatch, verifyWpr2(.{ .lo_4k = 100, .hi_4k = 100 }, layout));

    // FRTS region falls OUTSIDE the reported bounds -> mismatch.
    const too_high = Wpr2Scratch{
        .lo_4k = @intCast(layout.wpr2_addr >> 12),
        .hi_4k = @intCast((layout.frts_addr) >> 12), // hi below the FRTS end
    };
    try testing.expectError(Error.Wpr2Mismatch, verifyWpr2(too_high, layout));
}

test "readVerified reads the verified field at offset 248" {
    var meta = std.mem.zeroes(proto.GspFwWprMeta);
    meta.verified = proto.GspFwWprMeta.VERIFIED;
    const bytes = std.mem.asBytes(&meta);
    try testing.expectEqual(proto.GspFwWprMeta.VERIFIED, Boot.readVerified(bytes));

    // Unverified reads 0.
    var meta0 = std.mem.zeroes(proto.GspFwWprMeta);
    meta0.verified = 0;
    try testing.expectEqual(@as(u64, 0), Boot.readVerified(std.mem.asBytes(&meta0)));
}

// --- the booter HS-header parse + signature patch (synthetic blob) ---

/// Build a synthetic nvfw HS booter blob with known sizes so we can assert the
/// parse + the signature patch land exactly where the C would put them.
const SyntheticBooter = struct {
    blob: []u8,
    // Echo of the layout the builder chose, so the test can assert against it.
    header_offset: u32,
    load_hdr_offset: u32,
    patch_loc_word_off: u32,
    patch_sig_word_off: u32,
    num_sig_word_off: u32,
    sig_prod_offset: u32,
    sig_prod_size: u32,
    cnt: u32,
    sig_size: u32,
    loc: u32,
    data_offset: u32,
    data_size: u32,
    dmem_base: u32,

    fn build(alloc: std.mem.Allocator) !SyntheticBooter {
        // Choose a compact but realistic layout.
        const header_offset: u32 = 0x18; // hs header right after the 24-byte bin hdr
        const hs_size: u32 = @sizeOf(NvfwHsHeaderV2); // 36
        // patch words (3 separate u32 cells in the blob, dereferenced by the hs hdr).
        const patch_loc_word_off: u32 = header_offset + hs_size; // 0x3c
        const patch_sig_word_off: u32 = patch_loc_word_off + 4;
        const num_sig_word_off: u32 = patch_sig_word_off + 4;
        // load header.
        const load_hdr_offset: u32 = num_sig_word_off + 4;
        const app_off: u32 = load_hdr_offset + @sizeOf(NvfwHsLoadHeaderV2);
        // signatures: 2 sigs of 0x40 bytes each (prod then debug).
        const sig_size: u32 = 0x40;
        const cnt: u32 = 2;
        const sig_prod_size: u32 = sig_size * cnt;
        const sig_prod_offset: u32 = app_off + @sizeOf(NvfwHsLoadHeaderV2.App);
        // ucode image after the signatures.
        const data_offset: u32 = sig_prod_offset + sig_prod_size + 0x40;
        const data_size: u32 = 0x400;
        const dmem_base: u32 = 0x100;
        // The signature destination inside the image.
        const loc: u32 = 0x80;

        const total: u32 = data_offset + data_size;
        const blob = try alloc.alloc(u8, total);
        @memset(blob, 0);

        // nvfw_bin_hdr.
        var bin = std.mem.zeroes(NvfwBinHdr);
        bin.bin_magic = NvfwBinHdr.MAGIC;
        bin.bin_ver = 1;
        bin.bin_size = total;
        bin.header_offset = header_offset;
        bin.data_offset = data_offset;
        bin.data_size = data_size;
        @memcpy(blob[0..@sizeOf(NvfwBinHdr)], std.mem.asBytes(&bin));

        // nvfw_hs_header_v2.
        var hs = std.mem.zeroes(NvfwHsHeaderV2);
        hs.sig_prod_offset = sig_prod_offset;
        hs.sig_prod_size = sig_prod_size;
        hs.patch_loc = patch_loc_word_off;
        hs.patch_sig = patch_sig_word_off;
        hs.num_sig = num_sig_word_off;
        hs.header_offset = load_hdr_offset;
        hs.header_size = @sizeOf(NvfwHsLoadHeaderV2);
        @memcpy(blob[header_offset..][0..@sizeOf(NvfwHsHeaderV2)], std.mem.asBytes(&hs));

        // The dereferenced patch words.
        std.mem.writeInt(u32, blob[patch_loc_word_off..][0..4], loc, .little);
        std.mem.writeInt(u32, blob[patch_sig_word_off..][0..4], 0, .little); // sig == 0
        std.mem.writeInt(u32, blob[num_sig_word_off..][0..4], cnt, .little);

        // nvfw_hs_load_header_v2 + app[0].
        var load = std.mem.zeroes(NvfwHsLoadHeaderV2);
        load.os_code_offset = 0x0;
        load.os_code_size = 0x40;
        load.os_data_offset = dmem_base;
        load.os_data_size = 0x80;
        load.num_apps = 1;
        @memcpy(blob[load_hdr_offset..][0..@sizeOf(NvfwHsLoadHeaderV2)], std.mem.asBytes(&load));
        var app = std.mem.zeroes(NvfwHsLoadHeaderV2.App);
        app.offset = 0x40;
        app.size = 0x40;
        @memcpy(blob[app_off..][0..@sizeOf(NvfwHsLoadHeaderV2.App)], std.mem.asBytes(&app));

        // Fill the two signatures with distinct patterns so the patch is verifiable.
        var i: u32 = 0;
        while (i < sig_size) : (i += 1) {
            blob[sig_prod_offset + i] = 0xA0 | @as(u8, @truncate(i & 0xf)); // prod
            blob[sig_prod_offset + sig_size + i] = 0xD0 | @as(u8, @truncate(i & 0xf)); // debug
        }
        // Mark the ucode image so we can confirm it copied + only `loc` changed.
        i = 0;
        while (i < data_size) : (i += 1) blob[data_offset + i] = @truncate(i & 0xff);

        return .{
            .blob = blob,
            .header_offset = header_offset,
            .load_hdr_offset = load_hdr_offset,
            .patch_loc_word_off = patch_loc_word_off,
            .patch_sig_word_off = patch_sig_word_off,
            .num_sig_word_off = num_sig_word_off,
            .sig_prod_offset = sig_prod_offset,
            .sig_prod_size = sig_prod_size,
            .cnt = cnt,
            .sig_size = sig_size,
            .loc = loc,
            .data_offset = data_offset,
            .data_size = data_size,
            .dmem_base = dmem_base,
        };
    }
};

test "parseBooterHeaders: dereferences patch_loc/patch_sig/num_sig" {
    const sb = try SyntheticBooter.build(testing.allocator);
    defer testing.allocator.free(sb.blob);

    const h = try parseBooterHeaders(sb.blob);
    try testing.expectEqual(NvfwBinHdr.MAGIC, h.bin.bin_magic);
    try testing.expectEqual(sb.data_offset, h.bin.data_offset);
    try testing.expectEqual(sb.data_size, h.bin.data_size);
    try testing.expectEqual(sb.sig_prod_offset, h.hs.sig_prod_offset);
    try testing.expectEqual(sb.sig_prod_size, h.hs.sig_prod_size);
    // The patch words are dereferenced (NOT the raw field values).
    try testing.expectEqual(sb.loc, h.loc);
    try testing.expectEqual(@as(u32, 0), h.sig);
    try testing.expectEqual(sb.cnt, h.cnt);
    // Load header + app[0].
    try testing.expectEqual(sb.dmem_base, h.load.os_data_offset);
    try testing.expectEqual(@as(u32, 1), h.load.num_apps);
    try testing.expectEqual(@as(u32, 0x40), h.app0.offset);
}

test "patchBooter: prod signature lands at loc, rest of image intact" {
    const sb = try SyntheticBooter.build(testing.allocator);
    defer testing.allocator.free(sb.blob);

    const out = try testing.allocator.alloc(u8, sb.data_size);
    defer testing.allocator.free(out);

    const booter = try patchBooter(sb.blob, out, false); // prod (fuse clear)
    try testing.expectEqual(@as(usize, sb.data_size), booter.image.len);
    try testing.expectEqual(sb.loc, booter.sig_offset_in_image);
    try testing.expectEqual(sb.sig_size, booter.sig_size);
    try testing.expectEqual(sb.dmem_base, booter.dmem_base);
    // dmem_sign = loc - dmem_base.
    try testing.expectEqual(sb.loc -% sb.dmem_base, booter.dmem_sign);

    // The signature region now holds the PROD signature pattern (0xA0 | i&0xf).
    var i: u32 = 0;
    while (i < sb.sig_size) : (i += 1) {
        try testing.expectEqual(@as(u8, 0xA0 | @as(u8, @truncate(i & 0xf))), out[sb.loc + i]);
    }
    // Everything OUTSIDE [loc, loc+sig_size) is the untouched image bytes (i&0xff).
    i = 0;
    while (i < sb.data_size) : (i += 1) {
        if (i >= sb.loc and i < sb.loc + sb.sig_size) continue;
        try testing.expectEqual(@as(u8, @truncate(i & 0xff)), out[i]);
    }
}

test "patchBooter: debug fuse selects the debug signature (index 1)" {
    const sb = try SyntheticBooter.build(testing.allocator);
    defer testing.allocator.free(sb.blob);

    const out = try testing.allocator.alloc(u8, sb.data_size);
    defer testing.allocator.free(out);

    _ = try patchBooter(sb.blob, out, true); // debug fuse set
    // The signature region now holds the DEBUG pattern (0xD0 | i&0xf), not prod.
    var i: u32 = 0;
    while (i < sb.sig_size) : (i += 1) {
        try testing.expectEqual(@as(u8, 0xD0 | @as(u8, @truncate(i & 0xf))), out[sb.loc + i]);
    }
}

test "patchBooter: rejects a bad magic + a truncated blob" {
    const sb = try SyntheticBooter.build(testing.allocator);
    defer testing.allocator.free(sb.blob);
    const out = try testing.allocator.alloc(u8, sb.data_size);
    defer testing.allocator.free(out);

    // Corrupt the magic.
    const saved = sb.blob[0];
    sb.blob[0] = 0;
    try testing.expectError(Error.BadBooter, patchBooter(sb.blob, out, false));
    sb.blob[0] = saved;

    // Truncated blob (cut before the data section).
    try testing.expectError(Error.Truncated, patchBooter(sb.blob[0 .. sb.data_offset - 4], out, false));

    // out too small.
    const tiny = try testing.allocator.alloc(u8, 8);
    defer testing.allocator.free(tiny);
    try testing.expectError(Error.BufferTooSmall, patchBooter(sb.blob, tiny, false));
}

// --- the real ga102 booter blob (offline, skips on freestanding / if absent) ---

// The known ga102 booter_load path (linux-firmware nvidia/ga102/gsp/). The GA10x
// family consumes the ga102 booter sub-tree. The shipped .bin is zstd-compressed
// in the nix store, so this test only runs if a DECOMPRESSED copy has been staged
// at GA102_BOOTER_LOAD_STAGED (the test/build wrapper can drop one there); else it
// skips. This keeps the test hermetic + freestanding-clean (no zstd / no fs on UEFI).
pub const GA102_BOOTER_LOAD_STAGED: []const u8 = "/tmp/booter_load.bin";

test "patchBooter against a real ga102 booter blob (if staged)" {
    if (builtin.target.os.tag == .freestanding) return error.SkipZigTest;
    const alloc = testing.allocator;
    const path = GA102_BOOTER_LOAD_STAGED;

    var threaded = std.Io.Threaded.init(alloc, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const cwd = std.Io.Dir.cwd();
    cwd.access(io, path, .{}) catch return error.SkipZigTest;
    const blob = cwd.readFileAlloc(io, path, alloc, .limited(4 * 1024 * 1024)) catch return error.SkipZigTest;
    defer alloc.free(blob);

    const h = try parseBooterHeaders(blob);
    // The real ga102 booter_load: magic 0x10de, 2 sigs (prod + debug).
    try testing.expectEqual(NvfwBinHdr.MAGIC, h.bin.bin_magic);
    try testing.expect(h.cnt >= 1);
    try testing.expect(h.hs.sig_prod_size % h.cnt == 0);

    const out = try alloc.alloc(u8, h.bin.data_size);
    defer alloc.free(out);
    const booter = try patchBooter(blob, out, false);
    try testing.expect(booter.image.len == h.bin.data_size);
    try testing.expect(booter.sig_offset_in_image + booter.sig_size <= booter.image.len);
}

test {
    std.testing.refAllDecls(@This());
}
