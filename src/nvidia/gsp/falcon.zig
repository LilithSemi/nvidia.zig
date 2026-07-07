//! FALCON / NVRISCV microcontroller driver (plan phase P1): the substrate the
//! GSP boot (P4) drives. NVIDIA GPUs carry several FALCON microcontrollers (PMU,
//! SEC2, GSP). On Ampere+ (GA10x) the GSP engine runs a RISC-V core (NVRISCV)
//! that the FALCON shell boots through a BCR (Boot Config Register) kick.
//!
//! This is PURE register logic over a hw.Bar (BAR0) plus a per-engine sub-aperture
//! base offset. Each operation is a register-SEQUENCE BUILDER: it computes the
//! writes, which on hardware hit the GPU and in the unit tests hit a fake Bar.
//! No std.os - it compiles freestanding/UEFI.
//!
//! Reference (nouveau, drivers/gpu/drm/nouveau/nvkm/falcon/, kernel master):
//!   - falcon/gm200.c  : the FALCON-shell primitives (boot, disable/enable, PIO
//!                       IMEM/DMEM, mem-scrubbing wait, bind).
//!   - falcon/ga102.c  : the NVRISCV additions (DMA microcode load, the BCR boot
//!                       kick, riscv_active poll, core select).
//!   - falcon/base.c   : nvkm_falcon_dma_wr (the 256-byte DMA-load loop) and
//!                       nvkm_falcon_reset (disable -> enable).
//!   - core/falcon.h   : the IMEM/DMEM/EMEM enum (no register offsets live here -
//!                       the offsets are the literals in gm200.c/ga102.c).
//!
//! Important addressing fact (nouveau nvkm_falcon_rd32/wr32 in falcon/priv.h):
//! every falcon access adds the falcon's BAR0 base `addr` to the register offset.
//! The NVRISCV/BCR registers add a further `addr2` sub-offset (0x1000 on ga102).
//! So a "BCR_CTRL" write goes to `bar[ addr + addr2 + 0x180 ]`. We fold all three
//! (bar.base + base + RISCV_BASE + reg) inside the accessors below.

const std = @import("std");
const hw = @import("../hw/mmio.zig");

// ===========================================================================
// Per-engine BAR0 sub-aperture base offsets (GA10x / Ampere).
// These are hardcoded per-chip in nouveau; CONFIRMED against the nouveau master:
//   - GSP  falcon base 0x110000  : nvkm/subdev/gsp/base.c
//        `nvkm_falcon_ctor(gsp->func->flcn, ..., 0x110000, &gsp->falcon);`
//   - SEC2 falcon base 0x840000  : nvkm/engine/sec2/ga102.c  `const u32 addr = 0x840000;`
//   - both use ga102_*/gm200_* funcs with `.addr2 = 0x1000` (the NVRISCV region).
// ===========================================================================

/// GSP FALCON BAR0 sub-aperture base on GA10x. CONFIRMED: nouveau gsp/base.c.
pub const GSP_BASE: u32 = 0x110000;
/// SEC2 FALCON BAR0 sub-aperture base on GA10x (the booter engine). CONFIRMED:
/// nouveau engine/sec2/ga102.c `const u32 addr = 0x840000;`.
pub const SEC2_BASE: u32 = 0x840000;
/// PMU FALCON BAR0 sub-aperture base. FLAG: NOT independently re-confirmed from a
/// nouveau ga102 PMU ctor in this pass (the boot path needs only GSP + SEC2). The
/// historical PMU falcon base across Maxwell..Ampere is 0x10a000; treat as
/// UNCONFIRMED for GA10x until grounded in nvkm/subdev/pmu before use.
pub const PMU_BASE: u32 = 0x10a000;

/// The NVRISCV / BCR sub-region offset within a falcon aperture (nouveau
/// `.addr2 = 0x1000` for ga102_gsp_flcn and ga102_sec2_flcn). The RISC-V boot
/// config + signature + active-status registers live at base + addr2 + reg.
pub const RISCV_BASE: u32 = 0x1000;

// ===========================================================================
// FALCON register offsets (relative to the engine base). Every literal here is
// the exact offset used by nouveau gm200.c / ga102.c (cited inline).
// ===========================================================================

/// IRQSCLR / interrupt clear (gm200_flcn_fw_boot writes irqsclr here; also the
/// disable path writes 0xffffffff). nouveau offset 0x004.
pub const IRQSCLR: u32 = 0x004;
/// IRQSTAT (interrupt status). nouveau 0x008.
pub const IRQSTAT: u32 = 0x008;
/// IRQMASK / IRQMSET region. nouveau 0x010 (IRQMSET) / 0x014 (IRQMCLR). The
/// disable path writes 0x014 = 0xffffffff (mask all). We expose 0x014 as IRQMCLR.
pub const IRQMSET: u32 = 0x010;
pub const IRQMCLR: u32 = 0x014;
/// MAILBOX0 (PMBOX0). nouveau gm200_flcn_fw_boot writes/reads 0x040.
pub const MAILBOX0: u32 = 0x040;
/// MAILBOX1 (PMBOX1). nouveau 0x044.
pub const MAILBOX1: u32 = 0x044;
/// IDLESTATE / engine control bit (disable masks 0x048 bits 0..1). nouveau 0x048.
pub const IDLESTATE: u32 = 0x048;
/// ENABLE / scratch-restore register written in gm200_flcn_enable. nouveau 0x084.
pub const ENABLE: u32 = 0x084;
/// CPUCTL. start = write 0x2 (STARTCPU); HALTED = bit 0x10. nouveau gm200 0x100.
pub const CPUCTL: u32 = 0x100;
/// BOOTVEC. nouveau gm200_flcn_fw_boot 0x104.
pub const BOOTVEC: u32 = 0x104;
/// CPUCTL2 / scrub + reset status (gm200 mem-scrubbing waits on 0x10c bit 0x6).
/// ga102_flcn_fw_load writes 0x10c = 0 before the DMA load. nouveau 0x10c.
pub const CPUCTL2: u32 = 0x10c;
/// DMATRFBASE (DMA transfer base, dma_addr >> 8). nouveau ga102_flcn_dma_init 0x110.
pub const DMATRFBASE: u32 = 0x110;
/// DMATRFMOFFS (mem offset within IMEM/DMEM). nouveau ga102_flcn_dma_xfer 0x114.
pub const DMATRFMOFFS: u32 = 0x114;
/// DMATRFCMD (kick + status; done = bit 0x2). nouveau ga102_flcn_dma_xfer 0x118.
pub const DMATRFCMD: u32 = 0x118;
/// DMATRFFBOFFS (FB/source byte offset). nouveau ga102_flcn_dma_xfer 0x11c.
pub const DMATRFFBOFFS: u32 = 0x11c;
/// DMATRFBASE1 (high bits of the DMA base; ga102 writes 0). nouveau 0x128.
pub const DMATRFBASE1: u32 = 0x128;
/// IMEMC[port] init (auto-inc + secure + imem_base). nouveau gm200 0x180 + port*0x10.
pub const IMEMC: u32 = 0x180;
/// IMEMD[port] data port. nouveau 0x184 + port*0x10.
pub const IMEMD: u32 = 0x184;
/// IMEMT[port] tag port. nouveau 0x188 + port*0x10.
pub const IMEMT: u32 = 0x188;
/// DMEMC[port] init (auto-inc + dmem_base). nouveau gm200 0x1c0 + port*8.
pub const DMEMC: u32 = 0x1c0;
/// DMEMD[port] data port. nouveau 0x1c4 + port*8.
pub const DMEMD: u32 = 0x1c4;
/// FBIF transcfg/regioncfg select base. ga102_flcn_fw_load masks 0x600 / 0x624.
/// 0x600 = FBIF_TRANSCFG[0], 0x624 = FBIF_CTL. nouveau ga102_flcn_fw_load.
pub const FBIF_TRANSCFG: u32 = 0x600;
pub const FBIF_CTL: u32 = 0x624;
/// HWCFG (engine sizing - IMEM/DMEM size in 256-byte blocks). nouveau reads
/// HWCFG at 0x108 in nvkm_falcon_oneinit. Exposed for callers that size IMEM/DMEM.
pub const HWCFG: u32 = 0x108;

// --- NVRISCV / BCR registers (relative to base + RISCV_BASE) ---
// All from nouveau ga102.c (ga102_flcn_fw_boot, riscv_active, select).

/// BCR_CTRL: write 0x1 to validate the boot config (core_select=RISCV + valid).
/// nouveau ga102_flcn_fw_boot: wr32(addr2 + 0x180, 1) - LAST in the BCR sequence.
pub const BCR_CTRL: u32 = 0x180;
/// BCR_DMACFG / ucode id. nouveau ga102_flcn_fw_boot: wr32(addr2 + 0x198, ucode_id).
pub const BCR_UCODE_ID: u32 = 0x198;
/// BCR engine id. nouveau ga102_flcn_fw_boot: wr32(addr2 + 0x19c, engine_id).
pub const BCR_ENGINE_ID: u32 = 0x19c;
/// BR (boot-rom) signature / dmem_sign. nouveau ga102_flcn_fw_boot:
/// wr32(addr2 + 0x210, dmem_sign) - FIRST in the BCR sequence.
pub const BCR_DMEM_SIGN: u32 = 0x210;
/// RISCV active status (bit 0x80 set => RISC-V running). nouveau
/// ga102_flcn_riscv_active: rd32(addr2 + 0x388) & 0x80.
pub const RISCV_CPUCTL: u32 = 0x388;
/// Core select (FALCON vs RISCV). nouveau ga102_flcn_select operates on
/// addr2 + 0x668: bit 0x10 = currently FALCON, write 0 then poll bit 0x1.
pub const RISCV_BCR_DMACFG_SEC: u32 = 0x668;

// ===========================================================================
// Bit values (nouveau, cited).
// ===========================================================================

/// CPUCTL STARTCPU: gm200_flcn_fw_boot writes 0x2 to CPUCTL to start the core.
pub const CPUCTL_STARTCPU: u32 = 0x00000002;
/// CPUCTL HALTED: gm200 polls CPUCTL bit 0x10 for the halt.
pub const CPUCTL_HALTED: u32 = 0x00000010;
/// RISCV active bit (riscv_active poll, addr2 + 0x388).
pub const RISCV_ACTIVE: u32 = 0x00000080;
/// DMATRFCMD done bit (ga102_flcn_dma_done: 0x118 & 0x2).
pub const DMATRFCMD_DONE: u32 = 0x00000002;
/// DMATRFCMD IMEM target bit (ga102_flcn_dma_init: cmd |= 0x10 for IMEM).
pub const DMATRFCMD_IMEM: u32 = 0x00000010;
/// DMATRFCMD secure bit (ga102_flcn_dma_init: cmd |= 0x4 if sec).
pub const DMATRFCMD_SEC: u32 = 0x00000004;
/// IMEMC secure bit (gm200_flcn_pio_imem_wr_init: BIT(28)).
pub const IMEMC_SECURE: u32 = 1 << 28;
/// IMEMC/DMEMC auto-increment-write bit (gm200 PIO init: BIT(24)).
pub const MEMC_AINCW: u32 = 1 << 24;

/// The fixed DMA transfer chunk size nouveau uses (base.c nvkm_falcon_dma_wr:
/// `const int dmalen = 256;`). The DMATRFCMD size field encodes log2(len)-2.
pub const DMA_CHUNK: u32 = 256;

/// Which memory the load targets (matches nouveau core/falcon.h enum nvkm_falcon_mem).
pub const Mem = enum { imem, dmem };

/// A single MMIO write the sequence builders emit (used by tests + a record mode).
pub const Write = struct { off: u32, val: u32 };

// ===========================================================================
// Falcon: one microcontroller instance at a BAR0 sub-aperture.
// ===========================================================================

pub const Falcon = struct {
    bar: hw.Bar,
    /// The engine BAR0 base offset (GSP_BASE / SEC2_BASE / PMU_BASE).
    base: u32,
    /// The NVRISCV/BCR sub-region offset (RISCV_BASE = 0x1000 on ga102). Kept a
    /// field so a pure-FALCON engine (no RISC-V) could set it to 0.
    riscv_base: u32 = RISCV_BASE,

    pub fn init(bar: hw.Bar, base: u32) Falcon {
        return .{ .bar = bar, .base = base };
    }

    pub fn initRiscv(bar: hw.Bar, base: u32, riscv_base: u32) Falcon {
        return .{ .bar = bar, .base = base, .riscv_base = riscv_base };
    }

    // --- low-level accessors: fold bar.base + engine base + reg ---

    /// Read a falcon register (offset relative to the engine base).
    pub inline fn rd32(self: Falcon, reg: u32) u32 {
        return self.bar.read32(self.base + reg);
    }

    /// Write a falcon register (offset relative to the engine base).
    pub inline fn wr32(self: Falcon, reg: u32, val: u32) void {
        self.bar.write32(self.base + reg, val);
    }

    /// Read-modify-write a falcon register (nvkm_falcon_mask).
    pub inline fn mask(self: Falcon, reg: u32, m: u32, val: u32) void {
        self.bar.mask32(self.base + reg, m, val);
    }

    /// Read an NVRISCV/BCR register (offset relative to base + riscv_base).
    pub inline fn rdRiscv(self: Falcon, reg: u32) u32 {
        return self.bar.read32(self.base + self.riscv_base + reg);
    }

    /// Write an NVRISCV/BCR register (offset relative to base + riscv_base).
    pub inline fn wrRiscv(self: Falcon, reg: u32, val: u32) void {
        self.bar.write32(self.base + self.riscv_base + reg, val);
    }

    // --- mailboxes ---

    /// Write mailbox idx (0 or 1) -> MAILBOX0/MAILBOX1.
    pub fn mailboxWrite(self: Falcon, idx: u1, val: u32) void {
        self.wr32(if (idx == 0) MAILBOX0 else MAILBOX1, val);
    }

    /// Read mailbox idx (0 or 1).
    pub fn mailboxRead(self: Falcon, idx: u1) u32 {
        return self.rd32(if (idx == 0) MAILBOX0 else MAILBOX1);
    }

    // --- reset ---

    /// The FALCON reset sequence (nouveau nvkm_falcon_reset = disable -> enable).
    /// disable (gm200_flcn_disable): clear IDLESTATE[0..1] (0x048), write
    ///   IRQMCLR(0x014) = 0xffffffff (mask all interrupts).
    /// enable (gm200_flcn_enable): wait mem-scrubbing (poll CPUCTL2 0x10c bit 0x6
    ///   clear - a READ poll, not emitted as a write), then write ENABLE(0x084).
    /// We emit the deterministic WRITES in order; the scrub poll is a read loop the
    /// caller drives via waitScrubbed(). The 0x084 value on hardware is rd32(0x0)
    /// (the boot-0 chip id echo); offline we cannot read device 0x0 through the
    /// engine aperture, so we write 0 and document the hardware value.
    pub fn reset(self: Falcon) void {
        // disable
        self.mask(IDLESTATE, 0x00000003, 0x00000000);
        self.wr32(IRQMCLR, 0xffffffff);
        // enable: the scrub wait is a poll (waitScrubbed); then the enable write.
        // nouveau writes nvkm_rd32(device, 0x0) here (the global boot-0). Offline
        // we have no global 0x0 through this aperture; emit 0. On hardware, pass
        // the device boot-0 value via enableWrite() if a non-zero echo is needed.
        self.wr32(ENABLE, 0x00000000);
    }

    /// The reset write SEQUENCE as records (for tests / record-mode). Mirrors reset().
    pub fn resetSeq(self: Falcon, out: []Write) usize {
        const b = self.base;
        var n: usize = 0;
        // mask(IDLESTATE,0x3,0) is RMW; record it as the intended cleared write.
        out[n] = .{ .off = b + IDLESTATE, .val = 0 };
        n += 1;
        out[n] = .{ .off = b + IRQMCLR, .val = 0xffffffff };
        n += 1;
        out[n] = .{ .off = b + ENABLE, .val = 0 };
        n += 1;
        return n;
    }

    /// Poll the mem-scrubbing-done status (gm200_flcn_reset_wait_mem_scrubbing:
    /// CPUCTL2 0x10c bit 0x6 must be clear). Returns true when scrubbing is done.
    pub fn isScrubbed(self: Falcon) bool {
        return (self.rd32(CPUCTL2) & 0x00000006) == 0;
    }

    // --- microcode load ---

    /// DMA-load microcode into IMEM or DMEM (ga102 path: ga102_flcn_fw_load +
    /// nvkm_falcon_dma_wr). `dma_addr` is the GPU-visible source phys address of
    /// the firmware image; `mem_off` is the destination byte offset within
    /// IMEM/DMEM; `len` is the byte length; `secure` tags the IMEM block secure.
    ///
    /// Register sequence (per nouveau):
    ///   init  : DMATRFBASE(0x110) = dma_addr >> 8 ; DMATRFBASE1(0x128) = 0
    ///   per 256-byte chunk:
    ///           DMATRFMOFFS(0x114) = mem_off + i*256
    ///           DMATRFFBOFFS(0x11c) = i*256           (offset from dma_start)
    ///           DMATRFCMD(0x118)   = ((log2(256)-2)<<8) | (IMEM?0x10) | (sec?0x4)
    ///                              = 0x600 | ...       then poll done (bit 0x2)
    /// Returns the number of chunks issued. A trailing partial (< 256) chunk is
    /// NOT emitted (nouveau's loop is `while (len >= dmalen)`; the firmware images
    /// are 256-aligned, matching nouveau's assumption).
    pub fn loadDma(self: Falcon, mem: Mem, dma_addr: u64, mem_off: u32, len: u32, secure: bool) usize {
        self.wr32(DMATRFBASE, @truncate(dma_addr >> 8));
        self.wr32(DMATRFBASE1, 0);

        const cmd = dmaCmd(mem, secure);
        var i: u32 = 0;
        while ((i + 1) * DMA_CHUNK <= len) : (i += 1) {
            const off = i * DMA_CHUNK;
            self.wr32(DMATRFMOFFS, mem_off + off);
            self.wr32(DMATRFFBOFFS, off);
            self.wr32(DMATRFCMD, cmd);
            // hardware: poll DMATRFCMD bit 0x2 (done) here. Offline: no read loop.
        }
        return i;
    }

    /// Emit the full ga102_flcn_fw_load PREP writes (FBIF + 0x10c) then the DMA
    /// load for IMEM (secure) and DMEM. Returns the write count for tests.
    pub fn loadImem(self: Falcon, dma_addr: u64, mem_off: u32, len: u32, secure: bool) usize {
        // ga102_flcn_fw_load prep, emitted once before IMEM:
        //   mask(FBIF_CTL 0x624, 0x80, 0x80)
        //   wr32(0x10c, 0)
        //   mask(FBIF_TRANSCFG 0x600, 0x10007, (0<<16)|(1<<2)|1)
        self.mask(FBIF_CTL, 0x00000080, 0x00000080);
        self.wr32(CPUCTL2, 0x00000000);
        self.mask(FBIF_TRANSCFG, 0x00010007, (1 << 2) | 1);
        return self.loadDma(.imem, dma_addr, mem_off, len, secure);
    }

    /// DMEM DMA load (no FBIF prep; the IMEM prep above covers a paired load).
    pub fn loadDmem(self: Falcon, dma_addr: u64, mem_off: u32, len: u32) usize {
        return self.loadDma(.dmem, dma_addr, mem_off, len, false);
    }

    /// Compute the DMATRFCMD value for a 256-byte chunk: (log2(256)-2)<<8 plus the
    /// IMEM and secure bits. log2(256)=8 -> 6<<8 = 0x600.
    pub fn dmaCmd(mem: Mem, secure: bool) u32 {
        const size_field: u32 = (@as(u32, @ctz(DMA_CHUNK)) - 2) << 8;
        var cmd: u32 = size_field;
        if (mem == .imem) cmd |= DMATRFCMD_IMEM;
        if (secure) cmd |= DMATRFCMD_SEC;
        return cmd;
    }

    /// PIO IMEM load fallback (gm200_flcn_pio_imem_wr): no DMA, push words through
    /// the IMEMC/IMEMD/IMEMT ports. `tag` seeds the IMEM block tag; `data` length
    /// must be a multiple of 4. Provided as a simpler fallback path.
    pub fn loadImemPio(self: Falcon, data: []const u8, imem_base: u32, tag: u32, secure: bool) void {
        std.debug.assert(data.len % 4 == 0);
        // init: IMEMC = (secure?BIT28) | AINCW(BIT24) | imem_base
        self.wr32(IMEMC, (if (secure) IMEMC_SECURE else 0) | MEMC_AINCW | imem_base);
        // tag once, then stream 32-bit words.
        self.wr32(IMEMT, tag);
        var i: usize = 0;
        while (i < data.len) : (i += 4) {
            self.wr32(IMEMD, std.mem.readInt(u32, data[i..][0..4], .little));
        }
    }

    /// PIO DMEM load fallback (gm200_flcn_pio_dmem_wr): DMEMC init (AINCW + base),
    /// then stream words through DMEMD.
    pub fn loadDmemPio(self: Falcon, data: []const u8, dmem_base: u32) void {
        std.debug.assert(data.len % 4 == 0);
        self.wr32(DMEMC, MEMC_AINCW | dmem_base);
        var i: usize = 0;
        while (i < data.len) : (i += 4) {
            self.wr32(DMEMD, std.mem.readInt(u32, data[i..][0..4], .little));
        }
    }

    // --- start / halt ---

    /// Start the FALCON core (gm200_flcn_fw_boot core): BOOTVEC then CPUCTL
    /// STARTCPU. Mailboxes, if any, must be primed BEFORE this call.
    pub fn start(self: Falcon, bootvec: u32) void {
        self.wr32(BOOTVEC, bootvec);
        self.wr32(CPUCTL, CPUCTL_STARTCPU);
    }

    /// True when the FALCON has halted (CPUCTL bit 0x10, gm200_flcn_fw_boot poll).
    pub fn isHalted(self: Falcon) bool {
        return (self.rd32(CPUCTL) & CPUCTL_HALTED) != 0;
    }

    // --- NVRISCV (Ampere/ga102) ---

    /// The NVRISCV BCR boot kick (nouveau ga102_flcn_fw_boot). EXACT order, READ
    /// from nouveau (not inferred):
    ///   1. BR sig      : wrRiscv(0x210, dmem_sign)
    ///   2. engine id   : wrRiscv(0x19c, engine_id)
    ///   3. ucode id    : wrRiscv(0x198, ucode_id)
    ///   4. BCR_CTRL    : wrRiscv(0x180, 1)            <- validates the boot config
    /// ga102_flcn_fw_boot then tail-calls gm200_flcn_fw_boot, which primes the
    /// mailboxes + BOOTVEC + CPUCTL STARTCPU. We expose that tail as a separate
    /// `start()` so callers can prime mailboxes between the BCR kick and the CPU
    /// start exactly as the boot phase (P4) needs.
    ///
    /// FLAG: the FOUR BCR writes + their order are read line-for-line from
    /// ga102_flcn_fw_boot. What is INFERRED (not in that function) is whether a
    /// `select()` (core-select to RISCV via 0x668) must precede this on a cold
    /// engine; nouveau runs select() inside disable/enable (the reset), so a
    /// reset() before riscvBootKick() satisfies it. P4 should reset() first.
    pub fn riscvBootKick(self: Falcon, dmem_sign: u32, engine_id: u32, ucode_id: u32) void {
        self.wrRiscv(BCR_DMEM_SIGN, dmem_sign);
        self.wrRiscv(BCR_ENGINE_ID, engine_id);
        self.wrRiscv(BCR_UCODE_ID, ucode_id);
        self.wrRiscv(BCR_CTRL, 0x00000001);
    }

    /// The BCR kick SEQUENCE as records (tests / record-mode). The offsets are the
    /// folded BAR0 offsets (base + riscv_base + reg) so a test can assert absolute.
    pub fn riscvBootKickSeq(self: Falcon, dmem_sign: u32, engine_id: u32, ucode_id: u32, out: []Write) usize {
        const b = self.base + self.riscv_base;
        out[0] = .{ .off = b + BCR_DMEM_SIGN, .val = dmem_sign };
        out[1] = .{ .off = b + BCR_ENGINE_ID, .val = engine_id };
        out[2] = .{ .off = b + BCR_UCODE_ID, .val = ucode_id };
        out[3] = .{ .off = b + BCR_CTRL, .val = 0x00000001 };
        return 4;
    }

    /// True when the RISC-V core is active (ga102_flcn_riscv_active:
    /// rdRiscv(0x388) & 0x80).
    pub fn riscvActive(self: Falcon) bool {
        return (self.rdRiscv(RISCV_CPUCTL) & RISCV_ACTIVE) != 0;
    }

    /// Select the RISCV core (ga102_flcn_select): if 0x668 bit 0x10 is set (FALCON
    /// selected) write 0 to 0x668, then poll bit 0x1. We emit the deterministic
    /// write; the poll is a read loop (selectDone()).
    pub fn riscvSelect(self: Falcon) void {
        if ((self.rdRiscv(RISCV_BCR_DMACFG_SEC) & 0x00000010) != 0) {
            self.wrRiscv(RISCV_BCR_DMACFG_SEC, 0x00000000);
        }
    }
};

// ===========================================================================
// Tests (offline, against the fake Bar from mmio.zig).
// ===========================================================================

const testing = std.testing;

test "mailbox r/w round-trips through the fake Bar at base+0x40/0x44" {
    var backing = [_]u8{0} ** (GSP_BASE + 0x1000);
    const bar = hw.Bar.init(@intFromPtr(&backing), backing.len);
    const f = Falcon.init(bar, GSP_BASE);

    f.mailboxWrite(0, 0xCAFEBEEF);
    f.mailboxWrite(1, 0x12345678);
    try testing.expectEqual(@as(u32, 0xCAFEBEEF), f.mailboxRead(0));
    try testing.expectEqual(@as(u32, 0x12345678), f.mailboxRead(1));
    // Confirm the absolute offsets: base + 0x40 and base + 0x44.
    try testing.expectEqual(@as(u32, 0xCAFEBEEF), bar.read32(GSP_BASE + 0x40));
    try testing.expectEqual(@as(u32, 0x12345678), bar.read32(GSP_BASE + 0x44));
}

test "reset writes IDLESTATE clear, IRQMCLR all, ENABLE in order" {
    var backing = [_]u8{0} ** (GSP_BASE + 0x1000);
    const bar = hw.Bar.init(@intFromPtr(&backing), backing.len);
    const f = Falcon.init(bar, GSP_BASE);

    // Seed IDLESTATE with bits that reset must clear, prove the RMW masks them.
    bar.write32(GSP_BASE + IDLESTATE, 0xFFFFFFFF);
    f.reset();

    // IDLESTATE bits 0..1 cleared (rest preserved).
    try testing.expectEqual(@as(u32, 0xFFFFFFFC), bar.read32(GSP_BASE + IDLESTATE));
    try testing.expectEqual(@as(u32, 0xFFFFFFFF), bar.read32(GSP_BASE + IRQMCLR));
    try testing.expectEqual(@as(u32, 0x00000000), bar.read32(GSP_BASE + ENABLE));

    // The record-mode sequence matches the same three offsets in order.
    var seq: [4]Write = undefined;
    const n = f.resetSeq(&seq);
    try testing.expectEqual(@as(usize, 3), n);
    try testing.expectEqual(GSP_BASE + IDLESTATE, seq[0].off);
    try testing.expectEqual(GSP_BASE + IRQMCLR, seq[1].off);
    try testing.expectEqual(@as(u32, 0xffffffff), seq[1].val);
    try testing.expectEqual(GSP_BASE + ENABLE, seq[2].off);
}

test "dmaCmd encodes size field 0x600 + IMEM + secure bits" {
    // 256-byte chunk: log2(256)=8 -> (8-2)<<8 = 0x600.
    try testing.expectEqual(@as(u32, 0x600), Falcon.dmaCmd(.dmem, false));
    try testing.expectEqual(@as(u32, 0x610), Falcon.dmaCmd(.imem, false));
    try testing.expectEqual(@as(u32, 0x614), Falcon.dmaCmd(.imem, true));
    try testing.expectEqual(@as(u32, 0x604), Falcon.dmaCmd(.dmem, true));
}

test "loadDma emits DMATRFBASE/MOFFS/FBOFFS/CMD with exact values per chunk" {
    var backing = [_]u8{0} ** (GSP_BASE + 0x1000);
    const bar = hw.Bar.init(@intFromPtr(&backing), backing.len);
    const f = Falcon.init(bar, GSP_BASE);

    // 512 bytes = 2 chunks of 256, into IMEM at mem_off 0, secure, from phys 0x40000000.
    const dma_addr: u64 = 0x40000000;
    const chunks = f.loadDma(.imem, dma_addr, 0, 512, true);
    try testing.expectEqual(@as(usize, 2), chunks);

    // init writes:
    try testing.expectEqual(@as(u32, 0x40000000 >> 8), bar.read32(GSP_BASE + DMATRFBASE));
    try testing.expectEqual(@as(u32, 0), bar.read32(GSP_BASE + DMATRFBASE1));

    // After the loop the per-chunk regs hold the LAST chunk's values (chunk 1):
    //   MOFFS = 256, FBOFFS = 256, CMD = 0x614 (IMEM|sec).
    try testing.expectEqual(@as(u32, 256), bar.read32(GSP_BASE + DMATRFMOFFS));
    try testing.expectEqual(@as(u32, 256), bar.read32(GSP_BASE + DMATRFFBOFFS));
    try testing.expectEqual(@as(u32, 0x614), bar.read32(GSP_BASE + DMATRFCMD));
}

test "loadDmem single chunk writes the exact DMEM DMA values" {
    // A single 256-byte DMEM chunk from phys 0x1000, mem_off 0: assert the exact
    // register values the builder lands (init + the chunk's three regs).
    var backing = [_]u8{0} ** (GSP_BASE + 0x1000);
    const bar = hw.Bar.init(@intFromPtr(&backing), backing.len);
    const f = Falcon.init(bar, GSP_BASE);

    const chunks = f.loadDmem(0x1000, 0, 256);
    try testing.expectEqual(@as(usize, 1), chunks);

    try testing.expectEqual(@as(u32, 0x1000 >> 8), bar.read32(GSP_BASE + DMATRFBASE));
    try testing.expectEqual(@as(u32, 0), bar.read32(GSP_BASE + DMATRFBASE1));
    try testing.expectEqual(@as(u32, 0), bar.read32(GSP_BASE + DMATRFMOFFS));
    try testing.expectEqual(@as(u32, 0), bar.read32(GSP_BASE + DMATRFFBOFFS));
    // DMEM, not secure -> CMD == 0x600 (no IMEM/sec bits).
    try testing.expectEqual(@as(u32, 0x600), bar.read32(GSP_BASE + DMATRFCMD));
}

test "loadImem emits the FBIF prep then the IMEM DMA" {
    var backing = [_]u8{0} ** (GSP_BASE + 0x1000);
    const bar = hw.Bar.init(@intFromPtr(&backing), backing.len);
    const f = Falcon.init(bar, GSP_BASE);

    const chunks = f.loadImem(0x80000000, 0, 256, true);
    try testing.expectEqual(@as(usize, 1), chunks);

    // FBIF prep: 0x624 bit 0x80 set, 0x10c == 0, 0x600 == (1<<2)|1 == 0x5.
    try testing.expectEqual(@as(u32, 0x80), bar.read32(GSP_BASE + FBIF_CTL) & 0x80);
    try testing.expectEqual(@as(u32, 0), bar.read32(GSP_BASE + CPUCTL2));
    try testing.expectEqual(@as(u32, 0x5), bar.read32(GSP_BASE + FBIF_TRANSCFG) & 0x10007);
    // The DMA actually ran (CMD register reflects IMEM|sec).
    try testing.expectEqual(@as(u32, 0x614), bar.read32(GSP_BASE + DMATRFCMD));
}

test "start writes BOOTVEC then CPUCTL STARTCPU" {
    var backing = [_]u8{0} ** (GSP_BASE + 0x1000);
    const bar = hw.Bar.init(@intFromPtr(&backing), backing.len);
    const f = Falcon.init(bar, GSP_BASE);

    f.start(0xDEAD0000);
    try testing.expectEqual(@as(u32, 0xDEAD0000), bar.read32(GSP_BASE + BOOTVEC));
    try testing.expectEqual(CPUCTL_STARTCPU, bar.read32(GSP_BASE + CPUCTL));
}

test "isHalted reads CPUCTL and masks bit 0x10" {
    var backing = [_]u8{0} ** (GSP_BASE + 0x1000);
    const bar = hw.Bar.init(@intFromPtr(&backing), backing.len);
    const f = Falcon.init(bar, GSP_BASE);

    bar.write32(GSP_BASE + CPUCTL, 0x00000000);
    try testing.expect(!f.isHalted());
    bar.write32(GSP_BASE + CPUCTL, 0x00000010);
    try testing.expect(f.isHalted());
    // Other bits set but not 0x10 -> not halted.
    bar.write32(GSP_BASE + CPUCTL, 0xFFFFFFEF);
    try testing.expect(!f.isHalted());
}

test "riscvBootKick writes BR sig, engine id, ucode id, BCR_CTRL in EXACT order" {
    var backing = [_]u8{0} ** (GSP_BASE + 0x2000);
    const bar = hw.Bar.init(@intFromPtr(&backing), backing.len);
    const f = Falcon.init(bar, GSP_BASE); // riscv_base defaults to 0x1000

    f.riscvBootKick(0xAAAA1111, 0x00000008, 0x00000005);

    const rb = GSP_BASE + RISCV_BASE;
    try testing.expectEqual(@as(u32, 0xAAAA1111), bar.read32(rb + BCR_DMEM_SIGN)); // 0x210
    try testing.expectEqual(@as(u32, 0x00000008), bar.read32(rb + BCR_ENGINE_ID)); // 0x19c
    try testing.expectEqual(@as(u32, 0x00000005), bar.read32(rb + BCR_UCODE_ID)); // 0x198
    try testing.expectEqual(@as(u32, 0x00000001), bar.read32(rb + BCR_CTRL)); // 0x180

    // The record-mode sequence asserts the ORDER explicitly (the riskiest path).
    var seq: [4]Write = undefined;
    const n = f.riscvBootKickSeq(0xAAAA1111, 0x8, 0x5, &seq);
    try testing.expectEqual(@as(usize, 4), n);
    const want = [_]Write{
        .{ .off = rb + BCR_DMEM_SIGN, .val = 0xAAAA1111 },
        .{ .off = rb + BCR_ENGINE_ID, .val = 0x8 },
        .{ .off = rb + BCR_UCODE_ID, .val = 0x5 },
        .{ .off = rb + BCR_CTRL, .val = 0x1 },
    };
    try testing.expectEqualSlices(Write, &want, &seq);
}

test "riscvActive reads addr2+0x388 and masks bit 0x80" {
    var backing = [_]u8{0} ** (GSP_BASE + 0x2000);
    const bar = hw.Bar.init(@intFromPtr(&backing), backing.len);
    const f = Falcon.init(bar, GSP_BASE);

    const rb = GSP_BASE + RISCV_BASE;
    bar.write32(rb + RISCV_CPUCTL, 0x00000000);
    try testing.expect(!f.riscvActive());
    bar.write32(rb + RISCV_CPUCTL, 0x00000080);
    try testing.expect(f.riscvActive());
    // bit set elsewhere but not 0x80.
    bar.write32(rb + RISCV_CPUCTL, 0xFFFFFF7F);
    try testing.expect(!f.riscvActive());
}

test "SEC2 base addresses fold correctly (the booter engine)" {
    var backing = [_]u8{0} ** (SEC2_BASE + 0x2000);
    const bar = hw.Bar.init(@intFromPtr(&backing), backing.len);
    const f = Falcon.init(bar, SEC2_BASE);

    // P4 writes the WprMeta phys to SEC2 mbox0/mbox1; prove the SEC2 aperture.
    f.mailboxWrite(0, 0x0000FE00); // lower 32
    f.mailboxWrite(1, 0x00000001); // upper 32
    try testing.expectEqual(@as(u32, 0x0000FE00), bar.read32(SEC2_BASE + MAILBOX0));
    try testing.expectEqual(@as(u32, 0x00000001), bar.read32(SEC2_BASE + MAILBOX1));
}

test {
    std.testing.refAllDecls(@This());
}
