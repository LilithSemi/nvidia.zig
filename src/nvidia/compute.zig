//! NVIDIA compute dispatch on Blackwell (class 0xCEC0): builds a QMDV05_00 launch
//! descriptor and the GPFIFO method sequence to run a compute grid on the SM
//! cores. Verified live: a hand-assembled SASS kernel (see sass.zig) runs and
//! writes a sentinel to memory.
//!
//! Compute lives on GPFIFO subchannel 1 (the 3D engine is subchannel 0); the
//! host routes methods to the engine by subchannel, so the launch methods here
//! are all emitted on subchannel 1.

const std = @import("std");
const sdk = @import("sdk.zig");

pub const BLACKWELL_COMPUTE_B: sdk.NvV32 = 0xcec0;
pub const QMD_DWORDS = 64; // QMDV05_00 is 256 bytes

const SUBCH = 1; // compute subchannel

// Compute method offsets (clcec0 / inherited clc6c0).
const SET_OBJECT = 0x0000;
const SET_SHADER_SHARED_MEMORY_WINDOW_A = 0x02a0; // + _B at 0x02a4
const SEND_PCAS_A = 0x02b4;
const SEND_SIGNALING_PCAS2_B = 0x02c0;
const SET_SHADER_LOCAL_MEMORY_WINDOW_A = 0x07b0; // + _B at 0x07b4
const PCAS_INVALIDATE_COPY_SCHEDULE = 3;
const WFI = 0x0078; // host method, wait for engine idle

/// Set `width` QMD bits at bit offset `lo` (the clcdc0qmd.h MW(hi:lo) ranges map
/// to lo = the low bit, width = hi - lo + 1).
fn qset(qmd: []u32, lo: usize, width: usize, val: u64) void {
    var i: usize = 0;
    while (i < width) : (i += 1) {
        const bit = lo + i;
        const off: u5 = @intCast(bit % 32);
        const b: u32 = @intCast((val >> @intCast(i)) & 1);
        qmd[bit / 32] = (qmd[bit / 32] & ~(@as(u32, 1) << off)) | (b << off);
    }
}

/// A compute grid launch. `prog_va` is the GPU VA of the assembled SASS program
/// (256-byte alignment is fine). Kernel parameters are delivered through constant
/// bank 0: set `cbuf0_va` to the GPU VA of a constant buffer and `cbuf0_size` to
/// its byte size, and the kernel's LDC c[0][off] reads land in that buffer.
pub const Grid = struct {
    prog_va: u64,
    grid: [3]u32 = .{ 1, 1, 1 }, // CTAs in x, y, z
    block: [3]u32 = .{ 1, 1, 1 }, // threads per CTA in x, y, z
    register_count: u32 = 8, // GPRs per thread (>= the kernel's usage)
    cbuf0_va: u64 = 0, // constant bank 0 GPU VA (0 = no constant buffer bound)
    cbuf0_size: u32 = 0, // constant bank 0 byte size (>= the highest LDC offset read)
};

/// Fill a zeroed 64-dword (256-byte) QMD to launch `g`. The QMD must then be
/// placed at a 256-byte-aligned GPU VA and handed to `Stream.dispatch`.
pub fn buildQmd(qmd: *[QMD_DWORDS]u32, g: Grid) void {
    @memset(qmd, 0);
    qset(qmd, 464, 4, 0); // QMD_MINOR_VERSION
    qset(qmd, 468, 4, 5); // QMD_MAJOR_VERSION (QMDV05_00)
    qset(qmd, 144, 6, 0x3f); // QMD_GROUP_ID
    qset(qmd, 151, 3, 2); // QMD_TYPE = GRID_CTA
    qset(qmd, 448, 8, 1); // SASS_VERSION (NAK uses 1 on every shader)
    qset(qmd, 456, 1, 1); // API_VISIBLE_CALL_LIMIT = NO_CHECK
    qset(qmd, 457, 1, 1); // SAMPLER_INDEX = VIA_HEADER_INDEX
    // Cache invalidations at launch (NAK sets these on every QMD); the constant
    // cache invalidate is what makes a freshly bound parameter buffer's reads land.
    qset(qmd, 472, 1, 1); // INVALIDATE_TEXTURE_HEADER_CACHE
    qset(qmd, 473, 1, 1); // INVALIDATE_TEXTURE_SAMPLER_CACHE
    qset(qmd, 474, 1, 1); // INVALIDATE_TEXTURE_DATA_CACHE
    qset(qmd, 475, 1, 1); // INVALIDATE_SHADER_DATA_CACHE
    qset(qmd, 477, 1, 1); // INVALIDATE_SHADER_CONSTANT_CACHE
    qset(qmd, 624, 2, 1); // CWD_MEMBAR_TYPE = L1_SYSMEMBAR
    qset(qmd, 1024, 32, (g.prog_va >> 4) & 0xFFFFFFFF); // PROGRAM_ADDRESS_LOWER (shifted4)
    qset(qmd, 1056, 21, (g.prog_va >> 4) >> 32); // PROGRAM_ADDRESS_UPPER
    // Program prefetch: addr is shifted-8, size in 256-byte units (a 4 KB kernel
    // page is the upper bound here).
    qset(qmd, 1888, 32, (g.prog_va >> 8) & 0xFFFFFFFF); // PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED
    qset(qmd, 1920, 17, g.prog_va >> 40); // PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED
    qset(qmd, 1077, 9, 0x10); // PROGRAM_PREFETCH_SIZE (256-byte units: 4 KB)
    qset(qmd, 1088, 16, g.block[0]); // CTA_THREAD_DIMENSION0
    qset(qmd, 1104, 16, g.block[1]);
    qset(qmd, 1120, 8, g.block[2]);
    qset(qmd, 1128, 9, g.register_count); // REGISTER_COUNT
    qset(qmd, 1137, 5, 1); // BARRIER_COUNT
    // SM shared-memory config: min/target = 0 KB, max = 100 KB. hw = kB/4 + 1.
    qset(qmd, 1163, 6, 1); // MIN_SM_CONFIG_SHARED_MEM_SIZE
    qset(qmd, 1169, 6, 26); // MAX_SM_CONFIG_SHARED_MEM_SIZE
    qset(qmd, 1175, 6, 1); // TARGET_SM_CONFIG_SHARED_MEM_SIZE
    qset(qmd, 1248, 32, g.grid[0]); // GRID_WIDTH
    qset(qmd, 1280, 16, g.grid[1]); // GRID_HEIGHT
    qset(qmd, 1312, 16, g.grid[2]); // GRID_DEPTH

    // Constant bank 0: bind the parameter buffer so the kernel's LDC c[0][off]
    // reads from it through the const cache. The address is shifted-6 (64-byte
    // alignment), split lower/upper; the size is shifted-4. VALID(0) gates the bank.
    // QMDV05_00 field offsets (clcdc0qmd.h, index 0): ADDR_LOWER_SHIFTED6 1375:1344,
    // ADDR_UPPER_SHIFTED6 1394:1376, SIZE_SHIFTED4 1407:1395, VALID 1856,
    // INVALIDATE 1859.
    if (g.cbuf0_va != 0) {
        const addr_s6 = g.cbuf0_va >> 6;
        qset(qmd, 1344, 32, addr_s6 & 0xFFFFFFFF); // CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6(0)
        qset(qmd, 1376, 19, addr_s6 >> 32); // CONSTANT_BUFFER_ADDR_UPPER_SHIFTED6(0)
        qset(qmd, 1395, 13, (g.cbuf0_size + 0xf) >> 4); // CONSTANT_BUFFER_SIZE_SHIFTED4(0)
        qset(qmd, 1856, 1, 1); // CONSTANT_BUFFER_VALID(0) = TRUE
        qset(qmd, 1859, 1, 1); // CONSTANT_BUFFER_INVALIDATE(0) = TRUE
    }
}

/// Builds the compute launch method stream (subchannel 1) into a caller buffer.
/// Order: `setup` once, then `dispatch` per grid, then optionally `fence`.
pub const Stream = struct {
    buf: []u32,
    n: usize = 0,

    fn m1(self: *Stream, addr: u32, v: u32) void {
        self.buf[self.n] = sdk.gpfifo.methodHeader(addr, SUBCH, 1);
        self.buf[self.n + 1] = v;
        self.n += 2;
    }
    fn mm(self: *Stream, addr: u32, vals: []const u32) void {
        self.buf[self.n] = sdk.gpfifo.methodHeader(addr, SUBCH, @intCast(vals.len));
        self.n += 1;
        for (vals) |v| {
            self.buf[self.n] = v;
            self.n += 1;
        }
    }

    /// Bind the compute class and program the shader memory windows (Blackwell:
    /// shared at 4 GB, local at 0xff<<24). Required once before any dispatch.
    pub fn setup(self: *Stream) void {
        self.m1(SET_OBJECT, BLACKWELL_COMPUTE_B);
        self.mm(SET_SHADER_SHARED_MEMORY_WINDOW_A, &.{ 1, 0 }); // (1 << 32)
        self.mm(SET_SHADER_LOCAL_MEMORY_WINDOW_A, &.{ 0, 0xFF000000 }); // (0xff << 24)
    }

    /// Launch the QMD at `qmd_va` (256-byte aligned): point the work distributor
    /// at it and schedule it (invalidating the caches first).
    pub fn dispatch(self: *Stream, qmd_va: u64) void {
        self.m1(SEND_PCAS_A, @intCast(qmd_va >> 8));
        self.m1(SEND_SIGNALING_PCAS2_B, PCAS_INVALIDATE_COPY_SCHEDULE);
    }

    /// Wait for the compute to go idle, then release `seq` to GPU VA `sem_va`
    /// (poll it on a CPU mapping to know the grid finished).
    pub fn fence(self: *Stream, sem_va: u64, seq: u32) void {
        self.m1(WFI, 1);
        const rel = sdk.gpfifo.semaphoreRelease(sem_va, seq);
        for (rel) |w| {
            self.buf[self.n] = w;
            self.n += 1;
        }
    }

    pub fn dwords(self: *const Stream) u32 {
        return @intCast(self.n);
    }
};

const rm = @import("rm.zig");
const sass = @import("sass.zig");

test "live: a hand-assembled SASS compute kernel runs on the SMs and stores" {
    var c = rm.Client.open() catch return error.SkipZigTest;
    defer c.deinit();
    const dev = c.allocDevice(0) catch return error.SkipZigTest;
    defer c.freeDevice(dev);
    const va = try c.allocVaSpace(dev);
    defer c.rmFree(dev.client, dev.device, va);

    const out_va: u64 = 0x5000000;
    const prog_va: u64 = 0x6000000;
    const qmd_va: u64 = 0x7000000;
    const gpf_va: u64 = 0x200000;
    const pb_va: u64 = 0x300000;
    const sem_va: u64 = 0x400000;

    // Kernel: *(u32*)out_va = 0xCAFE
    var code: [64]u32 = undefined;
    var asm_ = sass.Assembler{ .code = &code };
    asm_.movImm(0, @truncate(out_va), .{});
    asm_.movImm(1, @intCast(out_va >> 32), .{});
    asm_.movImm(2, 0xcafe, .{});
    asm_.stgU32(0, 2, .{});
    asm_.exit(.{ .stall = 1 });

    var qmd: [QMD_DWORDS]u32 = undefined;
    buildQmd(&qmd, .{ .prog_va = prog_va, .register_count = asm_.registerCount() });

    const userd = try c.allocMemory(dev, .vram, 0x1000);
    const gpfifo = try c.allocMemory(dev, .vram, 0x2000);
    const pbuf = try c.allocMemory(dev, .system, 0x1000);
    const sem = try c.allocMemory(dev, .system, 0x1000);
    const codem = try c.allocMemory(dev, .system_wc, 0x1000);
    const qmdm = try c.allocMemory(dev, .system_wc, 0x1000);
    const outm = try c.allocMemory(dev, .system, 0x1000);
    _ = try c.mapToGpu(dev, va, gpfifo, gpf_va);
    _ = try c.mapToGpu(dev, va, pbuf, pb_va);
    _ = try c.mapToGpu(dev, va, sem, sem_va);
    _ = try c.mapToGpu(dev, va, codem, prog_va);
    _ = try c.mapToGpu(dev, va, qmdm, qmd_va);
    _ = try c.mapToGpu(dev, va, outm, out_va);
    const uc = try c.mapMemory(dev, userd);
    const gc = try c.mapMemory(dev, gpfifo);
    const pc = try c.mapMemory(dev, pbuf);
    const sc = try c.mapMemory(dev, sem);
    const codec = try c.mapMemory(dev, codem);
    const qmdc = try c.mapMemory(dev, qmdm);
    const outc = try c.mapMemory(dev, outm);
    @memcpy(@as([*]u32, @ptrCast(@alignCast(codec.bytes.ptr)))[0..asm_.dwords()], code[0..asm_.dwords()]);
    @memcpy(@as([*]u32, @ptrCast(@alignCast(qmdc.bytes.ptr)))[0..QMD_DWORDS], &qmd);
    const outp: *volatile u32 = @ptrCast(@alignCast(outc.bytes.ptr));
    outp.* = 0;

    const ch = try c.allocChannel(dev, sdk.BLACKWELL_CHANNEL_GPFIFO_B, va, gpf_va, 0x100, userd);
    defer c.rmFree(dev.client, dev.device, ch.handle);
    _ = try c.allocObject(dev, ch, BLACKWELL_COMPUTE_B);
    try c.bindChannel(dev, ch, sdk.NV2080_ENGINE_TYPE_GRAPHICS);
    try c.scheduleChannel(dev, ch, true);
    const token = try c.workSubmitToken(dev, ch);
    const usermode = try c.allocUsermode(dev, sdk.BLACKWELL_USERMODE_A);
    const door = try c.mapMemory(dev, .{ .handle = usermode, .size = 0x1000, .location = .vram });

    var s = Stream{ .buf = @as([*]u32, @ptrCast(@alignCast(pc.bytes.ptr)))[0 .. pc.bytes.len / 4] };
    s.setup();
    s.dispatch(qmd_va);
    s.fence(sem_va, 0xc0de);

    const semp: *volatile u32 = @ptrCast(@alignCast(sc.bytes.ptr));
    semp.* = 0;
    var q = rm.Queue{ .channel = ch, .token = token, .userd = uc.bytes, .gpfifo = gc.bytes, .doorbell = door.bytes };
    q.submit(pb_va, s.dwords());
    var spins: u64 = 0;
    while (spins < 200_000_000) : (spins += 1) {
        if (semp.* == 0xc0de) break;
    }
    try std.testing.expectEqual(@as(u32, 0xcafe), outp.*);
}
