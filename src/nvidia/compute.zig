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
const INVALIDATE_SHADER_CACHES = 0x021c;
// Inline-to-memory (I2M): the payload rides in the pushbuffer instead of coming
// from a source buffer. 0x0180..0x0190 are five contiguous method offsets.
const LINE_LENGTH_IN = 0x0180; // + LINE_COUNT, OFFSET_OUT_UPPER, OFFSET_OUT, PITCH_OUT
const I2M_LAUNCH_DMA = 0x01b0;
const LOAD_INLINE_DATA = 0x01b4;
// DST_MEMORY_LAYOUT = PITCH, COMPLETION_TYPE = FLUSH_ONLY, everything else off.
const I2M_LAUNCH_PITCH_FLUSH: sdk.NvV32 = 0x11;
// INSTRUCTION | LOCKS | FLUSH_DATA | DATA | CONSTANT
const INVALIDATE_ALL_SHADER_CACHES: sdk.NvV32 = 0x1017;
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
    prog_va: u64 = 0, // filled in by `Runner.run`; set it by hand for a raw dispatch
    grid: [3]u32 = .{ 1, 1, 1 }, // CTAs in x, y, z
    block: [3]u32 = .{ 1, 1, 1 }, // threads per CTA in x, y, z
    register_count: u32 = 8, // GPRs per thread (>= the kernel's usage)
    cbuf0_va: u64 = 0, // constant bank 0 GPU VA (0 = no constant buffer bound)
    cbuf0_size: u32 = 0, // constant bank 0 byte size (>= the highest LDC offset read)
    shared_mem_bytes: u32 = 0, // shared memory per CTA; LDS/STS fault without it
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
    // Shared memory. The size field counts 128-byte units and the allocation is
    // rounded to 256 bytes; the min/target/max values name an SM partition size,
    // which the hardware encodes as kB/4 + 1. Max stays at the full 100 KB.
    const smem = std.mem.alignForward(u32, g.shared_mem_bytes, 0x100);
    qset(qmd, 1152, 11, smem >> 7); // SHARED_MEMORY_SIZE_SHIFTED7
    qset(qmd, 1163, 6, smemConfig(smem)); // MIN_SM_CONFIG_SHARED_MEM_SIZE
    qset(qmd, 1169, 6, 26); // MAX_SM_CONFIG_SHARED_MEM_SIZE
    qset(qmd, 1175, 6, smemConfig(smem)); // TARGET_SM_CONFIG_SHARED_MEM_SIZE
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

    /// Drop everything the SMs cached about the last program: its instructions,
    /// its data, and its constants. A dispatch that reuses a program address for
    /// new code needs this, or the SMs run the previous kernel out of the
    /// instruction cache.
    pub fn invalidateShaderCaches(self: *Stream) void {
        self.m1(INVALIDATE_SHADER_CACHES, INVALIDATE_ALL_SHADER_CACHES);
    }

    /// Write `data` into memory at `dst_va` from the pushbuffer itself, with no
    /// source buffer and no copy engine. The host unit feeds the payload to the
    /// compute engine as it reads the pushbuffer, which skips the copy engine's
    /// launch cost entirely; NVIDIA's own driver picks this path for small
    /// transfers. The cost is pushbuffer space, so it only pays below the
    /// crossover measured in the test at the bottom of this file.
    ///
    /// `dst_va` must be 4-byte aligned. The caller's buffer has to hold
    /// `data.len + 8` dwords or so of headers on top of the payload.
    pub fn uploadInline(self: *Stream, dst_va: u64, data: []const u32) void {
        std.debug.assert(dst_va % 4 == 0);
        if (data.len == 0) return;
        self.mm(LINE_LENGTH_IN, &.{
            @intCast(data.len * 4), // LINE_LENGTH_IN, in bytes
            1, // LINE_COUNT: one line
            @intCast(dst_va >> 32), // OFFSET_OUT_UPPER
            @truncate(dst_va), // OFFSET_OUT
            @intCast(data.len * 4), // PITCH_OUT
        });
        self.m1(I2M_LAUNCH_DMA, I2M_LAUNCH_PITCH_FLUSH);
        // One method header carries at most 8191 dwords, so a long payload needs
        // several. The engine keeps writing where the last chunk left off.
        var sent: usize = 0;
        while (sent < data.len) {
            const chunk = @min(data.len - sent, sdk.gpfifo.MAX_METHOD_COUNT);
            self.buf[self.n] = sdk.gpfifo.methodHeaderNonInc(LOAD_INLINE_DATA, SUBCH, @intCast(chunk));
            self.n += 1;
            @memcpy(self.buf[self.n..][0..chunk], data[sent..][0..chunk]);
            self.n += chunk;
            sent += chunk;
        }
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

const copy = @import("copy.zig");
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

/// The SM shared-memory partition that holds `bytes`, encoded the way the QMD
/// wants it: kB/4 + 1. The SM only splits its memory at these sizes, so round up
/// to the next one instead of asking for an arbitrary amount.
fn smemConfig(bytes: u32) u32 {
    const sizes_kb = [_]u32{ 0, 8, 16, 32, 64, 100 };
    for (sizes_kb) |kb| {
        if (kb * 1024 >= bytes) return kb / 4 + 1;
    }
    return 26; // 100 KB, the largest the SM offers
}

/// One allocation: the CPU bytes and the GPU virtual address they appear at.
pub const Buffer = struct {
    va: u64,
    bytes: []u8,
    memory: rm.Memory,

    /// A typed view, for filling the buffer before a launch.
    pub fn slice(self: Buffer, comptime T: type) []T {
        return @as([*]T, @ptrCast(@alignCast(self.bytes.ptr)))[0 .. self.bytes.len / @sizeOf(T)];
    }

    /// Read one element the GPU wrote. The read is volatile because this thread
    /// did not write the value and must not reuse a cached copy of it.
    pub fn read(self: Buffer, comptime T: type, index: usize) T {
        const p: [*]volatile T = @ptrCast(@alignCast(self.bytes.ptr));
        return p[index];
    }
};

/// The transfer size at which the copy engine overtakes the inline path on this
/// hardware. Below it, put the payload in the pushbuffer with
/// `Stream.uploadInline`; above it, hand the copy engine a source buffer with
/// `Runner.copyLinear`.
///
/// Measured on the GB10 in September 2026, nanoseconds per transfer, submission
/// and fence included:
///
///     bytes | inline |    CE
///       256 |   4265 |  4415
///      4096 |   4435 |  4589
///     16384 |   5206 |  5204
///     24576 |   5692 |  5439
///     65536 |   8875 |  7159
///
/// The crossover sits between 16 and 24 KiB, close to the 24 KiB that NVIDIA.s
/// own driver uses on Ampere (arXiv:2604.26889). Note what else the table says:
/// both paths cost about 4.3 us at the small end, so a single small transfer is
/// almost entirely submission overhead rather than transfer. Cutting the number
/// of submissions matters more than choosing between these two.
pub const inline_upload_limit_bytes: u32 = 16 * 1024;

/// Whether a transfer of `bytes` is cheaper through the pushbuffer than through
/// the copy engine.
pub fn preferInlineUpload(bytes: u32) bool {
    return bytes < inline_upload_limit_bytes;
}

/// A ready-to-use compute context: a GPFIFO channel bound to the compute class,
/// a VA space with a bump allocator behind `alloc`, and the code, descriptor,
/// pushbuffer and semaphore a launch needs. `run` uploads a kernel, dispatches
/// it, and waits for it, so a kernel test is a few lines instead of sixty.
pub const Runner = struct {
    client: rm.Client,
    dev: rm.Device,
    vaspace: sdk.NvHandle,
    channel: rm.Channel,
    queue: rm.Queue,
    next_va: u64 = VA_BASE,
    seq: u32 = 0,
    code: Buffer,
    descriptor: Buffer,
    push: Buffer,
    sem: Buffer,
    /// A second channel on the copy engine, brought up only when something asks
    /// for a copy. The copy engine has its own runlist, so it cannot share the
    /// compute channel; keeping both here lets one test time the two paths
    /// against each other.
    copy_channel: ?rm.Channel = null,
    copy_queue: rm.Queue = undefined,
    copy_push: Buffer = undefined,
    copy_sem: Buffer = undefined,
    copy_seq: u32 = 0,
    doorbell: []u8 = &.{},

    /// Where the bump allocator starts, and the step between allocations. One
    /// 2 MB step per buffer keeps every allocation on its own big page.
    const VA_BASE: u64 = 0x1000_0000;
    const VA_STEP: u64 = 0x20_0000;
    const CODE_BYTES = 0x1000; // matches the QMD's 4 KB program prefetch
    /// How many launch descriptors the batch area holds, and so the most
    /// dispatches one submission can carry.
    pub const MAX_BATCH = 64;
    // Big enough for a sizeable inline upload: the payload rides in here.
    const PUSH_BYTES = 0x20000;
    const GPFIFO_BYTES = 0x2000;
    const GPFIFO_ENTRIES = 0x100;
    /// How long to poll the completion semaphore before calling the grid hung.
    const SPIN_LIMIT = 200_000_000;

    /// Open GPU 0 and bring up a compute channel on it. Returns
    /// `error.SkipZigTest` when no GPU is reachable, so tests skip instead of
    /// failing on a machine without one.
    pub fn init() !Runner {
        var client = rm.Client.open() catch return error.SkipZigTest;
        errdefer client.deinit();
        const dev = client.allocDevice(0) catch return error.SkipZigTest;
        errdefer client.freeDevice(dev);
        const vaspace = try client.allocVaSpace(dev);

        var self: Runner = .{
            .client = client,
            .dev = dev,
            .vaspace = vaspace,
            .channel = undefined,
            .queue = undefined,
            .code = undefined,
            .descriptor = undefined,
            .push = undefined,
            .sem = undefined,
        };

        // The ring and the USERD are read by the host unit, so they live in VRAM.
        // Only the ring needs a GPU virtual address.
        const gpfifo = try self.client.allocMemory(dev, .vram, GPFIFO_BYTES);
        const gpfifo_va = self.takeVa(GPFIFO_BYTES);
        _ = try self.client.mapToGpu(dev, vaspace, gpfifo, gpfifo_va);
        const gpfifo_cpu = try self.client.mapMemory(dev, gpfifo);
        const userd = try self.client.allocMemory(dev, .vram, 0x1000);
        const userd_cpu = try self.client.mapMemory(dev, userd);

        // Code and descriptor are write-combined: the CPU only writes them.
        self.code = try self.alloc(.system_wc, CODE_BYTES);
        self.descriptor = try self.alloc(.system_wc, MAX_BATCH * QMD_DWORDS * 4);
        self.push = try self.alloc(.system, PUSH_BYTES);
        self.sem = try self.alloc(.system, 0x1000);

        const ch = try self.client.allocChannel(
            dev,
            sdk.BLACKWELL_CHANNEL_GPFIFO_B,
            vaspace,
            gpfifo_va,
            GPFIFO_ENTRIES,
            userd,
        );
        errdefer self.client.rmFree(dev.client, dev.device, ch.handle);
        _ = try self.client.allocObject(dev, ch, BLACKWELL_COMPUTE_B);
        try self.client.bindChannel(dev, ch, sdk.NV2080_ENGINE_TYPE_GRAPHICS);
        try self.client.scheduleChannel(dev, ch, true);
        const token = try self.client.workSubmitToken(dev, ch);
        const usermode = try self.client.allocUsermode(dev, sdk.BLACKWELL_USERMODE_A);
        const door = try self.client.mapMemory(dev, .{
            .handle = usermode,
            .size = 0x1000,
            .location = .vram,
        });

        self.channel = ch;
        self.doorbell = door.bytes;
        self.queue = .{
            .channel = ch,
            .token = token,
            .userd = userd_cpu.bytes,
            .gpfifo = gpfifo_cpu.bytes,
            .doorbell = door.bytes,
        };
        return self;
    }

    /// Bring up the copy-engine channel on first use. It needs its own ring,
    /// USERD, pushbuffer and semaphore, but shares the VA space and the doorbell
    /// page with the compute channel.
    fn ensureCopyChannel(self: *Runner) !void {
        if (self.copy_channel != null) return;
        const gpfifo = try self.client.allocMemory(self.dev, .vram, GPFIFO_BYTES);
        const gpfifo_va = self.takeVa(GPFIFO_BYTES);
        _ = try self.client.mapToGpu(self.dev, self.vaspace, gpfifo, gpfifo_va);
        const gpfifo_cpu = try self.client.mapMemory(self.dev, gpfifo);
        const userd = try self.client.allocMemory(self.dev, .vram, 0x1000);
        const userd_cpu = try self.client.mapMemory(self.dev, userd);
        self.copy_push = try self.alloc(.system, 0x1000);
        self.copy_sem = try self.alloc(.system, 0x1000);

        const ch = try self.client.allocChannelEngine(
            self.dev,
            sdk.BLACKWELL_CHANNEL_GPFIFO_B,
            self.vaspace,
            gpfifo_va,
            GPFIFO_ENTRIES,
            userd,
            sdk.NV2080_ENGINE_TYPE_COPY0,
        );
        _ = try self.client.allocObject(self.dev, ch, sdk.BLACKWELL_DMA_COPY_B);
        try self.client.bindChannel(self.dev, ch, sdk.NV2080_ENGINE_TYPE_COPY0);
        try self.client.scheduleChannel(self.dev, ch, true);
        const token = try self.client.workSubmitToken(self.dev, ch);
        self.copy_channel = ch;
        self.copy_queue = .{
            .channel = ch,
            .token = token,
            .userd = userd_cpu.bytes,
            .gpfifo = gpfifo_cpu.bytes,
            .doorbell = self.doorbell,
        };
    }

    /// Copy `bytes` from `src_va` to `dst_va` on the copy engine and wait for the
    /// engine's own semaphore to land.
    pub fn copyLinear(self: *Runner, dst_va: u64, src_va: u64, bytes: u32) !void {
        try self.ensureCopyChannel();
        self.copy_seq += 1;
        var s = copy.Stream{ .buf = self.copy_push.slice(u32) };
        s.setup();
        s.linear(.{
            .src_va = src_va,
            .dst_va = dst_va,
            .bytes = bytes,
            .sem_va = self.copy_sem.va,
            .sem_seq = self.copy_seq,
        });
        const semp: *volatile u32 = @ptrCast(@alignCast(self.copy_sem.bytes.ptr));
        semp.* = 0;
        self.copy_queue.submit(self.copy_push.va, s.dwords());
        var spins: u64 = 0;
        while (spins < SPIN_LIMIT) : (spins += 1) {
            if (semp.* == self.copy_seq) return;
        }
        return error.CopyTimeout;
    }

    pub fn deinit(self: *Runner) void {
        self.client.rmFree(self.dev.client, self.dev.device, self.channel.handle);
        self.client.freeDevice(self.dev);
        self.client.deinit();
    }

    fn takeVa(self: *Runner, size: u64) u64 {
        // Step over the span the driver keeps for itself. A mapping in there
        // succeeds and then swallows every store, so the allocator must never
        // hand one out.
        if (rm.Client.vaIsReserved(self.next_va, size)) self.next_va = rm.Client.RESERVED_VA_END;
        const va = self.next_va;
        self.next_va += std.mem.alignForward(u64, size, VA_STEP);
        return va;
    }

    /// Allocate `size` bytes, map them to the GPU and to the CPU, and zero them.
    pub fn alloc(self: *Runner, location: rm.Memory.Location, size: u64) !Buffer {
        const rounded = std.mem.alignForward(u64, @max(size, 0x1000), 0x1000);
        const mem = try self.client.allocMemory(self.dev, location, rounded);
        const va = self.takeVa(rounded);
        _ = try self.client.mapToGpu(self.dev, self.vaspace, mem, va);
        const cpu = try self.client.mapMemory(self.dev, mem);
        @memset(cpu.bytes[0..rounded], 0);
        return .{ .va = va, .bytes = cpu.bytes[0..rounded], .memory = mem };
    }

    /// Send `data` into `dst_va` through the pushbuffer and wait for it to land.
    /// This is the small-transfer path: no source buffer and no copy engine, at
    /// the cost of carrying the payload in the pushbuffer.
    pub fn uploadInline(self: *Runner, dst_va: u64, data: []const u32) !void {
        var s = Stream{ .buf = self.push.slice(u32) };
        s.setup();
        s.uploadInline(dst_va, data);
        self.seq += 1;
        s.fence(self.sem.va, self.seq);
        return self.submitAndWait(s.dwords());
    }

    /// Ring the doorbell for the pushbuffer and spin until the fence lands.
    fn submitAndWait(self: *Runner, dwords: u32) !void {
        const semp: *volatile u32 = @ptrCast(@alignCast(self.sem.bytes.ptr));
        semp.* = 0;
        self.queue.submit(self.push.va, dwords);
        var spins: u64 = 0;
        while (spins < SPIN_LIMIT) : (spins += 1) {
            if (semp.* == self.seq) return;
        }
        return error.GridTimeout;
    }

    /// Upload `code`, dispatch grid `g`, and wait for it. `g.prog_va` is filled
    /// in here. A grid that never signals gives `error.GridTimeout`, which means
    /// the kernel hung or faulted.
    pub fn run(self: *Runner, code: []const u32, g: Grid) !void {
        return self.runBatch(code, &.{g});
    }

    /// Dispatch every grid in `grids` from one pushbuffer, with one ring of the
    /// doorbell and one fence at the end. All of them run the same program.
    ///
    /// The grids are not ordered against each other: the work distributor may
    /// overlap them, so they must not write the same memory. What they save is
    /// the per-submission cost, which is most of what a small kernel costs.
    ///
    /// Measured on the GB10 in September 2026, nanoseconds per dispatch:
    ///
    ///     dispatches | batched | one at a time
    ///              1 |   19711 |         19569
    ///              4 |    9084 |         19598
    ///             16 |    6663 |         19596
    ///             64 |    6228 |         19901
    ///
    /// A dispatch on its own costs about 19.6 us however many there are. In a
    /// batch that falls to about 6.2 us, so roughly two thirds of a small
    /// launch is the submission round trip rather than the launch itself.
    pub fn runBatch(self: *Runner, code: []const u32, grids: []const Grid) !void {
        std.debug.assert(code.len * 4 <= self.code.bytes.len);
        std.debug.assert(grids.len > 0 and grids.len <= MAX_BATCH);
        @memcpy(self.code.slice(u32)[0..code.len], code);

        const descriptors = self.descriptor.slice(u32);
        for (grids, 0..) |g, i| {
            var grid = g;
            grid.prog_va = self.code.va;
            var qmd: [QMD_DWORDS]u32 = undefined;
            buildQmd(&qmd, grid);
            @memcpy(descriptors[i * QMD_DWORDS ..][0..QMD_DWORDS], &qmd);
        }

        var s = Stream{ .buf = self.push.slice(u32) };
        s.setup();
        s.invalidateShaderCaches();
        for (0..grids.len) |i| s.dispatch(self.descriptor.va + i * QMD_DWORDS * 4);
        self.seq += 1;
        s.fence(self.sem.va, self.seq);
        return self.submitAndWait(s.dwords());
    }
};

test "live: integer ALU (IADD3, IMAD, IMAD.WIDE, ISETP, SEL) computes on the SMs" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);

    var code: [256]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.movImm(0, @truncate(out.va), .{});
    a.movImm(1, @intCast(out.va >> 32), .{});
    a.movImm(2, 7, .{});
    a.movImm(3, 5, .{});
    a.iadd3(4, sass.Src.reg(2), sass.Src.reg(3), sass.zero, .{}); // 12
    a.imad(5, sass.Src.reg(2), sass.Src.reg(3), sass.Src.reg(4), false, .{}); // 47
    a.isetp(0, .gt, true, sass.Src.reg(5), sass.Src.imm(46), .{}); // P0 = true
    a.sel(6, sass.Src.reg(2), sass.Src.reg(3), 0, false, .{}); // 7
    a.imadWide(8, sass.Src.reg(2), sass.Src.imm(4), sass.zero, false, .{}); // 28
    a.stg(0, 4, 0, .bits32, .{});
    a.stg(0, 5, 4, .bits32, .{});
    a.stg(0, 6, 8, .bits32, .{});
    a.stg(0, 8, 16, .bits64, .{});
    a.exit(.{ .stall = 1 });

    try r.run(code[0..a.dwords()], .{ .register_count = a.registerCount() });
    try std.testing.expectEqual(@as(u32, 12), out.read(u32, 0));
    try std.testing.expectEqual(@as(u32, 47), out.read(u32, 1));
    try std.testing.expectEqual(@as(u32, 7), out.read(u32, 2));
    try std.testing.expectEqual(@as(u64, 28), out.read(u64, 2));
}

test "live: single-precision FADD, FMUL and FFMA compute on the SMs" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);

    var code: [256]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.movImm(0, @truncate(out.va), .{});
    a.movImm(1, @intCast(out.va >> 32), .{});
    a.mov(2, sass.Src.float(2.0), .{});
    a.mov(3, sass.Src.float(3.0), .{});
    a.fadd(4, sass.Src.reg(2), sass.Src.reg(3), .{}); // 5.0
    a.fmul(5, sass.Src.reg(2), sass.Src.reg(3), .{}); // 6.0
    a.ffma(6, sass.Src.reg(2), sass.Src.reg(3), sass.Src.reg(4), .{}); // 11.0
    a.fadd(7, sass.Src.reg(2), sass.Src.float(1.5), .{}); // 3.5
    a.fadd(9, sass.Src.reg(2), sass.Src.reg(3).negated(), .{}); // -1.0
    a.stg(0, 4, 0, .bits32, .{});
    a.stg(0, 5, 4, .bits32, .{});
    a.stg(0, 6, 8, .bits32, .{});
    a.stg(0, 7, 12, .bits32, .{});
    a.stg(0, 9, 16, .bits32, .{});
    a.exit(.{ .stall = 1 });

    try r.run(code[0..a.dwords()], .{ .register_count = a.registerCount() });
    try std.testing.expectEqual(@as(f32, 5.0), out.read(f32, 0));
    try std.testing.expectEqual(@as(f32, 6.0), out.read(f32, 1));
    try std.testing.expectEqual(@as(f32, 11.0), out.read(f32, 2));
    try std.testing.expectEqual(@as(f32, 3.5), out.read(f32, 3));
    try std.testing.expectEqual(@as(f32, -1.0), out.read(f32, 4));
}

test "live: LDC reads kernel parameters out of constant bank 0" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);
    const params = try r.alloc(.system, 0x1000);
    const pw = params.slice(u32);
    pw[0] = 0xa11ce;
    pw[1] = 0xb0b;
    pw[2] = 0xfeed;

    var code: [256]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.movImm(0, @truncate(out.va), .{});
    a.movImm(1, @intCast(out.va >> 32), .{});
    // Constant reads have no fixed latency: each LDC sets a scoreboard and the
    // store that consumes the result waits on it.
    a.ldc(2, .{ .bank = 0, .offset = 0 }, sass.RZ, .bits32, .{ .wr_barrier = 0 });
    a.ldc(3, .{ .bank = 0, .offset = 4 }, sass.RZ, .bits32, .{ .wr_barrier = 1 });
    a.movImm(4, 8, .{});
    a.ldc(5, .{ .bank = 0, .offset = 0 }, 4, .bits32, .{ .wr_barrier = 2 }); // indexed
    a.stg(0, 2, 0, .bits32, .{ .wait_mask = 0b001 });
    a.stg(0, 3, 4, .bits32, .{ .wait_mask = 0b010 });
    a.stg(0, 5, 8, .bits32, .{ .wait_mask = 0b100 });
    a.exit(.{ .stall = 1 });

    try r.run(code[0..a.dwords()], .{
        .register_count = a.registerCount(),
        .cbuf0_va = params.va,
        .cbuf0_size = 256,
    });
    try std.testing.expectEqual(@as(u32, 0xa11ce), out.read(u32, 0));
    try std.testing.expectEqual(@as(u32, 0xb0b), out.read(u32, 1));
    try std.testing.expectEqual(@as(u32, 0xfeed), out.read(u32, 2));
}

test "live: LDG reads global memory back into a register" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);
    const in = try r.alloc(.system, 0x1000);
    for (in.slice(u32)[0..8], 0..) |*w, i| w.* = @intCast(0x100 + i);

    var code: [256]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.movImm(0, @truncate(out.va), .{});
    a.movImm(1, @intCast(out.va >> 32), .{});
    a.movImm(2, @truncate(in.va), .{});
    a.movImm(3, @intCast(in.va >> 32), .{});
    // A global load has no fixed latency, so it sets scoreboard 0 and the store
    // that consumes the result waits on it.
    a.ldg(4, 2, 12, .bits32, .{ .wr_barrier = 0 });
    a.ldg(6, 2, 0, .bits64, .{ .wr_barrier = 1 });
    a.stg(0, 4, 0, .bits32, .{ .wait_mask = 0b01 });
    a.stg(0, 6, 8, .bits64, .{ .wait_mask = 0b10 });
    a.exit(.{ .stall = 1 });

    try r.run(code[0..a.dwords()], .{ .register_count = a.registerCount() });
    try std.testing.expectEqual(@as(u32, 0x103), out.read(u32, 0));
    try std.testing.expectEqual(@as(u32, 0x100), out.read(u32, 2));
    try std.testing.expectEqual(@as(u32, 0x101), out.read(u32, 3));
}

test "live: a predicated backward branch runs a counted loop" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);

    var code: [256]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.movImm(0, @truncate(out.va), .{});
    a.movImm(1, @intCast(out.va >> 32), .{});
    a.movImm(2, 0, .{}); // accumulator
    a.movImm(3, 0, .{}); // counter
    const loop = a.here();
    a.iadd3(2, sass.Src.reg(2), sass.Src.reg(3), sass.zero, .{});
    a.iadd3(3, sass.Src.reg(3), sass.Src.imm(1), sass.zero, .{});
    a.isetp(0, .lt, true, sass.Src.reg(3), sass.Src.imm(10), .{});
    a.bra(loop, .{ .pred = 0 });
    a.stg(0, 2, 0, .bits32, .{});
    a.exit(.{ .stall = 1 });

    try r.run(code[0..a.dwords()], .{ .register_count = a.registerCount() });
    try std.testing.expectEqual(@as(u32, 45), out.read(u32, 0)); // 0+1+..+9
}

test "live: a forward branch skips the instructions it jumps over" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);

    var code: [256]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.movImm(0, @truncate(out.va), .{});
    a.movImm(1, @intCast(out.va >> 32), .{});
    a.movImm(2, 0x600d, .{});
    const skip = a.braForward(.{});
    // A branch that lands short writes one of these markers instead.
    a.movImm(2, 0xbad1, .{});
    a.movImm(2, 0xbad2, .{});
    a.movImm(2, 0xbad3, .{});
    a.patchBranch(skip, a.here());
    a.stg(0, 2, 0, .bits32, .{});
    a.exit(.{ .stall = 1 });

    try r.run(code[0..a.dwords()], .{ .register_count = a.registerCount() });
    try std.testing.expectEqual(@as(u32, 0x600d), out.read(u32, 0));
}

test "live: every thread of a grid writes its own global index" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);

    var code: [256]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.s2r(2, sass.SR_TID_X, .{ .wr_barrier = 0 });
    a.s2r(3, sass.SR_CTAID_X, .{ .wr_barrier = 1 });
    // index = ctaid.x * 8 + tid.x
    a.imad(4, sass.Src.reg(3), sass.Src.imm(8), sass.Src.reg(2), false, .{ .wait_mask = 0b11 });
    // The output address is the base plus index*4, built with a wide multiply.
    a.movImm(6, @truncate(out.va), .{});
    a.movImm(7, @intCast(out.va >> 32), .{});
    a.imadWide(0, sass.Src.reg(4), sass.Src.imm(4), sass.Src.reg(6), false, .{});
    a.stg(0, 4, 0, .bits32, .{});
    a.exit(.{ .stall = 1 });

    try r.run(code[0..a.dwords()], .{
        .register_count = a.registerCount(),
        .grid = .{ 4, 1, 1 },
        .block = .{ 8, 1, 1 },
    });
    for (0..32) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i)), out.read(u32, i));
    }
}

test "live: a carry chain of two IADD3 makes a 64-bit add" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);

    var code: [256]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.movImm(0, @truncate(out.va), .{});
    a.movImm(1, @intCast(out.va >> 32), .{});
    a.movImm(2, 0xffffffff, .{});
    a.movImm(3, 0, .{});
    // R2:R3 += 1. The low add produces the carry, the high add consumes it.
    a.iadd3Carry(2, sass.Src.reg(2), sass.Src.imm(1), sass.zero, 0, null, .{});
    a.iadd3Carry(3, sass.Src.reg(3), sass.zero, sass.zero, sass.PT, 0, .{});
    a.stg(0, 2, 0, .bits64, .{});
    // A second step, this time with no carry out of the low half.
    a.iadd3Carry(2, sass.Src.reg(2), sass.Src.imm(7), sass.zero, 0, null, .{});
    a.iadd3Carry(3, sass.Src.reg(3), sass.zero, sass.zero, sass.PT, 0, .{});
    a.stg(0, 2, 8, .bits64, .{});
    a.exit(.{ .stall = 1 });

    try r.run(code[0..a.dwords()], .{ .register_count = a.registerCount() });
    try std.testing.expectEqual(@as(u64, 0x1_0000_0000), out.read(u64, 0));
    try std.testing.expectEqual(@as(u64, 0x1_0000_0007), out.read(u64, 1));
}

/// Byte offsets of the matmul kernel's parameters inside constant bank 0.
const mm_params = struct {
    const a_ptr = 0;
    const b_ptr = 8;
    const c_ptr = 16;
    const rows = 24; // M
    const cols = 28; // N
    const depth = 32; // K
    const size = 36;
};

/// Assemble a naive single-precision matmul: C[M,N] = A[M,K] * B[K,N], all
/// row-major. One thread computes one element of C, so the launch is a grid of
/// `tile` by `tile` blocks covering C. K must be at least 1: the inner loop
/// tests its counter at the bottom.
///
/// Registers: R0:R1 and R2:R3 walk A and B, R4 accumulates, R5 counts k,
/// R6/R7 hold the loaded elements, R8/R9 are the row and column of this thread,
/// R10..R13 hold M, N, K and the byte stride of a B row, R14:R15 addresses C,
/// R16 is scratch, and R20..R25 hold the three base pointers.
fn buildMatmul(a: *sass.Assembler, tile: u32) void {
    const Src = sass.Src;
    const zero = sass.zero;

    // row = ctaid.y * tile + tid.y, col = ctaid.x * tile + tid.x
    a.s2r(8, sass.SR_TID_Y, .{ .wr_barrier = 0 });
    a.s2r(16, sass.SR_CTAID_Y, .{ .wr_barrier = 1 });
    a.imad(8, Src.reg(16), Src.imm(tile), Src.reg(8), false, .{ .wait_mask = 0b11 });
    a.s2r(9, sass.SR_TID_X, .{ .wr_barrier = 0 });
    a.s2r(16, sass.SR_CTAID_X, .{ .wr_barrier = 1 });
    a.imad(9, Src.reg(16), Src.imm(tile), Src.reg(9), false, .{ .wait_mask = 0b11 });

    // Every parameter read is a separate scoreboard, so each use waits only on
    // the value it needs.
    a.ldc(10, .{ .offset = mm_params.rows }, sass.RZ, .bits32, .{ .wr_barrier = 0 });
    a.ldc(11, .{ .offset = mm_params.cols }, sass.RZ, .bits32, .{ .wr_barrier = 1 });
    a.ldc(12, .{ .offset = mm_params.depth }, sass.RZ, .bits32, .{ .wr_barrier = 2 });
    a.ldc(20, .{ .offset = mm_params.a_ptr }, sass.RZ, .bits64, .{ .wr_barrier = 3 });
    a.ldc(22, .{ .offset = mm_params.b_ptr }, sass.RZ, .bits64, .{ .wr_barrier = 4 });
    a.ldc(24, .{ .offset = mm_params.c_ptr }, sass.RZ, .bits64, .{ .wr_barrier = 5 });

    // The grid is rounded up to whole tiles, so the threads past the edge of C
    // leave without touching memory.
    a.isetp(0, .ge, true, Src.reg(8), Src.reg(10), .{ .wait_mask = 0b000001 });
    const row_outside = a.braForward(.{ .pred = 0 });
    a.isetp(0, .ge, true, Src.reg(9), Src.reg(11), .{ .wait_mask = 0b000010 });
    const col_outside = a.braForward(.{ .pred = 0 });

    a.imad(16, Src.reg(8), Src.reg(12), zero, false, .{ .wait_mask = 0b000100 }); // row*K
    a.imadWide(0, Src.reg(16), Src.imm(4), Src.reg(20), false, .{ .wait_mask = 0b001000 });
    a.imadWide(2, Src.reg(9), Src.imm(4), Src.reg(22), false, .{ .wait_mask = 0b010000 });
    a.imad(13, Src.reg(11), Src.imm(4), zero, false, .{}); // bytes per B row
    a.movImm(4, 0, .{}); // accumulator = 0.0f
    a.movImm(5, 0, .{}); // k = 0

    // The loop walks A along its row and B down its column. Both loads set a
    // write scoreboard for the multiply and a read scoreboard for the pointer
    // step, so the step cannot outrun the address the load still needs.
    const loop = a.here();
    a.ldg(6, 0, 0, .bits32, .{ .wr_barrier = 0, .rd_barrier = 2 });
    a.ldg(7, 2, 0, .bits32, .{ .wr_barrier = 1, .rd_barrier = 3 });
    a.iadd3Carry(0, Src.reg(0), Src.imm(4), zero, 1, null, .{ .wait_mask = 0b001100 });
    a.iadd3Carry(1, Src.reg(1), zero, zero, sass.PT, 1, .{});
    a.iadd3Carry(2, Src.reg(2), Src.reg(13), zero, 1, null, .{});
    a.iadd3Carry(3, Src.reg(3), zero, zero, sass.PT, 1, .{});
    a.ffma(4, Src.reg(6), Src.reg(7), Src.reg(4), .{ .wait_mask = 0b000011 });
    a.iadd3(5, Src.reg(5), Src.imm(1), zero, .{});
    a.isetp(0, .lt, true, Src.reg(5), Src.reg(12), .{});
    a.bra(loop, .{ .pred = 0 });

    a.imad(16, Src.reg(8), Src.reg(11), Src.reg(9), false, .{}); // row*N + col
    a.imadWide(14, Src.reg(16), Src.imm(4), Src.reg(24), false, .{ .wait_mask = 0b100000 });
    a.stg(14, 4, 0, .bits32, .{});

    a.patchBranch(row_outside, a.here());
    a.patchBranch(col_outside, a.here());
    a.exit(.{ .stall = 1 });
}

test "live: a naive FP32 matmul kernel matches a CPU reference" {
    var r = try Runner.init();
    defer r.deinit();

    // Sizes that are not multiples of the tile, so the edge guards are exercised.
    const rows = 37;
    const cols = 29;
    const depth = 23;
    const tile = 16;

    const a_buf = try r.alloc(.system, rows * depth * 4);
    const b_buf = try r.alloc(.system, depth * cols * 4);
    const c_buf = try r.alloc(.system, rows * cols * 4);
    const params = try r.alloc(.system, mm_params.size);

    // Small whole numbers keep every product and every partial sum exact in
    // f32, so the comparison below can demand equality rather than a tolerance.
    const av = a_buf.slice(f32);
    for (0..rows * depth) |i| av[i] = @floatFromInt(@as(i32, @intCast(i % 7)) - 3);
    const bv = b_buf.slice(f32);
    for (0..depth * cols) |i| bv[i] = @floatFromInt(@as(i32, @intCast(i % 5)) - 2);

    const p = params.slice(u32);
    p[0] = @truncate(a_buf.va);
    p[1] = @intCast(a_buf.va >> 32);
    p[2] = @truncate(b_buf.va);
    p[3] = @intCast(b_buf.va >> 32);
    p[4] = @truncate(c_buf.va);
    p[5] = @intCast(c_buf.va >> 32);
    p[6] = rows;
    p[7] = cols;
    p[8] = depth;

    var code: [1024]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    buildMatmul(&a, tile);

    try r.run(code[0..a.dwords()], .{
        .register_count = a.registerCount(),
        .grid = .{ (cols + tile - 1) / tile, (rows + tile - 1) / tile, 1 },
        .block = .{ tile, tile, 1 },
        .cbuf0_va = params.va,
        .cbuf0_size = mm_params.size,
    });

    for (0..rows) |i| {
        for (0..cols) |j| {
            var want: f32 = 0;
            for (0..depth) |k| want += av[i * depth + k] * bv[k * cols + j];
            try std.testing.expectEqual(want, c_buf.read(f32, i * cols + j));
        }
    }
}

test "live: shared memory and BAR.SYNC exchange values across a block" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);
    const threads = 32;

    var code: [256]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.s2r(2, sass.SR_TID_X, .{ .wr_barrier = 0 });
    a.imad(3, sass.Src.reg(2), sass.Src.imm(4), sass.zero, false, .{ .wait_mask = 0b01 });
    a.sts(3, 2, 0, .bits32, .{});
    // Every thread must have written before any thread reads its neighbour.
    a.bar(.{});
    a.movImm(7, threads - 1, .{});
    // R4 = (threads - 1) - tid
    a.imad(4, sass.Src.reg(2), sass.Src.imm(0xffffffff), sass.Src.reg(7), true, .{});
    a.imad(5, sass.Src.reg(4), sass.Src.imm(4), sass.zero, false, .{});
    a.lds(6, 5, 0, .bits32, .{ .wr_barrier = 0 });
    a.movImm(8, @truncate(out.va), .{});
    a.movImm(9, @intCast(out.va >> 32), .{});
    a.imadWide(0, sass.Src.reg(2), sass.Src.imm(4), sass.Src.reg(8), false, .{});
    a.stg(0, 6, 0, .bits32, .{ .wait_mask = 0b01 });
    a.exit(.{ .stall = 1 });

    try r.run(code[0..a.dwords()], .{
        .register_count = a.registerCount(),
        .block = .{ threads, 1, 1 },
        .shared_mem_bytes = threads * 4,
    });
    for (0..threads) |i| {
        try std.testing.expectEqual(@as(u32, threads - 1 - @as(u32, @intCast(i))), out.read(u32, i));
    }
}

test "live: SHF shifts a register left and right" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);

    var code: [256]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.movImm(0, @truncate(out.va), .{});
    a.movImm(1, @intCast(out.va >> 32), .{});
    a.movImm(2, 0x1234, .{});
    a.movImm(3, 0x80000000, .{});
    a.shl(4, sass.Src.reg(2), sass.Src.imm(4), .{}); // 0x12340
    a.shr(5, sass.Src.reg(3), sass.Src.imm(28), false, .{}); // 0x8
    a.shr(6, sass.Src.reg(3), sass.Src.imm(28), true, .{}); // 0xfffffff8
    a.stg(0, 4, 0, .bits32, .{});
    a.stg(0, 5, 4, .bits32, .{});
    a.stg(0, 6, 8, .bits32, .{});
    a.exit(.{ .stall = 1 });

    try r.run(code[0..a.dwords()], .{ .register_count = a.registerCount() });
    try std.testing.expectEqual(@as(u32, 0x12340), out.read(u32, 0));
    try std.testing.expectEqual(@as(u32, 0x8), out.read(u32, 1));
    try std.testing.expectEqual(@as(u32, 0xfffffff8), out.read(u32, 2));
}

/// Assemble a tiled single-precision matmul with the same interface as
/// `buildMatmul`. Each block stages a `tile` by `tile` square of A and of B in
/// shared memory, so every element loaded from global memory is used `tile`
/// times instead of once. `tile` must be a power of two.
///
/// Shared memory holds A's tile at offset 0 and B's tile right after it, so the
/// launch needs `2 * tile * tile * 4` bytes.
///
/// The control fields are left to `Assembler.schedule`, so this reads as plain
/// instructions with no scoreboard bookkeeping.
///
/// The staging code branches nowhere. A thread whose element is off the edge of
/// A or B still loads (from a clamped address, so the read stays inside the
/// matrix) and then selects a zero into the tile. Divergent branches around a
/// barrier let part of a warp reach the barrier at a different time from the
/// rest, which corrupts the staged tile, so the guards are predicated instead.
/// No thread leaves early either: only the final store is skipped.
///
/// Registers: R0:R1 and R2:R3 address the elements to stage, R4 accumulates,
/// R5 counts tiles, R6/R7 hold the staged values, R8/R9 are this thread's row
/// and column, R10..R12 hold M, N and K, R13/R14 are tid.x and tid.y,
/// R16..R19 are scratch, R20..R25 hold the base pointers, R26 counts the inner
/// loop, R27 is this thread's slot in the staged tile, R28/R29 are clamped
/// indices, R30/R31 walk the tile, R32/R33 hold the values read back out of it,
/// and R34 holds the tile count.
fn buildTiledMatmul(a: *sass.Assembler, comptime tile: u32) void {
    const Src = sass.Src;
    const zero = sass.zero;
    comptime std.debug.assert(std.math.isPowerOfTwo(tile));
    const tile_shift: u32 = @ctz(tile);
    const b_base: i32 = tile * tile * 4; // shared byte offset of B's tile

    a.s2r(13, sass.SR_TID_X, .{});
    a.s2r(14, sass.SR_TID_Y, .{});
    a.s2r(16, sass.SR_CTAID_X, .{});
    a.imad(9, Src.reg(16), Src.imm(tile), Src.reg(13), false, .{}); // col
    a.s2r(16, sass.SR_CTAID_Y, .{});
    a.imad(8, Src.reg(16), Src.imm(tile), Src.reg(14), false, .{}); // row

    a.ldc(10, .{ .offset = mm_params.rows }, sass.RZ, .bits32, .{});
    a.ldc(11, .{ .offset = mm_params.cols }, sass.RZ, .bits32, .{});
    a.ldc(12, .{ .offset = mm_params.depth }, sass.RZ, .bits32, .{});
    a.ldc(20, .{ .offset = mm_params.a_ptr }, sass.RZ, .bits64, .{});
    a.ldc(22, .{ .offset = mm_params.b_ptr }, sass.RZ, .bits64, .{});
    a.ldc(24, .{ .offset = mm_params.c_ptr }, sass.RZ, .bits64, .{});

    // This thread's slot in the staged tile, and the row of A and column of B
    // it walks once the tile is staged.
    a.imad(16, Src.reg(14), Src.imm(tile), Src.reg(13), false, .{}); // ty*tile + tx
    a.imad(27, Src.reg(16), Src.imm(4), zero, false, .{});
    a.imad(18, Src.reg(14), Src.imm(tile * 4), zero, false, .{});
    a.imad(19, Src.reg(13), Src.imm(4), zero, false, .{});
    a.iadd3(19, Src.reg(19), Src.imm(@intCast(b_base)), zero, .{});

    a.movImm(4, 0, .{}); // accumulator
    a.movImm(5, 0, .{}); // tile index
    // Tile count = (K + tile - 1) / tile.
    a.iadd3(34, Src.reg(12), Src.imm(tile - 1), zero, .{});
    a.shr(34, Src.reg(34), Src.imm(tile_shift), false, .{});

    const tile_loop = a.here();

    // Stage A[row][t*tile + tx]. P0 and P1 record whether the element exists;
    // the clamped indices keep the address inside A either way.
    a.imad(29, Src.reg(5), Src.imm(tile), Src.reg(13), false, .{});
    a.isetp(0, .lt, true, Src.reg(29), Src.reg(12), .{});
    a.isetp(1, .lt, true, Src.reg(8), Src.reg(10), .{});
    a.sel(28, Src.reg(29), zero, 0, false, .{});
    a.sel(17, Src.reg(8), zero, 1, false, .{});
    a.imad(16, Src.reg(17), Src.reg(12), Src.reg(28), false, .{});
    a.imadWide(0, Src.reg(16), Src.imm(4), Src.reg(20), false, .{});
    a.ldg(6, 0, 0, .bits32, .{});
    a.sel(6, Src.reg(6), zero, 0, false, .{});
    a.sel(6, Src.reg(6), zero, 1, false, .{});

    // Stage B[t*tile + ty][col] the same way.
    a.imad(29, Src.reg(5), Src.imm(tile), Src.reg(14), false, .{});
    a.isetp(0, .lt, true, Src.reg(29), Src.reg(12), .{});
    a.isetp(1, .lt, true, Src.reg(9), Src.reg(11), .{});
    a.sel(28, Src.reg(29), zero, 0, false, .{});
    a.sel(17, Src.reg(9), zero, 1, false, .{});
    a.imad(16, Src.reg(28), Src.reg(11), Src.reg(17), false, .{});
    a.imadWide(2, Src.reg(16), Src.imm(4), Src.reg(22), false, .{});
    a.ldg(7, 2, 0, .bits32, .{});
    a.sel(7, Src.reg(7), zero, 0, false, .{});
    a.sel(7, Src.reg(7), zero, 1, false, .{});

    a.sts(27, 6, 0, .bits32, .{});
    a.sts(27, 7, b_base, .bits32, .{});
    a.bar(.{});

    // Walk the staged tile: A along a row, B down a column.
    a.movImm(26, 0, .{});
    a.movReg(30, 18, .{});
    a.movReg(31, 19, .{});
    const inner_loop = a.here();
    a.lds(32, 30, 0, .bits32, .{});
    a.lds(33, 31, 0, .bits32, .{});
    // The pointer step must not overtake the load that still needs the address.
    a.iadd3(30, Src.reg(30), Src.imm(4), zero, .{});
    a.iadd3(31, Src.reg(31), Src.imm(tile * 4), zero, .{});
    a.ffma(4, Src.reg(32), Src.reg(33), Src.reg(4), .{});
    a.iadd3(26, Src.reg(26), Src.imm(1), zero, .{});
    a.isetp(0, .lt, true, Src.reg(26), Src.imm(tile), .{});
    a.bra(inner_loop, .{ .pred = 0 });

    // Hold every thread here until the block has finished reading the tile,
    // otherwise the next round would overwrite it too early.
    a.bar(.{});
    a.iadd3(5, Src.reg(5), Src.imm(1), zero, .{});
    a.isetp(0, .lt, true, Src.reg(5), Src.reg(34), .{});
    a.bra(tile_loop, .{ .pred = 0 });

    // Past the last barrier, so the threads outside C can leave.
    a.isetp(0, .ge, true, Src.reg(8), Src.reg(10), .{});
    const row_outside = a.braForward(.{ .pred = 0 });
    a.isetp(0, .ge, true, Src.reg(9), Src.reg(11), .{});
    const col_outside = a.braForward(.{ .pred = 0 });
    a.imad(16, Src.reg(8), Src.reg(11), Src.reg(9), false, .{});
    a.imadWide(2, Src.reg(16), Src.imm(4), Src.reg(24), false, .{});
    a.stg(2, 4, 0, .bits32, .{});
    a.patchBranch(row_outside, a.here());
    a.patchBranch(col_outside, a.here());
    a.exit(.{});
}
test "live: a tiled FP32 matmul kernel matches a CPU reference" {
    var r = try Runner.init();
    defer r.deinit();

    const rows = 37;
    const cols = 29;
    const depth = 23;
    const tile = 16;

    const a_buf = try r.alloc(.system, rows * depth * 4);
    const b_buf = try r.alloc(.system, depth * cols * 4);
    const c_buf = try r.alloc(.system, rows * cols * 4);
    const params = try r.alloc(.system, mm_params.size);

    const av = a_buf.slice(f32);
    for (0..rows * depth) |i| av[i] = @floatFromInt(@as(i32, @intCast(i % 7)) - 3);
    const bv = b_buf.slice(f32);
    for (0..depth * cols) |i| bv[i] = @floatFromInt(@as(i32, @intCast(i % 5)) - 2);

    const p = params.slice(u32);
    p[0] = @truncate(a_buf.va);
    p[1] = @intCast(a_buf.va >> 32);
    p[2] = @truncate(b_buf.va);
    p[3] = @intCast(b_buf.va >> 32);
    p[4] = @truncate(c_buf.va);
    p[5] = @intCast(c_buf.va >> 32);
    p[6] = rows;
    p[7] = cols;
    p[8] = depth;

    var code: [1024]u32 = undefined;
    var deps: [256]sass.Dep = @splat(.{});
    var a = sass.Assembler{ .code = &code, .deps = &deps };
    buildTiledMatmul(&a, tile);
    try a.schedule();

    try r.run(code[0..a.dwords()], .{
        .register_count = a.registerCount(),
        .grid = .{ (cols + tile - 1) / tile, (rows + tile - 1) / tile, 1 },
        .block = .{ tile, tile, 1 },
        .cbuf0_va = params.va,
        .cbuf0_size = mm_params.size,
        .shared_mem_bytes = 2 * tile * tile * 4,
    });
    for (0..rows) |i| {
        for (0..cols) |j| {
            var want: f32 = 0;
            for (0..depth) |k| want += av[i * depth + k] * bv[k * cols + j];
            try std.testing.expectEqual(want, c_buf.read(f32, i * cols + j));
        }
    }
}

test "live: a Runner reuses its channel for different kernels" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);

    // The second launch reuses the program address, so it only sees the new
    // code if the dispatch invalidated the instruction cache.
    for ([_]u8{ 1, 2, 3, 8 }) |n| {
        var code: [512]u32 = undefined;
        var a = sass.Assembler{ .code = &code };
        a.movImm(0, @truncate(out.va), .{});
        a.movImm(1, @intCast(out.va >> 32), .{});
        a.movImm(2, 0, .{});
        var i: u8 = 0;
        while (i < n) : (i += 1) a.iadd3(2, sass.Src.reg(2), sass.Src.imm(1), sass.zero, .{});
        a.stg(0, 2, 0, .bits32, .{});
        a.exit(.{ .stall = 1 });
        out.slice(u32)[0] = 0;
        try r.run(code[0..a.dwords()], .{ .register_count = a.registerCount() });
        try std.testing.expectEqual(@as(u32, n), out.read(u32, 0));
    }
}

// ---------------------------------------------------------------------------
// Instruction latency probes. Each probe writes a known value with one
// instruction and reads it back with the next, so a stall that is too short
// shows up as the value the destination held before. `measureLatency` searches
// for the smallest stall that reads the new value, which is the result latency
// the scheduler has to leave. The table in sass.zig comes from these numbers,
// and the test at the end asserts the table still covers the silicon.
// ---------------------------------------------------------------------------

const probe_good: u32 = 0x1234;
const probe_stale: u32 = 0x9999;
const probe_stale_hi: u32 = 0x8888;
const probe_float_bits: u32 = 0x40000000; // 2.0f

/// The producer-to-consumer pairs the harness times.
const Probe = enum {
    mov_to_alu,
    iadd3_to_alu,
    imad_to_alu,
    lop3_to_alu,
    shf_to_alu,
    sel_to_alu,
    fadd_to_alu,
    fmul_to_alu,
    ffma_to_alu,
    imadwide_lo_to_alu,
    imadwide_hi_to_alu,
    imadwide_to_ldg_address,
    imadwide_hi_to_imad,
    imadwide_hi_to_stg_data,
    imad_to_imadwide,
    sel_to_imad,
    iadd3_to_lds_address,
    iadd3_to_sts_address,
    sel_to_sts_data,
    iadd3_to_stg_data,
    ffma_to_stg_data,
    iadd3_to_isetp,
    iadd3_to_imad,
    iadd3_to_sel,
    iadd3_to_mov,
    iadd3_to_shf,
    iadd3_to_lop3,
    ffma_to_ffma,
    mov_to_imad,
    imad_to_mov,
    sel_to_isetp,
    shf_to_lop3,
    isetp_to_sel,
    isetp_to_guard,
    isetp_to_branch,

    fn expected(self: Probe) u32 {
        return switch (self) {
            .imadwide_hi_to_alu, .imadwide_hi_to_imad, .imadwide_hi_to_stg_data => 0,
            .fadd_to_alu, .fmul_to_alu, .ffma_to_alu, .ffma_to_stg_data, .ffma_to_ffma => probe_float_bits,
            else => probe_good,
        };
    }

    /// Shared memory the probe needs, in bytes.
    fn sharedBytes(self: Probe) u32 {
        return switch (self) {
            .iadd3_to_lds_address, .iadd3_to_sts_address, .sel_to_sts_data => 256,
            else => 0,
        };
    }
};

/// Every thread runs the same probe and stores its own answer.
///
/// One warp, deliberately. At 256 threads two of the probes below measure a
/// cycle less than they do here, because while one warp waits the scheduler
/// issues from another and a stall that is one cycle short never shows. That
/// makes a high-occupancy measurement an average rather than a floor, and a
/// scheduler needs the floor. Credit to a vulcan session for the hypothesis.
const probe_threads = 32;

fn buildProbe(a: *sass.Assembler, p: Probe, stall: u4, out_va: u64, in_va: u64) void {
    const Src = sass.Src;
    const zero = sass.zero;
    const st = sass.Control{ .stall = stall };
    a.movImm(22, @truncate(out_va), .{});
    a.movImm(23, @intCast(out_va >> 32), .{});
    a.s2r(24, sass.SR_TID_X, .{ .wr_barrier = 5 });
    a.imadWide(20, Src.reg(24), Src.imm(4), Src.reg(22), false, .{ .wait_mask = 0b100000 });

    switch (p) {
        .mov_to_alu, .iadd3_to_alu, .imad_to_alu, .lop3_to_alu, .shf_to_alu, .sel_to_alu => {
            a.movImm(2, probe_good, .{});
            a.movImm(3, probe_stale, .{});
            a.movImm(5, 7, .{});
            a.isetp(0, .eq, true, Src.reg(5), Src.imm(7), .{}); // P0 true, for SEL
            switch (p) {
                .mov_to_alu => a.movReg(3, 2, st),
                .iadd3_to_alu => a.iadd3(3, Src.reg(2), Src.imm(0), zero, st),
                .imad_to_alu => a.imad(3, Src.reg(2), Src.imm(1), zero, false, st),
                .lop3_to_alu => a.lop3(3, Src.reg(2), zero, zero, 0xf0, st), // dst = a
                .shf_to_alu => a.shl(3, Src.reg(2), Src.imm(0), st),
                .sel_to_alu => a.sel(3, Src.reg(2), zero, 0, false, st),
                else => unreachable,
            }
            a.iadd3(4, Src.reg(3), Src.imm(0), zero, .{});
            a.stg(20, 4, 0, .bits32, .{});
        },
        .fadd_to_alu, .fmul_to_alu, .ffma_to_alu => {
            a.mov(2, Src.float(2.0), .{});
            a.mov(10, Src.float(0.0), .{});
            a.mov(11, Src.float(1.0), .{});
            a.mov(3, Src.float(9.0), .{}); // the stale value
            switch (p) {
                .fadd_to_alu => a.fadd(3, Src.reg(2), Src.reg(10), st),
                .fmul_to_alu => a.fmul(3, Src.reg(2), Src.reg(11), st),
                .ffma_to_alu => a.ffma(3, Src.reg(2), Src.reg(11), Src.reg(10), st),
                else => unreachable,
            }
            a.fadd(4, Src.reg(3), Src.reg(10), .{});
            a.stg(20, 4, 0, .bits32, .{});
        },
        .imadwide_lo_to_alu, .imadwide_hi_to_alu => {
            a.movImm(2, probe_good, .{});
            a.movImm(6, probe_stale, .{});
            a.movImm(7, probe_stale_hi, .{});
            a.imadWide(6, Src.reg(2), Src.imm(1), zero, false, st);
            const half: u8 = if (p == .imadwide_lo_to_alu) 6 else 7;
            a.iadd3(4, Src.reg(half), Src.imm(0), zero, .{});
            a.stg(20, 4, 0, .bits32, .{});
        },
        .imadwide_hi_to_imad, .imadwide_hi_to_stg_data => {
            a.movImm(2, probe_good, .{});
            a.movImm(6, probe_stale, .{});
            a.movImm(7, probe_stale_hi, .{});
            a.imadWide(6, Src.reg(2), Src.imm(1), zero, false, st);
            if (p == .imadwide_hi_to_imad) {
                a.imad(4, Src.reg(7), Src.imm(1), zero, false, .{});
                a.stg(20, 4, 0, .bits32, .{});
            } else {
                a.stg(20, 7, 0, .bits32, .{});
            }
        },
        .imad_to_imadwide => {
            a.movImm(2, probe_good, .{});
            a.movImm(3, probe_stale, .{});
            a.imad(3, Src.reg(2), Src.imm(1), zero, false, st);
            a.imadWide(6, Src.reg(3), Src.imm(1), zero, false, .{});
            a.stg(20, 6, 0, .bits32, .{});
        },
        .sel_to_imad => {
            a.movImm(2, probe_good, .{});
            a.movImm(3, probe_stale, .{});
            a.movImm(5, 7, .{});
            a.isetp(0, .eq, true, Src.reg(5), Src.imm(7), .{});
            a.sel(3, Src.reg(2), zero, 0, false, st);
            a.imad(4, Src.reg(3), Src.imm(1), zero, false, .{});
            a.stg(20, 4, 0, .bits32, .{});
        },
        .imadwide_to_ldg_address => {
            // A stale address reads the decoy 64 bytes in, not the real element.
            a.movImm(6, @truncate(in_va + 64), .{});
            a.movImm(7, @intCast((in_va + 64) >> 32), .{});
            a.movImm(4, @truncate(in_va), .{});
            a.movImm(5, @intCast(in_va >> 32), .{});
            a.movImm(2, 0, .{});
            a.imadWide(6, Src.reg(2), Src.imm(1), Src.reg(4), false, st);
            a.ldg(8, 6, 0, .bits32, .{ .wr_barrier = 0 });
            a.stg(20, 8, 0, .bits32, .{ .wait_mask = 0b1 });
        },
        .iadd3_to_lds_address => {
            a.movImm(8, 0, .{});
            a.movImm(9, probe_good, .{});
            a.sts(8, 9, 0, .bits32, .{});
            a.movImm(10, probe_stale, .{});
            a.sts(8, 10, 64, .bits32, .{});
            a.bar(.{});
            a.movImm(11, 64, .{});
            a.iadd3(11, Src.reg(11), Src.imm(@bitCast(@as(i32, -64))), zero, st);
            a.lds(12, 11, 0, .bits32, .{ .wr_barrier = 0 });
            a.stg(20, 12, 0, .bits32, .{ .wait_mask = 0b1 });
        },
        .iadd3_to_sts_address => {
            a.movImm(8, 0, .{});
            a.movImm(13, probe_stale, .{});
            a.sts(8, 13, 0, .bits32, .{}); // shared[0] starts wrong
            a.bar(.{});
            a.movImm(9, probe_good, .{});
            a.movImm(11, 64, .{});
            a.iadd3(11, Src.reg(11), Src.imm(@bitCast(@as(i32, -64))), zero, st);
            a.sts(11, 9, 0, .bits32, .{}); // a stale address stores 64 bytes in
            a.bar(.{});
            a.lds(12, 8, 0, .bits32, .{ .wr_barrier = 0 });
            a.stg(20, 12, 0, .bits32, .{ .wait_mask = 0b1 });
        },
        .sel_to_sts_data => {
            a.movImm(2, probe_good, .{});
            a.movImm(3, probe_stale, .{});
            a.movImm(5, 7, .{});
            a.movImm(8, 0, .{});
            a.isetp(0, .eq, true, Src.reg(5), Src.imm(7), .{});
            a.sel(3, Src.reg(2), zero, 0, false, st);
            a.sts(8, 3, 0, .bits32, .{});
            a.bar(.{});
            a.lds(12, 8, 0, .bits32, .{ .wr_barrier = 0 });
            a.stg(20, 12, 0, .bits32, .{ .wait_mask = 0b1 });
        },
        .iadd3_to_stg_data => {
            a.movImm(2, probe_good, .{});
            a.movImm(3, probe_stale, .{});
            a.iadd3(3, Src.reg(2), Src.imm(0), zero, st);
            a.stg(20, 3, 0, .bits32, .{});
        },
        .ffma_to_stg_data => {
            a.mov(2, Src.float(2.0), .{});
            a.mov(10, Src.float(0.0), .{});
            a.mov(11, Src.float(1.0), .{});
            a.mov(3, Src.float(9.0), .{});
            a.ffma(3, Src.reg(2), Src.reg(11), Src.reg(10), st);
            a.stg(20, 3, 0, .bits32, .{});
        },
        .iadd3_to_isetp, .iadd3_to_imad, .iadd3_to_sel, .iadd3_to_mov, .iadd3_to_shf, .iadd3_to_lop3 => {
            // One producer, six different consumers of its result.
            a.movImm(2, probe_good, .{});
            a.movImm(3, probe_stale, .{});
            a.movImm(4, 0, .{});
            a.movImm(5, 7, .{});
            a.isetp(0, .eq, true, Src.reg(5), Src.imm(7), .{}); // P0 true, for SEL
            a.iadd3(3, Src.reg(2), Src.imm(0), zero, st);
            switch (p) {
                .iadd3_to_isetp => {
                    a.isetp(1, .eq, true, Src.reg(3), Src.imm(probe_good), .{});
                    a.sel(4, Src.reg(2), zero, 1, false, .{});
                },
                .iadd3_to_imad => a.imad(4, Src.reg(3), Src.imm(1), zero, false, .{}),
                .iadd3_to_sel => a.sel(4, Src.reg(3), zero, 0, false, .{}),
                .iadd3_to_mov => a.movReg(4, 3, .{}),
                .iadd3_to_shf => a.shl(4, Src.reg(3), Src.imm(0), .{}),
                .iadd3_to_lop3 => a.lop3(4, Src.reg(3), zero, zero, 0xf0, .{}),
                else => unreachable,
            }
            a.stg(20, 4, 0, .bits32, .{});
        },
        .mov_to_imad, .imad_to_mov, .sel_to_isetp, .shf_to_lop3 => {
            // Pairs the pipe model says are same-pipe, so 4 cycles should do.
            a.movImm(2, probe_good, .{});
            a.movImm(3, probe_stale, .{});
            a.movImm(4, 0, .{});
            a.movImm(5, 7, .{});
            a.isetp(0, .eq, true, Src.reg(5), Src.imm(7), .{});
            switch (p) {
                .mov_to_imad => {
                    a.movReg(3, 2, st);
                    a.imad(4, Src.reg(3), Src.imm(1), zero, false, .{});
                },
                .imad_to_mov => {
                    a.imad(3, Src.reg(2), Src.imm(1), zero, false, st);
                    a.movReg(4, 3, .{});
                },
                .sel_to_isetp => {
                    a.sel(3, Src.reg(2), zero, 0, false, st);
                    a.isetp(1, .eq, true, Src.reg(3), Src.imm(probe_good), .{});
                    a.sel(4, Src.reg(2), zero, 1, false, .{});
                },
                .shf_to_lop3 => {
                    a.shl(3, Src.reg(2), Src.imm(0), st);
                    a.lop3(4, Src.reg(3), zero, zero, 0xf0, .{});
                },
                else => unreachable,
            }
            a.stg(20, 4, 0, .bits32, .{});
        },
        .ffma_to_ffma => {
            a.mov(2, Src.float(2.0), .{});
            a.mov(10, Src.float(0.0), .{});
            a.mov(11, Src.float(1.0), .{});
            a.mov(3, Src.float(9.0), .{});
            a.ffma(3, Src.reg(2), Src.reg(11), Src.reg(10), st);
            a.ffma(4, Src.reg(3), Src.reg(11), Src.reg(10), .{});
            a.stg(20, 4, 0, .bits32, .{});
        },
        .isetp_to_sel, .isetp_to_guard => {
            a.movImm(2, probe_good, .{});
            a.movImm(4, 0, .{});
            a.movImm(5, 7, .{});
            a.isetp(0, .ne, true, Src.reg(5), Src.imm(7), .{}); // P0 false
            a.isetp(0, .eq, true, Src.reg(5), Src.imm(7), st); // P0 true
            if (p == .isetp_to_sel) {
                a.sel(4, Src.reg(2), zero, 0, false, .{});
            } else {
                a.movReg(4, 2, .{ .pred = 0 });
            }
            a.stg(20, 4, 0, .bits32, .{});
        },
        .isetp_to_branch => {
            a.movImm(4, 0, .{});
            a.movImm(5, 1, .{});
            a.isetp(0, .eq, true, Src.reg(5), Src.imm(1), .{}); // P0 true
            a.isetp(0, .ne, true, Src.reg(5), Src.imm(1), st); // P0 false
            const skip = a.braForward(.{ .pred = 0 }); // must fall through
            a.movImm(4, probe_good, .{});
            a.patchBranch(skip, a.here());
            a.stg(20, 4, 0, .bits32, .{});
        },
    }
    a.exit(.{ .stall = 1 });
}

fn runProbe(r: *Runner, p: Probe, stall: u4, out: Buffer, in: Buffer) !bool {
    var code: [512]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    buildProbe(&a, p, stall, out.va, in.va);
    @memset(out.slice(u32)[0..probe_threads], 0);
    try r.run(code[0..a.dwords()], .{
        .register_count = a.registerCount(),
        .block = .{ probe_threads, 1, 1 },
        .shared_mem_bytes = @max(p.sharedBytes(), 256),
    });
    // One thread reading a stale value is a failure for the whole probe.
    for (0..probe_threads) |i| {
        if (out.read(u32, i) != p.expected()) return false;
    }
    return true;
}

/// The smallest stall at which `p` reads the new value. A longer stall always
/// works, so the search halves the range each step.
fn measureLatency(r: *Runner, p: Probe, out: Buffer, in: Buffer) !u4 {
    if (!try runProbe(r, p, 15, out, in)) return error.ProbeNeverSettles;
    var lo: u4 = 1;
    var hi: u4 = 15;
    while (lo < hi) {
        const mid: u4 = lo + (hi - lo) / 2;
        if (try runProbe(r, p, mid, out, in)) hi = mid else lo = mid + 1;
    }
    return lo;
}

test "live: the scheduler's latency table still covers the silicon" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);
    const in = try r.alloc(.system, 0x1000);
    in.slice(u32)[0] = probe_good;
    in.slice(u32)[16] = probe_stale;

    var short = false;
    for (std.enums.values(Probe)) |p| {
        const measured = try measureLatency(&r, p, out, in);
        // The pipe model: a value read on the pipe that produced it costs
        // `same_pipe`, one read on the other compute pipe costs `cross_pipe`,
        // and a predicate costs more again when a guard or a branch reads it.
        const budget: u8 = switch (p) {
            .mov_to_alu,
            .imad_to_alu,
            .sel_to_alu,
            .imadwide_lo_to_alu,
            .imadwide_hi_to_alu,
            .iadd3_to_isetp,
            .iadd3_to_imad,
            .iadd3_to_sel,
            .iadd3_to_mov,
            => sass.Latency.cross_pipe,
            .isetp_to_sel => sass.Latency.pred,
            .isetp_to_guard, .isetp_to_branch => sass.Latency.pred_guard,
            else => sass.Latency.same_pipe,
        };
        if (@as(u8, measured) > budget) {
            std.debug.print("probe {t} = {d} (table allows {d})\n", .{ p, measured, budget });
            short = true;
        }
    }
    if (short) return error.LatencyTableTooShort;
}

test "live: BAR.SYNC synchronises every warp of a block" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);
    const threads = 256; // eight warps, so a barrier that only arrives shows up

    var code: [256]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.s2r(2, sass.SR_TID_X, .{ .wr_barrier = 0 });
    a.imad(3, sass.Src.reg(2), sass.Src.imm(4), sass.zero, false, .{ .wait_mask = 0b01 });
    a.sts(3, 2, 0, .bits32, .{});
    a.bar(.{});
    a.movImm(7, threads - 1, .{});
    a.imad(4, sass.Src.reg(2), sass.Src.imm(0xffffffff), sass.Src.reg(7), true, .{});
    a.imad(5, sass.Src.reg(4), sass.Src.imm(4), sass.zero, false, .{});
    a.lds(6, 5, 0, .bits32, .{ .wr_barrier = 0 });
    a.movImm(8, @truncate(out.va), .{});
    a.movImm(9, @intCast(out.va >> 32), .{});
    a.imadWide(0, sass.Src.reg(2), sass.Src.imm(4), sass.Src.reg(8), false, .{});
    a.stg(0, 6, 0, .bits32, .{ .wait_mask = 0b01 });
    a.exit(.{ .stall = 1 });

    // Run it a few times: a barrier that does not block fails intermittently.
    for (0..8) |_| {
        @memset(out.slice(u32)[0..threads], 0xff);
        try r.run(code[0..a.dwords()], .{
            .register_count = a.registerCount(),
            .block = .{ threads, 1, 1 },
            .shared_mem_bytes = threads * 4,
        });
        for (0..threads) |i| {
            try std.testing.expectEqual(@as(u32, threads - 1 - @as(u32, @intCast(i))), out.read(u32, i));
        }
    }
}

test "live: a scheduled shared-memory dot product loop matches its reference" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);
    const Src = sass.Src;
    const steps = 16;
    const threads = 256;

    var code: [512]u32 = undefined;
    var deps: [128]sass.Dep = @splat(.{});
    var a = sass.Assembler{ .code = &code, .deps = &deps };
    // Fill shared memory: slot i holds i + 1 as a float, in both halves.
    a.s2r(2, sass.SR_TID_X, .{});
    a.imad(3, Src.reg(2), Src.imm(4), sass.zero, false, .{});
    // Slot i holds 1.0 for an even i and 2.0 for an odd one, so a pointer step
    // that lands wrong changes the answer.
    a.lop3(5, Src.reg(2), Src.imm(1), sass.zero, 0xc0, .{});
    a.isetp(0, .eq, true, Src.reg(5), Src.imm(0), .{});
    a.mov(6, Src.float(1.0), .{});
    a.mov(7, Src.float(2.0), .{});
    a.sel(4, Src.reg(6), Src.reg(7), 0, false, .{});
    a.sts(3, 4, 0, .bits32, .{});
    a.bar(.{});
    // The loop under test: two shared loads, two pointer steps, one multiply-add.
    a.movImm(30, 0, .{});
    a.movImm(31, 0, .{});
    a.movImm(26, 0, .{});
    a.mov(20, Src.float(0.0), .{});
    const loop = a.here();
    a.lds(10, 30, 0, .bits32, .{});
    a.lds(11, 31, 0, .bits32, .{});
    a.iadd3(30, Src.reg(30), Src.imm(4), sass.zero, .{});
    a.iadd3(31, Src.reg(31), Src.imm(4), sass.zero, .{});
    a.ffma(20, Src.reg(10), Src.reg(11), Src.reg(20), .{});
    a.iadd3(26, Src.reg(26), Src.imm(1), sass.zero, .{});
    a.isetp(0, .lt, true, Src.reg(26), Src.imm(steps), .{});
    a.bra(loop, .{ .pred = 0 });
    a.movImm(6, @truncate(out.va), .{});
    a.movImm(7, @intCast(out.va >> 32), .{});
    a.imadWide(0, Src.reg(2), Src.imm(4), Src.reg(6), false, .{});
    a.stg(0, 20, 0, .bits32, .{});
    a.exit(.{});
    try a.schedule();

    var want: f32 = 0;
    for (0..steps) |i| {
        const v: f32 = if (i % 2 == 0) 1.0 else 2.0;
        want += v * v;
    }
    for (0..8) |_| {
        @memset(out.slice(u32)[0..threads], 0);
        try r.run(code[0..a.dwords()], .{
            .register_count = a.registerCount(),
            .block = .{ threads, 1, 1 },
            .shared_mem_bytes = threads * 4,
        });
        for (0..threads) |i| try std.testing.expectEqual(want, out.read(f32, i));
    }
}

test "live: LOP3 applies its truth table to the three operands in order" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);
    const Src = sass.Src;

    var code: [512]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.movImm(0, @truncate(out.va), .{});
    a.movImm(1, @intCast(out.va >> 32), .{});
    a.movImm(2, 0xf0f0f0f0, .{});
    a.movImm(3, 0xcccccccc, .{});
    a.movImm(4, 0xaaaaaaaa, .{});
    const luts = [_]u8{ 0xc0, 0x88, 0xa0, 0xf0, 0xcc, 0xaa, 0xfc, 0x3c };
    for (luts, 0..) |lut, i| {
        a.lop3(10, Src.reg(2), Src.reg(3), Src.reg(4), lut, .{});
        a.stg(0, 10, @intCast(i * 4), .bits32, .{});
    }
    // The same with an immediate in the second operand slot.
    a.lop3(11, Src.reg(2), Src.imm(0xcccccccc), sass.zero, 0xc0, .{});
    a.stg(0, 11, 32, .bits32, .{});
    a.exit(.{ .stall = 1 });
    try r.run(code[0..a.dwords()], .{ .register_count = a.registerCount() });

    // With a = 0xf0, b = 0xcc and c = 0xaa as the truth-table inputs, the LUT
    // value is the function itself, so each result is the LUT byte repeated.
    for (luts, 0..) |lut, i| {
        const want = @as(u32, lut) * 0x01010101;
        try std.testing.expectEqual(want, out.read(u32, i));
    }
    try std.testing.expectEqual(@as(u32, 0xc0c0c0c0), out.read(u32, 8));
}

test "live: every register the count covers is writable" {
    const Src = sass.Src;
    // The hardware keeps the top two GPRs of the allocation, so `registerCount`
    // has to leave room for them. Without that, a write to the highest register
    // the kernel uses is silently dropped.
    for ([_]u8{ 8, 16, 23, 24, 29, 30, 31, 40, 62, 63 }) |n| {
        var r = try Runner.init();
        defer r.deinit();
        const out = try r.alloc(.system, 0x1000);
        var code: [512]u32 = undefined;
        var a = sass.Assembler{ .code = &code };
        a.movImm(0, @truncate(out.va), .{});
        a.movImm(1, @intCast(out.va >> 32), .{});
        a.movImm(n, 0, .{});
        var i: u8 = 0;
        while (i < 4) : (i += 1) a.iadd3(n, Src.reg(n), Src.imm(4), sass.zero, .{});
        a.stg(0, n, 0, .bits32, .{});
        a.exit(.{ .stall = 1 });
        const rc = a.registerCount();
        out.slice(u32)[0] = 0xdead;
        try r.run(code[0..a.dwords()], .{ .register_count = rc });
        try std.testing.expectEqual(@as(u32, 16), out.read(u32, 0));
    }
}

test "live: each block gets its own shared memory" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);
    const blocks = 6;
    const threads = 256;
    const slots = 512; // 2 KB of shared memory, as the tiled matmul uses
    const Src = sass.Src;

    var code: [512]u32 = undefined;
    var deps: [128]sass.Dep = @splat(.{});
    var a = sass.Assembler{ .code = &code, .deps = &deps };
    a.s2r(2, sass.SR_TID_X, .{});
    a.s2r(3, sass.SR_CTAID_X, .{});
    // Every thread fills two slots with this block's index.
    a.imad(4, Src.reg(2), Src.imm(4), sass.zero, false, .{});
    a.sts(4, 3, 0, .bits32, .{});
    a.sts(4, 3, threads * 4, .bits32, .{});
    a.bar(.{});
    // Then every thread checks two slots of its own block's window.
    a.lds(5, 4, 0, .bits32, .{});
    a.lds(6, 4, threads * 4, .bits32, .{});
    a.iadd3(7, Src.reg(5), Src.reg(6), sass.zero, .{});
    a.imad(8, Src.reg(3), Src.imm(2), sass.zero, false, .{});
    a.iadd3(9, Src.reg(7), Src.reg(8).negated(), sass.zero, .{}); // zero when right
    // Accumulate any mismatch into out[block] with a store per thread.
    a.movImm(10, @truncate(out.va), .{});
    a.movImm(11, @intCast(out.va >> 32), .{});
    a.isetp(0, .ne, true, Src.reg(9), Src.imm(0), .{});
    a.imadWide(12, Src.reg(3), Src.imm(4), Src.reg(10), false, .{});
    a.stg(12, 9, 0, .bits32, .{ .pred = 0 });
    a.exit(.{});
    try a.schedule();

    _ = slots;
    for (0..8) |_| {
        @memset(out.slice(u32)[0..blocks], 0);
        try r.run(code[0..a.dwords()], .{
            .register_count = a.registerCount(),
            .grid = .{ blocks, 1, 1 },
            .block = .{ threads, 1, 1 },
            .shared_mem_bytes = 2048,
        });
        for (0..blocks) |b| try std.testing.expectEqual(@as(u32, 0), out.read(u32, b));
    }
}

test "live: a loop that stages through shared memory keeps its blocks in step" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);
    const threads = 256;
    const rounds = 4;
    const Src = sass.Src;

    var code: [512]u32 = undefined;
    var deps: [128]sass.Dep = @splat(.{});
    var a = sass.Assembler{ .code = &code, .deps = &deps };
    a.s2r(2, sass.SR_TID_X, .{}); // tid
    a.imad(3, Src.reg(2), Src.imm(4), sass.zero, false, .{}); // this thread's slot
    a.movImm(4, threads - 1, .{});
    a.imad(5, Src.reg(2), Src.imm(0xffffffff), Src.reg(4), true, .{}); // partner index
    a.imad(6, Src.reg(5), Src.imm(4), sass.zero, false, .{}); // partner slot
    a.movImm(7, 0, .{}); // round
    a.movImm(8, 0, .{}); // mismatches

    const loop = a.here();
    // Each round writes a value only this round produces, so a thread that runs
    // ahead of the block reads the wrong one.
    a.imad(9, Src.reg(7), Src.imm(1000), Src.reg(2), false, .{});
    a.sts(3, 9, 0, .bits32, .{});
    a.bar(.{});
    a.lds(10, 6, 0, .bits32, .{});
    a.imad(11, Src.reg(7), Src.imm(1000), Src.reg(5), false, .{});
    a.iadd3(12, Src.reg(10), Src.reg(11).negated(), sass.zero, .{});
    a.isetp(0, .ne, true, Src.reg(12), Src.imm(0), .{});
    a.iadd3(8, Src.reg(8), Src.imm(1), sass.zero, .{ .pred = 0 });
    a.bar(.{});
    a.iadd3(7, Src.reg(7), Src.imm(1), sass.zero, .{});
    a.isetp(0, .lt, true, Src.reg(7), Src.imm(rounds), .{});
    a.bra(loop, .{ .pred = 0 });

    a.movImm(14, @truncate(out.va), .{});
    a.movImm(15, @intCast(out.va >> 32), .{});
    a.imadWide(16, Src.reg(2), Src.imm(4), Src.reg(14), false, .{});
    a.stg(16, 8, 0, .bits32, .{});
    a.exit(.{});
    try a.schedule();

    for (0..8) |_| {
        @memset(out.slice(u32)[0..threads], 0xff);
        try r.run(code[0..a.dwords()], .{
            .register_count = a.registerCount(),
            .block = .{ threads, 1, 1 },
            .shared_mem_bytes = threads * 4,
        });
        for (0..threads) |i| try std.testing.expectEqual(@as(u32, 0), out.read(u32, i));
    }
}

test "inline upload encodes a non-incrementing payload after the launch" {
    var buf: [64]u32 = undefined;
    var s = Stream{ .buf = &buf };
    const payload = [_]u32{ 0xaa, 0xbb, 0xcc };
    s.uploadInline(0x1234_5000, &payload);

    // Five contiguous geometry methods, then the launch, then the payload.
    try std.testing.expectEqual(sdk.gpfifo.methodHeader(LINE_LENGTH_IN, SUBCH, 5), buf[0]);
    try std.testing.expectEqual(@as(u32, 12), buf[1]); // bytes
    try std.testing.expectEqual(@as(u32, 1), buf[2]); // one line
    try std.testing.expectEqual(@as(u32, 0), buf[3]); // destination high
    try std.testing.expectEqual(@as(u32, 0x1234_5000), buf[4]); // destination low
    try std.testing.expectEqual(sdk.gpfifo.methodHeader(I2M_LAUNCH_DMA, SUBCH, 1), buf[6]);
    try std.testing.expectEqual(I2M_LAUNCH_PITCH_FLUSH, buf[7]);
    // The payload goes to one method address, so the header is non-incrementing.
    try std.testing.expectEqual(sdk.gpfifo.methodHeaderNonInc(LOAD_INLINE_DATA, SUBCH, 3), buf[8]);
    try std.testing.expectEqual(@as(u32, 0xaa), buf[9]);
    try std.testing.expectEqual(@as(u32, 0xcc), buf[11]);
    try std.testing.expectEqual(@as(u32, 12), s.dwords());
}

test "inline upload splits a payload too long for one method header" {
    const long = sdk.gpfifo.MAX_METHOD_COUNT + 5;
    const payload = try std.testing.allocator.alloc(u32, long);
    defer std.testing.allocator.free(payload);
    for (payload, 0..) |*w, i| w.* = @intCast(i);
    const buf = try std.testing.allocator.alloc(u32, long + 16);
    defer std.testing.allocator.free(buf);

    var s = Stream{ .buf = buf };
    s.uploadInline(0x2000, payload);
    // The first chunk fills a header, the second carries the remainder.
    try std.testing.expectEqual(
        sdk.gpfifo.methodHeaderNonInc(LOAD_INLINE_DATA, SUBCH, sdk.gpfifo.MAX_METHOD_COUNT),
        buf[8],
    );
    const second = 9 + sdk.gpfifo.MAX_METHOD_COUNT;
    try std.testing.expectEqual(sdk.gpfifo.methodHeaderNonInc(LOAD_INLINE_DATA, SUBCH, 5), buf[second]);
    try std.testing.expectEqual(@as(u32, sdk.gpfifo.MAX_METHOD_COUNT), buf[second + 1]);
    try std.testing.expectEqual(@as(u32, long - 1), buf[second + 5]);
}

test "live: an inline upload lands in memory without a source buffer" {
    var r = try Runner.init();
    defer r.deinit();
    const dst = try r.alloc(.system, 0x10000);

    // A short payload, the size NVIDIA's driver would send this way.
    var small: [256]u32 = undefined;
    for (&small, 0..) |*w, i| w.* = @intCast(0xc0de_0000 + i);
    try r.uploadInline(dst.va, &small);
    for (small, 0..) |want, i| try std.testing.expectEqual(want, dst.read(u32, i));

    // A payload longer than one method header can carry, to cover the chunking.
    const long = try std.testing.allocator.alloc(u32, sdk.gpfifo.MAX_METHOD_COUNT + 37);
    defer std.testing.allocator.free(long);
    for (long, 0..) |*w, i| w.* = @intCast(0x5a5a_0000 +% i);
    @memset(dst.slice(u32)[0..long.len], 0);
    try r.uploadInline(dst.va, long);
    for (long, 0..) |want, i| try std.testing.expectEqual(want, dst.read(u32, i));
}

test "live: a linear copy on the copy engine moves a buffer" {
    var r = try Runner.init();
    defer r.deinit();
    const src = try r.alloc(.system, 0x4000);
    const dst = try r.alloc(.system, 0x4000);
    const sv = src.slice(u32);
    for (sv[0..1024], 0..) |*w, i| w.* = @intCast(0x1234_0000 + i);

    try r.copyLinear(dst.va, src.va, 1024 * 4);
    for (0..1024) |i| try std.testing.expectEqual(sv[i], dst.read(u32, i));
}

/// Time `iters` transfers of `words` dwords through both paths. `awake` is the
/// clock that counts forward without jumping; on Linux it is CLOCK_MONOTONIC.
fn timeBothPaths(r: *Runner, dst: Buffer, src: Buffer, payload: []const u32, words: usize) !struct { inline_ns: u64, ce_ns: u64 } {
    const iters = 200;
    const bytes: u32 = @intCast(words * 4);
    try r.uploadInline(dst.va, payload[0..words]); // warm the path
    var start: std.Io.Timestamp = .now(std.testing.io, .awake);
    for (0..iters) |_| try r.uploadInline(dst.va, payload[0..words]);
    const inline_ns: u64 = @intCast(@divTrunc(start.durationTo(.now(std.testing.io, .awake)).nanoseconds, iters));

    try r.copyLinear(dst.va, src.va, bytes);
    start = .now(std.testing.io, .awake);
    for (0..iters) |_| try r.copyLinear(dst.va, src.va, bytes);
    const ce_ns: u64 = @intCast(@divTrunc(start.durationTo(.now(std.testing.io, .awake)).nanoseconds, iters));
    return .{ .inline_ns = inline_ns, .ce_ns = ce_ns };
}

test "live: the inline path costs more per byte than the copy engine" {
    var r = try Runner.init();
    defer r.deinit();
    const big = 64 * 1024;
    const small = 4 * 1024;
    const src = try r.alloc(.system, big);
    const dst = try r.alloc(.system, big);
    const payload = try std.testing.allocator.alloc(u32, big / 4);
    defer std.testing.allocator.free(payload);
    for (payload, 0..) |*w, i| w.* = @intCast(0x600d_0000 +% i);
    @memcpy(src.slice(u32)[0 .. big / 4], payload);

    const at_small = try timeBothPaths(&r, dst, src, payload, small / 4);
    // Both paths have to deliver the same bytes whichever one is faster.
    for (0..small / 4) |i| try std.testing.expectEqual(payload[i], dst.read(u32, i));
    const at_big = try timeBothPaths(&r, dst, src, payload, big / 4);
    for (0..big / 4) |i| try std.testing.expectEqual(payload[i], dst.read(u32, i));

    // The payload rides in the pushbuffer, so the inline path pays for every
    // byte twice: once for the host to fetch the pushbuffer and once to write
    // the destination. Its cost therefore climbs faster with size, which is why
    // a crossover exists at all and why `inline_upload_limit_bytes` is finite.
    const inline_growth = at_big.inline_ns - at_small.inline_ns;
    const ce_growth = at_big.ce_ns - at_small.ce_ns;
    if (inline_growth <= ce_growth) {
        std.debug.print(
            "inline grew {d} ns over {d} bytes, the copy engine {d} ns\n",
            .{ inline_growth, big - small, ce_growth },
        );
        return error.CrossoverGone;
    }
    // And the constant has to sit where the measurement puts it.
    try std.testing.expect(preferInlineUpload(small));
    try std.testing.expect(!preferInlineUpload(big));
}

test "live: one submission carries a batch of dispatches" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);
    const params = try r.alloc(.system, 0x4000);
    const batch = 16;
    const stride = 256; // constant banks bind on a 64-byte boundary
    const Src = sass.Src;

    // Every dispatch runs the same program but binds its own constant bank, so
    // each one writes a different slot and they never touch the same memory.
    for (0..batch) |i| params.slice(u32)[i * stride / 4] = @intCast(i);

    var code: [256]u32 = undefined;
    var deps: [64]sass.Dep = @splat(.{});
    var a = sass.Assembler{ .code = &code, .deps = &deps };
    a.ldc(2, .{ .offset = 0 }, sass.RZ, .bits32, .{});
    a.movImm(4, @truncate(out.va), .{});
    a.movImm(5, @intCast(out.va >> 32), .{});
    a.imadWide(0, Src.reg(2), Src.imm(4), Src.reg(4), false, .{});
    a.iadd3(6, Src.reg(2), Src.imm(0x100), sass.zero, .{});
    a.stg(0, 6, 0, .bits32, .{});
    a.exit(.{});
    try a.schedule();

    var grids: [batch]Grid = undefined;
    for (&grids, 0..) |*g, i| g.* = .{
        .register_count = a.registerCount(),
        .cbuf0_va = params.va + i * stride,
        .cbuf0_size = 64,
    };
    @memset(out.slice(u32)[0..batch], 0);
    try r.runBatch(code[0..a.dwords()], &grids);
    for (0..batch) |i| {
        try std.testing.expectEqual(@as(u32, 0x100 + @as(u32, @intCast(i))), out.read(u32, i));
    }
}

/// Build the batching benchmark's kernel: read a slot index out of constant
/// bank 0 and write a marker to that slot. Every dispatch runs this, and its
/// bound constant bank decides which slot it touches.
fn buildSlotWriter(a: *sass.Assembler, out_va: u64) void {
    const Src = sass.Src;
    a.ldc(2, .{ .offset = 0 }, sass.RZ, .bits32, .{});
    a.movImm(4, @truncate(out_va), .{});
    a.movImm(5, @intCast(out_va >> 32), .{});
    a.imadWide(0, Src.reg(2), Src.imm(4), Src.reg(4), false, .{});
    a.iadd3(6, Src.reg(2), Src.imm(0x100), sass.zero, .{});
    a.stg(0, 6, 0, .bits32, .{});
    a.exit(.{});
}

test "live: batching dispatches costs far less per dispatch than submitting each" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);
    const params = try r.alloc(.system, 0x4000);
    const batch = 16;
    const stride = 256;
    for (0..batch) |i| params.slice(u32)[i * stride / 4] = @intCast(i);

    var code: [256]u32 = undefined;
    var deps: [64]sass.Dep = @splat(.{});
    var a = sass.Assembler{ .code = &code, .deps = &deps };
    buildSlotWriter(&a, out.va);
    try a.schedule();
    const program = code[0..a.dwords()];

    var grids: [batch]Grid = undefined;
    for (&grids, 0..) |*g, i| g.* = .{
        .register_count = a.registerCount(),
        .cbuf0_va = params.va + i * stride,
        .cbuf0_size = 64,
    };

    const iters = 100;
    try r.runBatch(program, &grids); // warm the path
    var start: std.Io.Timestamp = .now(std.testing.io, .awake);
    for (0..iters) |_| try r.runBatch(program, &grids);
    const batched_per: u64 = @intCast(@divTrunc(start.durationTo(.now(std.testing.io, .awake)).nanoseconds, iters * batch));

    start = .now(std.testing.io, .awake);
    for (0..iters) |_| {
        for (0..batch) |i| try r.runBatch(program, grids[i .. i + 1]);
    }
    const separate_per: u64 = @intCast(@divTrunc(start.durationTo(.now(std.testing.io, .awake)).nanoseconds, iters * batch));

    // The last round left every slot written, whichever way it was submitted.
    for (0..batch) |i| {
        try std.testing.expectEqual(@as(u32, 0x100 + @as(u32, @intCast(i))), out.read(u32, i));
    }
    // Measured at about a third of the cost; fail well before that regresses to
    // parity, so ordinary run-to-run noise cannot trip this.
    if (batched_per * 5 >= separate_per * 3) {
        std.debug.print(
            "batched {d} ns per dispatch against {d} ns submitted one at a time\n",
            .{ batched_per, separate_per },
        );
        return error.BatchingNoLongerPays;
    }
}

test "live: the FP32 multiply-add pipes reach their expected rate" {
    var r = try Runner.init();
    defer r.deinit();
    const Src = sass.Src;
    const threads = 256;
    const accs = 8; // independent chains, to cover the FFMA result latency
    const rounds = 8; // unrolled groups per loop iteration
    const trips = 1024;
    const max_blocks = 2048;
    const out = try r.alloc(.system, max_blocks * threads * 4);

    var code: [2048]u32 = undefined;
    var deps: [512]sass.Dep = @splat(.{});
    var a = sass.Assembler{ .code = &code, .deps = &deps };
    a.s2r(20, sass.SR_CTAID_X, .{});
    a.s2r(21, sass.SR_TID_X, .{});
    a.mov(10, Src.float(1.0), .{});
    for (0..accs) |i| a.mov(@intCast(2 + i), Src.float(0.0), .{});
    a.movImm(12, 0, .{});
    const loop = a.here();
    for (0..rounds) |_| {
        for (0..accs) |i| {
            const acc: u8 = @intCast(2 + i);
            a.ffma(acc, Src.reg(acc), Src.reg(10), Src.reg(10), .{});
        }
    }
    a.iadd3(12, Src.reg(12), Src.imm(1), sass.zero, .{});
    a.isetp(0, .lt, true, Src.reg(12), Src.imm(trips), .{});
    a.bra(loop, .{ .pred = 0 });
    // One value per thread, so nothing can be optimised away by accident.
    a.imad(22, Src.reg(20), Src.imm(threads), Src.reg(21), false, .{});
    a.movImm(16, @truncate(out.va), .{});
    a.movImm(17, @intCast(out.va >> 32), .{});
    a.imadWide(0, Src.reg(22), Src.imm(4), Src.reg(16), false, .{});
    a.stg(0, 2, 0, .bits32, .{});
    a.exit(.{});
    try a.schedule();

    // Nothing here touches memory until the end, so this is the rate the
    // multiply-add pipes can issue at: the denominator any kernel is measured
    // against. Measured at about 22.7 TFLOP/s on the GB10 in September 2026.
    const want: f32 = @floatFromInt(trips * rounds);
    var best: f64 = 0;
    for ([_]u32{ 512, 2048 }) |blocks| {
        const g = Grid{
            .register_count = a.registerCount(),
            .grid = .{ blocks, 1, 1 },
            .block = .{ threads, 1, 1 },
        };
        try r.run(code[0..a.dwords()], g); // warm
        const start: std.Io.Timestamp = .now(std.testing.io, .awake);
        try r.run(code[0..a.dwords()], g);
        const ns: u64 = @intCast(start.durationTo(.now(std.testing.io, .awake)).nanoseconds);
        try std.testing.expectEqual(want, out.read(f32, 0));
        try std.testing.expectEqual(want, out.read(f32, blocks * threads - 1));
        const flops = @as(f64, @floatFromInt(blocks)) * threads * accs * rounds * trips * 2;
        best = @max(best, flops / @as(f64, @floatFromInt(ns)));
    }
    // Well under the measured rate, so this catches a real collapse rather than
    // ordinary variation or a slower part in the same family.
    if (best < 10_000) {
        std.debug.print("FP32 multiply-add rate fell to {d:.0} GFLOP/s\n", .{best});
        return error.ArithmeticRateCollapsed;
    }
}

test "live: a mapping in the driver's reserved span is refused, not silently dead" {
    var r = try Runner.init();
    defer r.deinit();
    const mem = try r.client.allocMemory(r.dev, .system, 0x1000);

    // Inside the span, at both edges and in the middle.
    for ([_]u64{ 0xFF00_0000, 0x1_0000_0000, 0x1_FFF0_0000 }) |va| {
        try std.testing.expectError(
            error.ReservedGpuAddress,
            r.client.mapToGpu(r.dev, r.vaspace, mem, va),
        );
    }
    // A range that only overlaps the start of the span is refused too.
    try std.testing.expectError(
        error.ReservedGpuAddress,
        r.client.mapToGpu(r.dev, r.vaspace, mem, rm.Client.RESERVED_VA_START - 0x800),
    );
    // Either side of it is fine, and a store there actually lands.
    for ([_]u64{ 0xF000_0000, rm.Client.RESERVED_VA_END }) |va| {
        r.next_va = va;
        const buf = try r.alloc(.system, 0x1000);
        try std.testing.expectEqual(va, buf.va);
        var code: [128]u32 = undefined;
        var deps: [32]sass.Dep = @splat(.{});
        var a = sass.Assembler{ .code = &code, .deps = &deps };
        a.movImm(0, @truncate(buf.va), .{});
        a.movImm(1, @intCast(buf.va >> 32), .{});
        a.movImm(2, 0xabcd, .{});
        a.stg(0, 2, 0, .bits32, .{});
        a.exit(.{});
        try a.schedule();
        try r.run(code[0..a.dwords()], .{ .register_count = a.registerCount() });
        try std.testing.expectEqual(@as(u32, 0xabcd), buf.read(u32, 0));
    }
}

test "the address allocator steps over the driver's reserved span" {
    // Walk the bump allocator up to the span and check it lands past the end
    // rather than inside, where stores would vanish.
    var next: u64 = rm.Client.RESERVED_VA_START - 0x20_0000;
    try std.testing.expect(!rm.Client.vaIsReserved(next, 0x1000));
    next += 0x20_0000;
    try std.testing.expect(rm.Client.vaIsReserved(next, 0x1000));
    if (rm.Client.vaIsReserved(next, 0x1000)) next = rm.Client.RESERVED_VA_END;
    try std.testing.expectEqual(rm.Client.RESERVED_VA_END, next);
    try std.testing.expect(!rm.Client.vaIsReserved(next, 0x1000));
}

test "live: a load keeps its address until it has read it, at high occupancy" {
    var r = try Runner.init();
    defer r.deinit();
    const blocks = 16384;
    const threads = 32;
    const count = blocks * threads;
    const in = try r.alloc(.system, count * 4);
    const out = try r.alloc(.system, count * 4);
    for (in.slice(u32)[0..count], 0..) |*w, i| w.* = @intCast(i);
    const Src = sass.Src;

    var code: [256]u32 = undefined;
    var deps: [64]sass.Dep = @splat(.{});
    var a = sass.Assembler{ .code = &code, .deps = &deps };
    a.s2r(2, sass.SR_TID_X, .{});
    a.s2r(3, sass.SR_CTAID_X, .{});
    a.imad(4, Src.reg(3), Src.imm(threads), Src.reg(2), false, .{});
    a.movImm(6, @truncate(in.va), .{});
    a.movImm(7, @intCast(in.va >> 32), .{});
    a.imadWide(0, Src.reg(4), Src.imm(4), Src.reg(6), false, .{});
    const load_at = a.here();
    a.ldg(8, 0, 0, .bits32, .{});
    // Step the address straight afterwards. Without a read scoreboard the load
    // can pick up the stepped address and return the next element instead.
    a.iadd3(0, Src.reg(0), Src.imm(4), sass.zero, .{});
    a.movImm(10, @truncate(out.va), .{});
    a.movImm(11, @intCast(out.va >> 32), .{});
    a.imadWide(12, Src.reg(4), Src.imm(4), Src.reg(10), false, .{});
    a.stg(12, 8, 0, .bits32, .{});
    a.exit(.{});
    try a.schedule();

    // The scheduler has to have guarded it, whatever the hardware tolerates.
    const rd = (code[load_at * 4 + 3] >> 17) & 0x7;
    try std.testing.expect(rd != 7);

    @memset(out.slice(u32)[0..count], 0xffff_ffff);
    try r.run(code[0..a.dwords()], .{
        .register_count = a.registerCount(),
        .grid = .{ blocks, 1, 1 },
        .block = .{ threads, 1, 1 },
    });
    for (0..count) |i| {
        if (out.read(u32, i) != @as(u32, @intCast(i))) {
            std.debug.print("thread {d} read {d}\n", .{ i, out.read(u32, i) });
            return error.LoadUsedTheSteppedAddress;
        }
    }
}

/// One dependent pair, built with a chosen destination register and chosen
/// extra sources for the consumer, so the register indices are the only thing
/// that varies between measurements.
const PairShape = struct {
    dst: u8, // the register the producer writes and the consumer reads
    src_b: u8, // the consumer's other sources, RZ for none
    src_c: u8,
    consumer_is_fma: bool,
    producer_is_alu: bool = false,
};

fn runPair2(r: *Runner, out: Buffer, shape: PairShape, cdst: u8, stall: u4) !bool {
    const Src = sass.Src;
    var code: [256]u32 = undefined;
    var a = sass.Assembler{ .code = &code };
    a.movImm(0, @truncate(out.va), .{});
    a.movImm(1, @intCast(out.va >> 32), .{});
    a.movImm(shape.dst, 0x9999, .{}); // the stale value
    if (shape.src_b != sass.RZ) a.movImm(shape.src_b, 1, .{});
    if (shape.src_c != sass.RZ) a.movImm(shape.src_c, 0, .{});
    // Producer, then consumer, with nothing in between.
    if (shape.producer_is_alu) {
        a.movImm(shape.src_b, 0x1234, .{});
        a.iadd3(shape.dst, Src.reg(shape.src_b), Src.imm(0), sass.zero, .{ .stall = stall });
    } else a.movImm(shape.dst, 0x1234, .{ .stall = stall });
    if (shape.consumer_is_fma) {
        a.imad(cdst, Src.reg(shape.dst), Src.reg(shape.src_b), Src.reg(shape.src_c), false, .{});
    } else {
        a.iadd3(cdst, Src.reg(shape.dst), Src.imm(0), sass.zero, .{});
    }
    a.stg(0, cdst, 0, .bits32, .{});
    a.exit(.{});
    out.slice(u32)[0] = 0;
    try r.run(code[0..a.dwords()], .{ .register_count = a.registerCount() });
    return out.read(u32, 0) == 0x1234;
}

fn minPairStall2(r: *Runner, out: Buffer, shape: PairShape, cdst: u8) !u4 {
    _ = try runPair2(r, out, shape, cdst, 15); // warm the program
    var lo: u4 = 1;
    var hi: u4 = 15;
    while (lo < hi) {
        const mid: u4 = lo + (hi - lo) / 2;
        if (try runPair2(r, out, shape, cdst, mid)) hi = mid else lo = mid + 1;
    }
    return lo;
}

fn minPairStall(r: *Runner, out: Buffer, shape: PairShape) !u4 {
    return minPairStall2(r, out, shape, 3);
}

test "live: only a cross-pipe result depends on which register carries it" {
    var r = try Runner.init();
    defer r.deinit();
    const out = try r.alloc(.system, 0x1000);

    // Both ends on the integer pipe. The register the value travels in makes no
    // difference here, so `same_pipe` needs no register-aware adjustment.
    var reg: u8 = 4;
    while (reg <= 19) : (reg += 1) {
        if (reg == 12) continue; // the producer's own scratch
        const stall = try minPairStall(&r, out, .{
            .dst = reg,
            .src_b = 12,
            .src_c = sass.RZ,
            .consumer_is_fma = false,
            .producer_is_alu = true,
        });
        try std.testing.expectEqual(@as(u4, sass.Latency.same_pipe), stall);
    }

    // Crossing from the multiply pipe to the integer pipe, the cost alternates
    // with bit 1 of the register: 5 cycles for R4, R5, R8, R9 and so on, 4 for
    // R6, R7, R10, R11. `cross_pipe` is the worst of the two, so the table is
    // safe everywhere and a cycle pessimistic on half the registers. An
    // allocator that prefers a destination with bit 1 set gets that cycle back.
    // Which register is cheap follows bit 1 of its index, but that pattern is
    // not stable enough run to run to pin here. What matters to the scheduler
    // is asserted instead: nothing ever needs more than `cross_pipe`, and the
    // variation is real rather than the constant being pure padding.
    var saw_worst = false;
    var saw_cheap = false;
    reg = 4;
    while (reg <= 19) : (reg += 1) {
        const stall = try minPairStall(&r, out, .{
            .dst = reg,
            .src_b = sass.RZ,
            .src_c = sass.RZ,
            .consumer_is_fma = false,
        });
        if (stall > sass.Latency.cross_pipe) {
            std.debug.print("cross-pipe through R{d} needed {d} cycles\n", .{ reg, stall });
            return error.CrossPipeLatencyExceeded;
        }
        if (stall == sass.Latency.cross_pipe) saw_worst = true;
        if (stall <= sass.Latency.same_pipe) saw_cheap = true;
    }
    try std.testing.expect(saw_worst);
    try std.testing.expect(saw_cheap);
}
