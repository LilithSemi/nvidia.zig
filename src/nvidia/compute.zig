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

    /// Where the bump allocator starts, and the step between allocations. One
    /// 2 MB step per buffer keeps every allocation on its own big page.
    const VA_BASE: u64 = 0x1000_0000;
    const VA_STEP: u64 = 0x20_0000;
    const CODE_BYTES = 0x1000; // matches the QMD's 4 KB program prefetch
    const PUSH_BYTES = 0x1000;
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
        self.descriptor = try self.alloc(.system_wc, QMD_DWORDS * 4);
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
        self.queue = .{
            .channel = ch,
            .token = token,
            .userd = userd_cpu.bytes,
            .gpfifo = gpfifo_cpu.bytes,
            .doorbell = door.bytes,
        };
        return self;
    }

    pub fn deinit(self: *Runner) void {
        self.client.rmFree(self.dev.client, self.dev.device, self.channel.handle);
        self.client.freeDevice(self.dev);
        self.client.deinit();
    }

    fn takeVa(self: *Runner, size: u64) u64 {
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

    /// Upload `code`, dispatch grid `g`, and wait for it. `g.prog_va` is filled
    /// in here. A grid that never signals gives `error.GridTimeout`, which means
    /// the kernel hung or faulted.
    pub fn run(self: *Runner, code: []const u32, g: Grid) !void {
        std.debug.assert(code.len * 4 <= self.code.bytes.len);
        @memcpy(self.code.slice(u32)[0..code.len], code);

        var grid = g;
        grid.prog_va = self.code.va;
        var qmd: [QMD_DWORDS]u32 = undefined;
        buildQmd(&qmd, grid);
        @memcpy(self.descriptor.slice(u32)[0..QMD_DWORDS], &qmd);

        var s = Stream{ .buf = self.push.slice(u32) };
        s.setup();
        s.dispatch(self.descriptor.va);
        self.seq += 1;
        s.fence(self.sem.va, self.seq);

        const semp: *volatile u32 = @ptrCast(@alignCast(self.sem.bytes.ptr));
        semp.* = 0;
        self.queue.submit(self.push.va, s.dwords());
        var spins: u64 = 0;
        while (spins < SPIN_LIMIT) : (spins += 1) {
            if (semp.* == self.seq) return;
        }
        return error.GridTimeout;
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

    a.s2r(13, sass.SR_TID_X, .{ .wr_barrier = 0 });
    a.s2r(14, sass.SR_TID_Y, .{ .wr_barrier = 1 });
    a.s2r(16, sass.SR_CTAID_X, .{ .wr_barrier = 2 });
    a.imad(9, Src.reg(16), Src.imm(tile), Src.reg(13), false, .{ .wait_mask = 0b101 }); // col
    a.s2r(16, sass.SR_CTAID_Y, .{ .wr_barrier = 2 });
    a.imad(8, Src.reg(16), Src.imm(tile), Src.reg(14), false, .{ .wait_mask = 0b110 }); // row

    a.ldc(10, .{ .offset = mm_params.rows }, sass.RZ, .bits32, .{ .wr_barrier = 0 });
    a.ldc(11, .{ .offset = mm_params.cols }, sass.RZ, .bits32, .{ .wr_barrier = 1 });
    a.ldc(12, .{ .offset = mm_params.depth }, sass.RZ, .bits32, .{ .wr_barrier = 2 });
    a.ldc(20, .{ .offset = mm_params.a_ptr }, sass.RZ, .bits64, .{ .wr_barrier = 3 });
    a.ldc(22, .{ .offset = mm_params.b_ptr }, sass.RZ, .bits64, .{ .wr_barrier = 4 });
    a.ldc(24, .{ .offset = mm_params.c_ptr }, sass.RZ, .bits64, .{ .wr_barrier = 5 });

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
    a.iadd3(34, Src.reg(12), Src.imm(tile - 1), zero, .{ .wait_mask = 0b000100 });
    a.shr(34, Src.reg(34), Src.imm(tile_shift), false, .{});

    const tile_loop = a.here();

    // Stage A[row][t*tile + tx]. P0 and P1 record whether the element exists;
    // the clamped indices keep the address inside A either way.
    a.imad(29, Src.reg(5), Src.imm(tile), Src.reg(13), false, .{});
    a.isetp(0, .lt, true, Src.reg(29), Src.reg(12), .{});
    a.isetp(1, .lt, true, Src.reg(8), Src.reg(10), .{ .wait_mask = 0b000001 });
    a.sel(28, Src.reg(29), zero, 0, false, .{});
    a.sel(17, Src.reg(8), zero, 1, false, .{});
    a.imad(16, Src.reg(17), Src.reg(12), Src.reg(28), false, .{});
    a.imadWide(0, Src.reg(16), Src.imm(4), Src.reg(20), false, .{ .wait_mask = 0b001000 });
    a.ldg(6, 0, 0, .bits32, .{ .wr_barrier = 0 });
    a.sel(6, Src.reg(6), zero, 0, false, .{ .wait_mask = 0b000001 });
    a.sel(6, Src.reg(6), zero, 1, false, .{});

    // Stage B[t*tile + ty][col] the same way.
    a.imad(29, Src.reg(5), Src.imm(tile), Src.reg(14), false, .{});
    a.isetp(0, .lt, true, Src.reg(29), Src.reg(12), .{});
    a.isetp(1, .lt, true, Src.reg(9), Src.reg(11), .{ .wait_mask = 0b000010 });
    a.sel(28, Src.reg(29), zero, 0, false, .{});
    a.sel(17, Src.reg(9), zero, 1, false, .{});
    a.imad(16, Src.reg(28), Src.reg(11), Src.reg(17), false, .{});
    a.imadWide(2, Src.reg(16), Src.imm(4), Src.reg(22), false, .{ .wait_mask = 0b010000 });
    a.ldg(7, 2, 0, .bits32, .{ .wr_barrier = 1 });
    a.sel(7, Src.reg(7), zero, 0, false, .{ .wait_mask = 0b000010 });
    a.sel(7, Src.reg(7), zero, 1, false, .{});

    a.sts(27, 6, 0, .bits32, .{});
    a.sts(27, 7, b_base, .bits32, .{});
    a.bar(.{});

    // Walk the staged tile: A along a row, B down a column.
    a.movImm(26, 0, .{});
    a.movReg(30, 18, .{});
    a.movReg(31, 19, .{});
    const inner_loop = a.here();
    a.lds(32, 30, 0, .bits32, .{ .wr_barrier = 0, .rd_barrier = 2 });
    a.lds(33, 31, 0, .bits32, .{ .wr_barrier = 1, .rd_barrier = 3 });
    // The pointer step must not overtake the load that still needs the address.
    a.iadd3(30, Src.reg(30), Src.imm(4), zero, .{ .wait_mask = 0b001100 });
    a.iadd3(31, Src.reg(31), Src.imm(tile * 4), zero, .{});
    a.ffma(4, Src.reg(32), Src.reg(33), Src.reg(4), .{ .wait_mask = 0b000011 });
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
    a.imadWide(2, Src.reg(16), Src.imm(4), Src.reg(24), false, .{ .wait_mask = 0b100000 });
    a.stg(2, 4, 0, .bits32, .{});
    a.patchBranch(row_outside, a.here());
    a.patchBranch(col_outside, a.here());
    a.exit(.{ .stall = 1 });
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
    var a = sass.Assembler{ .code = &code };
    buildTiledMatmul(&a, tile);

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
