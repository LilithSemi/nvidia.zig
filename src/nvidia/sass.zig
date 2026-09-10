//! Minimal hand-assembler for NVIDIA SASS, the native GPU ISA. The instruction
//! encoding is shared Volta..Blackwell (only per-instruction latencies differ),
//! so one encoder serves all of them. Each instruction is 128 bits (4 dwords).
//! Verified live on Blackwell (sm_120): the kernels the tests at the bottom of
//! this file assemble run on the SMs.
//!
//! Scheduling: each instruction carries a `Control` with a stall count (cycles
//! before the next instruction) and optional scoreboard barriers. Write those by
//! hand, or give the assembler a `Dep` buffer and call `schedule`, which derives
//! them from the register traffic using the measured `Latency` table below.
//!
//! Uniform registers (UGPR) are encoded in 8 bits here, which is the Blackwell
//! (sm >= 100) width. Volta..Ada use 6 bits and need a per-generation switch
//! before this encoder can target them.

const std = @import("std");

pub const RZ: u8 = 255; // the zero GPR
pub const URZ: u8 = 255; // the zero uniform GPR (8-bit UGPR field, sm >= 100)
pub const PT: u8 = 7; // the always-true predicate register

/// Set `width` bits at bit offset `lo` within a 128-bit instruction.
fn setBits(inst: []u32, lo: usize, width: usize, val: u64) void {
    var i: usize = 0;
    while (i < width) : (i += 1) {
        const bit = lo + i;
        const off: u5 = @intCast(bit % 32);
        const b: u32 = @intCast((val >> @intCast(i)) & 1);
        inst[bit / 32] = (inst[bit / 32] & ~(@as(u32, 1) << off)) | (b << off);
    }
}

/// Keep the low `width` bits of a two's-complement value, for the signed
/// immediate fields (memory offsets, branch offsets).
fn signedBits(val: i64, width: usize) u64 {
    const mask: u64 = if (width >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(width)) - 1;
    return @as(u64, @bitCast(val)) & mask;
}

/// A constant-bank reference, `c[bank][offset]`. `offset` is a byte offset and
/// must be 4-aligned. Kernel parameters live in bank 0, which the QMD binds.
pub const CBuf = struct {
    bank: u5 = 0,
    offset: u16,
};

/// An ALU operand. The hardware reads the first operand from a register only.
/// The second and third can each be a register or a 32-bit immediate, and which
/// one holds the immediate selects the instruction "form" in bits 9..11.
///
/// The ISA also lets an ALU operand read a constant bank directly. That form is
/// not here: every encoding of it tried so far raises "Illegal Instruction
/// Encoding" on Blackwell silicon, so a kernel loads the constant with `ldc`
/// first. Add it back only with a live test that passes.
pub const Src = struct {
    ref: Ref,
    abs: bool = false, // absolute value (float sources only)
    neg: bool = false, // negate (float sources, and integer add/multiply)

    pub const Ref = union(enum) {
        reg: u8,
        imm: u32,
    };

    pub fn reg(n: u8) Src {
        return .{ .ref = .{ .reg = n } };
    }
    pub fn imm(v: u32) Src {
        return .{ .ref = .{ .imm = v } };
    }
    pub fn float(v: f32) Src {
        return .{ .ref = .{ .imm = @bitCast(v) } };
    }

    pub fn negated(self: Src) Src {
        var s = self;
        s.neg = !s.neg;
        return s;
    }
    pub fn absolute(self: Src) Src {
        var s = self;
        s.abs = true;
        return s;
    }

    fn isReg(self: Src) bool {
        return switch (self.ref) {
            .reg => true,
            else => false,
        };
    }
    fn regIndex(self: Src) u8 {
        return switch (self.ref) {
            .reg => |r| r,
            else => unreachable, // this operand slot only accepts a register
        };
    }
};

/// The zero register as an operand. Reads as 0, writes are dropped.
pub const zero = Src.reg(RZ);

/// The width of one memory access, and how the loaded bits are extended.
pub const MemType = enum(u3) {
    unsigned8 = 0,
    signed8 = 1,
    unsigned16 = 2,
    signed16 = 3,
    bits32 = 4,
    bits64 = 5,
    bits128 = 6,

    /// How many consecutive GPRs the access reads or writes.
    fn regs(self: MemType) u8 {
        return switch (self) {
            .bits64 => 2,
            .bits128 => 4,
            else => 1,
        };
    }
};

/// How far a memory barrier reaches.
pub const MemScope = enum(u3) {
    cta = 0,
    gpu = 2,
    system = 3,
};

/// The integer comparison an ISETP applies.
pub const IntCmp = enum(u3) {
    never = 0,
    lt = 1,
    eq = 2,
    le = 3,
    gt = 4,
    ne = 5,
    ge = 6,
    always = 7,
};

/// Per-instruction scheduling and predication. On Blackwell (sm >= 120) the
/// yield/reuse bits are dropped, so only the stall, barriers, wait mask, and
/// guard predicate are encoded.
pub const Control = struct {
    stall: u4 = 15, // conservative; covers a fixed-latency register dependency
    wr_barrier: u3 = 7, // write scoreboard to set on completion (7 = none)
    rd_barrier: u3 = 7, // read scoreboard to set (7 = none)
    wait_mask: u6 = 0, // scoreboards to wait on before issue
    /// Guard predicate. The instruction runs only when predicate register `pred`
    /// holds the value opposite to `pred_not`. The default PT is always true, so
    /// the instruction is unconditional.
    pred: u8 = PT,
    pred_not: bool = false,
};

fn putControl(inst: []u32, c: Control) void {
    setBits(inst, 12, 3, c.pred);
    setBits(inst, 15, 1, @intFromBool(c.pred_not));
    setBits(inst, 105, 4, c.stall);
    setBits(inst, 110, 3, c.wr_barrier);
    setBits(inst, 113, 3, c.rd_barrier);
    setBits(inst, 116, 6, c.wait_mask);
}

/// Which functional pipe an instruction issues to. A reader on the pipe that
/// produced a value sees it one cycle sooner than a reader on the other pipe,
/// which is the whole of the fixed-latency model on this hardware.
pub const Pipe = enum {
    /// The integer pipe: IADD3, LOP3, SHF.
    alu,
    /// The multiply pipe: MOV, SEL, IMAD, IMAD.WIDE, ISETP, FADD, FMUL, FFMA.
    fma,
    /// Memory, control flow, and anything whose result a scoreboard covers.
    /// These never pay the cross-pipe cycle.
    other,
};

/// Result latencies in cycles, measured on Blackwell by the probes in
/// compute.zig. `Assembler.schedule` leaves this many cycles between an
/// instruction and a reader of its result. The live test
/// "the scheduler's latency table still covers the silicon" re-measures every
/// figure here, so a chip that needs longer fails loudly instead of quietly
/// computing the wrong answer.
pub const Latency = struct {
    /// A fixed-latency result read on the pipe that produced it.
    pub const same_pipe: u8 = 4;
    /// The same result read on the other compute pipe. Measured across every
    /// pair of ALU and FMA instructions this encoder emits.
    pub const cross_pipe: u8 = 5;
    /// A predicate read as an instruction operand: a SEL condition, or the
    /// carry-in of an extended IADD3.
    pub const pred: u8 = 5;
    /// A predicate read as a guard, or as a branch condition. The hardware tests
    /// those far later in the pipeline, so they cost much more than an operand
    /// read of the same predicate.
    pub const pred_guard: u8 = 13;

    /// Cycles the scheduler adds on top of every measured figure above.
    ///
    /// The figures are measured one dependency at a time, at one warp so that no
    /// other warp can hide a shortfall. Two things are known to sit outside that
    /// model. Register banking: a cross-pipe result costs a cycle less through
    /// half the registers, which makes `cross_pipe` the worst case and safe
    /// rather than short. And something in dense code that neither banking nor
    /// occupancy explains, which an unrolled matmul inner loop reproduces and
    /// which no single-instruction analysis has localised.
    ///
    /// So this is not padding against a known effect, it is honesty about an
    /// unknown one. What is known about that unknown, from unrolling the tiled
    /// matmul inner loop and failing to ship it twice:
    ///
    /// The trigger is the outer tile loop running more than once. One block with
    /// one tile is correct; one block with two tiles is wrong. Four blocks with
    /// one tile are correct, so it is not about blocks. Sizes that are exact
    /// multiples of the tile fail too, so it is not the boundary guards, and the
    /// guards are predicated rather than branched anyway.
    ///
    /// It is not probabilistic, which is easy to misread from the symptom. The
    /// unrolled kernel is wrong on every run at the shapes that trigger it, and
    /// the looped one is right on every run at those same shapes, 225 of them
    /// across a threefold stretch of the tile walk. What moves between runs is
    /// only which rows of the staged tile lose. So the defect is present in one
    /// version and absent in the other; it is not one defect that the slower
    /// version keeps winning against.
    ///
    /// Ruled out by measurement, not by argument: register banking, the latency
    /// table being an average, scoreboard pressure (one load pair in flight
    /// fails the same as four), shared memory visibility at either barrier (a
    /// CTA membar on both sides changes nothing), outstanding shared loads at
    /// the barrier (every load is already waited on by name where its value is
    /// used, and forcing the fence to drain all six scoreboards changes
    /// nothing), the barrier's predicate source at bits 87..90, which ptxas also
    /// leaves zero so the field is unused for this form, the barrier's
    /// DEFER_BLOCKING bit at 80, which ptxas sets on every __syncthreads() and
    /// neither NAK nor this encoder does, and code layout, since a stall bump
    /// moves no instruction bytes at all.
    ///
    /// One architectural fact worth keeping, from a ptxas dump of a tiled
    /// matmul: it issues shared loads that cross a BAR.SYNC and consumes them
    /// after it, with an empty wait mask on every barrier in the kernel. So the
    /// hardware finishes the shared access before the barrier releases and only
    /// the register writeback is late. Draining scoreboards at a barrier is
    /// therefore wrong on the merits rather than merely useless, and this kernel
    /// is already stricter than NVIDIA's, since every load here is consumed
    /// before the barrier rather than across it.
    ///
    /// The one structural difference left: the looped inner loop leaves P0
    /// uniformly false at the second barrier, because that is the condition that
    /// ended it. The unrolled version leaves whatever the staging guard put
    /// there, which is divergent at some shapes.
    ///
    /// Instructions that depend on nothing are unaffected: they still issue back
    /// to back.
    pub const margin: u8 = 1;
};

/// The number of scoreboards the hardware gives each warp.
pub const BARRIERS: u8 = 6;

const NO_BARRIER: u8 = 0xff;

/// What one instruction does to registers, recorded while it is emitted so
/// `Assembler.schedule` can work out the stalls and scoreboards for the program.
pub const Dep = struct {
    /// GPRs written. Unused slots hold RZ.
    writes: [4]u8 = .{ RZ, RZ, RZ, RZ },
    /// GPRs read. Unused slots hold RZ.
    reads: [6]u8 = .{ RZ, RZ, RZ, RZ, RZ, RZ },
    /// Predicate written, and predicate read. PT means neither.
    pred_write: u8 = PT,
    pred_read: u8 = PT,
    /// The result arrives at an unpredictable time, so a consumer has to wait on
    /// a scoreboard instead of counting cycles.
    ///
    /// Memory and S2R are the obvious ones. NAK also classes the conversions and
    /// bit-counting ops as decoupled: F2F, F2I, I2F, FRND, POPC, FLO and BREV.
    /// This encoder emits none of those yet; whoever adds one has to set this,
    /// because a fixed stall for them is a guess rather than a guarantee.
    variable: bool = false,
    /// The instruction reads its sources after it issues (a memory op holding an
    /// address), so an instruction that overwrites those sources must wait too.
    late_read: bool = false,
    /// Every outstanding scoreboard must drain before this instruction.
    fence: bool = false,
    /// Control leaves this instruction, so nothing after it is reachable in a
    /// straight line.
    branch: bool = false,
    /// Control can arrive here from elsewhere.
    target: bool = false,
    /// The pipe this instruction issues to, which sets how long a reader of its
    /// result has to wait. Ignored when `variable` is set, because a scoreboard
    /// covers those instead.
    pipe: Pipe = .other,

    /// Fills a `Dep` one operand at a time, so an emitter can describe itself
    /// without counting array slots.
    pub const Builder = struct {
        dep: Dep = .{},
        w: usize = 0,
        r: usize = 0,

        pub fn write(self: *Builder, reg: u8) void {
            self.writeRun(reg, 1);
        }
        pub fn writeRun(self: *Builder, first: u8, count: u8) void {
            if (first == RZ) return;
            var i: u8 = 0;
            while (i < count) : (i += 1) {
                self.dep.writes[self.w] = first + i;
                self.w += 1;
            }
        }
        pub fn read(self: *Builder, reg: u8) void {
            self.readRun(reg, 1);
        }
        pub fn readRun(self: *Builder, first: u8, count: u8) void {
            if (first == RZ) return;
            var i: u8 = 0;
            while (i < count) : (i += 1) {
                self.dep.reads[self.r] = first + i;
                self.r += 1;
            }
        }
        /// Read an ALU operand, which contributes nothing when it is a constant.
        pub fn readSrc(self: *Builder, s: Src) void {
            switch (s.ref) {
                .reg => |reg| self.read(reg),
                .imm => {},
            }
        }
        pub fn readSrcRun(self: *Builder, s: Src, count: u8) void {
            switch (s.ref) {
                .reg => |reg| self.readRun(reg, count),
                .imm => {},
            }
        }
    };
};

/// Emits 128-bit SASS instructions into a caller-provided dword buffer (4 dwords
/// per instruction). Tracks the highest GPR touched so the launch descriptor can
/// derive a register count.
///
/// Give it a `deps` buffer of one entry per instruction to use `schedule`, which
/// replaces every hand-written `Control` with stalls and scoreboards derived
/// from the register traffic. Leave `deps` empty to keep full manual control.
pub const Assembler = struct {
    code: []u32,
    deps: []Dep = &.{},
    n: usize = 0, // dwords emitted
    max_reg: u8 = 0, // highest GPR index written/read (excluding RZ)

    /// Record what the instruction just emitted does to registers.
    fn dep(self: *Assembler, b: Dep.Builder) void {
        if (self.deps.len == 0) return;
        self.deps[self.n / 4 - 1] = b.dep;
    }

    fn next(self: *Assembler) []u32 {
        const w = self.code[self.n..][0..4];
        @memset(w, 0);
        if (self.deps.len != 0) {
            // A branch can name its target before the target is emitted, so the
            // reset keeps that mark.
            const was_target = self.deps[self.n / 4].target;
            self.deps[self.n / 4] = .{ .target = was_target };
        }
        self.n += 4;
        return w;
    }
    fn note(self: *Assembler, reg: u8) void {
        if (reg != RZ and reg > self.max_reg) self.max_reg = reg;
    }
    /// Record a run of `count` registers starting at `first`.
    fn noteRun(self: *Assembler, first: u8, count: u8) void {
        if (first == RZ) return;
        self.note(first + count - 1);
    }

    /// Program size in dwords (bytes = dwords()*4).
    pub fn dwords(self: *const Assembler) usize {
        return self.n;
    }
    /// Index of the instruction that the next emit lands on. Use it as a branch
    /// target and as the argument to `patchBranch`.
    pub fn here(self: *const Assembler) usize {
        return self.n / 4;
    }
    /// A safe register count for the QMD. The hardware keeps the top two GPRs of
    /// each thread's allocation for itself: writes to them are dropped and reads
    /// return zero, verified live. So the count covers the highest GPR used plus
    /// those two, rounded up to the allocation granularity of 8, minimum 16.
    pub const RESERVED_GPRS: u32 = 2;
    pub fn registerCount(self: *const Assembler) u32 {
        const used = @as(u32, self.max_reg) + 1 + RESERVED_GPRS;
        return @max(16, (used + 7) & ~@as(u32, 7));
    }

    /// Rewrite every instruction's stall count and scoreboards from the register
    /// traffic recorded during assembly. Call it once, after the last
    /// instruction and before uploading the code.
    ///
    /// The model is a cycle counter. Each instruction issues as late as its
    /// operands demand and no later, so instructions that depend on nothing
    /// issue back to back at a stall of 1 instead of paying a fixed worst-case
    /// delay. A result that arrives at an unpredictable time gets a scoreboard
    /// instead of a delay, and the reader waits on it.
    ///
    /// Control flow is handled by draining: a branch and a branch target both
    /// wait for every outstanding scoreboard, and a branch stalls long enough to
    /// cover every fixed-latency result still in flight. That costs a few cycles
    /// per loop and removes the need to reason about which path arrived.
    pub fn schedule(self: *Assembler) error{OutOfBarriers}!void {
        std.debug.assert(self.deps.len * 4 >= self.n);
        const count = self.n / 4;
        if (count == 0) return;

        var reg_ready = [_]u32{0} ** 256; // cycle a same-pipe reader can use a GPR
        var reg_pipe = [_]Pipe{.other} ** 256; // the pipe that wrote it
        var pred_ready = [_]u32{0} ** 8; // when an ALU can read a predicate
        var pred_ready_guard = [_]u32{0} ** 8; // when a guard or branch can read it
        var wr_bar = [_]u8{NO_BARRIER} ** 256; // scoreboard guarding a pending write
        var rd_bar = [_]u8{NO_BARRIER} ** 256; // scoreboard guarding a pending read
        var bar_used = [_]bool{false} ** BARRIERS;

        var cycle: u32 = 0;
        var prev_issue: u32 = 0;
        var prev: usize = 0;
        var have_prev = false;

        for (0..count) |i| {
            const d = self.deps[i];
            var ctl = Control{};
            var wait: u6 = 0;
            var need = cycle;

            // A branch or a target ends the straight line, so nothing may still
            // be in flight across it.
            if (d.target or d.branch or d.fence) {
                for (bar_used, 0..) |used, b| {
                    if (used) wait |= @as(u6, 1) << @intCast(b);
                }
            }
            for (d.reads) |r| {
                if (r == RZ) continue;
                need = @max(need, reg_ready[r] + crossPipeCost(reg_pipe[r], d.pipe));
                if (wr_bar[r] != NO_BARRIER) wait |= @as(u6, 1) << @intCast(wr_bar[r]);
            }
            for (d.writes) |w| {
                if (w == RZ) continue;
                need = @max(need, reg_ready[w]); // write after read of an older value
                if (wr_bar[w] != NO_BARRIER) wait |= @as(u6, 1) << @intCast(wr_bar[w]);
                if (rd_bar[w] != NO_BARRIER) wait |= @as(u6, 1) << @intCast(rd_bar[w]);
            }
            if (d.pred_read != PT) {
                const ready = if (d.branch) pred_ready_guard[d.pred_read] else pred_ready[d.pred_read];
                need = @max(need, ready);
            }
            // The guard predicate lives in the encoded instruction, not in the
            // dependency record, so read it back out.
            const guard = self.instrPred(i);
            if (guard != PT) need = @max(need, pred_ready_guard[guard]);
            if (d.pred_write != PT) need = @max(need, pred_ready[d.pred_write]);

            // Space the previous instruction so this one issues no earlier than
            // `need`. The gap never exceeds the longest fixed latency, so the
            // 4-bit stall field always holds it.
            if (have_prev) {
                const gap = need - prev_issue;
                std.debug.assert(gap <= 15);
                const stall: u4 = @intCast(@max(1, gap));
                self.setStall(prev, stall);
                cycle = prev_issue + stall;
            }

            releaseBarriers(wait, &bar_used, &wr_bar, &rd_bar);

            const has_write_barrier = d.variable and d.writes[0] != RZ;
            var own_barrier: u8 = NO_BARRIER;
            if (has_write_barrier) {
                const b = try takeBarrier(&bar_used, &wait, &wr_bar, &rd_bar, NO_BARRIER);
                own_barrier = b;
                ctl.wr_barrier = @intCast(b);
                for (d.writes) |w| {
                    if (w == RZ) continue;
                    wr_bar[w] = b;
                    reg_ready[w] = cycle;
                    reg_pipe[w] = .other;
                }
            } else {
                const wait_cycles: u32 = if (d.variable) 0 else Latency.same_pipe + Latency.margin;
                for (d.writes) |w| {
                    if (w == RZ) continue;
                    wr_bar[w] = NO_BARRIER;
                    reg_ready[w] = cycle + wait_cycles;
                    reg_pipe[w] = if (d.variable) .other else d.pipe;
                }
            }
            if (d.late_read and self.writesAnyLater(i, d.reads, has_write_barrier)) {
                const b = try takeBarrier(&bar_used, &wait, &wr_bar, &rd_bar, own_barrier);
                ctl.rd_barrier = @intCast(b);
                for (d.reads) |r| {
                    if (r == RZ) continue;
                    rd_bar[r] = b;
                }
            }
            if (d.pred_write != PT) {
                pred_ready[d.pred_write] = cycle + Latency.pred + Latency.margin;
                pred_ready_guard[d.pred_write] = cycle + Latency.pred_guard + Latency.margin;
            }

            ctl.wait_mask = wait;
            ctl.pred = self.instrPred(i);
            ctl.pred_not = self.instrPredNot(i);
            self.setControl(i, ctl);

            if (d.branch) {
                // Give the branch a stall long enough that every fixed-latency
                // result in flight has landed wherever control goes next.
                var pending: u32 = 0;
                // Add the cross-pipe cycle: whatever runs next may be on the
                // other pipe from whatever is still in flight.
                for (reg_ready) |t| pending = @max(pending, (t + 1) -| cycle);
                for (pred_ready) |t| pending = @max(pending, t -| cycle);
                for (pred_ready_guard) |t| pending = @max(pending, t -| cycle);
                self.setStall(i, @intCast(@max(1, @min(15, pending))));
                @memset(&reg_ready, 0);
                @memset(&pred_ready, 0);
                @memset(&pred_ready_guard, 0);
                have_prev = false;
                cycle = 0;
                prev_issue = 0;
                continue;
            }

            prev = i;
            prev_issue = cycle;
            have_prev = true;
        }
        // The last instruction has nothing after it to space out.
        if (have_prev) self.setStall(prev, 1);
    }

    /// The extra cycle a reader pays when the value came off the other compute
    /// pipe. Memory and scoreboard-covered results never pay it.
    fn crossPipeCost(writer: Pipe, reader: Pipe) u32 {
        if (writer == .other or reader == .other) return 0;
        if (writer == reader) return 0;
        return Latency.cross_pipe - Latency.same_pipe;
    }

    fn releaseBarriers(wait: u6, used: *[BARRIERS]bool, wr: *[256]u8, rd: *[256]u8) void {
        for (0..BARRIERS) |b| {
            if (wait & (@as(u6, 1) << @intCast(b)) == 0) continue;
            used[b] = false;
            for (wr) |*e| {
                if (e.* == b) e.* = NO_BARRIER;
            }
            for (rd) |*e| {
                if (e.* == b) e.* = NO_BARRIER;
            }
        }
    }

    /// Take a free scoreboard, waiting on the lowest-numbered one in use when
    /// they are all taken.
    fn takeBarrier(used: *[BARRIERS]bool, wait: *u6, wr: *[256]u8, rd: *[256]u8, avoid: u8) error{OutOfBarriers}!u8 {
        for (0..BARRIERS) |b| {
            if (!used[b]) {
                used[b] = true;
                return @intCast(b);
            }
        }
        // All in use, so wait one out. Never the one this same instruction just
        // took: an instruction cannot set a scoreboard twice and still have the
        // model track what it guards.
        for (0..BARRIERS) |b| {
            const bit = @as(u6, 1) << @intCast(b);
            if (b == avoid or wait.* & bit != 0) continue;
            wait.* |= bit;
            releaseBarriers(bit, used, wr, rd);
            used[b] = true;
            return @intCast(b);
        }
        return error.OutOfBarriers;
    }

    /// Whether a later instruction can overwrite one of `regs` while this one is
    /// still reading them.
    ///
    /// A branch or a branch target ends the scan, because what runs after it is
    /// unknowable. What the scan answers there depends on `has_write_barrier`:
    /// an instruction that already set one is covered, since the drain at that
    /// boundary waits for it and so for its reads as well. One that did not, a
    /// store for instance, has to assume the worst.
    fn writesAnyLater(self: *const Assembler, i: usize, regs: [6]u8, has_write_barrier: bool) bool {
        // Nothing to guard when the instruction reads no register at all.
        if (regs[0] == RZ) return false;
        var j = i + 1;
        while (j < self.n / 4) : (j += 1) {
            const d = self.deps[j];
            for (regs) |g| {
                if (g == RZ) continue;
                for (d.writes) |w| if (w == g) return true;
            }
            if (d.branch or d.target) return !has_write_barrier;
        }
        return false;
    }

    fn instrPred(self: *const Assembler, i: usize) u8 {
        return @intCast((self.code[i * 4] >> 12) & 0x7);
    }
    fn instrPredNot(self: *const Assembler, i: usize) bool {
        return (self.code[i * 4] >> 15) & 1 == 1;
    }
    fn setStall(self: *Assembler, i: usize, stall: u4) void {
        setBits(self.code[i * 4 ..][0..4], 105, 4, stall);
    }
    fn setControl(self: *Assembler, i: usize, ctl: Control) void {
        putControl(self.code[i * 4 ..][0..4], ctl);
    }

    // -----------------------------------------------------------------------
    // ALU operand encoding. The three operand slots are: src0 at bits 24..31
    // (register only), src1 at bits 32..63 (register, immediate, or constant
    // bank), and src2 at bits 64..71 (register). A non-register operand always
    // occupies the 32..63 slot, so when the third operand is the non-register
    // one the second moves into the 64..71 slot and the form changes.
    // -----------------------------------------------------------------------

    fn putSrc2(self: *Assembler, w: []u32, s: ?Src) void {
        const src = s orelse return;
        const r = src.regIndex();
        self.note(r);
        setBits(w, 64, 8, r);
        setBits(w, 74, 1, @intFromBool(src.abs));
        setBits(w, 75, 1, @intFromBool(src.neg));
    }

    /// Fill the 32..63 slot and return the form it selects.
    fn putSrc1(self: *Assembler, w: []u32, s: ?Src) u32 {
        const src = s orelse return 1;
        switch (src.ref) {
            .reg => |r| {
                self.note(r);
                setBits(w, 32, 8, r);
                setBits(w, 62, 1, @intFromBool(src.abs));
                setBits(w, 63, 1, @intFromBool(src.neg));
                return 1;
            },
            .imm => |v| {
                setBits(w, 32, 32, v);
                return 4;
            },
        }
    }

    fn encodeAlu(self: *Assembler, w: []u32, opcode: u32, dst: ?u8, src0: ?Src, src1: ?Src, src2: ?Src) void {
        if (dst) |d| {
            self.note(d);
            setBits(w, 16, 8, d);
        }
        if (src0) |s| {
            const r = s.regIndex();
            self.note(r);
            setBits(w, 24, 8, r);
            setBits(w, 73, 1, @intFromBool(s.abs));
            setBits(w, 72, 1, @intFromBool(s.neg));
        }
        var form: u32 = 1;
        if (src2) |s2| {
            switch (s2.ref) {
                .reg => {
                    self.putSrc2(w, s2);
                    form = self.putSrc1(w, src1);
                },
                .imm => |v| {
                    setBits(w, 32, 32, v);
                    self.putSrc2(w, src1);
                    form = 2;
                },
            }
        } else {
            form = self.putSrc1(w, src1);
        }
        setBits(w, 0, 9, opcode);
        setBits(w, 9, 3, form);
    }

    // -----------------------------------------------------------------------
    // Moves
    // -----------------------------------------------------------------------

    /// MOV dst, src - copy 32 bits from a register or an immediate.
    pub fn mov(self: *Assembler, dst: u8, src: Src, ctl: Control) void {
        const w = self.next();
        self.encodeAlu(w, 0x002, dst, null, src, null);
        setBits(w, 72, 4, 0xf); // all quad lanes
        var dp = Dep.Builder{};
        dp.dep.pipe = .fma;
        dp.write(dst);
        dp.readSrc(src);
        self.dep(dp);
        putControl(w, ctl);
    }

    /// MOV dst, imm32 - load a 32-bit immediate into a GPR.
    pub fn movImm(self: *Assembler, dst: u8, imm: u32, ctl: Control) void {
        self.mov(dst, Src.imm(imm), ctl);
    }

    /// MOV dst, src - copy a 32-bit GPR.
    pub fn movReg(self: *Assembler, dst: u8, src: u8, ctl: Control) void {
        self.mov(dst, Src.reg(src), ctl);
    }

    /// SEL dst, a, b - dst = a when the guard predicate holds, else b. Unlike a
    /// predicated MOV this always writes dst, so it needs no fall-through path.
    pub fn sel(self: *Assembler, dst: u8, a: Src, b: Src, pred: u8, pred_not: bool, ctl: Control) void {
        const w = self.next();
        self.encodeAlu(w, 0x007, dst, a, b, null);
        setBits(w, 87, 3, pred);
        setBits(w, 90, 1, @intFromBool(pred_not));
        var dp = Dep.Builder{};
        dp.dep.pipe = .fma;
        dp.write(dst);
        dp.readSrc(a);
        dp.readSrc(b);
        dp.dep.pred_read = pred;
        self.dep(dp);
        putControl(w, ctl);
    }

    // -----------------------------------------------------------------------
    // Integer arithmetic. IADD3 and IMAD carry nearly all integer work on this
    // ISA, address math included.
    // -----------------------------------------------------------------------

    /// IADD3 dst, a, b, c - dst = a + b + c, 32-bit. Negate an operand with
    /// `Src.negated` to subtract. The hardware needs at least one of a and b
    /// unmodified.
    pub fn iadd3(self: *Assembler, dst: u8, a: Src, b: Src, c: Src, ctl: Control) void {
        self.iadd3Carry(dst, a, b, c, PT, null, ctl);
    }

    /// IADD3 with the carry chain exposed: `carry_out` is the predicate register
    /// that receives the carry (PT discards it) and `carry_in` adds the carry
    /// from an earlier add (null for none). A pair of these is a 64-bit add,
    /// which is how a kernel steps a pointer through an array.
    pub fn iadd3Carry(
        self: *Assembler,
        dst: u8,
        a: Src,
        b: Src,
        c: Src,
        carry_out: u8,
        carry_in: ?u8,
        ctl: Control,
    ) void {
        std.debug.assert(!a.neg or !b.neg);
        const w = self.next();
        self.encodeAlu(w, 0x010, dst, a, b, c);
        if (carry_in) |p| {
            setBits(w, 87, 3, p);
            setBits(w, 90, 1, 0);
            setBits(w, 74, 1, 1); // .X, the extended add that reads a carry
        } else {
            setBits(w, 87, 3, PT); // carry-in = false
            setBits(w, 90, 1, 1);
        }
        setBits(w, 77, 3, PT); // second carry-in = false
        setBits(w, 80, 1, 1);
        setBits(w, 81, 3, carry_out);
        setBits(w, 84, 3, PT); // no second carry-out
        var dp = Dep.Builder{};
        dp.dep.pipe = .alu;
        dp.write(dst);
        dp.readSrc(a);
        dp.readSrc(b);
        dp.readSrc(c);
        dp.dep.pred_write = carry_out;
        if (carry_in) |p| dp.dep.pred_read = p;
        self.dep(dp);
        putControl(w, ctl);
    }

    /// IADD dst, a, b - the two-operand add (IADD3 with a zero third operand).
    pub fn iadd(self: *Assembler, dst: u8, a: Src, b: Src, ctl: Control) void {
        self.iadd3(dst, a, b, zero, ctl);
    }

    /// IMAD dst, a, b, c - dst = a*b + c, 32-bit.
    pub fn imad(self: *Assembler, dst: u8, a: Src, b: Src, c: Src, signed: bool, ctl: Control) void {
        const w = self.next();
        self.encodeAlu(w, 0x024, dst, a, b, c);
        setBits(w, 81, 3, PT); // no predicate destination
        setBits(w, 73, 1, @intFromBool(signed));
        var dp = Dep.Builder{};
        dp.dep.pipe = .fma;
        dp.write(dst);
        dp.readSrc(a);
        dp.readSrc(b);
        dp.readSrc(c);
        self.dep(dp);
        putControl(w, ctl);
    }

    /// IMAD.WIDE dst:dst+1, a, b, c:c+1 - a 32x32 multiply added to a 64-bit
    /// value, giving a 64-bit result. This builds global addresses: multiply an
    /// element index by the element size and add the 64-bit base pointer.
    /// `dst` and a register `c` must both be even.
    pub fn imadWide(self: *Assembler, dst: u8, a: Src, b: Src, c: Src, signed: bool, ctl: Control) void {
        std.debug.assert(dst % 2 == 0);
        self.noteRun(dst, 2);
        if (c.isReg()) {
            std.debug.assert(c.regIndex() % 2 == 0 or c.regIndex() == RZ);
            self.noteRun(c.regIndex(), 2);
        }
        const w = self.next();
        self.encodeAlu(w, 0x025, dst, a, b, c);
        setBits(w, 81, 3, PT);
        setBits(w, 73, 1, @intFromBool(signed));
        var dp = Dep.Builder{};
        dp.dep.pipe = .fma;
        dp.writeRun(dst, 2);
        dp.readSrc(a);
        dp.readSrc(b);
        dp.readSrcRun(c, 2);
        self.dep(dp);
        putControl(w, ctl);
    }

    /// LOP3.LUT dst, a, b, c - an arbitrary bitwise function of three operands.
    /// `lut` is the truth table: with a = 0xf0, b = 0xcc and c = 0xaa as inputs,
    /// evaluate the function you want and pass the result (a & b is 0xc0,
    /// a | b is 0xfc, a ^ b is 0x3c).
    pub fn lop3(self: *Assembler, dst: u8, a: Src, b: Src, c: Src, lut: u8, ctl: Control) void {
        const w = self.next();
        self.encodeAlu(w, 0x012, dst, a, b, c);
        setBits(w, 72, 8, lut);
        setBits(w, 80, 1, 0); // no .PAND
        setBits(w, 81, 3, PT); // no predicate destination
        setBits(w, 87, 3, PT); // predicate input = false
        setBits(w, 90, 1, 1);
        var dp = Dep.Builder{};
        dp.dep.pipe = .alu;
        dp.write(dst);
        dp.readSrc(a);
        dp.readSrc(b);
        dp.readSrc(c);
        self.dep(dp);
        putControl(w, ctl);
    }

    /// SHF.L.U32 dst, a, shift - shift left. This ISA has only the funnel shift,
    /// so a plain shift is a funnel shift with a zero high half.
    pub fn shl(self: *Assembler, dst: u8, a: Src, shift: Src, ctl: Control) void {
        const w = self.next();
        self.encodeAlu(w, 0x019, dst, a, shift, zero);
        setBits(w, 73, 2, 3); // U32
        setBits(w, 75, 1, 0); // clamp the shift amount, do not wrap it
        setBits(w, 76, 1, 0); // left
        setBits(w, 80, 1, 0); // take the low half of the funnel
        var dp = Dep.Builder{};
        dp.dep.pipe = .alu;
        dp.write(dst);
        dp.readSrc(a);
        dp.readSrc(shift);
        self.dep(dp);
        putControl(w, ctl);
    }

    /// SHF.R.U32.HI dst, a, shift - logical shift right. The value goes in the
    /// high half of the funnel and the result comes from the high half too.
    pub fn shr(self: *Assembler, dst: u8, a: Src, shift: Src, signed: bool, ctl: Control) void {
        const w = self.next();
        self.encodeAlu(w, 0x019, dst, zero, shift, a);
        setBits(w, 73, 2, if (signed) 2 else 3); // I32 / U32
        setBits(w, 75, 1, 0);
        setBits(w, 76, 1, 1); // right
        setBits(w, 80, 1, 1); // take the high half of the funnel
        var dp = Dep.Builder{};
        dp.dep.pipe = .alu;
        dp.write(dst);
        dp.readSrc(a);
        dp.readSrc(shift);
        self.dep(dp);
        putControl(w, ctl);
    }

    /// ISETP.<cmp> dst, a, b - compare two integers into a predicate register.
    pub fn isetp(self: *Assembler, dst: u8, cmp: IntCmp, signed: bool, a: Src, b: Src, ctl: Control) void {
        const w = self.next();
        self.encodeAlu(w, 0x00c, null, a, b, null);
        setBits(w, 68, 3, PT); // low-half compare input = false
        setBits(w, 71, 1, 1);
        setBits(w, 87, 3, PT); // accumulator = true, so dst is the compare alone
        setBits(w, 90, 1, 0);
        setBits(w, 72, 1, 0); // not the 64-bit extended form
        setBits(w, 73, 1, @intFromBool(signed));
        setBits(w, 74, 2, 0); // combine with the accumulator through AND
        setBits(w, 76, 3, @intFromEnum(cmp));
        setBits(w, 81, 3, dst);
        setBits(w, 84, 3, PT); // no second destination
        var dp = Dep.Builder{};
        dp.dep.pipe = .fma;
        dp.dep.pred_write = dst;
        dp.readSrc(a);
        dp.readSrc(b);
        self.dep(dp);
        putControl(w, ctl);
    }

    // -----------------------------------------------------------------------
    // Floating point (32-bit). All three round to nearest even.
    // -----------------------------------------------------------------------

    /// FADD dst, a, b - single-precision add.
    pub fn fadd(self: *Assembler, dst: u8, a: Src, b: Src, ctl: Control) void {
        const w = self.next();
        // A non-register second operand has to go through the third slot, which
        // is also what fills the 64..71 register slot with the zero register.
        if (b.isReg()) {
            self.encodeAlu(w, 0x021, dst, a, b, null);
        } else {
            self.encodeAlu(w, 0x021, dst, a, zero, b);
        }
        setBits(w, 77, 1, 0); // no saturate
        setBits(w, 78, 2, 0); // round to nearest even
        setBits(w, 80, 1, 0); // no flush-to-zero
        var dp = Dep.Builder{};
        dp.dep.pipe = .fma;
        dp.write(dst);
        dp.readSrc(a);
        dp.readSrc(b);
        self.dep(dp);
        putControl(w, ctl);
    }

    /// FMUL dst, a, b - single-precision multiply.
    pub fn fmul(self: *Assembler, dst: u8, a: Src, b: Src, ctl: Control) void {
        const w = self.next();
        self.encodeAlu(w, 0x020, dst, a, b, zero);
        setBits(w, 76, 1, 0); // no denormal-to-zero
        setBits(w, 77, 1, 0); // no saturate
        setBits(w, 78, 2, 0); // round to nearest even
        setBits(w, 80, 1, 0); // no flush-to-zero
        setBits(w, 84, 3, 4); // no post-multiply divide
        var dp = Dep.Builder{};
        dp.dep.pipe = .fma;
        dp.write(dst);
        dp.readSrc(a);
        dp.readSrc(b);
        self.dep(dp);
        putControl(w, ctl);
    }

    /// FFMA dst, a, b, c - single-precision fused multiply-add, dst = a*b + c.
    /// This is the inner loop of every dense matrix kernel.
    pub fn ffma(self: *Assembler, dst: u8, a: Src, b: Src, c: Src, ctl: Control) void {
        const w = self.next();
        self.encodeAlu(w, 0x023, dst, a, b, c);
        setBits(w, 76, 1, 0); // no denormal-to-zero
        setBits(w, 77, 1, 0); // no saturate
        setBits(w, 78, 2, 0); // round to nearest even
        setBits(w, 80, 1, 0); // no flush-to-zero
        var dp = Dep.Builder{};
        dp.dep.pipe = .fma;
        dp.write(dst);
        dp.readSrc(a);
        dp.readSrc(b);
        dp.readSrc(c);
        self.dep(dp);
        putControl(w, ctl);
    }

    // -----------------------------------------------------------------------
    // Memory
    // -----------------------------------------------------------------------

    fn putGlobalAccess(w: []u32, mem_type: MemType, offset: i32) void {
        std.debug.assert(offset >= -(1 << 23) and offset < (1 << 23));
        setBits(w, 40, 24, signedBits(offset, 24));
        setBits(w, 73, 3, @intFromEnum(mem_type));
        setBits(w, 77, 4, 0xa); // order STRONG, scope SYS
        setBits(w, 84, 3, 1); // eviction NORMAL
        setBits(w, 90, 1, 1); // the GPR address is a 64-bit register pair
        setBits(w, 91, 1, 1); // UGPR mode (required, or the SM traps)
    }

    /// LDG.E dst, [addr:addr+1 + offset] - load from global memory through the
    /// 64-bit address in the register pair (addr, addr+1). `addr` must be even.
    /// Variable latency: set `ctl.wr_barrier` and make the consumer wait on it.
    pub fn ldg(self: *Assembler, dst: u8, addr: u8, offset: i32, mem_type: MemType, ctl: Control) void {
        self.noteRun(dst, mem_type.regs());
        self.noteRun(addr, 2);
        const w = self.next();
        setBits(w, 0, 12, 0x981);
        setBits(w, 16, 8, dst);
        setBits(w, 24, 8, addr);
        setBits(w, 32, 8, URZ); // no uniform base register
        setBits(w, 72, 1, 1); // ... and it counts as 64-bit
        setBits(w, 64, 3, 0); // guard predicate = true (this field counts down)
        setBits(w, 67, 1, 0);
        setBits(w, 81, 3, PT); // no predicate destination
        putGlobalAccess(w, mem_type, offset);
        var dp = Dep.Builder{};
        dp.dep.pipe = .alu;
        dp.writeRun(dst, mem_type.regs());
        dp.readRun(addr, 2);
        dp.dep.variable = true;
        dp.dep.late_read = true;
        self.dep(dp);
        putControl(w, ctl);
    }

    /// STG.E.STRONG.SYS [addr:addr+1 + offset], data - store to global memory
    /// through the 64-bit address in the register pair (addr, addr+1). `addr`
    /// must be even.
    pub fn stg(self: *Assembler, addr: u8, data: u8, offset: i32, mem_type: MemType, ctl: Control) void {
        self.noteRun(addr, 2);
        self.noteRun(data, mem_type.regs());
        const w = self.next();
        setBits(w, 0, 12, 0x986);
        setBits(w, 24, 8, addr);
        setBits(w, 32, 8, data);
        setBits(w, 64, 8, URZ); // no uniform base register
        setBits(w, 72, 1, 1); // ... and it counts as 64-bit
        putGlobalAccess(w, mem_type, offset);
        var dp = Dep.Builder{};
        dp.dep.pipe = .alu;
        dp.readRun(addr, 2);
        dp.readRun(data, mem_type.regs());
        dp.dep.late_read = true;
        self.dep(dp);
        putControl(w, ctl);
    }

    /// STG of one 32-bit register, the common case.
    pub fn stgU32(self: *Assembler, addr: u8, data: u8, ctl: Control) void {
        self.stg(addr, data, 0, .bits32, ctl);
    }

    /// LDS dst, [addr + offset] - load from the CTA's shared memory. The address
    /// is a byte offset inside the block's shared window, not a global address.
    pub fn lds(self: *Assembler, dst: u8, addr: u8, offset: i32, mem_type: MemType, ctl: Control) void {
        std.debug.assert(offset >= -(1 << 23) and offset < (1 << 23));
        self.noteRun(dst, mem_type.regs());
        self.note(addr);
        const w = self.next();
        setBits(w, 0, 12, 0x984);
        setBits(w, 16, 8, dst);
        setBits(w, 24, 8, addr);
        setBits(w, 32, 8, URZ);
        setBits(w, 40, 24, signedBits(offset, 24));
        setBits(w, 73, 3, @intFromEnum(mem_type));
        setBits(w, 78, 2, 0); // address stride x1
        setBits(w, 87, 1, 0); // no predicate result
        setBits(w, 91, 1, 1);
        var dp = Dep.Builder{};
        dp.dep.pipe = .alu;
        dp.writeRun(dst, mem_type.regs());
        dp.read(addr);
        dp.dep.variable = true;
        dp.dep.late_read = true;
        self.dep(dp);
        putControl(w, ctl);
    }

    /// STS [addr + offset], data - store to the CTA's shared memory.
    pub fn sts(self: *Assembler, addr: u8, data: u8, offset: i32, mem_type: MemType, ctl: Control) void {
        std.debug.assert(offset >= -(1 << 23) and offset < (1 << 23));
        self.note(addr);
        self.noteRun(data, mem_type.regs());
        const w = self.next();
        setBits(w, 0, 12, 0x988);
        setBits(w, 24, 8, addr);
        setBits(w, 32, 8, data);
        setBits(w, 64, 8, URZ);
        setBits(w, 40, 24, signedBits(offset, 24));
        setBits(w, 73, 3, @intFromEnum(mem_type));
        setBits(w, 78, 2, 0); // address stride x1
        setBits(w, 91, 1, 1);
        var dp = Dep.Builder{};
        dp.dep.pipe = .alu;
        dp.read(addr);
        dp.readRun(data, mem_type.regs());
        dp.dep.late_read = true;
        self.dep(dp);
        putControl(w, ctl);
    }

    /// LDC dst, c[bank][cb.offset + index] - read the constant bank through the
    /// constant cache. `index` is a GPR holding a byte offset, or RZ for a
    /// static read. Kernel parameters bound by the QMD in bank 0 are read here.
    pub fn ldc(self: *Assembler, dst: u8, cb: CBuf, index: u8, mem_type: MemType, ctl: Control) void {
        std.debug.assert(cb.offset % 4 == 0);
        self.noteRun(dst, mem_type.regs());
        self.note(index);
        const w = self.next();
        setBits(w, 0, 12, 0xb82);
        setBits(w, 16, 8, dst);
        setBits(w, 24, 8, index);
        setBits(w, 38, 16, cb.offset);
        setBits(w, 54, 5, cb.bank);
        setBits(w, 73, 3, @intFromEnum(mem_type));
        setBits(w, 78, 2, 0); // indexed mode
        setBits(w, 80, 2, 0); // no texture-header unpack (sm >= 120)
        setBits(w, 91, 1, 0); // bound bank, not a bindless handle
        var dp = Dep.Builder{};
        dp.dep.pipe = .alu;
        dp.writeRun(dst, mem_type.regs());
        dp.read(index);
        dp.dep.variable = true;
        dp.dep.late_read = true;
        self.dep(dp);
        putControl(w, ctl);
    }

    /// MEMBAR - order this thread's memory traffic up to `scope` before any that
    /// follows. Needed between a store and a read of it by another CTA.
    pub fn membar(self: *Assembler, scope: MemScope, ctl: Control) void {
        const w = self.next();
        setBits(w, 0, 12, 0x992);
        setBits(w, 72, 1, 0); // not MMIO
        setBits(w, 76, 3, @intFromEnum(scope));
        setBits(w, 80, 1, 0); // not the strong-cached form
        var dp = Dep.Builder{};
        dp.dep.fence = true;
        self.dep(dp);
        putControl(w, ctl);
    }

    // -----------------------------------------------------------------------
    // Control flow
    // -----------------------------------------------------------------------

    /// BAR.SYNC - the CTA barrier. Every thread of the block waits here, and all
    /// shared-memory writes made before it are visible to the block after it.
    pub fn bar(self: *Assembler, ctl: Control) void {
        const w = self.next();
        setBits(w, 0, 12, 0xb1d);
        var dp = Dep.Builder{};
        dp.dep.fence = true;
        self.dep(dp);
        putControl(w, ctl);
    }

    fn putBranchOffset(w: []u32, from: usize, target: usize) void {
        // The hardware offset is relative to the instruction after the branch
        // and counts 4-byte units, so one instruction is 4.
        const delta = (@as(i64, @intCast(target)) - @as(i64, @intCast(from)) - 1) * 4;
        const bits = signedBits(delta, 56);
        setBits(w, 16, 8, bits & 0xff);
        setBits(w, 34, 48, bits >> 8);
    }

    /// BRA target - branch to instruction index `target`. `ctl.pred` is the
    /// branch condition, so a non-PT predicate makes the branch conditional.
    pub fn bra(self: *Assembler, target: usize, ctl: Control) void {
        self.emitBra(target, ctl);
        self.markTarget(target);
    }

    fn emitBra(self: *Assembler, target: usize, ctl: Control) void {
        const from = self.here();
        const w = self.next();
        setBits(w, 0, 12, 0x947);
        setBits(w, 32, 1, 0); // not the uniform-branch form
        putBranchOffset(w, from, target);
        // A branch takes its condition from the 87..89 field, not the usual
        // guard predicate, so the guard itself stays unconditional.
        var guard = ctl;
        guard.pred = PT;
        guard.pred_not = false;
        putControl(w, guard);
        setBits(w, 87, 3, ctl.pred);
        setBits(w, 90, 1, @intFromBool(ctl.pred_not));
        var dp = Dep.Builder{};
        dp.dep.branch = true;
        dp.dep.pred_read = ctl.pred;
        self.dep(dp);
    }

    /// Note that control can reach instruction `target` from somewhere else.
    fn markTarget(self: *Assembler, target: usize) void {
        if (self.deps.len == 0) return;
        std.debug.assert(target < self.deps.len);
        self.deps[target].target = true;
    }

    /// Emit a BRA whose target is not known yet, and return its instruction
    /// index. Call `patchBranch` with that index once the target is emitted.
    pub fn braForward(self: *Assembler, ctl: Control) usize {
        const at = self.here();
        self.emitBra(at + 1, ctl); // provisional: fall through
        return at;
    }

    /// Point the branch at instruction index `at` to instruction index `target`.
    pub fn patchBranch(self: *Assembler, at: usize, target: usize) void {
        std.debug.assert(at * 4 < self.n);
        putBranchOffset(self.code[at * 4 ..][0..4], at, target);
        self.markTarget(target);
    }

    /// EXIT - terminate the warp.
    pub fn exit(self: *Assembler, ctl: Control) void {
        const w = self.next();
        setBits(w, 0, 12, 0x94d);
        setBits(w, 87, 3, 7); // condition-code test = always
        putControl(w, ctl);
    }

    // -----------------------------------------------------------------------
    // Special registers and shader attributes
    // -----------------------------------------------------------------------

    /// S2R dst, sysval - read a special/system register (thread id, CTA id, the
    /// vertex id) into a GPR. Variable latency: set a `wr_barrier` and have the
    /// consumer wait on it.
    pub fn s2r(self: *Assembler, dst: u8, sysval: u8, ctl: Control) void {
        self.note(dst);
        const w = self.next();
        setBits(w, 0, 12, 0x919);
        setBits(w, 16, 8, dst);
        setBits(w, 72, 8, sysval);
        var dp = Dep.Builder{};
        dp.write(dst);
        dp.dep.variable = true;
        self.dep(dp);
        putControl(w, ctl);
    }

    /// ALD dst..dst+comps-1, a[addr] - load `comps` shader input-attribute words
    /// (e.g. a fetched vertex attribute) into consecutive GPRs. Variable latency.
    pub fn ald(self: *Assembler, dst: u8, addr: u16, comps: u8, ctl: Control) void {
        self.note(dst + comps - 1);
        const w = self.next();
        setBits(w, 0, 12, 0x321);
        setBits(w, 16, 8, dst);
        setBits(w, 32, 8, RZ); // vertex (RZ: not per-vertex addressed)
        setBits(w, 24, 8, RZ); // dynamic offset (RZ: static)
        setBits(w, 40, 10, addr);
        setBits(w, 74, 2, comps - 1);
        var dp = Dep.Builder{};
        dp.writeRun(dst, comps);
        dp.dep.variable = true;
        self.dep(dp);
        putControl(w, ctl);
    }

    /// AST o[addr], data..data+comps-1 - store `comps` GPRs to a shader
    /// output attribute (e.g. the clip-space position at ATTR_POSITION).
    pub fn ast(self: *Assembler, addr: u16, data: u8, comps: u8, ctl: Control) void {
        self.note(data + comps - 1);
        const w = self.next();
        setBits(w, 0, 12, 0x322);
        setBits(w, 32, 8, data);
        setBits(w, 64, 8, RZ); // vertex
        setBits(w, 24, 8, RZ); // dynamic offset
        setBits(w, 40, 10, addr);
        setBits(w, 74, 2, comps - 1);
        var dp = Dep.Builder{};
        dp.readRun(data, comps);
        dp.dep.late_read = true;
        self.dep(dp);
        putControl(w, ctl);
    }

    /// IPA dst, a[addr] - interpolate one component (one dword) of a fragment
    /// input attribute. On SM70+ a single IPA with InterpLoc::Default(0) and
    /// InterpFreq::Pass(0) does the full perspective-correct interpolation
    /// implicitly (the barycentrics are hardware-provided, no explicit 1/w
    /// multiply, no load_barycentric setup). `addr` is the attribute BYTE
    /// address (must be 4-aligned); the encoder stores addr>>2. Variable latency
    /// like ALD: set a `wr_barrier` and drain it before consuming the result.
    pub fn ipa(self: *Assembler, dst: u8, addr: u16, ctl: Control) void {
        std.debug.assert(addr % 4 == 0);
        self.note(dst);
        const w = self.next();
        setBits(w, 0, 12, 0x326); // OpIpa
        setBits(w, 16, 8, dst); // dst
        setBits(w, 64, 8, addr >> 2); // attribute addr / 4
        setBits(w, 76, 2, 0); // loc = InterpLoc::Default
        setBits(w, 78, 2, 0); // freq = InterpFreq::Pass (implicit perspective)
        setBits(w, 32, 8, RZ); // offset reg src = RZ (required for Default loc)
        setBits(w, 81, 3, PT); // pred_dst = none (PT)
        var dp = Dep.Builder{};
        dp.write(dst);
        dp.dep.variable = true;
        self.dep(dp);
        putControl(w, ctl);
    }
};

/// Shader attribute addresses (the `addr` for ald/ast). Position output is at
/// 0x70; generic varyings / vertex inputs start at 0x80.
pub const ATTR_POSITION: u16 = 0x70;
pub const ATTR_GENERIC0: u16 = 0x80;

/// System-value indices for `s2r`.
pub const SR_LANE_ID: u8 = 0x00;
pub const SR_TID_X: u8 = 0x21;
pub const SR_TID_Y: u8 = 0x22;
pub const SR_TID_Z: u8 = 0x23;
pub const SR_CTAID_X: u8 = 0x25;
pub const SR_CTAID_Y: u8 = 0x26;
pub const SR_CTAID_Z: u8 = 0x27;
pub const SR_VERTEX_ID: u8 = 0x2f;

test "sass encodes the live-verified store kernel" {
    var code: [64]u32 = undefined;
    var a = Assembler{ .code = &code };
    a.movImm(0, 0x05000000, .{}); // R0 = addr lo
    a.movImm(1, 0, .{}); // R1 = addr hi
    a.movImm(2, 0xcafe, .{}); // R2 = value
    a.stgU32(0, 2, .{}); // STG [R0:R1], R2
    a.exit(.{ .stall = 1 });

    try std.testing.expectEqual(@as(usize, 20), a.dwords());
    try std.testing.expectEqual(@as(u32, 16), a.registerCount()); // R0..R2 -> floored to 16
    // The STG instruction, bit-for-bit, as verified on hardware (bit 91 set).
    try std.testing.expectEqual(@as(u32, 0x00007986), code[12]);
    try std.testing.expectEqual(@as(u32, 0x00000002), code[13]);
    try std.testing.expectEqual(@as(u32, 0x0c1149ff), code[14]);
    // MOV R2, 0xcafe: ALU MOV (0x002) form 4 -> 0x802, dst R2, imm in word+1.
    try std.testing.expectEqual(@as(u32, 0xcafe), code[9]);
}

test "sass encodes MOV register-to-register" {
    var code: [8]u32 = undefined;
    var a = Assembler{ .code = &code };
    a.movReg(0, 4, .{}); // MOV R0, R4
    try std.testing.expectEqual(@as(u32, 0x002), code[0] & 0x1ff); // ALU MOV opcode
    try std.testing.expectEqual(@as(u32, 1), (code[0] >> 9) & 0x7); // form 1 = reg src
    try std.testing.expectEqual(@as(u32, 0), (code[0] >> 16) & 0xff); // dst R0
    try std.testing.expectEqual(@as(u32, 4), code[1] & 0xff); // src R4 at bits 32..39
}

test "sass ALU forms follow the operand that is not a register" {
    var code: [32]u32 = undefined;
    var a = Assembler{ .code = &code };
    // All-register FFMA: form 1, with the third operand in the 64..71 slot.
    a.ffma(4, Src.reg(1), Src.reg(2), Src.reg(3), .{});
    try std.testing.expectEqual(@as(u32, 0x023), code[0] & 0x1ff);
    try std.testing.expectEqual(@as(u32, 1), (code[0] >> 9) & 0x7);
    try std.testing.expectEqual(@as(u32, 1), (code[0] >> 24) & 0xff); // src0 R1
    try std.testing.expectEqual(@as(u32, 2), code[1] & 0xff); // src1 R2
    try std.testing.expectEqual(@as(u32, 3), code[2] & 0xff); // src2 R3
    // An immediate third operand takes the 32..63 slot, pushing the second
    // operand into the register slot at 64..71: form 2.
    a.ffma(4, Src.reg(1), Src.reg(2), Src.float(1.0), .{});
    try std.testing.expectEqual(@as(u32, 2), (code[4] >> 9) & 0x7);
    try std.testing.expectEqual(@as(u32, 0x3f800000), code[5]); // 1.0f
    try std.testing.expectEqual(@as(u32, 2), code[6] & 0xff); // src1 moved to 64..71
}

test "sass IMAD tracks the wide destination pair in the register count" {
    var code: [16]u32 = undefined;
    var a = Assembler{ .code = &code };
    a.imadWide(6, Src.reg(0), Src.imm(4), Src.reg(2), false, .{});
    try std.testing.expectEqual(@as(u32, 0x025), code[0] & 0x1ff);
    try std.testing.expectEqual(@as(u32, 4), (code[0] >> 9) & 0x7); // form 4 = imm
    // R6:R7 is written, so the kernel needs at least 8 registers.
    try std.testing.expectEqual(@as(u32, 16), a.registerCount());
}

test "sass encodes a branch offset relative to the following instruction" {
    var code: [32]u32 = undefined;
    var a = Assembler{ .code = &code };
    a.movImm(0, 0, .{}); // instruction 0
    const at = a.braForward(.{}); // instruction 1
    a.movImm(1, 0, .{}); // instruction 2
    a.exit(.{}); // instruction 3
    a.patchBranch(at, 3);

    const w = code[4..8];
    try std.testing.expectEqual(@as(u32, 0x947), w[0] & 0xfff);
    // Target 3 from branch 1: (3 - 1 - 1) * 4 = 4.
    const low: u64 = (w[0] >> 16) & 0xff;
    const high: u64 = @as(u64, (w[1] >> 2) | (@as(u64, w[2]) << 30)) & 0xffffffffffff;
    try std.testing.expectEqual(@as(u64, 4), low | (high << 8));
}

test "sass encodes a backward branch as a negative offset" {
    var code: [32]u32 = undefined;
    var a = Assembler{ .code = &code };
    a.movImm(0, 0, .{}); // instruction 0
    a.movImm(1, 0, .{}); // instruction 1
    a.bra(0, .{ .pred = 0 }); // instruction 2, conditional on P0
    const w = code[8..12];
    // Target 0 from branch 2: (0 - 2 - 1) * 4 = -12, kept in 56 bits.
    const low: u64 = (w[0] >> 16) & 0xff;
    const high: u64 = @as(u64, (w[1] >> 2) | (@as(u64, w[2]) << 30)) & 0xffffffffffff;
    try std.testing.expectEqual(signedBits(-12, 56), low | (high << 8));
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> 23) & 0x7); // condition P0
    try std.testing.expectEqual(@as(u32, PT), (w[0] >> 12) & 0x7); // guard stays true
}

test "sass encodes IPA (interpolate attribute)" {
    var code: [16]u32 = undefined;
    var a = Assembler{ .code = &code };
    // IPA R0, a[0x80] - interpolate the first component of generic varying 0.
    a.ipa(0, ATTR_GENERIC0, .{ .wr_barrier = 0 });
    try std.testing.expectEqual(@as(u32, 0x326), code[0] & 0xfff); // OpIpa opcode
    try std.testing.expectEqual(@as(u32, PT), (code[0] >> 12) & 0x7); // predicate PT
    try std.testing.expectEqual(@as(u32, 0), (code[0] >> 16) & 0xff); // dst = R0
    // addr>>2 at bits 64..71 -> word 2, low byte
    try std.testing.expectEqual(@as(u32, ATTR_GENERIC0 >> 2), code[2] & 0xff);
    // loc(76..77)=0, freq(78..79)=0 -> bits 12..15 of word 2 all zero
    try std.testing.expectEqual(@as(u32, 0), (code[2] >> 12) & 0xf);
    // offset reg src = RZ at bits 32..39 -> word 1 low byte
    try std.testing.expectEqual(@as(u32, RZ), code[1] & 0xff);
    // pred_dst = PT(7) at bits 81..83 -> word 2 bits 17..19
    try std.testing.expectEqual(@as(u32, PT), (code[2] >> 17) & 0x7);

    // A second component into R1 from a[0x84] (the byte addr/4 = 0x21).
    a.ipa(1, ATTR_GENERIC0 + 4, .{});
    try std.testing.expectEqual(@as(u32, (ATTR_GENERIC0 + 4) >> 2), code[6] & 0xff);
    try std.testing.expectEqual(@as(u32, 1), (code[4] >> 16) & 0xff);
}

test "sass encodes a passthrough vertex shader (ald -> ast)" {
    var code: [64]u32 = undefined;
    var a = Assembler{ .code = &code };
    // Read the fetched position into R0..R3, write it to the position output.
    a.ald(0, ATTR_GENERIC0, 4, .{ .wr_barrier = 0 }); // load, set scoreboard 0
    a.ast(ATTR_POSITION, 0, 4, .{ .wait_mask = 1 }); // wait on scoreboard 0, store
    a.exit(.{ .stall = 1 });
    try std.testing.expectEqual(@as(usize, 12), a.dwords());
    try std.testing.expectEqual(@as(u32, 0x321), code[0] & 0xfff); // ALD opcode
    try std.testing.expectEqual(@as(u32, 0x322), code[4] & 0xfff); // AST opcode
    // ALD attribute addr (bits 40..49) = ATTR_GENERIC0
    try std.testing.expectEqual(@as(u32, ATTR_GENERIC0), (code[1] >> 8) & 0x3ff);
    // AST attribute addr = ATTR_POSITION
    try std.testing.expectEqual(@as(u32, ATTR_POSITION), (code[5] >> 8) & 0x3ff);
}

fn stallOf(code: []const u32, i: usize) u32 {
    return (code[i * 4 + 3] >> 9) & 0xf;
}
fn wrBarrierOf(code: []const u32, i: usize) u32 {
    return (code[i * 4 + 3] >> 14) & 0x7;
}
fn rdBarrierOf(code: []const u32, i: usize) u32 {
    return (code[i * 4 + 3] >> 17) & 0x7;
}
fn waitMaskOf(code: []const u32, i: usize) u32 {
    return (code[i * 4 + 3] >> 20) & 0x3f;
}

test "schedule packs independent instructions and spaces dependent ones" {
    var code: [64]u32 = undefined;
    var deps: [16]Dep = @splat(.{});
    var a = Assembler{ .code = &code, .deps = &deps };
    a.movImm(0, 1, .{});
    a.movImm(1, 2, .{}); // reads nothing the first one wrote
    a.iadd3(2, Src.reg(0), Src.reg(1), zero, .{}); // reads both
    a.exit(.{});
    try a.schedule();

    // The second move depends on nothing, so it issues on the next cycle.
    try std.testing.expectEqual(@as(u32, 1), stallOf(&code, 0));
    // MOV writes on the multiply pipe and IADD3 reads on the integer pipe, so
    // the add pays the cross-pipe cycle on top of the base latency.
    try std.testing.expectEqual(@as(u32, Latency.cross_pipe + Latency.margin), stallOf(&code, 1));
    // Nothing needs a scoreboard: every result here has a fixed latency.
    try std.testing.expectEqual(@as(u32, 7), wrBarrierOf(&code, 0));
    try std.testing.expectEqual(@as(u32, 0), waitMaskOf(&code, 2));
}

test "schedule gives a load a scoreboard and makes its reader wait" {
    var code: [64]u32 = undefined;
    var deps: [16]Dep = @splat(.{});
    var a = Assembler{ .code = &code, .deps = &deps };
    a.movImm(0, 0x1000, .{});
    a.movImm(1, 0, .{});
    a.ldg(4, 0, 0, .bits32, .{}); // instruction 2
    a.stg(0, 4, 0, .bits32, .{}); // instruction 3 reads R4
    a.exit(.{});
    try a.schedule();

    try std.testing.expectEqual(@as(u32, 0), wrBarrierOf(&code, 2));
    try std.testing.expectEqual(@as(u32, 0b1), waitMaskOf(&code, 3));
    // The store does not overwrite the address, so the load needs no read
    // scoreboard.
    try std.testing.expectEqual(@as(u32, 7), rdBarrierOf(&code, 2));
}

test "schedule guards a source that a later instruction overwrites" {
    var code: [64]u32 = undefined;
    var deps: [16]Dep = @splat(.{});
    var a = Assembler{ .code = &code, .deps = &deps };
    a.movImm(0, 0x1000, .{});
    a.movImm(1, 0, .{});
    a.ldg(4, 0, 0, .bits32, .{}); // instruction 2, address in R0:R1
    a.iadd3(0, Src.reg(0), Src.imm(4), zero, .{}); // instruction 3 steps R0
    a.exit(.{});
    try a.schedule();

    // The step must not overtake the load that still holds the address.
    const rd = rdBarrierOf(&code, 2);
    try std.testing.expect(rd != 7);
    try std.testing.expect(waitMaskOf(&code, 3) & (@as(u32, 1) << @intCast(rd)) != 0);
}

test "schedule drains every scoreboard at a branch" {
    var code: [64]u32 = undefined;
    var deps: [16]Dep = @splat(.{});
    var a = Assembler{ .code = &code, .deps = &deps };
    a.movImm(0, 0x1000, .{});
    a.movImm(1, 0, .{});
    const loop = a.here();
    a.ldg(4, 0, 0, .bits32, .{}); // instruction 2
    a.isetp(0, .lt, true, Src.reg(4), Src.imm(9), .{}); // instruction 3
    a.bra(loop, .{ .pred = 0 }); // instruction 4
    a.exit(.{});
    try a.schedule();

    // The compare reads the loaded value, so it waits on the load's scoreboard,
    // and the branch leaves nothing outstanding.
    try std.testing.expectEqual(@as(u32, 0b1), waitMaskOf(&code, 3));
    try std.testing.expect(deps[loop].target);
    // The branch covers the predicate the compare produced.
    try std.testing.expect(stallOf(&code, 3) >= Latency.pred);
}

// The three encodings below were each a silent wrong-answer bug in another
// implementation of this ISA (the vulcan compiler, on a 5070). None of them
// faults when wrong; the kernel just computes something else. They are pinned
// bit-for-bit here so a refactor cannot quietly reintroduce them.

test "IADD3 marks its extended form at bit 74, not in the predicate field" {
    var code: [16]u32 = undefined;
    var a = Assembler{ .code = &code };
    // The carry-consuming half of a 64-bit add.
    a.iadd3Carry(1, Src.reg(1), zero, zero, PT, 0, .{});
    // Bit 74 is the .X flag. Reading the 87..89 predicate as the selector
    // instead leaves this clear, and every 64-bit pointer add drops its carry.
    try std.testing.expectEqual(@as(u32, 1), (code[2] >> 10) & 1);
    try std.testing.expectEqual(@as(u32, 0), (code[2] >> 23) & 0x7); // carry-in P0
    try std.testing.expectEqual(@as(u32, 0), (code[2] >> 26) & 1); // not inverted

    // Without a carry-in the same field has to be clear, or an ordinary add
    // picks up a carry that is not there.
    a.iadd3(2, Src.reg(2), Src.imm(1), zero, .{});
    try std.testing.expectEqual(@as(u32, 0), (code[6] >> 10) & 1);
    try std.testing.expectEqual(@as(u32, PT), (code[6] >> 23) & 0x7);
    try std.testing.expectEqual(@as(u32, 1), (code[6] >> 26) & 1); // inverted, so false
}

test "LDG puts its uniform base at bit 32 where STG puts it at 64" {
    var code: [16]u32 = undefined;
    var a = Assembler{ .code = &code };
    a.ldg(4, 0, 0, .bits32, .{});
    // The load's uniform base sits at 32..39. Putting it at 64 instead lands on
    // the guard predicate field below and addresses through a stale register.
    try std.testing.expectEqual(@as(u32, URZ), code[1] & 0xff);
    try std.testing.expectEqual(@as(u32, 1), (code[2] >> 8) & 1); // 64-bit uniform

    // The store really does use 64..71, so the two are not interchangeable.
    a.stg(0, 4, 0, .bits32, .{});
    try std.testing.expectEqual(@as(u32, URZ), code[6] & 0xff);
    try std.testing.expectEqual(@as(u32, 1), (code[6] >> 8) & 1);
    try std.testing.expectEqual(@as(u32, 4), code[5] & 0xff); // data register at 32..39
}

test "LDG encodes its guard predicate backwards and its predicate destination as PT" {
    var code: [16]u32 = undefined;
    var a = Assembler{ .code = &code };
    a.ldg(4, 0, 0, .bits32, .{});
    // The guard at 64..66 counts down: an always-true guard is 0, not PT. Seven
    // there names a real predicate register and the load runs conditionally.
    try std.testing.expectEqual(@as(u32, 0), code[2] & 0x7);
    try std.testing.expectEqual(@as(u32, 0), (code[2] >> 3) & 1); // not inverted
    // The predicate destination is the ordinary way round, so it is PT for none.
    try std.testing.expectEqual(@as(u32, PT), (code[2] >> 17) & 0x7);
}
