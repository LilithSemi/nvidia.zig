//! Minimal hand-assembler for NVIDIA SASS, the native GPU ISA. The instruction
//! encoding is shared Volta..Blackwell (only per-instruction latencies differ),
//! so one encoder serves all of them. Each instruction is 128 bits (4 dwords).
//! Verified live on Blackwell (sm_120): a MOV/STG/EXIT kernel runs on the SMs.
//!
//! Scheduling: each instruction carries a `Control` with a stall count (cycles
//! before the next instruction) and optional scoreboard barriers. The default
//! stall is conservative - enough to cover a back-to-back fixed-latency register
//! dependency. Variable-latency ops (global loads, texture) need scoreboard
//! barriers via `Control.wr_barrier`/`wait_mask`; a real scheduler is future work.

const std = @import("std");

pub const RZ: u8 = 255; // the zero GPR
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

/// Per-instruction scheduling. On Blackwell (sm>=120) the yield/reuse bits are
/// dropped, so only the stall, barriers, and wait mask are encoded.
pub const Control = struct {
    stall: u4 = 15, // conservative; covers a fixed-latency register dependency
    wr_barrier: u3 = 7, // write scoreboard to set on completion (7 = none)
    rd_barrier: u3 = 7, // read scoreboard to set (7 = none)
    wait_mask: u6 = 0, // scoreboards to wait on before issue
};

fn putControl(inst: []u32, c: Control) void {
    setBits(inst, 105, 4, c.stall);
    setBits(inst, 110, 3, c.wr_barrier);
    setBits(inst, 113, 3, c.rd_barrier);
    setBits(inst, 116, 6, c.wait_mask);
}

fn putPredTrue(inst: []u32) void {
    setBits(inst, 12, 3, PT); // unconditional (predicate PT)
}

/// Emits 128-bit SASS instructions into a caller-provided dword buffer (4 dwords
/// per instruction). Tracks the highest GPR touched so the launch descriptor can
/// derive a register count.
pub const Assembler = struct {
    code: []u32,
    n: usize = 0, // dwords emitted
    max_reg: u8 = 0, // highest GPR index written/read (excluding RZ)

    fn next(self: *Assembler) []u32 {
        const w = self.code[self.n..][0..4];
        @memset(w, 0);
        self.n += 4;
        return w;
    }
    fn note(self: *Assembler, reg: u8) void {
        if (reg != RZ and reg > self.max_reg) self.max_reg = reg;
    }

    /// Program size in dwords (bytes = dwords()*4).
    pub fn dwords(self: *const Assembler) usize {
        return self.n;
    }
    /// A safe register count for the QMD: highest GPR used + 1, rounded up to the
    /// hardware allocation granularity (multiples of 8, minimum 16).
    pub fn registerCount(self: *const Assembler) u32 {
        const used = @as(u32, self.max_reg) + 1;
        return @max(16, (used + 7) & ~@as(u32, 7));
    }

    /// MOV dst, imm32 - load a 32-bit immediate into a GPR.
    pub fn movImm(self: *Assembler, dst: u8, imm: u32, c: Control) void {
        self.note(dst);
        const w = self.next();
        setBits(w, 0, 9, 0x002); // ALU MOV
        setBits(w, 9, 3, 4); // form: 32-bit immediate
        putPredTrue(w);
        setBits(w, 16, 8, dst);
        setBits(w, 32, 32, imm);
        setBits(w, 72, 4, 0xf); // all quad lanes
        putControl(w, c);
    }

    /// MOV dst, src - copy a 32-bit GPR. ALU MOV (0x002) with the register source
    /// form (form 1): src register at bits 32..39, no immediate.
    pub fn movReg(self: *Assembler, dst: u8, src: u8, c: Control) void {
        self.note(dst);
        self.note(src);
        const w = self.next();
        setBits(w, 0, 9, 0x002); // ALU MOV
        setBits(w, 9, 3, 1); // form: register source
        putPredTrue(w);
        setBits(w, 16, 8, dst);
        setBits(w, 32, 8, src); // src register (encode_alu_reg bits 32..40)
        setBits(w, 72, 4, 0xf); // all quad lanes
        putControl(w, c);
    }

    /// STG.E.STRONG.SYS [addr:addr+1], data - store the 32-bit GPR `data` to the
    /// 64-bit global address held in the register pair (addr, addr+1). `addr`
    /// must be even.
    pub fn stgU32(self: *Assembler, addr: u8, data: u8, c: Control) void {
        self.note(addr + 1);
        self.note(data);
        const w = self.next();
        setBits(w, 0, 12, 0x986); // STG global (UGPR form)
        putPredTrue(w);
        setBits(w, 24, 8, addr);
        setBits(w, 90, 1, 1); // 64-bit GPR address (addr:addr+1)
        setBits(w, 64, 8, RZ); // URZ uniform base
        setBits(w, 72, 1, 1); // 64-bit uniform
        setBits(w, 32, 8, data);
        setBits(w, 73, 3, 4); // type B32
        setBits(w, 77, 4, 0xa); // order STRONG, scope SYS
        setBits(w, 84, 3, 1); // eviction NORMAL
        setBits(w, 91, 1, 1); // UGPR mode (required, or the SM traps)
        putControl(w, c);
    }

    /// S2R dst, sysval - read a special/system register (e.g. the vertex ID)
    /// into a GPR. Variable latency: set a `wr_barrier` and have the consumer
    /// wait on it.
    pub fn s2r(self: *Assembler, dst: u8, sysval: u8, c: Control) void {
        self.note(dst);
        const w = self.next();
        setBits(w, 0, 12, 0x919);
        putPredTrue(w);
        setBits(w, 16, 8, dst);
        setBits(w, 72, 8, sysval);
        putControl(w, c);
    }

    /// ALD dst..dst+comps-1, a[addr] - load `comps` shader input-attribute words
    /// (e.g. a fetched vertex attribute) into consecutive GPRs. Variable latency.
    pub fn ald(self: *Assembler, dst: u8, addr: u16, comps: u8, c: Control) void {
        self.note(dst + comps - 1);
        const w = self.next();
        setBits(w, 0, 12, 0x321);
        putPredTrue(w);
        setBits(w, 16, 8, dst);
        setBits(w, 32, 8, RZ); // vertex (RZ: not per-vertex addressed)
        setBits(w, 24, 8, RZ); // dynamic offset (RZ: static)
        setBits(w, 40, 10, addr);
        setBits(w, 74, 2, comps - 1);
        putControl(w, c);
    }

    /// AST o[addr], data..data+comps-1 - store `comps` GPRs to a shader
    /// output attribute (e.g. the clip-space position at ATTR_POSITION).
    pub fn ast(self: *Assembler, addr: u16, data: u8, comps: u8, c: Control) void {
        self.note(data + comps - 1);
        const w = self.next();
        setBits(w, 0, 12, 0x322);
        putPredTrue(w);
        setBits(w, 32, 8, data);
        setBits(w, 64, 8, RZ); // vertex
        setBits(w, 24, 8, RZ); // dynamic offset
        setBits(w, 40, 10, addr);
        setBits(w, 74, 2, comps - 1);
        putControl(w, c);
    }

    /// IPA dst, a[addr] - interpolate one component (one dword) of a fragment
    /// input attribute. On SM70+ a single IPA with InterpLoc::Default(0) and
    /// InterpFreq::Pass(0) does the full perspective-correct interpolation
    /// implicitly (the barycentrics are hardware-provided, no explicit 1/w
    /// multiply, no load_barycentric setup). `addr` is the attribute BYTE
    /// address (must be 4-aligned); the encoder stores addr>>2. Variable latency
    /// like ALD: set a `wr_barrier` and drain it before consuming the result.
    pub fn ipa(self: *Assembler, dst: u8, addr: u16, c: Control) void {
        std.debug.assert(addr % 4 == 0);
        self.note(dst);
        const w = self.next();
        setBits(w, 0, 12, 0x326); // OpIpa
        putPredTrue(w);
        setBits(w, 16, 8, dst); // dst
        setBits(w, 64, 8, addr >> 2); // attribute addr / 4
        setBits(w, 76, 2, 0); // loc = InterpLoc::Default
        setBits(w, 78, 2, 0); // freq = InterpFreq::Pass (implicit perspective)
        setBits(w, 32, 8, RZ); // offset reg src = RZ (required for Default loc)
        setBits(w, 81, 3, PT); // pred_dst = none (PT)
        putControl(w, c);
    }

    /// EXIT - terminate the warp.
    pub fn exit(self: *Assembler, c: Control) void {
        const w = self.next();
        setBits(w, 0, 12, 0x94d);
        putPredTrue(w);
        setBits(w, 87, 3, 7);
        putControl(w, c);
    }
};

/// Shader attribute addresses (the `addr` for ald/ast). Position output is at
/// 0x70; generic varyings / vertex inputs start at 0x80.
pub const ATTR_POSITION: u16 = 0x70;
pub const ATTR_GENERIC0: u16 = 0x80;
/// System-value index for S2R: the vertex ID.
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
