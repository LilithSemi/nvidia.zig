//! Minimal NVIDIA 3D-engine renderer: rasterizes solid/gradient triangles via
//! scissor-bounded GPU CLEARs (no shaders). Verified live on Blackwell (3D class
//! 0xce97). The GPU's clear unit does the pixel writes; the CPU only computes
//! per-scanline spans + colors. Method offsets are NV9097-derived (stable across
//! 3D class generations); a SET_OBJECT binds the CLASS id (not the RM handle).

const std = @import("std");
const sdk = @import("sdk.zig");

pub const BLACKWELL_A: sdk.NvV32 = 0xce97; // Blackwell 3D class (GB20x)

// NV9097-derived method offsets.
const SET_OBJECT = 0x0000;
const RT_ADDR = 0x0800; // 9-dword color-target block
const CLEAR_COLOR = 0x0d80; // 4 RGBA floats
const SCREEN_SCISSOR_HORIZ = 0x0ff4; // (x | width<<16), then (y | height<<16)
const CLEAR_FLAGS = 0x10f8;
const RT_CONTROL = 0x121c;
const ZETA_ENABLE = 0x1538;
const COND_MODE = 0x1558;
const MULTISAMPLE_MODE = 0x15d0;
const CLEAR_BUFFERS = 0x19d0;
const QUERY = 0x1b00; // addr_hi, addr_lo, sequence, get

const FORMAT_A8R8G8B8 = 0xcf;
const TILE_LINEAR = 0x1000;
const CLEAR_BUFFERS_RGBA = 0x3c;
const CLEAR_FLAGS_SCISSOR = 0x100;
const COND_MODE_ALWAYS = 1;
const QUERY_GET_FENCE = 0x10; // MODE_WRITE | FENCE: flush GR writes then report

fn f2u(f: f32) u32 {
    return @bitCast(f);
}

/// Byte pitch of a w-wide A8R8G8B8 render target. The 3D engine requires the
/// pitch aligned to 256 bytes; the framebuffer's row stride must match this
/// (not w*4) when reading it back / presenting it.
pub fn pitchBytes(w: u32) u32 {
    return (w * 4 + 0xff) & ~@as(u32, 0xff);
}

/// A method-stream (pushbuffer) builder. Write methods, then submit the
/// populated dword range (`dwords()`) via nvidia.Queue.submit.
pub const Stream = struct {
    buf: []u32,
    n: usize = 0,

    fn header(addr: u32, count: u32) u32 {
        return (1 << 29) | (count << 16) | (addr >> 2); // INCR method, subchannel 0
    }
    pub fn m1(self: *Stream, addr: u32, v: u32) void {
        self.buf[self.n] = header(addr, 1);
        self.buf[self.n + 1] = v;
        self.n += 2;
    }
    pub fn mm(self: *Stream, addr: u32, vals: []const u32) void {
        self.buf[self.n] = header(addr, @intCast(vals.len));
        self.n += 1;
        for (vals) |v| {
            self.buf[self.n] = v;
            self.n += 1;
        }
    }
    /// Non-incrementing (P_1INC / NON_INCR) method: opcode 3 in the top bits, so every data word
    /// lands at the SAME register `addr`. Used for LOAD_CONSTANT_BUFFER / LOAD_ROOT_TABLE, where a
    /// run of dwords streams into one window register.
    pub fn ni(self: *Stream, addr: u32, vals: []const u32) void {
        self.buf[self.n] = (3 << 29) | (@as(u32, @intCast(vals.len)) << 16) | (addr >> 2);
        self.n += 1;
        for (vals) |v| {
            self.buf[self.n] = v;
            self.n += 1;
        }
    }
    pub fn reset(self: *Stream) void {
        self.n = 0;
    }
    pub fn dwords(self: *const Stream) u32 {
        return @intCast(self.n);
    }
};

/// Bind the 3D `class` and set a linear A8R8G8B8 render target at GPU VA
/// `rt_gpu_va` of size `w`x`h`. Must be first in the stream.
pub fn begin(s: *Stream, class: sdk.NvV32, rt_gpu_va: u64, w: u32, h: u32) void {
    s.m1(SET_OBJECT, class);
    s.m1(COND_MODE, COND_MODE_ALWAYS);
    s.m1(RT_CONTROL, 1);
    s.mm(RT_ADDR, &.{
        @intCast(rt_gpu_va >> 32), @truncate(rt_gpu_va),
        pitchBytes(w),   h, // pitch must be 256-aligned, not w*4
        FORMAT_A8R8G8B8, TILE_LINEAR,
        1,               0,
        0,
    });
    s.m1(ZETA_ENABLE, 0);
    s.m1(MULTISAMPLE_MODE, 0);
}

/// Clear the whole render target to (r,g,b,a) (0..1). Use ALONE: a full-RT clear
/// pipelines against any following scissored clears (fillSpan/fillTriangle) and
/// races them, so don't chain it before a triangle. To clear a background under
/// a triangle, clear on the CPU instead and let fillTriangle write only the spans.
pub fn clear(s: *Stream, w: u32, h: u32, r: f32, g: f32, b: f32, a: f32) void {
    s.mm(SCREEN_SCISSOR_HORIZ, &.{ w << 16, h << 16 }); // a full clear needs a screen scissor
    s.m1(CLEAR_FLAGS, CLEAR_FLAGS_SCISSOR);
    s.mm(CLEAR_COLOR, &.{ f2u(r), f2u(g), f2u(b), f2u(a) });
    s.m1(CLEAR_BUFFERS, CLEAR_BUFFERS_RGBA);
}

/// Enable scissor-bounded clears (call once before a run of fillSpan).
pub fn scissorsOn(s: *Stream) void {
    s.m1(CLEAR_FLAGS, CLEAR_FLAGS_SCISSOR);
}

/// GPU-clear the span [x, x+width) x [y, y+1) to (r,g,b,a).
pub fn fillSpan(s: *Stream, x: u32, y: u32, width: u32, r: f32, g: f32, b: f32, a: f32) void {
    s.mm(SCREEN_SCISSOR_HORIZ, &.{ x | (width << 16), y | (1 << 16) });
    s.mm(CLEAR_COLOR, &.{ f2u(r), f2u(g), f2u(b), f2u(a) });
    s.m1(CLEAR_BUFFERS, CLEAR_BUFFERS_RGBA);
}

/// Finish the stream with a fenced query: flush the GR writes to memory and
/// write `seq` to GPU VA `sem_gpu_va` (poll it on a CPU mapping for completion).
pub fn fence(s: *Stream, sem_gpu_va: u64, seq: u32) void {
    s.mm(QUERY, &.{ @intCast(sem_gpu_va >> 32), @truncate(sem_gpu_va), seq, QUERY_GET_FENCE });
}

pub const Vertex = struct { x: f32, y: f32, r: f32, g: f32, b: f32 };

fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) f32 {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

/// The inclusive x-span [lo, hi] of triangle `tri` covered at scanline `y` in a
/// w-wide target, or null if the scanline misses the triangle. Same coverage
/// rule fillTriangle uses, so a presenter can copy exactly the GPU-written spans.
pub fn spanAt(tri: [3]Vertex, w: u32, y: u32) ?[2]u32 {
    const area = edge(tri[0].x, tri[0].y, tri[1].x, tri[1].y, tri[2].x, tri[2].y);
    if (area == 0) return null;
    const fy: f32 = @as(f32, @floatFromInt(y)) + 0.5;
    var xl: i32 = -1;
    var xr: i32 = -1;
    var x: u32 = 0;
    while (x < w) : (x += 1) {
        const fx: f32 = @as(f32, @floatFromInt(x)) + 0.5;
        const w0 = edge(tri[1].x, tri[1].y, tri[2].x, tri[2].y, fx, fy) / area;
        const w1 = edge(tri[2].x, tri[2].y, tri[0].x, tri[0].y, fx, fy) / area;
        const w2 = edge(tri[0].x, tri[0].y, tri[1].x, tri[1].y, fx, fy) / area;
        if (w0 >= 0 and w1 >= 0 and w2 >= 0) {
            if (xl < 0) xl = @intCast(x);
            xr = @intCast(x);
        }
    }
    if (xl < 0) return null;
    return .{ @intCast(xl), @intCast(xr) };
}

/// Rasterize a gradient triangle (`tri`, pixel space) into the current render
/// target via per-scanline scissor-bounded GPU clears. Call after begin()
/// (+ an optional clear()); finish with fence(). The CPU finds each scanline's
/// span + interpolated color, the GPU writes the pixels.
pub fn fillTriangle(s: *Stream, tri: [3]Vertex, w: u32, h: u32) void {
    scissorsOn(s);
    const area = edge(tri[0].x, tri[0].y, tri[1].x, tri[1].y, tri[2].x, tri[2].y);
    if (area == 0) return;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        const span = spanAt(tri, w, y) orelse continue;
        const fy: f32 = @as(f32, @floatFromInt(y)) + 0.5;
        const mx = (@as(f32, @floatFromInt(span[0])) + @as(f32, @floatFromInt(span[1]))) * 0.5 + 0.5;
        const w0 = edge(tri[1].x, tri[1].y, tri[2].x, tri[2].y, mx, fy) / area;
        const w1 = edge(tri[2].x, tri[2].y, tri[0].x, tri[0].y, mx, fy) / area;
        const w2 = edge(tri[0].x, tri[0].y, tri[1].x, tri[1].y, mx, fy) / area;
        fillSpan(
            s,
            span[0],
            y,
            span[1] - span[0] + 1,
            w0 * tri[0].r + w1 * tri[1].r + w2 * tri[2].r,
            w0 * tri[0].g + w1 * tri[1].g + w2 * tri[2].g,
            w0 * tri[0].b + w1 * tri[1].b + w2 * tri[2].b,
            1.0,
        );
    }
}

test "stream encodes an increasing method header + data" {
    var buf: [16]u32 = undefined;
    var s = Stream{ .buf = &buf };
    s.m1(0x1558, 1);
    try std.testing.expectEqual(@as(u32, (1 << 29) | (1 << 16) | (0x1558 >> 2)), buf[0]);
    try std.testing.expectEqual(@as(u32, 1), buf[1]);
    s.mm(0x0d80, &.{ 1, 2, 3, 4 });
    try std.testing.expectEqual(@as(u32, (1 << 29) | (4 << 16) | (0x0d80 >> 2)), buf[2]);
    try std.testing.expectEqual(@as(u32, 7), s.dwords()); // 2 (m1) + 1 header + 4 data
}
