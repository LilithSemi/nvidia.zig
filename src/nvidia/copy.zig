//! NVIDIA copy-engine (CE) method stream for Blackwell (BLACKWELL_DMA_COPY_B,
//! class 0xCAB5). Builds the LAUNCH_DMA sequence that detiles a block-linear
//! (GOB-tiled) color surface in VRAM into a PITCH-linear destination buffer,
//! performed entirely on the GPU's copy engine. This is the GPU-side analog of
//! the CPU de-swizzle in graphics.blColorPixelOffset / device.deswizzleBlockLinear.
//!
//! The CE runs on its own GPFIFO channel bound to NV2080_ENGINE_TYPE_COPY0; the
//! copy class is bound with SET_OBJECT on subchannel 4 (the copy engine's
//! canonical subchannel, nvk/nouveau SUBC_COPY) and every method here is emitted
//! on that subchannel. Emitting the copy class on subchannel 0 instead made the
//! host PBDMA reject the whole pushbuffer with Xid 32 (corrupted push buffer
//! stream) on the real GB20x; subchannel 4 is required. Method offsets are
//! NVC5B5_/NVCAB5_-derived (stable across CE generations); the block-size
//! encoding follows clcab5.h (Blackwell).

const std = @import("std");
const sdk = @import("sdk.zig");

const SUBCH = 4; // the copy engine's canonical subchannel (nvk/nouveau SUBC_COPY)

// CE method offsets (NVC5B5 / NVCAB5; stable across gens).
const SET_OBJECT = 0x0000;
const LAUNCH_DMA = 0x0300;
const SET_SEMAPHORE_A = 0x0240; // semaphore VA upper (16:0)
const SET_SEMAPHORE_B = 0x0244; // semaphore VA lower (31:0)
const SET_SEMAPHORE_PAYLOAD = 0x0248; // release payload (31:0)
const OFFSET_IN_UPPER = 0x0400;
const OFFSET_IN_LOWER = 0x0404;
const OFFSET_OUT_UPPER = 0x0408;
const OFFSET_OUT_LOWER = 0x040C;
const PITCH_IN = 0x0410;
const PITCH_OUT = 0x0414;
const LINE_LENGTH_IN = 0x0418;
const LINE_COUNT = 0x041C;
const SET_SRC_BLOCK_SIZE = 0x0728;
const SET_SRC_WIDTH = 0x072C;
const SET_SRC_HEIGHT = 0x0730;
const SET_SRC_DEPTH = 0x0734;
const SET_SRC_LAYER = 0x0738;
const SET_SRC_ORIGIN = 0x073C; // packed { x: 15:0, y: 31:16 }

// LAUNCH_DMA field encodings (clc5b5.h).
const DATA_TRANSFER_TYPE_NON_PIPELINED: u32 = 0x2; // 1:0
const FLUSH_ENABLE_TRUE: u32 = 1 << 2; // 2:2
const SRC_MEMORY_LAYOUT_BLOCKLINEAR: u32 = 0 << 7; // 7:7
const DST_MEMORY_LAYOUT_PITCH: u32 = 1 << 8; // 8:8
const MULTI_LINE_ENABLE_TRUE: u32 = 1 << 9; // 9:9
const SEMAPHORE_TYPE_RELEASE_ONE_WORD: u32 = 1 << 3; // 4:3: release a 1-word semaphore

// SET_SRC_BLOCK_SIZE field encodings: WIDTH 3:0, HEIGHT 7:4, DEPTH 11:8,
// GOB_HEIGHT 15:12, KIND_BPP 17:16 (clcab5.h).
const GOB_HEIGHT_FERMI_8: u32 = 1 << 12;
const KIND_BPP_BL_32: u32 = 0 << 16; // TuringColor2D (Blackwell >=4-byte color)

const GOB_WIDTH_BYTES: u32 = 64;

/// Parameters for a block-linear -> pitch detile copy of an A8R8G8B8 (4 byte/px)
/// color surface. The src tiling matches the block-linear color render targets
/// (graphics.beginBlockLinear / blColorPixelOffset): 1-GOB-wide x
/// `src_block_height_gobs`-tall blocks, GOB_HEIGHT_FERMI_8, TuringColor2D kind.
pub const Detile = struct {
    src_va: u64, // block-linear source surface VA
    dst_va: u64, // pitch-linear destination VA
    width: u32, // pixels per row
    height: u32, // rows
    dst_pitch: u32, // destination row stride in bytes (>= width*4)
    src_block_height_gobs: u32 = 16, // ZT_BLOCK_HEIGHT_GOBS (matches the color RT)
    // CE-native completion fence: when sem_va != 0, the copy LAUNCH_DMA releases
    // `sem_seq` to `sem_va` after the transfer flushes through L2. The CPU polls
    // that semaphore to know the detile landed. (The CE has no host WFI fence; the
    // engine releases its own semaphore - this is how nvk/nouveau fence a copy.)
    sem_va: u64 = 0,
    sem_seq: u32 = 0,
};

/// The GOB-aligned source row stride in bytes for a `width`-wide A8R8G8B8
/// block-linear surface (align(width*4, 64)). The CE uses the stride (divided by
/// bpp) as SET_SRC_WIDTH because the copy hardware has no tile-width concept.
pub fn srcRowStrideBytes(width: u32) u32 {
    return std.mem.alignForward(u32, width * 4, GOB_WIDTH_BYTES);
}

/// log2 of the block height in GOBs (e.g. 16 -> 4 = SIXTEEN_GOBS), encoded into
/// SET_SRC_BLOCK_SIZE.HEIGHT.
fn blockHeightField(gobs: u32) u32 {
    return std.math.log2_int(u32, gobs);
}

/// A CE method-stream builder. Bind the class with `setup`, append one or more
/// `detile` copies, finish with `fence`, then submit `dwords()` via nvidia.Queue.
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

    /// Bind the copy class (SET_OBJECT). Required once before any detile.
    pub fn setup(self: *Stream) void {
        self.m1(SET_OBJECT, sdk.BLACKWELL_DMA_COPY_B);
    }

    /// Emit one block-linear -> pitch detile copy of `d`. The src surface is read
    /// GOB-tiled (block-linear) and written row-major (pitch) to the destination.
    /// When `d.sem_va` is set, the copy also releases `d.sem_seq` to that VA on
    /// completion (the CE-native fence).
    pub fn detile(self: *Stream, d: Detile) void {
        const src_stride = srcRowStrideBytes(d.width); // align(w*4, 64)
        // Configure the CE-native release semaphore (before the launch).
        if (d.sem_va != 0) {
            self.mm(SET_SEMAPHORE_A, &.{
                @intCast((d.sem_va >> 32) & 0x1ffff), // SET_SEMAPHORE_A (upper 16:0)
                @truncate(d.sem_va), // SET_SEMAPHORE_B (lower)
                d.sem_seq, // SET_SEMAPHORE_PAYLOAD
            });
        }
        // Addresses, pitches, and the per-line geometry.
        self.mm(OFFSET_IN_UPPER, &.{
            @intCast(d.src_va >> 32), @truncate(d.src_va),
            @intCast(d.dst_va >> 32), @truncate(d.dst_va),
            src_stride,               d.dst_pitch,
            d.width * 4,              d.height,
        }); // OFFSET_IN/OUT_UPPER/LOWER, PITCH_IN/OUT, LINE_LENGTH_IN, LINE_COUNT (contiguous 0x400..0x41C)

        // Describe the block-linear source surface. The copy HW has no tile-width
        // concept, so SET_SRC_WIDTH is the row stride / bpp * bpp = the GOB-aligned
        // byte stride; HEIGHT/DEPTH bound the surface; ORIGIN is the top-left.
        const block_size = (blockHeightField(d.src_block_height_gobs) << 4) | GOB_HEIGHT_FERMI_8 | KIND_BPP_BL_32;
        self.m1(SET_SRC_BLOCK_SIZE, block_size);
        self.mm(SET_SRC_WIDTH, &.{
            src_stride, // SET_SRC_WIDTH (stride_el * bpp = byte stride)
            d.height, // SET_SRC_HEIGHT
            1, // SET_SRC_DEPTH
            0, // SET_SRC_LAYER
            0, // SET_SRC_ORIGIN (x=0, y=0)
        }); // 0x72C..0x73C contiguous

        var launch = DATA_TRANSFER_TYPE_NON_PIPELINED | FLUSH_ENABLE_TRUE |
            SRC_MEMORY_LAYOUT_BLOCKLINEAR | DST_MEMORY_LAYOUT_PITCH | MULTI_LINE_ENABLE_TRUE;
        if (d.sem_va != 0) launch |= SEMAPHORE_TYPE_RELEASE_ONE_WORD;
        self.m1(LAUNCH_DMA, launch);
    }

    pub fn dwords(self: *const Stream) u32 {
        return @intCast(self.n);
    }
};

test "CE detile stream encodes the block-linear -> pitch launch" {
    var buf: [64]u32 = undefined;
    var s = Stream{ .buf = &buf };
    s.setup();
    s.detile(.{ .src_va = 0x10000000, .dst_va = 0x20000000, .width = 64, .height = 64, .dst_pitch = 64 * 4, .sem_va = 0x30000000, .sem_seq = 0xc0de });
    // SET_OBJECT binds the copy class.
    try std.testing.expectEqual(sdk.gpfifo.methodHeader(SET_OBJECT, SUBCH, 1), buf[0]);
    try std.testing.expectEqual(sdk.BLACKWELL_DMA_COPY_B, buf[1]);
    // SET_SEMAPHORE_A/B/PAYLOAD (3 words) come first because sem_va is set.
    try std.testing.expectEqual(sdk.gpfifo.methodHeader(SET_SEMAPHORE_A, SUBCH, 3), buf[2]);
    try std.testing.expectEqual(@as(u32, 0x30000000), buf[4]); // SET_SEMAPHORE_B (lower)
    try std.testing.expectEqual(@as(u32, 0xc0de), buf[5]); // SET_SEMAPHORE_PAYLOAD
    // Walk to the OFFSET_IN_UPPER block and check the geometry words.
    var i: usize = 6;
    try std.testing.expectEqual(sdk.gpfifo.methodHeader(OFFSET_IN_UPPER, SUBCH, 8), buf[i]);
    try std.testing.expectEqual(@as(u32, 0), buf[i + 1]); // src hi
    try std.testing.expectEqual(@as(u32, 0x10000000), buf[i + 2]); // src lo
    try std.testing.expectEqual(@as(u32, 0x20000000), buf[i + 4]); // dst lo
    try std.testing.expectEqual(srcRowStrideBytes(64), buf[i + 5]); // PITCH_IN
    try std.testing.expectEqual(@as(u32, 64 * 4), buf[i + 6]); // PITCH_OUT
    try std.testing.expectEqual(@as(u32, 64 * 4), buf[i + 7]); // LINE_LENGTH_IN
    try std.testing.expectEqual(@as(u32, 64), buf[i + 8]); // LINE_COUNT
    i += 9;
    // SET_SRC_BLOCK_SIZE: HEIGHT=SIXTEEN_GOBS(4)<<4 | FERMI_8(1<<12) | BL_32(0).
    try std.testing.expectEqual(sdk.gpfifo.methodHeader(SET_SRC_BLOCK_SIZE, SUBCH, 1), buf[i]);
    try std.testing.expectEqual(@as(u32, (4 << 4) | (1 << 12)), buf[i + 1]);
    i += 2;
    try std.testing.expectEqual(sdk.gpfifo.methodHeader(SET_SRC_WIDTH, SUBCH, 5), buf[i]);
    i += 6;
    // LAUNCH_DMA: NON_PIPELINED | FLUSH | SRC_BL | DST_PITCH | MULTI_LINE
    // | SEMAPHORE_TYPE_RELEASE_ONE_WORD(1<<3) = 0x306 | 0x8 = 0x30E.
    try std.testing.expectEqual(sdk.gpfifo.methodHeader(LAUNCH_DMA, SUBCH, 1), buf[i]);
    try std.testing.expectEqual(@as(u32, 0x30E), buf[i + 1]);
}
