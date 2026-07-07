//! NVIDIA graphics (3D) shader-pipeline scaffolding for Blackwell (class 0xce97).
//!
//! This is rung 1 of the real shaded-draw path: the Shader Program Header (SPH)
//! builder and the 3D pipeline/draw method offsets. A draw with programmable
//! shaders needs, on top of this: the full 3D engine init, vertex + fragment
//! shaders assembled via sass.zig (with the I/O ops - S2R, attribute store,
//! interpolation), the vertex pipeline + viewport, and DRAW_VERTEX_ARRAY. Those
//! are the hardware bring-up rungs that build on the pieces here.
//!
//! Method offsets are NV9097 (inherited unchanged by Blackwell's 3D class). The
//! SPH layout is SPHV3 (cla097sph.h); on Volta+ (sm>=73) the header version is 4.

const std = @import("std");
const threed = @import("threed.zig");

// --- 3D pipeline + draw method offsets (NV9097 / clce97) ---
pub const INVALIDATE_SHADER_CACHES = 0x021c;
pub const SET_PROGRAM_REGION_A = 0x1608; // + _B at 0x160c; base of the shader heap
pub const DRAW_VERTEX_ARRAY = 0x0d78; // vertex count to draw
pub const BEGIN = 0x1618; // start a primitive (topology in the low bits)
pub const END = 0x1614; // end the primitive

// Vertex input: SET_VERTEX_ATTRIBUTE_A(i) describes attribute i (stream, offset,
// component widths, numeric type); SET_VERTEX_STREAM_A_*(j) binds stream j's
// buffer + stride.
pub fn vertexAttribute(i: u32) u32 {
    return 0x1160 + i * 4;
}
pub fn vertexStreamFormat(j: u32) u32 {
    return 0x1c00 + j * 16; // stride in the low bits
}
pub fn vertexStreamLocationHi(j: u32) u32 {
    return 0x1c04 + j * 16; // buffer VA[39:32]
}
pub fn vertexStreamLocationLo(j: u32) u32 {
    return 0x1c08 + j * 16; // buffer VA[31:0]
}
// Turing+ vertex-stream extent: SET_VERTEX_STREAM_SIZE is the byte SIZE of the
// bound buffer (NOT the pre-Turing SET_VERTEX_STREAM_LIMIT end-address at 0x1f00,
// which hangs the Data Assembler on Blackwell). Without a non-zero size the DA
// fetches nothing and the vertex shader's attribute loads read out of range.
pub fn vertexStreamSizeHi(j: u32) u32 {
    return 0x0600 + j * 8; // size[39:32]
}
pub fn vertexStreamSizeLo(j: u32) u32 {
    return 0x0604 + j * 8; // size[31:0]
}

// Viewport: SET_VIEWPORT_SCALE_X/OFFSET_X(j) map clip space to pixels;
// CLIP_HORIZONTAL/VERTICAL(j) is the scissor-like viewport clip.
pub fn viewportScaleX(j: u32) u32 {
    return 0x0a00 + j * 32; // + Y at +4, Z at +8
}
pub fn viewportOffsetX(j: u32) u32 {
    return 0x0a0c + j * 32; // + Y at +4, Z at +8
}
pub fn viewportClipHorizontal(j: u32) u32 {
    return 0x0c00 + j * 16; // x0 (15:0) | width (31:16); vertical at +4
}
pub const SET_WINDOW_ORIGIN = 0x13ac;

// SET_VERTEX_ATTRIBUTE_A component-width values (NVA097 COMPONENT_BIT_WIDTHS, in
// bits 26:21). The DA fetches exactly this many 32-bit components per vertex; a
// width WIDER than the buffer provides for the last vertex over-reads past the
// stream SIZE and the Data Assembler faults (a null read at VA 0). So the width
// must match the real attribute's component count (a vec2 = R32_G32, vec3 =
// R32_G32_B32), not a fixed vec4.
// (NVC597_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_*, from the Blackwell class
// header clc597.h - these are NOT sequential.)
pub const ATTR_R32 = 0x12;
pub const ATTR_R32_G32 = 0x04;
pub const ATTR_R32_G32_B32 = 0x02;
pub const ATTR_R32_G32_B32_A32 = 0x01;
// SET_VERTEX_ATTRIBUTE numeric type (in bits 30:27): FLOAT.
pub const ATTR_NUM_TYPE_FLOAT = 0x07;

/// The COMPONENT_BIT_WIDTHS value for `comps` (1..4) 32-bit float components.
pub fn attrBitWidths(comps: u32) u32 {
    return switch (comps) {
        1 => ATTR_R32,
        2 => ATTR_R32_G32,
        3 => ATTR_R32_G32_B32,
        else => ATTR_R32_G32_B32_A32,
    };
}

/// Per-pipeline-slot shader-bind methods: SET_PIPELINE_*(j) = base + j*STRIDE.
/// `j` is the pipeline slot (0 = vertex-cull/A, 1 = vertex/B, ... 5 = pixel).
pub const pipeline = struct {
    pub const STRIDE = 64;
    pub const SHADER = 0x2000; // enable (bit 0) + type (bits 4:7)
    pub const PROGRAM = 0x2004; // pre-Turing 32-bit SPH offset (NOT used on Volta+)
    pub const REGISTER_COUNT = 0x200c;
    pub const BINDING = 0x2010; // constant-buffer bind group (per stage, not the slot)
    pub const PROGRAM_ADDRESS_A = 0x2014; // Volta+: full 64-bit SPH address, hi then lo
    pub const PROGRAM_ADDRESS_B = 0x2018;
    pub fn shader(j: u32) u32 {
        return SHADER + j * STRIDE;
    }
    pub fn program(j: u32) u32 {
        return PROGRAM + j * STRIDE;
    }
    /// Volta+/Blackwell shader address (verified live): write the full 64-bit SPH
    /// address as { hi, lo } starting here. `j` is the slot = the shader-type value.
    pub fn programAddress(j: u32) u32 {
        return PROGRAM_ADDRESS_A + j * STRIDE;
    }
    pub fn registerCount(j: u32) u32 {
        return REGISTER_COUNT + j * STRIDE;
    }
    pub fn binding(j: u32) u32 {
        return BINDING + j * STRIDE;
    }
};

// Shader-local-memory setup (the SMs need this to run any shader).
pub const SET_SHADER_LOCAL_MEMORY_WINDOW = 0x077c; // = 0xff << 24
pub const SET_SHADER_LOCAL_MEMORY_A = 0x0790; // A..E: base hi/lo, size hi/lo, per-warp

// --- ZETA (depth/stencil) surface + the fixed-function depth test (NVA097) ---
// The ZETA surface is the depth buffer. Unlike the COLOR render target (whose
// SET_COLOR_TARGET_MEMORY has a LAYOUT=PITCH bit, i.e. linear), the ZETA surface
// has NO pitch/linear option - SET_ZT_BLOCK_SIZE only selects a block-linear GOB
// tiling, so the depth buffer MUST be block-linear. The CPU never reads it back
// (only the COLOR output is checked), so the exact byte layout is internal: the
// clear writes it block-linear and the depth test reads/writes it block-linear,
// so depth occlusion is self-consistent regardless of the tiling.
pub const SET_ZT_A = 0x0fe0; // depth surface address[39:32] (upper byte)
pub const SET_ZT_B = 0x0fe4; // depth surface address[31:0] (lower)
pub const SET_ZT_FORMAT = 0x0fe8; // V (bits 4:0): ZF32 = 0x0A
pub const SET_ZT_BLOCK_SIZE = 0x0fec; // width/height/depth GOB counts (block-linear)
pub const SET_ZT_ARRAY_PITCH = 0x0ff0; // per-layer pitch (in units of 4 bytes)
pub const SET_ZT_SIZE_A = 0x1228; // row stride in elements (NOT pixel width; see bindDepth)
pub const SET_ZT_SIZE_B = 0x122c; // height (pixels)
pub const SET_ZT_SIZE_C = 0x1230; // third_dimension | control
pub const SET_ZT_LAYER = 0x179c; // base array layer (0)
pub const SET_ZT_SPARSE = 0x1208; // sparse enable - MUST be disabled for a plain ZETA
pub const SET_ZT_SELECT = 0x1538; // target_count: 0 = no ZETA, 1 = one ZETA bound
// SEPARATE STENCIL plane (NVCD97, Blackwell): an S8 (or separate depth+stencil) ZETA needs the
// stencil plane bound HERE in ADDITION to SET_ZT_* - without it the ROP-Z reads the stencil at
// VA 0 (Xid 31 MMU fault from GPCCLIENT_PROP). For an S8-only surface the stencil plane IS the
// bound surface (same address + stride as the ZT plane).
pub const SET_ST_A = 0x0f00; // stencil plane address[39:32]
pub const SET_ST_B = 0x0f04; // stencil plane address[31:0]
pub const SET_ST_BLOCK_SIZE = 0x0f08; // width/height/depth GOB counts
pub const SET_ST_ARRAY_PITCH = 0x0f0c; // per-layer pitch (units of 4 bytes)
pub const SET_ST_SIZE_A = 0x120c; // row stride in elements
pub const SET_ST_SIZE_B = 0x1210; // height (pixels)
pub const SET_DEPTH_TEST = 0x12cc; // enable (bit 0)
pub const SET_DEPTH_WRITE = 0x12e8; // enable (bit 0)
pub const SET_DEPTH_FUNC = 0x130c; // V = the OGL compare op (LESS = 0x201)
pub const SET_Z_CLEAR_VALUE = 0x0d90; // depth clear value (f32 bits)
pub const SET_STENCIL_CLEAR_VALUE = 0x0da0; // stencil clear value (NVC597, V = bits 7:0)
pub const CLEAR_SURFACE = 0x19d0; // Z_ENABLE bit 0; STENCIL_ENABLE bit 1; RGBA at bits 2..5
pub const ZT_FORMAT_ZF32 = 0x0A; // SET_ZT_FORMAT_V_ZF32
pub const ZT_FORMAT_Z24S8 = 0x14; // SET_ZT_FORMAT_V_Z24S8 (24-bit depth + 8-bit stencil, 4 B/px)
pub const ZT_FORMAT_ZF32_X24S8 = 0x19; // SET_ZT_FORMAT_V_ZF32_X24S8 (fp32 depth + X24S8, 8 B/px)
pub const ZT_FORMAT_S8 = 0x17; // SET_ZT_FORMAT_V_S8 (stencil-only, 1 B/px, single plane)

// SET_COLOR_TARGET_FORMAT_V values (NVC597). A8R8G8B8 is the default 8-bit color RT; the
// float variants render + read back at full precision (HDR render targets). RF32/RF16 store
// R,G,B,A IN ORDER (no B<->R swap, unlike A8R8G8B8's little-endian [B,G,R,A]).
pub const CT_FORMAT_A8R8G8B8 = 0xcf; // 4 B/px (the historical color RT)
pub const CT_FORMAT_RF16_GF16_BF16_AF16 = 0xca; // rgba16f, 8 B/px
pub const CT_FORMAT_RF32_GF32_BF32_AF32 = 0xc0; // rgba32f, 16 B/px

// Stencil methods (NVC597 / BLACKWELL_A). func + ops take OGL enum values, exactly like
// SET_DEPTH_FUNC. The stencil lives in the bound ZETA's stencil component (Z24S8 here).
pub const SET_STENCIL_TEST = 0x1380; // enable (bit 0)
pub const SET_STENCIL_OP_FAIL = 0x1384; // OGL stencil-op (stencil test fails)
pub const SET_STENCIL_OP_ZFAIL = 0x1388; // OGL stencil-op (stencil passes, depth fails)
pub const SET_STENCIL_OP_ZPASS = 0x138c; // OGL stencil-op (both pass)
pub const SET_STENCIL_FUNC = 0x1390; // OGL compare func (NEVER=0x200 .. ALWAYS=0x207)
pub const SET_STENCIL_FUNC_REF = 0x1394; // reference value (u8)
pub const SET_STENCIL_FUNC_MASK = 0x1398; // compare/read mask (u8)
pub const SET_STENCIL_MASK = 0x139c; // write mask (u8)
pub const SET_TWO_SIDED_STENCIL_TEST = 0x1594; // enable (bit 0); when 1 the BACK_* methods apply
// BACK-face stencil methods (used only when SET_TWO_SIDED_STENCIL_TEST=1). The front methods
// above then govern only front-facing primitives, these the back-facing ones.
pub const SET_BACK_STENCIL_OP_FAIL = 0x1598;
pub const SET_BACK_STENCIL_OP_ZFAIL = 0x159c;
pub const SET_BACK_STENCIL_OP_ZPASS = 0x15a0;
pub const SET_BACK_STENCIL_FUNC = 0x15a4; // OGL compare func
pub const SET_BACK_STENCIL_FUNC_REF = 0x0f54; // reference (u8)
pub const SET_BACK_STENCIL_FUNC_MASK = 0x0f5c; // compare/read mask (u8)
pub const SET_BACK_STENCIL_MASK = 0x0f58; // write mask (u8)

/// OGL stencil-op values for SET_STENCIL_OP_* (NVC597_SET_STENCIL_OP_FAIL_V_OGL_*). These
/// mirror hal.StencilOp; the HAL maps to these.
pub const StencilOp = enum(u32) {
    keep = 0x1E00,
    zero = 0x0000,
    replace = 0x1E01,
    incr_clamp = 0x1E02, // OGL_INCRSAT
    decr_clamp = 0x1E03, // OGL_DECRSAT
    invert = 0x150A,
    incr_wrap = 0x8507, // OGL_INCR
    decr_wrap = 0x8508, // OGL_DECR
};

/// OGL depth-compare-function values for SET_DEPTH_FUNC (NVA097_SET_DEPTH_FUNC_V).
/// These mirror VkCompareOp / hal.CompareOp; the HAL maps to these.
pub const DepthFunc = enum(u32) {
    never = 0x200,
    less = 0x201,
    equal = 0x202,
    less_or_equal = 0x203,
    greater = 0x204,
    not_equal = 0x205,
    greater_or_equal = 0x206,
    always = 0x207,
};

/// The block height (in GOBs) the ZETA depth surface is tiled at. A GOB is 64
/// bytes wide x 8 rows; a block is BLOCK_HEIGHT_GOBS GOBs tall. 16 is nvk's common
/// depth choice and is large enough for the render targets here (the surface is
/// over-allocated to the block footprint so the GPU never writes past it).
pub const ZT_BLOCK_HEIGHT_GOBS: u32 = 16;
const GOB_WIDTH_BYTES: u32 = 64;
const GOB_HEIGHT_ROWS: u32 = 8;

/// The byte size of a `w`x`h` ZF32 (4 bytes/px) block-linear ZETA depth surface,
/// at block height ZT_BLOCK_HEIGHT_GOBS. The width is rounded up to a GOB (64
/// bytes = 16 px) and the height to a full block (BLOCK_HEIGHT_GOBS * 8 rows), so
/// the allocation covers the whole tiled footprint the GPU addresses.
pub fn ztSizeBytes(w: u32, h: u32) u32 {
    return ztSizeBytesBpp(w, h, 4);
}

/// The byte size of a `w`x`h` block-linear ZETA at `bpp` bytes/px (4 for ZF32/Z24S8,
/// 8 for ZF32_X24S8) and block height ZT_BLOCK_HEIGHT_GOBS.
pub fn ztSizeBytesBpp(w: u32, h: u32, bpp: u32) u32 {
    const row_bytes = std.mem.alignForward(u32, w * bpp, GOB_WIDTH_BYTES);
    const block_rows = ZT_BLOCK_HEIGHT_GOBS * GOB_HEIGHT_ROWS;
    const tiled_h = std.mem.alignForward(u32, h, block_rows);
    return row_bytes * tiled_h;
}

/// Bind `zt_va` as the ZETA depth surface (ZF32, block-linear) of `w`x`h` and
/// enable it (SET_ZT_SELECT target_count=1). Call after threed.begin (which set
/// SET_ZT_SELECT=0 / no depth) when the framebuffer has a depth attachment. The
/// block size + array pitch describe the block-linear tiling the GPU reads/writes.
pub fn bindDepth(s: *threed.Stream, zt_va: u64, w: u32, h: u32) void {
    bindZetaFmt(s, zt_va, w, h, ZT_FORMAT_ZF32, 4);
}

// SET_ZT_FORMAT_STENCIL_IS_SEPARATE (NVCE97, bit 8): the stencil lives in the SET_ST_* plane, not
// packed into the ZT surface. Set for combined depth+stencil so the ROP reads/writes stencil from
// the separate S8 plane - plain ZF32 (no stencil bits) makes the stencil test SILENTLY IGNORED
// (the combined oracle rendered content everywhere, un-clipped, until this bit was set).
pub const SET_ZT_FORMAT_STENCIL_IS_SEPARATE: u32 = 1 << 8;

/// Combined depth+stencil (Blackwell separate-z/s model): a depth ZETA at `depth_va` PLUS a
/// SEPARATE S8 stencil plane at `stencil_va`. Verified live on the RTX 5070: depth occludes AND
/// stencil clips in one framebuffer, Xid-clean, with the depth surface on the GENERIC page kind
/// (NO packed-format page kind the open RM cannot set - the depth-wall stays dodged).
/// TWO details are load-bearing (each cost an Xid 69 / wrong-output debug cycle):
///   1. FORMAT CODE = ZF32_X24S8 (the combined z+s code nvk uses for the packet), NOT plain ZF32:
///      ZF32|STENCIL_IS_SEPARATE faults Xid 69 (the HW needs a recognized z+s format), and plain
///      ZF32 (no STENCIL_IS_SEPARATE) makes the stencil test SILENTLY IGNORED (content un-clipped).
///   2. STENCIL_IS_SEPARATE (bit 8) so the stencil lives in the SET_ST_* S8 plane, while the ZT
///      surface is sized at 4 B/px (ZF32 depth only) - the X24S8 part is not backed here.
pub fn bindDepthStencilSeparate(s: *threed.Stream, depth_va: u64, stencil_va: u64, w: u32, h: u32) void {
    bindZetaFmt(s, depth_va, w, h, ZT_FORMAT_ZF32_X24S8 | SET_ZT_FORMAT_STENCIL_IS_SEPARATE, 4);
    bindStencilPlane(s, stencil_va, w, h);
}

/// Bind an S8 (stencil-only, single-plane, 1 B/px) ZETA for a stencil-only UI clip. The
/// packed depth+stencil formats (Z24S8, ZF32_X24S8) FAULT Xid 69 / ErrorCode 0x13 with the
/// GENERIC block-linear page kind this open-RM path maps, because a stencil-containing
/// surface needs its format's own PTE kind (which the open RM would not let us set). S8 is a
/// SINGLE stencil plane - the untried format - so try it: if the ROP accepts an S8-only ZETA
/// with the generic kind, stencil-only clipping works without any kind override. Call after
/// threed.begin.
pub fn bindStencilZeta(s: *threed.Stream, zt_va: u64, w: u32, h: u32) void {
    bindZetaFmt(s, zt_va, w, h, ZT_FORMAT_S8, 1);
    // For an S8-only surface the separate stencil plane IS this surface (same VA + stride).
    bindStencilPlane(s, zt_va, w, h);
}

/// Bind a SEPARATE S8 stencil plane at `st_va` via the SET_ST_* methods. Blackwell reads the
/// stencil from this plane whenever the stencil test is enabled - required for BOTH an S8-only
/// ZETA (the plane IS the ZT surface) AND a combined depth+stencil framebuffer (a ZF32 depth ZT
/// plane via bindDepth + this SEPARATE S8 stencil plane). Without it the ROP-Z reads the stencil
/// at VA 0 (Xid 31 GPCCLIENT_PROP MMU fault). Mirrors nvk's SET_ST block (separate z/s model:
/// the depth and stencil live in DISTINCT surfaces, not a packed Z24S8 whose page kind the open
/// RM will not set). Call after bindDepth (for combined) or bindZetaFmt(S8) (for stencil-only).
pub fn bindStencilPlane(s: *threed.Stream, st_va: u64, w: u32, h: u32) void {
    s.mm(SET_ST_A, &.{ @intCast(st_va >> 32), @truncate(st_va) });
    const height_field: u32 = std.math.log2_int(u32, ZT_BLOCK_HEIGHT_GOBS);
    s.m1(SET_ST_BLOCK_SIZE, (height_field << 4));
    s.m1(SET_ST_ARRAY_PITCH, ztSizeBytesBpp(w, h, 1) >> 2);
    const row_stride_el: u32 = std.mem.alignForward(u32, w * 1, 64) / 1;
    s.m1(SET_ST_SIZE_A, row_stride_el);
    s.m1(SET_ST_SIZE_B, h);
}

fn bindZetaFmt(s: *threed.Stream, zt_va: u64, w: u32, h: u32, format: u32, bpp: u32) void {
    s.mm(SET_ZT_A, &.{ @intCast(zt_va >> 32), @truncate(zt_va) });
    s.m1(SET_ZT_FORMAT, format);
    // BLOCK_SIZE: width = 1 GOB, height = ZT_BLOCK_HEIGHT_GOBS, depth = 1 GOB.
    // The height field encodes log2(gobs): ONE=0, TWO=1, FOUR=2, EIGHT=3, SIXTEEN=4.
    // This MUST match the block height ztSizeBytes used to size the allocation (nvk
    // sets it from the surface's actual tiling - level->tiling.y_log2).
    const height_field: u32 = std.math.log2_int(u32, ZT_BLOCK_HEIGHT_GOBS); // 16 -> 4
    s.m1(SET_ZT_BLOCK_SIZE, (height_field << 4)); // width=ONE_GOB(0), depth=ONE_GOB(0)
    s.m1(SET_ZT_ARRAY_PITCH, ztSizeBytesBpp(w, h, bpp) >> 2); // per-layer pitch in 4-byte units
    s.m1(SET_ZT_SELECT, 1); // target_count = 1: one ZETA bound
    // SET_ZT_SIZE_A is the ROW STRIDE IN ELEMENTS, not the pixel width (the Z/S hardware has
    // no tile-width concept, so nvk passes row_stride_B / bpp). row_stride_B = align(w*bpp,
    // GOB=64); stride in elements = align(w*bpp,64)/bpp.
    const row_stride_el: u32 = std.mem.alignForward(u32, w * bpp, 64) / bpp;
    s.m1(SET_ZT_SIZE_A, row_stride_el);
    s.m1(SET_ZT_SIZE_B, h);
    // THIRD_DIMENSION = base_layer + layer_count = 0 + 1 = 1, control = THIRD_DIMENSION
    // _DEFINES_ARRAY_SIZE (0). (nvk uses control=0 here, NOT ARRAY_SIZE_IS_ONE.)
    s.m1(SET_ZT_SIZE_C, 1);
    s.m1(SET_ZT_LAYER, 0); // base array layer
    s.m1(0x19cc, 0); // SET_Z_COMPRESSION = FALSE (uncompressed depth)
    // SET_ZT_SPARSE = DISABLED. nvk emits this on Maxwell+; without it the engine may
    // treat the ZT as sparse and reject the depth draw (the Xid 69 / ErrorCode 0x9c
    // Class Error). This is the method the from-scratch bind was missing.
    s.m1(SET_ZT_SPARSE, 0);
    // SET_DEPTH_BOUNDS_TEST (0x19bc) = disabled. nvk emits this; with a ZETA bound
    // the ROP would otherwise run a depth-bounds test against an unset bounds range.
    s.m1(0x19bc, 0);
}

/// Enable the fixed-function depth test: SET_DEPTH_TEST on, SET_DEPTH_FUNC = the
/// compare op, SET_DEPTH_WRITE per write_enable. The VS's interpolated gl_Position.z
/// (mapped to [0,1] by the viewport) is compared against the stored depth.
pub fn setDepthTest(s: *threed.Stream, func: DepthFunc, write_enable: bool) void {
    s.m1(SET_DEPTH_TEST, 1);
    s.m1(SET_DEPTH_FUNC, @intFromEnum(func));
    s.m1(SET_DEPTH_WRITE, if (write_enable) 1 else 0);
}

// Occlusion query (ZPASS pixel count): the ROP counts samples that pass depth/stencil into a
// running hardware counter while SET_ZPASS_PIXEL_COUNT is enabled. A REPORT_SEMAPHORE with the
// ZPASS_PIXEL_CNT64 report writes that cumulative count to memory (GL_ANY_SAMPLES_PASSED reads the
// begin/end delta). Mirrors nvk's query pool (SET_ZPASS_PIXEL_COUNT + REPORT_ZPASS_PIXEL_CNT64).
pub const SET_ZPASS_PIXEL_COUNT = 0x1514;
pub const CLEAR_REPORT_VALUE = 0x1530;
const CLEAR_REPORT_VALUE_TYPE_ZPASS = 0x01;
const SET_REPORT_SEMAPHORE_A = 0x1b00; // A=off_hi, B=off_lo, C=payload, D=execute

/// Enable/disable ZPASS pixel counting (the ROP increments its counter for passing samples).
pub fn setZpassPixelCount(s: *threed.Stream, enable: bool) void {
    s.m1(SET_ZPASS_PIXEL_COUNT, if (enable) 1 else 0);
}

/// Reset the ZPASS hardware counter to 0.
pub fn clearZpassCount(s: *threed.Stream) void {
    s.m1(CLEAR_REPORT_VALUE, CLEAR_REPORT_VALUE_TYPE_ZPASS);
}

/// Stall the method pipe until the engine is idle. Used to order the ZPASS report's memory write
/// (a posted, FLUSH_DISABLE SET_REPORT_SEMAPHORE) BEFORE the following fence the CPU waits on:
/// without it the report write had not committed when the fence signalled, so occlusionSampleCount
/// read the untouched buffer (count=0, ts=0) - an intermittent "0 samples passed".
pub fn waitForIdle(s: *threed.Stream) void {
    s.m1(0x0110, 0); // WAIT_FOR_IDLE
}

/// Write the current ZPASS pixel count (a 4-word {count64, timestamp64} report) to `va` via
/// SET_REPORT_SEMAPHORE. SET_REPORT_SEMAPHORE_D fields (mirrors nvk's occlusion query): OPERATION=
/// REPORT_ONLY(0x2, bits 1:0), FLUSH_DISABLE(bit 2), PIPELINE_LOCATION=ALL(0xF, bits 15:12), REPORT=
/// ZPASS_PIXEL_CNT64(0x15, bits 27:23), STRUCTURE_SIZE=FOUR_WORDS(0, bit 28). NOTE: OPERATION must be
/// REPORT_ONLY, not RELEASE - RELEASE with a report type faults Xid 69 Class Error.
pub fn reportZpass(s: *threed.Stream, va: u64) void {
    const d: u32 = 0x2 | (1 << 2) | (0xF << 12) | (0x15 << 23);
    s.mm(SET_REPORT_SEMAPHORE_A, &.{ @intCast(va >> 32), @truncate(va), 0, d });
}

// Transform feedback (GPU stream-out): the vertex pipeline streams selected VS outputs to buffers.
// SET_STREAM_OUT_BUFFER_*(j) at 0x0380+j*32; SET_STREAM_OUT_CONTROL_*(j) at 0x0700+j*16;
// SET_STREAM_OUT_LAYOUT_SELECT(b,k) at 0x2800+b*128+k*4; SET_RASTER_ENABLE at 0x037c.
pub const SET_RASTER_ENABLE = 0x037c;

/// Rasterizer discard (GL_RASTERIZER_DISCARD): 0 = the pipeline runs the VS (and stream-out) but the
/// rasterizer/ROP are OFF, so nothing is drawn - the transform-feedback capture path.
pub fn setRasterEnable(s: *threed.Stream, enable: bool) void {
    s.m1(SET_RASTER_ENABLE, if (enable) 1 else 0);
}

/// GLOBAL stream-out enable (SET_STREAM_OUTPUT, 0x0744): the master switch - stream-out captures
/// NOTHING without this on, even with buffers + control + layout set. nvk sets it in BeginTransformFeedback.
pub fn setStreamOutputEnable(s: *threed.Stream, enable: bool) void {
    s.m1(0x0744, if (enable) 1 else 0);
}

/// Bind stream-out buffer `idx` at `va` (`size` bytes) and reset its write pointer to the start.
pub fn setStreamOutBuffer(s: *threed.Stream, idx: u32, va: u64, size: u32) void {
    const base = 0x0380 + idx * 32;
    s.m1(base, 1); // ENABLE = TRUE
    s.mm(base + 4, &.{ @intCast(va >> 32), @truncate(va) }); // ADDRESS_A(hi), _B(lo)
    s.m1(base + 12, size); // SIZE (bytes)
    s.m1(base + 16, 0); // LOAD_WRITE_POINTER = 0 (write from the start)
}

/// Disable stream-out buffer `idx` (so a later normal draw is not streamed).
pub fn disableStreamOutBuffer(s: *threed.Stream, idx: u32) void {
    s.m1(0x0380 + idx * 32, 0);
}

/// Stream-out control for buffer `b`: which vertex `stream`, how many f32 `components` per vertex,
/// and the per-vertex `stride` in bytes.
pub fn setStreamOutControl(s: *threed.Stream, b: u32, stream: u32, components: u32, stride: u32) void {
    const base = 0x0700 + b * 16;
    s.m1(base, stream);
    s.m1(base + 4, components);
    s.m1(base + 8, stride);
}

/// The stream-out layout for buffer `b`: `attr` is one byte per streamed component, each the VS
/// output ATTRIBUTE INDEX (byte address/4) that feeds that SO slot. Uploaded packed 4 bytes/dword.
pub fn setStreamOutLayout(s: *threed.Stream, b: u32, attr: []const u8) void {
    var dws: [32]u32 = undefined;
    const n = (attr.len + 3) / 4;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var v: u32 = 0;
        var k: usize = 0;
        while (k < 4) : (k += 1) {
            const idx = i * 4 + k;
            if (idx < attr.len) v |= @as(u32, attr[idx]) << @intCast(k * 8);
        }
        dws[i] = v;
    }
    s.mm(0x2800 + b * 128, dws[0..n]);
}

pub const SET_CT_WRITE = 0x1a00; // per-target(i*4) color write enables: R@0 G@4 B@8 A@12

/// Set the per-channel color write mask (glColorMask) for color target `index`. Each channel's
/// enable is one bit (R bit0, G bit4, B bit8, A bit12). Emitted per-pipeline; the default is all
/// channels enabled (0x1111). A stencil-only mask pass sets all-false (0) to write no color.
pub fn setColorWriteMask(s: *threed.Stream, index: u32, r: bool, g: bool, b: bool, a: bool) void {
    const v: u32 = (@as(u32, @intFromBool(r)) << 0) | (@as(u32, @intFromBool(g)) << 4) |
        (@as(u32, @intFromBool(b)) << 8) | (@as(u32, @intFromBool(a)) << 12);
    s.m1(SET_CT_WRITE + index * 4, v);
}

pub const SET_POLY_OFFSET_FILL = 0x0dc8; // ENABLE bit 0 (glPolygonOffset for filled polys)
pub const SET_POLY_OFFSET_POINT = 0x0dc0;
pub const SET_POLY_OFFSET_LINE = 0x0dc4;
pub const SET_DEPTH_BIAS = 0x15bc; // constant factor (f32)
pub const SET_SLOPE_SCALE_DEPTH_BIAS = 0x156c; // slope-scaled factor (f32)
pub const SET_DEPTH_BIAS_CLAMP = 0x187c; // clamp (f32)
pub const SET_DEPTH_BIAS_CONTROL = 0x1110; // DEPTH_FORMAT_DEPENDENT bit 0

/// Configure depth bias (glPolygonOffset). `enable` gates POLY_OFFSET for all three primitive
/// modes; the constant + slope factors go to SET_DEPTH_BIAS / SET_SLOPE_SCALE_DEPTH_BIAS as f32,
/// with DEPTH_FORMAT_DEPENDENT so the constant is scaled by the depth format's least-representable
/// value (GL/Vulkan `r`). Mirrors nvk's depth-bias block. Emitted per-pipeline in the depth branch.
pub fn setDepthBias(s: *threed.Stream, enable: bool, constant: f32, slope: f32, clamp: f32) void {
    const en: u32 = if (enable) 1 else 0;
    s.m1(SET_POLY_OFFSET_POINT, en);
    s.m1(SET_POLY_OFFSET_LINE, en);
    s.m1(SET_POLY_OFFSET_FILL, en);
    if (enable) {
        s.m1(SET_DEPTH_BIAS_CONTROL, 1); // DEPTH_FORMAT_DEPENDENT = TRUE (format's r)
        s.m1(SET_DEPTH_BIAS, f2u(constant));
        s.m1(SET_SLOPE_SCALE_DEPTH_BIAS, f2u(slope));
        s.m1(SET_DEPTH_BIAS_CLAMP, f2u(clamp));
    }
}

pub const SET_ZCULL_BOUNDS = 0x196c; // z_min_unbounded (bit 0) + z_max_unbounded (bit 4)
pub const SET_API_MANDATED_EARLY_Z = 0x0210; // 1 = force EARLY depth test/write (pre-FS)

/// Configure the ROP-Z for a gl_FragDepth (depth-replace) draw. Two things break when the
/// FS writes its own depth and the defaults are left in place:
///   1. EARLY-Z: the ROP tests+WRITES depth BEFORE the FS using the interpolated z. A shader
///      that replaces the depth cannot participate, so the early WRITE is suppressed and no
///      late WRITE is configured -> the ZETA keeps the clear value -> occlusion degrades to
///      pure paint order. Forcing API_MANDATED_EARLY_Z = 0 makes the ROP do LATE-Z (test AND
///      write AFTER the FS, using the shader's depth). nvk sets this per shader.
///   2. ZCULL: the coarse per-tile early reject keys off the interpolated-z bounds, which a
///      shader depth breaks; unbounding it (z_min/z_max_unbounded) lets the fine late-Z govern.
/// Pass `replace`=true when the FS writes gl_FragDepth. (Left set for later non-replace draws
/// it only costs the early-Z perf optimization, never correctness, so it need not be restored.)
pub fn setDepthReplace(s: *threed.Stream, replace: bool) void {
    if (!replace) return;
    s.m1(SET_API_MANDATED_EARLY_Z, 0); // late-Z: test + write after the FS with the shader depth
    s.m1(SET_ZCULL_BOUNDS, 1 | (1 << 4)); // z_min + z_max unbounded
}

/// Emit a VALID but DISABLED depth state (test off, a valid compare func, no write). When a
/// ZETA is bound for a stencil-only draw the depth test isn't used, but the ROP-Z still
/// validates the depth state at draw time - leaving SET_DEPTH_FUNC unset (0, not a valid OGL
/// func) faults. nvk always emits the full depth state with the ZETA, so we do too.
pub fn setDepthDisabled(s: *threed.Stream) void {
    s.m1(SET_DEPTH_TEST, 0);
    s.m1(SET_DEPTH_FUNC, @intFromEnum(DepthFunc.always));
    s.m1(SET_DEPTH_WRITE, 0);
}

/// Clear the bound ZETA depth surface to `value` (e.g. 1.0 at render-pass begin):
/// set the Z clear value then CLEAR_SURFACE with only the Z bit. Must follow
/// bindDepth (so a ZETA is bound) and a SET_SURFACE_CLIP covering the area.
pub fn clearDepth(s: *threed.Stream, value: f32) void {
    s.m1(SET_Z_CLEAR_VALUE, f2u(value));
    s.m1(CLEAR_SURFACE, 1); // Z_ENABLE bit 0 only
}

/// Enable the fixed-function stencil test (single face, applied via the front methods).
/// `func` is the OGL compare value (reuses the DepthFunc enum), the ops are OGL stencil-op
/// values, and ref/compare_mask/write_mask are 8-bit. A bound ZETA with a stencil component
/// (Z24S8) must already be set; the ROP reads/writes the stencil there.
pub fn setStencilTest(s: *threed.Stream, func: DepthFunc, fail: StencilOp, zfail: StencilOp, zpass: StencilOp, ref: u8, compare_mask: u8, write_mask: u8) void {
    s.m1(SET_STENCIL_TEST, 1);
    // SINGLE-sided: initDrawState enables SET_TWO_SIDED_STENCIL_TEST, which would require
    // the BACK_* stencil methods to be defined or the draw faults (Xid 69 / ErrorCode 0x13).
    // The HAL StencilState is single-face, so disable two-sided and let the FRONT methods
    // below govern both windings.
    s.m1(SET_TWO_SIDED_STENCIL_TEST, 0);
    s.m1(SET_STENCIL_OP_FAIL, @intFromEnum(fail));
    s.m1(SET_STENCIL_OP_ZFAIL, @intFromEnum(zfail));
    s.m1(SET_STENCIL_OP_ZPASS, @intFromEnum(zpass));
    s.m1(SET_STENCIL_FUNC, @intFromEnum(func));
    s.m1(SET_STENCIL_FUNC_REF, ref);
    s.m1(SET_STENCIL_FUNC_MASK, compare_mask);
    s.m1(SET_STENCIL_MASK, write_mask);
}

/// Enable TWO-SIDED stencil (GLES glStencilFuncSeparate / glStencilOpSeparate, Vulkan front/back
/// VkStencilOpState): the FRONT methods govern front-facing primitives, the BACK_* methods the
/// back-facing ones. Each face has its own func/ops/ref/masks. Used for e.g. shadow-volume z-fail.
pub fn setStencilTestTwoSided(
    s: *threed.Stream,
    f_func: DepthFunc,
    f_fail: StencilOp,
    f_zfail: StencilOp,
    f_zpass: StencilOp,
    f_ref: u8,
    f_cmask: u8,
    f_wmask: u8,
    b_func: DepthFunc,
    b_fail: StencilOp,
    b_zfail: StencilOp,
    b_zpass: StencilOp,
    b_ref: u8,
    b_cmask: u8,
    b_wmask: u8,
) void {
    s.m1(SET_STENCIL_TEST, 1);
    s.m1(SET_TWO_SIDED_STENCIL_TEST, 1);
    // FRONT face.
    s.m1(SET_STENCIL_OP_FAIL, @intFromEnum(f_fail));
    s.m1(SET_STENCIL_OP_ZFAIL, @intFromEnum(f_zfail));
    s.m1(SET_STENCIL_OP_ZPASS, @intFromEnum(f_zpass));
    s.m1(SET_STENCIL_FUNC, @intFromEnum(f_func));
    s.m1(SET_STENCIL_FUNC_REF, f_ref);
    s.m1(SET_STENCIL_FUNC_MASK, f_cmask);
    s.m1(SET_STENCIL_MASK, f_wmask);
    // BACK face.
    s.m1(SET_BACK_STENCIL_OP_FAIL, @intFromEnum(b_fail));
    s.m1(SET_BACK_STENCIL_OP_ZFAIL, @intFromEnum(b_zfail));
    s.m1(SET_BACK_STENCIL_OP_ZPASS, @intFromEnum(b_zpass));
    s.m1(SET_BACK_STENCIL_FUNC, @intFromEnum(b_func));
    s.m1(SET_BACK_STENCIL_FUNC_REF, b_ref);
    s.m1(SET_BACK_STENCIL_FUNC_MASK, b_cmask);
    s.m1(SET_BACK_STENCIL_MASK, b_wmask);
}

/// Clear the bound ZETA's stencil component to `value`: set the stencil clear value then
/// CLEAR_SURFACE with the STENCIL_ENABLE bit (bit 1). Must follow the ZETA bind + a valid
/// SET_SURFACE_CLIP (like clearDepth).
pub fn clearStencil(s: *threed.Stream, value: u8) void {
    s.m1(SET_STENCIL_CLEAR_VALUE, value);
    s.m1(CLEAR_SURFACE, 2); // STENCIL_ENABLE bit 1 only
}

// --- block-linear color render targets (required when pairing with a ZETA) ---
//
// THE depth wall, root-caused: a draw with a ZETA selected (SET_ZT_SELECT=1)
// faults Xid 69 / Class Error / ErrorCode 0x9c if the COLOR render target is
// LINEAR (pitch). nvk encodes this rule directly (nvk_rendering_linear: "Depth
// and stencil are never linear", and the linear color fast-path is disabled the
// moment a depth attachment exists). The ROP requires the color + depth targets
// to share a tiled (block-linear) layout. So a depth-tested draw MUST render into
// a block-linear color surface; the CPU reads it back by de-swizzling the GOBs.
//
// The block-linear color surface uses the SAME GOB tiling as the ZETA (1 GOB
// wide x ZT_BLOCK_HEIGHT_GOBS tall blocks, GOB_TYPE_FERMI_8), so ztSizeBytes
// also sizes the color surface and blColorRowStrideEl gives its element stride.

/// Element stride (in pixels) of a `w`-wide block-linear color surface: the GOB-aligned row
/// width. SET_COLOR_TARGET_WIDTH takes this for a block-linear target (the ROP has no tile-width
/// concept, like the ZETA's SET_ZT_SIZE_A). `bpp` = bytes per pixel (4 = A8R8G8B8, 8 = rgba16f,
/// 16 = rgba32f).
pub fn blColorRowStrideElBpp(w: u32, bpp: u32) u32 {
    return std.mem.alignForward(u32, w * bpp, GOB_WIDTH_BYTES) / bpp;
}
pub fn blColorRowStrideEl(w: u32) u32 {
    return blColorRowStrideElBpp(w, 4);
}

/// Emit the BLOCK-LINEAR color render target block (RT slot 0) + the standard
/// begin state (SET_OBJECT, COND_MODE, RT_CONTROL, ZT off, multisample off). Use
/// in place of threed.begin when the draw will bind a ZETA. The surface at
/// `rt_va` must be allocated block-linear (ztSizeBytes footprint, BIG pages).
pub fn beginBlockLinear(s: *threed.Stream, class: u32, rt_va: u64, w: u32, h: u32) void {
    beginBlockLinearFmt(s, class, rt_va, w, h, CT_FORMAT_A8R8G8B8, 4);
}

/// `beginBlockLinear` with an explicit color-target FORMAT + bytes-per-pixel, so a float RT
/// (RF16/RF32) renders at full precision. bpp drives the GOB-aligned WIDTH stride + ARRAY_PITCH.
pub fn beginBlockLinearFmt(s: *threed.Stream, class: u32, rt_va: u64, w: u32, h: u32, ct_format: u32, bpp: u32) void {
    s.m1(0x0000, class); // SET_OBJECT = the 3D class id
    s.m1(0x1558, 1); // SET_RENDER_ENABLE_C COND_MODE = ALWAYS
    s.m1(0x121c, 1); // SET_CT_SELECT target_count = 1
    const height_field: u32 = std.math.log2_int(u32, ZT_BLOCK_HEIGHT_GOBS); // 16 -> 4
    s.mm(0x0800, &.{
        @intCast(rt_va >> 32), @truncate(rt_va), // SET_COLOR_TARGET_A/B (address)
        blColorRowStrideElBpp(w, bpp), // WIDTH = GOB-aligned element stride (block-linear)
        h, // HEIGHT (pixels)
        ct_format, // FORMAT (A8R8G8B8 / RF16x4 / RF32x4)
        (height_field << 4), // MEMORY: block_width=ONE_GOB(0), block_height, layout=BLOCKLINEAR(bit12=0)
        1, // THIRD_DIMENSION
        ztSizeBytesBpp(w, h, bpp) >> 2, // ARRAY_PITCH (4-byte units)
        0, // LAYER
    });
    s.m1(0x1538, 0); // SET_ZT_SELECT = 0 (a ZETA is bound later if depth is used)
    s.m1(0x15d0, 0); // SET_ANTI_ALIAS / MULTISAMPLE_MODE = 1x1
}

/// MRT: bind a BLOCK-LINEAR A8R8G8B8 color surface at render-target `slot` (1..7). Each
/// color-target block is 0x40 bytes apart (SET_COLOR_TARGET_A(slot) at 0x0800 + slot*0x40),
/// same 9-dword layout as beginBlockLinear's RT0. Call after beginBlockLinear (which binds
/// slot 0), then setColorTargetCount(N) to tell the ROP how many targets are live. The FS's
/// SPH omap_targets must declare each target too. Every MRT surface uses the SAME GOB tiling
/// as RT0 so blColorPixelOffset de-swizzles it on readback.
pub fn bindColorTargetBlockLinear(s: *threed.Stream, slot: u32, rt_va: u64, w: u32, h: u32) void {
    const height_field: u32 = std.math.log2_int(u32, ZT_BLOCK_HEIGHT_GOBS);
    s.mm(0x0800 + slot * 0x40, &.{
        @intCast(rt_va >> 32), @truncate(rt_va), // SET_COLOR_TARGET_A/B (address)
        blColorRowStrideEl(w), // WIDTH = GOB-aligned element stride (block-linear)
        h, // HEIGHT (pixels)
        0xcf, // FORMAT = A8R8G8B8
        (height_field << 4), // MEMORY: block-linear
        1, // THIRD_DIMENSION
        ztSizeBytes(w, h) >> 2, // ARRAY_PITCH (4-byte units)
        0, // LAYER
    });
}

/// MRT: set how many color targets the ROP reads AND the output-slot -> physical-target map
/// (SET_CT_SELECT). target_count is bits 3:0; target0..target7 are 3-bit fields at 4 + i*3
/// selecting which physical CT each shader output slot writes. Setting ONLY the count leaves
/// slot 1+ mapping to physical CT0 (so extra outputs vanish) - we emit the identity map
/// (slot i -> target i). Must match the bound color surfaces AND the FS's SPH omap_targets.
pub fn setColorTargetCount(s: *threed.Stream, count: u32) void {
    var v: u32 = count & 0xf; // TARGET_COUNT
    var i: u32 = 0;
    while (i < 8) : (i += 1) v |= (i & 0x7) << @intCast(4 + i * 3); // TARGET_i = i (identity)
    s.m1(0x121c, v);
}

/// De-swizzle pixel (x, y) from a block-linear A8R8G8B8 surface produced by
/// beginBlockLinear (GOB_TYPE_FERMI_8, 1-GOB-wide x ZT_BLOCK_HEIGHT_GOBS-tall
/// blocks). A GOB is 64 bytes (16 px) wide x 8 rows with the fixed Fermi/Turing
/// intra-GOB byte swizzle. Returns the packed pixel. Verified live on Blackwell
/// (a green triangle's center reads 0xff00ff00, the clear corner 0xff202020).
pub fn blColorPixelOffset(x: u32, y: u32, w: u32) usize {
    return blColorPixelOffsetBpp(x, y, w, 4);
}

/// `blColorPixelOffset` for a `bpp`-byte-per-pixel color surface (4 = A8R8G8B8, 8 = rgba16f,
/// 16 = rgba32f). Returns the byte offset of the texel's FIRST byte; the texel's `bpp` bytes are
/// CONTIGUOUS from there, because the Mesa TuringColor2D layout copies 16-byte lines (so an 8- or
/// 16-byte texel, 16-aligned within the GOB, never straddles a line boundary).
pub fn blColorPixelOffsetBpp(x: u32, y: u32, w: u32, bpp: u32) usize {
    const gob_w: u32 = GOB_WIDTH_BYTES; // 64
    const gob_h: u32 = GOB_HEIGHT_ROWS; // 8
    const row_bytes = std.mem.alignForward(u32, w * bpp, gob_w);
    const gobs_per_row = row_bytes / gob_w;
    const block_bytes = gob_w * gob_h * ZT_BLOCK_HEIGHT_GOBS;
    const bx = x * bpp;
    const gob_col = bx / gob_w;
    const gob_row = y / gob_h;
    const block_row = gob_row / ZT_BLOCK_HEIGHT_GOBS;
    const gob_in_block = gob_row % ZT_BLOCK_HEIGHT_GOBS;
    const block_base = (block_row * gobs_per_row + gob_col) * block_bytes;
    const gob_base = block_base + gob_in_block * (gob_w * gob_h);
    const xb = bx % gob_w;
    const yb = y % gob_h;
    const off = blGobByteOffset(xb, yb);
    return @intCast(gob_base + off);
}

/// Intra-GOB byte offset of linear byte-column `xb` (0..63) at row `yb` (0..7)
/// within a 64x8-byte GOB, for the Blackwell/Turing 2D COLOR sector layout
/// (NIL GOBType::TuringColor2D, which Blackwell uses for all >=4-byte color
/// formats - see Mesa nil/tiling.rs CopyGOBTuring2D::for_each_gob_line). The
/// GOB's 16 32-byte sectors are arranged as two 16-byte-wide columns of 8
/// sectors (4 rows x 2 of 4-sector groups), NOT the older Fermi 2-row layout.
/// Using the Fermi layout (yb/2,*64,*32,yb%2) mis-maps half the pixels in every
/// GOB; for a uniform region that permutation is invisible, but at a covered
/// region's edge it reads an adjacent (uncovered/black) byte = the steep-edge
/// "speckle" the rasterizer was wrongly blamed for. This was VERIFIED byte-for-
/// byte against Mesa's CopyGOBTuring2D (see the unit test below).
pub fn blGobByteOffset(xb: u32, yb: u32) u32 {
    return ((xb / 32) * 256) + ((yb / 4) * 128) + (((xb % 32) / 16) * 64) + ((yb % 4) * 16) + (xb % 16);
}

test "blGobByteOffset matches Mesa CopyGOBTuring2D byte-for-byte (Blackwell color GOB)" {
    // Build the reference (lin byte-col, row) -> tiled-offset map straight from
    // Mesa nil/tiling.rs CopyGOBTuring2D::for_each_gob_line: each f(tiled, x, y, _)
    // copies a 16-byte (LINE_WIDTH_B) line from linear (x..x+16, y) to tiled..+16.
    const Line = struct { toff: u32, xadd: u32, row: u32 };
    const lines = [_]Line{
        .{ .toff = 0x00, .xadd = 0, .row = 0 },  .{ .toff = 0x10, .xadd = 0, .row = 1 },
        .{ .toff = 0x20, .xadd = 0, .row = 2 },  .{ .toff = 0x30, .xadd = 0, .row = 3 },
        .{ .toff = 0x40, .xadd = 16, .row = 0 }, .{ .toff = 0x50, .xadd = 16, .row = 1 },
        .{ .toff = 0x60, .xadd = 16, .row = 2 }, .{ .toff = 0x70, .xadd = 16, .row = 3 },
        .{ .toff = 0x80, .xadd = 0, .row = 4 },  .{ .toff = 0x90, .xadd = 0, .row = 5 },
        .{ .toff = 0xa0, .xadd = 0, .row = 6 },  .{ .toff = 0xb0, .xadd = 0, .row = 7 },
        .{ .toff = 0xc0, .xadd = 16, .row = 4 }, .{ .toff = 0xd0, .xadd = 16, .row = 5 },
        .{ .toff = 0xe0, .xadd = 16, .row = 6 }, .{ .toff = 0xf0, .xadd = 16, .row = 7 },
    };
    var ref = [_]u32{0xffff} ** (64 * 8); // [row*64 + xbyte] -> tiled offset
    var i: u32 = 0;
    while (i < 2) : (i += 1) {
        for (lines) |ln| {
            const tiled = i * 0x100 + ln.toff;
            const lin_x = i * 32 + ln.xadd;
            var k: u32 = 0;
            while (k < 16) : (k += 1) ref[ln.row * 64 + lin_x + k] = tiled + k;
        }
    }
    var yb: u32 = 0;
    while (yb < 8) : (yb += 1) {
        var xb: u32 = 0;
        while (xb < 64) : (xb += 1) {
            try std.testing.expectEqual(ref[yb * 64 + xb], blGobByteOffset(xb, yb));
        }
    }
}

test "blColorPixelOffsetBpp: bpp=4 unchanged; wide (rgba16f/rgba32f) texels are byte-contiguous" {
    // The bpp=4 path is byte-identical to the base A8R8G8B8 function (zero regression).
    for ([_]u32{ 40, 64, 100 }) |w| {
        var y: u32 = 0;
        while (y < 24) : (y += 1) {
            var x: u32 = 0;
            while (x < 32) : (x += 1) {
                try std.testing.expectEqual(blColorPixelOffset(x, y, w), blColorPixelOffsetBpp(x, y, w, 4));
            }
        }
    }
    // A wide texel de-swizzles CONTIGUOUSLY: within a 16-byte line the intra-GOB swizzle is +1
    // per byte, so an 8-byte (rgba16f) or 16-byte (rgba32f) texel at an aligned start column
    // occupies `bpp` consecutive tiled bytes (the whole feature rests on this).
    var yb: u32 = 0;
    while (yb < GOB_HEIGHT_ROWS) : (yb += 1) {
        for ([_]u32{ 0, 8, 16, 24, 32, 40, 48, 56 }) |xb| { // rgba16f start columns
            var k: u32 = 0;
            while (k < 8) : (k += 1) try std.testing.expectEqual(blGobByteOffset(xb, yb) + k, blGobByteOffset(xb + k, yb));
        }
        for ([_]u32{ 0, 16, 32, 48 }) |xb| { // rgba32f start columns
            var k: u32 = 0;
            while (k < 16) : (k += 1) try std.testing.expectEqual(blGobByteOffset(xb, yb) + k, blGobByteOffset(xb + k, yb));
        }
    }
}

// --- Textures: TIC (texture image control) + TSC (texture sampler control) ---
//
// A sampled texture on Blackwell is described by two 8-dword (32-byte) descriptors
// living in GPU pools the 3D engine is pointed at:
//   - TIC (texture image control / "texture header"): the texture's GPU address,
//     format, dimensions, and block-linear tiling. Layout = TEXHEAD_V2_BL (the
//     Hopper/Blackwell V2 block-linear header, NVCB97/NVCE97), filled to match Mesa
//     nvk's nvcb97_fill_image_view_desc for an RGBA8 UNORM 2D image.
//   - TSC (texture sampler control / "sampler header"): the filter + wrap modes
//     (NVB097 TEXSAMP0/1), filled to match nvk's nvk_sampler_get_header.
// The shader's bindless TEX reads pool[index] using the handle (tic | tsc<<20); the
// pools are bound with SET_TEX_HEADER_POOL_A/B/C + SET_TEX_SAMPLER_POOL_A/B/C.
//
// The texture pixels are uploaded BLOCK-LINEAR (GOB-tiled), the same Fermi/Turing
// GOB swizzle as the block-linear color render targets, so blTexPixelOffset mirrors
// blColorPixelOffset. A small texture (<= 8 rows, e.g. the 2x2 verification texture)
// fits one GOB tall, so the block height is ONE_GOB (y_log2 = 0).

/// Block height (in GOBs, log2) for a sampled texture of `h` rows: one GOB (8 rows)
/// covers up to 8 px tall; taller textures step up to 2/4/8/16 GOBs like nvk's
/// choose_gob_height. The TIC's GOBS_PER_BLOCK_HEIGHT and blTexPixelOffset must agree.
pub fn texBlockHeightLog2(h: u32) u5 {
    var gobs: u32 = 1;
    var log2: u5 = 0;
    while (gobs * GOB_HEIGHT_ROWS < h and log2 < 4) : (log2 += 1) gobs *= 2;
    return log2;
}

/// A sampled-texture texel format the TIC + swizzle understand. rgba8_unorm/_srgb are the
/// 8-bit paths (4 bytes/texel; sRGB sets the TIC's hardware sRGB->linear conversion);
/// rgba16_float / rgba32_float are the IEEE half / single HDR formats (8 / 16 bytes/texel,
/// DATA_TYPE = FLOAT). The wider float texels stay contiguous in the GOB (8 and 16 divide
/// the 16-byte GOB sector), so the same block-linear swizzle covers them with the right bpp.
pub const TicFormat = enum {
    rgba8_unorm,
    rgba8_srgb,
    rgba16_float,
    rgba32_float,
    /// A SAMPLED depth texture (Z32_FLOAT). One 32-bit float component (the stored depth),
    /// COMPONENTS = ZF32 (0x2f), DATA_TYPE = FLOAT, swizzle R001 (R = depth, G/B = 0, A = 1).
    /// This is the format the fixed-function texture unit's DEPTH_COMPARE (sampler2DShadow)
    /// engages on - a color format returns the raw texel and the compare is a HW no-op. The
    /// backing is plain block-linear (generic kind) 4 B/px memory, identical tiling to rgba8.
    zf32,

    pub fn bytesPerTexel(self: TicFormat) u32 {
        return switch (self) {
            .rgba8_unorm, .rgba8_srgb, .zf32 => 4,
            .rgba16_float => 8,
            .rgba32_float => 16,
        };
    }
    /// TEXHEAD_V2_BL_COMPONENTS (MW 118:112): A8B8G8R8 / R16_G16_B16_A16 / R32_G32_B32_A32 / ZF32.
    fn components(self: TicFormat) u32 {
        return switch (self) {
            .rgba8_unorm, .rgba8_srgb => 0x08,
            .rgba16_float => 0x03,
            .rgba32_float => 0x01,
            .zf32 => 0x2f, // NVCB97_TEXHEAD_V2_BL_COMPONENTS_SIZES_ZF32
        };
    }
    /// TEXHEAD_V2_BL_DATA_TYPE (MW 111:108): UNORM = 0, FLOAT = 2. ZF32 depth is a float.
    fn dataType(self: TicFormat) u32 {
        return switch (self) {
            .rgba8_unorm, .rgba8_srgb => 0,
            .rgba16_float, .rgba32_float, .zf32 => 2,
        };
    }
    /// TEXHEAD_V2_BL_S_R_G_B_CONVERSION (MW 149): the texture unit decodes sRGB -> linear.
    fn srgbBit(self: TicFormat) u32 {
        return if (self == .rgba8_srgb) 1 else 0;
    }
    /// The TIC X/Y/Z/W_SOURCE swizzle for this format (TEXHEAD_V2_BL_*_SOURCE): color formats
    /// map identity R,G,B,A (IN_R=2, IN_G=3, IN_B=4, IN_A=5); a ZF32 depth texture is R001
    /// (X=IN_R=2, Y=IN_ZERO=0, Z=IN_ZERO=0, W=IN_ONE_FLOAT=7), matching NAK's nil format table
    /// so the depth lands in R (what the DEPTH_COMPARE fetches).
    pub fn swizzleSources(self: TicFormat) [4]u32 {
        return switch (self) {
            .zf32 => .{ 2, 0, 0, 7 },
            else => .{ 2, 3, 4, 5 },
        };
    }
};

/// Byte size of a `w`x`h` block-linear sampled texture of `fmt` at block height
/// `texBlockHeightLog2(h)`: width rounded to a GOB (64 B), height rounded to a full block.
/// Sizes the texture allocation to cover the whole tiled footprint.
pub fn texSizeBytes(w: u32, h: u32, fmt: TicFormat) u32 {
    const row_bytes = std.mem.alignForward(u32, w * fmt.bytesPerTexel(), GOB_WIDTH_BYTES);
    const block_gobs = @as(u32, 1) << texBlockHeightLog2(h);
    const block_rows = block_gobs * GOB_HEIGHT_ROWS;
    const tiled_h = std.mem.alignForward(u32, h, block_rows);
    return row_bytes * tiled_h;
}

/// A 3D texture's block-linear footprint: `depth` 2D block-linear slices stacked
/// (GOBS_PER_BLOCK_DEPTH = ONE_GOB, so each slice is an independent 2D block-linear image).
pub fn tex3dSizeBytes(w: u32, h: u32, depth: u32, fmt: TicFormat) u32 {
    return texSizeBytes(w, h, fmt) * @max(1, depth);
}

/// Mip level `level`'s dimensions for a `w`x`h` base (each axis halved, floored, clamped to 1).
pub fn mipLevelDims(w: u32, h: u32, level: u32) [2]u32 {
    const sh: u5 = @intCast(@min(level, 31));
    return .{ @max(1, w >> sh), @max(1, h >> sh) };
}

/// The block-linear byte OFFSET of mip `level` within a mipped sampled texture: the cumulative
/// sum of the (tile-aligned, per-level block height) sizes of all lower levels. Matches nvk nil's
/// level offset_B layout (each level uses its own clamped block height, via texSizeBytes' internal
/// texBlockHeightLog2(level_h)), which is exactly what the HW TEX addresses from the base + LOD.
pub fn blMipLevelOffset(w: u32, h: u32, level: u32, fmt: TicFormat) u32 {
    var off: u32 = 0;
    var i: u32 = 0;
    while (i < level) : (i += 1) {
        const d = mipLevelDims(w, h, i);
        off += texSizeBytes(d[0], d[1], fmt);
    }
    return off;
}

/// The total block-linear byte size of a `levels`-level mip chain (the GPU allocation footprint).
pub fn mippedTexSizeBytes(w: u32, h: u32, levels: u32, fmt: TicFormat) u32 {
    return blMipLevelOffset(w, h, levels, fmt);
}

/// The byte offset of mip `level` in the TIGHTLY-PACKED staging chain (the CPU/GLES upload layout:
/// level 0 = w*h*bpt, then w/2 x h/2, ...). Matches hal.mipLevelOffset, so uploadTexture reads each
/// level from where the GLES ensureTextureHal wrote it. NOT the block-linear GPU offset.
pub fn stagingMipLevelOffset(w: u32, h: u32, level: u32, bpt: usize) usize {
    var off: usize = 0;
    var i: u32 = 0;
    while (i < level) : (i += 1) {
        const d = mipLevelDims(w, h, i);
        off += @as(usize, d[0]) * d[1] * bpt;
    }
    return off;
}

/// Total tightly-packed staging bytes for a `levels`-level chain.
pub fn mipChainStagingBytes(w: u32, h: u32, levels: u32, bpt: usize) usize {
    return stagingMipLevelOffset(w, h, levels, bpt);
}

/// A sampler's mip-blend mode (glGenerateMipmap + a GL_*_MIPMAP_* min filter). `none` = the
/// single-level path (MIP_NONE).
pub const TexMipFilter = enum { none, nearest, linear };

/// The block-linear byte offset of texel (x, y) in a `w`x`h` RGBA8 sampled texture
/// (the same GOB swizzle as a block-linear color RT, but with the texture's own block
/// height). The CPU writes the texture pixels through this offset so the GPU's TEX
/// reads them back correctly. Mirrors blColorPixelOffset with a per-texture block.
pub fn blTexPixelOffset(x: u32, y: u32, w: u32, h: u32, fmt: TicFormat) usize {
    const gob_w: u32 = GOB_WIDTH_BYTES; // 64
    const gob_h: u32 = GOB_HEIGHT_ROWS; // 8
    const bpp: u32 = fmt.bytesPerTexel();
    const block_gobs = @as(u32, 1) << texBlockHeightLog2(h);
    const row_bytes = std.mem.alignForward(u32, w * bpp, gob_w);
    const gobs_per_row = row_bytes / gob_w;
    const block_bytes = gob_w * gob_h * block_gobs;
    const bx = x * bpp;
    const gob_col = bx / gob_w;
    const gob_row = y / gob_h;
    const block_row = gob_row / block_gobs;
    const gob_in_block = gob_row % block_gobs;
    const block_base = (block_row * gobs_per_row + gob_col) * block_bytes;
    const gob_base = block_base + gob_in_block * (gob_w * gob_h);
    const xb = bx % gob_w;
    const yb = y % gob_h;
    const off = blGobByteOffset(xb, yb);
    return @intCast(gob_base + off);
}

/// A texture sampler's magnification/minification filter.
pub const TexFilter = enum { nearest, linear };
/// A sampler's coordinate wrap mode (out of [0,1]).
pub const TexAddress = enum { repeat, clamp_to_edge, mirror };

/// Set a bit field [lo, lo+width) in an 8-dword descriptor (TIC/TSC are 256-bit).
fn descSet(d: *[8]u32, lo: usize, width: usize, val: u32) void {
    var i: usize = 0;
    while (i < width) : (i += 1) {
        const bit = lo + i;
        const off: u5 = @intCast(bit % 32);
        const b: u32 = (val >> @intCast(i)) & 1;
        d[bit / 32] = (d[bit / 32] & ~(@as(u32, 1) << off)) | (b << off);
    }
}

/// Build a TEXHEAD_V2_BL TIC (8 dwords) for an RGBA8_UNORM, `w`x`h`, block-linear,
/// single-level 2D texture at `tex_va`. Byte-for-byte the field set Mesa nvk's
/// nvcb97_fill_image_view_desc writes for this format on Blackwell (HEADER_VERSION =
/// SELECT_BLOCKLINEAR_V2, COMPONENTS = A8B8G8R8, DATA_TYPE = UNORM, X/Y/Z/W_SOURCE =
/// R/G/B/A, the block tiling, WIDTH/HEIGHT/DEPTH minus one, NORMALIZED_COORDS).
pub fn fillTic(tex_va: u64, w: u32, h: u32, fmt: TicFormat, max_mip_level: u32) [8]u32 {
    var d = [_]u32{0} ** 8;
    // MAX_MIP_LEVEL (V2_BL MW 95:92) = the highest mip index the HW may sample (levels-1). 0 keeps
    // the single-level behavior. The block tiling below is level 0's; the HW derives each lower
    // level's offset + clamped block height itself (matching blMipLevelOffset / nvk nil).
    descSet(&d, 92, 4, @min(max_mip_level, 15));
    // RES_VIEW MIN/MAX mip level (V2_BL MW 227:224 / 231:228): the sampled level is CLAMPED to this
    // resource-view range. They DEFAULT to 0, which pins sampling to the base level even with
    // MAX_MIP_LEVEL set - this was the wall. Set the view to span the whole chain [0, levels-1].
    descSet(&d, 224, 4, 0); // RES_VIEW_MIN_MIP_LEVEL
    descSet(&d, 228, 4, @min(max_mip_level, 15)); // RES_VIEW_MAX_MIP_LEVEL
    // Word-0 component description (TEXHEAD_V2_BL_*):
    descSet(&d, 112, 7, fmt.components()); // COMPONENTS (MW 118:112)
    descSet(&d, 108, 4, fmt.dataType()); // DATA_TYPE (UNORM=0 / FLOAT=2) (MW 111:108)
    descSet(&d, 149, 1, fmt.srgbBit()); // S_R_G_B_CONVERSION (MW 149): sRGB textures decode
    const src = fmt.swizzleSources(); // color: R,G,B,A; ZF32 depth: R,0,0,1
    descSet(&d, 96, 3, src[0]); // X_SOURCE (MW 98:96)
    descSet(&d, 99, 3, src[1]); // Y_SOURCE (MW 101:99)
    descSet(&d, 102, 3, src[2]); // Z_SOURCE (MW 104:102)
    descSet(&d, 105, 3, src[3]); // W_SOURCE (MW 107:105)
    // Address (block-linear): bits 9..32 and 32..57.
    const a9_31: u32 = @intCast((tex_va >> 9) & 0x7fffff); // 23 bits (MW 31:9)
    const a32_56: u32 = @intCast((tex_va >> 32) & 0x1ffffff); // 25 bits (MW 56:32)
    descSet(&d, 9, 23, a9_31);
    descSet(&d, 32, 25, a32_56);
    // Block tiling: width 1 GOB, height = texBlockHeightLog2, depth 1 GOB, tile width 1.
    descSet(&d, 64, 3, 0); // GOBS_PER_BLOCK_WIDTH = ONE_GOB (MW 66:64)
    descSet(&d, 67, 3, texBlockHeightLog2(h)); // GOBS_PER_BLOCK_HEIGHT (MW 69:67)
    descSet(&d, 70, 3, 0); // GOBS_PER_BLOCK_DEPTH = ONE_GOB (MW 72:70)
    descSet(&d, 74, 3, 0); // TILE_WIDTH_IN_GOBS = ONE_GOB (MW 76:74)
    descSet(&d, 81, 1, 1); // LOD_ANISO_QUALITY = HIGH (MW 81)
    descSet(&d, 82, 1, 1); // LOD_ISO_QUALITY = HIGH (MW 82)
    descSet(&d, 124, 4, 3); // HEADER_VERSION = SELECT_BLOCKLINEAR_V2 (MW 127:124)
    descSet(&d, 128, 17, w - 1); // WIDTH_MINUS_ONE (MW 144:128)
    descSet(&d, 145, 1, 1); // NORMALIZED_COORDS (MW 145)
    descSet(&d, 150, 4, 1); // TEXTURE_TYPE = TWO_D (MW 153:150)
    descSet(&d, 154, 2, 0); // SECTOR_PROMOTION = NO_PROMOTION (MW 155:154)
    descSet(&d, 156, 1, 1); // BORDER_SOURCE = BORDER_COLOR (MW 156)
    descSet(&d, 160, 17, h - 1); // HEIGHT_MINUS_ONE (MW 176:160)
    descSet(&d, 177, 15, 0); // DEPTH_MINUS_ONE = 0 (MW 191:177)
    descSet(&d, 215, 2, 2); // ANISO_FINE_SPREAD_FUNC = SPREAD_FUNC_TWO (MW 216:215)
    descSet(&d, 217, 2, 1); // ANISO_COARSE_SPREAD_FUNC = SPREAD_FUNC_ONE (MW 218:217)
    return d;
}

/// Override a TIC's RES_VIEW mip range (GL_TEXTURE_BASE_LEVEL / GL_TEXTURE_MAX_LEVEL). The HW
/// clamps the sampled mip level to [view_min, view_max] within the resource. Call AFTER fillTic
/// (which defaults the view to the whole chain [0, MAX_MIP_LEVEL]). view_max is expected already
/// clamped to the chain length by the caller; both are capped at the 4-bit field max (15).
pub fn setTicMipView(tic: *[8]u32, view_min: u32, view_max: u32) void {
    descSet(tic, 224, 4, @min(view_min, 15)); // RES_VIEW_MIN_MIP_LEVEL (V2_BL MW 227:224)
    descSet(tic, 228, 4, @min(view_max, 15)); // RES_VIEW_MAX_MIP_LEVEL (V2_BL MW 231:228)
}

/// Override a TIC's component swizzle (GL_TEXTURE_SWIZZLE_R/G/B/A). `swizzle` is per output channel
/// X/Y/Z/W: 0=R 1=G 2=B 3=A 4=zero 5=one. Maps to the TIC *_SOURCE fields (IN_R=2..IN_A=5, IN_ZERO=0,
/// IN_ONE_FLOAT=7) at MW 98:96 / 101:99 / 104:102 / 107:105. Call after fillTic (identity default).
pub fn setTicSwizzle(tic: *[8]u32, swizzle: [4]u8) void {
    const src = struct {
        fn v(s: u8) u32 {
            return switch (s) {
                0, 1, 2, 3 => @as(u32, s) + 2, // IN_R..IN_A
                4 => 0, // IN_ZERO
                else => 7, // IN_ONE_FLOAT
            };
        }
    }.v;
    descSet(tic, 96, 3, src(swizzle[0])); // X_SOURCE
    descSet(tic, 99, 3, src(swizzle[1])); // Y_SOURCE
    descSet(tic, 102, 3, src(swizzle[2])); // Z_SOURCE
    descSet(tic, 105, 3, src(swizzle[3])); // W_SOURCE
}

/// Set a TSC's LOD bias (GL_TEXTURE_LOD_BIAS) - added to the computed mip LOD. TEXSAMP1 MIP_LOD_BIAS
/// (24:12, TEXSAMP1 word base bit 32) is a SIGNED S4.8 fixed-point (13 bits): value = round(bias*256),
/// clamped to the field range [-4096, 4095] (about +-16 LOD). Call after fillTsc (default 0 = no bias).
pub fn setTscLodBias(tsc: *[8]u32, bias: f32) void {
    const fixed = std.math.clamp(@round(bias * 256.0), -4096.0, 4095.0);
    const bits: u32 = @as(u32, @bitCast(@as(i32, @intFromFloat(fixed)))) & 0x1fff;
    descSet(tsc, 32 + 12, 13, bits); // TEXSAMP1 MIP_LOD_BIAS
}

/// Set a TSC's LOD clamp (GL_TEXTURE_MIN_LOD / GL_TEXTURE_MAX_LOD). TEXSAMP2 MIN_LOD_CLAMP (11:0) /
/// MAX_LOD_CLAMP (23:12) are UNSIGNED U4.8 fixed-point (12 bits each): value = round(lod*256) clamped
/// to [0, 0xfff] (about 0..16). A negative min_lod clamps to 0. Call after fillTsc (default 0/0xfff).
pub fn setTscLodClamp(tsc: *[8]u32, min_lod: f32, max_lod: f32) void {
    const enc = struct {
        fn v(l: f32) u32 {
            return @intFromFloat(std.math.clamp(@round(l * 256.0), 0.0, 4095.0));
        }
    }.v;
    descSet(tsc, 64 + 0, 12, enc(min_lod)); // TEXSAMP2 MIN_LOD_CLAMP
    descSet(tsc, 64 + 12, 12, enc(max_lod)); // TEXSAMP2 MAX_LOD_CLAMP
}

/// Enable/disable a TSC's hardware DEPTH-COMPARE (GL_TEXTURE_COMPARE_MODE == COMPARE_REF_TO_TEXTURE,
/// for sampler2DShadow). TEXSAMP0 (dword 0): DEPTH_COMPARE (bit 9) enables the compare; DEPTH_COMPARE_
/// FUNC (bits 12:10) is the ZC_* comparison function. The ZC_* enum (ZC_NEVER=0 LESS=1 EQUAL=2 LEQUAL=3
/// GREATER=4 NOTEQUAL=5 GEQUAL=6 ALWAYS=7) exactly equals hal.CompareOp's ordinal, so `op` maps directly
/// (no remap). The TSC holds the compare FUNCTION; the TEX's z_cmpr bit triggers the compare + returns the
/// scalar pass fraction. Call after fillTsc (default disabled). See the isel's texShadow (z_cmpr, bit 78).
pub fn setTscDepthCompare(tsc: *[8]u32, enable: bool, op: u32) void {
    descSet(tsc, 9, 1, if (enable) 1 else 0); // TEXSAMP0 DEPTH_COMPARE (enable)
    if (enable) descSet(tsc, 10, 3, op & 0x7); // TEXSAMP0 DEPTH_COMPARE_FUNC (ZC_* == hal.CompareOp ordinal)
}

/// Build a TEXHEAD_V2_BL TIC for a `w`x`h`x`depth` 3D texture (sampler3D) at `tex_va`. Same as
/// `fillTic` (single-level, block-linear) but TEXTURE_TYPE = THREE_D and DEPTH_MINUS_ONE = the
/// slice count minus one. The volume is `depth` stacked 2D block-linear slices
/// (GOBS_PER_BLOCK_DEPTH = ONE_GOB), which the HW addresses per slice like a 2D image.
pub fn fillTic3d(tex_va: u64, w: u32, h: u32, depth: u32, fmt: TicFormat) [8]u32 {
    var d = fillTic(tex_va, w, h, fmt, 0);
    descSet(&d, 150, 4, 2); // TEXTURE_TYPE = THREE_D (MW 153:150) [was TWO_D=1]
    descSet(&d, 177, 15, if (depth > 0) depth - 1 else 0); // DEPTH_MINUS_ONE (MW 191:177)
    return d;
}

/// Build a TEXHEAD_V2_BL TIC for a `w`x`h` 2D ARRAY (sampler2DArray) of `layers` layers at `tex_va`.
/// Same block-linear layout as `fillTic3d` (the layers are stacked block-linear 2D images, one GOB
/// deep each), but TEXTURE_TYPE = TWO_D_ARRAY (=5). The array size reuses DEPTH_MINUS_ONE (the field
/// the HW reads as the layer count for an array target). The HW samples one layer by a raw index -
/// no cross-layer filtering (vs THREE_D's trilinear).
pub fn fillTic2dArray(tex_va: u64, w: u32, h: u32, layers: u32, fmt: TicFormat) [8]u32 {
    var d = fillTic(tex_va, w, h, fmt, 0);
    descSet(&d, 150, 4, 5); // TEXTURE_TYPE = TWO_D_ARRAY (MW 153:150)
    descSet(&d, 177, 15, if (layers > 0) layers - 1 else 0); // (array size) - 1, reuses DEPTH_MINUS_ONE
    return d;
}

// NOTE: there is no fillTicCube. The native Blackwell cube TEX modes do not select the face from
// the direction (see vulcan encode.TexDim), so a samplerCube is lowered (in the nvidia isel) to the
// major-axis math + a 2D sample of a 6-face-WIDE atlas, and the cube resource is bound with a plain
// 2D `fillTic(6*face_w, face_h)`.

/// The 32-byte-aligned row pitch a PITCH-linear sampled texture needs (the V2_PITCH
/// TIC encodes pitch>>5, so the row stride must be a multiple of 32 bytes).
pub fn texPitchBytes(w: u32) u32 {
    return std.mem.alignForward(u32, w * 4, 32);
}

/// Build a TEXHEAD_V2_PITCH TIC (8 dwords) for an RGBA8_UNORM, `w`x`h`, PITCH-LINEAR
/// (row-major, no GOB swizzle) 2D texture at `tex_va` with row stride `pitch_bytes`
/// (32-byte aligned). The component/source/dimension fields share the V2_BL offsets;
/// only HEADER_VERSION (SELECT_PITCH_V2), the address split, and the pitch differ.
/// A pitch-linear texture is the natural layout for a small sampled image: the CPU
/// writes it row-major exactly as the sampler reads it (no tiling to get wrong).
pub fn fillTicPitch(tex_va: u64, w: u32, h: u32, pitch_bytes: u32) [8]u32 {
    var d = [_]u32{0} ** 8;
    descSet(&d, 112, 7, 0x08); // COMPONENTS = A8B8G8R8
    descSet(&d, 108, 4, 0); // DATA_TYPE = UNORM
    descSet(&d, 96, 3, 2); // X_SOURCE = IN_R
    descSet(&d, 99, 3, 3); // Y_SOURCE = IN_G
    descSet(&d, 102, 3, 4); // Z_SOURCE = IN_B
    descSet(&d, 105, 3, 5); // W_SOURCE = IN_A
    // Address (pitch): bits 5..32 and 32..57. The low 5 bits MUST be zero.
    const a5_31: u32 = @intCast((tex_va >> 5) & 0x7ffffff); // 27 bits (MW 31:5)
    const a32_56: u32 = @intCast((tex_va >> 32) & 0x1ffffff); // 25 bits (MW 56:32)
    descSet(&d, 5, 27, a5_31);
    descSet(&d, 32, 25, a32_56);
    descSet(&d, 64, 17, pitch_bytes >> 5); // PITCH_BITS21TO5 (MW 80:64) = pitch / 32
    descSet(&d, 81, 1, 1); // LOD_ANISO_QUALITY2
    descSet(&d, 82, 1, 1); // LOD_ANISO_QUALITY = HIGH
    descSet(&d, 83, 1, 1); // LOD_ISO_QUALITY = HIGH
    descSet(&d, 124, 4, 2); // HEADER_VERSION = SELECT_PITCH_V2
    descSet(&d, 128, 17, w - 1); // WIDTH_MINUS_ONE
    descSet(&d, 145, 1, 1); // NORMALIZED_COORDS
    descSet(&d, 150, 4, 1); // TEXTURE_TYPE = TWO_D
    descSet(&d, 154, 2, 0); // SECTOR_PROMOTION = NO_PROMOTION
    descSet(&d, 156, 1, 1); // BORDER_SOURCE = BORDER_COLOR
    descSet(&d, 160, 17, h - 1); // HEIGHT_MINUS_ONE
    descSet(&d, 177, 15, 0); // DEPTH_MINUS_ONE = 0
    descSet(&d, 215, 2, 2); // ANISO_FINE_SPREAD_FUNC = TWO
    descSet(&d, 217, 2, 1); // ANISO_COARSE_SPREAD_FUNC = ONE
    return d;
}

fn tscAddr(a: TexAddress) u32 {
    return switch (a) {
        .repeat => 0, // WRAP
        .mirror => 1, // MIRROR
        .clamp_to_edge => 2, // CLAMP_TO_EDGE
    };
}

/// Build a TEXSAMP TSC (8 dwords) for the given filter + wrap modes (NVB097
/// TEXSAMP0/1), matching nvk_sampler_get_header for a non-anisotropic, single-level
/// sampler. MAG/MIN use the filter; MIP_FILTER = NONE (single-level texture);
/// FLOAT_COORD_NORMALIZATION = USE_HEADER_SETTING (the TIC sets NORMALIZED_COORDS).
/// Map a max-anisotropy ratio to the TSC 3-bit MAX_ANISOTROPY level (1:1=0, 2:1=1, 4:1=2, 6:1=3,
/// 8:1=4, 10:1=5, 12:1=6, 16:1=7). Mirrors nvk's vk_to_9097_max_anisotropy thresholds.
fn anisoLevel(max_anisotropy: f32) u32 {
    if (max_anisotropy >= 16) return 7;
    if (max_anisotropy >= 12) return 6;
    if (max_anisotropy >= 10) return 5;
    if (max_anisotropy >= 8) return 4;
    if (max_anisotropy >= 6) return 3;
    if (max_anisotropy >= 4) return 2;
    if (max_anisotropy >= 2) return 1;
    return 0;
}

test "anisoLevel maps the ratio to the TSC 3-bit level" {
    try std.testing.expectEqual(@as(u32, 0), anisoLevel(1));
    try std.testing.expectEqual(@as(u32, 1), anisoLevel(2));
    try std.testing.expectEqual(@as(u32, 2), anisoLevel(4));
    try std.testing.expectEqual(@as(u32, 4), anisoLevel(8));
    try std.testing.expectEqual(@as(u32, 7), anisoLevel(16));
    try std.testing.expectEqual(@as(u32, 7), anisoLevel(100)); // clamped
    try std.testing.expectEqual(@as(u32, 0), anisoLevel(1.5)); // below 2x -> off
}

pub fn fillTsc(filter: TexFilter, address_u: TexAddress, address_v: TexAddress, mip_filter: TexMipFilter, max_anisotropy: f32) [8]u32 {
    var d = [_]u32{0} ** 8;
    descSet(&d, 20, 3, anisoLevel(max_anisotropy)); // TEXSAMP0 MAX_ANISOTROPY (22:20)
    // TEXSAMP0 (word 0): ADDRESS_U (2:0), ADDRESS_V (5:3), ADDRESS_P (8:6).
    descSet(&d, 0, 3, tscAddr(address_u));
    descSet(&d, 3, 3, tscAddr(address_v));
    descSet(&d, 6, 3, tscAddr(.clamp_to_edge)); // P (unused for 2D)
    // TEXSAMP0_S_R_G_B_CONVERSION (bit 13): ENABLE sRGB->linear decode. This is the sampler
    // gate; the TIC's own sRGB flag decides per-texture whether it actually happens (an
    // rgba8_unorm TIC is unaffected). nvk sets this true unconditionally, so we do too.
    descSet(&d, 13, 1, 1);
    // TEXSAMP1 (word 1, bit base 32): MAG_FILTER (2:0), MIN_FILTER (5:4), MIP_FILTER (7:6).
    const mag: u32 = if (filter == .linear) 2 else 1; // MAG_LINEAR / MAG_POINT
    const min: u32 = if (filter == .linear) 2 else 1; // MIN_LINEAR / MIN_POINT
    descSet(&d, 32 + 0, 3, mag);
    descSet(&d, 32 + 4, 2, min);
    // MIP_FILTER (TEXSAMP1 7:6): MIP_NONE=1 (base only), MIP_POINT=2 (nearest level), MIP_LINEAR=3
    // (trilinear). Set from the GL_*_MIPMAP_* min filter; a texture with no chain stays MIP_NONE.
    const mipf: u32 = switch (mip_filter) {
        .none => 1,
        .nearest => 2,
        .linear => 3,
    };
    descSet(&d, 32 + 6, 2, mipf);
    // TEXSAMP2 (word 2, bit base 64): MIN_LOD_CLAMP (11:0) + MAX_LOD_CLAMP (23:12), unsigned
    // fixed-point LOD clamps. They DEFAULT to 0, which pins the LOD to [0,0] = the base level -
    // so a mipmapped texture never minifies without this. Set MAX to the field max (0xfff, a LOD
    // far above any mip level) so the HW's implicit LOD is unclamped. MIP_NONE ignores the LOD, so
    // this is harmless for a single-level texture.
    descSet(&d, 64 + 0, 12, 0); // MIN_LOD_CLAMP = 0
    descSet(&d, 64 + 12, 12, 0xfff); // MAX_LOD_CLAMP = no clamp
    return d;
}

/// SET_TEX_HEADER_POOL_A/B/C and SET_TEX_SAMPLER_POOL_A/B/C (NVCE97). Point the 3D
/// engine at the TIC pool (`tic_va`, `tic_max` = highest valid index) and the TSC
/// pool (`tsc_va`, `tsc_max`). The bindless TEX indexes these with its handle. Call
/// before the draw; the pools live in GPU memory the dispatch built.
pub const SET_TEX_HEADER_POOL_A = 0x1574; // OFFSET_UPPER (24:0)
pub const SET_TEX_HEADER_POOL_B = 0x1578; // OFFSET_LOWER (31:0)
pub const SET_TEX_HEADER_POOL_C = 0x157c; // MAXIMUM_INDEX
pub const SET_TEX_SAMPLER_POOL_A = 0x155c; // OFFSET_UPPER (24:0)
pub const SET_TEX_SAMPLER_POOL_B = 0x1560; // OFFSET_LOWER (31:0)
pub const SET_TEX_SAMPLER_POOL_C = 0x1564; // MAXIMUM_INDEX
pub fn bindTexturePools(s: *threed.Stream, tic_va: u64, tic_max: u32, tsc_va: u64, tsc_max: u32) void {
    s.mm(SET_TEX_HEADER_POOL_A, &.{ @intCast(tic_va >> 32), @truncate(tic_va), tic_max });
    s.mm(SET_TEX_SAMPLER_POOL_A, &.{ @intCast(tsc_va >> 32), @truncate(tsc_va), tsc_max });
    // INVALIDATE_TEXTURE_HEADER_CACHE + DATA_CACHE so the engine re-reads the freshly
    // written pools (a stale header/data cache would sample garbage or the wrong texel).
    s.m1(0x1334, 0); // INVALIDATE_TEXTURE_HEADER_CACHE (LINES_ALL)
    s.m1(0x1338, 0); // INVALIDATE_TEXTURE_DATA_CACHE (LINES_ALL)
}

// --- MME (Macro Method Engine): the SET_PRIV_REG firmware workaround ---
//
// The Blackwell 3D front-end has a programmable Macro Method Engine. nvk uploads
// MME macro programs and drives most render-pass/draw state through them. The
// from-scratch raw-method path replicates the draw itself fine (nvk's draw macro,
// with view_mask == 0, reduces to exactly the SET_DRAW_CONTROL_A + DRAW_VERTEX_
// ARRAY_BEGIN_END this driver already emits), so the draw macro is NOT needed.
//
// What the raw path was MISSING is nvk's two SET_PRIV_REG calls in
// nvk_push_draw_state_init. On GSP-firmware GPUs (Hopper+, which includes
// Blackwell) a privileged SM register can only be written by handing the request
// to the GSP firmware via the MME's SET_FALCON04 handshake - a raw pushbuffer
// cannot do it because it requires reading back a scratch register in a spin
// loop, which only the MME can do. The two writes nvk performs:
//
//   1. clear bit 3 of gr_gpcs_tpcs_sm_disp_ctrl (Hopper+ reg 0x4243a4) - enables
//      FP helper-invocation memory loads.
//   2. clear bit 14 of gr_gpcs_tpcs_sms_hww_warp_esr_report_mask (Hopper+ reg
//      0x4246a8) - DISABLES the "Out Of Range Address" SM exception. nvk's own
//      comment: when a final geometry stage writes an output the hardware thinks
//      is unused, "any writes to outputs from the final shader stage generates an
//      Out Of Range Address exception ... the easiest solution is to just disable
//      the exception."
//
// Write #2 is the load-bearing one for the depth wall: binding a valid ZETA and
// drawing makes the ROP-Z path active, and the engine raises that masked-by-
// default-elsewhere SM/ROP exception which surfaces as the opaque GSP-decoded
// Xid 69 / Class Error / ErrorCode 0x9c. Clearing the report-mask bit via the
// firmware suppresses it, which is exactly what the raw path could not do.
//
// The macro bytecode below is the Blackwell (tu104 MME ISA) encoding of nvk's
// `nvk_mme_set_priv_reg`, produced byte-for-byte by Mesa's own MME builder +
// encoder (src/nouveau/mme/mme_tu104_*). It takes 3 inline params via mme_load:
// value, mask, reg. It emits WAIT_FOR_IDLE, stuffs {0, value, mask} into the
// FALCON_0/1/2 scratch, writes the reg id to SET_FALCON04, then spins on
// SET_MME_SHADOW_SCRATCH(FALCON_0) until the firmware sets it to 1.

// MME upload + dispatch method offsets (clcd97 / clce97; NV9097-stable).
pub const LOAD_MME_INSTRUCTION_RAM_POINTER = 0x0114;
pub const LOAD_MME_INSTRUCTION_RAM = 0x0118;
pub const LOAD_MME_START_ADDRESS_RAM_POINTER = 0x011c;
pub const LOAD_MME_START_ADDRESS_RAM = 0x0120;
pub const SET_FALCON04 = 0x2310;
pub const WAIT_FOR_IDLE = 0x0110;
/// CALL_MME_MACRO(j) = 0x3800 + j*8: trigger the macro whose start IP is stored
/// at dispatch-table slot j. The first data word is the macro's first mme_load.
pub fn callMmeMacro(j: u32) u32 {
    return 0x3800 + j * 8;
}

/// The Blackwell MME bytecode for nvk_mme_set_priv_reg: 12 instructions x 3
/// dwords. Generated by Mesa's mme_tu104 builder/encoder for cls_eng3d=0xce97,
/// so it is byte-identical to what nvk uploads. Do not hand-edit.
pub const MME_SET_PRIV_REG = [_]u32{
    0x00000003, 0x1d080000, 0x31c00300,
    0x00d40003, 0x18c00411, 0x31c10300,
    0x10d40003, 0x18c00740, 0x300c0300,
    0x2a200003, 0x02c00631, 0x301c0300,
    0x00000003, 0x18c00000, 0x31818300,
    0x00000003, 0x18c02000, 0xf18c6300,
    0x00000003, 0x18c00340, 0x3191db00,
    0x00d40003, 0x18c00410, 0x318c0300,
    0x00000007, 0x18c007ff, 0xb43c7700,
    0x00000003, 0x18c00000, 0x318c0300,
    0x00000003, 0x18c00000, 0x318c0301,
    0x00000003, 0x18c00000, 0x318c0300,
};

/// The MME macro slot the SET_PRIV_REG macro is uploaded into (and its start IP,
/// since it is the only macro: both are 0).
pub const MME_SLOT_SET_PRIV_REG: u32 = 0;

// The two privileged registers nvk pokes during state init, for Hopper+ (which
// covers Blackwell 0xce97). On earlier generations the offsets differ; this
// driver only targets Blackwell.
pub const PRIV_REG_SM_DISP_CTRL: u32 = 0x4243a4; // clear bit 3: FP helper loads
pub const PRIV_REG_WARP_ESR_REPORT_MASK: u32 = 0x4246a8; // clear bit 14: OOR-addr exc

/// Upload the SET_PRIV_REG MME macro into the macro RAM at slot
/// MME_SLOT_SET_PRIV_REG. Call ONCE at context/channel init, after SET_OBJECT,
/// before any priv-reg write or draw. Mirrors nvk_push_draw_state_init's upload
/// loop (LOAD_MME_START_ADDRESS_RAM* sets the dispatch entry; the macro words go
/// into the instruction RAM via a non-incrementing run at LOAD_MME_INSTRUCTION_
/// RAM).
pub fn uploadMmeMacros(s: *threed.Stream) void {
    const start_ip: u32 = 0;
    s.mm(LOAD_MME_START_ADDRESS_RAM_POINTER, &.{ MME_SLOT_SET_PRIV_REG, start_ip });
    // LOAD_MME_INSTRUCTION_RAM_POINTER then stream the macro words (non-incr, so
    // every word lands in the instruction RAM auto-advancing the RAM pointer).
    s.m1(LOAD_MME_INSTRUCTION_RAM_POINTER, start_ip);
    s.ni(LOAD_MME_INSTRUCTION_RAM, &MME_SET_PRIV_REG);
}

/// Emit a ONE_INC (P_1INC) method run: the first data word lands at `addr`, every
/// subsequent word at `addr + 4`. This is the calling convention for CALL_MME_
/// MACRO: word 0 triggers the macro at slot `addr`, and the macro reads its
/// mme_load parameters from the words streamed to `addr + 4`. A NON_INC run sends
/// every word to `addr`, which never feeds the macro's parameter port and faults
/// MISSING_MACRO_DATA.
fn oneInc(s: *threed.Stream, addr: u32, vals: []const u32) void {
    s.buf[s.n] = (5 << 29) | (@as(u32, @intCast(vals.len)) << 16) | (addr >> 2);
    s.n += 1;
    for (vals) |v| {
        s.buf[s.n] = v;
        s.n += 1;
    }
}

/// Hand the firmware a masked privileged-register write via the SET_PRIV_REG MME
/// macro: reg[bits in mask] = value. CALL_MME_MACRO (ONE_INC) with 3 inline
/// params (value, mask, reg) the macro reads via mme_load. Must follow
/// uploadMmeMacros and SET_OBJECT.
pub fn setPrivReg(s: *threed.Stream, value: u32, mask: u32, reg: u32) void {
    oneInc(s, callMmeMacro(MME_SLOT_SET_PRIV_REG), &.{ value, mask, reg });
}

/// Replicate nvk's two state-init priv-reg writes for Blackwell: enable FP helper
/// loads (clear bit 3 of SM_DISP_CTRL) and DISABLE the SM "Out Of Range Address"
/// exception (clear bit 14 of WARP_ESR_REPORT_MASK). The latter is what lets a
/// draw with a bound ZETA succeed instead of faulting Xid 69 / 0x9c. Run once at
/// init after uploadMmeMacros.
pub fn disableDrawExceptions(s: *threed.Stream) void {
    setPrivReg(s, 0, 1 << 3, PRIV_REG_SM_DISP_CTRL);
    setPrivReg(s, 0, 1 << 14, PRIV_REG_WARP_ESR_REPORT_MASK);
}

/// Constant-buffer bind group for a stage (the SET_PIPELINE_BINDING value):
/// vertex = 0 ... fragment = 4, matching the Mesa shader-stage order.
pub fn bindGroup(t: ShaderType) u32 {
    return switch (t) {
        .vertex => 0,
        .tessellation_init => 1,
        .tessellation => 2,
        .geometry => 3,
        .pixel => 4,
    };
}

/// SET_PIPELINE_SHADER_TYPE values (which engine stage a slot drives).
pub const ShaderType = enum(u32) {
    vertex = 1,
    tessellation_init = 2,
    tessellation = 3,
    geometry = 4,
    pixel = 5,
};

/// Primitive topology for BEGIN / DRAW_VERTEX_ARRAY.
/// The enum VALUES are the SET_DRAW_CONTROL_A_TOPOLOGY encoding (POINTS=0, LINES=1,
/// TRIANGLES=4). SET_PRIMITIVE_TOPOLOGY (0x1970) uses a DIFFERENT encoding
/// (POINTLIST=1, LINELIST=2, TRIANGLELIST=4) - use primTopoV() for it. (Writing the
/// DRAW_CONTROL_A value to 0x1970 faults: 0 is invalid, 1 = POINTLIST not LINELIST.)
pub const Topology = enum(u32) {
    points = 0,
    lines = 1,
    triangles = 4,
    triangle_strip = 5,
    triangle_fan = 6,

    /// SET_PRIMITIVE_TOPOLOGY_V value (the separate-topology-state method 0x1970).
    pub fn primTopoV(self: Topology) u32 {
        return switch (self) {
            .points => 1, // POINTLIST
            .lines => 2, // LINELIST
            .triangles => 4, // TRIANGLELIST
            .triangle_strip => 5, // TRIANGLESTRIP
            .triangle_fan => 6, // TRIANGLEFAN
        };
    }
};

/// A Shader Program Header (SPHV3): the ~80-byte descriptor the 3D engine reads
/// to learn a shader's type and its input/output attribute maps. One per shader,
/// uploaded just before its SASS code; SET_PIPELINE_PROGRAM(j) points the engine
/// at it (as an offset from SET_PROGRAM_REGION).
pub const Sph = struct {
    // SPHV4 (sm>=73) is 32 dwords (128 bytes). The fragment per-vertex/barycentric
    // imap and the VTG generic omap live above dword 17, so the full 32-dword
    // header must be uploaded (older code used 20 dwords, which was enough for a
    // position-only passthrough but truncates the generic varying maps).
    data: [32]u32 = [_]u32{0} ** 32, // 128 bytes, SPHV4

    fn set(self: *Sph, lo: usize, width: usize, val: u64) void {
        var i: usize = 0;
        while (i < width) : (i += 1) {
            const bit = lo + i;
            const off: u5 = @intCast(bit % 32);
            const b: u32 = @intCast((val >> @intCast(i)) & 1);
            self.data[bit / 32] = (self.data[bit / 32] & ~(@as(u32, 1) << off)) | (b << off);
        }
    }

    /// Header for a vertex shader (a "VTG"-type program).
    pub fn vertex() Sph {
        var s = Sph{};
        s.set(0, 5, 0x01); // SPH_TYPE = TYPE_01_VTG
        s.set(5, 5, 4); // VERSION = 4 (sm>=73)
        s.set(10, 4, @intFromEnum(ShaderType.vertex)); // SHADER_TYPE
        s.set(17, 4, 1); // SASS_VERSION (NAK sets this to 1 on every shader)
        return s;
    }

    /// Header for a fragment (pixel) shader.
    pub fn fragment() Sph {
        var s = Sph{};
        s.set(0, 5, 0x02); // SPH_TYPE = TYPE_02_PS
        s.set(5, 5, 4); // VERSION = 4
        s.set(10, 4, @intFromEnum(ShaderType.pixel)); // SHADER_TYPE = PIXEL
        s.set(17, 4, 1); // SASS_VERSION
        s.set(14, 1, 1); // MRT_ENABLE (NAK always sets this true for fragment)
        // REQUIRED on every fragment shader: imap_system_values_ab bit 31 (absolute
        // SPH bit 191). NAK: "otherwise it cause a trap." Without it the front-end
        // data-scheduler raises a SHADER exception while validating the SPH (the
        // Xid 13 / ESR 0x405840 SHADER / 0x405848 fault) the instant a draw is issued.
        s.set(191, 1, 1);
        return s;
    }

    /// VS: this shader writes clip-space position (x, y, z, w). Also marks the
    /// store-request range (the output attributes the VS actually writes) - the
    /// engine needs this or it thinks the VS produces nothing. Position is at
    /// ATTR_POSITION (0x70), 4 components -> bytes 0x70..0x80; NAK encodes the
    /// range as start = addr/4, end = (addr_end-1)/4 => [0x1c, 0x1f].
    pub fn writesPosition(self: *Sph) void {
        self.set(428, 4, 0xf); // OMAP_POSITION_X..W
        self.set(140, 8, 0x70 / 4); // STORE_REQ_START
        self.set(152, 8, (0x80 - 1) / 4); // STORE_REQ_END
    }

    /// VS: enable stream 0 for transform-feedback capture. STREAM_OUT_MASK = SPH T1 MW(31:28) = bit
    /// offset 28, width 4 (one bit per vertex stream); 0x1 = stream 0. Without it the HW computes the
    /// VS outputs but does NOT route them to the stream-out engine (the capture buffer stays zero).
    /// Harmless for a normal draw: stream-out only happens when SO buffers are also bound + enabled.
    pub fn enablesStreamOut(self: *Sph) void {
        self.set(28, 4, 0x1);
    }

    /// VS: this shader writes gl_PointSize (a scalar to attribute a[0x6c]). Sets
    /// OMAP_POINT_SIZE (SPH bit 427) so the DA takes the point size from the shader, and
    /// extends STORE_REQ_START down to 0x6c so the store is in the declared output range.
    /// Call AFTER writesPosition (which sets STORE_REQ to the 0x70 base); this lowers the start.
    pub fn writesPointSize(self: *Sph) void {
        self.set(427, 1, 1); // OMAP_POINT_SIZE
        self.set(140, 8, 0x6c / 4); // STORE_REQ_START down to the point-size slot
    }

    /// PS: this shader writes color (r, g, b, a) to render target 0. NAK's
    /// "omap_targets" is the RT write mask at bits 576..608 (8 targets x 4
    /// components); RT0 RGBA = the low nibble. (Not OMAP_COLOR at 560, which is
    /// the legacy fixed-function color.)
    pub fn writesColor(self: *Sph) void {
        self.set(576, 4, 0xf); // OMAP_TARGET RT0 R..A
    }

    /// PS (MRT): this shader writes render target `index` (RGBA). omap_targets is 8 targets x
    /// 4 component bits starting at bit 576, so target `index` = bits 576+index*4 .. +4. RT0 is
    /// writesColor(); this generalizes it to any target for multiple render targets.
    pub fn writesColorTarget(self: *Sph, index: u32) void {
        self.set(576 + index * 4, 4, 0xf); // OMAP_TARGET RT<index> R..A
    }

    /// PS: this shader reads the interpolated fragment position.
    pub fn readsPosition(self: *Sph) void {
        self.set(188, 4, 0xf); // IMAP_POSITION_X..W
    }

    /// PS: this shader writes gl_FragDepth. Sets OMAP_DEPTH (SPHV4T2 bit 609, from
    /// clcb97sph.h NVCB97_SPHV4T2_OMAP_DEPTH = MW(609:609)) so the ROP takes the fragment
    /// depth from the shader's depth-output register instead of the interpolated z.
    pub fn writesDepth(self: *Sph) void {
        self.set(609, 1, 1); // OMAP_DEPTH
    }

    /// VS: this shader reads the DA-delivered vertex id (gl_VertexIndex) from the
    /// system-value attribute interface (ALD a[0x2fc]). The vertex-id sysval lives in
    /// imap_system_values_c, which for a VTG shader is SPH bits 336..352, at index
    /// (0x2fc-0x2c0)/4 = 15 -> absolute bit 351 (NAK FragmentIoInfo/VtgIoInfo:
    /// sysvals_in.c |= 1 << ((addr-0x2c0)/4)). Without this the DA does not deliver
    /// the vertex id into the attribute RAM, so the ALD reads 0.
    pub fn readsVertexId(self: *Sph) void {
        self.set(336 + 15, 1, 1); // imap_system_values_c bit 15 (vertex id at a[0x2fc])
    }

    /// VS: this shader reads the DA-delivered instance id (gl_InstanceIndex) from
    /// a[0x2f8] -> imap_system_values_c index (0x2f8-0x2c0)/4 = 14 -> absolute bit 350.
    pub fn readsInstanceId(self: *Sph) void {
        self.set(336 + 14, 1, 1); // imap_system_values_c bit 14 (instance id at a[0x2f8])
    }

    /// PS: this shader reads gl_PointCoord (the point-sprite s/t coord). The two sprite
    /// inputs live at a[0x2e0] (s) and a[0x2e4] (t), sysval indices (0x2e0-0x2c0)/4 = 8
    /// and 9 in imap_system_values_c. CRITICAL: for a FRAGMENT shader that field is at SPH
    /// bits 464..480 (NAK sph.rs imap_system_values_c: Fragment -> 464..480, VTG -> 336..352),
    /// so S = bit 464+8 = 472, T = bit 473. (The cla097sph.h MW(344:344) macro is the VTG
    /// layout; a fragment SPH bit 344 lands in imap_g_ps (192..448), NOT the sysval field,
    /// so the sprite coord is never declared and the IPA a[0x2e0] reads 0.) Pairs with
    /// SET_POINT_SPRITE + SET_POINT_SPRITE_SELECT in the draw state.
    pub fn readsPointSprite(self: *Sph) void {
        self.set(472, 1, 1); // IMAP_POINT_SPRITE_S (a[0x2e0]) - fragment imap_system_values_c bit 8
        self.set(473, 1, 1); // IMAP_POINT_SPRITE_T (a[0x2e4]) - fragment imap_system_values_c bit 9
    }

    /// VS: this shader reads generic input attribute `index` (a fetched vertex
    /// attribute, e.g. a position from a vertex buffer), all 4 components. The
    /// generic input map (imap_g_vtg) starts at bit 192, 4 bits per attribute.
    pub fn readsGeneric(self: *Sph, index: u32) void {
        self.set(192 + index * 4, 4, 0xf);
    }

    /// VS: this shader writes generic OUTPUT varying `index` (4 components), to be
    /// interpolated and consumed by the fragment shader. The VTG generic output
    /// map (omap_g) starts at bit 432, 4 bits (one per X/Y/Z/W component) per
    /// generic output. Generic output 0 -> o[0x80], output 1 -> o[0x90], etc.
    /// Also extends the store-request range to cover the varying so the engine
    /// knows the VS produces it: o[index] lives at byte 0x80 + index*0x10, 4
    /// comps -> dwords [.. (0x80 + index*0x10 + 0x10 - 1)/4]. The start is the
    /// position start (0x70/4); writesPosition() must also be called.
    pub fn writesVarying(self: *Sph, index: u32) void {
        self.set(432 + index * 4, 4, 0xf); // OMAP_G output `index` X..W
        const end_byte: u32 = 0x80 + index * 0x10 + 0x10; // exclusive
        self.set(152, 8, (end_byte - 1) / 4); // STORE_REQ_END extends to cover it
    }

    /// PS: this shader reads (perspective-interpolated) generic INPUT varying
    /// `index`, all 4 components, via IPA. The fragment generic input map
    /// (imap_g_ps) starts at bit 192, 2 bits per component (PixelImap:
    /// Unused=0, Constant=1, ScreenLinear=2, Perspective=3). Varying `index`
    /// occupies attribute words [index*4 .. index*4+4) at a[0x80 + index*0x10].
    /// A plain IPA(Pass,Default) does perspective-correct interpolation
    /// implicitly on SM70+; NO barycentric (pervertex_imap / ldtram) declaration
    /// is needed (NAK only marks barycentric_attr_in for explicit ldtram_nv,
    /// never for a simple ipa_nv), so this imap is the whole story.
    pub fn readsVarying(self: *Sph, index: u32) void {
        const base = 192 + index * 8; // 4 comps * 2 bits each
        var c: u32 = 0;
        while (c < 4) : (c += 1) self.set(base + c * 2, 2, 3); // Perspective
    }

    pub fn bytes(self: *const Sph) []const u8 {
        return std.mem.sliceAsBytes(self.data[0..]);
    }
};

fn f2u(f: f32) u32 {
    return @bitCast(f);
}

/// The full nvk_push_draw_state_init port: programs every piece of fixed-function
/// 3D state a shaded draw needs (SLM/TLS, watermarks, CT_MRT/CT_WRITE, DA
/// defaults, blend/point/zcull defaults, viewport clip control, scissor disable,
/// ROOT_TABLE, CB0 select+bind+zero, mesh-off, separate-topology). Call after
/// threed.begin(); `w`x`h` is the render target, `tls_va` the shader-local-memory
/// base, `cb0_va` the constant-buffer-0 base. Copied verbatim from the proven
/// gradient-triangle probe.
/// Full per-submit draw-state init (kept for cold paths + tests): the static state plus the two
/// per-submit-reset pieces. The hot draw path instead emits `initStaticState` ONCE per context
/// (channel method state persists across submits) and calls `resetVertexStreams` + `zeroConstantBuffer0`
/// every submit - see context.zig submit(). w/h are unused (viewport is set separately).
pub fn initDrawState(s: *threed.Stream, w: u32, h: u32, tls_va: u64, cb0_va: u64) void {
    _ = w;
    _ = h;
    initStaticState(s, tls_va, cb0_va);
    resetVertexStreams(s);
    zeroConstantBuffer0(s);
}

/// Disable all 32 vertex streams. MUST run every submit (channel state persists): the draw path
/// re-enables only the stream(s) it uses, so a stream enabled by a prior submit would otherwise
/// leak and the DA could fetch from a stale binding.
pub fn resetVertexStreams(s: *threed.Stream) void {
    var b: u32 = 0;
    while (b < 32) : (b += 1) s.m1(0x1c00 + b * 16, 0);
}

/// Zero the first 256 dwords of the selected constant buffer (CB0). MUST run every submit: CB0
/// holds the per-draw bindless texture handles + UBO addresses, and an UNBOUND sampler slot reads
/// handle 0 -> the null TIC. Without re-zeroing, a stale non-zero handle from a prior submit would
/// point a would-be-unbound sampler at a recycled descriptor (Xid MMU fault / wrong texture). CB0
/// must already be selected + bound (initStaticState does that, once, and it persists).
/// The HW root table the graphics UBO base addresses + bindless texture handles live in (read by
/// the shader as c[24 + ROOT_TABLE_GRAPHICS] = c[25]). MUST match vulcan encode.graphics_root_table.
/// The old bound-cb0 + LOAD_CONSTANT_BUFFER path is incoherent at high TPC occupancy on Blackwell.
pub const ROOT_TABLE_GRAPHICS: u32 = 1;

/// Zero the graphics root table (c[25]) each submit so unbound sampler slots read handle 0 (the
/// null TIC descriptor). 64 dwords = the full 256-byte root table.
pub fn zeroConstantBuffer0(s: *threed.Stream) void {
    s.ni(0x0504, &.{ROOT_TABLE_GRAPHICS}); // SET_ROOT_TABLE_SELECTOR table=ROOT_TABLE_GRAPHICS offset=0
    const zeros = [_]u32{0} ** 64;
    s.ni(0x0508, &zeros); // LOAD_ROOT_TABLE: 64 dwords
}

/// The STATIC draw-state defaults: everything that does not have to be reset per submit (it is a
/// fixed constant, or the begin/emitDrawState path re-emits it every submit/draw anyway). Emitted
/// ONCE per context by the hot path; the GPU retains method state across submits on the channel.
/// Excludes the vertex-stream disable and the CB0 zero (see resetVertexStreams / zeroConstantBuffer0).
pub fn initStaticState(s: *threed.Stream, tls_va: u64, cb0_va: u64) void {
    s.m1(0x0320, 0); // SET_TESSELLATION_PARAMETERS off
    s.m1(0x1558, 1); // SET_RENDER_ENABLE_C MODE_TRUE
    s.m1(0x19cc, 1); // SET_Z_COMPRESSION ENABLE_TRUE (bindDepth overrides to 0 for the
    // uncompressed ZETA; the default-on value here matches the proven color path)
    {
        var i: u32 = 0;
        while (i < 8) : (i += 1) s.m1(0x19e0 + i * 4, 1); // SET_COLOR_COMPRESSION(0..7)
    }
    s.m1(0x121c, 1); // SET_CT_SELECT target_count=1
    s.m1(0x020c, 1); // SET_ALIASED_LINE_WIDTH_ENABLE
    s.m1(0x0de8, 0); // SET_DA_PRIMITIVE_RESTART off
    s.m1(0x133c, 1); // SET_BLEND_SEPARATE_FOR_ALPHA
    s.m1(0x0f90, 1); // SET_SINGLE_CT_WRITE_CONTROL
    s.m1(0x135c, 0); // SET_SINGLE_ROP_CONTROL off
    s.m1(0x1594, 1); // SET_TWO_SIDED_STENCIL_TEST
    s.m1(0x16b4, 1); // SET_ALPHA_TO_COVERAGE_OVERRIDE
    s.m1(0x12d4, 0x1d01); // SET_SHADE_MODE OGL_SMOOTH
    s.m1(0x0d64, 8); // SET_API_VISIBLE_CALL_LIMIT V__128
    s.m1(0x151c, 1); // SET_ZCULL_STATS
    s.m1(0x0d9c, 0); // SET_REDUCE_COLOR_THRESHOLDS_ENABLE
    s.m1(0x0d94, 1); // SET_SHADER_CACHE_CONTROL
    s.m1(0x16a8, 0x30003); // CHECK_SPH_VERSION current=3 oldest=3
    s.m1(0x1794, 0x20002); // CHECK_AAM_VERSION current=2 oldest=2
    s.m1(0x1140, 1 << 4); // SET_BLEND_PER_FORMAT_ENABLE
    s.m1(0x1610, 0xE); // SET_ATTRIBUTE_DEFAULT
    s.m1(0x164c, 0x1000); // SET_DA_OUTPUT vertex_id_uses_array_start
    s.m1(0x030c, 0); // SET_RENDER_ENABLE_CONTROL conditional_load_cb=FALSE
    s.m1(0x0300, 0x3); // SET_PS_OUTPUT_SAMPLE_MASK_USAGE
    s.m1(0x0fdc, 1); // SET_BLEND_OPT_CONTROL
    s.m1(0x19c0, 1); // SET_BLEND_FLOAT_OPTION
    s.m1(0x12e4, 1); // SET_BLEND_STATE_PER_TARGET
    s.m1(0x12ec, 0); // SET_ALPHA_TEST off
    s.m1(0x1688, 0); // SET_TWO_SIDED_LIGHT off
    s.m1(0x2600, 1); // SET_COLOR_CLAMP
    s.m1(0x13a8, 0); // SET_PS_SATURATE off
    s.m1(0x1514, 0); // SET_ZPASS_PIXEL_COUNT off
    s.m1(0x0d68, 0); // SET_STATISTICS_COUNTER off
    s.m1(0x1518, f2u(16.0)); // SET_POINT_SIZE
    s.m1(0x1910, 1); // SET_ATTRIBUTE_POINT_SIZE
    s.m1(0x1520, 1); // SET_POINT_SPRITE
    // SET_POINT_SPRITE_SELECT: RMODE_ZERO (bits 1:0=0), ORIGIN_TOP (bit 2:2=1), all
    // TEXTURE0..9 PASSTHROUGH (gl_PointCoord is delivered via the NAK_ATTR_POINT_SPRITE
    // imap, not the legacy fixed-function TEXTURE#_GENERATE path). Matches NVK's
    // nvk_cmd_draw. ORIGIN_TOP makes t=0 at the sprite top to match gl_PointCoord.
    s.m1(0x1604, 1 << 2); // SET_POINT_SPRITE_SELECT: ORIGIN_TOP (bit 2)
    s.m1(0x1658, 0); // SET_ANTI_ALIASED_POINT off
    s.m1(0x0db4, 0); // SET_POLY_SMOOTH off
    s.m1(0x1924, 0); // SET_VIEWPORT_PIXEL CENTER_AT_HALF_INTEGERS
    s.m1(0x1534, 1); // SET_ANTI_ALIAS_ENABLE
    // SET_HYBRID_ANTI_ALIAS_CONTROL (0x0754): PASSES (bits 3:0) + CENTROID (bit 4).
    // For single-sample, no-sample-shading draws nvk programs PASSES=1,
    // CENTROID=PER_FRAGMENT(0) => 0x01 (nvk_mme_set_anti_alias: passes=1<<passes_log2,
    // centroid = passes>1 ? PER_PASS : PER_FRAGMENT). This forces the pixel shader to
    // run ONCE PER PIXEL at the pixel center, which is what makes the SM pack fragments
    // as spatial 2x2 quads into consecutive warp lanes - the prerequisite for the
    // SHFL.BFLY quad-shuffle screen-space derivatives (dFdx/dFdy) to read the correct
    // neighbour. Leaving this at its reset default left lanes scattered so a lane XOR 1
    // reached a distant pixel, not the horizontal quad neighbour.
    s.m1(0x0754, 0x01); // SET_HYBRID_ANTI_ALIAS_CONTROL PASSES=1 CENTROID=PER_FRAGMENT
    // SET_VARIABLE_PIXEL_RATE_SHADING_CONTROL(0) (0x2a00): ENABLE bit 0 = FALSE.
    // Variable-rate / coarse shading makes one fragment cover an NxM pixel block, so
    // the 2x2 quad would span many pixels and the SHFL.BFLY derivative neighbour would
    // be the wrong pixel. nvk keeps VRS disabled unless the app requests it; force it
    // off so every fragment is exactly one pixel and the quad is a true 2x2 pixel quad.
    s.m1(0x2a00, 0x00); // VRS disabled (one fragment == one pixel)
    // SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER (0x0280 + i*4, i=0..12): the per-tile
    // fragment->sample ORDERING the blob (and nvk, Turing+) always programs (13 opaque
    // dwords, copied verbatim "from the way the blob sets up the hardware",
    // nvk_cmd_draw.c cls_eng3d >= TURING_A). nvk-matched hardening; for single-sample
    // draws it does not change the quad lane->pixel packing (verified: the frame-based
    // partner distance is unchanged with vs without it), but it matches nvk's init.
    s.mm(0x0280, &.{
        0xa23eb139, 0xfb72ea61, 0xd950c843, 0x88fac4e5,
        0x1ab3e1b6, 0xa98fedc2, 0x2107654b, 0xe0539773,
        0x698badcf, 0x71032547, 0xdef05397, 0x56789abc,
        0x1234,
    });
    s.m1(0x13ac, 0); // SET_WINDOW_ORIGIN UPPER_LEFT
    s.mm(0x0df8, &.{ 0, 0 }); // SET_WINDOW_OFFSET_X/Y
    s.m1(0x196c, 0); // SET_ZCULL_BOUNDS bounded
    s.m1(0x1968, 1); // SET_ZCULL z_enable=1
    s.m1(0x197c, 0); // SET_CLIP_ID_TEST off
    s.m1(0x192c, 1); // SET_VIEWPORT_SCALE_OFFSET enable
    s.m1(0x193c, 0x18); // SET_VIEWPORT_CLIP_CONTROL pixel_min/max_z=CLAMP
    {
        var i: u32 = 0;
        while (i < 16) : (i += 1) s.m1(0x0e00 + i * 16, 0); // SET_SCISSOR_ENABLE(0..15) off
    }
    s.m1(0x0fac, 1); // SET_CT_MRT_ENABLE
    {
        var i: u32 = 0;
        while (i < 8) : (i += 1) s.m1(0x1a00 + i * 4, 0x1111); // SET_CT_WRITE(0..7) RGBA
    }
    // Turing+ ROOT_TABLE: visibility + 8 tables x 64 dwords zeroed.
    {
        var i: u32 = 0;
        while (i < 8) : (i += 1) s.m1(0x0240 + i * 4, 0x3 | (0x3 << 4) | (0x3 << 8) | (0x3 << 12) | (0x3 << 16));
    }
    {
        var t: u32 = 0;
        while (t < 8) : (t += 1) {
            s.ni(0x0504, &.{t & 0x7}); // SET_ROOT_TABLE_SELECTOR
            const zeros = [_]u32{0} ** 64;
            s.ni(0x0508, &zeros); // LOAD_ROOT_TABLE x64
        }
    }
    s.m1(0x02d0, 0x3f); // SET_ROOT_TABLE_PREFETCH
    s.m1(0x07ac, 1); // SET_TEXTURE_HEADER_VERSION
    // shader local memory window + TLS base/size.
    s.m1(SET_SHADER_LOCAL_MEMORY_WINDOW, 0xff << 24);
    s.mm(SET_SHADER_LOCAL_MEMORY_A, &.{ @intCast(tls_va >> 32), @truncate(tls_va), 0, 0x200000, 0 });
    // BIND_GROUP_CONSTANT_BUFFER: unbind all (group0..4, slot0..15).
    {
        var g: u32 = 0;
        while (g < 5) : (g += 1) {
            var sl: u32 = 0;
            while (sl < 16) : (sl += 1) s.m1(0x2410 + g * 32, sl << 4);
        }
    }
    s.m1(0x15cc, 0); // SET_RT_LAYER
    s.m1(0x165c, 0); // SET_POINT_CENTER_MODE V_OGL
    s.m1(0x15e4, 1); // SET_EDGE_FLAG
    s.m1(0x1234, 0); // SET_SAMPLER_BINDING V_INDEPENDENTLY
    s.m1(0x1948, 1); // SET_PRIMITIVE_TOPOLOGY_CONTROL: separate topology state
    s.m1(0x1450, 0x8 | (0x40 << 16)); // SET_PS_WARP_WATERMARKS low=8 high=64
    s.m1(0x1454, 0x80 | (0x1000 << 16)); // SET_PS_REGISTER_WATERMARKS
    // CB0: select + bind to all 5 groups slot0 (persists across submits; the per-submit
    // zeroConstantBuffer0 re-zeroes its contents each submit while this binding stays live).
    s.mm(0x2380, &.{ 0x10000, @intCast(cb0_va >> 32), @truncate(cb0_va) }); // SET_CONSTANT_BUFFER_SELECTOR size=64KB
    {
        var g: u32 = 0;
        while (g < 5) : (g += 1) s.m1(0x2410 + g * 32, 1 | (0 << 4)); // BIND_GROUP_CONSTANT_BUFFER valid slot=0
    }
    s.m1(0x1438, 0); // SET_GLOBAL_BASE_INSTANCE_INDEX
    s.m1(0x1434, 0); // SET_GLOBAL_BASE_VERTEX_INDEX
    s.m1(0x1118, 0); // SET_VERTEX_ID_BASE
    s.m1(0x0d74, 0); // SET_VERTEX_ARRAY_START
    s.m1(0x114c, 0); // SET_MESH_CONTROL off: first stage is a classic VS
    // raster: disable cull, solid fill, CCW front face.
    s.m1(0x1918, 0); // OGL_SET_CULL disable
    s.m1(0x12d0, 3); // SET_FILL_MODE SOLID
    s.m1(0x191c, 0x901); // OGL_SET_FRONT_FACE CCW
}

/// Write `vals` into the currently selected constant buffer (CB0, selected by
/// initDrawState and bound to all shader groups at slot 0) starting at byte
/// `byte_offset` in the graphics HW ROOT TABLE (read back by the shader as LDC c[25][byte_offset..]).
/// Used to bind UBO base addresses + bindless texture handles so a SPIR-V-compiled VS/PS can load
/// its uniforms. Writes via SET_ROOT_TABLE_SELECTOR (0x0504: table in bits 2:0, BYTE offset in bits
/// 15:8 - so byte_offset must stay < 256) + LOAD_ROOT_TABLE (0x0508, streams dwords). This replaced
/// the SET_LOAD_CONSTANT_BUFFER_OFFSET/LOAD_CONSTANT_BUFFER (cb0) path, which is NOT coherent with
/// the SM's LDC at high TPC occupancy on Blackwell (the glmark2 fault). See [[prism-glmark2-perf-cliff]].
pub fn loadConstantBuffer(s: *threed.Stream, byte_offset: u32, vals: []const u32) void {
    s.ni(0x0504, &.{ROOT_TABLE_GRAPHICS | (byte_offset << 8)}); // SET_ROOT_TABLE_SELECTOR (table | byteoff<<8)
    s.ni(0x0508, vals); // LOAD_ROOT_TABLE: stream dwords at the selected table+offset
}

/// Bind vertex stream `idx` to GPU buffer `va` of `size_bytes`, with per-vertex
/// `stride`. Uses the Turing+ SET_VERTEX_STREAM_SIZE (byte size), NOT the
/// pre-Turing LIMIT (which hangs the DA on Blackwell).
pub fn setVertexStream(s: *threed.Stream, idx: u32, va: u64, size_bytes: u32, stride: u32) void {
    s.m1(vertexStreamFormat(idx), stride | (1 << 12)); // stride | enable
    s.m1(0x1880 + idx * 4, 0); // SET_VERTEX_STREAM_INSTANCE(idx)=0: per-vertex
    s.m1(0x1c0c + idx * 16, 0); // SET_VERTEX_STREAM_A_FREQUENCY(idx)=0
    s.mm(vertexStreamLocationHi(idx), &.{ @intCast(va >> 32), @truncate(va) });
    s.mm(vertexStreamSizeHi(idx), &.{ 0, size_bytes }); // size_A upper=0, size_B=bytes
}

/// Mark vertex attribute `idx` active: sourced from `stream` at byte `offset`,
/// `comps` (1..4) 32-bit FLOAT components. The component width must match the real
/// attribute (a vec2/vec3 input) so the DA fetch stays within the bound stream's
/// SIZE; declaring a wider vec4 for a tightly-packed vec2/vec3 over-reads the last
/// vertex and faults the Data Assembler.
pub fn setVertexAttribute(s: *threed.Stream, idx: u32, stream: u32, offset_bytes: u32, comps: u32) void {
    s.m1(vertexAttribute(idx), stream | (offset_bytes << 7) | (attrBitWidths(comps) << 21) | (ATTR_NUM_TYPE_FLOAT << 27));
}

/// Mark vertex attribute `idx` inactive (constant source, not fetched).
pub fn setVertexAttributeInactive(s: *threed.Stream, idx: u32) void {
    s.m1(vertexAttribute(idx), (1 << 6) | (1 << 21));
}

/// Bind the shader whose SPH is at `sph_va` into pipeline `slot` (VS=1, PS=5),
/// with `reg_count` registers and constant-buffer `bind_group` (VS=0, PS=4).
pub fn bindShader(s: *threed.Stream, slot: u32, sph_va: u64, reg_count: u32, bind_group: u32) void {
    s.m1(pipeline.shader(slot), 1 | (slot << 4)); // enable | type
    s.mm(pipeline.programAddress(slot), &.{ @intCast(sph_va >> 32), @truncate(sph_va) });
    s.m1(pipeline.registerCount(slot), reg_count);
    s.m1(pipeline.binding(slot), bind_group);
}

/// Program viewport 0: clip space [-1,1] -> the `w`x`h` pixel rect, z mapped
/// [-1,1] NDC -> [0,1], plus the horizontal/vertical clip extent.
///
/// Y MAPS NDC y=-1 -> row 0 (top), y=+1 -> row h (bottom): window_y = (ndc_y+1)/2*h
/// (a POSITIVE Y scale). This is the SAME framebuffer Y-origin the software driver
/// uses (drivers/software/context.zig: `(ndc_y*0.5+0.5)*H`, no GL flip), so the two
/// HAL backends render identical orientation for the same NDC geometry - the EGL/GLES
/// path, the Vulkan ICD, and every oracle agree. (Was a NEGATIVE Y scale = the
/// OpenGL-style flip, which mirrored every nvidia frame vertically vs software and
/// upside-down'd GLES textures; the keyed dFdy-sign compensation in vulcan's nvidia
/// isel was reverted in lock-step with this.)
/// Program viewport 0 to the window rect (x, y, w, h) in top-left pixels: NDC [-1,1] maps into
/// [x, x+w] x [y, y+h] (scale/offset), and the VIEWPORT_CLIP extent clips rasterization to it. A
/// full-RT viewport is (0, 0, w, h) - identical to the pre-viewport programming. The SET_SCREEN_
/// SCISSOR full-surface clip is programmed separately (per submit) with the render-target size.
pub fn setViewport(s: *threed.Stream, x: i32, y: i32, w: u32, h: u32, znear: f32, zfar: f32) void {
    const fx: f32 = @floatFromInt(x);
    const fy: f32 = @floatFromInt(y);
    const fw: f32 = @floatFromInt(w);
    const fh: f32 = @floatFromInt(h);
    s.m1(viewportScaleX(0), f2u(fw / 2.0));
    s.m1(viewportOffsetX(0), f2u(fx + fw / 2.0));
    s.m1(viewportScaleX(0) + 4, f2u(fh / 2.0)); // scale Y (NDC -1 -> top, matches software)
    s.m1(viewportOffsetX(0) + 4, f2u(fy + fh / 2.0)); // offset Y
    // glDepthRangef: z_win = near + z_ndc*(far-near). Prism NDC z is [0,1] (Vulkan), so scaleZ =
    // far-near, offsetZ = near. Default [0,1] = identity (scaleZ=1, offsetZ=0).
    s.m1(viewportScaleX(0) + 8, f2u(zfar - znear)); // scale Z
    s.m1(viewportOffsetX(0) + 8, f2u(znear)); // offset Z
    // VIEWPORT_CLIP extent: XMIN|XMAX<<16, YMIN|YMAX<<16 (XMAX/YMAX exclusive). Clamp the min edges
    // to >= 0 (a partially-offscreen viewport still transforms via the f32 offset above; the clip
    // only bounds the visible region, and the full-surface SET_SCREEN_SCISSOR bounds the max edges).
    const xmin: u32 = @intCast(@max(@as(i32, 0), x));
    const ymin: u32 = @intCast(@max(@as(i32, 0), y));
    const xmax: u32 = @intCast(@max(@as(i32, 0), x + @as(i32, @intCast(w))));
    const ymax: u32 = @intCast(@max(@as(i32, 0), y + @as(i32, @intCast(h))));
    s.mm(viewportClipHorizontal(0), &.{ xmin | (xmax << 16), ymin | (ymax << 16) });
    // SET_VIEWPORT_CLIP_MIN_Z / MAX_Z (0x0c08 / 0x0c0c): the [0,1] depth clip range
    // the ROP-Z clamps fragment depth to. nvk sets these on Volta+ directly (not via
    // MME). They default to garbage; with a ZETA bound the ROP-Z validates against
    // them, so leaving them unset faults the depth draw (Xid 69 / ErrorCode 0x9c).
    s.m1(viewportClipHorizontal(0) + 8, f2u(@min(znear, zfar))); // SET_VIEWPORT_CLIP_MIN_Z
    s.m1(viewportClipHorizontal(0) + 12, f2u(@max(znear, zfar))); // SET_VIEWPORT_CLIP_MAX_Z
}

// Per-viewport scissor (NVCE97 SET_SCISSOR_*, index 0). Distinct from SET_SCREEN_SCISSOR
// (0x0ff4, the always-on full-surface clip setViewport programs): this is the app-controlled
// scissor rectangle (glScissor / vkCmdSetScissor). ENABLE defaults FALSE (nvk sets it false
// at init), so an un-scissored draw is unaffected.
pub const SET_SCISSOR_ENABLE = 0x0e00; // (j)*16; j=0. V bit 0
pub const SET_SCISSOR_HORIZONTAL = 0x0e04; // XMIN 15:0, XMAX 31:16 (exclusive)
pub const SET_SCISSOR_VERTICAL = 0x0e08; // YMIN 15:0, YMAX 31:16 (exclusive)

/// Enable + set the scissor rectangle in window pixels, top-left origin (matching Prism's
/// viewport: NDC y=-1 -> row 0). `x`/`y` are the inclusive top-left; `x+w`/`y+h` the
/// exclusive bottom-right. The caller clamps to the surface so XMIN/XMAX fit u16.
pub fn setScissor(s: *threed.Stream, x: u32, y: u32, w: u32, h: u32) void {
    s.m1(SET_SCISSOR_ENABLE, 1);
    s.m1(SET_SCISSOR_HORIZONTAL, (x & 0xffff) | (((x + w) & 0xffff) << 16));
    s.m1(SET_SCISSOR_VERTICAL, (y & 0xffff) | (((y + h) & 0xffff) << 16));
}

/// Disable the app scissor (fall back to the full-screen SET_SCREEN_SCISSOR only).
pub fn disableScissor(s: *threed.Stream) void {
    s.m1(SET_SCISSOR_ENABLE, 0);
}

/// Which triangle face the fixed-function rasterizer discards.
pub const CullFace = enum { none, front, back };
/// Which screen-space winding the rasterizer treats as the front face.
pub const FrontWinding = enum { cw, ccw };

/// Program the OpenGL-style back-face cull state (NVB097 OGL_SET_CULL /
/// OGL_SET_FRONT_FACE / OGL_SET_CULL_FACE), matching nvk's
/// nvk_mme_set_cull / vk_to_nv9097_cull_mode emit. The default initDrawState
/// disables culling (draw both windings); a pipeline that requests culling re-emits
/// this with the face + front winding it wants. The front winding is interpreted in
/// the SAME window-space orientation Prism's positive-Y viewport produces (NDC y=-1
/// -> row 0), so it agrees with the software rasterizer's signed-area winding test.
pub fn setCull(s: *threed.Stream, face: CullFace, front: FrontWinding) void {
    s.m1(0x191c, switch (front) { // OGL_SET_FRONT_FACE
        .cw => 0x900,
        .ccw => 0x901,
    });
    if (face == .none) {
        s.m1(0x1918, 0); // OGL_SET_CULL disable
        return;
    }
    s.m1(0x1918, 1); // OGL_SET_CULL enable
    s.m1(0x1920, switch (face) { // OGL_SET_CULL_FACE
        .front => 0x404,
        .back => 0x405,
        .none => unreachable,
    });
}

/// An OpenGL-style blend coefficient (NV9097 SET_BLEND_*_COEFF OGL_* value, verified
/// against cl9097.h). The numeric value is the method payload written verbatim.
pub const BlendCoeff = enum(u32) {
    zero = 0x4000,
    one = 0x4001,
    src_color = 0x4300,
    one_minus_src_color = 0x4301,
    src_alpha = 0x4302,
    one_minus_src_alpha = 0x4303,
    dst_alpha = 0x4304,
    one_minus_dst_alpha = 0x4305,
    dst_color = 0x4306,
    one_minus_dst_color = 0x4307,
    src_alpha_saturate = 0x4308,
    constant_color = 0xC001,
    one_minus_constant_color = 0xC002,
    constant_alpha = 0xC003,
    one_minus_constant_alpha = 0xC004,
};

/// An OpenGL-style blend equation (NV9097 SET_BLEND_*_OP OGL_* value, per cl9097.h).
pub const BlendEquation = enum(u32) {
    add = 0x8006,
    subtract = 0x800A,
    reverse_subtract = 0x800B,
    min = 0x8007,
    max = 0x8008,
};

/// The per-draw blend state the context maps from the HAL BlendState. `enable` toggles
/// SET_BLEND(0); the separate RGB / alpha coefficients + ops drive the fixed-function
/// blender; `constant` is glBlendColor (the CONSTANT_* coefficients read it).
pub const BlendState = struct {
    enable: bool,
    src_color: BlendCoeff = .one,
    dst_color: BlendCoeff = .zero,
    src_alpha: BlendCoeff = .one,
    dst_alpha: BlendCoeff = .zero,
    color_op: BlendEquation = .add,
    alpha_op: BlendEquation = .add,
    constant: [4]f32 = .{ 0, 0, 0, 0 },
};

/// Program the NV9097 blend state for render target 0 for this draw. The default
/// initDrawState set SET_BLEND_STATE_PER_TARGET=1 (0x12e4), so the GPU reads the
/// PER_TARGET blend methods (0x1e00 + j*32) for target j, NOT the global SET_BLEND_COLOR_OP
/// (0x1340) set - this matches nvk's nvk_cmd_draw blend emit. When `b.enable` is false this
/// just disables SET_BLEND(0) (a pure passthrough - the fragment overwrites the destination,
/// matching the no-blend path; the default-passthrough draws stay byte-identical). When
/// enabled it writes the per-target separate-for-alpha flag, the color/alpha ops + source/
/// dest coefficients, the blend constant (glBlendColor), and enables SET_BLEND for target 0.
pub fn setBlend(s: *threed.Stream, b: BlendState) void {
    if (!b.enable) {
        s.m1(0x1360, 0); // SET_BLEND(0) = FALSE
        return;
    }
    // The blend constant (glBlendColor): SET_BLEND_CONST_RED/GREEN/BLUE/ALPHA (0x131c..0x1328).
    s.m1(0x131c, f2u(b.constant[0])); // SET_BLEND_CONST_RED
    s.m1(0x1320, f2u(b.constant[1])); // SET_BLEND_CONST_GREEN
    s.m1(0x1324, f2u(b.constant[2])); // SET_BLEND_CONST_BLUE
    s.m1(0x1328, f2u(b.constant[3])); // SET_BLEND_CONST_ALPHA
    // Per-target (target 0) blend state: methods 0x1e00 + 0*32.
    s.m1(0x1e00, 1); // SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA(0) = TRUE
    s.m1(0x1e04, @intFromEnum(b.color_op)); // SET_BLEND_PER_TARGET_COLOR_OP(0)
    s.m1(0x1e08, @intFromEnum(b.src_color)); // SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF(0)
    s.m1(0x1e0c, @intFromEnum(b.dst_color)); // SET_BLEND_PER_TARGET_COLOR_DEST_COEFF(0)
    s.m1(0x1e10, @intFromEnum(b.alpha_op)); // SET_BLEND_PER_TARGET_ALPHA_OP(0)
    s.m1(0x1e14, @intFromEnum(b.src_alpha)); // SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF(0)
    s.m1(0x1e18, @intFromEnum(b.dst_alpha)); // SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF(0)
    s.m1(0x1360, 1); // SET_BLEND(0) = TRUE
}

/// Issue a draw of `count` vertices from `first` with the given `topology`, via
pub const SET_GLOBAL_BASE_INSTANCE_INDEX = 0x1438;

/// Set the DA's base instance index: the VS's instance-id attribute a[0x2f8]
/// (gl_InstanceIndex) reads this value. An instanced draw replays the draw once per
/// instance, bumping this so each replay's gl_InstanceIndex = first_instance + inst.
pub fn setBaseInstance(s: *threed.Stream, base: u32) void {
    s.m1(SET_GLOBAL_BASE_INSTANCE_INDEX, base);
}

/// the Blackwell combined DRAW_VERTEX_ARRAY_BEGIN_END (NVC597).
pub fn draw(s: *threed.Stream, topology: Topology, first: u32, count: u32) void {
    drawInstanced(s, topology, first, count, 1);
}

pub const SET_ALIASED_LINE_WIDTH_FLOAT = 0x13b4; // f32 line width (aliased/non-AA lines)

/// Set the aliased line width (glLineWidth). initDrawState enables aliased line width, so
/// this float method controls the width of a LINES-topology draw's segments.
pub fn setLineWidth(s: *threed.Stream, width: f32) void {
    s.m1(SET_ALIASED_LINE_WIDTH_FLOAT, @bitCast(width));
}

pub const SET_ATTRIBUTE_POINT_SIZE = 0x1910; // ENABLE bit 0: take point size from the shader

/// Enable/disable taking the point size from the shader's gl_PointSize output (a[0x6c])
/// instead of the fixed SET_POINT_SIZE. Enabled for a POINTS draw whose VS writes gl_PointSize.
pub fn setAttributePointSize(s: *threed.Stream, enable: bool) void {
    s.m1(SET_ATTRIBUTE_POINT_SIZE, if (enable) 1 else 0);
}

/// SET_DRAW_CONTROL_A.INSTANCE_ITERATE_ENABLE (bit 9). Without it the DA never iterates
/// the vertex array per instance, so gl_InstanceIndex (a[0x2f8]) stays 0 for every vertex.
pub const DRAW_CONTROL_A_INSTANCE_ITERATE_ENABLE = 1 << 9;

/// Instanced draw: the DA iterates the vertex array `instance_count` times, delivering
/// gl_InstanceIndex (a[0x2f8]) = 0..instance_count-1 (plus SET_GLOBAL_BASE_INSTANCE_INDEX).
/// Matches nvk's Turing+ draw loop: SET_DRAW_CONTROL_A carries the topology plus
/// INSTANCE_ITERATE_ENABLE, and its second field (DRAW_CONTROL_B = 0x0264) is the instance
/// count; then DRAW_VERTEX_ARRAY_BEGIN_END_A (0x0270) supplies start + count.
pub fn drawInstanced(s: *threed.Stream, topology: Topology, first: u32, count: u32, instance_count: u32) void {
    const t = @intFromEnum(topology);
    const draw_control_a = t | DRAW_CONTROL_A_INSTANCE_ITERATE_ENABLE;
    s.m1(0x1970, topology.primTopoV()); // SET_PRIMITIVE_TOPOLOGY (its own encoding, 1/2/4)
    s.mm(0x0260, &.{ draw_control_a, @max(instance_count, 1) }); // SET_DRAW_CONTROL_A, DRAW_CONTROL_B (instance_count)
    s.mm(0x0270, &.{ first, count }); // DRAW_VERTEX_ARRAY_BEGIN_END start, count
}

test "sph encodes vertex + fragment headers" {
    var vs = Sph.vertex();
    vs.writesPosition();
    // SPH_TYPE=1 (0:4) | VERSION=4 (5:9) | SHADER_TYPE=1 (10:13) | SASS_VERSION=1 (17:20)
    try std.testing.expectEqual(@as(u32, 1 | (4 << 5) | (1 << 10) | (1 << 17)), vs.data[0]);
    // OMAP_POSITION_X..W at bit 428 -> word 13, offset 12
    try std.testing.expectEqual(@as(u32, 0xf << 12), vs.data[13]);

    var ps = Sph.fragment();
    ps.writesColor();
    // PS: + MRT_ENABLE (14) + SASS_VERSION (17)
    try std.testing.expectEqual(@as(u32, 2 | (4 << 5) | (5 << 10) | (1 << 14) | (1 << 17)), ps.data[0]);
    // OMAP_TARGET RT0 at bit 576 -> word 18, offset 0
    try std.testing.expectEqual(@as(u32, 0xf), ps.data[18]);
    // REQUIRED imap_system_values_ab bit 31 -> word 5 bit 31 (or the DS traps)
    try std.testing.expectEqual(@as(u32, 0x80000000), ps.data[5]);
}

test "sph encodes a varying VS output + PS input" {
    var vs = Sph.vertex();
    vs.writesPosition();
    vs.readsGeneric(0); // attr0 = position in
    vs.writesVarying(0); // generic output 0 = the color varying
    // OMAP_G output 0 at bit 432 -> word 13 (432/32), bit 16 (432%32) = 0xf<<16.
    // writesPosition already set OMAP_POSITION at bit 428 = bits 12..15 -> 0xf<<12.
    try std.testing.expectEqual(@as(u32, (0xf << 12) | (0xf << 16)), vs.data[13]);
    // STORE_REQ_END (bit 152 -> word 4 byte 3) extends to (0x90-1)/4 = 0x23.
    try std.testing.expectEqual(@as(u32, 0x23), (vs.data[4] >> 24) & 0xff);
    // STORE_REQ_START (bit 140 -> word 4) stays 0x70/4 = 0x1c.
    try std.testing.expectEqual(@as(u32, 0x1c), (vs.data[4] >> 12) & 0xff);

    var ps = Sph.fragment();
    ps.writesColor();
    ps.readsVarying(0); // perspective-interpolated generic input 0
    // imap_g_ps base = bit 192 -> word 6 bit 0. 4 comps x 2 bits, all Perspective(3)
    // -> 0b11111111 = 0xff in the low byte of word 6.
    try std.testing.expectEqual(@as(u32, 0xff), ps.data[6] & 0xff);
    // The required imap_system_values_ab bit 31 (word 5) survives.
    try std.testing.expectEqual(@as(u32, 0x80000000), ps.data[5]);
    // No barycentric pervertex_imap (bits 672..800 -> words 21..24) is set.
    try std.testing.expectEqual(@as(u32, 0), ps.data[21]);
}

test "pipeline slot method offsets" {
    try std.testing.expectEqual(@as(u32, 0x2000), pipeline.shader(0));
    try std.testing.expectEqual(@as(u32, 0x2040), pipeline.shader(1)); // + 1*64
    try std.testing.expectEqual(@as(u32, 0x2004), pipeline.program(0));
    try std.testing.expectEqual(@as(u32, 0x200c), pipeline.registerCount(0));
}

test "setVertexStream emits format/size/location" {
    var buf: [32]u32 = undefined;
    var s = threed.Stream{ .buf = &buf };
    setVertexStream(&s, 0, 0x3000000, 0x1000, 32);
    // first method: SET_VERTEX_STREAM_A_FORMAT(0) = stride | enable(1<<12)
    try std.testing.expectEqual(@as(u32, vertexStreamFormat(0) >> 2), buf[0] & 0x1fff);
    try std.testing.expectEqual(@as(u32, 32 | (1 << 12)), buf[1]);
}

test "setVertexAttribute marks active R32G32B32A32 FLOAT" {
    var buf: [8]u32 = undefined;
    var s = threed.Stream{ .buf = &buf };
    setVertexAttribute(&s, 1, 0, 16, 4); // attr1, stream0, offset 16, vec4
    try std.testing.expectEqual(@as(u32, vertexAttribute(1) >> 2), buf[0] & 0x1fff);
    try std.testing.expectEqual(@as(u32, (16 << 7) | (ATTR_R32_G32_B32_A32 << 21) | (7 << 27)), buf[1]);
}

test "setVertexAttribute honors the real component width (vec2/vec3 fetch the right comps)" {
    // A vec2 attribute declares R32_G32 (NOT a vec4) so the DA does not over-read a
    // tightly-packed stream's last vertex and fault; a vec3 declares R32_G32_B32.
    {
        var buf: [8]u32 = undefined;
        var s = threed.Stream{ .buf = &buf };
        setVertexAttribute(&s, 0, 0, 0, 2);
        try std.testing.expectEqual(@as(u32, (0 << 7) | (ATTR_R32_G32 << 21) | (7 << 27)), buf[1]);
    }
    {
        var buf: [8]u32 = undefined;
        var s = threed.Stream{ .buf = &buf };
        setVertexAttribute(&s, 1, 0, 8, 3);
        try std.testing.expectEqual(@as(u32, (8 << 7) | (ATTR_R32_G32_B32 << 21) | (7 << 27)), buf[1]);
    }
}

test "bindShader writes enable/address/regs/binding" {
    var buf: [16]u32 = undefined;
    var s = threed.Stream{ .buf = &buf };
    bindShader(&s, 1, 0x1000080, 64, 0); // VS slot 1, group 0
    try std.testing.expectEqual(@as(u32, pipeline.shader(1) >> 2), buf[0] & 0x1fff);
    try std.testing.expectEqual(@as(u32, 1 | (1 << 4)), buf[1]); // enable | type
}

test "bindDepth emits ZT address/format/size/select" {
    var buf: [64]u32 = undefined;
    var s = threed.Stream{ .buf = &buf };
    bindDepth(&s, 0x12340000, 256, 256);
    // First method is SET_ZT_A (address upper), then the lower.
    try std.testing.expectEqual(@as(u32, SET_ZT_A >> 2), buf[0] & 0x1fff);
    try std.testing.expectEqual(@as(u32, 0x12340000 >> 32), buf[1]);
    try std.testing.expectEqual(@as(u32, 0x12340000 & 0xffffffff), buf[2]);
    // Walk the stream and confirm ZT_FORMAT=ZF32 and SET_ZT_SELECT=1 are present.
    var i: usize = 0;
    var saw_format = false;
    var saw_select = false;
    while (i < s.dwords()) {
        const addr = (buf[i] & 0x1fff) << 2;
        const cnt = (buf[i] >> 16) & 0x1fff;
        if (addr == SET_ZT_FORMAT) {
            saw_format = buf[i + 1] == ZT_FORMAT_ZF32;
        }
        if (addr == SET_ZT_SELECT) saw_select = buf[i + 1] == 1;
        i += 1 + cnt;
    }
    try std.testing.expect(saw_format);
    try std.testing.expect(saw_select);
}

test "setDepthTest enables test + func LESS + write" {
    var buf: [16]u32 = undefined;
    var s = threed.Stream{ .buf = &buf };
    setDepthTest(&s, .less, true);
    try std.testing.expectEqual(@as(u32, SET_DEPTH_TEST >> 2), buf[0] & 0x1fff);
    try std.testing.expectEqual(@as(u32, 1), buf[1]); // test enable
    try std.testing.expectEqual(@as(u32, SET_DEPTH_FUNC >> 2), buf[2] & 0x1fff);
    try std.testing.expectEqual(@as(u32, 0x201), buf[3]); // OGL_LESS
    try std.testing.expectEqual(@as(u32, SET_DEPTH_WRITE >> 2), buf[4] & 0x1fff);
    try std.testing.expectEqual(@as(u32, 1), buf[5]); // write enable
}

test "setBlend disabled emits only SET_BLEND(0)=0" {
    var buf: [8]u32 = undefined;
    var s = threed.Stream{ .buf = &buf };
    setBlend(&s, .{ .enable = false });
    try std.testing.expectEqual(@as(u32, 0x1360 >> 2), buf[0] & 0x1fff); // SET_BLEND(0)
    try std.testing.expectEqual(@as(u32, 0), buf[1]); // disabled
}

test "setBlend enabled emits the per-target NV9097 blend methods + enable" {
    var buf: [32]u32 = undefined;
    var s = threed.Stream{ .buf = &buf };
    setBlend(&s, .{
        .enable = true,
        .src_color = .src_alpha,
        .dst_color = .one_minus_src_alpha,
        .src_alpha = .src_alpha,
        .dst_alpha = .one_minus_src_alpha,
        .color_op = .add,
        .alpha_op = .add,
        .constant = .{ 0.25, 0, 0, 1 },
    });
    // Method k occupies buf[2k] (header, addr>>2 in bits 0..12) + buf[2k+1] (value).
    const m = struct {
        fn addr(b: []const u32, k: usize) u32 {
            return b[2 * k] & 0x1fff;
        }
        fn val(b: []const u32, k: usize) u32 {
            return b[2 * k + 1];
        }
    };
    try std.testing.expectEqual(@as(u32, 0x131c >> 2), m.addr(&buf, 0)); // SET_BLEND_CONST_RED
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 0.25))), m.val(&buf, 0));
    try std.testing.expectEqual(@as(u32, 0x1e00 >> 2), m.addr(&buf, 4)); // PER_TARGET_SEPARATE_FOR_ALPHA(0)
    try std.testing.expectEqual(@as(u32, 1), m.val(&buf, 4));
    try std.testing.expectEqual(@as(u32, 0x1e04 >> 2), m.addr(&buf, 5)); // PER_TARGET_COLOR_OP(0)
    try std.testing.expectEqual(@as(u32, 0x8006), m.val(&buf, 5)); // OGL_FUNC_ADD
    try std.testing.expectEqual(@as(u32, 0x1e08 >> 2), m.addr(&buf, 6)); // COLOR_SOURCE_COEFF(0)
    try std.testing.expectEqual(@as(u32, 0x4302), m.val(&buf, 6)); // OGL_SRC_ALPHA
    try std.testing.expectEqual(@as(u32, 0x1e0c >> 2), m.addr(&buf, 7)); // COLOR_DEST_COEFF(0)
    try std.testing.expectEqual(@as(u32, 0x4303), m.val(&buf, 7)); // OGL_ONE_MINUS_SRC_ALPHA
    try std.testing.expectEqual(@as(u32, 0x1e14 >> 2), m.addr(&buf, 9)); // ALPHA_SOURCE_COEFF(0)
    try std.testing.expectEqual(@as(u32, 0x4302), m.val(&buf, 9)); // OGL_SRC_ALPHA
    try std.testing.expectEqual(@as(u32, 0x1360 >> 2), m.addr(&buf, 11)); // SET_BLEND(0)
    try std.testing.expectEqual(@as(u32, 1), m.val(&buf, 11)); // enabled
}

test "clearDepth sets clear value + CLEAR_SURFACE Z bit" {
    var buf: [16]u32 = undefined;
    var s = threed.Stream{ .buf = &buf };
    clearDepth(&s, 1.0);
    try std.testing.expectEqual(@as(u32, SET_Z_CLEAR_VALUE >> 2), buf[0] & 0x1fff);
    try std.testing.expectEqual(@as(u32, @as(u32, @bitCast(@as(f32, 1.0)))), buf[1]);
    try std.testing.expectEqual(@as(u32, CLEAR_SURFACE >> 2), buf[2] & 0x1fff);
    try std.testing.expectEqual(@as(u32, 1), buf[3]); // Z_ENABLE only
}

test "ztSizeBytes rounds to GOB + block height footprint" {
    // 256x256 ZF32: row = align(256*4,64)=1024; height = align(256, 16*8=128)=256.
    try std.testing.expectEqual(@as(u32, 1024 * 256), ztSizeBytes(256, 256));
    // A height not a block multiple rounds up to the block (128 rows).
    try std.testing.expectEqual(@as(u32, 1024 * 128), ztSizeBytes(256, 100));
}

test "fillTic encodes an RGBA8 block-linear 2D texture header (V2_BL)" {
    const tic = fillTic(0x12340a00, 2, 2, .rgba8_unorm, 0); // 2x2, address GOB-aligned (bits 0..9 == 0)
    // HEADER_VERSION = SELECT_BLOCKLINEAR_V2 (3) at MW 127:124 -> word 3 bits 28..31.
    try std.testing.expectEqual(@as(u32, 3), (tic[3] >> 28) & 0xf);
    // COMPONENTS = A8B8G8R8 (0x08) at MW 118:112 -> word 3 bits 16..22.
    try std.testing.expectEqual(@as(u32, 0x08), (tic[3] >> 16) & 0x7f);
    // X/Y/Z/W_SOURCE = R/G/B/A (2,3,4,5) at MW 98:96.. -> word 3 bits 0..11.
    try std.testing.expectEqual(@as(u32, 2 | (3 << 3) | (4 << 6) | (5 << 9)), tic[3] & 0xfff);
    // WIDTH_MINUS_ONE (1) at MW 144:128 -> word 4 bits 0..16.
    try std.testing.expectEqual(@as(u32, 1), tic[4] & 0x1ffff);
    // HEIGHT_MINUS_ONE (1) at MW 176:160 -> word 5 bits 0..16.
    try std.testing.expectEqual(@as(u32, 1), tic[5] & 0x1ffff);
    // The address low bits 9..32 land in word 0 bits 9.. (the rest of the address).
    try std.testing.expectEqual(@as(u32, (0x12340a00 >> 9) & 0x7fffff), (tic[0] >> 9) & 0x7fffff);
}

test "fillTicPitch encodes an RGBA8 pitch-linear 2D texture header (V2_PITCH)" {
    const pitch = texPitchBytes(2); // 32 (align(8,32))
    try std.testing.expectEqual(@as(u32, 32), pitch);
    const tic = fillTicPitch(0x12340020, 2, 2, pitch);
    // HEADER_VERSION = SELECT_PITCH_V2 (2) at MW 127:124 -> word 3 bits 28..31.
    try std.testing.expectEqual(@as(u32, 2), (tic[3] >> 28) & 0xf);
    // COMPONENTS = A8B8G8R8 at MW 118:112.
    try std.testing.expectEqual(@as(u32, 0x08), (tic[3] >> 16) & 0x7f);
    // PITCH_BITS21TO5 at MW 80:64 -> word 2 bits 0..16 = pitch/32 = 1.
    try std.testing.expectEqual(@as(u32, 1), tic[2] & 0x1ffff);
    // WIDTH/HEIGHT minus one.
    try std.testing.expectEqual(@as(u32, 1), tic[4] & 0x1ffff);
    try std.testing.expectEqual(@as(u32, 1), tic[5] & 0x1ffff);
    // Address bits 5..32 land in word0 bits 5.. (0x12340020 >> 5).
    try std.testing.expectEqual(@as(u32, (0x12340020 >> 5) & 0x7ffffff), (tic[0] >> 5) & 0x7ffffff);
}

test "fillTsc encodes NEAREST + REPEAT sampler" {
    const tsc = fillTsc(.nearest, .repeat, .repeat, .none, 1);
    // TEXSAMP0: ADDRESS_U/V = WRAP (0).
    try std.testing.expectEqual(@as(u32, 0), tsc[0] & 0x3f);
    // TEXSAMP1 (word 1): MAG_FILTER = MAG_POINT (1) at 2:0, MIN_FILTER = MIN_POINT (1) at 5:4.
    try std.testing.expectEqual(@as(u32, 1), tsc[1] & 0x7);
    try std.testing.expectEqual(@as(u32, 1), (tsc[1] >> 4) & 0x3);
    try std.testing.expectEqual(@as(u32, 1), (tsc[1] >> 6) & 0x3); // MIP_NONE
    // A LINEAR sampler uses MAG/MIN_LINEAR (2).
    const lin = fillTsc(.linear, .clamp_to_edge, .clamp_to_edge, .none, 1);
    try std.testing.expectEqual(@as(u32, 2), lin[1] & 0x7);
    try std.testing.expectEqual(@as(u32, 2), (lin[0]) & 0x7); // ADDRESS_U = CLAMP_TO_EDGE (2)
}

test "bindTexturePools emits header + sampler pool methods" {
    var buf: [32]u32 = undefined;
    var s = threed.Stream{ .buf = &buf };
    bindTexturePools(&s, 0x5000000, 0, 0x6000000, 0);
    try std.testing.expectEqual(@as(u32, SET_TEX_HEADER_POOL_A >> 2), buf[0] & 0x1fff);
    try std.testing.expectEqual(@as(u32, 0x5000000 >> 32), buf[1]);
    try std.testing.expectEqual(@as(u32, 0x5000000 & 0xffffffff), buf[2]);
}

test "texSizeBytes + blTexPixelOffset cover a small texture in one GOB" {
    // A 2x2 texture: one GOB wide (64 B), one GOB tall (8 rows) = 512 B.
    try std.testing.expectEqual(@as(u5, 0), texBlockHeightLog2(2));
    try std.testing.expectEqual(@as(u32, 64 * 8), texSizeBytes(2, 2, .rgba8_unorm));
    // Texel (0,0) is at byte 0; the four 2x2 texels map to distinct in-GOB offsets.
    const o00 = blTexPixelOffset(0, 0, 2, 2, .rgba8_unorm);
    const o10 = blTexPixelOffset(1, 0, 2, 2, .rgba8_unorm);
    const o01 = blTexPixelOffset(0, 1, 2, 2, .rgba8_unorm);
    const o11 = blTexPixelOffset(1, 1, 2, 2, .rgba8_unorm);
    try std.testing.expectEqual(@as(usize, 0), o00);
    try std.testing.expect(o10 != o00 and o01 != o00 and o11 != o00);
    try std.testing.expect(o10 != o01 and o10 != o11 and o01 != o11);
}

test "draw emits topology + begin/end" {
    var buf: [16]u32 = undefined;
    var s = threed.Stream{ .buf = &buf };
    draw(&s, .triangles, 0, 3);
    try std.testing.expectEqual(@as(u32, 0x1970 >> 2), buf[0] & 0x1fff);
    try std.testing.expectEqual(@as(u32, 4), buf[1]); // triangles topology value
}
