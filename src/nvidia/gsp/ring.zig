//! The GSP message-queue ring transport (plan phase P5). Pure logic over a
//! memory region: on hardware that region is the sysmem the GPU DMAs, but the
//! ring math and framing are identical for an offline test - which is why this
//! whole file is unit-testable against a second in-memory endpoint with no GPU.
//!
//! A ring is one backing store laid out (per OGK msgq.c msgqTxCreate) as:
//!   [ msgqTxHeader @ 0 ][ msgqRxHeader @ rxHdrOff ][ entry slots @ entryOff ]
//! The PRODUCER owns the txHeader (writePtr) + the entry slots; the CONSUMER
//! owns the rxHeader (readPtr) in that same store. For the GSP RM transport the
//! host->GSP command queue (cmdq) and the GSP->host status queue (msgq) are two
//! separate backing stores, each with one producer and one consumer.
//!
//! Index math + framing are matched field-for-field to OGK 595.71.05:
//!   - src/nvidia/src/libraries/msgq/msgq.c        (free/wrap, write/read slots)
//!   - src/nvidia/src/kernel/gpu/gsp/message_queue_cpu.c (send/recv framing)
//!   - src/nvidia/inc/kernel/gpu/gsp/message_queue_priv.h (_checkSum32, BytesToElements)
//!
//! Confidential Compute: GspMsgQueueSendCommand gates AES-GCM on
//! gpuIsCCFeatureEnabled(pGpu); consumer GA10x has CC OFF, so this implements
//! the plaintext path (checksum over [hdr + rpc.length], auth/aad buffers left
//! zero). The checksum is XOR-fold (NOT additive): _checkSum32 XORs the element
//! as NvU64 words then folds hi^lo, and check_sum is chosen so the whole element
//! XOR-folds to 0.

const std = @import("std");
const proto = @import("proto.zig");
const sdk = @import("../sdk.zig");

pub const Error = error{
    /// The backing store is too small to hold the headers + at least one slot.
    RegionTooSmall,
    /// No free slots for the (possibly multi-slot) element being sent.
    QueueFull,
    /// recv() found the ring empty (readPtr == writePtr).
    Empty,
    /// The received element failed checksum verification.
    BadChecksum,
    /// The received element's seqNum did not match the expected rx sequence.
    BadSeqNum,
    /// The framed RPC length is outside [sizeof(element), elementSizeMax].
    BadLength,
    /// The payload does not fit in a single element's max byte span.
    PayloadTooLarge,
};

/// queueHeaderAlign == 4 -> 1<<4 == 16 (msgq.c hdrAlign).
const HDR_ALIGN: u32 = @as(u32, 1) << proto.QUEUE_HEADER_ALIGN_SHIFT;
/// queueElementAlign == 12 -> 1<<12 == 4096 (msgq.c entryAlign, == slot size).
const ENTRY_ALIGN: u32 = @as(u32, 1) << proto.QUEUE_ELEMENT_ALIGN_SHIFT;
/// queueElementSizeMin == 4096: the per-slot byte size (msgSize).
pub const SLOT_SIZE: u32 = proto.ELEMENT_SIZE_MIN;

fn alignUp(v: u32, a: u32) u32 {
    return (v + a - 1) & ~(a - 1);
}

/// rxHdrOff = ALIGN_UP(sizeof(msgqTxHeader)=32, 16) == 32.
pub const RX_HDR_OFF: u32 = alignUp(@sizeOf(proto.MsgqTxHeader), HDR_ALIGN);
/// entryOff = ALIGN_UP(rxHdrOff + sizeof(msgqRxHeader)=4, 4096) == 4096.
pub const ENTRY_OFF: u32 = alignUp(RX_HDR_OFF + @sizeOf(proto.MsgqRxHeader), ENTRY_ALIGN);

comptime {
    std.debug.assert(RX_HDR_OFF == 32);
    std.debug.assert(ENTRY_OFF == 4096);
}

/// ceil(bytes / SLOT_SIZE): gspMsgQueueBytesToElements with queueElementSizeMin.
pub fn bytesToElements(bytes: u32) u32 {
    return (bytes + SLOT_SIZE - 1) / SLOT_SIZE;
}

/// Volatile load of a u32 at byte offset `off` in `base` (the GSP reads/writes
/// these ring pointers concurrently, so the compiler must not cache them).
inline fn rd(base: usize, off: u32) u32 {
    const p: *volatile u32 = @ptrFromInt(base + off);
    return p.*;
}

/// Volatile store of a u32 ring pointer.
inline fn wr(base: usize, off: u32, value: u32) void {
    const p: *volatile u32 = @ptrFromInt(base + off);
    p.* = value;
}

/// Acquire-load of the producer's writePtr: pairs with the release-store in
/// advanceWriteOrdered so the consumer never observes the pointer ahead of the
/// slot bytes. (Zig 0.16 dropped the standalone @fence builtin; the ordering
/// rides on the atomic pointer access instead.)
inline fn loadAcquire(base: usize, off: u32) u32 {
    const p: *u32 = @ptrFromInt(base + off);
    return @atomicLoad(u32, p, .acquire);
}

/// Release-store of a ring pointer: the slot bytes written before this call are
/// visible to any party that acquire-loads the pointer afterwards.
inline fn storeRelease(base: usize, off: u32, value: u32) void {
    const p: *u32 = @ptrFromInt(base + off);
    @atomicStore(u32, p, value, .release);
}

/// The 32-bit checksum from message_queue_priv.h `_checkSum32`: XOR the region
/// as NvU64 words (the region is zero-padded to an 8-byte boundary), then fold
/// hi32 ^ lo32. `len` need not be a multiple of 8; the trailing bytes are read
/// from the (zero-padded) backing slot, exactly like the C reads past the end
/// to the 8-byte alignment.
pub fn checkSum32(base: usize, len: u32) u32 {
    const words = (len + 7) / 8; // round up to the 8-byte boundary
    var acc: u64 = 0;
    var i: u32 = 0;
    while (i < words) : (i += 1) {
        const p: *align(1) const u64 = @ptrFromInt(base + i * 8);
        acc ^= p.*;
    }
    const hi: u32 = @truncate(acc >> 32);
    const lo: u32 = @truncate(acc & 0xffff_ffff);
    return hi ^ lo;
}

/// A view over one backing-store ring. `base` is the CPU-visible address of the
/// store; `size` its byte length. The txHeader/rxHeader/entries are at fixed
/// offsets. The same struct serves the producer (writePtr/send) and the consumer
/// (readPtr/recv) of the same store - on hardware the GSP is the other party.
pub const Ring = struct {
    base: usize,
    size: u32,
    msg_count: u32,

    const TX_OFF: u32 = 0;
    const WRITE_PTR_OFF: u32 = @offsetOf(proto.MsgqTxHeader, "write_ptr");
    const READ_PTR_OFF: u32 = RX_HDR_OFF; // msgqRxHeader.read_ptr is its only field

    /// Wrap a backing store. Does not write anything (use `format` to lay the
    /// headers down first). `size` must hold the headers plus >= one slot.
    pub fn init(base: usize, size: u32) Error!Ring {
        if (size < ENTRY_OFF + SLOT_SIZE) return Error.RegionTooSmall;
        const msg_count = (size - ENTRY_OFF) / SLOT_SIZE;
        return .{ .base = base, .size = size, .msg_count = msg_count };
    }

    /// Lay down a fresh msgqTxHeader + zeroed rxHeader for this store (the
    /// producer side initialises this; mirrors msgq.c msgqTxCreate). writePtr
    /// and readPtr start at 0.
    pub fn format(self: Ring, flags: u32) void {
        const tx: *volatile proto.MsgqTxHeader = @ptrFromInt(self.base + TX_OFF);
        tx.version = proto.MsgqTxHeader.VERSION;
        tx.size = self.size;
        tx.msg_size = SLOT_SIZE;
        tx.msg_count = self.msg_count;
        tx.write_ptr = 0;
        tx.flags = flags;
        tx.rx_hdr_off = RX_HDR_OFF;
        tx.entry_off = ENTRY_OFF;
        wr(self.base, READ_PTR_OFF, 0); // rxHeader.read_ptr
    }

    inline fn writePtr(self: Ring) u32 {
        return rd(self.base, WRITE_PTR_OFF);
    }
    inline fn readPtr(self: Ring) u32 {
        return rd(self.base, READ_PTR_OFF);
    }

    /// Free slots available to the producer. msgq.c msgqTxGetFreeSpace:
    ///   free = readPtr + msgCount - writePtr - 1; if (free >= msgCount) free -= msgCount
    /// The "-1" keeps one slot empty so writePtr never catches readPtr from
    /// behind (full and empty would otherwise alias).
    pub fn freeSlots(self: Ring) u32 {
        var free = self.readPtr() +% self.msg_count -% self.writePtr() -% 1;
        if (free >= self.msg_count) free -%= self.msg_count;
        return free;
    }

    /// Elements available to the consumer. msgq.c msgqRxGetReadAvailable:
    ///   avail = writePtr + msgCount - readPtr; if (avail >= msgCount) avail -= msgCount
    pub fn available(self: Ring) u32 {
        var avail = self.writePtrAcquire() +% self.msg_count -% self.readPtr();
        if (avail >= self.msg_count) avail -%= self.msg_count;
        return avail;
    }

    /// Byte address of the n-th slot ahead of writePtr (wrapping), without
    /// advancing. msgq.c msgqTxGetWriteBuffer: wp = (writePtr + n) % msgCount.
    fn writeSlotAddr(self: Ring, n: u32) usize {
        var wp = self.writePtr() +% n;
        if (wp >= self.msg_count) wp -%= self.msg_count;
        return self.base + ENTRY_OFF + wp * SLOT_SIZE;
    }

    /// Byte address of the n-th slot ahead of readPtr (wrapping), without
    /// advancing. msgq.c msgqRxGetReadBuffer: rp = (readPtr + n) % msgCount.
    fn readSlotAddr(self: Ring, n: u32) usize {
        var rp = self.readPtr() +% n;
        if (rp >= self.msg_count) rp -%= self.msg_count;
        return self.base + ENTRY_OFF + rp * SLOT_SIZE;
    }

    /// Advance writePtr by `n` (producer commit) with release ordering, so the
    /// slot bytes written before the advance are visible to a consumer that
    /// acquire-loads writePtr. msgq.c msgqTxSubmitBuffers + the store fence in
    /// GspMsgQueueSendCommand.
    fn advanceWrite(self: Ring, n: u32) void {
        var wp = self.writePtr() +% n;
        if (wp >= self.msg_count) wp -%= self.msg_count;
        storeRelease(self.base, WRITE_PTR_OFF, wp);
    }

    /// Acquire-load of writePtr, used by the consumer's available()/read path so
    /// it pairs with the producer's release-store and never sees the pointer run
    /// ahead of the slot data.
    inline fn writePtrAcquire(self: Ring) u32 {
        return loadAcquire(self.base, WRITE_PTR_OFF);
    }

    /// Advance readPtr by `n` (consumer consume). msgq.c msgqRxMarkConsumed.
    fn advanceRead(self: Ring, n: u32) void {
        var rp = self.readPtr() +% n;
        if (rp >= self.msg_count) rp -%= self.msg_count;
        wr(self.base, READ_PTR_OFF, rp);
    }
};

/// A complete GSP message-queue endpoint: a producer ring (we write commands /
/// status into) and a consumer ring (we drain replies / events from), plus the
/// tx/rx sequence counters. On the host this is { cmdq=producer, msgq=consumer };
/// the test "GSP" end is the mirror { cmdq=consumer, msgq=producer }.
pub const Endpoint = struct {
    tx: Ring, // we produce into this store (host: cmdq, gsp: msgq)
    rx: Ring, // we consume from this store (host: msgq, gsp: cmdq)
    tx_seq: u32 = 0, // next seqNum we stamp on send
    rx_seq: u32 = 0, // next seqNum we expect on recv
    // Scratch staging for the receive path (one max element). Owned by the
    // Endpoint so the returned `rpc` slice stays valid until the next recv.
    rx_scratch: [proto.ELEMENT_SIZE_MAX]u8 = undefined,

    /// queueElementHdrSize (== NV_OFFSETOF(GSP_MSG_QUEUE_ELEMENT, rpc)).
    const ELEM_HDR: u32 = @intCast(proto.GspMsgQueueElement.HDR_SIZE);

    /// Frame and enqueue an RPC. `rpc_bytes` is the already-marshalled payload
    /// starting at the rpc_message_header_v (header + function body); its
    /// rpc.length field must already be set. We prefix the GSP_MSG_QUEUE_ELEMENT
    /// header, stamp seqNum/elemCount, zero-pad, compute the XOR-fold checksum so
    /// the element folds to 0, copy into the next free slot(s), store-fence, and
    /// advance writePtr. Mirrors GspMsgQueueSendCommand (plaintext path).
    pub fn send(self: *Endpoint, rpc_bytes: []const u8) Error!void {
        const msg_len: u32 = ELEM_HDR + @as(u32, @intCast(rpc_bytes.len));
        if (msg_len < @sizeOf(proto.GspMsgQueueElement) or msg_len > proto.ELEMENT_SIZE_MAX)
            return Error.BadLength;

        const elem_count = bytesToElements(msg_len);
        if (elem_count > self.tx.freeSlots()) return Error.QueueFull;

        // Build the element in the first slot's storage directly: zero the whole
        // span we will checksum (elem_count slots), lay the element header, then
        // the rpc payload. We assemble across the (logically contiguous, possibly
        // physically wrapping) slots one slot at a time, like the C.
        //
        // Stage into a single contiguous scratch first so the checksum + copy are
        // simple, then scatter into the (wrapping) ring slots.
        var scratch: [proto.ELEMENT_SIZE_MAX]u8 = undefined;
        const span = elem_count * SLOT_SIZE;
        @memset(scratch[0..span], 0);

        const elem: *proto.GspMsgQueueElement = @ptrCast(@alignCast(&scratch[0]));
        // auth_tag_buffer / aad_buffer stay zero (CC OFF, plaintext path).
        elem.auth_tag_buffer = [_]u8{0} ** 16;
        elem.aad_buffer = [_]u8{0} ** 16;
        elem.seq_num = self.tx_seq;
        elem.elem_count = elem_count;
        elem.check_sum = 0; // included in the checksum, so zero before computing
        elem._pad = 0;
        @memcpy(scratch[ELEM_HDR .. ELEM_HDR + rpc_bytes.len], rpc_bytes);

        // Checksum over [hdr + rpc.length] == msg_len (plaintext path). The
        // scratch is zero-padded past msg_len to the 8-byte boundary already.
        elem.check_sum = checkSum32(@intFromPtr(&scratch[0]), msg_len);

        // Scatter into the ring slots (each slot may wrap independently).
        var i: u32 = 0;
        while (i < elem_count) : (i += 1) {
            const dst: [*]u8 = @ptrFromInt(self.tx.writeSlotAddr(i));
            @memcpy(dst[0..SLOT_SIZE], scratch[i * SLOT_SIZE ..][0..SLOT_SIZE]);
        }

        // The slot bytes must land before writePtr advances, else the consumer
        // (GSP) could observe the pointer ahead of the data. advanceWrite uses a
        // release store to enforce that ordering.
        self.tx.advanceWrite(elem_count);
        self.tx_seq +%= 1;
    }

    /// A received RPC: the element's framing plus a slice of the payload (the
    /// rpc_message_header_v + function body), valid until the next recv.
    pub const Received = struct {
        function: u32,
        seq_num: u32,
        elem_count: u32,
        rpc: []const u8, // rpc_message_header_v + body, length == rpc.length
    };

    /// Poll the consumer ring; if an element is present, copy it out, verify the
    /// XOR-fold checksum and the seqNum, then advance readPtr. Mirrors
    /// GspMsgQueueReceiveStatus (plaintext path). Returns Error.Empty if nothing
    /// is queued.
    pub fn recv(self: *Endpoint) Error!Received {
        if (self.rx.available() == 0) return Error.Empty;

        // First slot carries elem_count; copy it, read the count, then copy the
        // rest. The available() check above acquire-loaded writePtr, pairing with
        // the producer's release store so the slot bytes are visible here.
        const first: [*]const u8 = @ptrFromInt(self.rx.readSlotAddr(0));
        @memcpy(self.rx_scratch[0..SLOT_SIZE], first[0..SLOT_SIZE]);

        const elem: *const proto.GspMsgQueueElement = @ptrCast(@alignCast(&self.rx_scratch[0]));
        const elem_count = elem.elem_count;
        if (elem_count == 0 or elem_count > bytesToElements(proto.ELEMENT_SIZE_MAX))
            return Error.BadLength;
        if (elem_count > self.rx.available()) return Error.Empty; // partial, not ready

        var i: u32 = 1;
        while (i < elem_count) : (i += 1) {
            const src: [*]const u8 = @ptrFromInt(self.rx.readSlotAddr(i));
            @memcpy(self.rx_scratch[i * SLOT_SIZE ..][0..SLOT_SIZE], src[0..SLOT_SIZE]);
        }

        // The rpc header sits right after the element header; its length field
        // bounds the checksum (plaintext path): checksum over [hdr + rpc.length].
        const rpc_hdr: *const proto.RpcMessageHeader = @ptrCast(@alignCast(&self.rx_scratch[ELEM_HDR]));
        const rpc_len = rpc_hdr.length;
        const msg_len = ELEM_HDR + rpc_len;
        if (msg_len < @sizeOf(proto.GspMsgQueueElement) or msg_len > proto.ELEMENT_SIZE_MAX)
            return Error.BadLength;

        if (checkSum32(@intFromPtr(&self.rx_scratch[0]), msg_len) != 0)
            return Error.BadChecksum;
        if (elem.seq_num != self.rx_seq) return Error.BadSeqNum;

        self.rx_seq +%= 1;
        self.rx.advanceRead(elem_count);

        return .{
            .function = rpc_hdr.function,
            .seq_num = elem.seq_num,
            .elem_count = elem_count,
            .rpc = self.rx_scratch[ELEM_HDR .. ELEM_HDR + rpc_len],
        };
    }
};

// ---------------------------------------------------------------------------
// RPC marshalling helpers (build a ready-to-send rpc_message_header_v + body).
// ---------------------------------------------------------------------------

/// Write the common rpc_message_header_v into `out` for `function`, with
/// `body_len` bytes of function body to follow. Returns the header byte count
/// (32). The caller fills out[32..32+body_len] with the function body, then sets
/// length to 32 + body_len (done here).
fn writeRpcHeader(out: []u8, function: u32, body_len: u32) void {
    const hdr: *proto.RpcMessageHeader = @ptrCast(@alignCast(&out[0]));
    hdr.* = .{
        .header_version = proto.RpcMessageHeader.HEADER_VERSION,
        .signature = proto.RpcMessageHeader.SIGNATURE,
        .length = @sizeOf(proto.RpcMessageHeader) + body_len,
        .function = function,
        .rpc_result = 0,
        .rpc_result_private = 0,
        .sequence = 0,
        .u = 0,
    };
}

/// Marshal a GSP_RM_ALLOC (function 103) RPC into `out`: rpc_message_header_v +
/// rpc_gsp_rm_alloc_v header + the sdk alloc `params` blob. Returns the number
/// of bytes written. Asserts 8-byte alignment of the params and that length /
/// params_size / function are consistent. `out` must be at least
/// 32 + 32 + params.len bytes.
pub fn buildAllocElement(
    out: []u8,
    h_client: sdk.NvHandle,
    h_parent: sdk.NvHandle,
    h_object: sdk.NvHandle,
    h_class: u32,
    flags: u32,
    params: []const u8,
) usize {
    const body_len: u32 = @sizeOf(proto.RpcGspRmAlloc) + @as(u32, @intCast(params.len));
    const total = @sizeOf(proto.RpcMessageHeader) + body_len;
    std.debug.assert(out.len >= total);

    writeRpcHeader(out, @intFromEnum(proto.Function.gsp_rm_alloc), body_len);

    const a: *proto.RpcGspRmAlloc = @ptrCast(@alignCast(&out[@sizeOf(proto.RpcMessageHeader)]));
    a.* = .{
        .h_client = h_client,
        .h_parent = h_parent,
        .h_object = h_object,
        .h_class = h_class,
        .status = 0,
        .params_size = @intCast(params.len),
        .flags = flags,
        .reserved = [_]u8{0} ** 4,
    };

    const params_off = @sizeOf(proto.RpcMessageHeader) + @sizeOf(proto.RpcGspRmAlloc);
    // The alloc header is 32 + 32 == 64 bytes, so the params land 8-byte aligned.
    std.debug.assert(params_off % 8 == 0);
    @memcpy(out[params_off .. params_off + params.len], params);
    return total;
}

/// Marshal a GSP_RM_CONTROL (function 76) RPC into `out`: rpc_message_header_v +
/// rpc_gsp_rm_control_v header + the sdk control `params` blob. Returns the
/// number of bytes written. `out` must be at least 32 + 40 + params.len bytes.
pub fn buildControlElement(
    out: []u8,
    h_client: sdk.NvHandle,
    h_object: sdk.NvHandle,
    cmd: u32,
    flags: u32,
    params: []const u8,
) usize {
    const body_len: u32 = @sizeOf(proto.RpcGspRmControl) + @as(u32, @intCast(params.len));
    const total = @sizeOf(proto.RpcMessageHeader) + body_len;
    std.debug.assert(out.len >= total);

    writeRpcHeader(out, @intFromEnum(proto.Function.gsp_rm_control), body_len);

    const c: *proto.RpcGspRmControl = @ptrCast(@alignCast(&out[@sizeOf(proto.RpcMessageHeader)]));
    c.* = .{
        .h_client = h_client,
        .h_object = h_object,
        .cmd = cmd,
        .status = 0,
        .params_size = @intCast(params.len),
        .rmapi_rpc_flags = flags,
        .rmctrl_flags = 0,
        .rmctrl_access_right = 0,
        .reserved0 = 0,
    };

    const params_off = @sizeOf(proto.RpcMessageHeader) + @sizeOf(proto.RpcGspRmControl);
    // 32 + 40 == 72, 8-byte aligned.
    std.debug.assert(params_off % 8 == 0);
    @memcpy(out[params_off .. params_off + params.len], params);
    return total;
}

// ===========================================================================
// Tests (all offline, no GPU).
// ===========================================================================

const testing = std.testing;

test "ring layout offsets match the OGK msgq.c constants" {
    try testing.expectEqual(@as(u32, 32), RX_HDR_OFF);
    try testing.expectEqual(@as(u32, 4096), ENTRY_OFF);
    try testing.expectEqual(@as(u32, 4096), SLOT_SIZE);
}

test "bytesToElements ceils against the 4KB slot" {
    try testing.expectEqual(@as(u32, 1), bytesToElements(1));
    try testing.expectEqual(@as(u32, 1), bytesToElements(4096));
    try testing.expectEqual(@as(u32, 2), bytesToElements(4097));
    try testing.expectEqual(@as(u32, 2), bytesToElements(8192));
    try testing.expectEqual(@as(u32, 3), bytesToElements(8193));
    // A header-only element (48 bytes) is one slot.
    try testing.expectEqual(@as(u32, 1), bytesToElements(@intCast(proto.GspMsgQueueElement.HDR_SIZE)));
}

test "ring free/available index math wraps correctly" {
    // 4 slots: entryOff(4096) + 4*4096.
    const size: u32 = ENTRY_OFF + 4 * SLOT_SIZE;
    const region = try testing.allocator.alignedAlloc(u8, .@"8", size);
    defer testing.allocator.free(region);
    @memset(region, 0);

    var ring = try Ring.init(@intFromPtr(region.ptr), size);
    try testing.expectEqual(@as(u32, 4), ring.msg_count);
    ring.format(0);

    // Empty: free = msgCount - 1 (one reserved), available = 0.
    try testing.expectEqual(@as(u32, 3), ring.freeSlots());
    try testing.expectEqual(@as(u32, 0), ring.available());

    // Advance writePtr 2 -> 2 available to consumer, 1 free to producer.
    ring.advanceWrite(2);
    try testing.expectEqual(@as(u32, 2), ring.available());
    try testing.expectEqual(@as(u32, 1), ring.freeSlots());

    // Consume 2 -> back to empty, full free again.
    ring.advanceRead(2);
    try testing.expectEqual(@as(u32, 0), ring.available());
    try testing.expectEqual(@as(u32, 3), ring.freeSlots());

    // Push write near the wrap: advance 3, consume 3, advance 3 again -> wptr
    // wraps past msgCount. Verify available stays correct across the boundary.
    ring.advanceWrite(3);
    ring.advanceRead(3);
    ring.advanceWrite(3); // wptr = (3+3) % 4 == 2, rptr == 3
    try testing.expectEqual(@as(u32, 3), ring.available()); // (2 + 4 - 3) % 4 == 3
    try testing.expectEqual(@as(u32, 0), ring.freeSlots()); // (3 + 4 - 2 - 1) % 4 == 0
}

test "checkSum32 makes a framed element fold to zero" {
    // A small element with arbitrary payload: after stamping check_sum the whole
    // [hdr + rpc.length] span must XOR-fold to 0.
    var buf: [SLOT_SIZE]u8 = undefined;
    @memset(&buf, 0);
    const elem: *proto.GspMsgQueueElement = @ptrCast(@alignCast(&buf[0]));
    elem.* = std.mem.zeroes(proto.GspMsgQueueElement);
    elem.seq_num = 7;
    elem.elem_count = 1;
    // Put some non-zero payload after the 48-byte header.
    const payload = buf[proto.GspMsgQueueElement.HDR_SIZE..][0..16];
    for (payload, 0..) |*b, i| b.* = @intCast(0xA0 + i);
    const msg_len: u32 = @as(u32, @intCast(proto.GspMsgQueueElement.HDR_SIZE)) + 16;

    elem.check_sum = 0;
    const cs = checkSum32(@intFromPtr(&buf[0]), msg_len);
    elem.check_sum = cs;
    try testing.expectEqual(@as(u32, 0), checkSum32(@intFromPtr(&buf[0]), msg_len));
}

test "two-ended ring round-trips an RPC end to end (no GPU)" {
    // One in-memory cmdq (host produces, gsp consumes) and one msgq (gsp
    // produces, host consumes). Both ends are CPU code over the same stores.
    const ring_size: u32 = ENTRY_OFF + 8 * SLOT_SIZE;

    const cmdq = try testing.allocator.alignedAlloc(u8, .@"8", ring_size);
    defer testing.allocator.free(cmdq);
    const msgq = try testing.allocator.alignedAlloc(u8, .@"8", ring_size);
    defer testing.allocator.free(msgq);
    @memset(cmdq, 0);
    @memset(msgq, 0);

    var cmdq_ring = try Ring.init(@intFromPtr(cmdq.ptr), ring_size);
    var msgq_ring = try Ring.init(@intFromPtr(msgq.ptr), ring_size);
    // The producer of each store formats its header.
    cmdq_ring.format(0);
    msgq_ring.format(0);

    var host = Endpoint{ .tx = cmdq_ring, .rx = msgq_ring };
    var gsp = Endpoint{ .tx = msgq_ring, .rx = cmdq_ring };

    // Host sends a GSP_RM_ALLOC root alloc (no params blob).
    var out: [256]u8 = undefined;
    const nv01_root: u32 = 0; // NV01_ROOT class id
    const n = buildAllocElement(&out, 0xCAF30001, 0, 0xCAF30002, nv01_root, 0, &[_]u8{});
    try host.send(out[0..n]);

    // GSP drains it: verifies checksum + seqNum, reads function + payload.
    const got = try gsp.recv();
    try testing.expectEqual(@intFromEnum(proto.Function.gsp_rm_alloc), got.function);
    try testing.expectEqual(@as(u32, 0), got.seq_num);
    try testing.expectEqual(@as(u32, 1), got.elem_count);
    const alloc_hdr: *const proto.RpcGspRmAlloc = @ptrCast(@alignCast(&got.rpc[@sizeOf(proto.RpcMessageHeader)]));
    try testing.expectEqual(@as(sdk.NvHandle, 0xCAF30001), alloc_hdr.h_client);
    try testing.expectEqual(@as(u32, 0), alloc_hdr.params_size);

    // GSP replies on the msgq with a GSP_INIT_DONE-style event element.
    var reply: [64]u8 = undefined;
    writeRpcHeader(&reply, @intFromEnum(proto.Event.gsp_init_done), 0);
    try gsp.send(reply[0..@sizeOf(proto.RpcMessageHeader)]);

    // Host receives the reply and round-trips.
    const r = try host.recv();
    try testing.expectEqual(@intFromEnum(proto.Event.gsp_init_done), r.function);
    try testing.expectEqual(@as(u32, 0), r.seq_num);

    // Ring is empty again on both ends.
    try testing.expectError(Error.Empty, host.recv());
    try testing.expectError(Error.Empty, gsp.recv());
}

test "two-ended ring: seqNum increments and a multi-slot payload spans slots" {
    const ring_size: u32 = ENTRY_OFF + 8 * SLOT_SIZE;
    const cmdq = try testing.allocator.alignedAlloc(u8, .@"8", ring_size);
    defer testing.allocator.free(cmdq);
    const msgq = try testing.allocator.alignedAlloc(u8, .@"8", ring_size);
    defer testing.allocator.free(msgq);
    @memset(cmdq, 0);
    @memset(msgq, 0);

    var cmdq_ring = try Ring.init(@intFromPtr(cmdq.ptr), ring_size);
    var msgq_ring = try Ring.init(@intFromPtr(msgq.ptr), ring_size);
    cmdq_ring.format(0);
    msgq_ring.format(0);

    var host = Endpoint{ .tx = cmdq_ring, .rx = msgq_ring };
    var gsp = Endpoint{ .tx = msgq_ring, .rx = cmdq_ring };

    // A payload large enough to need two slots: body > (4096 - 48 - 32) so the
    // alloc element header + params exceeds one slot.
    var big_params: [5000]u8 = undefined;
    for (&big_params, 0..) |*b, i| b.* = @truncate(i);
    var out: [proto.ELEMENT_SIZE_MAX]u8 = undefined;

    const n0 = buildAllocElement(&out, 0x111, 0x222, 0x333, 0x444, 0, &big_params);
    try host.send(out[0..n0]);
    // The framed element must be 2 slots (msg_len = 48 + 32 + 5000 = 5080 > 4096).
    const got0 = try gsp.recv();
    try testing.expectEqual(@as(u32, 2), got0.elem_count);
    try testing.expectEqual(@as(u32, 0), got0.seq_num);
    const a0: *const proto.RpcGspRmAlloc = @ptrCast(@alignCast(&got0.rpc[@sizeOf(proto.RpcMessageHeader)]));
    try testing.expectEqual(@as(u32, 5000), a0.params_size);
    // The params survived the slot scatter/gather intact.
    const got_params = got0.rpc[@sizeOf(proto.RpcMessageHeader) + @sizeOf(proto.RpcGspRmAlloc) ..];
    try testing.expectEqualSlices(u8, &big_params, got_params[0..5000]);

    // A second send increments the seqNum.
    const n1 = buildControlElement(&out, 0xABC, 0xDEF, 0x20800110, 0, &[_]u8{ 1, 2, 3, 4 });
    try host.send(out[0..n1]);
    const got1 = try gsp.recv();
    try testing.expectEqual(@as(u32, 1), got1.seq_num);
    try testing.expectEqual(@intFromEnum(proto.Function.gsp_rm_control), got1.function);
}

test "buildAllocElement on-wire bytes for a known root alloc" {
    // NV01_ROOT alloc: function 103, params == an Os21-shaped blob. Assert the
    // exact header fields land on the wire.
    var out: [128]u8 = undefined;
    const params = [_]u8{0xAA} ** 8; // stand-in 8-byte params blob
    const nv01_root: u32 = 0x0;
    const n = buildAllocElement(&out, 0x1000, 0x2000, 0x3000, nv01_root, 0x5, &params);

    try testing.expectEqual(@as(usize, 32 + 32 + 8), n);

    const rpc: *const proto.RpcMessageHeader = @ptrCast(@alignCast(&out[0]));
    try testing.expectEqual(proto.RpcMessageHeader.HEADER_VERSION, rpc.header_version);
    try testing.expectEqual(proto.RpcMessageHeader.SIGNATURE, rpc.signature);
    try testing.expectEqual(@intFromEnum(proto.Function.gsp_rm_alloc), rpc.function);
    // length == rpc header(32) + alloc header(32) + params(8) == 72. writeRpcHeader
    // sets length = sizeof(rpc_message_header_v) + body_len.
    try testing.expectEqual(@as(u32, @sizeOf(proto.RpcMessageHeader) + @sizeOf(proto.RpcGspRmAlloc) + 8), rpc.length);

    const a: *const proto.RpcGspRmAlloc = @ptrCast(@alignCast(&out[32]));
    try testing.expectEqual(@as(sdk.NvHandle, 0x1000), a.h_client);
    try testing.expectEqual(@as(sdk.NvHandle, 0x2000), a.h_parent);
    try testing.expectEqual(@as(sdk.NvHandle, 0x3000), a.h_object);
    try testing.expectEqual(@as(u32, 0x0), a.h_class);
    try testing.expectEqual(@as(u32, 8), a.params_size);
    try testing.expectEqual(@as(u32, 0x5), a.flags);
    try testing.expectEqualSlices(u8, &params, out[64..72]);
}

test "buildControlElement marshals the 595.71.05 control header" {
    var out: [128]u8 = undefined;
    const params = [_]u8{0xBB} ** 16;
    const n = buildControlElement(&out, 0xC11E, 0x0B7E, 0x20800142, 0, &params);
    try testing.expectEqual(@as(usize, 32 + 40 + 16), n);

    const rpc: *const proto.RpcMessageHeader = @ptrCast(@alignCast(&out[0]));
    try testing.expectEqual(@intFromEnum(proto.Function.gsp_rm_control), rpc.function);

    const c: *const proto.RpcGspRmControl = @ptrCast(@alignCast(&out[32]));
    try testing.expectEqual(@as(sdk.NvHandle, 0xC11E), c.h_client);
    try testing.expectEqual(@as(sdk.NvHandle, 0x0B7E), c.h_object);
    try testing.expectEqual(@as(u32, 0x20800142), c.cmd);
    try testing.expectEqual(@as(u32, 16), c.params_size);
    try testing.expectEqualSlices(u8, &params, out[72..88]);
}

test "send rejects an oversized payload and a full queue" {
    const ring_size: u32 = ENTRY_OFF + 2 * SLOT_SIZE; // msg_count == 2, free == 1
    const cmdq = try testing.allocator.alignedAlloc(u8, .@"8", ring_size);
    defer testing.allocator.free(cmdq);
    const msgq = try testing.allocator.alignedAlloc(u8, .@"8", ring_size);
    defer testing.allocator.free(msgq);
    @memset(cmdq, 0);
    @memset(msgq, 0);

    var cmdq_ring = try Ring.init(@intFromPtr(cmdq.ptr), ring_size);
    var msgq_ring = try Ring.init(@intFromPtr(msgq.ptr), ring_size);
    cmdq_ring.format(0);
    msgq_ring.format(0);
    var host = Endpoint{ .tx = cmdq_ring, .rx = msgq_ring };

    // free == 1 slot. A 1-slot send works; a second (without a drain) is full.
    var out: [256]u8 = undefined;
    const n = buildAllocElement(&out, 1, 2, 3, 4, 0, &[_]u8{});
    try host.send(out[0..n]);
    try testing.expectError(Error.QueueFull, host.send(out[0..n]));

    // An over-max payload is rejected up front.
    var huge: [proto.ELEMENT_SIZE_MAX + 8]u8 = undefined;
    try testing.expectError(Error.BadLength, host.send(&huge));
}
