//! Version-pinned (NVIDIA 595.71.05) GSP transport protocol structures, ported
//! from open-gpu-kernel-modules tag 595.71.05. These are the on-wire envelopes
//! the GSP RPC ring carries; the RM alloc/control PAYLOADS they wrap are the
//! existing sdk.zig structs (NOT re-ported here). Layouts must match the GSP
//! firmware ABI exactly (extern struct), so every struct carries a
//! `comptime`/test @sizeOf/@offsetOf assert against the C reference - this
//! mirrors sdk.zig's "RM struct layouts match the NVIDIA ABI" test.
//!
//! OGK headers matched (tag 595.71.05):
//!   - src/nvidia/generated/g_rpc-message-header.h   (rpc_message_header_v)
//!   - src/nvidia/generated/g_rpc-structures.h       (rpc_gsp_rm_alloc_v, rpc_gsp_rm_control_v)
//!   - src/nvidia/inc/kernel/vgpu/rpc_global_enums.h (NV_VGPU_MSG_FUNCTION_* / _EVENT_*)
//!   - src/nvidia/inc/kernel/vgpu/rpc_headers.h      (header version + signature)
//!   - src/nvidia/inc/kernel/gpu/gsp/message_queue_priv.h (GSP_MSG_QUEUE_ELEMENT)
//!   - src/nvidia/inc/libraries/msgq/msgq_priv.h     (msgqTxHeader, msgqRxHeader)

const std = @import("std");

// Reuse the RM scalar aliases so the GSP envelopes read like the C and stay
// consistent with sdk.zig (NvHandle == NvU32 etc).
const sdk = @import("../sdk.zig");
const NvU8 = sdk.NvU8;
const NvU32 = sdk.NvU32;
const NvU64 = sdk.NvU64;
const NvHandle = sdk.NvHandle;

// ---------------------------------------------------------------------------
// rpc_message_header_v (g_rpc-message-header.h, rpc_message_header_v03_00)
// ---------------------------------------------------------------------------

/// The common RPC message header that prefixes every GSP RPC payload. The
/// function body follows immediately (`rpc_message_data[]` in the C, omitted
/// here since Zig handles the trailing bytes separately). 32 bytes.
pub const RpcMessageHeader = extern struct {
    /// header_version: NV_VGPU_MSG_HEADER_VERSION MAJOR(31:24)=3, MINOR(23:16)=0.
    header_version: NvU32,
    /// signature: must equal NV_VGPU_MSG_SIGNATURE_VALID ("RPCV").
    signature: NvU32,
    /// length: total bytes of header + function body (NOT including the queue
    /// element header). The receiver sanity-checks this.
    length: NvU32,
    /// function: one of Function (NV_VGPU_MSG_FUNCTION_*) on cmds, or an Event
    /// (NV_VGPU_MSG_EVENT_*) on async GSP->host messages.
    function: NvU32,
    /// rpc_result: NV_STATUS the GSP returns (0 == NV_OK).
    rpc_result: NvU32,
    rpc_result_private: NvU32,
    /// sequence: RPC sequence number (distinct from the queue-element seqNum).
    sequence: NvU32,
    /// u: union { spare; cpuRmGfid } (NvU32). Spare on the consumer path.
    u: NvU32,

    /// NV_VGPU_MSG_HEADER_VERSION_MAJOR_TOT(3)<<24 | _MINOR_TOT(0)<<16.
    pub const HEADER_VERSION: NvU32 = 0x03000000;
    /// NV_VGPU_MSG_SIGNATURE_VALID ("RPCV").
    pub const SIGNATURE: NvU32 = 0x43505256;
};

// ---------------------------------------------------------------------------
// NV_VGPU_MSG_FUNCTION_* (rpc_global_enums.h)
// ---------------------------------------------------------------------------

/// The RPC function numbers the GSP transport carries (the subset Prism uses;
/// values are exact from rpc_global_enums.h at 595.71.05).
pub const Function = enum(NvU32) {
    alloc_root = 2,
    alloc_memory = 4,
    map_memory = 7,
    free = 10,
    unmap_memory = 13,
    map_memory_dma = 14,
    unmap_memory_dma = 15,
    get_gsp_static_info = 65,
    gsp_set_system_info = 72,
    set_registry = 73,
    gsp_rm_control = 76,
    get_static_info2 = 77,
    gsp_rm_alloc = 103,
    _,
};

// ---------------------------------------------------------------------------
// NV_VGPU_MSG_EVENT_* (rpc_global_enums.h, base FIRST_EVENT = 0x1000)
// ---------------------------------------------------------------------------

/// Async events the GSP posts on the status (msgq) ring. The `function` field
/// of an incoming element carries one of these when it is an event.
pub const Event = enum(NvU32) {
    first_event = 0x1000,
    gsp_init_done = 0x1001,
    rc_triggered = 0x1004,
    _,
};

// ---------------------------------------------------------------------------
// rpc_gsp_rm_alloc_v (g_rpc-structures.h, rpc_gsp_rm_alloc_v03_00)
// ---------------------------------------------------------------------------

/// GSP_RM_ALLOC (function 103) header. The RM alloc params (an sdk.zig struct,
/// e.g. Os21Params-described class params) follow `params[]`. 32-byte header.
pub const RpcGspRmAlloc = extern struct {
    h_client: NvHandle,
    h_parent: NvHandle,
    h_object: NvHandle,
    h_class: NvU32,
    status: NvU32,
    params_size: NvU32,
    flags: NvU32,
    reserved: [4]NvU8,
    // NvU8 params[] follows (handled out-of-band).
};

// ---------------------------------------------------------------------------
// rpc_gsp_rm_control_v (g_rpc-structures.h, rpc_gsp_rm_control_v03_00)
// ---------------------------------------------------------------------------

/// GSP_RM_CONTROL (function 76) header. NOTE: at 595.71.05 this header is LARGER
/// than older drops - it carries rmapiRpcFlags / rmctrlFlags / rmctrlAccessRight
/// and an 8-byte-aligned reserved0 (the plan's "24B header with a single flags"
/// reflected an older layout; the 595.71.05 source has the expanded form). The
/// control params (an sdk Os54-style blob) follow `params[]`. 40-byte header.
pub const RpcGspRmControl = extern struct {
    h_client: NvHandle,
    h_object: NvHandle,
    cmd: NvU32,
    status: NvU32,
    params_size: NvU32,
    rmapi_rpc_flags: NvU32,
    rmctrl_flags: NvU32,
    rmctrl_access_right: NvU32,
    reserved0: NvU64 align(8),
    // NvU8 params[] follows (handled out-of-band).
};

// ---------------------------------------------------------------------------
// GSP_MSG_QUEUE_ELEMENT (message_queue_priv.h)
// ---------------------------------------------------------------------------

/// The framing wrapped around every RPC payload on the shared-mem ring. The
/// `rpc` header (and its function body) is 8-byte aligned and follows the 44
/// declared bytes, so NV_DECLARE_ALIGNED(rpc, 8) pads the element header to 48
/// bytes (queueElementHdrSize = NV_OFFSETOF(GSP_MSG_QUEUE_ELEMENT, rpc) = 48).
///
/// authTagBuffer / aadBuffer are the Confidential-Compute AES-GCM tag + AAD; on
/// consumer GA10x (CC OFF) they stay zero and the plaintext checksum path is
/// used (see ring.zig). checkSum is the 32-bit XOR-fold value that makes the
/// whole element checksum to 0 (NOT an additive sum - _checkSum32 XORs 8-byte
/// words then folds hi^lo). seqNum / elemCount are maintained by the ring.
pub const GspMsgQueueElement = extern struct {
    auth_tag_buffer: [16]NvU8,
    aad_buffer: [16]NvU8,
    check_sum: NvU32,
    seq_num: NvU32,
    elem_count: NvU32,
    // padding to 8-byte align the rpc that follows; the rpc header + body live
    // out-of-band (not a struct field here, since they are variable length).
    _pad: NvU32,

    /// queueElementHdrSize = NV_OFFSETOF(GSP_MSG_QUEUE_ELEMENT, rpc) = 48.
    pub const HDR_SIZE: usize = 48;
};

// ---------------------------------------------------------------------------
// msgqTxHeader / msgqRxHeader (msgq_priv.h)
// ---------------------------------------------------------------------------

/// The producer-side ring header at the front of each backing store. The writer
/// owns this; the reader reads writePtr to know how much is available. 32 bytes.
pub const MsgqTxHeader = extern struct {
    version: NvU32, // MSGQ_VERSION (0)
    size: NvU32, // backing-store bytes, page aligned
    msg_size: NvU32, // entry size, power-of-2, >= 16
    msg_count: NvU32, // number of entry slots = (size - entryOff) / msgSize
    write_ptr: NvU32, // message id of next slot (advances on submit)
    flags: NvU32, // MSGQ_FLAGS_*
    rx_hdr_off: NvU32, // offset of the msgqRxHeader from the backing-store start
    entry_off: NvU32, // offset of entry slots from the backing-store start

    pub const VERSION: NvU32 = 0; // MSGQ_VERSION
};

/// The consumer-side ring header (lives at rxHdrOff within the producer's
/// backing store). The reader owns readPtr; the writer reads it to compute free
/// space. 4 bytes.
pub const MsgqRxHeader = extern struct {
    read_ptr: NvU32, // message id of last message read (advances on consume)
};

// Backing-store layout constants (msgq.c msgqTxCreate):
//   rxHdrOff = ALIGN_UP(sizeof(msgqTxHeader)=32, 1<<hdrAlign)
//   entryOff = ALIGN_UP(rxHdrOff + sizeof(msgqRxHeader)=4, 1<<entryAlign)
// For the GSP RM queue (message_queue_cpu.c): queueHeaderAlign = 4 (1<<4 == 16),
// queueElementAlign = RM_PAGE_SHIFT (12 -> 4096), queueElementSizeMin = RM_PAGE_SIZE.

/// queueHeaderAlign = 4 -> header alignment of 1<<4 == 16 bytes.
pub const QUEUE_HEADER_ALIGN_SHIFT: u5 = 4;
/// queueElementAlign = RM_PAGE_SHIFT == 12 -> entries aligned to 1<<12 == 4096.
pub const QUEUE_ELEMENT_ALIGN_SHIFT: u5 = 12;
/// queueElementSizeMin = RM_PAGE_SIZE == 4096 (the per-slot byte size).
pub const ELEMENT_SIZE_MIN: u32 = 1 << QUEUE_ELEMENT_ALIGN_SHIFT;
/// queueElementSizeMax = RM_PAGE_SIZE * 16.
pub const ELEMENT_SIZE_MAX: u32 = ELEMENT_SIZE_MIN * 16;

// ---------------------------------------------------------------------------
// GspFwWprMeta (gsp_fw_wpr_meta.h) - the Booter <-> BL handoff descriptor.
// ---------------------------------------------------------------------------

/// The 256-byte "cold" descriptor the GSP boot consumes: it points the Booter at
/// the radix3 ELF, the bootloader, the per-chip signature, and lays out WPR2 /
/// FRTS / the heaps in VRAM. Filled by fw.zig from the WPR2 layout + radix3 root.
/// Field-for-field from gsp_fw_wpr_meta.h at 595.71.05. The two unions are
/// modelled with their initial-boot arm inlined (the resume arm + the crashcat
/// arm overlay the SAME bytes - we keep names + an explicit padding-equivalence
/// assert below). 256 bytes exactly (the struct is self-padded by the C).
pub const GspFwWprMeta = extern struct {
    /// = GSP_FW_WPR_META_MAGIC. The Booter verifies this.
    magic: NvU64 align(8),
    /// = GSP_FW_WPR_META_REVISION (1). Booter<->BL interface revision.
    revision: NvU64,

    // ---- data in SYSMEM (consumed by Booter for DMA) ----
    sysmem_addr_of_radix3_elf: NvU64,
    size_of_radix3_elf: NvU64,
    sysmem_addr_of_bootloader: NvU64,
    size_of_bootloader: NvU64,
    bootloader_code_offset: NvU64,
    bootloader_data_offset: NvU64,
    bootloader_manifest_offset: NvU64,

    // union { initial-boot { sysmemAddrOfSignature; sizeOfSignature }
    //         resume       { gspFwHeapFreeListWprOffset(u32); unused0(u32); unused1(u64) } }
    // Both arms are 16 bytes. We model the initial-boot arm (what we fill at
    // cold boot); the resume arm is the same bytes.
    sysmem_addr_of_signature: NvU64,
    size_of_signature: NvU64,

    // ---- FB layout ----
    gsp_fw_rsvd_start: NvU64,
    non_wpr_heap_offset: NvU64,
    non_wpr_heap_size: NvU64,
    gsp_fw_wpr_start: NvU64,
    gsp_fw_heap_offset: NvU64,
    gsp_fw_heap_size: NvU64,
    gsp_fw_offset: NvU64,
    boot_bin_offset: NvU64,
    frts_offset: NvU64,
    frts_size: NvU64,
    gsp_fw_wpr_end: NvU64,
    fb_size: NvU64,

    // ---- other ----
    vga_workspace_offset: NvU64,
    vga_workspace_size: NvU64,
    boot_count: NvU64,

    // union { partitionRpc {partitionRpcAddr(u64); partitionRpcRequestOffset(u16);
    //         partitionRpcReplyOffset(u16); elfCodeOffset(u32); elfDataOffset(u32);
    //         elfCodeSize(u32); elfDataSize(u32); lsUcodeVersion(u32) }
    //         crashcat-overlay { partitionRpcPadding[4](u32);
    //         sysmemAddrOfCrashReportQueue(u64); sizeOfCrashReportQueue(u32);
    //         lsUcodeVersionPadding[1](u32) } } - both arms 32 bytes.
    // We model the partitionRpc arm (zeroed at cold boot).
    partition_rpc_addr: NvU64,
    partition_rpc_request_offset: u16,
    partition_rpc_reply_offset: u16,
    elf_code_offset: NvU32,
    elf_data_offset: NvU32,
    elf_code_size: NvU32,
    elf_data_size: NvU32,
    ls_ucode_version: NvU32,

    gsp_fw_heap_vf_partition_count: NvU8,
    flags: NvU8,
    padding: [2]NvU8,
    pmu_reserved_size: NvU32,

    /// 0 -> unverified; the Booter writes GSP_FW_WPR_META_VERIFIED on success.
    verified: NvU64,

    pub const MAGIC: NvU64 = 0xdc3aae21371a60b3;
    pub const REVISION: NvU64 = 1;
    pub const VERIFIED: NvU64 = 0xa0a0a0a0a0a0a0a0;
};

// ---------------------------------------------------------------------------
// GspFwSRMeta (gsp_fw_sr_meta.h) - suspend/resume descriptor (256 bytes).
// ---------------------------------------------------------------------------

/// 256-byte suspend/resume metadata. Not on the cold-boot critical path (cold
/// boot uses GspFwWprMeta), but ported here for completeness + layout assert so
/// the SR path (a later phase) has it. Field-for-field from gsp_fw_sr_meta.h.
pub const GspFwSrMeta = extern struct {
    magic: NvU64 align(8),
    revision: NvU64,
    sysmem_addr_of_suspend_resume_data: NvU64,
    size_of_suspend_resume_data: NvU64,
    internal: [32]NvU32,
    flags: NvU32,
    subrevision: NvU32,
    padding: [22]NvU32,

    pub const MAGIC: NvU64 = 0x8a3bb9e6c6c39d93;
    pub const REVISION: NvU64 = 2;
    /// GSP_FW_SR_META_INTERNAL_SIZE: the `internal` array is exactly 128 bytes.
    pub const INTERNAL_SIZE: usize = 128;
};

// ---------------------------------------------------------------------------
// MESSAGE_QUEUE_INIT_ARGUMENTS / GSP_SR_INIT_ARGUMENTS / GSP_ARGUMENTS_CACHED
// (gsp_init_args.h). NvLength == NvU64 on the 64-bit GSP. NvBool == NvU8.
// ---------------------------------------------------------------------------

/// MESSAGE_QUEUE_INIT_ARGUMENTS: tells the GSP where the shared cmd/stat queues
/// live + their geometry. NOTE this carries MORE than the plan listed - the
/// 595.71.05 header also has the queueElement* + queue*Align fields (matched
/// here). All NvLength fields are NvU64.
pub const MessageQueueInitArguments = extern struct {
    shared_mem_phys_addr: NvU64 align(8),
    page_table_entry_count: NvU32,
    cmd_queue_offset: NvU64 align(8),
    stat_queue_offset: NvU64,
    queue_element_hdr_size: NvU64,
    queue_element_size_min: NvU64,
    queue_element_size_max: NvU64,
    queue_header_align: NvU32,
    queue_element_align: NvU32,
};

/// GSP_SR_INIT_ARGUMENTS (suspend/resume init args, zeroed on cold boot).
pub const GspSrInitArguments = extern struct {
    old_level: NvU32,
    flags: NvU32,
    b_in_pm_transition: NvU8, // NvBool
};

/// GSP_ARGUMENTS_CACHED: the top-level boot args blob the GSP reads. profilerArgs
/// / sysmemHeapArgs / rmStateMonitorBufferArgs are {pa,size} pairs (zeroed at
/// cold boot). bDmemStack is NvBool (NvU8).
pub const GspArgumentsCached = extern struct {
    message_queue_init_arguments: MessageQueueInitArguments align(8),
    sr_init_arguments: GspSrInitArguments,
    gpu_instance: NvU32,
    b_dmem_stack: NvU8, // NvBool

    profiler_args: PaSize align(8),
    sysmem_heap_args: PaSize,
    rm_state_monitor_buffer_args: PaSize,

    pub const PaSize = extern struct {
        pa: NvU64 align(8),
        size: NvU64,
    };
};

// ---------------------------------------------------------------------------
// LibosMemoryRegionInitArgument (libos_init_args.h)
// ---------------------------------------------------------------------------

/// LibosMemoryRegionKind. RADIX3 == 2 (the kind used for the radix3 ELF region).
pub const LibosMemoryRegionKind = enum(NvU8) {
    none = 0,
    contiguous = 1,
    radix3 = 2,
};

/// LibosMemoryRegionLoc. SYSMEM == 1 (the boot regions live in sysmem the GPU DMAs).
pub const LibosMemoryRegionLoc = enum(NvU8) {
    none = 0,
    sysmem = 1,
    fb = 2,
};

/// One libos memory region descriptor (id8/pa/size 64-bit, kind/loc u8). The 4
/// boot regions (LOGINIT/LOGINTR/LOGRM/RMARGS) are built from these.
pub const LibosMemoryRegionInitArgument = extern struct {
    id8: NvU64 align(8), // LibosAddress (== NvU64), an 8-char id tag packed little-endian
    pa: NvU64,
    size: NvU64,
    kind: u8, // LibosMemoryRegionKind
    loc: u8, // LibosMemoryRegionLoc

    /// Pack an up-to-8-char ASCII tag into the id8 field (little-endian, the way
    /// the C does `*(NvU64*)"LOGINIT"`).
    pub fn idFromTag(tag: []const u8) NvU64 {
        var v: NvU64 = 0;
        var i: usize = 0;
        while (i < tag.len and i < 8) : (i += 1) {
            v |= @as(NvU64, tag[i]) << @intCast(i * 8);
        }
        return v;
    }
};

comptime {
    // rpc_message_header_v == 32 bytes, all NvU32 in order.
    std.debug.assert(@sizeOf(RpcMessageHeader) == 32);
    std.debug.assert(@offsetOf(RpcMessageHeader, "header_version") == 0);
    std.debug.assert(@offsetOf(RpcMessageHeader, "signature") == 4);
    std.debug.assert(@offsetOf(RpcMessageHeader, "length") == 8);
    std.debug.assert(@offsetOf(RpcMessageHeader, "function") == 12);
    std.debug.assert(@offsetOf(RpcMessageHeader, "rpc_result") == 16);
    std.debug.assert(@offsetOf(RpcMessageHeader, "rpc_result_private") == 20);
    std.debug.assert(@offsetOf(RpcMessageHeader, "sequence") == 24);
    std.debug.assert(@offsetOf(RpcMessageHeader, "u") == 28);

    // rpc_gsp_rm_alloc_v header == 32 bytes.
    std.debug.assert(@sizeOf(RpcGspRmAlloc) == 32);
    std.debug.assert(@offsetOf(RpcGspRmAlloc, "h_client") == 0);
    std.debug.assert(@offsetOf(RpcGspRmAlloc, "h_parent") == 4);
    std.debug.assert(@offsetOf(RpcGspRmAlloc, "h_object") == 8);
    std.debug.assert(@offsetOf(RpcGspRmAlloc, "h_class") == 12);
    std.debug.assert(@offsetOf(RpcGspRmAlloc, "status") == 16);
    std.debug.assert(@offsetOf(RpcGspRmAlloc, "params_size") == 20);
    std.debug.assert(@offsetOf(RpcGspRmAlloc, "flags") == 24);
    std.debug.assert(@offsetOf(RpcGspRmAlloc, "reserved") == 28);

    // rpc_gsp_rm_control_v header == 40 bytes (595.71.05 expanded layout).
    std.debug.assert(@sizeOf(RpcGspRmControl) == 40);
    std.debug.assert(@offsetOf(RpcGspRmControl, "h_client") == 0);
    std.debug.assert(@offsetOf(RpcGspRmControl, "h_object") == 4);
    std.debug.assert(@offsetOf(RpcGspRmControl, "cmd") == 8);
    std.debug.assert(@offsetOf(RpcGspRmControl, "status") == 12);
    std.debug.assert(@offsetOf(RpcGspRmControl, "params_size") == 16);
    std.debug.assert(@offsetOf(RpcGspRmControl, "rmapi_rpc_flags") == 20);
    std.debug.assert(@offsetOf(RpcGspRmControl, "rmctrl_flags") == 24);
    std.debug.assert(@offsetOf(RpcGspRmControl, "rmctrl_access_right") == 28);
    std.debug.assert(@offsetOf(RpcGspRmControl, "reserved0") == 32);

    // GSP_MSG_QUEUE_ELEMENT header == 48 bytes (44 declared + 4 align pad).
    std.debug.assert(@sizeOf(GspMsgQueueElement) == 48);
    std.debug.assert(GspMsgQueueElement.HDR_SIZE == 48);
    std.debug.assert(@offsetOf(GspMsgQueueElement, "auth_tag_buffer") == 0);
    std.debug.assert(@offsetOf(GspMsgQueueElement, "aad_buffer") == 16);
    std.debug.assert(@offsetOf(GspMsgQueueElement, "check_sum") == 32);
    std.debug.assert(@offsetOf(GspMsgQueueElement, "seq_num") == 36);
    std.debug.assert(@offsetOf(GspMsgQueueElement, "elem_count") == 40);

    // msgqTxHeader == 32 bytes, msgqRxHeader == 4 bytes.
    std.debug.assert(@sizeOf(MsgqTxHeader) == 32);
    std.debug.assert(@offsetOf(MsgqTxHeader, "version") == 0);
    std.debug.assert(@offsetOf(MsgqTxHeader, "size") == 4);
    std.debug.assert(@offsetOf(MsgqTxHeader, "msg_size") == 8);
    std.debug.assert(@offsetOf(MsgqTxHeader, "msg_count") == 12);
    std.debug.assert(@offsetOf(MsgqTxHeader, "write_ptr") == 16);
    std.debug.assert(@offsetOf(MsgqTxHeader, "flags") == 20);
    std.debug.assert(@offsetOf(MsgqTxHeader, "rx_hdr_off") == 24);
    std.debug.assert(@offsetOf(MsgqTxHeader, "entry_off") == 28);
    std.debug.assert(@sizeOf(MsgqRxHeader) == 4);

    // GspFwWprMeta == 256 bytes. Spot-check the leading + key offsets vs the C.
    std.debug.assert(@sizeOf(GspFwWprMeta) == 256);
    std.debug.assert(@offsetOf(GspFwWprMeta, "magic") == 0);
    std.debug.assert(@offsetOf(GspFwWprMeta, "revision") == 8);
    std.debug.assert(@offsetOf(GspFwWprMeta, "sysmem_addr_of_radix3_elf") == 16);
    std.debug.assert(@offsetOf(GspFwWprMeta, "size_of_radix3_elf") == 24);
    std.debug.assert(@offsetOf(GspFwWprMeta, "sysmem_addr_of_bootloader") == 32);
    std.debug.assert(@offsetOf(GspFwWprMeta, "bootloader_code_offset") == 48);
    std.debug.assert(@offsetOf(GspFwWprMeta, "bootloader_manifest_offset") == 64);
    std.debug.assert(@offsetOf(GspFwWprMeta, "sysmem_addr_of_signature") == 72);
    std.debug.assert(@offsetOf(GspFwWprMeta, "size_of_signature") == 80);
    std.debug.assert(@offsetOf(GspFwWprMeta, "gsp_fw_rsvd_start") == 88);
    std.debug.assert(@offsetOf(GspFwWprMeta, "non_wpr_heap_offset") == 96);
    std.debug.assert(@offsetOf(GspFwWprMeta, "gsp_fw_wpr_start") == 112);
    std.debug.assert(@offsetOf(GspFwWprMeta, "gsp_fw_heap_offset") == 120);
    std.debug.assert(@offsetOf(GspFwWprMeta, "gsp_fw_offset") == 136);
    std.debug.assert(@offsetOf(GspFwWprMeta, "boot_bin_offset") == 144);
    std.debug.assert(@offsetOf(GspFwWprMeta, "frts_offset") == 152);
    std.debug.assert(@offsetOf(GspFwWprMeta, "frts_size") == 160);
    std.debug.assert(@offsetOf(GspFwWprMeta, "gsp_fw_wpr_end") == 168);
    std.debug.assert(@offsetOf(GspFwWprMeta, "fb_size") == 176);
    std.debug.assert(@offsetOf(GspFwWprMeta, "vga_workspace_offset") == 184);
    std.debug.assert(@offsetOf(GspFwWprMeta, "boot_count") == 200);
    std.debug.assert(@offsetOf(GspFwWprMeta, "partition_rpc_addr") == 208);
    std.debug.assert(@offsetOf(GspFwWprMeta, "partition_rpc_request_offset") == 216);
    std.debug.assert(@offsetOf(GspFwWprMeta, "elf_code_offset") == 220);
    std.debug.assert(@offsetOf(GspFwWprMeta, "ls_ucode_version") == 236);
    std.debug.assert(@offsetOf(GspFwWprMeta, "gsp_fw_heap_vf_partition_count") == 240);
    std.debug.assert(@offsetOf(GspFwWprMeta, "flags") == 241);
    std.debug.assert(@offsetOf(GspFwWprMeta, "pmu_reserved_size") == 244);
    std.debug.assert(@offsetOf(GspFwWprMeta, "verified") == 248);

    // GspFwSrMeta == 256 bytes.
    std.debug.assert(@sizeOf(GspFwSrMeta) == 256);
    std.debug.assert(@offsetOf(GspFwSrMeta, "internal") == 32);
    std.debug.assert(@sizeOf(@FieldType(GspFwSrMeta, "internal")) == GspFwSrMeta.INTERNAL_SIZE);
    std.debug.assert(@offsetOf(GspFwSrMeta, "flags") == 160);
    std.debug.assert(@offsetOf(GspFwSrMeta, "subrevision") == 164);
    std.debug.assert(@offsetOf(GspFwSrMeta, "padding") == 168);

    // MESSAGE_QUEUE_INIT_ARGUMENTS (NvLength == NvU64). 56 bytes.
    std.debug.assert(@offsetOf(MessageQueueInitArguments, "shared_mem_phys_addr") == 0);
    std.debug.assert(@offsetOf(MessageQueueInitArguments, "page_table_entry_count") == 8);
    std.debug.assert(@offsetOf(MessageQueueInitArguments, "cmd_queue_offset") == 16);
    std.debug.assert(@offsetOf(MessageQueueInitArguments, "stat_queue_offset") == 24);
    std.debug.assert(@offsetOf(MessageQueueInitArguments, "queue_element_hdr_size") == 32);
    std.debug.assert(@offsetOf(MessageQueueInitArguments, "queue_element_size_min") == 40);
    std.debug.assert(@offsetOf(MessageQueueInitArguments, "queue_element_size_max") == 48);
    std.debug.assert(@offsetOf(MessageQueueInitArguments, "queue_header_align") == 56);
    std.debug.assert(@offsetOf(MessageQueueInitArguments, "queue_element_align") == 60);
    std.debug.assert(@sizeOf(MessageQueueInitArguments) == 64);

    // GSP_SR_INIT_ARGUMENTS.
    std.debug.assert(@offsetOf(GspSrInitArguments, "old_level") == 0);
    std.debug.assert(@offsetOf(GspSrInitArguments, "flags") == 4);
    std.debug.assert(@offsetOf(GspSrInitArguments, "b_in_pm_transition") == 8);

    // GSP_ARGUMENTS_CACHED.
    std.debug.assert(@offsetOf(GspArgumentsCached, "message_queue_init_arguments") == 0);
    std.debug.assert(@offsetOf(GspArgumentsCached, "sr_init_arguments") == 64);
    std.debug.assert(@offsetOf(GspArgumentsCached, "gpu_instance") == 76);
    std.debug.assert(@offsetOf(GspArgumentsCached, "b_dmem_stack") == 80);
    std.debug.assert(@offsetOf(GspArgumentsCached, "profiler_args") == 88);
    std.debug.assert(@offsetOf(GspArgumentsCached, "sysmem_heap_args") == 104);
    std.debug.assert(@offsetOf(GspArgumentsCached, "rm_state_monitor_buffer_args") == 120);

    // LibosMemoryRegionInitArgument. id8/pa/size + kind/loc u8 -> 32 bytes
    // (8-byte aligned struct, 6 bytes tail pad after the two u8).
    std.debug.assert(@offsetOf(LibosMemoryRegionInitArgument, "id8") == 0);
    std.debug.assert(@offsetOf(LibosMemoryRegionInitArgument, "pa") == 8);
    std.debug.assert(@offsetOf(LibosMemoryRegionInitArgument, "size") == 16);
    std.debug.assert(@offsetOf(LibosMemoryRegionInitArgument, "kind") == 24);
    std.debug.assert(@offsetOf(LibosMemoryRegionInitArgument, "loc") == 25);
    std.debug.assert(@sizeOf(LibosMemoryRegionInitArgument) == 32);
    std.debug.assert(@intFromEnum(LibosMemoryRegionKind.radix3) == 2);
    std.debug.assert(@intFromEnum(LibosMemoryRegionLoc.sysmem) == 1);
}

test "GSP transport struct layouts match the NVIDIA 595.71.05 ABI" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(RpcMessageHeader));
    try std.testing.expectEqual(@as(NvU32, 0x03000000), RpcMessageHeader.HEADER_VERSION);
    try std.testing.expectEqual(@as(NvU32, 0x43505256), RpcMessageHeader.SIGNATURE);

    try std.testing.expectEqual(@as(usize, 32), @sizeOf(RpcGspRmAlloc));
    try std.testing.expectEqual(@as(usize, 28), @offsetOf(RpcGspRmAlloc, "reserved"));

    try std.testing.expectEqual(@as(usize, 40), @sizeOf(RpcGspRmControl));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(RpcGspRmControl, "reserved0"));

    try std.testing.expectEqual(@as(usize, 48), @sizeOf(GspMsgQueueElement));
    try std.testing.expectEqual(@as(usize, 48), GspMsgQueueElement.HDR_SIZE);
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(GspMsgQueueElement, "check_sum"));
    try std.testing.expectEqual(@as(usize, 36), @offsetOf(GspMsgQueueElement, "seq_num"));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(GspMsgQueueElement, "elem_count"));

    try std.testing.expectEqual(@as(usize, 32), @sizeOf(MsgqTxHeader));
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(MsgqRxHeader));

    // The enum function numbers carried on the wire.
    try std.testing.expectEqual(@as(NvU32, 2), @intFromEnum(Function.alloc_root));
    try std.testing.expectEqual(@as(NvU32, 10), @intFromEnum(Function.free));
    try std.testing.expectEqual(@as(NvU32, 76), @intFromEnum(Function.gsp_rm_control));
    try std.testing.expectEqual(@as(NvU32, 103), @intFromEnum(Function.gsp_rm_alloc));
    try std.testing.expectEqual(@as(NvU32, 0x1001), @intFromEnum(Event.gsp_init_done));
    try std.testing.expectEqual(@as(NvU32, 0x1004), @intFromEnum(Event.rc_triggered));
}
