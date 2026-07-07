//! GSP (GPU System Processor) transport for the baremetal NVIDIA bring-up. This
//! aggregates the version-pinned (595.71.05) protocol structs and the message-
//! queue ring transport that the freestanding RM-over-GSP path (phases P3/P4/P6)
//! builds on. Pure logic, no OS dependency - it compiles on the UEFI/freestanding
//! target as well as Linux.
//!
//! - `proto`: the on-wire structs (rpc_message_header_v, rpc_gsp_rm_alloc_v,
//!   rpc_gsp_rm_control_v, GSP_MSG_QUEUE_ELEMENT, msgqTx/RxHeader) + the
//!   NV_VGPU_MSG_FUNCTION_* / _EVENT_* enums, each layout-asserted vs the C.
//! - `ring`: the cmdq/msgq ring (index math, element framing, XOR-fold checksum,
//!   seqNum/elemCount) + the RPC marshalling helpers (buildAllocElement /
//!   buildControlElement) wrapping the existing sdk.zig payloads.

const std = @import("std");

pub const proto = @import("proto.zig");
pub const ring = @import("ring.zig");
pub const fw = @import("fw.zig");
pub const falcon = @import("falcon.zig");
pub const bios = @import("bios.zig");
pub const boot = @import("boot.zig");

// Re-export the most-used surface so callers can write gsp.Endpoint etc.
pub const RpcMessageHeader = proto.RpcMessageHeader;
pub const Function = proto.Function;
pub const Event = proto.Event;
pub const GspMsgQueueElement = proto.GspMsgQueueElement;
pub const Ring = ring.Ring;
pub const Endpoint = ring.Endpoint;
pub const GspFwWprMeta = proto.GspFwWprMeta;
pub const GspArgumentsCached = proto.GspArgumentsCached;
pub const Wpr2Layout = fw.Wpr2Layout;
pub const Radix3 = fw.Radix3;
pub const BumpAllocator = fw.BumpAllocator;
pub const Falcon = falcon.Falcon;
pub const Bit = bios.Bit;
pub const PmuTable = bios.PmuTable;
pub const Fwsec = bios.Fwsec;
pub const findFwsec = bios.findFwsec;
pub const extractAndPatchFrts = bios.extractAndPatchFrts;
pub const buildAllocElement = ring.buildAllocElement;
pub const buildControlElement = ring.buildControlElement;
pub const Boot = boot.Boot;
pub const Booter = boot.Booter;
pub const parseBooterHeaders = boot.parseBooterHeaders;
pub const patchBooter = boot.patchBooter;
pub const buildFmcBootParams = boot.buildFmcBootParams;
pub const splitPhys = boot.splitPhys;
pub const verifyWpr2 = boot.verifyWpr2;
pub const GspFmcBootParams = boot.GspFmcBootParams;

test {
    std.testing.refAllDecls(@This());
}
