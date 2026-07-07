//! NVIDIA RM ioctl escape numbers + encoding, ported from NVIDIA's open kernel
//! modules: kernel-open/common/inc/nv-ioctl-numbers.h.
//!
//! The /dev/nvidia* control ioctls use Linux _IOWR(magic='F', nr, struct) for
//! small parameter blocks; oversized blocks go through NV_ESC_IOCTL_XFER_CMD
//! (not needed for the basic handshake).

const std = @import("std");

pub const NV_IOCTL_MAGIC: u8 = 'F'; // 0x46
pub const NV_IOCTL_BASE: u32 = 200;

pub const NV_ESC_CARD_INFO: u32 = NV_IOCTL_BASE + 0;
pub const NV_ESC_REGISTER_FD: u32 = NV_IOCTL_BASE + 1;
pub const NV_ESC_ALLOC_OS_EVENT: u32 = NV_IOCTL_BASE + 6;
pub const NV_ESC_FREE_OS_EVENT: u32 = NV_IOCTL_BASE + 7;
pub const NV_ESC_STATUS_CODE: u32 = NV_IOCTL_BASE + 9;
pub const NV_ESC_CHECK_VERSION_STR: u32 = NV_IOCTL_BASE + 10;
pub const NV_ESC_IOCTL_XFER_CMD: u32 = NV_IOCTL_BASE + 11;
pub const NV_ESC_ATTACH_GPUS_TO_FD: u32 = NV_IOCTL_BASE + 12;
pub const NV_ESC_SYS_PARAMS: u32 = NV_IOCTL_BASE + 14;

// RM escape numbers use a separate, raw numbering (nv_escape.h), NOT BASE+n.
pub const NV_ESC_RM_ALLOC_MEMORY: u32 = 0x27;
pub const NV_ESC_RM_ALLOC_OBJECT: u32 = 0x28;
pub const NV_ESC_RM_FREE: u32 = 0x29;
pub const NV_ESC_RM_CONTROL: u32 = 0x2A;
pub const NV_ESC_RM_ALLOC: u32 = 0x2B;
pub const NV_ESC_RM_DUP_OBJECT: u32 = 0x34;
pub const NV_ESC_RM_VID_HEAP_CONTROL: u32 = 0x4A;
pub const NV_ESC_RM_MAP_MEMORY: u32 = 0x4E;
pub const NV_ESC_RM_MAP_MEMORY_DMA: u32 = 0x57;
pub const NV_ESC_RM_UNMAP_MEMORY_DMA: u32 = 0x58;

/// Linux _IOWR(NV_IOCTL_MAGIC, nr, T) - the request value for a direct RM ioctl.
pub fn iowr(nr: u32, comptime T: type) u32 {
    return std.os.linux.IOCTL.IOWR(NV_IOCTL_MAGIC, @intCast(nr), T);
}

test "CHECK_VERSION_STR encodes to the kernel-confirmed value" {
    const sdk = @import("sdk.zig");
    // Verified live against the 595.71.05 open kernel module: 0xc04846d2.
    try std.testing.expectEqual(@as(u32, 0xc04846d2), iowr(NV_ESC_CHECK_VERSION_STR, sdk.RmApiVersion));
}
