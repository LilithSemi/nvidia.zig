//! NVIDIA RM SDK types, ported from NVIDIA's open kernel modules
//! (kernel-open/common/inc/nv-ioctl.h and src/common/sdk/nvidia/inc/nvos.h +
//! class headers). Layouts must match the kernel ABI exactly (extern struct).

const std = @import("std");

// Base RM scalar types (nvtypes.h). NvP64 is the 64-bit pointer variant.
pub const NvU8 = u8;
pub const NvU16 = u16;
pub const NvU32 = u32;
pub const NvV32 = u32;
pub const NvU64 = u64;
pub const NvBool = u8;
pub const NvHandle = u32;
pub const NvP64 = u64;

pub const NV_MAX_DEVICES = 32; // nvlimits.h

/// nv_pci_info_t (nv-ioctl.h): PCI location of a GPU.
pub const PciInfo = extern struct {
    domain: NvU32,
    bus: NvU8,
    slot: NvU8,
    function: NvU8,
    vendor_id: NvU16,
    device_id: NvU16,
};

/// nv_ioctl_card_info_t (nv-ioctl.h): one NV_ESC_CARD_INFO entry per GPU.
pub const CardInfo = extern struct {
    valid: NvBool,
    pci_info: PciInfo,
    gpu_id: NvU32,
    interrupt_line: NvU16,
    reg_address: NvU64 align(8),
    reg_size: NvU64 align(8),
    fb_address: NvU64 align(8),
    fb_size: NvU64 align(8),
    minor_number: NvU32,
    dev_name: [10]u8,
};

/// nv_ioctl_register_fd_t (nv-ioctl.h): NV_ESC_REGISTER_FD links a per-GPU node
/// fd to the control fd (required before a device can be allocated).
pub const RegisterFd = extern struct {
    ctl_fd: i32,
};

/// nv_ioctl_rm_api_version_t (nv-ioctl.h): the NV_ESC_CHECK_VERSION_STR block.
pub const RmApiVersion = extern struct {
    cmd: NvU32,
    reply: NvU32,
    version_string: [STRING_LENGTH]u8,

    pub const STRING_LENGTH = 64; // NV_RM_API_VERSION_STRING_LENGTH

    // cmd values.
    pub const CMD_STRICT: NvU32 = 0; // exact match required
    pub const CMD_RELAXED: NvU32 = '1'; // major-version match
    pub const CMD_QUERY: NvU32 = '2'; // kernel fills in its own version

    // reply values.
    pub const REPLY_UNRECOGNIZED: NvU32 = 0;
    pub const REPLY_RECOGNIZED: NvU32 = 1;
};

/// NVOS00_PARAMETERS (nvos.h): the NV_ESC_RM_FREE parameter block.
pub const Os00Params = extern struct {
    h_root: NvHandle,
    h_object_parent: NvHandle,
    h_object_old: NvHandle,
    status: NvV32,
};

/// NVOS21_PARAMETERS (nvos.h): the NV_ESC_RM_ALLOC (NV04_ALLOC) parameter block.
pub const Os21Params = extern struct {
    h_root: NvHandle,
    h_object_parent: NvHandle,
    h_object_new: NvHandle,
    h_class: NvV32,
    p_alloc_parms: NvP64 align(8),
    params_size: NvU32,
    status: NvV32,
};

/// NV0000_ALLOC_PARAMETERS (cl0000.h): params for allocating the root client.
pub const Nv0000AllocParameters = extern struct {
    h_client: NvHandle,
    process_id: NvU32,
    process_name: [PROC_NAME_MAX_LENGTH]u8,
    p_os_pid_info: NvP64 align(8),

    pub const PROC_NAME_MAX_LENGTH = 100; // NV_PROC_NAME_MAX_LENGTH
};

/// NV0080_ALLOC_PARAMETERS (cl0080.h): params for allocating a device.
pub const Nv0080AllocParameters = extern struct {
    device_id: NvU32,
    h_client_share: NvHandle,
    h_target_client: NvHandle,
    h_target_device: NvHandle,
    flags: NvV32,
    va_space_size: NvU64 align(8),
    va_start_internal: NvU64 align(8),
    va_limit_internal: NvU64 align(8),
    va_mode: NvV32,
};

/// NV2080_ALLOC_PARAMETERS (cl2080.h): params for allocating a subdevice.
pub const Nv2080AllocParameters = extern struct {
    sub_device_id: NvU32,
};

/// NV_MEMORY_ALLOCATION_PARAMS (nvos.h): pAllocParms for RM_ALLOC of a memory
/// class (NV01_MEMORY_SYSTEM / NV01_MEMORY_LOCAL_USER). The legacy NVOS02
/// RM_ALLOC_MEMORY path returns EINVAL on the open kernel modules; this is the
/// supported path. `size` and `attr`/`attr2` are in/out (kernel fills actuals).
pub const MemAllocParams = extern struct {
    owner: NvU32 = 0, // memory owner tag (for tracking/debug)
    surface_type: NvU32 = 0, // NVOS32_TYPE_IMAGE = 0
    flags: NvU32 = 0,
    width: NvU32 = 0,
    height: NvU32 = 0,
    pitch: i32 = 0,
    attr: NvU32 = 0, // NVOS32_ATTR_* (location/physicality/coherency/format)
    attr2: NvU32 = 0,
    format: NvU32 = 0,
    compr_covg: NvU32 = 0,
    zcull_covg: NvU32 = 0,
    range_lo: NvU64 align(8) = 0,
    range_hi: NvU64 align(8) = 0,
    size: NvU64 align(8) = 0,
    alignment: NvU64 align(8) = 0,
    offset: NvU64 align(8) = 0,
    limit: NvU64 align(8) = 0,
    address: NvP64 align(8) = 0,
    ctag_offset: NvU32 = 0,
    h_va_space: NvHandle = 0,
    internal_flags: NvU32 = 0,
    tag: NvU32 = 0,
    numa_node: i32 = 0,
};

/// NV_OS_DESC_MEMORY_ALLOCATION_PARAMS (nvos.h): pAllocParms for
/// NV01_MEMORY_SYSTEM_OS_DESCRIPTOR. Imports an existing OS memory range
/// (here a userspace VA) as an RM memory object the GPU can map - the basis for
/// zero-copy present (render straight into a Wayland shm buffer's pages).
pub const OsDescMemAllocParams = extern struct {
    surface_type: NvU32 = 0, // NVOS32_TYPE_IMAGE
    flags: NvU32 = 0,
    attr: NvU32 = 0, // NVOS32_ATTR_* (location/physicality/coherency)
    attr2: NvU32 = 0,
    descriptor: NvP64 align(8) = 0, // the userspace VA to import
    limit: NvU64 align(8) = 0, // size - 1
    descriptor_type: NvU32 = 0, // NVOS32_DESCRIPTOR_TYPE_VIRTUAL_ADDRESS
};

/// NVOS32_DESCRIPTOR_TYPE_VIRTUAL_ADDRESS: the OS-descriptor is a userspace VA.
pub const NVOS32_DESCRIPTOR_TYPE_VIRTUAL_ADDRESS: NvU32 = 0;

/// NVOS33_PARAMETERS (nvos.h): the NV_ESC_RM_MAP_MEMORY parameter block.
pub const Os33Params = extern struct {
    h_client: NvHandle,
    h_device: NvHandle, // device or subdevice
    h_memory: NvHandle,
    offset: NvU64 align(8),
    length: NvU64 align(8),
    p_linear_address: NvP64 align(8), // out: mapping cookie
    status: NvU32,
    flags: NvU32,
};

/// nv_ioctl_nvos33_parameters_with_fd (nv-unix-nvos-params-wrappers.h): the
/// actual NV_ESC_RM_MAP_MEMORY ioctl payload. The open modules wrap NVOS33 with
/// the fd whose per-fd mmap context the kernel sets up (then you mmap that fd).
pub const Nvos33WithFd = extern struct {
    params: Os33Params,
    fd: i32,
};

/// NVOS54_PARAMETERS (nvos.h): the NV_ESC_RM_CONTROL parameter block. `cmd` is
/// an NVxxxx_CTRL_CMD_* code; `params` points at the command-specific struct.
pub const Os54Params = extern struct {
    h_client: NvHandle,
    h_object: NvHandle, // the object the control targets (e.g. subdevice)
    cmd: NvV32,
    flags: NvU32,
    params: NvP64 align(8),
    params_size: NvU32,
    status: NvV32,
};

/// NV_VASPACE_ALLOCATION_PARAMETERS (nvos.h): pAllocParms for FERMI_VASPACE_A.
/// All-zero requests a default GPU virtual address space.
pub const VaSpaceAllocParams = extern struct {
    index: NvU32 = 0,
    flags: NvV32 = 0,
    va_size: NvU64 align(8) = 0,
    va_start_internal: NvU64 align(8) = 0,
    va_limit_internal: NvU64 align(8) = 0,
    big_page_size: NvU32 = 0,
    va_base: NvU64 align(8) = 0,
    pasid: NvU32 = 0,
};

// Control command codes (NVxxxx_CTRL_CMD_*).
pub const NV2080_CTRL_CMD_GPU_GET_ID: NvU32 = 0x20800142;
pub const NV2080_CTRL_CMD_GPU_GET_NAME_STRING: NvU32 = 0x20800110;
pub const GPU_NAME_MAX_LENGTH = 64; // NV2080_GPU_MAX_NAME_STRING_LENGTH

/// NV2080_CTRL_GPU_GET_ID_PARAMS.
pub const GpuGetIdParams = extern struct {
    gpu_id: NvU32 = 0,
};

/// NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS (ASCII variant).
pub const GpuGetNameStringParams = extern struct {
    flags: NvU32 = 0, // 0 = ASCII
    ascii: [GPU_NAME_MAX_LENGTH]u8 = [_]u8{0} ** GPU_NAME_MAX_LENGTH,
};

/// NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS (cl0070.h): pAllocParms for
/// NV01_MEMORY_VIRTUAL - reserves a GPU VA range [offset, limit] in a VA space.
pub const VirtMemAllocParams = extern struct {
    offset: NvU64 align(8) = 0,
    limit: NvU64 align(8) = 0, // last valid VA (in/out)
    h_vaspace: NvHandle = 0,
};

/// NVOS46_PARAMETERS (nvos.h): the NV_ESC_RM_MAP_MEMORY_DMA parameter block -
/// binds a physical memory object into a VA space at a GPU virtual address.
pub const Os46Params = extern struct {
    h_client: NvHandle,
    h_device: NvHandle,
    h_dma: NvHandle, // an NV01_MEMORY_VIRTUAL object (carved from the VA space)
    h_memory: NvHandle, // the physical memory to map
    offset: NvU64 align(8),
    length: NvU64 align(8),
    flags: NvV32,
    flags2: NvV32,
    kind_override: NvV32,
    dma_offset: NvU64 align(8), // in (for virtual hDma): the GPU VA; out: actual
    status: NvV32,
};

/// NVOS46.flags bitfields (value << shift), from nvos.h NVOS46_FLAGS_*.
pub const dma_flags = struct {
    pub const ACCESS_READ_WRITE: NvU32 = 0 << 0;
    pub const ACCESS_READ_ONLY: NvU32 = 1 << 0;
    pub const PAGE_SIZE_DEFAULT: NvU32 = 0 << 8;
    pub const PAGE_SIZE_4KB: NvU32 = 1 << 8;
    pub const PAGE_SIZE_BIG: NvU32 = 2 << 8;
    pub const DMA_OFFSET_FIXED: NvU32 = 1 << 15; // dmaOffset is the requested VA
};

pub const NV01_MEMORY_VIRTUAL: NvV32 = 0x70;

// Channel-stack object classes (toward GPFIFO command submission).
pub const FERMI_VASPACE_A: NvV32 = 0x90f1; // GPU virtual address space
pub const KEPLER_CHANNEL_GROUP_A: NvV32 = 0xa06c; // TSG (timeslice group)
pub const AMPERE_CHANNEL_GPFIFO_A: NvV32 = 0xc56f;
pub const HOPPER_CHANNEL_GPFIFO_A: NvV32 = 0xc86f;
pub const BLACKWELL_CHANNEL_GPFIFO_A: NvV32 = 0xc96f;
pub const BLACKWELL_CHANNEL_GPFIFO_B: NvV32 = 0xca6f; // GB202 (RTX 50-series)

pub const NV_MAX_SUBDEVICES = 8; // nvlimits.h
pub const NV2080_ENGINE_TYPE_GRAPHICS: NvU32 = 0x1;
// The first asynchronous copy engine (CE0). NV2080_ENGINE_TYPE_COPY(i) =
// COPY0 + i; bind a GPFIFO channel to this to run BLACKWELL_DMA_COPY_B work.
pub const NV2080_ENGINE_TYPE_COPY0: NvU32 = 0x9;

// Blackwell DMA copy engine class (GB20x, the 0xCA family). Allocated under a
// channel bound to NV2080_ENGINE_TYPE_COPY0; its methods (LAUNCH_DMA etc.) drive
// a copy-engine 2D/block-linear<->pitch transfer entirely on the GPU.
pub const BLACKWELL_DMA_COPY_B: NvV32 = 0xcab5;

/// NV_MEMORY_DESC_PARAMS (alloc_channel.h).
pub const MemoryDescParams = extern struct {
    base: NvU64 align(8) = 0,
    size: NvU64 align(8) = 0,
    address_space: NvU32 = 0,
    cache_attrib: NvU32 = 0,
};

/// NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS (nvos.h): pAllocParms for
/// KEPLER_CHANNEL_GROUP_A (a TSG/timeslice group).
pub const TsgAllocParams = extern struct {
    h_object_error: NvHandle = 0,
    h_object_ecc_error: NvHandle = 0,
    h_vaspace: NvHandle = 0,
    engine_type: NvU32 = 0,
    b_vgpu_plugin: NvBool = 0,
};

/// NV_CHANNEL_ALLOC_PARAMS (alloc_channel.h) - the GPFIFO channel alloc params.
///
/// VERSION-SPECIFIC (see version.Abi): this is the 595.71.05 layout. The latest
/// open-gpu-kernel-modules inserts `hHandleVASpace` right after `hVASpace`;
/// 595.71 has no such field. Using the wrong layout shifts every later field
/// and the RM rejects the alloc with NV_ERR_INVALID_ARGUMENT.
pub const ChannelAllocParams = extern struct {
    h_object_error: NvHandle = 0,
    h_object_buffer: NvHandle = 0,
    gp_fifo_offset: NvU64 align(8) = 0, // GPU VA of the GPFIFO ring
    gp_fifo_entries: NvU32 = 0,
    flags: NvU32 = 0,
    h_context_share: NvHandle = 0,
    h_vaspace: NvHandle = 0,
    h_userd_memory: [NV_MAX_SUBDEVICES]NvHandle = [_]NvHandle{0} ** NV_MAX_SUBDEVICES,
    userd_offset: [NV_MAX_SUBDEVICES]NvU64 align(8) = [_]NvU64{0} ** NV_MAX_SUBDEVICES,
    engine_type: NvU32 = 0,
    cid: NvU32 = 0,
    sub_device_id: NvU32 = 0,
    h_object_ecc_error: NvHandle = 0,
    instance_mem: MemoryDescParams align(8) = .{},
    userd_mem: MemoryDescParams align(8) = .{},
    ramfc_mem: MemoryDescParams align(8) = .{},
    mthdbuf_mem: MemoryDescParams align(8) = .{},
    h_phys_channel_group: NvHandle = 0,
    internal_flags: NvU32 = 0,
    error_notifier_mem: MemoryDescParams align(8) = .{},
    ecc_error_notifier_mem: MemoryDescParams align(8) = .{},
    process_id: NvU32 = 0,
    sub_process_id: NvU32 = 0,
    encrypt_iv: [3]NvU32 = [_]NvU32{0} ** 3,
    decrypt_iv: [3]NvU32 = [_]NvU32{0} ** 3,
    hmac_nonce: [8]NvU32 = [_]NvU32{0} ** 8,
    tpc_config_id: NvU32 = 0,
};

// USERMODE doorbell classes (per GPU generation). The mapped aperture's
// NOTIFY_CHANNEL_PENDING register receives a channel's work-submit token.
pub const VOLTA_USERMODE_A: NvV32 = 0xc361;
pub const TURING_USERMODE_A: NvV32 = 0xc461;
pub const AMPERE_USERMODE_A: NvV32 = 0xc561;
pub const HOPPER_USERMODE_A: NvV32 = 0xc661;
pub const BLACKWELL_USERMODE_A: NvV32 = 0xc761;

// Channel control commands.
pub const NVA06F_CTRL_CMD_GPFIFO_SCHEDULE: NvU32 = 0xa06f0103;
pub const NVA06F_CTRL_CMD_BIND: NvU32 = 0xa06f0104;
pub const NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN: NvU32 = 0xc36f0108;

/// NVA06F_CTRL_BIND_PARAMS.
pub const ChannelBindParams = extern struct { engine_type: NvU32 = 0 };

/// NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS (three NvBool, 3 bytes).
pub const GpfifoScheduleParams = extern struct {
    b_enable: NvBool = 0,
    b_skip_submit: NvBool = 0,
    b_skip_enable: NvBool = 0,
};

/// NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS.
pub const WorkSubmitTokenParams = extern struct { token: NvU32 = 0 };

// Submission-stream encodings (NV906F GPFIFO + NVC86F host semaphore methods).
pub const gpfifo = struct {
    /// GP_ENTRY (8 bytes / 2 dwords) pointing at a pushbuffer of `len_dwords`
    /// methods at GPU VA `pb_va`.
    pub fn entry(pb_va: u64, len_dwords: u32) [2]NvU32 {
        return .{
            @truncate(pb_va & 0xFFFFFFFC), // GP_ENTRY0_GET (VA[31:2])
            @truncate(((pb_va >> 32) & 0xFF) | (@as(u64, len_dwords) << 10)), // GET_HI | LENGTH
        };
    }
    /// An "increasing method" header (NV906F SEC_OP=INC): `count` data dwords to
    /// method `addr` on `subch` follow.
    pub fn methodHeader(addr: u32, subch: u32, count: u32) NvU32 {
        return (1 << 29) | (count << 16) | (subch << 13) | (addr >> 2);
    }
    /// A "non-increasing method" header (SEC_OP=NON_INC): all `count` data dwords
    /// go to the same method `addr`. This is how a payload rides in the
    /// pushbuffer itself, as `LOAD_INLINE_DATA` does.
    pub fn methodHeaderNonInc(addr: u32, subch: u32, count: u32) NvU32 {
        return (3 << 29) | (count << 16) | (subch << 13) | (addr >> 2);
    }
    /// The largest `count` a single method header can carry (the field is 13 bits).
    pub const MAX_METHOD_COUNT: u32 = (1 << 13) - 1;
    // USERD byte offsets (NV_RAMUSERD, Volta+): GP_GET = word 34, GP_PUT = word 35.
    pub const USERD_GP_GET_OFFSET = 0x88;
    pub const USERD_GP_PUT_OFFSET = 0x8c;
    // Doorbell offset WITHIN the mapped USERMODE doorbell page. The page-relative
    // offset is 0x90 across generations even though the absolute VF register
    // moves (Volta 0x90, gb100 0x2200, gb202 0x30090 -> all 0x..090 in-page).
    pub const USERMODE_DOORBELL_OFFSET = 0x90;
    // NVC86F host semaphore methods.
    pub const SEM_ADDR_LO = 0x5c;
    pub const SEM_EXECUTE_RELEASE: NvU32 = 1;

    /// Build the 6-dword pushbuffer method stream that releases `payload` (32-bit)
    /// to the semaphore at GPU VA `sem_va` (NVC86F host SEM_* methods).
    pub fn semaphoreRelease(sem_va: u64, payload: NvU32) [6]NvU32 {
        return .{
            methodHeader(SEM_ADDR_LO, 0, 5),
            @truncate(sem_va & 0xFFFFFFFC), // SEM_ADDR_LO
            @truncate((sem_va >> 32) & 0x00FFFFFF), // SEM_ADDR_HI
            payload, // SEM_PAYLOAD_LO
            0, // SEM_PAYLOAD_HI
            SEM_EXECUTE_RELEASE, // SEM_EXECUTE
        };
    }
};

// Memory object classes.
pub const NV01_MEMORY_SYSTEM: NvV32 = 0x3e; // system (host) memory
pub const NV01_MEMORY_SYSTEM_OS_DESCRIPTOR: NvV32 = 0x71; // imported OS memory
pub const NV01_MEMORY_LOCAL_USER: NvV32 = 0x40; // device-local VRAM

/// NV_MEMORY_ALLOCATION_PARAMS.attr bitfields (value << shift), from nvos.h
/// NVOS32_ATTR_*. OR these together for the `attr` field.
pub const mem_attr = struct {
    pub const FORMAT_PITCH: NvU32 = 0 << 16; // linear
    // NVOS32_ATTR_FORMAT (bits 17:16): BLOCK_LINEAR is the GOB-tiled layout a depth
    // (ZETA) surface must use - the 3D engine's ROP-Z unit only writes block-linear
    // depth, and the GPU page-table kind must agree, so the depth surface memory is
    // allocated BLOCK_LINEAR (not the default PITCH). A PITCH-kind page under a
    // block-linear ZETA draw faults with an Xid 69 Class Error (ErrorCode 0x9c).
    pub const FORMAT_BLOCK_LINEAR: NvU32 = 2 << 16;
    // NVOS32_ATTR_DEPTH (bits 2:0): bit depth of a pixel. 32 = a 32-bit ZF32 depel.
    pub const DEPTH_32: NvU32 = 4 << 0;
    // NVOS32_ATTR_Z_TYPE (bit 18): FLOAT for ZF32 (vs FIXED for Z24).
    pub const Z_TYPE_FLOAT: NvU32 = 1 << 18;
    // NVOS32_ATTR_ZS_PACKING (bits 21:19): Z32 = a pure 32-bit Z (no stencil).
    pub const ZS_PACKING_Z32: NvU32 = 2 << 19;
    // NVOS32_ATTR_COMPR (bits 13:12): NONE - the depth surface is uncompressed (no
    // comptag lines are allocated; the 3D init keeps Z-compression off for it).
    pub const COMPR_NONE: NvU32 = 0 << 12;
    // NVOS32_ATTR_PAGE_SIZE (bits 24:23): BIG - a block-linear surface's page size
    // must be BIG (the block-linear PTE kind is only valid on big pages), so the
    // depth surface is allocated big-page and mapped big-page to match.
    pub const PAGE_SIZE_BIG: NvU32 = 2 << 23;
    pub const LOCATION_VIDMEM: NvU32 = 0 << 25;
    pub const LOCATION_PCI: NvU32 = 1 << 25;
    pub const PHYSICALITY_NONCONTIGUOUS: NvU32 = 1 << 27;
    pub const PHYSICALITY_CONTIGUOUS: NvU32 = 2 << 27;
    pub const COHERENCY_UNCACHED: NvU32 = 0 << 29;
    pub const COHERENCY_CACHED: NvU32 = 1 << 29;
    pub const COHERENCY_WRITE_COMBINE: NvU32 = 2 << 29;
    pub const COHERENCY_WRITE_BACK: NvU32 = 5 << 29;
};

// Object classes.
pub const NV01_NULL_OBJECT: NvHandle = 0x0;
pub const NV01_ROOT: NvV32 = 0x0; // cl0000.h
pub const NV01_ROOT_CLIENT: NvV32 = 0x41;
pub const NV01_DEVICE_0: NvV32 = 0x80; // cl0080.h
pub const NV20_SUBDEVICE_0: NvV32 = 0x2080; // cl2080.h

test "RM struct layouts match the NVIDIA ABI" {
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(RmApiVersion));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Os21Params));
    // p_alloc_parms must sit at offset 16 (8-aligned after four u32s).
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(Os21Params, "p_alloc_parms"));
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(Nv0080AllocParameters));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(Nv0080AllocParameters, "va_space_size"));
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(Nv2080AllocParameters));
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(PciInfo));
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(CardInfo));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(CardInfo, "gpu_id"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(CardInfo, "reg_address"));
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(MemAllocParams));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(MemAllocParams, "range_lo"));
    try std.testing.expectEqual(@as(usize, 64), @offsetOf(MemAllocParams, "size"));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(Os33Params));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(Os33Params, "p_linear_address"));
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(Nvos33WithFd));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(Nvos33WithFd, "fd"));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Os54Params));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(Os54Params, "params"));
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(VaSpaceAllocParams));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(VaSpaceAllocParams, "va_base"));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(VirtMemAllocParams));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(Os46Params));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(Os46Params, "dma_offset"));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(MemoryDescParams));
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(TsgAllocParams));
    // 595.71.05 channel params: 368 bytes (no hHandleVASpace).
    try std.testing.expectEqual(@as(usize, 368), @sizeOf(ChannelAllocParams));
    try std.testing.expectEqual(@as(usize, 64), @offsetOf(ChannelAllocParams, "userd_offset"));
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(ChannelBindParams));
    try std.testing.expectEqual(@as(usize, 3), @sizeOf(GpfifoScheduleParams));
}

test "GPFIFO submission encodings" {
    const e = gpfifo.entry(0x210000, 6);
    try std.testing.expectEqual(@as(NvU32, 0x210000), e[0]); // GP_ENTRY0 = VA[31:2]
    try std.testing.expectEqual(@as(NvU32, 6 << 10), e[1]); // LENGTH=6, GET_HI=0
    // increasing method to SEM_ADDR_LO (0x5c), 5 data dwords, subch 0.
    try std.testing.expectEqual(@as(NvU32, 0x20050017), gpfifo.methodHeader(0x5c, 0, 5));
}
