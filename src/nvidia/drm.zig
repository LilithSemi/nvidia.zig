//! nvidia-drm node helpers: open the nvidia-drm render node
//! (/dev/dri/renderD128), import a userspace CPU mapping as a GEM object via
//! DRM_IOCTL_NVIDIA_GEM_IMPORT_USERSPACE_MEMORY, export it as a real Linux
//! dma-buf fd via DRM_IOCTL_PRIME_HANDLE_TO_FD, and clean up with
//! DRM_IOCTL_GEM_CLOSE.
//!
//! Only works with system memory (allocMemory(.system) + mapMemory): the
//! kernel locks user pages with get_user_pages(), which requires a real CPU
//! VA backed by page-faulted pages. VRAM BAR mappings do not work here.
//!
//! The CPU mapping must stay alive (unmapMemory must NOT be called) until the
//! dma-buf consumer is finished with the buffer.

const std = @import("std");

const DRM_IOCTL_MAGIC: u8 = 'd'; // 0x64

// ---------------------------------------------------------------------------
// Ioctl request numbers
// ---------------------------------------------------------------------------

/// DRM_IOCTL_NVIDIA_GEM_IMPORT_USERSPACE_MEMORY
/// DRM_IOWR('d', DRM_COMMAND_BASE + 0x02, drm_nvidia_gem_import_userspace_memory_params)
/// DRM_COMMAND_BASE = 0x40, so nr = 0x42, size = 24 -> 0xc0186442
const NVIDIA_GEM_IMPORT_USERSPACE_MEMORY_NR: u8 = 0x42;

/// DRM_IOCTL_PRIME_HANDLE_TO_FD
/// DRM_IOWR('d', 0x2d, drm_prime_handle), size = 12 -> 0xc00c642d
const PRIME_HANDLE_TO_FD_NR: u8 = 0x2d;

/// DRM_IOCTL_GEM_CLOSE
/// DRM_IOW('d', 0x09, drm_gem_close), size = 8 -> 0x40086409
const GEM_CLOSE_NR: u8 = 0x09;

// ---------------------------------------------------------------------------
// Parameter structs
// ---------------------------------------------------------------------------

/// drm_nvidia_gem_import_userspace_memory_params (24 bytes)
/// IN:  size (page-aligned bytes), address (CPU virtual address from mapMemory)
/// OUT: handle (GEM object handle)
pub const GemImportUserspaceMemory = extern struct {
    size: u64, // IN: size in bytes, page-aligned
    address: u64, // IN: CPU virtual address
    handle: u32, // OUT: GEM handle
    _pad: u32 = 0,

    /// Computed request: DRM_IOWR('d', 0x42, GemImportUserspaceMemory)
    pub const req: u32 = std.os.linux.IOCTL.IOWR(DRM_IOCTL_MAGIC, NVIDIA_GEM_IMPORT_USERSPACE_MEMORY_NR, GemImportUserspaceMemory);

    comptime {
        // Must be 24 bytes: 8+8+4+4
        if (@sizeOf(GemImportUserspaceMemory) != 24) @compileError("GemImportUserspaceMemory must be 24 bytes");
    }
};

/// drm_prime_handle (12 bytes)
/// IN:  handle (GEM handle), flags (O_RDWR|O_CLOEXEC = 0x80002)
/// OUT: fd (dma-buf fd)
pub const PrimeHandleToFd = extern struct {
    handle: u32, // IN: GEM handle
    flags: u32, // IN: O_RDWR|O_CLOEXEC
    fd: i32 = -1, // OUT: dma-buf fd

    /// Computed request: DRM_IOWR('d', 0x2d, PrimeHandleToFd)
    pub const req: u32 = std.os.linux.IOCTL.IOWR(DRM_IOCTL_MAGIC, PRIME_HANDLE_TO_FD_NR, PrimeHandleToFd);
    /// flags: O_RDWR (0x2) | O_CLOEXEC (0x80000) = 0x80002
    pub const FLAGS_RDWR_CLOEXEC: u32 = 0x80002;

    comptime {
        // Must be 12 bytes: 4+4+4
        if (@sizeOf(PrimeHandleToFd) != 12) @compileError("PrimeHandleToFd must be 12 bytes");
    }
};

/// drm_gem_close (8 bytes)
/// IN: handle (GEM handle to release)
pub const GemClose = extern struct {
    handle: u32, // IN: GEM handle
    _pad: u32 = 0,

    /// Computed request: DRM_IOW('d', 0x09, GemClose)
    pub const req: u32 = std.os.linux.IOCTL.IOW(DRM_IOCTL_MAGIC, GEM_CLOSE_NR, GemClose);

    comptime {
        // Must be 8 bytes: 4+4
        if (@sizeOf(GemClose) != 8) @compileError("GemClose must be 8 bytes");
    }
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

pub const DrmError = error{
    /// Could not open or identify the nvidia-drm render node
    DrmOpenFailed,
    /// GEM_IMPORT_USERSPACE_MEMORY ioctl failed (bad VA, not system memory, etc.)
    GemImportFailed,
    /// PRIME_HANDLE_TO_FD ioctl failed
    PrimeExportFailed,
};

// ---------------------------------------------------------------------------
// Open + identify the nvidia-drm render node
// ---------------------------------------------------------------------------

/// Scan /dev/dri/ for renderD* entries via getdents64, check each via the sysfs
/// driver symlink (/sys/class/drm/<name>/device/driver), and open the first one
/// whose driver basename is "nvidia" O_RDWR|O_CLOEXEC.
/// Returns DrmError.DrmOpenFailed if no matching node is found or accessible.
pub fn openNvidiaDrmNode() DrmError!std.posix.fd_t {
    const dri_fd_rc = std.os.linux.open("/dev/dri", .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, 0);
    switch (std.os.linux.errno(dri_fd_rc)) {
        .SUCCESS => {},
        else => return error.DrmOpenFailed,
    }
    const dri_fd: i32 = @intCast(dri_fd_rc);
    defer _ = std.os.linux.close(dri_fd);

    var dents_buf: [4096]u8 align(@alignOf(std.os.linux.dirent64)) = undefined;
    while (true) {
        const n = std.os.linux.getdents64(dri_fd, &dents_buf, dents_buf.len);
        switch (std.os.linux.errno(n)) {
            .SUCCESS => {},
            else => return error.DrmOpenFailed,
        }
        if (n == 0) break; // end of directory

        var offset: usize = 0;
        while (offset < n) {
            const dent: *const std.os.linux.dirent64 = @ptrCast(@alignCast(&dents_buf[offset]));
            offset += dent.reclen;

            // Extract the name: it lives right after the fixed fields.
            const name_ptr: [*:0]const u8 = @ptrCast(&dent.name);
            const name = std.mem.sliceTo(name_ptr, 0);

            // Only consider renderD* entries.
            if (!std.mem.startsWith(u8, name, "renderD")) continue;

            // Build the sysfs driver symlink path: /sys/class/drm/<name>/device/driver
            var sysfs_buf: [128]u8 = undefined;
            const sysfs_path = std.fmt.bufPrintZ(&sysfs_buf, "/sys/class/drm/{s}/device/driver", .{name}) catch continue;

            var link_buf: [256]u8 = undefined;
            const link_len = std.os.linux.readlink(sysfs_path.ptr, &link_buf, link_buf.len);
            if (@as(isize, @bitCast(link_len)) < 0) continue;
            const link_str = link_buf[0..link_len];

            // Extract the basename correctly: if no slash, whole string is the basename.
            const basename = if (std.mem.lastIndexOfScalar(u8, link_str, '/')) |i| link_str[i + 1 ..] else link_str;
            if (!std.mem.eql(u8, basename, "nvidia")) continue;

            // Found the nvidia-drm node -- open it.
            var dev_buf: [64]u8 = undefined;
            const dev_path = std.fmt.bufPrintZ(&dev_buf, "/dev/dri/{s}", .{name}) catch continue;
            const rc = std.os.linux.open(dev_path.ptr, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, 0);
            switch (std.os.linux.errno(rc)) {
                .SUCCESS => return @intCast(rc),
                else => continue,
            }
        }
    }
    return error.DrmOpenFailed;
}

// ---------------------------------------------------------------------------
// memToDmaBuf: CPU VA -> real Linux dma-buf fd
// ---------------------------------------------------------------------------

/// Turn a CPU virtual address (from Client.mapMemory on a .system memory object)
/// into a real Linux dma-buf fd via the nvidia-drm node.
///
/// Steps:
///   1. Open /dev/dri/renderD128 (verified as nvidia-drm).
///   2. GEM_IMPORT_USERSPACE_MEMORY: lock `size` bytes at `va` via get_user_pages.
///   3. PRIME_HANDLE_TO_FD: export the GEM object as a dma-buf fd.
///   4. GEM_CLOSE: release the GEM handle (dma-buf holds its own ref).
///   5. Close the renderD128 fd (dma-buf holds the buffer alive independently).
///
/// IMPORTANT: `va` must remain mapped (unmapMemory must NOT be called) until
/// the dma-buf consumer is done. System memory only -- VRAM BAR mappings will
/// return EFAULT/EINVAL from get_user_pages.
///
/// `size` must not exceed the length of the mapped region that produced `va`.
/// GEM_IMPORT pins exactly this range via get_user_pages; a size larger than
/// the mapped VMA fails or over-pins memory.
pub fn memToDmaBuf(va: usize, size: usize) DrmError!std.posix.fd_t {
    // Page-align size up.
    const page_size: usize = 4096;
    const aligned_size: u64 = (size + page_size - 1) & ~(page_size - 1);

    const drm_fd = try openNvidiaDrmNode();
    errdefer _ = std.os.linux.close(drm_fd);

    // Step 1: Import the userspace pages as a GEM object.
    var import = GemImportUserspaceMemory{
        .size = aligned_size,
        .address = @intCast(va),
        .handle = 0,
    };
    {
        const rc = std.os.linux.ioctl(drm_fd, GemImportUserspaceMemory.req, @intFromPtr(&import));
        switch (std.os.linux.errno(rc)) {
            .SUCCESS => {},
            else => return error.GemImportFailed,
        }
    }
    // Always close the GEM handle when we leave this scope (even on prime error).
    var gem_closed = false;
    defer {
        if (!gem_closed) {
            var gc = GemClose{ .handle = import.handle };
            _ = std.os.linux.ioctl(drm_fd, GemClose.req, @intFromPtr(&gc));
        }
    }

    // Step 2: Export the GEM object as a real dma-buf fd.
    var prime = PrimeHandleToFd{
        .handle = import.handle,
        .flags = PrimeHandleToFd.FLAGS_RDWR_CLOEXEC,
        .fd = -1,
    };
    {
        const rc = std.os.linux.ioctl(drm_fd, PrimeHandleToFd.req, @intFromPtr(&prime));
        switch (std.os.linux.errno(rc)) {
            .SUCCESS => {},
            else => return error.PrimeExportFailed,
        }
    }

    // Step 3: Close GEM handle (dma-buf holds its own ref).
    {
        var gc = GemClose{ .handle = import.handle };
        const rc = std.os.linux.ioctl(drm_fd, GemClose.req, @intFromPtr(&gc));
        gem_closed = true;
        // Non-fatal if close fails: the dma-buf is already exported.
        _ = rc;
    }

    // Close the render node fd; the dma-buf keeps the buffer alive independently.
    _ = std.os.linux.close(drm_fd);

    return @intCast(prime.fd);
}

// ---------------------------------------------------------------------------
// ABI / @sizeOf sanity tests
// ---------------------------------------------------------------------------

test "GemImportUserspaceMemory is 24 bytes" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(GemImportUserspaceMemory));
}

test "PrimeHandleToFd is 12 bytes" {
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(PrimeHandleToFd));
}

test "GemClose is 8 bytes" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(GemClose));
}

test "GemImportUserspaceMemory ioctl number is 0xc0186442" {
    try std.testing.expectEqual(@as(u32, 0xc0186442), GemImportUserspaceMemory.req);
}

test "PrimeHandleToFd ioctl number is 0xc00c642d" {
    try std.testing.expectEqual(@as(u32, 0xc00c642d), PrimeHandleToFd.req);
}

test "GemClose ioctl number is 0x40086409" {
    try std.testing.expectEqual(@as(u32, 0x40086409), GemClose.req);
}
