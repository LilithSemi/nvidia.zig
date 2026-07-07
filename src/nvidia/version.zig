//! Driver-version detection + ABI generation selection.
//!
//! The NVIDIA RM ABI (struct layouts, control commands, class params) shifts
//! between driver versions. Because we learn the driver version at runtime
//! (NV_ESC_CHECK_VERSION_STR), version-specific code switches on `Abi` (an enum
//! you can `switch` on, or key a tagged union by) instead of hardcoding one
//! layout - keeping us compatible across releases.

const std = @import("std");

pub const Version = struct {
    major: u32 = 0,
    minor: u32 = 0,
    patch: u32 = 0,

    /// Parse "595.71.05" -> {595, 71, 5}. Trailing junk is ignored.
    pub fn parse(s: []const u8) ?Version {
        var it = std.mem.splitScalar(u8, std.mem.trim(u8, s, " \t\r\n"), '.');
        const maj = std.fmt.parseInt(u32, it.next() orelse return null, 10) catch return null;
        const min = if (it.next()) |m| std.fmt.parseInt(u32, m, 10) catch 0 else 0;
        const pat = if (it.next()) |p| std.fmt.parseInt(u32, p, 10) catch 0 else 0;
        return .{ .major = maj, .minor = min, .patch = pat };
    }

    pub fn atLeast(self: Version, major: u32, minor: u32) bool {
        if (self.major != major) return self.major > major;
        return self.minor >= minor;
    }
};

/// RM ABI generation, picked from the detected driver version. Add variants as
/// new generations diverge; version-specific structs/commands switch on this.
pub const Abi = enum {
    /// 5xx-series open kernel module (the layouts ported so far, e.g. 595.71).
    open_5xx,
    /// Unrecognized version - callers should treat as the newest known with care.
    unknown,

    pub fn fromVersion(v: Version) Abi {
        if (v.major >= 500 and v.major < 600) return .open_5xx;
        return .unknown;
    }
};

test "version parse + abi selection" {
    const v = Version.parse("595.71.05").?;
    try std.testing.expectEqual(@as(u32, 595), v.major);
    try std.testing.expectEqual(@as(u32, 71), v.minor);
    try std.testing.expectEqual(@as(u32, 5), v.patch);
    try std.testing.expectEqual(Abi.open_5xx, Abi.fromVersion(v));
    try std.testing.expect(v.atLeast(595, 0));
    try std.testing.expect(!v.atLeast(600, 0));
}
