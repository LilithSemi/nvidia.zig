const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root_module = b.addModule("nvidia", .{
        .root_source_file = b.path("src/nvidia.zig"),
        .target = target,
        .optimize = optimize,
    });

    // conduit (device discovery + MMIO) for the PCI enumeration in nvidia-info:
    // /sys on Linux, EFI_PCI_IO on UEFI. Only the example module imports it;
    // Zig's lazy analysis pulls in just the OS-specific backend the referenced
    // branch (linuxMain / uefiMain) touches.
    const conduit_dep = b.dependency("conduit", .{
        .target = target,
        .optimize = optimize,
    });
    const conduit_module = conduit_dep.module("conduit");

    // The shared EFI-console subproject (con_out writer + std.log/panic opt-in).
    const uefi_dep = b.dependency("uefi", .{
        .target = target,
        .optimize = optimize,
    });
    const uefi_module = uefi_dep.module("uefi");

    const test_step = b.step("test", "Run nvidia subproject tests");
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{
        .root_module = root_module,
    })).step);

    // The single dual-target nvidia-info probe (parity with asahi-info):
    //   * Linux -> a normal executable "nvidia-info" in bin/ (the kernel-RM probe).
    //   * UEFI  -> a UEFI application named BOOTAA64/BOOTX64 by arch, installed to
    //     EFI/BOOT/ (the baremetal milestone-1 PCI + BAR0 + chip-id probe).
    const is_uefi = target.result.os.tag == .uefi;

    const exe_name = if (is_uefi) switch (target.result.cpu.arch) {
        .x86_64 => "BOOTX64",
        .aarch64 => "BOOTAA64",
        else => "BOOTEFI",
    } else "nvidia-info";

    const info = b.addExecutable(.{
        .name = exe_name,
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/nvidia-info.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nvidia", .module = root_module },
                .{ .name = "conduit", .module = conduit_module },
                .{ .name = "uefi", .module = uefi_module },
            },
        }),
    });

    // UEFI artifacts land in EFI/BOOT/ (the removable-media default boot path);
    // Linux artifacts land in bin/.
    const install = if (is_uefi)
        b.addInstallArtifact(info, .{
            .dest_dir = .{ .override = .{ .custom = "EFI/BOOT" } },
        })
    else
        b.addInstallArtifact(info, .{});

    b.getInstallStep().dependOn(&install.step);

    const probe_step = b.step("nvidia-info", "Build the nvidia-info probe (Linux bin / UEFI EFI/BOOT)");
    probe_step.dependOn(&install.step);

    // A convenience run step (Linux only - the UEFI build is run on the metal).
    if (!is_uefi) {
        const run_probe = b.addRunArtifact(info);
        if (b.args) |args| run_probe.addArgs(args);
        const run_step = b.step("run-nvidia-info", "Run the nvidia-info probe (needs an NVIDIA GPU)");
        run_step.dependOn(&run_probe.step);
    }
}
