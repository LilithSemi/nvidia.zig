//! nvidia-info - the single dual-target NVIDIA probe (parity with asahi-info).
//!
//! ONE example, comptime target-gated. Zig only analyzes the referenced branch
//! (the same lazy-analysis trick the comptime driver registry uses), so the UEFI
//! branch's std.os.uefi usage does not break the Linux build and vice versa.
//!
//!   * On UEFI (bare metal): boots as a UEFI application, enumerates PCI via
//!     EFI_PCI_IO_PROTOCOL, finds the first NVIDIA display/3D controller, maps
//!     BAR0 and reads + decodes the chip id from NV_PMC_BOOT_0 / NV_PMC_BOOT_42,
//!     then halts so the output can be read off a real machine's console. This is
//!     milestone 1 of the baremetal bring-up and is strictly READ-ONLY against
//!     the GPU. The freestanding RM-over-GSP + HAL path is force-analysed via the
//!     gated gspBringUpHook below.
//!
//!   * On Linux: a normal executable that opens the kernel RM over the
//!     user-accessible /dev/nvidiactl + /dev/nvidia* nodes (NO root), brings up
//!     the GPU, and prints its identity (name, gpu id, driver version) plus a
//!     small sysmem alloc/map/write+readback roundtrip. Output is plain +
//!     greppable (one "key: value" per line; a final "RESULT:" line). On a box
//!     with no NVIDIA GPU it prints "RESULT: SKIP no-nvidia-device" and exits 0.
//!
//! Output style mirrors asahi-info: greppable "key: value" lines and a final
//! RESULT: OK / RESULT: SKIP / RESULT: FAIL.

const std = @import("std");
const builtin = @import("builtin");
const conduit = @import("conduit");
// The shared UEFI console subproject. On UEFI it gives the con_out std.Io.Writer
// (UTF-8 -> UTF-16, '\n' -> "\r\n") + the std.log / panic opt-in below; on Linux
// it is imported harmlessly (its std.os.uefi usage is comptime-gated) and unused.
const uefi_support = @import("uefi");

const is_uefi = builtin.os.tag == .uefi;

// Opt into UEFI console support from the ROOT, but ONLY in the uefi build branch
// (on Linux std.log + panic already work natively and we must not override them).
// These re-exports route std.log.* and panics to the EFI console; the greppable
// data lines below stay on a PLAIN writer (out()) so existing greps are intact.
// NOTE (honest, Zig 0.16): std.debug.print is NOT routable to con_out from the
// root - it goes to std.fs.File.stderr(), which has no UEFI fd. std.log IS the
// supported routable path. See subproject/uefi/src/uefi.zig.
pub const std_options: std.Options = if (is_uefi) uefi_support.std_options else .{};
pub const panic = if (is_uefi) uefi_support.panic else std.debug.FullPanic(std.debug.defaultPanic);

// Comptime target-gated entry. Zig analyzes only the referenced branch, so the
// UEFI std.os.uefi usage and the Linux std.os.linux usage never clash.
pub const main = if (is_uefi) uefiMain else linuxMain;

// The data sink for the greppable "key: value" / RESULT lines is the single
// per-target writer from the shared uefi subproject: `uefi_support.init()`
// returns con_out on UEFI and a buffered stdout writer on Linux. std.log + panic
// (routed to con_out on UEFI via the opt-in above) are reserved for diagnostics,
// NOT the data.

const NVIDIA_VENDOR_ID: u16 = 0x10DE;
const PCI_CLASS_DISPLAY: u8 = 0x03; // base class: display (3D / VGA) controller

// ===========================================================================
// Linux path: kernel RM probe (mirrors asahi-info's shape + output style).
// ===========================================================================

const nvidia = @import("nvidia");

fn linuxMain(init: std.process.Init.Minimal) void {
    _ = init;

    // The single per-target writer (buffered stdout on Linux) for the greppable
    // data lines. flush on return so the buffered lines are not lost. std.log /
    // std.debug.print also work natively on Linux.
    const lw = uefi_support.init();
    defer uefi_support.flush();

    // Open the kernel RM (version handshake over /dev/nvidiactl, no root). A
    // failure here means no usable NVIDIA device/driver - a graceful SKIP, like
    // asahi-info on a GPU-less box.
    var client = nvidia.Client.open() catch {
        lw.print("RESULT: SKIP no-nvidia-device (could not open the kernel RM /dev/nvidiactl)\n", .{}) catch {};
        return;
    };
    defer client.deinit();
    lw.print("device: opened the kernel RM\n", .{}) catch {};

    const ver = client.driverVersion();
    lw.print("driver_version: {d}.{d}.{d}\n", .{ ver.major, ver.minor, ver.patch }) catch {};
    lw.print("driver_abi: {s}\n", .{@tagName(client.abi())}) catch {};

    // Bring up the first GPU end to end (register node, attach, alloc client/
    // device/subdevice). If there is no GPU behind the driver this fails - SKIP.
    const dev = client.allocDevice(0) catch {
        lw.print("RESULT: SKIP no-nvidia-device (the RM opened but no GPU could be brought up)\n", .{}) catch {};
        return;
    };
    defer client.freeDevice(dev);

    // Identity (the same RM controls lib/prism/drivers/nvidia caps() reuses).
    var name_buf: [128]u8 = undefined;
    const name = client.getGpuName(dev, &name_buf) catch "NVIDIA GPU";
    lw.print("gpu_name: {s}\n", .{name}) catch {};

    const gpu_id = client.getGpuId(dev) catch 0;
    lw.print("gpu_id: 0x{x}\n", .{gpu_id}) catch {};
    lw.print("rm_gpu_id: 0x{x}\n", .{dev.gpu_id}) catch {};
    lw.print("device_minor: {d}\n", .{dev.minor}) catch {};

    // PCI identity via conduit's /sys backend (no root, addresses only). The RM
    // above is the authoritative device; this adds the bus geometry + BAR map.
    printPciIdentity(lw);

    // Parity stretch (asahi-info does a GEM roundtrip): a small sysmem alloc +
    // CPU map + write/readback. Reuses the existing RM Client - no new machinery.
    const size: u64 = 16 * 1024;
    const mem = client.allocMemory(dev, .system, size) catch {
        // Identity already printed; the alloc is a bonus, so still report OK.
        lw.print("roundtrip: SKIP sysmem alloc failed\n", .{}) catch {};
        lw.print("RESULT: OK\n", .{}) catch {};
        return;
    };
    defer client.freeMemory(dev, mem);

    const mapping = client.mapMemory(dev, mem) catch {
        lw.print("roundtrip: SKIP sysmem map failed\n", .{}) catch {};
        lw.print("RESULT: OK\n", .{}) catch {};
        return;
    };
    defer client.unmapMemory(mapping);

    var ok = true;
    var i: usize = 0;
    while (i < mapping.bytes.len) : (i += 1) {
        mapping.bytes[i] = @truncate(i *% 31 +% 7);
    }
    i = 0;
    while (i < mapping.bytes.len) : (i += 1) {
        const expect: u8 = @truncate(i *% 31 +% 7);
        if (mapping.bytes[i] != expect) {
            ok = false;
            break;
        }
    }

    if (ok) {
        lw.print("roundtrip: {d} bytes sysmem write+readback OK\n", .{mapping.bytes.len}) catch {};
        lw.print("RESULT: OK\n", .{}) catch {};
    } else {
        lw.print("roundtrip: MISMATCH at byte {d}\n", .{i}) catch {};
        lw.print("RESULT: FAIL roundtrip\n", .{}) catch {};
    }
}

// Enumerate PCI via conduit's Linux /sys backend (read-only, NO root) and print
// the NVIDIA GPU's PCI identity: segment:bus:dev.fn, vendor:device, class, and
// BAR0 base/size. /sys exposes BAR ADDRESSES only - we do NOT read BAR0 MMIO on
// Linux (that needs root; honour conduit-no-root-on-linux). The kernel RM above
// already proved the device; this is the bus-level view through conduit.
fn printPciIdentity(lw: *std.Io.Writer) void {
    var be = conduit.backend.pci.PciBackend.init();
    const reg = conduit.Registry.init(be.any(), &.{conduit.pci_matcher});

    var it = reg.iter(.pci);
    while (it.next() catch null) |m| {
        const info = m.pci orelse continue;
        if (info.vendor_id != NVIDIA_VENDOR_ID or info.class_code != PCI_CLASS_DISPLAY) continue;

        lw.print("pci_address: {x:0>4}:{x:0>2}:{x:0>2}.{x}\n", .{ info.segment, info.bus, info.device, info.function }) catch {};
        lw.print("pci_vendor_device: {x:0>4}:{x:0>4}\n", .{ info.vendor_id, info.device_id }) catch {};
        lw.print("pci_class: 0x{x:0>2}{x:0>2}{x:0>2}\n", .{ info.class_code, info.subclass, info.prog_if }) catch {};
        if (m.mmio()) |bar0| {
            lw.print("pci_bar0_base: 0x{x}\n", .{bar0.base}) catch {};
            lw.print("pci_bar0_size: 0x{x}\n", .{bar0.size}) catch {};
        } else {
            lw.print("pci_bar0: (unassigned)\n", .{}) catch {};
        }
        return; // first NVIDIA display controller is enough.
    }
    lw.print("pci: no NVIDIA display controller found via /sys\n", .{}) catch {};
}

// ===========================================================================
// UEFI path: the baremetal GPU probe (milestone 1), now via conduit's EFI_PCI_IO
// backend (the hand-rolled pci_io.zig moved into conduit).
// ===========================================================================

const uefi = std.os.uefi;
const hw = nvidia.hw;

// Force the whole HAL -> Transport -> GSP path to be semantically analyzed on the
// UEFI/freestanding target. On freestanding, transport.zig comptime-selects the
// freestanding (RM-over-GSP) Transport, so referencing rm.Client (which embeds the
// Transport) + the transport-agnostic draw path (graphics/threed) makes Zig compile
// the entire freestanding RM-over-GSP + HAL chain here - the P6 "the freestanding
// build COMPILES the complete rm.zig + HAL + transport + gsp path" requirement.
// This guards against any of those modules regressing the freestanding build even
// before open()-on-metal is actually called. Gated behind the UEFI target so it is
// only analyzed on the freestanding build.
comptime {
    if (builtin.os.tag == .uefi) {
        _ = nvidia.gsp; // the GSP transport core (rings, RPC envelopes, boot)
        _ = nvidia.rm; // rm.Client -> the comptime-selected (freestanding) Transport
        _ = nvidia.Client; // the HAL client over the freestanding Transport
        _ = nvidia.Queue; // GPFIFO submission (the draw path submits through this)
        _ = nvidia.graphics; // the transport-agnostic shaded-draw method builders
        _ = nvidia.threed; // the transport-agnostic clear/triangle method builders
        _ = &gspBringUpHook; // analyse the whole freestanding HAL->Transport->GSP chain
    }
}

// A pointer to the shared con_out writer, set up in uefiMain so every helper can
// format through it. The shared uefi subproject writer translates UTF-8 to UTF-16
// and '\n' to "\r\n" (so a "\n" terminator emits CRLF on con_out), exactly what
// the old per-subproject console.zig did.
var w: *std.Io.Writer = undefined;

fn uefiMain() uefi.Status {
    // Capture + reset con_out via the shared uefi subproject; init() returns the
    // con_out writer. After this, std.log.* and panics (opted in at the root
    // above) also reach con_out. pauseHalt() flushes before spinning.
    w = uefi_support.init();

    w.print("\n", .{}) catch {};
    w.print("=== Prism UEFI GPU probe (milestone 1) ===\n", .{}) catch {};
    w.print("Enumerating PCI via conduit (EFI_PCI_IO_PROTOCOL)...\n", .{}) catch {};
    w.print("\n", .{}) catch {};

    // conduit's UEFI PCI backend wraps EFI_PCI_IO_PROTOCOL; discover/match drive
    // it exactly like the device-tree backend. The pci_matcher claims every PCI
    // node; we refine on the numeric vendor/class to find the NVIDIA GPU.
    var be = conduit.backend.pci.PciBackend.init();
    const reg = conduit.Registry.init(be.any(), &.{conduit.pci_matcher});

    w.print("seg:bus:dev.fn  vendor:device  class  rev\n", .{}) catch {};
    w.print("----------------------------------------------\n", .{}) catch {};

    var found: ?conduit.discover.Match = null;
    var it = reg.iter(.pci);
    while (it.next() catch null) |m| {
        const info = m.pci orelse continue;
        printDevice(info);
        if (info.vendor_id == NVIDIA_VENDOR_ID and info.class_code == PCI_CLASS_DISPLAY and found == null) {
            found = m;
        }
    }

    w.print("\n", .{}) catch {};
    const gpu = found orelse {
        w.print("RESULT: no NVIDIA display/3D controller (vendor 0x10DE, class 0x03) found\n", .{}) catch {};
        w.print("        (under QEMU+OVMF without a real GPU this is expected; the\n", .{}) catch {};
        w.print("         enumeration + NVIDIA-match path above still ran)\n", .{}) catch {};
        w.print("=== probe done ===\n", .{}) catch {};
        pauseHalt();
    };
    const ginfo = gpu.pci.?;

    w.print("Found NVIDIA GPU at ", .{}) catch {};
    printLoc(ginfo);
    w.print("  vendor:device = 0x{X:0>4}:0x{X:0>4}\n", .{ ginfo.vendor_id, ginfo.device_id }) catch {};

    // BAR0 = the GPU MMIO register aperture, lowered by conduit into the first
    // MMIO resource (the EFI_PCI_IO backend reads it straight from config space).
    const bar0_res = gpu.mmio() orelse {
        w.print("RESULT: BAR0 not assigned; cannot read GPU registers\n", .{}) catch {};
        pauseHalt();
    };
    const bar0_base: u64 = bar0_res.base;
    const bar0_size: u64 = if (bar0_res.size != 0) bar0_res.size else 16 * 1024 * 1024;

    w.print("BAR0 base = 0x{X:0>16}  size = 0x{X:0>16}\n", .{ bar0_base, bar0_size }) catch {};

    // BAR1 (the VRAM/window aperture) is informational this milestone.
    if (gpu.mmioAt(1)) |bar1| {
        w.print("BAR1 base = 0x{X:0>16}  size = 0x{X:0>16}\n", .{ bar1.base, bar1.size }) catch {};
    }

    if (bar0_base == 0) {
        w.print("RESULT: BAR0 not assigned; cannot read GPU registers\n", .{}) catch {};
        pauseHalt();
    }

    // Read the boot registers straight off BAR0 through conduit's MMIO HAL
    // (identity-mapped volatile pointer), then decode the chip id. Read-only.
    const mmio = conduit.Mmio.direct(bar0_base);
    const boot0 = mmio.read(u32, hw.chip.NV_PMC_BOOT_0);
    const boot42 = mmio.read(u32, hw.chip.NV_PMC_BOOT_42);
    const id = hw.chip.decode(boot0, boot42);

    w.print("\n", .{}) catch {};
    w.print("--- GPU chip id ---\n", .{}) catch {};
    w.print("PMC_BOOT_0  (0x0000) = 0x{X:0>8}\n", .{id.boot0}) catch {};
    w.print("PMC_BOOT_42 (0x0A00) = 0x{X:0>8}\n", .{id.boot42}) catch {};
    w.print("architecture = 0x{X:0>8} ({s})\n", .{ id.architecture, id.arch().name() }) catch {};
    w.print("chip id      = 0x{X:0>8}\n", .{id.chip_id}) catch {};
    w.print("implementation = 0x{X:0>8}  revision = 0x{X:0>8}.0x{X:0>8}\n", .{ id.implementation, id.major_revision, id.minor_revision }) catch {};

    w.print("\n", .{}) catch {};
    w.print(">>> {s} detected - metal path is live <<<\n", .{id.arch().name()}) catch {};
    if (id.arch() == .ga10x) {
        w.print(">>> Ampere (30-series): the primary baremetal bring-up target <<<\n", .{}) catch {};
    } else if (!id.arch().isGspBased()) {
        w.print(">>> note: pre-Turing or unrecognized arch - GSP bring-up path may differ <<<\n", .{}) catch {};
    }

    w.print("\n", .{}) catch {};
    w.print("=== probe done ===\n", .{}) catch {};
    pauseHalt();
}

// Halt with the output left on screen, via the shared uefi subproject (it prints
// the same ">>> HALTED ... <<<" banner then spins forever). Without this the probe
// returns to the UEFI firmware, which boots on (or resets) before the chip id can
// be read or photographed.
fn pauseHalt() noreturn {
    uefi_support.halt();
}

// ===========================================================================
// GSP bring-up hook (P6): the assembly point where, on metal, the BAR0 + chip-id
// the probe above read would be handed to the freestanding RM-over-GSP transport
// to boot the GSP and bring the HAL up. This is GATED OFF (never called from main)
// because the live boot is the user's deferred hardware test - but referencing it
// force-ANALYSES the whole freestanding rm.zig + HAL + transport + GSP path on the
// UEFI build, so a regression in any of those breaks the build here (the P6
// "the freestanding/UEFI build COMPILES the complete path" guarantee).
//
// To actually drive it on metal: gather the firmware (@embedFile gsp_ga10x.bin) +
// the booter blobs, package them with fw.zig/boot.zig into BootInputs, EFI-
// AllocatePages the identity-mapped sysmem pool + carve the VRAM, format the host
// ring endpoint over the shared region, then call Transport.openOnMetal(...). The
// returned Transport backs an rm.Client the existing graphics/threed draw path
// runs on unchanged.
fn gspBringUpHook(bar0: hw.Bar, arch: hw.chip.Architecture, inputs: nvidia.gsp.boot.BootInputs, endpoint: nvidia.gsp.ring.Endpoint, wpr_meta_cpu: []const u8, sysmem_base: u64, sysmem_size: u64, vram_base: u64, vram_size: u64) !void {
    const Transport = nvidia.transport.Transport; // freestanding on UEFI
    // Assemble the metal inputs the same way the live path will.
    const metal = Transport.MetalInputs{
        .bar = bar0,
        .arch = arch,
        .sysmem_base = sysmem_base,
        .sysmem_size = sysmem_size,
        .vram_base = vram_base,
        .vram_size = vram_size,
        .boot_inputs = inputs,
        .endpoint = endpoint,
        .wpr_meta_cpu = wpr_meta_cpu,
    };
    // open()-on-metal: boot the GSP + post-init, leaving the ring live. (Deferred:
    // this RUNS only on the real GPU; the compile here is the P6 deliverable.)
    const t = try Transport.openOnMetal(metal);
    defer t.deinit();

    // Bring the GPU up over GSP + exercise the RM-over-GSP surface the HAL uses.
    var client = nvidia.Client{ .t = t };
    const dev = try client.allocDevice(0);
    defer client.freeDevice(dev);
    const vaspace = try client.allocVaSpace(dev);
    const mem = try client.allocMemory(dev, .vram, 0x1000);
    const gmap = try client.mapToGpu(dev, vaspace, mem, 0x200000);
    _ = gmap;
    const cmap = try client.mapMemory(dev, mem);
    _ = cmap;
}

// ---- PCI line formatters (over the shared UEFI std.Io.Writer `w`) --------

/// seg:bus:dev.fn, each component as a 0x-prefixed hex (seg/bus/dev 2 digits,
/// function 1 digit) - the same shape the old printLoc emitted.
fn printLoc(info: conduit.backend.PciInfo) void {
    w.print("0x{X:0>2}:0x{X:0>2}:0x{X:0>2}.0x{X:0>1}", .{ info.segment, info.bus, info.device, info.function }) catch {};
}

/// One PCI enumeration line: indent, location, vendor:device, class, revision,
/// and the NVIDIA tag. Byte-identical to the old printDevice.
fn printDevice(info: conduit.backend.PciInfo) void {
    w.print("  ", .{}) catch {};
    printLoc(info);
    w.print("   0x{X:0>4}:0x{X:0>4}   0x{X:0>2}   0x{X:0>2}", .{ info.vendor_id, info.device_id, info.class_code, info.revision }) catch {};
    if (info.vendor_id == NVIDIA_VENDOR_ID) {
        w.print("   <- NVIDIA", .{}) catch {};
        if (info.class_code == PCI_CLASS_DISPLAY) w.print(" display/3D", .{}) catch {};
    }
    w.print("\n", .{}) catch {};
}
