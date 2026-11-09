#!/usr/bin/env python3
"""Lock PFA GF31/GF61 double-wide tail ownership and trig invariants."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
gpu = (ROOT / "src" / "Gpu.cpp").read_text()
tail = (ROOT / "src" / "cl" / "tailsquare.cl").read_text()
trig = (ROOT / "src" / "TrigBufCache.cpp").read_text()

fail = []


def between(text, start_marker, end_marker):
    start = text.find(start_marker)
    if start < 0:
        fail.append(f"missing start marker: {start_marker}")
        return ""

    end = text.find(end_marker, start)
    if end < 0:
        fail.append(f"missing end marker after: {start_marker}")
        return ""

    return text[start:end + len(end_marker)]


pfa_host = (
    "!tail_single_wide && !tail_single_kernel ? "
    "hN / nH - SMALL_H / nH * 2 * "
    "(fft.isPfa() ? fft.pfa_radix : 1) :"
)
legacy_host = (
    "!tail_single_wide && !tail_single_kernel ? "
    "hN / nH - SMALL_H / nH * 2 :"
)

if gpu.count(pfa_host) != 2:
    fail.append(
        f"expected two PFA-aware GF host formulas, got {gpu.count(pfa_host)}"
    )

if gpu.count(legacy_host) != 1:
    fail.append(
        f"expected one untouched FP host formula, got {gpu.count(legacy_host)}"
    )


for name, out_name in (
    ("tailSquareGF31", "out31"),
    ("tailSquareGF61", "out61"),
):
    final_write = (
        f"writeTailFusedLine(u, {out_name}, "
        "transPos(line, MIDDLE, WIDTH), lowMe);\n}"
    )

    block = between(
        tail,
        f"KERNEL(G_H * 2) {name}(",
        final_write,
    )

    required = (
        "#if PFA_RADIX",
        "const u32 per_row = WIDTH / 2u - 1u;",
        "const u32 row = group / per_row;",
        "const u32 binary_line = group - row * per_row + 1u;",
        "const u32 line_u = row * WIDTH + binary_line;",
        "const u32 line_v = row * WIDTH + (WIDTH - binary_line);",
        "binary_line*2 + isSecondHalf",
        "binary_line*G_H*2 + me",
        "revCrossLine(lds, u);",
        "pairSq(NH/2, u, u + NH/2, trig, false);",
    )

    for token in required:
        if token not in block:
            fail.append(f"{name}: missing {token}")

    if "line_u*2 + isSecondHalf" in block:
        fail.append(f"{name}: stale flattened multiplier trig index")

    if "line_u*G_H*2 + me" in block:
        fail.append(f"{name}: stale flattened full-trig index")


for name, out_name in (
    ("tailSquareZeroGF31", "out31"),
    ("tailSquareZeroGF61", "out61"),
):
    final_write = (
        f"writeTailFusedLine(u, {out_name}, "
        "transPos(line, MIDDLE, WIDTH), me);\n}"
    )

    block = between(
        tail,
        f"KERNEL(G_H) {name}(",
        final_write,
    )

    required = (
        "const u32 row = which >> 1;",
        "const bool upper = (which & 1u) != 0;",
        "const u32 binary_line = upper ? (WIDTH / 2u) : 0u;",
        "const u32 doublewide_trig_slot = upper ? 1u : 0u;",
        "G_H + doublewide_trig_slot",
        "doublewide_trig_slot*G_H + me",
    )

    for token in required:
        if token not in block:
            fail.append(f"{name}: missing {token}")

    if "G_H + which" in block:
        fail.append(f"{name}: stale row-dependent multiplier trig slot")

    if "which*G_H + me" in block:
        fail.append(f"{name}: stale row-dependent direct trig slot")


if "for (u32 v = 0; v < (tail_single_wide ? 1 : 2); ++v)" not in trig:
    fail.append("TrigBufCache: double-wide pair generation missing")

if trig.count("fft.isPfa() ? 1u : middle") < 2:
    fail.append("TrigBufCache: PFA GF31/GF61 middle=1 contract missing")


for radix in (3, 7):
    for width in (256, 512):
        per_row = width // 2 - 1
        groups = radix * per_row
        seen = set()

        for group in range(groups):
            row = group // per_row
            binary_line = group - row * per_row + 1

            if not (0 < binary_line < width // 2):
                fail.append(
                    f"bad binary line radix={radix} width={width}"
                )
                continue

            line_u = row * width + binary_line
            line_v = row * width + (width - binary_line)
            pair = tuple(sorted((line_u, line_v)))

            if pair in seen:
                fail.append(
                    f"duplicate pair radix={radix} width={width}: {pair}"
                )

            seen.add(pair)

        if len(seen) != groups:
            fail.append(
                f"pair count radix={radix} width={width}: "
                f"{len(seen)} != {groups}"
            )

        for row in range(radix):
            lower_slot = 0
            upper_slot = 1

            if (lower_slot, upper_slot) != (0, 1):
                fail.append(
                    f"zero trig slot model failed radix={radix} row={row}"
                )


for msg in fail:
    print("FAIL:", msg)

if fail:
    sys.exit(1)

print(
    "ok: PFA double-wide GF31/GF61 launch geometry, row ownership, "
    "packed trig indexing and zero-line trig slots are locked"
)

# Production policy remains exact-scope. The generic row-aware implementation
# supports PFA3 and PFA7, but automatic TK3 is enabled only on validated gfx906
# geometries. PFA3 also carries its separately validated companion defaults;
# PFA7 changes only TAIL_KERNELS.
policy_tokens = (
    "const bool pfa3_doublewide_gfx906_geometry =",
    "const bool pfa7_doublewide_geometry =",
    "const bool pfa7_doublewide_gfx906_default =",
    "const bool pfa7_doublewide_rtx3080_opencl_default =",
    'config["TAIL_KERNELS"] = "3"',
    'config.try_emplace("TAIL_KERNELS", "1")',
)

for token in policy_tokens:
    if token not in gpu:
        print("FAIL: production policy missing", token)
        sys.exit(1)

pfa3_pos = gpu.find("const bool pfa3_doublewide_gfx906_geometry =")
pfa7_pos = gpu.find("const bool pfa7_doublewide_geometry =")
rtx_pos = gpu.find("const bool pfa7_doublewide_rtx3080_opencl_default =")
generic_pos = gpu.find('config.try_emplace("TAIL_KERNELS", "1")')

if not (0 <= pfa3_pos < pfa7_pos < rtx_pos < generic_pos):
    print("FAIL: hardware-scoped PFA3/PFA7 TK3 policies must precede generic PFA default")
    sys.exit(1)

pfa3_block = gpu[pfa3_pos:pfa7_pos]
pfa7_block = gpu[pfa7_pos:generic_pos]

for token in (
    "fft.pfa_radix == 3",
    "fft.shape.width == 256",
    "fft.shape.middle == 3",
    "fft.shape.height == 512",
    "fft.variant == 101",
    'config.try_emplace("OUT_WG", "256")',
    'config.try_emplace("OUT_SIZEX", "8")',
    'config.try_emplace("MIDDLE_OUT_LDS_TRANSPOSE", "1")',
    'config.try_emplace("LOADS", "1")',
    'config.try_emplace("STORES", "1")',
    "explicit per-key settings retain priority",
):
    if token not in pfa3_block:
        print("FAIL: gfx906 PFA3 combined policy missing", token)
        sys.exit(1)

for token in (
    "fft.isPfa()",
    "fft.pfa_radix == 7",
    "fft.shape.fft_type == FFT3161",
    "fft.NTT_GF31 && fft.NTT_GF61",
    "fft.shape.width == 512",
    "fft.shape.middle == 7",
    "fft.shape.height == 512",
    "fft.variant == 202",
    'pfa7_doublewide_geometry',
    '!config.count("TAIL_KERNELS")',
    'getDeviceName(id).rfind("gfx906", 0) == 0',
    'getDeviceName(id) == "NVIDIA GeForce RTX 3080"',
    '#if !defined(CUDA_BACKEND)',
    'config["TAIL_KERNELS"] = "3"',
    "Validated RTX 3080 OpenCL PFA7 production default",
    "explicit TAIL_KERNELS retains priority",
):
    if token not in pfa7_block:
        print("FAIL: gfx906 PFA7 TK3 policy missing", token)
        sys.exit(1)

for forbidden in (
    'config.try_emplace("OUT_WG"',
    'config.try_emplace("OUT_SIZEX"',
    'config.try_emplace("MIDDLE_OUT_LDS_TRANSPOSE"',
    'config.try_emplace("LOADS"',
    'config.try_emplace("STORES"',
):
    if forbidden in pfa7_block:
        print("FAIL: PFA7 TK3 policy must not inherit PFA3 companion defaults", forbidden)
        sys.exit(1)


def policy_model(device, radix, width, middle, height, variant,
                 fft_type="FFT3161", gf31=True, gf61=True,
                 explicit_tail=None):
    pfa3 = (
        fft_type == "FFT3161"
        and gf31 and gf61
        and (radix, width, middle, height, variant)
            == (3, 256, 3, 512, 101)
    )
    pfa7 = (
        fft_type == "FFT3161"
        and gf31 and gf61
        and (radix, width, middle, height, variant)
            == (7, 512, 7, 512, 202)
    )

    if explicit_tail is not None:
        return explicit_tail

    if device.startswith("gfx906") and (pfa3 or pfa7):
        return 3

    if device == "NVIDIA GeForce RTX 3080" and pfa7:
        return 3

    return 1


assert policy_model("gfx906", 3, 256, 3, 512, 101) == 3
assert policy_model("gfx906:sramecc+", 3, 256, 3, 512, 101) == 3
assert policy_model("gfx906", 7, 512, 7, 512, 202) == 3
assert policy_model("gfx906:sramecc+:xnack-", 7, 512, 7, 512, 202) == 3
assert policy_model("NVIDIA GeForce RTX 3080", 7, 512, 7, 512, 202) == 3

# Explicit user/tuner selections retain priority.
assert policy_model(
    "gfx906", 3, 256, 3, 512, 101, explicit_tail=1
) == 1
assert policy_model(
    "gfx906", 7, 512, 7, 512, 202, explicit_tail=0
) == 0
assert policy_model(
    "NVIDIA GeForce RTX 3080", 7, 512, 7, 512, 202, explicit_tail=1
) == 1

# Negative scope.
assert policy_model("gfx1100", 3, 256, 3, 512, 101) == 1
assert policy_model("gfx1100", 7, 512, 7, 512, 202) == 1
assert policy_model("gfx906", 9, 512, 9, 512, 202) == 1
assert policy_model("gfx906", 7, 512, 7, 256, 202) == 1
assert policy_model("gfx906", 7, 1024, 7, 512, 202) == 1
assert policy_model("gfx906", 7, 512, 8, 512, 202) == 1
assert policy_model("gfx906", 7, 512, 7, 512, 101) == 1
assert policy_model("NVIDIA GeForce RTX 3080", 3, 256, 3, 512, 101) == 1
assert policy_model("NVIDIA GeForce RTX 3080", 7, 512, 7, 256, 202) == 1
assert policy_model("NVIDIA GeForce RTX 3080", 7, 1024, 7, 512, 202) == 1
assert policy_model("NVIDIA GeForce RTX 4070 Ti SUPER", 7, 512, 7, 512, 202) == 1

print(
    "ok: automatic double-wide policy is restricted to production-qualified "
    "gfx906 PFA3/PFA7 and RTX 3080 OpenCL PFA7 scope and preserves explicit TAIL_KERNELS"
)


def production_policy_model(device, radix, width, middle, height, variant,
                            fft_type="FFT3161", gf31=True, gf61=True,
                            explicit=None):
    config = dict(explicit or {})

    pfa3 = (
        fft_type == "FFT3161"
        and gf31 and gf61
        and (radix, width, middle, height, variant)
            == (3, 256, 3, 512, 101)
    )
    pfa7 = (
        fft_type == "FFT3161"
        and gf31 and gf61
        and (radix, width, middle, height, variant)
            == (7, 512, 7, 512, 202)
    )

    pfa3_auto = (
        pfa3
        and device.startswith("gfx906")
        and "TAIL_KERNELS" not in config
    )
    pfa7_auto = (
        pfa7
        and (
            device.startswith("gfx906")
            or device == "NVIDIA GeForce RTX 3080"
        )
        and "TAIL_KERNELS" not in config
    )

    if pfa3_auto:
        config["TAIL_KERNELS"] = "3"
        for key, value in (
            ("OUT_WG", "256"),
            ("OUT_SIZEX", "8"),
            ("MIDDLE_OUT_LDS_TRANSPOSE", "1"),
            ("LOADS", "1"),
            ("STORES", "1"),
        ):
            config.setdefault(key, value)
    elif pfa7_auto:
        config["TAIL_KERNELS"] = "3"
    else:
        config.setdefault("TAIL_KERNELS", "1")

    return config


qualified_pfa3 = production_policy_model(
    "gfx906:sramecc+:xnack-",
    3, 256, 3, 512, 101,
)

assert qualified_pfa3["TAIL_KERNELS"] == "3"
assert qualified_pfa3["OUT_WG"] == "256"
assert qualified_pfa3["OUT_SIZEX"] == "8"
assert qualified_pfa3["MIDDLE_OUT_LDS_TRANSPOSE"] == "1"
assert qualified_pfa3["LOADS"] == "1"
assert qualified_pfa3["STORES"] == "1"

qualified_pfa7 = production_policy_model(
    "gfx906:sramecc+:xnack-",
    7, 512, 7, 512, 202,
)
assert qualified_pfa7 == {"TAIL_KERNELS": "3"}

qualified_rtx3080_pfa7 = production_policy_model(
    "NVIDIA GeForce RTX 3080",
    7, 512, 7, 512, 202,
)
assert qualified_rtx3080_pfa7 == {"TAIL_KERNELS": "3"}

# Explicit per-key settings retain priority while the remaining PFA3 defaults
# are still supplied.
explicit_layout = production_policy_model(
    "gfx906",
    3, 256, 3, 512, 101,
    explicit={
        "OUT_WG": "128",
        "OUT_SIZEX": "16",
        "LOADS": "0",
        "STORES": "0",
    },
)

assert explicit_layout["TAIL_KERNELS"] == "3"
assert explicit_layout["OUT_WG"] == "128"
assert explicit_layout["OUT_SIZEX"] == "16"
assert explicit_layout["LOADS"] == "0"
assert explicit_layout["STORES"] == "0"
assert explicit_layout["MIDDLE_OUT_LDS_TRANSPOSE"] == "1"

# An explicit tail selection disables either automatic policy.
assert production_policy_model(
    "gfx906", 3, 256, 3, 512, 101,
    explicit={"TAIL_KERNELS": "1"},
) == {"TAIL_KERNELS": "1"}
assert production_policy_model(
    "gfx906", 7, 512, 7, 512, 202,
    explicit={"TAIL_KERNELS": "1"},
) == {"TAIL_KERNELS": "1"}
assert production_policy_model(
    "NVIDIA GeForce RTX 3080", 7, 512, 7, 512, 202,
    explicit={"TAIL_KERNELS": "1"},
) == {"TAIL_KERNELS": "1"}

# Negative scope keeps the generic safe tail default and receives no companion
# policy.
for cfg in (
    production_policy_model("gfx1100", 3, 256, 3, 512, 101),
    production_policy_model("gfx1100", 7, 512, 7, 512, 202),
    production_policy_model("gfx906", 3, 256, 3, 256, 101),
    production_policy_model("gfx906", 7, 512, 7, 256, 202),
    production_policy_model("gfx906", 7, 1024, 7, 512, 202),
    production_policy_model("gfx906", 7, 512, 8, 512, 202),
    production_policy_model("NVIDIA GeForce RTX 3080", 3, 256, 3, 512, 101),
    production_policy_model("NVIDIA GeForce RTX 3080", 7, 512, 7, 256, 202),
    production_policy_model("NVIDIA GeForce RTX 4070 Ti SUPER", 7, 512, 7, 512, 202),
):
    assert cfg == {"TAIL_KERNELS": "1"}

print(
    "ok: validated gfx906 PFA3/PFA7 and RTX 3080 OpenCL PFA7 TK3 defaults are "
    "exact-scope and preserve explicit configuration"
)
