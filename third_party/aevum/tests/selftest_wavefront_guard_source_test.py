#!/usr/bin/env python3
"""selftest.cl's inline-asm timing kernels must not pick their assembly from a guessed wavefront size.

The wave32 (RDNA) and wave64 (GCN/CDNA) variants differ (v_add_nc_i32 and a null carry-out against v_add_i32 and vcc)
and neither assembles on the other's hardware.  base.cl falls back to WAVEFRONT=64 on AMD when the host passes none, so a
host query that returns 0 on a wave32 device used to select the wave64 assembly.  The wavefront size now comes from the
compiler (its wavefront-size macro, then its architecture family macros) or from a host value that really is 32 or 64,
and the asm tests are left out when it is unknown.

Preprocesses selftest.cl for each situation and checks which assembly is left.  Skipped when no C preprocessor is found.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CL_DIR = ROOT / "src" / "cl"

cpp = shutil.which("clang") or shutil.which("gcc") or shutil.which("cpp")
if cpp is None:
    print("selftest wavefront guard test skipped: no C preprocessor found")
    sys.exit(0)

BASE = ["-DNW=4", "-DNH=4", "-DWordSize=8", "-DFFT_TYPE=1", "-DFFT_VARIANT=101", "-DWIDTH=256", "-DSMALL_HEIGHT=256",
        "-DMIDDLE=2", "-DEXP=1362763", "-DCARRY_LEN=8", "-DMAXBPW=4054"]

# (v_add_i32 spelling, v_mad_u64_u32 carry-out) of the assembly that is left, or NONE
WAVE32 = ("v_add_nc_i32", "null")          # RDNA, wave32
WAVE64 = ("v_add_i32", "vcc")              # GCN / CDNA, wave64
RDNA_WAVE64 = ("v_add_nc_i32", "vcc")      # RDNA in wave64 mode
NONE = "none"


def variant(defs):
    cmd = [cpp, "-E", "-P", "-x", "c", "-I", str(CL_DIR)] + BASE + defs + [str(CL_DIR / "selftest.cl")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"preprocessing failed for {defs}:\n{r.stderr}")
        sys.exit(1)
    out = r.stdout
    if '__asm("v_mad_u64_u32' not in out:
        return NONE
    add = "v_add_nc_i32" if '__asm("v_add_nc_i32"' in out.replace(" ", "") or '"v_add_nc_i32"' in out else "v_add_i32"
    carry = "null" if '" "null"' in out else ("vcc" if '" "vcc"' in out else "?")
    return (add, carry)


cases = [
    # (defines, expected)
    (["-DAMDGPU=1", "-DWAVEFRONT=32"], WAVE32),
    (["-DAMDGPU=1", "-DWAVEFRONT=64"], WAVE64),
    # The host query returned 0 or nothing: base.cl would guess 64 on AMD.  Unknown means no asm, not wave64 asm.
    (["-DAMDGPU=1", "-DWAVEFRONT=0"], NONE),
    (["-DAMDGPU=1"], NONE),
    (["-DAMDGPU=1", "-DWAVEFRONT=48"], NONE),
    # The compiler's architecture family tells the wavefront size when the host did not.
    (["-DAMDGPU=1", "-DWAVEFRONT=0", "-D__GFX10__=1"], WAVE32),
    (["-DAMDGPU=1", "-D__GFX11__=1"], WAVE32),
    (["-DAMDGPU=1", "-D__GFX12__=1"], WAVE32),
    (["-DAMDGPU=1", "-D__GFX9__=1"], WAVE64),
    (["-DAMDGPU=1", "-DWAVEFRONT=0", "-D__GFX9__=1"], WAVE64),
    # RDNA in wave64 mode: gfx10 spelling of the add, the 64-bit vcc carry-out.
    (["-DAMDGPU=1", "-DWAVEFRONT=64", "-D__GFX10__=1"], RDNA_WAVE64),
    # The compiler's own wavefront-size macro wins over the host's value.
    (["-DAMDGPU=1", "-DWAVEFRONT=64", "-D__AMDGCN_WAVEFRONT_SIZE__=32"], WAVE32),
    (["-DAMDGPU=1", "-DWAVEFRONT=32", "-D__AMDGCN_WAVEFRONT_SIZE=64"], WAVE64),
    # Not an AMD device: no asm whatever the wavefront.
    (["-DWAVEFRONT=32"], NONE),
    (["-DWAVEFRONT=64", "-DNVIDIAGPU=1", "-DCC=800"], NONE),
]

failed = 0
for defs, expected in cases:
    got = variant(defs)
    if got != expected:
        print(f"FAIL {' '.join(defs)}: selftest.cl keeps {got} assembly, expected {expected}")
        failed += 1

src = (CL_DIR / "selftest.cl").read_text()
if "WAVEFRONT != 64" in src:
    print("FAIL selftest.cl still decides on WAVEFRONT != 64")
    failed += 1

# With a clang that has the AMDGPU backend, assemble selftest.cl for real hardware targets: a wrong asm variant is an
# assembler error.  Each target is built with a host WAVEFRONT of 0 (the failing query), and with the right value.
clang = shutil.which("clang")


def errs(text):
    return "\n".join(l for l in text.splitlines() if "error" in l)[:600]


def amdgpu_compile(mcpu, extra):
    cmd = [clang, "--target=amdgcn-amd-amdhsa", f"-mcpu={mcpu}", "-nogpulib", "-x", "cl", "-cl-std=CL2.0",
           "-Dcl_khr_fp64=1", "-I", str(CL_DIR), "-DAMDGPU=1"] + BASE + extra + \
          ["-c", str(CL_DIR / "selftest.cl"), "-o", "/dev/null"]
    return subprocess.run(cmd, capture_output=True, text=True)


if clang and amdgpu_compile("gfx906", ["-DWAVEFRONT=64"]).returncode == 0:
    for mcpu, extra in (("gfx906", []), ("gfx906", ["-DWAVEFRONT=0"]), ("gfx906", ["-DWAVEFRONT=64"]),
                        ("gfx1030", []), ("gfx1030", ["-DWAVEFRONT=0"]), ("gfx1030", ["-DWAVEFRONT=32"]),
                        ("gfx1030", ["-mwavefrontsize64", "-DWAVEFRONT=64"]),
                        ("gfx1100", ["-DWAVEFRONT=0"]), ("gfx1100", ["-DWAVEFRONT=32"])):
        r = amdgpu_compile(mcpu, extra)
        if r.returncode != 0:
            print(f"FAIL assembling selftest.cl for {mcpu} {' '.join(extra)}:\n{errs(r.stderr)}")
            failed += 1
else:
    print("clang without the AMDGPU backend: assembler checks skipped")

if failed:
    sys.exit(1)
print("selftest wavefront guard source test: PASS")
