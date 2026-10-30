#!/usr/bin/env python3
"""carryFused's hand-offs between workgroups must be device-scope release/acquire pairs where the L1 caches are not coherent.

The carry shuttle (and, in place, the lines-read flag) is written and read with plain stores and loads, ordered only by
write_mem_fence/read_mem_fence/mem_fence (work-group scope) and a relaxed ready-flag load.  On Intel GPUs a stale carry can be read.
CS_FENCE adds device-scope fences to every hand-off and CS_ATOMIC makes the shuttle accesses device-scope atomics; both are on by
default except on AMD and nVidia (gpuowl 18ca429, 21f0dba, 994acfc), and off under OpenCL C 1.2 (Apple), which lacks them.

1. Text: every carryFused variant calls csDeviceRelease() after both hand-off write fences and csDeviceAcquire() after both read fences,
   and the lines-read flag helpers do the same.
2. clang: every FFT type compiles with CS on and off, in and out of place, OLD_FENCE 0 and 1, at OpenCL C 2.0, 3.0 and 1.2 (Apple).
3. clang -emit-llvm: the default build for a non-AMD, non-nVidia device has the device-scope fences and shuttle atomics; AMD, nVidia,
   explicit CS_FENCE=0/CS_ATOMIC=0 and OpenCL C 1.2 have none of them.
"""
import concurrent.futures
import pathlib
import re
import shutil
import subprocess
import sys

CL_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "cl"
SRC = CL_DIR / "carryfused.cl"
fail = []

# ---- 1. text ----
lines = SRC.read_text().split("\n")
kernels = [i for i, l in enumerate(lines) if re.match(r"KERNEL(_CAP)?\(", l) and "carryFused(" in l]
bounds = kernels + [len(lines)]


def next_code(i):
    for l in lines[i + 1:]:
        s = l.strip()
        if s and not s.startswith("//"):
            return s
    return ""


for k, start in enumerate(kernels):
    body = range(start, bounds[k + 1])
    wf = [i for i in body if lines[i].strip() == "write_mem_fence(CLK_GLOBAL_MEM_FENCE);"]
    rf = [i for i in body if lines[i].strip() in ("read_mem_fence(CLK_GLOBAL_MEM_FENCE);", "mem_fence(CLK_GLOBAL_MEM_FENCE);")]
    if len(wf) != 2 or len(rf) != 2:
        fail.append(f"carryfused.cl:{start + 1}: expected 2 write and 2 read hand-off fences, found {len(wf)} and {len(rf)}")
    for i in wf:
        if next_code(i) != "csDeviceRelease();":
            fail.append(f"carryfused.cl:{i + 1}: hand-off write fence not followed by csDeviceRelease()")
    for i in rf:
        if next_code(i) != "csDeviceAcquire();":
            fail.append(f"carryfused.cl:{i + 1}: hand-off read fence not followed by csDeviceAcquire()")

text = "\n".join(lines)
m = re.search(r"void signalLinesRead\(P\(u32\) ready\) \{(.*?)\n\}", text, re.S)
if not m or "csDeviceRelease();" not in m.group(1) or m.group(1).find("csDeviceRelease();") > m.group(1).find("bar();"):
    fail.append("signalLinesRead: no csDeviceRelease() before its bar()")
m = re.search(r"void waitLinesRead\(P\(u32\) ready\) \{(.*?)\n\}", text, re.S)
if not m or not m.group(1).rstrip().split("\n")[-1].split("//")[0].strip() == "csDeviceAcquire();":
    fail.append("waitLinesRead: no csDeviceAcquire() after its bar()")

# ---- 2 and 3. clang ----
TYPES = {  # name: (FFT_TYPE, FP64, FP32, GF31, GF61, WordSize, carry64 values)
    "FFT64": (0, 1, 0, 0, 0, 4, (0, 1)),
    "FFT3161": (1, 0, 0, 1, 1, 8, (1,)),
    "FFT3261": (2, 0, 1, 0, 1, 8, (1,)),
    "FFT61": (3, 0, 0, 0, 1, 4, (0, 1)),
    "FFT323161": (4, 0, 1, 1, 1, 8, (1,)),
    "FFT3231": (50, 0, 1, 1, 0, 4, (0,)),
    "FFT6431": (51, 1, 0, 1, 0, 8, (1,)),
    "FFT31": (52, 0, 0, 1, 0, 4, (0,)),
    "FFT32": (53, 0, 1, 0, 0, 4, (0,)),
}
COMMON = ["-Dcl_khr_fp64=1", "-I", str(CL_DIR), "-DNO_ASM=1",
          "-DEXP=1700023u", "-DWIDTH=512u", "-DSMALL_HEIGHT=256u", "-DMIDDLE=2u", "-DNW=8u", "-DNH=4u", "-DCARRY_LEN=8u",
          "-DPFA_RADIX=0u", "-DMAXBPW=3998u", "-DFFT_VARIANT=202u",
          "-DDISTGF31=0", "-DDISTWTRIGGF31=0", "-DDISTMTRIGGF31=0", "-DDISTHTRIGGF31=0",
          "-DDISTGF61=0", "-DDISTWTRIGGF61=0", "-DDISTMTRIGGF61=0", "-DDISTHTRIGGF61=0",
          "-DFRAC_BPW_HI=375134435u", "-DFRAC_BPW_LO=2386092941u", "-DWEIGHT_STEP=0.5", "-DIWEIGHT_STEP=0.5", "-DTRIG_SCALE=1",
          "-DTAILT=U2(1.0,0.0)", "-DTAILTGF31=U2(269176336u,500380354u)", "-DTAILTGF61=U2(807738046998073027ul,30095103403839256ul)",
          "-DTRIG_SIN={1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}", "-DTRIG_COS={1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"]
STDS = {
    "CL2.0": ["-cl-std=CL2.0", "-Xclang", "-cl-ext=+cl_khr_fp64"],
    "CL3.0": ["-cl-std=CL3.0", "--target=spir64", "-Xclang", "-cl-ext=+cl_khr_fp64,+__opencl_c_fp64",
              "-D__opencl_c_atomic_scope_device=1"],
    "CL1.2-apple": ["-cl-std=CL1.2", "-Xclang", "-cl-ext=+cl_khr_fp64",
                    "-DAEVUM_APPLE_OPENCL12=1"],
}
VENDORS = {"other": [], "amd": ["-DAMDGPU=1", "-DWAVEFRONT=64"], "nvidia": ["-DNVIDIAGPU=1", "-DCC=860"]}
CS = {"default": [], "off": ["-DCS_FENCE=0", "-DCS_ATOMIC=0"], "on": ["-DCS_FENCE=1", "-DCS_ATOMIC=1"],
      "fence-only": ["-DCS_FENCE=1", "-DCS_ATOMIC=0"], "atomic-only": ["-DCS_FENCE=0", "-DCS_ATOMIC=1"]}


def type_defs(name, carry64):
    t, fp64, fp32, g31, g61, ws, _ = TYPES[name]
    return [f"-DFFT_TYPE={t}", f"-DFFT_FP64={fp64}", f"-DFFT_FP32={fp32}", f"-DNTT_GF31={g31}", f"-DNTT_GF61={g61}",
            f"-DWordSize={ws}"] + (["-DCARRY64=1"] if carry64 else [])


clang = shutil.which("clang")
if clang:
    def run(args):
        p = subprocess.run([clang, "-x", "cl", *args, str(SRC)], capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

    jobs = []
    for name, spec in TYPES.items():
        for carry64 in spec[6]:
            for std, sargs in STDS.items():
                for vendor, vargs in VENDORS.items():
                    for cs, cargs in CS.items():
                        for extra in (["-DINPLACE=0"], ["-DINPLACE=1"], ["-DINPLACE=1", "-DOLD_FENCE=0"]):
                            if "-DOLD_FENCE=0" in extra and vendor != "amd":
                                continue   # OLD_FENCE=0 is refused off AMD without PTX (kernel_config_guards_test.py)
                            if std != "CL2.0" and cs not in ("default", "on"):
                                continue
                            label = f"{name} CARRY64={carry64} {std} {vendor} CS={cs} {' '.join(extra)}"
                            jobs.append((label, sargs + ["-fsyntax-only"] + COMMON + type_defs(name, carry64) + vargs + cargs + extra))
    with concurrent.futures.ThreadPoolExecutor() as ex:
        for (label, _), (rc, _, err) in zip(jobs, ex.map(lambda j: run(j[1]), jobs)):
            if rc != 0:
                fail.append(f"compile failed: {label}\n" + "\n".join(l for l in err.splitlines() if "error" in l)[:400])

    # IR: which device-scope operations the carryFused build contains
    FENCE = "atomic_work_item_fence"
    ATOMIC = re.compile(r"atomic_(store|load)_explicit")

    def ir(std, vendor, cs, name="FFT3161", carry64=1):
        rc, out, err = run(STDS[std] + ["-target", "spir64", "-emit-llvm", "-S", "-O0", "-o", "-", "-DINPLACE=1"]
                           + COMMON + type_defs(name, carry64) + VENDORS[vendor] + CS[cs])
        if rc != 0:
            fail.append(f"IR build failed: {std} {vendor} {cs}\n{err[:300]}")
            return 0, 0
        calls = [l for l in out.splitlines() if "call " in l]
        fences = sum(FENCE in l for l in calls)
        # the ready flag load is atomic_load_explicit in every build; count the stores and loads beyond the ready-flag ones
        atomics = sum(bool(ATOMIC.search(l)) for l in calls)
        return fences, atomics

    _, base_atomics = ir("CL2.0", "amd", "default")
    for name in ("FFT3161", "FFT64", "FFT323161"):
        carry64 = TYPES[name][6][-1]
        f_def, a_def = ir("CL2.0", "other", "default", name, carry64)
        if f_def == 0:
            fail.append(f"{name}: default build for a non-AMD, non-nVidia device has no device-scope fence in carryFused")
        _, a_amd = ir("CL2.0", "amd", "default", name, carry64)
        if a_def <= a_amd:
            fail.append(f"{name}: default build for a non-AMD, non-nVidia device does not access the carry shuttle with atomics")
    for std, vendor, cs in (("CL2.0", "amd", "default"), ("CL2.0", "nvidia", "default"), ("CL2.0", "other", "off"),
                            ("CL1.2-apple", "other", "default"), ("CL1.2-apple", "other", "on")):
        f, a = ir(std, vendor, cs)
        if f:
            fail.append(f"{std} {vendor} CS={cs}: has {f} device-scope fences, expected none")
        if a > base_atomics:
            fail.append(f"{std} {vendor} CS={cs}: has carry shuttle atomics, expected none")
    print(f"carryFused CS_FENCE/CS_ATOMIC: {len(jobs)} clang builds")
else:
    print("SKIP (partial): clang not found, only the carryfused.cl text was checked")

if fail:
    print("\n".join(fail))
    sys.exit(1)
print(f"carryFused device-scope hand-offs: {len(kernels)} variants ok")
