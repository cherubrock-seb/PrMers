#!/usr/bin/env python3
"""Compile Aevum's kernels with NVRTC the way the CUDA build does, on a machine with no GPU.

The CUDA build compiles its kernels (src/cl/*.cl through opencl_compat.cuh) only at run time, on the user's GPU, so a kernel
that OpenCL accepts and NVRTC rejects is invisible to every CI build.  This runs the real CUDA build against a fake CUDA driver
(fakecuda.cpp: one nVidia GPU of the chosen compute capability, no hardware behind it): aevum picks its kernels and their -D
options as it would on that GPU and the real NVRTC compiles them.  Kernels are not run.

usage: nvrtc-check.py <CUDA build of aevum> [--cc 89,min] [-j N] [--list] [--only REGEX]
  --cc    compute capabilities to compile for: numbers (89 = sm_89), "min" and "max" for the oldest and newest this NVRTC knows
  --list  print the runs that would be made and exit
Exit status 1 if any kernel fails to compile (the NVRTC diagnostics are printed, grouped), 2 if there is nothing to check.

The runs are a small set of FFTs, taken from the list aevum prints (-h), that together cover every FFT type, width, middle and
height, every variant digit, INPLACE=0/1/2, both carry widths and the TAIL_KERNELS settings, for each number type (FP64, FP32,
GF31, GF61) a type contains.  Those are PRP runs; for every FFT type there is also a small LL test and a proof verification, which
load the kernels the PRP loop does not (carry and carryFused for LL, tailMul with MUL_LOW, the *Zero and Hin kernels).

aevum must be able to load a libcuda.so.1 (the CUDA toolkit's driver stub is enough): the real NVRTC loads one itself and has to
find the driver's, not the fake, so the fake is LD_PRELOADed over aevum's own calls instead of replacing the library.
"""
import argparse, concurrent.futures as cf, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = ""
# The number types of each FFT type (FFTConfig.h FFT_TYPES): D = FP64, S = FP32, t = GF31, g = GF61
PARTS = {0: "D", 1: "tg", 2: "Sg", 4: "Stg"}
# Types that -h does not list: reachable only through an explicit -fft prefix.  Take a few shapes each
EXTRA_TYPES = ()  # Aevum hidden types are seeded by validated explicit specs below
EXTRA_SHAPES = ()
AEVUM_EXTRA_SPECS = ('4:256:2:256', '4:256:8:256', '4:512:8:512', '4:256:16:1K', '4:512:7:512', '4:512:9:512', '4:1K:8:1K', '0:256:2:256', '0:256:4:256', '0:256:2:512', '0:512:2:256', '0:1K:2:256', '0:256:8:256', '0:256:4:512', '0:512:4:256', '0:512:2:512', '0:1K:4:256', '0:1K:2:512', '0:256:16:256', '0:256:8:512', '0:512:8:256', '0:512:4:512', '0:512:2:1K', '0:1K:8:256', '0:1K:4:512', '0:1K:2:1K', '0:256:16:512', '0:512:16:256', '0:512:8:512', '0:512:4:1K', '0:4K:2:256', '0:1K:16:256', '0:1K:8:512', '0:1K:4:1K', '0:512:16:512', '0:512:8:1K', '0:4K:4:256', '0:4K:2:512', '0:1K:16:512', '0:1K:8:1K', '0:512:16:1K', '0:4K:8:256', '0:4K:4:512', '0:4K:2:1K', '0:1K:16:1K', '0:4K:16:256', '0:4K:8:512', '0:4K:4:1K', '0:4K:16:512', '0:4K:8:1K', '0:4K:16:1K', '2:256:2:256', '2:256:4:256', '2:256:2:512', '2:512:2:256', '2:1K:2:256', '2:256:8:256', '2:256:4:512', '2:512:4:256', '2:512:2:512', '2:1K:4:256', '2:1K:2:512', '2:256:16:256', '2:256:8:512', '2:512:8:256', '2:512:4:512', '2:512:2:1K', '2:1K:8:256', '2:1K:4:512', '2:1K:2:1K', '2:256:16:512', '2:512:16:256', '2:512:8:512', '2:512:4:1K', '2:4K:2:256', '2:1K:16:256', '2:1K:8:512', '2:1K:4:1K', '2:512:16:512', '2:512:8:1K', '2:4K:4:256', '2:4K:2:512', '2:1K:16:512', '2:1K:8:1K', '2:512:16:1K', '2:4K:8:256', '2:4K:4:512', '2:4K:2:1K', '2:1K:16:1K', '2:4K:16:256', '2:4K:8:512', '2:4K:4:1K', '2:4K:16:512', '2:4K:8:1K', '2:4K:16:1K')
# Largest FFT (words) to compile for: the host builds its tables on the CPU, and the kernel code does not depend on the size
MAX_N = 1 << 24
INPLACE = (0, 1, 2)
EFRACS = (0.97, 0.6)            # of the largest exponent for the FFT: the wide and the narrow carry
TAIL_KERNELS = (None, 0, 1, 3)  # None: the default


def writeSyntheticProof(path):
    E = 786433
    power = 1
    nbytes = (E - 1) // 8 + 1
    header = (
        "PRP PROOF\n"
        "VERSION=2\n"
        "HASHSIZE=64\n"
        f"POWER={power}\n"
        f"NUMBER=M{E}\n"
    ).encode()
    B = bytearray(nbytes)
    B[0] = 3
    with open(path, "wb") as f:
        f.write(header)
        f.write(B)
        f.write(bytes(nbytes))

def size(s):
    return int(s.rstrip("K")) * (1024 if s.endswith("K") else 1)

def run(aevum, args, env=None, **kw):
    p = subprocess.run([aevum] + args, capture_output=True, text=True, env=env, **kw)
    return p.stdout + p.stderr, p.returncode

def plan(aevum, env):
    """[(mode, spec, INPLACE, TAIL_KERNELS, exponent)]: the LL and verify runs, and a greedy cover of the PRP features"""
    out, _ = run(aevum, ["-h"], env)
    specs = sorted({s for line in out.splitlines() if re.match(r"^\s*[\d.]+[MK]?\s+[\d.]+M\s+[\d.]+\s", line) for s in line.split()[3].split(",")})
    specs += list(AEVUM_EXTRA_SPECS)
    # -info gives each spec's variants and the exponent each takes (in M, one decimal)
    info, _ = run(aevum, ["-info", ",".join(specs)], env)
    shapes = []     # (type, W, M, H, variant, words, max exponent)
    for m in re.finditer(r"^\S+ \S+\s+(\S+) \| ([\d.]+) \|\s+([\d.]+)$", info, re.M):
        f = m.group(1).split(":")
        w, mid, h, var = f[-4:]
        shapes.append((int(f[0]) if len(f) == 5 else 0, w, mid, h, var, 2 * size(w) * int(mid) * size(h), float(m.group(3)) * 1e6))
    chosen = []
    for typ in PARTS:
        # A small LL test and a proof verification of every type.  Verification takes the FFT as given only if it can handle the
        # proof's exponent, so only a tiny FFT will do.  Verifications at TAIL_KERNELS 0 and 3, the single-wide one-kernel and the
        # double-wide two-kernel tails: the PRP runs below build tailMul and tailSquare for each, but only a verification builds
        # tailMul with MUL_LOW and the *Zero kernels
        small = [s for s in shapes if s[:4] == (typ, "256", "2", "256")]
        if small:
            chosen.append(("ll", f"{typ}:256:2:256:{small[-1][4]}", 1, None, int(small[-1][6] * 0.9)))
            chosen += [("verify", f"{typ}:256:2:256", 1, tk, 786433) for tk in (0, 3)]
    items = []
    for typ, w, mid, h, var, n, maxexp in shapes:
        if n > MAX_N: continue
        spec = f"{typ}:{w}:{mid}:{h}:{var}"
        for ip in INPLACE:
            for ef in EFRACS:
                for tk in TAIL_KERNELS:
                    feats = {(typ, "carry", ef)}
                    for p in PARTS[typ]:
                        feats |= {(p, "W", w, var[0]), (p, "M", mid), (p, "H", h, var[2]), (p, "m", var[1]), (p, "IP", ip), (p, "TK", tk)}
                    items.append((n, ("prp", spec, ip, tk, int(maxexp * ef)), feats))
    want = set().union(*(i[2] for i in items)) if items else set()
    while want:
        # most new features per word of FFT (the host's CPU work); ties go to the smaller FFT
        best = max(items, key=lambda i: (len(i[2] & want) / (1 + i[0] / MAX_N), -i[0]))
        chosen.append(best[1])
        want -= best[2]
    return chosen

def runtimeSpec(spec):
    """Render an Aevum runtime FFT spec from the planner's shape spec."""
    parts = spec.split(":")
    if len(parts) < 5:
        return spec

    try:
        typ = int(parts[0])
        middle = int(parts[2])
    except ValueError:
        return spec

    if typ == 1 and middle in (3, 7, 9):
        return f"pfa{middle}:{spec}"

    if typ == 4 and middle in (7, 9):
        return f"pfa{middle}:{spec}"

    return spec

def command(mode, spec, ip, tk, exp):
    use = f"INPLACE={ip}" + (f",TAIL_KERNELS={tk}" if tk is not None else "")
    what = {"prp": ["-prp", str(exp)], "ll": ["-ll", str(exp)], "verify": ["-verify", PROOF]}[mode]
    # The kernels are all built by the first few iterations.  The fake GPU computes nothing, so what the run reports means nothing
    return ["-v", "1", "-device", "0", "-fft", runtimeSpec(spec), "-iters", "1000", "-log", "1000", "-use", use] + what

def check(aevum, env, tmp, item):
    d = tempfile.mkdtemp(dir=tmp)
    try:
        log, rc = run(aevum, ["-dir", d] + command(*item), dict(env, FAKECUDA_RES="4" if item[0] == "ll" else "3"), timeout=900)
    except subprocess.TimeoutExpired:
        log, rc = "timeout", 1
    shutil.rmtree(d, ignore_errors=True)
    # "Loaded" (-v 1) says kernels were built; rc < 0 is a crash.  A run that ends in an error of its own is fine: the fake GPU computes nothing
    return rc >= 0 and "NVRTC COMPILE FAILED" not in log and "Can't compile" not in log and "Loaded" in log, log

def errors(log):
    """The NVRTC diagnostics of a failed run: [(error line, the source line and caret under it)], else its last lines"""
    lines = log.splitlines()
    errs = [(l, "\n".join(lines[i + 1:i + 5])) for i, l in enumerate(lines) if re.search(r"\(\d+\): error", l)]
    return errs or [("\n".join(lines[-6:]), "")]

def main():
    global PROOF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aevum"); ap.add_argument("--cc", default="89,min"); ap.add_argument("-j", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--list", action="store_true"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    aevum = os.path.abspath(a.aevum)
    with tempfile.TemporaryDirectory(prefix="nvrtc-check-") as tmp:
        PROOF = os.path.join(tmp, "aevum-synthetic-786433-p1.proof")
        writeSyntheticProof(PROOF)
        fake = os.path.join(tmp, "fakecuda.so")
        subprocess.run([os.environ.get("CXX", "c++"), "-O1", "-shared", "-fPIC", "-o", fake, os.path.join(HERE, "fakecuda.cpp"), "-ldl"], check=True)
        env = dict(os.environ, LD_PRELOAD=fake, FAKECUDA_CC="89", CUDA_VISIBLE_DEVICES="", AEVUM_EXPERIMENT_FFT3261="1")
        items = plan(aevum, env)
        # A listing format that changed must not turn into a check of nothing
        fft_types = {int(i[1].split(":")[0]) if i[1].count(":") == 4 else 0 for i in items}
        if fft_types != set(PARTS) or {i[0] for i in items} != {"prp", "ll", "verify"}:
            print(f"::error::the plan has FFT types {sorted(fft_types)}, expected {sorted(PARTS)}, and runs of {sorted({i[0] for i in items})}: did the output of 'aevum -h' or '-info' change?")
            return 2
        items = [i for i in items if re.search(a.only, i[1])]
        if a.list:
            for i in items: print(*i)
            return 0
        print(f"{len(items)} runs: {sum(i[0] == 'prp' for i in items)} PRP, {sum(i[0] == 'll' for i in items)} LL, {sum(i[0] == 'verify' for i in items)} proof verifications", flush=True)
        bad = 0
        for cc in a.cc.split(","):
            env["FAKECUDA_CC"] = cc
            groups = {}     # NVRTC error -> (context, [runs that hit it])
            arch = "sm_" + cc
            nbad = 0
            with cf.ThreadPoolExecutor(a.j) as ex:
                for item, (ok, log) in zip(items, ex.map(lambda i: check(aevum, env, tmp, i), items)):
                    if m := re.search(r"Fake NVIDIA GPU \((sm_\d+)\)", log): arch = m.group(1)    # "min" and "max" are numbers only to the fake
                    if ok: continue
                    nbad += 1
                    for err, ctx in errors(log):
                        groups.setdefault(err, (ctx, []))[1].append(" ".join(command(*item)[4:]))
            for err, (ctx, runs) in groups.items():
                print(f"::error::Aevum no-GPU kernel compile/startup check failed for {arch} in {len(runs)} runs")
                print(f"{err}\n{ctx}\naevum options of the runs:\n  " + "\n  ".join(sorted(set(runs))[:20]) + ("\n  ..." if len(set(runs)) > 20 else ""), flush=True)
            print(f"{arch}: {len(items)} runs, {nbad} failed", flush=True)
            bad += nbad
        return 1 if bad else 0

sys.exit(main())
