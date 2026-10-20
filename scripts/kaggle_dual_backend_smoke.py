#!/usr/bin/env python3
"""Credential-free Kaggle T4 x2 validation of unified Aevum CUDA/OpenCL.

This is a short smoke/throughput probe; not a completed PRP, LL, P-1 or ECM
correctness campaign. Runs are isolated and interrupted after a bounded time.
No GMNet/Turso credentials, database writes or user campaign jobs.
"""
import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import time

def run(cmd, cwd, timeout=900, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout, check=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=int, default=45)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    devices = [int(x) for x in args.devices.split(",")]
    smi = run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], root, 20)
    if smi.returncode != 0:
        raise SystemExit("NVIDIA GPU unavailable:\n" + smi.stdout[-1500:])
    print("NVIDIA devices:\n" + smi.stdout, flush=True)
    uuid_map = {}
    for line in smi.stdout.splitlines():
        fields = [x.strip() for x in line.split(",", 2)]
        if len(fields) == 3:
            uuid_map[int(fields[0])] = fields[1]
    if not set(devices).issubset(uuid_map):
        raise SystemExit("Requested GPU index missing from nvidia-smi")

    if not args.no_build:
        jobs = min(4, os.cpu_count() or 2)
        print("Compiling PrMers in Kaggle for ABI compatibility...", flush=True)
        b = run(["make", f"-j{jobs}", "KERNEL_PATH=./kernels/"], root, 1800)
        if b.returncode:
            raise SystemExit("OpenCL source build failed:\n" + b.stdout[-9000:])
        print("Building optional CUDA engine from same source...", flush=True)
        b = run(["make", "aevum-dual"], root, 1800)
        if b.returncode:
            raise SystemExit("CUDA source build failed (check CUDA SDK/NVRTC):\n" + b.stdout[-9000:])
    cuda = root / "third_party/aevum/build-engine/libaevum_engine_cuda.so"
    if not cuda.is_file():
        raise SystemExit("CUDA plugin absent. Did you run make aevum-dual?")
    for path in (root / "prmers", cuda):
        dep = run(["ldd", str(path)], root, 15)
        if dep.returncode or "not found" in dep.stdout:
            raise SystemExit("Unresolved runtime dependency: " + str(path) + "\n" + dep.stdout[-4000:])
    print("Both plugins built; native CUDA linkage available.", flush=True)

    base = ["-aevum", "-proof", "0", "--noask"]
    modes = {
        "PRP": ["3704053", "-prp"] + base,
        "LL": ["3704053", "-ll", "-aevum", "--noask"],
        "P-1": ["3704053", "-pm1", "-b1", "100", "-b2", "100", "-aevum", "--noask"],
        "ECM": ["3704053", "-ecm", "-b1", "100", "-b2", "100", "-K", "1", "-aevum", "--noask"],
        "GM-P-1": ["3704053", "-gm-pm1", "-b1", "2", "-b2", "2", "-gm-family", "GM",
                   "-gm-sieve", "0", "-aevum"],
    }
    all_ok = True
    for gpu in devices:
        for mode, params in modes.items():
            for backend in ("opencl", "auto"):
                outdir = root / ".kaggle-smoke" / f"gpu{gpu}-{mode}-{backend}"
                outdir.mkdir(parents=True, exist_ok=True)
                kernels = outdir / "kernels"
                if not kernels.exists():
                    kernels.symlink_to(root / "kernels", target_is_directory=True)
                env = dict(os.environ, AEVUM_BACKEND=backend)
                env.pop("AEVUM_ENGINE_LIB", None)
                cmd = [str(root / "prmers")] + params + ["-d", str(gpu)]
                if mode == "GM-P-1":
                    cmd += ["-f", str(outdir)]
                print(f"\n--- gpu={gpu} mode={mode} backend={backend} ---", flush=True)
                proc = subprocess.Popen(cmd, cwd=outdir, env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                # A short run including JIT compilation. Do not convert timeout
                # into success; report whether arbitration was actually seen.
                try:
                    stdout, _ = proc.communicate(timeout=args.max_seconds)
                    outcome = "completed"
                except subprocess.TimeoutExpired:
                    proc.send_signal(signal.SIGINT)
                    try:
                        stdout, _ = proc.communicate(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        stdout, _ = proc.communicate()
                    outcome = "interrupted-smoke"
                interesting = [line.strip() for line in stdout.splitlines()
                               if "[Backend Aevum]" in line or "CUDA/OpenCL" in line
                               or "IPS:" in line or "[Gerbicz" in line]
                for line in interesting[:16]:
                    print(line[:500])
                is_active = any("adapter active" in line for line in interesting)
                if not is_active:
                    print("WARN: Aevum not confirmed; exit=", proc.returncode,
                          "tail=", stdout[-1200:])
                    all_ok = False
                elif backend == "auto" and not any("CUDA/OpenCL auto:" in line for line in interesting):
                    print("WARN: CUDA/OpenCL comparison not confirmed for", mode)
                    all_ok = False
                else:
                    print("SMOKE:", outcome, "exit=", proc.returncode)
    print("KAGGLE_DUAL_BACKEND_SMOKE:", "PASS" if all_ok else "INCOMPLETE")
    raise SystemExit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
