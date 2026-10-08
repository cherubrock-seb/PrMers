#!/usr/bin/env python3
"""A stopped run exits with the same non-zero code in every mode.

Starts prmers in each mode on a long enough job (OpenCL device 0; PoCL's CPU device is enough),
waits until it is computing, sends SIGINT to the process, and checks:
  - the exit code is 1 (core/ExitCodes.hpp: kExitInterrupted),
  - the mode saved a checkpoint (where it has one),
  - with a worktodo queue: the interrupted entry is still first in worktodo.txt, was not archived
    and the next entry did not start.
Usage: interrupt_exit_code_test.py PRMERS_BINARY [DEVICE] [SIGNAL ...]   (default signals: INT)
"""
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

EXPECTED = 1

# name, argv (after -d DEVICE --noask), worktodo lines (or None), marker regex in the log,
# checkpoint globs (one must match), extra check on the working directory
CASES = [
    ("prp-marin-engine", ["110503", "-prp"], None, r"Progress", ["m_110503.ckpt"]),
    ("prp-legacy", ["110503", "-prp", "-marin"], None, r"Progress", ["110503prp.mers"]),
    ("ll", ["110503", "-ll"], None, r"Progress", ["llsafe_m_110503.ckpt"]),
    ("llsafe", ["110503", "-llsafe"], None, r"Progress", ["m_110503.ckpt"]),
    ("llsafe2", ["110503", "-llsafe2"], None, r"Progress", ["llsafe2_m_110503.ckpt"]),
    ("pm1-stage1", ["1000003", "-pm1", "-b1", "2000000", "-b2", "20000000"], None, r"Stage 1|Progress|B1", ["pm1_m_1000003.ckpt"]),
    ("pm1-legacy", ["1000003", "-pm1", "-b1", "2000000", "-b2", "20000000", "-marin"], None, r"Stage 1|Progress|B1", ["1000003pm12000000.mers"]),
    ("pm1-stage2", ["20011", "-pm1", "-b1", "2000", "-b2", "4000000000"], None, r"V-trace baby window", ["pm1_s2_*_m_20011.ckpt"]),
    ("ecm-montgomery", ["100003", "-ecm", "-b1", "200000", "-b2", "2000000", "-K", "50"], None, r"urve|Stage|B1", ["ecm_te_m_100003_c0.ckpt"]),
    ("ecm-edwards", ["100003", "-ecm", "-b1", "200000", "-b2", "2000000", "-K", "50", "-edwards"], None, r"urve|Stage|B1", ["ecm_te_m_100003_c0.ckpt"]),
    ("gm-proth", ["200003", "-gm-proth", "-gm-sieve", "0"], None, r"Progress|iter", ["gm_proth_p200003.ckpt"]),
    ("gm-prp", ["200003", "-gm-prp", "-gm-sieve", "0"], None, r"Progress|iter", ["gm_prp_p200003.ckpt"]),
    ("gm-pm1", ["1000003", "-gm-pm1", "-b1", "3000000", "-b2", "30000000", "-gm-sieve", "0"], None, r"Loaded carryFused|Stage 1 bits", []),
    ("gm-ecm", ["100003", "-gm-ecm", "-b1", "300000", "-b2", "3000000", "-K", "50", "-gm-sieve", "0"], None, r"Progress|ECM|curve", ["gm_ecm_p100003_c0_stage1.ckpt"]),
    ("gmtf-direct", ["-gm-tf", "40", "62", "-gm-family", "BOTH", "15317251"], None, r"progress", ["*.checkpoint"]),
    # These two look for a GPU-type OpenCL device themselves; skipped when there is none (e.g. PoCL CPU only)
    ("bench", ["-bench"], None, r"\[1/\d+\] TS=", []),
    ("memtest", ["-memtest"], None, r"addr W|Memtest|memtest", []),
    # worktodo-driven runs: the entry stays queued, nothing is archived, the next entry does not start
    ("worktodo-prp", [], ["PRP=N/A,1,2,110503,-1,70,0", "PRP=N/A,1,2,9941,-1,70,0"], r"Progress", ["m_110503.ckpt"]),
    ("worktodo-pm1", [], ["Pminus1=1,2,1000003,-1,2000000,20000000", "PRP=N/A,1,2,9941,-1,70,0"], r"Stage 1|Progress|B1", ["pm1_m_1000003.ckpt"]),
    ("worktodo-pm1-stage2", [], ["Pminus1=1,2,20011,-1,2000,4000000000", "PRP=N/A,1,2,9941,-1,70,0"], r"V-trace baby window", ["pm1_s2_*_m_20011.ckpt"]),
    ("worktodo-ecm", [], ["ECM2=1,2,100003,-1,200000,2000000,50", "PRP=N/A,1,2,9941,-1,70,0"], r"urve|Stage|B1", ["ecm_te_m_100003_c0.ckpt"]),
    ("worktodo-ll", [], ["Test=N/A,1,2,110503,-1,70,0", "PRP=N/A,1,2,9941,-1,70,0"], r"Progress", ["llunsafe_m_110503.ckpt"]),
    ("worktodo-gmtf", [], ["GMTF=15317251,40,62,BOTH", "PRP=N/A,1,2,9941,-1,70,0"], r"progress", ["*.checkpoint"]),
    # interrupted in the second family of a BOTH chain, and in the ECM phase
    ("gmchain-both-ecm", [], ["GMCHAIN=1279,2000,20000,1000,10000,8,0,0,factor,BOTH", "PRP=N/A,1,2,9941,-1,70,0"],
     r"Gaussian pair ECM factoring", ["gq_ecm_p1279_c0_stage*.ckpt"]),
    ("gmchain-both-gm-family", [], ["GMCHAIN=1279,2000,20000,1000,10000,8,0,0,factor,BOTH", "PRP=N/A,1,2,9941,-1,70,0"],
     r"target family  : GM", []),
]


def run_case(prmers, device, sig, case, root):
    name, args, worktodo, marker, ckpts = case
    work = os.path.join(root, f"{name}.{sig}")
    os.makedirs(work)
    os.symlink(os.path.join(os.path.dirname(os.path.abspath(prmers)), "kernels"), os.path.join(work, "kernels"))
    if worktodo:
        with open(os.path.join(work, "worktodo.txt"), "w") as f:
            f.write("\n".join(worktodo) + "\n")
    # The Aevum kernels take minutes to compile on PoCL: share one cache between the cases.
    os.makedirs(os.path.join(root, "aevum-cache"), exist_ok=True)
    os.symlink(os.path.join(root, "aevum-cache"), os.path.join(work, ".aevum-kernel-cache"))
    env = dict(os.environ, AEVUM_AUTOTUNE="off")
    log_path = os.path.join(work, "run.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen([os.path.abspath(prmers), "-d", str(device), "--noask"] + args, cwd=work, env=env,
                                stdout=log, stderr=subprocess.STDOUT,
                                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    deadline = time.time() + 180
    seen = False
    while time.time() < deadline and proc.poll() is None:
        with open(log_path, errors="replace") as f:
            if re.search(marker, f.read()):
                seen = True
                break
        time.sleep(0.5)
    if proc.poll() is not None:
        with open(log_path, errors="replace") as f:
            text = f.read()
            if name in ("bench", "memtest") and ("No OpenCL GPU device found" in text or "clCreateContext failed" in text):
                return name, True, "SKIPPED: no usable GPU-type OpenCL device"
        return name, False, f"exited early with {proc.returncode} (marker seen: {seen})"
    time.sleep(2.0)
    if proc.poll() is not None:
        return name, False, f"exited before the signal with {proc.returncode}"
    proc.send_signal(getattr(signal, "SIG" + sig))
    try:
        rc = proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        return name, False, "did not stop within 120 s"
    problems = []
    if rc != EXPECTED:
        problems.append(f"exit code {rc}, expected {EXPECTED}")
    if ckpts and not any(glob.glob(os.path.join(work, c)) for c in ckpts):
        problems.append(f"no checkpoint matching {ckpts}")
    if worktodo:
        with open(os.path.join(work, "worktodo.txt")) as f:
            left = [l.strip() for l in f if l.strip()]
        if left != worktodo:
            problems.append(f"worktodo.txt changed: {left}")
        if os.path.exists(os.path.join(work, "worktodo_save.txt")):
            problems.append("an entry was archived")
        if glob.glob(os.path.join(work, "*9941*")):
            problems.append("the next entry started")
    return name, not problems, "; ".join(problems) or f"rc={rc}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    prmers = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "0"
    sigs = sys.argv[3:] or ["INT"]
    env_filter = os.environ.get("CASE_FILTER", "")
    root = tempfile.mkdtemp(prefix="prmers_interrupt_")
    failed = 0
    try:
        jobs = [(sig, c) for sig in sigs for c in CASES if re.search(env_filter, c[0])]
        with ThreadPoolExecutor(max_workers=int(os.environ.get("JOBS", "4"))) as pool:
            futs = [(sig, c[0], pool.submit(run_case, prmers, device, sig, c, root)) for sig, c in jobs]
            for sig, name, fut in futs:
                n, ok, msg = fut.result()
                print(("PASS " if ok else "FAIL ") + f"{n} [SIG{sig}]: {msg}", flush=True)
                failed += 0 if ok else 1
    finally:
        if failed == 0:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print("working directories kept in", root)
    print("interrupt exit code test " + ("passed" if failed == 0 else f"FAILED ({failed})"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
