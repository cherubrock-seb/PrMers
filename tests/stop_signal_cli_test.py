#!/usr/bin/env python3
"""SIGTERM and SIGHUP stop a CLI run exactly like SIGINT (Ctrl-C).

For every mode: start prmers on a job that runs long enough (OpenCL device 0; PoCL's CPU device is
enough), wait until it is computing, then send SIGINT, SIGTERM and SIGHUP in three separate runs and
check that all three behave the same:
  - the process stops on its own within the time limit (before: SIGTERM/SIGHUP killed it at once with
    no checkpoint, or - in modes that now catch it - would be ignored),
  - the exit code is the one SIGINT gives,
  - the mode saved a checkpoint (where it has one),
  - with a worktodo queue: the interrupted entry is still first in worktodo.txt, was not archived and the
    next entry did not start.
Further cases:
  - Prime95 hand-off (P-1 and ECM stage 2): a stub "mprime" that hangs; a Stop must reach it as SIGTERM
    and prmers must finish promptly without waiting for a result the stub will never write.
  - nohup: with SIGHUP inherited as ignored, SIGHUP must not stop the run.
Usage: stop_signal_cli_test.py PRMERS_BINARY [DEVICE] [SIGNAL ...]   (default: INT TERM HUP)
"""
import glob
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

# name, argv (after -d DEVICE --noask), worktodo lines (or None), marker regex in the log,
# checkpoint globs (one must match)
CASES = [
    ("prp-marin-engine", ["110503", "-prp"], None, r"Progress", ["m_110503.ckpt"]),
    ("prp-legacy", ["110503", "-prp", "-marin"], None, r"Progress", ["110503prp.mers"]),
    ("ll", ["110503", "-ll"], None, r"Progress", ["llsafe_m_110503.ckpt"]),
    ("llsafe2", ["110503", "-llsafe2"], None, r"Progress", ["llsafe2_m_110503.ckpt"]),
    ("pm1-stage1", ["1000003", "-pm1", "-b1", "2000000", "-b2", "20000000"], None, r"Stage 1|Progress|B1", ["pm1_m_1000003.ckpt"]),
    ("pm1-legacy", ["1000003", "-pm1", "-b1", "2000000", "-b2", "20000000", "-marin"], None, r"Stage 1|Progress|B1", ["1000003pm12000000.mers"]),
    ("pm1-stage2", ["20011", "-pm1", "-b1", "2000", "-b2", "4000000000"], None, r"V-trace baby window", ["pm1_s2_*_m_20011.ckpt"]),
    ("ecm-montgomery", ["100003", "-ecm", "-b1", "200000", "-b2", "2000000", "-K", "50"], None, r"urve|Stage|B1", ["ecm_te_m_100003_c0.ckpt"]),
    ("ecm-edwards", ["100003", "-ecm", "-b1", "200000", "-b2", "2000000", "-K", "50", "-edwards"], None, r"urve|Stage|B1", ["ecm_te_m_100003_c0.ckpt"]),
    ("gm-proth", ["200003", "-gm-proth", "-gm-sieve", "0"], None, r"Progress|iter", ["gm_proth_p200003.ckpt"]),
    ("gm-pm1", ["1000003", "-gm-pm1", "-b1", "3000000", "-b2", "30000000", "-gm-sieve", "0"], None, r"Loaded carryFused|Stage 1 bits", []),
    ("gm-ecm", ["100003", "-gm-ecm", "-b1", "300000", "-b2", "3000000", "-K", "50", "-gm-sieve", "0"], None, r"Progress|ECM|curve", ["gm_ecm_p100003_c0_stage1.ckpt"]),
    ("gmtf-direct", ["-gm-tf", "40", "62", "-gm-family", "BOTH", "15317251"], None, r"progress", ["*.checkpoint"]),
    ("bench", ["-bench"], None, r"\[1/\d+\] TS=", []),
    ("worktodo-prp", [], ["PRP=N/A,1,2,110503,-1,70,0", "PRP=N/A,1,2,9941,-1,70,0"], r"Progress", ["m_110503.ckpt"]),
    ("worktodo-gmtf", [], ["GMTF=15317251,40,62,BOTH", "PRP=N/A,1,2,9941,-1,70,0"], r"progress", ["*.checkpoint"]),
    ("gmchain-both-ecm", [], ["GMCHAIN=1279,2000,20000,1000,10000,8,0,0,factor,BOTH", "PRP=N/A,1,2,9941,-1,70,0"],
     r"Gaussian pair ECM factoring", ["gq_ecm_p1279_c0_stage*.ckpt"]),
]

# Prime95 hand-off cases: (name, argv). The stub records that it started and that it got SIGTERM.
P95_STUB = """#!/bin/sh
echo started > started.flag
trap 'echo got-TERM > got_term.flag; exit 143' TERM
sleep 300 &
wait $!
"""
P95_CASES = [
    ("p95-pm1", ["113", "-pm1", "-b1", "4", "-b2", "2141", "-p95path", "./p95"]),
    ("p95-ecm", ["1693", "-ecm", "-ced", "-notorsion", "-b1", "3", "-b2", "2000", "-K", "1", "-p95path", "./p95"]),
]

STOP_SECONDS = 120


def start(prmers, device, work, args, nohup=False):
    env = dict(os.environ, AEVUM_AUTOTUNE="off")

    def pre():
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGHUP, signal.SIG_IGN if nohup else signal.SIG_DFL)

    log_path = os.path.join(work, "run.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen([os.path.abspath(prmers), "-d", str(device), "--noask"] + args, cwd=work, env=env,
                                stdout=log, stderr=subprocess.STDOUT, preexec_fn=pre)
    return proc, log_path


def wait_for(proc, log_path, marker, flag_file=None, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline and proc.poll() is None:
        if flag_file is not None:
            if os.path.exists(flag_file):
                return True
        else:
            with open(log_path, errors="replace") as f:
                if re.search(marker, f.read()):
                    return True
        time.sleep(0.5)
    return False


def prepare(root, name, sig, worktodo):
    work = os.path.join(root, f"{name}.{sig}")
    os.makedirs(work)
    prmers_dir = os.path.dirname(os.path.abspath(sys.argv[1]))
    os.symlink(os.path.join(prmers_dir, "kernels"), os.path.join(work, "kernels"))
    # The Aevum kernels take minutes to compile on PoCL: share one cache between the cases.
    os.makedirs(os.path.join(root, "aevum-cache"), exist_ok=True)
    os.symlink(os.path.join(root, "aevum-cache"), os.path.join(work, ".aevum-kernel-cache"))
    if worktodo:
        with open(os.path.join(work, "worktodo.txt"), "w") as f:
            f.write("\n".join(worktodo) + "\n")
    return work


def run_case(prmers, device, sig, case, root):
    """Returns (rc, problems list)."""
    name, args, worktodo, marker, ckpts = case
    work = prepare(root, name, sig, worktodo)
    proc, log_path = start(prmers, device, work, args)
    seen = wait_for(proc, log_path, marker)
    if proc.poll() is not None:
        with open(log_path, errors="replace") as f:
            text = f.read()
        if name == "bench" and ("No OpenCL GPU device found" in text or "clCreateContext failed" in text):
            return None, []
        return proc.returncode, [f"exited before the signal with {proc.returncode} (marker seen: {seen})"]
    time.sleep(2.0)
    if proc.poll() is not None:
        return proc.returncode, [f"exited before the signal with {proc.returncode}"]
    proc.send_signal(getattr(signal, "SIG" + sig))
    try:
        rc = proc.wait(timeout=STOP_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        return None, [f"did not stop within {STOP_SECONDS} s"]
    problems = []
    if rc < 0:
        problems.append(f"killed by signal {-rc} instead of stopping cleanly")
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
    return rc, problems


def run_mode(prmers, device, sigs, case, root):
    name = case[0]
    results = {}
    for sig in sigs:
        rc, problems = run_case(prmers, device, sig, case, root)
        results[sig] = (rc, problems)
    out = []
    ok = True
    ref = results[sigs[0]][0]
    for sig in sigs:
        rc, problems = results[sig]
        if rc is None and not problems:
            out.append((name, sig, True, "SKIPPED: no usable GPU-type OpenCL device"))
            continue
        if sig != sigs[0] and rc != ref:
            problems = problems + [f"exit code {rc}, SIGINT gave {ref}"]
        out.append((name, sig, not problems, "; ".join(problems) or f"rc={rc}"))
    return out


def run_p95(prmers, device, case, root, sig):
    name, args = case
    work = prepare(root, name, sig, None)
    os.makedirs(os.path.join(work, "p95"))
    stub = os.path.join(work, "p95", "mprime")
    with open(stub, "w") as f:
        f.write(P95_STUB)
    os.chmod(stub, os.stat(stub).st_mode | stat.S_IXUSR)
    proc, log_path = start(prmers, device, work, args)
    started = wait_for(proc, log_path, None, flag_file=os.path.join(work, "p95", "started.flag"), timeout=300)
    if not started:
        return name, sig, False, f"the Prime95 stub never started (rc={proc.poll()})"
    time.sleep(1.0)
    t0 = time.time()
    proc.send_signal(getattr(signal, "SIG" + sig))
    try:
        rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        return name, sig, False, "prmers did not stop within 60 s"
    took = time.time() - t0
    problems = []
    if not os.path.exists(os.path.join(work, "p95", "got_term.flag")):
        problems.append("Prime95 never received SIGTERM")
    if rc < 0:
        problems.append(f"killed by signal {-rc}")
    with open(log_path, errors="replace") as f:
        text = f.read()
    if "did not produce results.json.txt" in text:
        problems.append("reported a Prime95 failure instead of a stop")
    if took > 40:
        problems.append(f"took {took:.0f} s to stop (waited for a result?)")
    return name, sig, not problems, "; ".join(problems) or f"rc={rc}, stopped in {took:.1f} s, Prime95 got SIGTERM"


def run_nohup(prmers, device, root):
    name, sig = "nohup", "HUP"
    work = prepare(root, name, sig, None)
    proc, log_path = start(prmers, device, work, ["110503", "-prp"], nohup=True)
    if not wait_for(proc, log_path, r"Progress"):
        return name, sig, False, "never started computing"
    time.sleep(2.0)
    proc.send_signal(signal.SIGHUP)
    time.sleep(8.0)
    alive = proc.poll() is None
    if alive:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=STOP_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
    return name, sig, alive, "SIGHUP ignored as inherited (nohup)" if alive else "SIGHUP stopped a nohup run"


def run_prompt(prmers, device, root, sig):
    """Waiting at the "Enter the exponent" prompt: a signal must still end the process (the handlers are
    installed only once the run starts), not leave it blocked on stdin."""
    name = "prompt"
    work = prepare(root, name, sig, None)
    env = dict(os.environ, AEVUM_AUTOTUNE="off")
    log_path = os.path.join(work, "run.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen([os.path.abspath(prmers), "-d", str(device)], cwd=work, env=env, stdin=subprocess.PIPE,
                                stdout=log, stderr=subprocess.STDOUT,
                                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    if not wait_for(proc, log_path, r"Enter the exponent"):
        return name, sig, False, "no prompt"
    proc.send_signal(getattr(signal, "SIG" + sig))
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        return name, sig, False, "still blocked at the prompt after 15 s"
    finally:
        proc.stdin.close()
    return name, sig, True, f"ended at the prompt (rc={proc.returncode})"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    prmers = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "0"
    sigs = sys.argv[3:] or ["INT", "TERM", "HUP"]
    if sigs[0] != "INT":
        sigs = ["INT"] + [s for s in sigs if s != "INT"]
    flt = os.environ.get("CASE_FILTER", "")
    root = tempfile.mkdtemp(prefix="prmers_stop_signal_")
    failed = 0
    try:
        with ThreadPoolExecutor(max_workers=int(os.environ.get("JOBS", "4"))) as pool:
            futs = [pool.submit(run_mode, prmers, device, sigs, c, root) for c in CASES if re.search(flt, c[0])]
            futs += [pool.submit(lambda c=c, s=s: [run_p95(prmers, device, c, root, s)])
                     for c in P95_CASES if re.search(flt, c[0]) for s in sigs]
            futs += [pool.submit(lambda s=s: [run_prompt(prmers, device, root, s)]) for s in sigs if re.search(flt, "prompt")]
            if re.search(flt, "nohup"):
                futs.append(pool.submit(lambda: [run_nohup(prmers, device, root)]))
            for fut in futs:
                for name, sig, ok, msg in fut.result():
                    print(("PASS " if ok else "FAIL ") + f"{name} [SIG{sig}]: {msg}", flush=True)
                    failed += 0 if ok else 1
    finally:
        if failed == 0:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print("working directories kept in", root)
    print("stop signal test " + ("passed" if failed == 0 else f"FAILED ({failed})"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
