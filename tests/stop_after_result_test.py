#!/usr/bin/env python3
"""A stop that lands after the result was saved must not leave the job half-finished.

Once the result is recorded, the worktodo entry has to be retired and the state deleted as after a normal
finish; otherwise the next run repeats the job and writes its result a second time. The stop only suppresses
the restart on the next worktodo entry, and the process exits 1 (core::kExitInterrupted): the run was
stopped, but its work is recorded.

Deterministic: each job runs under gdb, which delivers a real SIGINT to the program at a fixed point of the
end of the job (no timing involved):
  save    on entry to fopen() of the result file (results.txt; for P-1 the one that ends the job): the stop is
          pending while the result is written, so it is already set when the end-of-job code decides what to
          do with the entry;
  retire  on entry to rename(worktodo*): the stop arrives while the entry is being retired, i.e. between the
          result save and the restart for the next entry.
For every mode that retires a worktodo entry (PRP on the Marin and legacy backends, LL-SAFE, LL-unsafe, Wagstaff on both,
P-1 stage 1 only and stage 1 + 2, ECM Montgomery and Edwards, GMTF) it checks:
  - exit code 1,
  - the job's result is recorded exactly once (results.txt lines, or the per-job json file for GMTF),
  - the entry is gone from worktodo.txt and archived in worktodo_save.txt,
  - the job's checkpoint/state files are gone,
  - the next entry did not start (still queued, no result of its own),
  - a second run (no stop) runs only the next entry: the results then hold exactly one record per job.
Skipped (exit 0, "SKIPPED") when gdb is unavailable, cannot trace, or the platform is not x86-64 / aarch64.

Usage: stop_after_result_test.py PRMERS_BINARY [DEVICE] [CASE_REGEX]   (env JOBS: parallel cases, default 4)
"""
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor

# Each case: name, entry A (runs first and is stopped), entry B (must not start in the stopped run), extra argv,
# state globs that must be gone once A is retired, and the result as the job writes it:
#   hook   what the "save" stop waits for: the path suffix of the fopen of the result file, and which of the
#          opens of it is the one that ends the job (P-1 also records its stage 1 result in results.txt),
#   count  how many lines results.txt gets per job, or None when the result is a per-job json file (GMTF).
def case(name, a, b, flags, states, suffix="results.txt", nth=1, count=1):
    return dict(name=name, a=a, b=b, flags=flags, states=states, suffix=suffix, nth=nth, count=count)


CASES = [
    case("prp-marin", "PRP=N/A,1,2,4423,-1,70,0", "PRP=N/A,1,2,9941,-1,70,0", ["-t", "0", "-engine-marin"],
         ["m_4423.ckpt*", "4423prp.*"]),
    case("prp-legacy", "PRP=N/A,1,2,4423,-1,70,0", "PRP=N/A,1,2,9941,-1,70,0", ["-t", "0", "-marin"],
         ["4423prp.*", "m_4423.ckpt*"]),
    case("ll-unsafe", "Test=4423,70,1", "Test=9689,70,1", ["-t", "0", "-engine-marin"],
         ["llunsafe_m_4423.ckpt*"]),
    case("ll-safe", "DoubleCheck=4423,70,1", "DoubleCheck=9689,70,1", ["-t", "0"],
         ["llsafe_m_4423.ckpt*"]),
    case("wagstaff-marin", "PRP=1,2,5807,-1", "PRP=1,2,10501,-1", ["-t", "0", "-wagstaff", "-engine-marin"],
         ["wagstaff_m_11614.ckpt*"]),
    case("wagstaff-legacy", "PRP=1,2,5807,-1", "PRP=1,2,10501,-1", ["-t", "0", "-wagstaff", "-marin"],
         ["11614prp_wagstaff.*"]),
    # P-1: stage 1 only, and stage 1 + 2 (two result lines per job; the stop waits for the second)
    case("pm1-stage1", "Pminus1=1,2,9941,-1,1000,0", "Pminus1=1,2,9689,-1,1000,0", [],
         ["pm1_m_9941*.ckpt*"]),
    case("pm1-stage2", "Pminus1=1,2,9941,-1,1000,20000", "Pminus1=1,2,9689,-1,1000,20000", [],
         ["pm1_m_9941*.ckpt*", "pm1_s2_*_m_9941*"], nth=2, count=2),
    case("ecm-montgomery", "ECM2=1,2,269,-1,2000,20000,1", "ECM2=1,2,277,-1,2000,20000,1", [],
         ["ecm_*_269_*.ckpt*", "ecm_*269*.ckpt*"]),
    case("ecm-edwards", "ECM2=1,2,269,-1,2000,20000,1", "ECM2=1,2,277,-1,2000,20000,1", ["-edwards"],
         ["ecm_*_269_*.ckpt*", "ecm_*269*.ckpt*"]),
    case("gmtf", "GMTF=1009,20,24", "GMTF=1031,20,24", [], ["*p1009*.checkpoint"],
         suffix="_result.json", count=None),
]

HOOK = textwrap.dedent("""
    import gdb, os
    STOPAT = os.environ["STOPAT"]
    SUFFIX = os.environ["HOOK_SUFFIX"]
    NTH = int(os.environ["HOOK_NTH"])
    LOG = open(os.environ["HOOK_LOG"], "w")

    class Hook(gdb.Breakpoint):
        def __init__(self, spec, match, nth):
            super().__init__(spec, internal=True)
            self.match = match
            self.nth = nth
            self.seen = 0
            self.fired = False

        def stop(self):
            if self.fired:
                return False
            try:
                arch = str(gdb.selected_frame().architecture().name())
                reg = "$x0" if "aarch64" in arch else "$rdi"
                path = gdb.parse_and_eval("(char*)" + reg).string("utf-8", "replace")
            except Exception as e:
                LOG.write("error %s\\n" % e)
                LOG.flush()
                return False
            if self.match(path):
                self.seen += 1
                if self.seen == self.nth:
                    self.fired = True
                    LOG.write("HIT %s %s\\n" % (self.location, path))
                    LOG.flush()
                    return True
            return False

    for cmd in ("set pagination off", "set confirm off", "set startup-with-shell off",
                "set breakpoint pending on", "set print thread-events off", "set print inferior-events off"):
        gdb.execute(cmd)
    if STOPAT == "save":
        hook = Hook("fopen", lambda p: p.endswith(SUFFIX), NTH)
    else:
        hook = Hook("rename", lambda p: "worktodo" in p, 1)
    gdb.execute("run")
    if hook.fired:
        hook.enabled = False
        LOG.write("SIGNAL\\n")
        LOG.flush()
        # The exit status is read from the exit_group syscall (its first argument): gdb can lose the
        # exit of a multi-threaded program ("Couldn't get registers"), so $_exitcode is not reliable.
        gdb.execute("catch syscall exit_group")
        try:
            gdb.execute("signal SIGINT")
            while True:
                arch = str(gdb.selected_frame().architecture().name())
                arm = "aarch64" in arch
                nr = int(gdb.parse_and_eval("$x8" if arm else "$orig_rax"))
                if nr == (94 if arm else 231):
                    LOG.write("EXIT %s\\n" % (int(gdb.parse_and_eval("$x0" if arm else "$rdi")) & 255))
                    LOG.flush()
                    gdb.execute("continue")
                    break
                gdb.execute("continue")
        except gdb.error as e:
            LOG.write("gdb: %s\\n" % e)
    LOG.close()
""")


def tool_problem():
    if shutil.which("gdb") is None:
        return "gdb is not installed"
    if os.uname().machine not in ("x86_64", "AMD64", "aarch64", "arm64"):
        return "unsupported architecture " + os.uname().machine
    probe = subprocess.run(["gdb", "-q", "-batch", "-ex", "run", "--args", "true"],
                           capture_output=True, text=True, errors="replace")
    out = probe.stdout + probe.stderr
    if "ptrace" in out and "not permitted" in out:
        return "gdb cannot trace processes here (ptrace is not permitted)"
    return None


def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def lines(path):
    return [l.strip() for l in read(path).splitlines() if l.strip()]


def records(work, c):
    """How many results the job directory holds."""
    if c["count"] is None:
        return len(glob.glob(os.path.join(work, "gm_tf_*_result.json")))
    return len(lines(os.path.join(work, "results.txt")))


def prepare(root, prmers, tag, c):
    work = os.path.join(root, f"{c['name']}.{tag}")
    os.makedirs(work)
    os.symlink(os.path.join(os.path.dirname(os.path.abspath(prmers)), "kernels"), os.path.join(work, "kernels"))
    os.makedirs(os.path.join(root, "aevum-cache"), exist_ok=True)
    os.symlink(os.path.join(root, "aevum-cache"), os.path.join(work, ".aevum-kernel-cache"))
    with open(os.path.join(work, "worktodo.txt"), "w") as f:
        f.write(c["a"] + "\n" + c["b"] + "\n")
    return work


def run_prmers(prmers, device, work, c, log_name, stopat=None, hook_path=None):
    env = dict(os.environ, AEVUM_AUTOTUNE="off", AEVUM_CARRY_WMUL="1")
    cmd = [os.path.abspath(prmers), "-d", str(device), "-noask"] + c["flags"]
    if stopat:
        env.update(STOPAT=stopat, HOOK_LOG=os.path.join(work, "hook.log"),
                   HOOK_SUFFIX=c["suffix"], HOOK_NTH=str(c["nth"]))
        cmd = ["gdb", "-q", "-batch", "-x", hook_path, "--args"] + cmd
    with open(os.path.join(work, log_name), "w") as log:
        proc = subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900,
                              preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    text = read(os.path.join(work, log_name))
    if not stopat:
        return proc.returncode, text
    # the hook script records the exit status of the program ($_exitcode) once it has exited
    m = re.search(r"^EXIT (\d+)$", read(env["HOOK_LOG"]), re.M)
    return (int(m.group(1)) if m else None), text


def run_case(prmers, device, hook_path, root, c, stopat):
    work = prepare(root, prmers, stopat, c)
    problems = []
    rc, text = run_prmers(prmers, device, work, c, "run1.log", stopat, hook_path)
    hook_log = read(os.path.join(work, "hook.log"))
    if "SIGNAL" not in hook_log:
        return c["name"], stopat, False, "the stop was never delivered: " + (hook_log.strip() or "no hook log")
    if rc != 1:
        problems.append(f"exit code {rc}, expected 1")
    todo = lines(os.path.join(work, "worktodo.txt"))
    if todo != [c["b"]]:
        problems.append(f"worktodo.txt is {todo}, expected only the next entry")
    saved = lines(os.path.join(work, "worktodo_save.txt"))
    if saved != [c["a"]]:
        problems.append(f"worktodo_save.txt is {saved}, expected only the finished entry")
    n = records(work, c)
    if n != (c["count"] or 1):
        problems.append(f"{n} result records, expected {c['count'] or 1}")
    for g in c["states"]:
        left = glob.glob(os.path.join(work, g))
        if left:
            problems.append("state left behind: " + ", ".join(sorted(os.path.basename(x) for x in left)))
    if "Stop requested; not restarting" not in text and "Stop requested after the result was saved" not in text:
        problems.append("the log says neither that the stop was noted nor that the restart was refused")
    if stopat == "save" and "Stop requested after the result was saved" not in text:
        problems.append("no note that the stop came after the result was saved")
    if not problems:
        # next run: only entry B; A is not repeated
        rc2, _ = run_prmers(prmers, device, work, c, "run2.log")
        if rc2 != 0:
            problems.append(f"second run exit code {rc2}")
        n2 = records(work, c)
        if n2 != 2 * (c["count"] or 1):
            problems.append(f"after the second run {n2} result records, expected {2 * (c['count'] or 1)}")
        if lines(os.path.join(work, "worktodo.txt")):
            problems.append("the second run left an entry queued")
        if lines(os.path.join(work, "worktodo_save.txt")) != [c["a"], c["b"]]:
            problems.append("worktodo_save.txt after the second run: " + str(lines(os.path.join(work, "worktodo_save.txt"))))
    return c["name"], stopat, not problems, "; ".join(problems) or "ok"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    prmers = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "0"
    only = sys.argv[3] if len(sys.argv) > 3 else ""
    why = tool_problem()
    if why:
        print("SKIPPED: " + why)
        return 0
    root = tempfile.mkdtemp(prefix="prmers_stop_after_result_")
    hook_path = os.path.join(root, "hook.py")
    with open(hook_path, "w") as f:
        f.write(HOOK)
    failed = 0
    try:
        jobs = [(c, s) for c in CASES if re.search(only, c["name"]) for s in ("save", "retire")]
        with ThreadPoolExecutor(max_workers=int(os.environ.get("JOBS", "4"))) as pool:
            futs = [pool.submit(run_case, prmers, device, hook_path, root, c, s) for c, s in jobs]
            for fut in futs:
                name, what, ok, msg = fut.result()
                print(("PASS " if ok else "FAIL ") + f"{name} [{what}]: {msg}", flush=True)
                failed += 0 if ok else 1
    finally:
        if failed == 0:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print("working directories kept in", root)
    print("stop after result test " + ("passed" if failed == 0 else f"FAILED ({failed})"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
