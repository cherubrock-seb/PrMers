#!/usr/bin/env python3
"""Forced-Aevum OFF/ON P-1 screen; compares the complete exported residue.

This starts real GPU jobs. Run only on a downstream validation host.
It preserves logs and exports in a new directory and does not benchmark.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--work-dir", type=Path, required=True,
                    help="new directory for retained evidence")
parser.add_argument("--device", default="0")
parser.add_argument("--backend", choices=("opencl", "cuda"), required=True)
parser.add_argument("--fft", default="", help="optional explicit factor-3-safe FFT")
parser.add_argument("--timeout", type=int, default=300)
args = parser.parse_args()
binary = args.binary.resolve()
work = args.work_dir.resolve()
if not binary.is_file():
    parser.error("--binary must identify an existing PrMers binary")
if args.timeout <= 0:
    parser.error("--timeout must be positive")
work.mkdir(parents=True, exist_ok=False)
if shutil.disk_usage(work).free < 15 * 1024**3:
    parser.error("validation requires at least 15 GiB free")
kernels = binary.parent / "kernels"

enabled = "[PM1 base3] Experimental Aevum square-and-scale enabled"
fallback = "[PM1 base3] Verification failed"
mismatch = "[Gerbicz Li] Mismatch"


def run(name, b1, flag, extra=(), inject=False):
    directory = work / name
    directory.mkdir()
    if kernels.is_dir():
        (directory / "kernels").symlink_to(kernels, target_is_directory=True)
    env = os.environ.copy()
    env.update(PRMERS_PM1_AEVUM_BASE3=str(flag), AEVUM_BACKEND=args.backend,
               AEVUM_AUTOTUNE="off", AEVUM_CARRY_WMUL="1")
    command = [str(binary), "269", "-pm1", "-b1", str(b1), "-b2", str(b1),
               "-aevum", "-checklevel", "1", "-resume", "-noask",
               "-d", args.device, "-f", str(directory), *extra]
    if args.fft:
        command += ["-aevum-fft", args.fft]
    (directory / "command.txt").write_text(repr(command) + "\n", encoding="utf-8")
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=directory, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=args.timeout,
                                check=False)
    text = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
    adapter_lines = [line for line in text.splitlines()
                     if "[Backend Aevum] engine::Reg adapter active" in line]
    backend = "CUDA" if args.backend == "cuda" else "OpenCL"
    if len(adapter_lines) != 1 or f" | backend={backend} |" not in adapter_lines[0]:
        raise RuntimeError(f"{name}: requested backend was not actually used")
    if result.returncode not in (0, 1):
        raise RuntimeError(f"{name}: exit {result.returncode}; inspect run.log")
    if (enabled in text) != bool(flag):
        raise RuntimeError(f"{name}: candidate activation did not match the requested flag")
    if inject:
        if "Injected error" not in text or mismatch not in text or fallback not in text:
            raise RuntimeError(f"{name}: injection, detection or fallback missing")
    elif mismatch in text or fallback in text:
        raise RuntimeError(f"{name}: unexpected verification failure")
    export = directory / f"resume_p269_B1_{b1}.save"
    content = export.read_text(encoding="utf-8")
    values = re.findall(r"\bX=0x([0-9a-fA-F]+)\b", content)
    if len(values) != 1:
        raise RuntimeError(f"{name}: expected one complete exported X field")
    plan = re.search(r"\| plan=([^|]+)", adapter_lines[0])
    if plan is None or plan.group(1).strip() == "unknown":
        raise RuntimeError(f"{name}: actual FFT plan was not reported")
    return int(values[0], 16), plan.group(1).strip()


for b1 in (100, 2141):
    baseline = run(f"b{b1}-off", b1, 0)
    candidate = run(f"b{b1}-on", b1, 1)
    if candidate != baseline:
        raise RuntimeError(f"B1={b1}: full exported residues differ")
    if b1 == 100:
        recovered = run("b100-on-injected", b1, 1, ("-erroriter", "4"), inject=True)
        if recovered != baseline:
            raise RuntimeError("full exported residue differs after recovery")

low_off = run("lowmem-off", 100, 0, ("-pm1-lowmem",))
low_on = run("lowmem-on", 100, 1, ("-pm1-lowmem",))
if low_on != low_off:
    raise RuntimeError("low-memory full exported residues differ")

print(f"PM1_AEVUM_BASE3_GPU=PASS evidence={work}")
