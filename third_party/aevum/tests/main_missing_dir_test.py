#!/usr/bin/env python3
"""-dir naming a directory that does not exist must end with an error message and exit status 1, not std::terminate.

Runs the real aevum binary (argv[1], default build-release/aevum); it stops while parsing arguments, before any
OpenCL device is touched.
"""
import os
import subprocess
import sys
import tempfile

binary = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "build-release/aevum")

with tempfile.TemporaryDirectory() as tmp:
    missing = os.path.join(tmp, "no", "such", "dir")
    proc = subprocess.run([binary, "-dir", missing], cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=120)
    out = proc.stdout

problems = []
if proc.returncode != 1:
    problems.append("exit status %d, want 1" % proc.returncode)
if "terminate called" in out:
    problems.append("std::terminate was called")
if "Exiting because of exception" not in out or "no/such/dir" not in out:
    problems.append("no clear error naming the directory")

print("%s  aevum -dir <missing> -> exit %d" % ("FAIL" if problems else "ok  ", proc.returncode))
if problems:
    print("\n".join(problems))
    print(out[-800:])
sys.exit(1 if problems else 0)
