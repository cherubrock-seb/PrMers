#!/usr/bin/env python3
"""main() must exit 0 when a run ends on purpose and 1 when an exception ends it.

Runs the real aevum binary (argv[1], default build-release/aevum) in a scratch
directory.  Every case ends before any kernel is compiled or run, so no GPU is
needed (-device 99 only enumerates the OpenCL devices).
"""
import os
import subprocess
import sys
import tempfile

binary = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "build-release/aevum")

# args, expected exit status, text that must appear in the output
CASES = [
    (["-h"], 0, 'Exiting because "help"'),
    (["-version"], 0, 'Exiting because "version"'),
    (["-info", "1K:13:256"], 0, 'Exiting because "info"'),
    (["-info"], 1, 'Exiting because "-info <fft>"'),
    (["-workers", "9"], 1, 'Exiting because "Number of workers must be between 1 and 4"'),
    (["-bogusarg"], 1, 'Exiting because "args"'),
    (["-device", "99"], 1, "Exiting because"),
]

failed = 0
with tempfile.TemporaryDirectory() as tmp:
    for args, want, text in CASES:
        proc = subprocess.run([binary, "-dir", tmp] + args, cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=120)
        ok = proc.returncode == want and text in proc.stdout
        print("%s  aevum %s -> exit %d (want %d)" % ("ok  " if ok else "FAIL", " ".join(args), proc.returncode, want))
        if not ok:
            failed += 1
            print(proc.stdout[-800:])

sys.exit(1 if failed else 0)
