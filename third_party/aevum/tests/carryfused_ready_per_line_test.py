#!/usr/bin/env python3
"""carryFused's ready flags: one per READY_LANES = min(G_W, WAVEFRONT) lanes of a line, never G_W / WAVEFRONT per line.

With G_W < WAVEFRONT (G_W = 32 on a 64-lane AMD wavefront), G_W / WAVEFRONT is 0: every group would signal and wait on
flag 0 and the !OLD_FENCE hand-off would hang.  No shipped geometry has G_W < 64 today; this keeps the indexing safe
if one is added.  Checks that every ready-flag index in carryfused.cl goes through READY_LANES / READY_PER_LINE, and
evaluates the two macros with the C preprocessor for line widths below, at and above the wavefront.
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "cl" / "carryfused.cl"
text = SRC.read_text(encoding="utf-8")
fail = []

# Every per-wavefront index/test must use the READY_* macros (the "#if G_W > WAVEFRONT" barrier guards are fine).
for n, line in enumerate(text.splitlines(), 1):
    code = line.split("//")[0]
    if "WAVEFRONT" in code and not code.lstrip().startswith(("#if", "#define READY_LANES")):
        fail.append(f"carryfused.cl:{n}: WAVEFRONT used directly: {line.strip()}")

defs = [l for l in text.splitlines() if re.match(r"#define READY_(LANES|PER_LINE)\b", l)]
if len(defs) != 2:
    fail.append("READY_LANES / READY_PER_LINE definitions not found")

cpp = shutil.which("cpp") or shutil.which("clang")
if cpp and len(defs) == 2:
    for g_w, wavefront in [(32, 64), (32, 32), (64, 64), (64, 32), (128, 64), (256, 32), (512, 64)]:
        probe = "\n".join(defs) + f"""
#define G_W {g_w}
#define WAVEFRONT {wavefront}
#if READY_LANES < 1 || READY_PER_LINE < 1 || READY_LANES * READY_PER_LINE != G_W
#error bad
#endif
lanes READY_LANES perline READY_PER_LINE
"""
        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
            f.write(probe)
            name = f.name
        cmd = [cpp, "-P", name] if cpp.endswith("cpp") else [cpp, "-E", "-P", "-x", "c", name]
        p = subprocess.run(cmd, capture_output=True, text=True)
        pathlib.Path(name).unlink()
        if p.returncode:
            fail.append(f"G_W={g_w} WAVEFRONT={wavefront}: READY_LANES * READY_PER_LINE must equal G_W with both >= 1")

for f in fail:
    print("FAIL:", f)
if fail:
    sys.exit(1)
print("ok: carryFused ready flags are per READY_LANES lanes, one or more per line")
