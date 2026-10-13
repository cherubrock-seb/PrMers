#!/usr/bin/env python3
from pathlib import Path

use = Path("src/UseOptions.h").read_text()
base = Path("src/cl/base.cl").read_text()
carry = Path("src/cl/carryfused.cl").read_text()

assert '"OLD_FENCE"' in use
assert '#if !defined(OLD_FENCE) && !AMDGPU' in base
assert '#define OLD_FENCE 1' in base
assert '#if OLD_FENCE' in carry
assert 'u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;' in carry

print("ok: OLD_FENCE=0 exposes existing per-wavefront carryFused ready protocol while preserving Nvidia default")
