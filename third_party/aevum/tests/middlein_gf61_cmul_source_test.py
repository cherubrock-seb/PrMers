#!/usr/bin/env python3
from pathlib import Path

use = Path("src/UseOptions.h").read_text()
midin = Path("src/cl/fftmiddlein.cl").read_text()
midout = Path("src/cl/fftmiddleout.cl").read_text()

assert '"AEVUM_GF61_CMUL_ALT"' in use
assert "#ifndef AEVUM_GF61_CMUL_ALT" in midin
assert "#define AEVUM_GF61_CMUL_ALT 1" in midin
assert "#ifndef AEVUM_GF61_CMUL_ALT\n#define AEVUM_GF61_CMUL_ALT 1\n#endif\n#include \"base.cl\"" in midin
assert "#define AEVUM_GF61_CMUL_ALT 1" not in midout

print("ok: middle-in GF61 cmul ALT remains default but is explicitly overrideable")
