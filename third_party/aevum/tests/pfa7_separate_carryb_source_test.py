#!/usr/bin/env python3
from pathlib import Path

src = Path("src/Gpu.cpp").read_text()

needle = '''if (fft.pfa_radix == 7 && args.value("AEVUM_PFA7_SEPARATE_CARRYB", 0)) {
        // Profiling on RTX 3080 shows the lazy fused gather is substantially
        // more expensive than the already-existing canonical carryB + fftP
        // sequence.  Both produce the same retained width-transform state.
        carryB(out);
        fftP(buf1, out);
      } else {
        fftPCarryB(buf1, out);
      }'''

assert needle in src
assert src.count('AEVUM_PFA7_SEPARATE_CARRYB') == 1

print("ok: PFA7 retained-width experiment selects canonical carryB + fftP and preserves fused default")
