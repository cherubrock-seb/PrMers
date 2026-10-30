#!/usr/bin/env python3
"""getNvidiaComputeCapability() returns major * 100 + minor (sm_75 is 705, sm_86 is 806, sm_120 is 1200).

A literal written as major * 100 + minor * 10 (750 for sm_75, the __CUDA_ARCH__ style) silently means something else:
"< 750" refuses sm_75 and every 7.x part, while the message says sm_75+.  Every literal compared with the host's
compute capability must have a minor digit below 10.
"""
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parents[1]
fail = []
for path in sorted((root / "src").rglob("*.cpp")):
    text = path.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"getNvidiaComputeCapability\([^)]*\)\s*(?:[<>]=?|[!=]=)\s*(\d+)u?", text):
        value = int(m.group(1))
        if value % 100 >= 10:
            line = text.count("\n", 0, m.start()) + 1
            fail.append(f"{path.relative_to(root)}:{line}: compute capability literal {value} is not major * 100 + minor")
for f in fail:
    print("FAIL:", f)
if fail:
    sys.exit(1)
print("ok: NVIDIA compute capability literals use major * 100 + minor")
