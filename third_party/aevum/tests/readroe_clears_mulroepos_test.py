#!/usr/bin/env python3
# Gpu::readROE() must forget the multiplication positions on every path.  mul() records the current ROE slot in
# mulRoePos even while ROE sampling is off (roePos stays 0); if readROE() returned early without clearing them, the
# next window's first samples would be counted as multiplications instead of squarings.
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
src = (root / "src/Gpu.cpp").read_text(encoding="utf-8")
start = src.find("pair<RoeInfo, RoeInfo> Gpu::readROE() {")
if start < 0:
    raise SystemExit("Gpu::readROE() not found")
depth, i = 0, src.index("{", start)
for j in range(i, len(src)):
    depth += {"{": 1, "}": -1}.get(src[j], 0)
    if depth == 0:
        body = src[i + 1:j]
        break
else:
    raise SystemExit("unbalanced Gpu::readROE()")

lines = [l.split("//")[0] for l in body.splitlines()]
returns = [n for n, l in enumerate(lines) if re.search(r"\breturn\b", l)]
if len(returns) < 2:
    raise SystemExit("expected a return on both the sampled and the empty path of Gpu::readROE()")
for n in returns:
    k = n - 1
    while k >= 0 and not lines[k].rstrip().endswith("{"):
        k -= 1
    branch = "\n".join(lines[k:n + 1])
    if "mulRoePos.clear()" not in branch:
        raise SystemExit(f"Gpu::readROE() returns without clearing mulRoePos: {lines[n].strip()}")
print("readROE mulRoePos clear test passed")
