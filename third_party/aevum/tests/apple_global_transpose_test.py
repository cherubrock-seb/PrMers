#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cl = (root / "src/cl/transpose.cl").read_text()
gpu = (root / "src/Gpu.cpp").read_text()
api = (root / "src/EngineApi.cpp").read_text()

assert "transposeOutAppleGlobal" in cl
assert "transposeInAppleGlobal" in cl
assert "out[col * BIG_HEIGHT + row] = in[gid];" in cl
assert "out[col * WIDTH + row] = in[gid];" in cl

for name in ("transposeOutAppleGlobal", "transposeInAppleGlobal"):
    start = cl.index(f"KERNEL(64) {name}")
    end = cl.index("}\n", start) + 2
    body = cl[start:end]
    assert "local " not in body
    assert "bar()" not in body

assert '#if defined(__APPLE__)\n  // Apple OpenCL-to-Metal corrupts the stock 64x64 LDS transpose' in gpu
assert 'K(transpIn,  "transpose.cl", "transposeInAppleGlobal",  hN)' in gpu
assert 'K(transpOut, "transpose.cl", "transposeOutAppleGlobal", hN)' in gpu
assert 'K(transpIn,  "transpose.cl", "transposeIn",  hN / 64)' in gpu
assert 'K(transpOut, "transpose.cl", "transposeOut", hN / 64)' in gpu
assert "register upload/readback uses direct global transpose kernels without LDS" in api


def old_tiled_transpose(values, W, H):
    assert W % 64 == 0 and H % 64 == 0
    out = [None] * (W * H)
    gpw, gph = W // 64, H // 64
    for g in range(gpw * gph):
        gy = g % gph
        gx = (gy + g // gph) % gpw
        for i in range(64):
            for me in range(64):
                # This is the exact LDS transpose performed by transposeWords:
                # output(row=64*gx+i,col=64*gy+me) receives
                # input(row=64*gy+me,col=64*gx+i).
                src = (64 * gy + me) * W + (64 * gx + i)
                dst = (64 * gx + i) * H + (64 * gy + me)
                out[dst] = values[src]
    assert all(x is not None for x in out)
    return out


def direct_transpose(values, W, H):
    out = [None] * (W * H)
    for gid, value in enumerate(values):
        row, col = divmod(gid, W)
        out[col * H + row] = value
    return out

for W, H in ((128, 128), (256, 512), (1024, 512)):
    values = list(range(W * H))
    old = old_tiled_transpose(values, W, H)
    direct = direct_transpose(values, W, H)
    assert direct == old, (W, H)
    assert direct_transpose(direct, H, W) == values, (W, H)

print("Apple direct global transpose mapping audit passed")
