#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cl = (ROOT / "src/cl/fftmiddleout.cl").read_text()
gpu = (ROOT / "src/Gpu.cpp").read_text()
hdr = (ROOT / "src/Gpu.h").read_text()

names = (
    "fftMiddleOutGF61ApplePlaceholder",
    "fftMiddleOutGF61LoadScalarApple",
    "fftMiddleOutGF61MulScalarApple",
    "fftMiddleOutGF61FftApple",
    "fftMiddleOutGF61Mul2ScalarApple",
    "fftMiddleOutGF61WriteScalarApple",
)
for name in names:
    assert name in cl and name in gpu, name

for old in ("fftMiddleOutGF61CoreApple", "fftMiddleOutGF61WriteApple"):
    assert old not in cl and old not in gpu and old not in hdr, old

for member in (
    "kfftMidOutGF61LoadScalarApple",
    "kfftMidOutGF61MulScalarApple",
    "kfftMidOutGF61FftApple",
    "kfftMidOutGF61Mul2ScalarApple",
    "kfftMidOutGF61WriteScalarApple",
):
    assert f"Kernel {member};" in hdr

assert 'fft.shape.middle >= 8 ? "fftMiddleOutGF61ApplePlaceholder" : "fftMiddleOutGF61"' in gpu

# All stages except the isolated FFT are scalar in MIDDLE.
load = cl[cl.index("KERNEL(OUT_WG) fftMiddleOutGF61LoadScalarApple"):
          cl.index("KERNEL(OUT_WG) fftMiddleOutGF61MulScalarApple")]
mul = cl[cl.index("KERNEL(OUT_WG) fftMiddleOutGF61MulScalarApple"):
         cl.index("KERNEL(OUT_WG) fftMiddleOutGF61FftApple")]
fft = cl[cl.index("KERNEL(OUT_WG) fftMiddleOutGF61FftApple"):
         cl.index("KERNEL(OUT_WG) fftMiddleOutGF61Mul2ScalarApple")]
mul2 = cl[cl.index("KERNEL(OUT_WG) fftMiddleOutGF61Mul2ScalarApple"):
          cl.index("KERNEL(OUT_WG) fftMiddleOutGF61WriteScalarApple")]
write = cl[cl.index("KERNEL(OUT_WG) fftMiddleOutGF61WriteScalarApple"):
           cl.index("// Signature-compatible placeholder")]
for body in (load, mul, mul2, write):
    assert "GF61 u[MIDDLE]" not in body
assert "GF61 u[MIDDLE]" in fft and "fft_MIDDLE(u);" in fft
assert "dependentLaunchWait();" in load
assert "dependentLaunch();" in write
assert "PAD_SIZE=0" in load and "PAD_SIZE=0" in write

# PAD=0 scalar load is exactly readMiddleOutLine:
# in[y*MIDDLE*SMALL_HEIGHT + x + k*SMALL_HEIGHT].
for width, middle, small_h, out_wg, out_sizex in (
    (64, 2, 64, 64, 16),
    (128, 8, 64, 64, 16),
):
    size_y = out_wg // out_sizex
    groups_x = small_h // out_sizex
    groups_y = width // size_y
    scratch = set()
    outputs = set()
    for gy in range(groups_y):
        for gx in range(groups_x):
            g = gy * groups_x + gx
            for me in range(out_wg):
                mx, my = me % out_sizex, me // out_sizex
                x = gx * out_sizex + mx
                y = gy * size_y + my
                transposed = mx * size_y + my
                for k in range(middle):
                    input_index = y * middle * small_h + x + k * small_h
                    expected_input = y * middle * small_h + x + k * small_h
                    assert input_index == expected_input

                    tmp_index = g * middle * out_wg + k * out_wg + me
                    scratch.add(tmp_index)

                    output_index = (
                        gy * middle * out_wg
                        + gx * middle * width * out_sizex
                        + k * out_wg
                        + transposed
                    )
                    outputs.add(output_index)

    ND = width * middle * small_h
    assert scratch == set(range(ND))
    assert outputs == set(range(ND))

# Production MIDDLE=8 plan has exactly ND scalar groups and the same
# transpose formula; sample every workgroup/lane and the boundary k values.
width, middle, small_h, out_wg, out_sizex = 1024, 8, 256, 64, 16
size_y = out_wg // out_sizex
groups_x = small_h // out_sizex
groups_y = width // size_y
groups = groups_x * groups_y
assert groups * out_wg * middle == width * middle * small_h
for g in range(groups):
    gx, gy = g % groups_x, g // groups_x
    for me in range(out_wg):
        mx, my = me % out_sizex, me // out_sizex
        transposed = mx * size_y + my
        assert 0 <= transposed < out_wg
        for k in (0, middle - 1):
            tmp_index = g * middle * out_wg + k * out_wg + me
            assert 0 <= tmp_index < width * middle * small_h
            output_index = (
                gy * middle * out_wg
                + gx * middle * width * out_sizex
                + k * out_wg
                + transposed
            )
            assert 0 <= output_index < width * middle * small_h

# Scalar middleMul2 computes base*w^k, exactly the vector loop.
for middle in (2, 4, 8, 16):
    assert [k for k in range(middle)] == list(range(middle))

# Dispatch uses the existing GF61-only scratch in strict algorithm order.
needle = "if (fft.shape.fft_type == FFT3161 && fft.shape.middle >= 8)"
start = gpu.index(needle, gpu.index("if (kern == KMIDOUT)"))
block = gpu[start:gpu.index("} else {", start)]
ordered = [
    "kfftMidOutGF61LoadScalarApple(bufAppleTailMulGF61, *in);",
    "kfftMidOutGF61MulScalarApple(bufAppleTailMulGF61, bufTrigM);",
    "kfftMidOutGF61FftApple(bufAppleTailMulGF61);",
    "kfftMidOutGF61Mul2ScalarApple(bufAppleTailMulGF61, bufTrigM);",
    "kfftMidOutGF61WriteScalarApple(*out, bufAppleTailMulGF61);",
]
positions = [block.index(x) for x in ordered]
assert positions == sorted(positions)
assert "Buffer<double> bufAppleMiddleOut" not in hdr

print("Aevum Apple GF61 middle-out scalar five-stage test passed")
