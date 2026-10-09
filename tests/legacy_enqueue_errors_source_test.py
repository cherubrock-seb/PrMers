from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

src = (ROOT / "src/opencl/NttEngine.cpp").read_text()


def throws(text):
    # A failed call raises through util::throwClError, which throws a std::runtime_error (or the GPU-lost
    # error) whatever the code is.
    return "throw std::runtime_error" in text or "util::throwClError" in text


# A failed stage enqueue must stop the run: continuing skips an NTT stage and
# silently corrupts LL and P-1 residues, which have no Gerbicz check.
start = src.index("static void executeKernelAndDisplay")
end = src.index("int NttEngine::forward(", start)
body = src[start:end]
enqueue = body.index("clEnqueueNDRangeKernel")
tail = body[enqueue:]
check = tail.index("if (err != CL_SUCCESS)")
block = tail[check:check + 400]
assert throws(block), block
assert "std::cerr << \"Kernel \"" not in block, block

# The pointwise multiply and temporary-buffer paths must check their calls.
pm = src[src.index("int NttEngine::pointwiseMul"):src.index("void NttEngine::squareInPlace")]
assert "const cl_int argErr = clSetKernelArg" in pm and throws(pm)

copy = src[src.index("void NttEngine::copy"):src.index("void NttEngine::mulInPlace(")]
assert "const cl_int err = clEnqueueCopyBuffer" in copy and throws(copy)

for name in ("void NttEngine::mulInPlace(", "void NttEngine::mulInPlace3("):
    fn = src[src.index(name):]
    fn = fn[:fn.index("clReleaseMemObject(temp)")]
    assert "clCreateBuffer" in fn
    assert "if (err != CL_SUCCESS)" in fn and throws(fn), name

print("Legacy NTT enqueue error source regression: PASS")
