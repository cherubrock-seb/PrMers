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

# The legacy PRP/LL loop must check every read, write and wait: a failed read left the host buffer as it was and
# that was used as the GPU's answer (a Gerbicz-Li check passed, a residue of 0 was reported or saved).
import re


def code_only(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


loop = code_only((ROOT / "src/modes/RunPrpOrLl.cpp").read_text())
for call in ("clEnqueueReadBuffer(", "clEnqueueWriteBuffer(", "clFinish(", "clWaitForEvents("):
    for m in re.finditer(re.escape(call), loop):
        before = loop[max(0, m.start() - 20):m.start()]
        assert before.endswith("checkLegacyCl("), "unchecked " + call + " in RunPrpOrLl.cpp: " + loop[m.start():m.start() + 120]
assert "clEnqueueMarkerWithWaitList" not in loop and "cl_event" not in loop, "leaked event in RunPrpOrLl.cpp"
assert "cl_uint ok = 0u;" in loop, "the Gerbicz-Li flag must start as failed on the host"

backup = code_only((ROOT / "src/core/BackupManager.cpp").read_text())
reads = [m.start() for m in re.finditer(r"clEnqueueReadBuffer\(", backup)]
assert len(reads) == 1, "BackupManager reads must go through readForSave"
helper = backup[backup.index("void readForSave("):]
assert helper.index("clEnqueueReadBuffer(") < helper.index("util::throwClError")

print("Legacy NTT enqueue error source regression: PASS")
