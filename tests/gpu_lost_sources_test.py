"""Source checks for the "GPU reset or lost" message wiring (see tests/gpu_lost_message_test.cpp for behaviour).

Host-only.  Guards the places where a one-line change would quietly bring back the old text: the two copies of the
status table, the phase given to Marin's creation calls, the Aevum plugin's pass-through, and the GUI log.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text()


def body_of(text, start, end):
    a = text.index(start)
    return text[a:text.index(end, a)]


# 1. Aevum keeps its own copy of the table; apart from the namespace, the guard, the comments and the exception
#    class, the code must be identical.
def code_only(text):
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"#(ifndef|define|endif)[^\n]*\n", "", text)
    text = re.sub(r"namespace\s+(util\s*\{\s*namespace\s+)?gpulost\s*\{", "", text)
    text = re.sub(r"\}\s*//\s*namespace[^\n]*", "", text)
    text = re.sub(r"#include <stdexcept>", "", text)
    text = re.sub(r"class GpuLostError.*?\n\};", "", text, flags=re.S)
    return re.sub(r"\s+", " ", text).strip().rstrip("} ")


root_h = code_only(read("include/util/GpuLost.hpp"))
aevum_h = code_only(read("third_party/aevum/src/GpuLost.h"))
assert root_h == aevum_h, "third_party/aevum/src/GpuLost.h drifted from include/util/GpuLost.hpp"

# 2. Marin: every creation call passes the Create phase, and the run calls name themselves.
ocl = read("include/marin/ocl.h")
for call in ("clCreateContext", "clCreateProgramWithSource", "clBuildProgram", "clCreateBuffer"):
    assert re.search(r'fatal\([^;\n]*"%s"[^;\n]*create_phase\)' % call, ocl), call
assert re.search(r"fatal\(err_ccqF,[^;\n]*create_phase\)", ocl)
assert re.search(r"fatal\(err_ccqP,[^;\n]*create_phase\)", ocl)
assert re.search(r"fatal\(err, kernel_name, create_phase\)", ocl)
create_buffer = body_of(ocl, "cl_mem _create_buffer(", "static void _release_buffer")
assert 'fatal(clFinish(_queue), "clFinish", create_phase)' in create_buffer, "zero-fill after a creation is part of it"
for call in ("clEnqueueReadBuffer", "clEnqueueWriteBuffer", "clEnqueueCopyBuffer"):
    assert re.search(r'fatal\(%s\([^;\n]*\), "%s"\)' % (call, call), ocl), call
assert "util/GpuLost.hpp" in ocl

# 3. Legacy NTT: the run calls use the Run phase, the creation calls the Create phase.
ntt = read("src/opencl/NttEngine.cpp")
assert "util::throwClError(err, util::gpulost::Phase::Run" in body_of(ntt, "static void executeKernelAndDisplay", "int NttEngine::forward(")
assert "util::throwClError(err, util::gpulost::Phase::Run" in body_of(ntt, "void NttEngine::copy(", "void NttEngine::mulInPlace(")
carry = read("src/math/Carry.cpp")
assert carry.count("util::gpulost::Phase::Run") == 6 and carry.count("util::gpulost::Phase::Create") == 4

# 4. Aevum: creation calls use CHECK_CREATE, run calls plain CHECK; clCreateBuffer throws the descriptive bad_alloc.
clwrap = read("third_party/aevum/src/clwrap.cpp")
for call in ("clCreateContext", "clCreateProgramWithSource", "clCreateCommandQueue", "clCreateBuffer"):
    assert 'CHECK_CREATE(err, "%s")' % call in clwrap, call
assert 'CHECK_CREATE(err, ("clCreateKernel "s + name))' in clwrap
assert "throw gpu_alloc_error(err, size)" in clwrap and "throw bad_alloc{}" not in clwrap
for fn in ("clEnqueueReadBuffer", "clEnqueueWriteBuffer", "clEnqueueCopyBuffer", "clEnqueueFillBuffer", "clFinish", "clFlush"):
    assert re.search(r"CHECK1\(%s\(" % fn, clwrap), fn

# 4b. The CUDA backend must not turn every failure into CL_OUT_OF_RESOURCES (a lost GPU at run time).
cuda = read("third_party/aevum/src/cuda/clwrap_cuda.cpp")
assert "CL_OUT_OF_RESOURCES" not in cuda, "the CUDA backend maps results through cuda_error_map"
assert cuda.count("clStatusFromCu(r)") >= 9 and "cuda_error_map::toClStatus" in cuda

# 5. The plugin's message reaches the host as the lost-GPU exception, and the GUI log gets it.
engine = read("src/aevum/EngineAevum.cpp")
assert "is_lost_message(detail)" in body_of(engine, "[[noreturn]] void fail(", "void require(")
app = read("src/core/App.cpp")
run = body_of(app, "int App::run()", "int App::runInner()")
assert "catch (const util::gpulost::GpuLostError& e)" in run and "guiServer_->appendLog" in run and "throw;" in run

print("GPU lost message source regression: PASS")
