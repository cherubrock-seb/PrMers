// Host test, no OpenCL device: what an OpenCL error status reads like to the user.
//
// A GPU that was reset or lost used to surface as a bare "OUT_OF_RESOURCES (-5) ..." from an enqueue, and a failed
// allocation as std::bad_alloc.  gpu_error now words a reset or lost GPU as such, and an allocation failure as out
// of memory.  The rule (GpuLost.h): CL_OUT_OF_RESOURCES is out of memory from a creation call and a lost GPU from
// anything that runs on the device; the other lost codes (-2, -34, -36, -9999) are a lost GPU in both.
#include "clwrap.h"
#include "GpuLost.h"

#include <cstdio>
#include <new>
#include <stdexcept>
#include <string>

static int failures = 0;

#define EXPECT(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } else { std::printf("ok:   %s\n", msg); } } while (0)

static bool has(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }

static std::string thrown(int err, const char* mes, gpulost::Phase phase, gpulost::Kind* kind = nullptr) {
  try {
    check(err, "file.cpp", 7, "func", mes, phase);
  } catch (const gpu_error& e) {
    if (kind) { *kind = e.kind; }
    return e.what();
  }
  return "(no throw)";
}

int main() {
  using gpulost::Phase;
  using gpulost::Kind;

  EXPECT(thrown(CL_SUCCESS, "clFinish(q)", Phase::Run) == "(no throw)", "CL_SUCCESS does not throw");

  // Run time: every code that means the GPU is gone says so, with the code, the call, and what to do.
  for (int code : {-5, -36, -34, -2, -9999}) {
    Kind kind{};
    std::string m = thrown(code, "clEnqueueNDRangeKernel(queue, kernel, ...)", Phase::Run, &kind);
    char what[96];
    std::snprintf(what, sizeof(what), "run-time %d is reported as a reset or lost GPU", code);
    EXPECT(kind == Kind::DeviceLost && m.rfind(gpulost::kLostPrefix, 0) == 0, what);
    EXPECT(has(m, errMes(code).c_str()), "the message names the code");
    EXPECT(has(m, "in clEnqueueNDRangeKernel."), "the message names the call, not the whole expression");
    EXPECT(has(m, "last checkpoint") && has(m, "restart PrMers"), "the message says the work is safe and a restart resumes");
    EXPECT(!has(m, "memory"), "the message does not mention memory");
  }

  // Setup time: a creation call out of resources is out of memory, not a reset.
  {
    Kind kind{};
    std::string m = thrown(-5, "clCreateBuffer", Phase::Create, &kind);
    EXPECT(kind == Kind::OutOfMemory && m.rfind("Out of GPU memory", 0) == 0, "creation-time -5 is reported as out of memory");
    EXPECT(has(m, "OUT_OF_RESOURCES (-5)") && has(m, "clCreateBuffer"), "the out-of-memory message names the code and the call");
    EXPECT(!has(m, "reset"), "the out-of-memory message does not claim a reset");
  }
  // The codes that cannot come from a healthy context are a lost GPU whichever call returned them.
  for (int code : {-2, -34, -36, -9999}) {
    Kind kind{};
    thrown(code, "clCreateContext", Phase::Create, &kind);
    EXPECT(kind == Kind::DeviceLost, "a lost-GPU code from a creation call is still a lost GPU");
  }
  EXPECT(thrown(-4, "clEnqueueFillBuffer(q)", Phase::Run).rfind("Out of GPU memory", 0) == 0, "-4 is out of memory at run time too");

  // Everything else keeps the existing text: code, call, location.
  {
    Kind kind{};
    std::string m = thrown(-54, "clEnqueueNDRangeKernel(q)", Phase::Run, &kind);
    EXPECT(kind == Kind::Other && m == "INVALID_WORK_GROUP_SIZE (-54) clEnqueueNDRangeKernel(q) at file.cpp:7 func", "an unrelated code keeps the generic message");
    m = thrown(-12345, "clFinish(q)", Phase::Run, &kind);
    EXPECT(kind == Kind::Other && m == " (-12345) clFinish(q) at file.cpp:7 func", "an unknown code keeps the generic message");
  }

  // The 2-argument constructor (no location) classifies the same way.
  {
    gpu_error lost(-36, "clFinish(q)");
    EXPECT(lost.kind == Kind::DeviceLost && has(lost.what(), "in clFinish."), "gpu_error(err, mes) classifies a run-time call");
    gpu_error oom(-5, "clCreateBuffer", Phase::Create);
    EXPECT(oom.kind == Kind::OutOfMemory, "gpu_error(err, mes, Create) classifies a creation call");
  }

  // clCreateBuffer's failure stays a std::bad_alloc, with a message that says what happened.
  {
    bool caught = false;
    std::string m;
    try {
      throw gpu_alloc_error(-5, size_t(3) << 30);
    } catch (const std::bad_alloc& e) {
      caught = true;
      m = e.what();
    }
    EXPECT(caught, "gpu_alloc_error is a std::bad_alloc");
    EXPECT(m.rfind("Out of GPU memory", 0) == 0 && has(m, "3072 MB") && has(m, "OUT_OF_RESOURCES (-5)"), "gpu_alloc_error says out of memory, how much, and the code");
  }

  // The vendor code has a name; the plain table is unchanged.
  EXPECT(errMes(-9999) == "CL_NVIDIA_DRIVER_ERROR (-9999)", "errMes names -9999");
  EXPECT(errMes(-5) == "OUT_OF_RESOURCES (-5)", "errMes is unchanged for -5");

  return failures ? 1 : 0;
}
