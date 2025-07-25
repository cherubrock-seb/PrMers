#ifndef CORE_APP_HPP
#define CORE_APP_HPP

#include "io/CliParser.hpp"
#include "io/WorktodoParser.hpp"
#include "opencl/Context.hpp"
#include "opencl/Program.hpp"
#include "math/Precompute.hpp"
#include "opencl/Buffers.hpp"
#include "opencl/Kernels.hpp"
#include "opencl/NttEngine.hpp"
#include "math/Carry.hpp"
#include "core/BackupManager.hpp"
#include "core/Spinner.hpp"
#include "core/Printer.hpp"
#include "core/QuickChecker.hpp"
#include "core/ProofManager.hpp"
#include "core/Logger.hpp"
#include "util/Timer.hpp"
#include "io/JsonBuilder.hpp"
#include "io/CurlClient.hpp"
#include <memory>
#include <optional>
#include <atomic>
#include <gmp.h>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

namespace core {

/// Top-level application driver.
class App {
public:
    App(int argc, char** argv);
    int runPrpOrLl();
    int runPM1();
    int runPM1Stage2();
    int run();
    void tuneIterforce();
    double measureIps(uint64_t testIterforce, uint64_t testIters);
    void gpuSquareInPlace(
                                    cl_mem A,
                                    math::Carry& carry,
                                    size_t limbBytes,
                                    cl_mem blockCarryBuf);
    void gpuMulInPlace(
                                 cl_mem A, cl_mem B,
                                 math::Carry& carry,
                                 size_t limbBytes,
                                 cl_mem blockCarryBuf);
    void gpuMulInPlace2(
                                 cl_mem A, cl_mem B,
                                 math::Carry& carry,
                                 size_t limbBytes,
                                 cl_mem blockCarryBuf);
    void gpuMulInPlace3(
                                 cl_mem A, cl_mem B,
                                 math::Carry& carry,
                                 size_t limbBytes,
                                 cl_mem blockCarryBuf);
    void gpuMulInPlace5(
                                 cl_mem A, cl_mem B,
                                 math::Carry& carry,
                                 size_t limbBytes,
                                 cl_mem blockCarryBuf);
    void subOneGPU(cl_mem buf);
    void gpuCopy(cl_command_queue q, cl_mem src, cl_mem dst, size_t bytes);

private:
  int    argc_;
  char** argv_;
  std::unique_ptr<io::WorktodoParser> worktodoParser_;
  bool hasWorktodoEntry_{false};
  io::CliOptions                     options;
  opencl::Context                    context;
  math::Precompute                   precompute;
  std::optional<opencl::Program>     program;
  std::optional<opencl::Buffers>     buffers;
  std::optional<opencl::Kernels>     kernels;
  std::optional<opencl::NttEngine>   nttEngine;
  BackupManager                      backupManager;
  ProofManager                       proofManager;
  Spinner                            spinner;
  Logger                             logger;
  util::Timer                        timer;
  util::Timer                        timer2;
  double                             elapsed;
};

mpz_class buildE(uint64_t B1);
void vectToMpz(mpz_t out,
                const std::vector<uint64_t>& v,
                const std::vector<int>& widths);
void readGpuBufferWithProgress(cl_command_queue q,
                               cl_mem            deviceBuf,
                               void*             hostPtr,
                               size_t            bytes,
                               const char*       msg);
} // namespace core

#endif // CORE_APP_HPP
