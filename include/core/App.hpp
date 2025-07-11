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
#include "math/GerbiczLiChecker.hpp"
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
    int run();
    void tuneIterforce();
    double measureIps(uint32_t testIterforce, uint32_t testIters);

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
  std::unique_ptr<math::GerbiczLiChecker> checker;
};

mpz_class buildE(uint64_t B1);
} // namespace core

#endif // CORE_APP_HPP
