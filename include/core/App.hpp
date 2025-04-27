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
#include <optional>
#include <atomic>
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
    int run();

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
  double                             elapsed;
};

} // namespace core

#endif // CORE_APP_HPP
