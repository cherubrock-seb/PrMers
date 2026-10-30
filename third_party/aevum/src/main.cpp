// Copyright (C) Mihai Preda

#include "Args.h"
#include "Background.h"
#include "Queue.h"
#include "Signal.h"
#include "Task.h"
#include "Worktodo.h"
#include "version.h"
#include "AllocTrac.h"
#include "typeName.h"
#include "log.h"
#include "Context.h"
#include "TrigBufCache.h"
#include "GpuCommon.h"
#include "Gpu.h"
#include "tune.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <thread>
// #include <format> from GCC-13 onwards

void gpuWorker(GpuCommon shared, i32 instance) {
  // LogContext context{(instance ? shared.args->tailDir() : ""s) + to_string(instance) + ' '};
  // log("Starting worker %d\n", instance);
  if (instance > 0) {
    initLog(("aevum-"s + to_string(instance) + ".log").c_str());
    log("Aevum %s, instance %d\n", VERSION, instance);
  }

  try {
    while (auto task = Worktodo::getTask(*shared.args, instance)) { task->execute(shared, instance); }
  } catch (const char *mes) {
    log("Exception \"%s\"\n", mes);
  } catch (const string& mes) {
    log("Exception \"%s\"\n", mes.c_str());
  } catch (const std::exception& e) {
    log("Exception %s: %s\n", typeName(e), e.what());
  }
}

namespace {

class ThreadJoiner {
  vector<thread>& threads_;

public:
  explicit ThreadJoiner(vector<thread>& threads) : threads_(threads) {}
  ThreadJoiner(const ThreadJoiner&) = delete;
  ThreadJoiner& operator=(const ThreadJoiner&) = delete;

  ~ThreadJoiner() {
    for (auto& worker : threads_) {
      if (worker.joinable()) worker.join();
    }
  }
};

} // namespace


#if defined(__MINGW32__) || defined(__MINGW64__) || defined(__MSYS__) // for Windows
extern int putenv(char *);
#endif

// The exceptions that end a run on purpose: the user's stop, and the flags
// that only print (-h, -info, -version). Everything else thrown to main() is
// a failure.
static bool isCleanExit(const char *reason) {
  return !strcmp(reason, "stop requested") || !strcmp(reason, "help") || !strcmp(reason, "info") || !strcmp(reason, "version");
}

int main(int argc, char **argv) {
//!MSVC version support
#ifdef _MSC_VER
  _set_printf_count_output(1);    // I'm not sure what this does (it's from CrazeTheDragon)
#endif

#if defined(__MSYS__)
  // I was unable to get putenv to link in MSYS2
#elif defined(__MINGW32__) || defined(__MINGW64__)
  putenv("ROC_SIGNAL_POOL_SIZE=32");
#elif defined(_WIN32)
  _putenv_s("ROC_SIGNAL_POOL_SIZE", "32");  // For MSVC
#else
  // Required to work around a ROCm bug when using multiple queues
  setenv("ROC_SIGNAL_POOL_SIZE", "32", 0);
#endif

  int exitCode = 0;

  try {
    string mainLine = Args::mergeArgs(argc, argv);
    {
      Args args{true};
      args.parse(mainLine);
      if (!args.dir.empty()) {
        std::error_code ec;
        fs::current_path(args.dir, ec);
        if (ec) { throw std::runtime_error("cannot use -dir '" + args.dir + "': " + ec.message()); }
      }
    }

    fs::path poolDir;
    {
      Args args{true};
      args.readConfig("config.txt");
      args.parse(mainLine);
      poolDir = args.masterDir;
    }

    initLog("aevum-0.log");
    log("Aevum %s starting\n", VERSION);

    Args args;

    if (!poolDir.empty()) { args.readConfig(poolDir / "config.txt"); }
    args.readConfig("config.txt");
    args.parse(mainLine);
    args.setDefaults();

    if (args.maxAlloc) { AllocTrac::setMaxAlloc(args.maxAlloc); }

    Context context(getDevice(args.device));
    Signal signal;
    Background background;
    GpuCommon shared;
    shared.context = &context;
    shared.args = &args;
    TrigBufCache bufCache{&context};
    shared.bufCache = &bufCache;
    shared.background = &background;

    if (args.doCtune || args.doTune || args.doZtune || args.carryTune) {
      Tune tune{shared};

      if (args.doCtune) {
        tune.ctune();
      } else if (args.doTune) {
        tune.tune();
      } else if (args.doZtune) {
        tune.ztune();
      } else if (args.carryTune) {
        tune.carryTune();
      }
    } else {
      {
        vector<thread> threads;
        ThreadJoiner joiner{threads};
        for (int i = 1; i < int(args.workers); ++i) {
          threads.emplace_back(gpuWorker, shared, i);
        }
        gpuWorker(shared, 0);
      }

      // log("No more work. Add work to worktodo.txt , see -h for details.\n");
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
    exitCode = isCleanExit(mes) ? 0 : 1;
  } catch (const string& mes) {
    log("Exiting because \"%s\"\n", mes.c_str());
    exitCode = isCleanExit(mes.c_str()) ? 0 : 1;
  } catch (const std::exception& e) {
    // E.g. -dir naming a directory that does not exist.  Like the cases above this is logged and ends the run, with status 1.
    log("Exiting because of exception %s: %s\n", typeName(e), e.what());
    exitCode = 1;
  }

  log("Bye\n");
  return exitCode;
}
