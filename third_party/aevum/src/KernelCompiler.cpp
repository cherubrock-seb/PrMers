#include "KernelCompiler.h"
#include "Context.h"
#include "Sha3Hash.h"
#include "log.h"
#include "timeutil.h"
#include "Args.h"
#include "OpenCLStandard.h"
#include "OpenCLSourceBuilder.h"

#include <cassert>
#include <cinttypes>
#include <future>

using namespace std;

// Implemented in bundle.cpp
const std::vector<const char*>& getClFileNames();
const std::vector<const char*>& getClFiles();

static_assert(sizeof(Program) == sizeof(cl_program));

static string openclStandardArg(cl_device_id deviceId) {
#if defined(__APPLE__)
  return aevum_opencl_standard_arg(getOpenCLCVersion(deviceId),
                                   getOpenCLDeviceVersion(deviceId),
                                   true) + "-DAEVUM_APPLE_OPENCL12=1 ";
#else
  (void) deviceId;
  // Preserve the upstream PRPLL/Aevum build contract exactly on Linux,
  // Windows, OpenCL and CUDA backends.
  return "-cl-std=CL2.0 ";
#endif
}

// -cl-fast-relaxed-math  -cl-unsafe-math-optimizations -cl-denorms-are-zero -cl-mad-enable
// Other options:
// * -cl-uniform-work-group-size
// * -fno-bin-llvmir
// * various: -fno-bin-source -fno-bin-amdil

KernelCompiler::KernelCompiler(const Args& args, const Context* context, const string& clArgs) :
  cacheDir{args.cacheDir.string()},
  context{context->get()},
  linkArgs{"-cl-finite-math-only " },
  baseArgs{linkArgs + openclStandardArg(context->deviceId()) + clArgs},
  dump{args.dump},
  useCache{args.useCache},
  verbose{args.verbose},
  deviceId{context->deviceId()}
{

  string hw = getDriverVersion(deviceId) + ':' + getDeviceName(deviceId);
#if defined(__APPLE__) && !defined(CUDA_BACKEND)
  // Include the effective language/device versions in Apple cache keys because
  // the same hardware can receive a different legacy OpenCL frontend after a
  // macOS update. Keep the upstream cache key untouched everywhere else.
  hw += ':' + getOpenCLDeviceVersion(deviceId) + ':' + getOpenCLCVersion(deviceId);
#endif
  if (args.verbose) { log("OpenCL: %s, args %s\n", hw.c_str(), baseArgs.c_str()); }

  SHA3 hasher;
  hasher.update(hw);
  hasher.update(baseArgs);

  auto& clNames = getClFileNames();
  auto& clFiles = getClFiles();
  assert(clNames.size() == clFiles.size());
  int n = clNames.size();
  for (int i = 0; i < n; ++i) {
    auto &src = clFiles[i];
    files.push_back({clNames[i], src});
    clSources.push_back(loadSource(context->get(), src));

    hasher.update(clNames[i]);
    hasher.update(src);
  }
  contextHash = std::move(hasher).finish()[0];
  // log("OpenCL %d files, hash %016" PRIx64 "\n", n, contextHash);
}

Program KernelCompiler::compile(const string& fileName, const string& extraArgs) const {
  string args = baseArgs + ' ' + extraArgs;
  if (!dump.empty()) {
    args += " -save-temps="s + dump + "/" + fileName;
  }

#if defined(__APPLE__) && !defined(CUDA_BACKEND)
  // Apple's legacy OpenCL 1.2 stack can abort inside the separate
  // clCompileProgram + clLinkProgram route when bundled sources are supplied
  // as compile headers. Build one expanded source program instead.
  const string source = buildMonolithicOpenCLSource(fileName, files);
  Program program = loadSource(context, source);
  if (!program) throw runtime_error("Can't create Apple OpenCL source program for " + fileName);
  if (verbose) log("Apple OpenCL monolithic build begin: %s\n", fileName.c_str());
  const int err = clBuildProgram(program.get(), 1, &deviceId, args.c_str(), nullptr, nullptr);
  if (string mes = getBuildLog(program.get(), deviceId); !mes.empty()) { log("%s\n", mes.c_str()); }
  if (err != CL_SUCCESS) {
    log("Building '%s' error %s (args %s)\n", fileName.c_str(), errMes(err).c_str(), args.c_str());
    return {};
  }
  if (verbose) log("Apple OpenCL monolithic build end: %s\n", fileName.c_str());
  return program;
#else
  Program p1 = loadSource(context, "#include \""s + fileName + "\"\n");
  assert(p1);
#ifdef CUDA_BACKEND
  int err = clCompileProgram(p1.get(), 1, &deviceId, args.c_str(),
                             clSources.size(), (const cl_program*) (clSources.data()), getClFileNames().data(),
                             nullptr, nullptr);
#else
  // Skip first file (opencl_compat.cuh) if this is a standard openCL application rather than a CUDA translation
  int err = clCompileProgram(p1.get(), 1, &deviceId, args.c_str(),
                             clSources.size()-1, (const cl_program*) (clSources.data()+1), getClFileNames().data()+1,
                             nullptr, nullptr);
#endif
  if (string mes = getBuildLog(p1.get(), deviceId); !mes.empty()) { log("%s\n", mes.c_str()); }
  if (err != CL_SUCCESS) {
    log("Compiling '%s' error %s (args %s)\n", fileName.c_str(), errMes(err).c_str(), args.c_str());
    return {};
  }

  Program p2{clLinkProgram(context, 1, &deviceId, linkArgs.c_str(),
                           1, (cl_program *) &p1, nullptr, nullptr, &err)};
  if (string mes = getBuildLog(p1.get(), deviceId); !mes.empty()) { log("%s\n", mes.c_str()); }
  if (err != CL_SUCCESS) {
    log("Linking '%s' error %s (args %s)\n", fileName.c_str(), errMes(err).c_str(), linkArgs.c_str());
  }
  return p2;
#endif
}

static string to_hex(u64 d) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%016" PRIx64, d);
  return buf;
}

KernelHolder KernelCompiler::loadAux(const string& fileName, const string& kernelName, const string& args) const {
  Timer timer;
  bool fromCache = true;

  Program program;
  string cacheFile;

  if (useCache) {
    string f = kernelName + '-' + to_hex(SHA3::hash(contextHash, fileName, kernelName, args)[0]);
    cacheFile = cacheDir + '/' + f;
    program = loadBinary(context, deviceId, cacheFile);
  }

  if (!program) {
    fromCache = false;
    program = compile(fileName, args);
  }

  if (!program) {
    log("Can't compile %s\n", fileName.c_str());
    throw "Can't compile " + fileName;
  }

  KernelHolder ret;
#if defined(__APPLE__) && !defined(CUDA_BACKEND)
  string createError;
  auto tryCreate = [&](Program& candidate) -> KernelHolder {
    if (!candidate) return {};
    try {
      return KernelHolder{loadKernel(candidate.get(), kernelName.c_str())};
    } catch (const std::runtime_error& e) {
      createError = e.what();
      return {};
    }
  };

  ret = tryCreate(program);

  // A stale binary produced by a previous macOS/OpenCL configuration can build
  // successfully yet fail during Metal pipeline creation. Rebuild it once from
  // the current monolithic source before trying a lower-optimization fallback.
  if (!ret && fromCache) {
    if (verbose) {
      log("Apple OpenCL cached pipeline rejected for %s (%s); rebuilding source.\n",
          kernelName.c_str(), createError.c_str());
    }
    program = compile(fileName, args);
    fromCache = false;
    ret = tryCreate(program);
  }

  // Some Apple OpenCL-to-Metal revisions fail only in the optimized pipeline
  // compiler for large integer kernels. Retry the exact same source and work-
  // group contract with standard OpenCL optimization disabled. Linux, Windows,
  // CUDA, AMD and NVIDIA paths never enter this branch.
  if (!ret) {
    if (verbose) {
      log("Apple OpenCL pipeline retry for %s with -cl-opt-disable (%s).\n",
          kernelName.c_str(), createError.c_str());
    }
    Program conservative = compile(fileName, args + " -cl-opt-disable");
    KernelHolder retry = tryCreate(conservative);
    if (retry) {
      program = std::move(conservative);
      ret = std::move(retry);
      fromCache = false;
      if (verbose) log("Apple OpenCL conservative pipeline accepted: %s\n", kernelName.c_str());
    }
  }

  if (!ret && !createError.empty()) throw std::runtime_error(createError);
#else
  ret = KernelHolder{loadKernel(program.get(), kernelName.c_str())};
#endif
  if (!ret) {
    log("Can't find %s in %s\n", kernelName.c_str(), fileName.c_str());
    throw "Can't find "s + kernelName + " in " + fileName;
  }

  if (!fromCache) {
    if (useCache) {
      if (verbose) { log("saving binary to '%s'\n", cacheFile.c_str()); }
      saveBinary(program.get(), cacheFile);
    }
    if (verbose) { log("Loaded %s %s: %.0fms\n", kernelName.c_str(), args.c_str(), timer.at() * 1000); }
  }

  return ret;
}

std::future<KernelHolder> KernelCompiler::load(const string& fileName, const string& kernelName, const string& args) const {
#if 0
  // Do the compilation in parallel on a separate thread.
  // Unfortunatelly no benefit on ROCm (the compiler serializes).
  return async(std::launch::async, &KernelCompiler::loadAux, this, fileName, kernelName, args);
#else
  std::promise<KernelHolder> promise;
  promise.set_value(loadAux(fileName, kernelName, args));
  return promise.get_future();
#endif
}
