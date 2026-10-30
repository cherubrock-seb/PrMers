// Copyright (C) Mihai Preda

#pragma once

#include "clwrap.h"

#include <atomic>
#include <chrono>
#include <vector>
#include <string>
#include <future>

class Args;
class Context;

class KernelCompiler {
  std::string cacheDir;
  cl_context context;
  std::string linkArgs;
  std::string baseArgs;
  std::string dump;
  const bool useCache;
  const bool useMemCache;
  const int verbose;
  
  std::vector<Program> clSources;
  std::vector<std::pair<std::string, std::string>> files;

  u64 contextHash{};
  
  Program compile(const string& fileName, const string& args) const;
  KernelHolder loadAux(const string& fileName, const string& kernelName, const string& args) const;

public:
  const cl_device_id deviceId;

  KernelCompiler(const Args& args, const Context* context, const string& clArgs);
  
  std::future<KernelHolder> load(const string& fileName, const string& kernelName, const string& args) const;

  // Total time spent loading kernels (compiling them, or reading them from the kernel cache).
  std::chrono::steady_clock::duration loadTime() const { return std::chrono::steady_clock::duration(loadTicks.load()); }

private:
  mutable std::atomic<std::chrono::steady_clock::rep> loadTicks{0};
};
