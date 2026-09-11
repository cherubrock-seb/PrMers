// Copyright (C) Mihai Preda

#include "Kernel.h"
#include "KernelCompiler.h"
#include "log.h"

#include <stdexcept>
#include <cstdlib>
#include <cstring>

Kernel::Kernel(string_view name, KernelCompiler* compiler, TimeInfo* timeInfo, Queue* queue,
       string_view fileName, string_view nameInFile,
       size_t workSize, string_view defines):
  name{name},
  compiler{compiler},
  fileName{fileName},
  nameInFile{nameInFile},
  defines{defines},
  timeInfo{timeInfo},
  queue{queue},
  workSize{workSize}
{
#ifndef CUDA_BACKEND
#if !defined(__APPLE__)
  const char* value = std::getenv("AEVUM_CACHE_BUFFER_ARGS");
  cacheBufferArgs = value && std::strcmp(value, "1") == 0;
#endif
#endif
}

Kernel::~Kernel() = default;

void Kernel::startLoad(KernelCompiler* compiler) {
  assert(!kernel);
  assert(!pendingKernel.valid());
  pendingKernel = compiler->load(fileName, nameInFile, defines);
  deviceId = compiler->deviceId;
}

void Kernel::finishLoad() {
  pendingKernel.wait();
  kernel = pendingKernel.get();
  assert(kernel);
  groupSize = getWorkGroupSize(kernel.get(), deviceId, name.c_str());
  assert(groupSize);
  assert(workSize % groupSize == 0);
#ifndef CUDA_BACKEND
  const char* profiling = std::getenv("AEVUM_PROFILE_KERNELS");
  if (profiling && std::strcmp(profiling,"1")==0) {
    // OpenCL 1.1+ query identifiers; private bytes are not a VGPR count.
    u64 localBytes=0, privateBytes=0; size_t preferred=0;
    const int lr=clGetKernelWorkGroupInfo(kernel.get(),deviceId,0x11B2,sizeof(localBytes),&localBytes,nullptr);
    const int pr=clGetKernelWorkGroupInfo(kernel.get(),deviceId,0x11B4,sizeof(privateBytes),&privateBytes,nullptr);
    const int wr=clGetKernelWorkGroupInfo(kernel.get(),deviceId,0x11B3,sizeof(preferred),&preferred,nullptr);
    log("AEVUM_RESOURCE name=%s wg=%u local_bytes=%llu private_bytes=%llu preferred=%zu rc=%d,%d,%d\n",
        name.c_str(),groupSize,static_cast<unsigned long long>(localBytes),
        static_cast<unsigned long long>(privateBytes),preferred,lr,pr,wr);
  }
#endif

  for (auto [pos, arg] : pendingArgs) { setArgs(pos, arg); }
}

void Kernel::run() {
  assert(kernel);
  queue->run(kernel.get(), groupSize, workSize, timeInfo);
}
