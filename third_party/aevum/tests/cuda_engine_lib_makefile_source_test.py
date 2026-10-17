#!/usr/bin/env python3
from pathlib import Path

makefile = Path("Makefile").read_text()

assert "ENGINE_CUDA_SRCS = $(CUDASRCS1)" in makefile
assert "ENGINE_CUDA_OBJS = $(ENGINE_CUDA_SRCS:%.cpp=$(ENGINE_BIN)/cuda/%.o)" in makefile
assert "ENGINE_OBJS = $(ENGINE_CUDA_OBJS) $(ENGINE_SRCS:%.cpp=$(ENGINE_BIN)/%.o)" in makefile

engine_flags = next(
    line for line in makefile.splitlines()
    if line.startswith("ENGINE_FLAGS =")
)
assert "$(CUDAFLAGS)" in engine_flags

rule = """$(ENGINE_BIN)/%.o: src/%.cpp
	@mkdir -p $(dir $@) $(dir $(ENGINE_DEPDIR)/$*.d)
	$(CXX) $(ENGINE_FLAGS) $(CPPFLAGS) -MMD -MP -MF $(ENGINE_DEPDIR)/$*.d -c $< -o $@"""
assert rule in makefile

cuda_branch = makefile.split("ifeq ($(CUDA), 1)", 1)[1].split("else", 1)[0]
assert "CUDASRCS1 = clwrap_cuda.cpp cudawrap.cpp" in cuda_branch
assert "CUDAFLAGS = -DCUDA_BACKEND" in cuda_branch
assert "OPENCL_LIBS = -L/usr/local/cuda/lib64 -lcuda -lnvrtc" in cuda_branch

print("CUDA engine-lib Makefile source test PASS")
