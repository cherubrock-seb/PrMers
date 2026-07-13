# Use "make CUDA=1" for a CUDA build, use "make DEBUG=1" for a debug build

# The build artifacts are put in the "build-release" subfolder (or "build-debug" for a debug build).

# On Windows invoke with "make exe" or "make all"

# Uncomment below as desired to set a particular compiler or force a debug build:
# CXX = g++-12
# DEBUG = 1
# or export those into environment, or pass on the command line e.g.
# make all DEBUG=1 CXX=g++-12

HOST_OS = $(shell uname -s)
AEVUM_VERSION ?= v0.3.4

# Use the platform default C++20 compiler.  On macOS, /usr/bin/c++ is
# AppleClang and is fully supported by the engine build.
CXX ?= c++

DL_LIBS = -ldl
ENGINE_LINK_FLAGS = -shared -Wl,-Bsymbolic
EXAMPLE_RPATH = -Wl,-rpath,'$$ORIGIN/../$(ENGINE_BIN)'
ifeq ($(HOST_OS), Darwin)
DL_LIBS =
ENGINE_LINK_FLAGS = -dynamiclib
EXAMPLE_RPATH = -Wl,-rpath,@loader_path/../$(ENGINE_BIN)
endif

ifeq ($(CUDA), 1)
 BIN=build-cuda
 CUDASRCS1 = clwrap_cuda.cpp cudawrap.cpp
 CUDAFLAGS = -DCUDA_BACKEND -Isrc/cuda -I/usr/local/cuda/include
 CUDAOBJS = $(CUDASRCS1:%.cpp=$(BIN)/%.o)
 OPENCL_LIBS = -L/usr/local/cuda/lib64 -lcuda -lnvrtc
else
 BIN=build-release
 CUDAFLAGS =
 CUDAOBJS =
 ifeq ($(HOST_OS), Darwin)
  OPENCL_LIBS = -framework OpenCL
 else
  OPENCL_LIBS = -lOpenCL
 endif
endif

COMMON_FLAGS = -Wall $(CUDAFLAGS) -std=c++20
ifneq ($(HOST_OS),Darwin)
 COMMON_FLAGS += -static-libstdc++ -static-libgcc
 ifeq ($(findstring MINGW,$(HOST_OS)),MINGW)
  COMMON_FLAGS += -static
 endif
endif
# -fext-numeric-literals

ifeq ($(DEBUG), 1)

BIN=build-debug
CXXFLAGS = -g $(COMMON_FLAGS)
STRIP=

else

CXXFLAGS = -O3 -DNDEBUG $(COMMON_FLAGS)
STRIP=-s

endif

SRCS1 = fs.cpp Trig.cpp TuneEntry.cpp Primes.cpp tune.cpp CycleFile.cpp TrigBufCache.cpp Event.cpp Queue.cpp TimeInfo.cpp Profile.cpp bundle.cpp Saver.cpp KernelCompiler.cpp Kernel.cpp gpuid.cpp File.cpp Proof.cpp log.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp FFTConfig.cpp AllocTrac.cpp sha3.cpp md5.cpp version.cpp EngineApi.cpp

SRCS2 = test.cpp

# SRCS=$(addprefix src/, $(SRCS1))

OBJS = $(CUDAOBJS) $(SRCS1:%.cpp=$(BIN)/%.o)
DEPDIR := $(BIN)/.d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

all: aevum


ENGINE_BIN = build-engine
ENGINE_SRCS = $(filter-out main.cpp,$(SRCS1))
ENGINE_OBJS = $(ENGINE_SRCS:%.cpp=$(ENGINE_BIN)/%.o)
ENGINE_DEPDIR := $(ENGINE_BIN)/.d
ENGINE_FLAGS = -O3 -DNDEBUG -Wall -std=c++20 -fPIC -fvisibility=hidden -DAEVUM_ENGINE_LIBRARY
ENGINE_LIB = $(ENGINE_BIN)/libaevum_engine.so

.PHONY: engine-lib
engine-lib: $(ENGINE_LIB)

$(ENGINE_LIB): $(ENGINE_OBJS)
	$(CXX) $(ENGINE_LINK_FLAGS) -o $@ $^ $(LIBPATH) $(OPENCL_LIBS) $(DL_LIBS)

$(ENGINE_BIN)/%.o: src/%.cpp
	@mkdir -p $(dir $@) $(ENGINE_DEPDIR)
	$(CXX) $(ENGINE_FLAGS) $(CPPFLAGS) -MMD -MP -MF $(ENGINE_DEPDIR)/$*.d -c $< -o $@

aevum: $(BIN)/aevum

amd: $(BIN)/aevum-amd

#$(BIN)/test: $(BIN)/test.o
#	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBPATH) ${STRIP}

$(BIN)/aevum: ${OBJS}
	$(CXX) $(CXXFLAGS) -o $@ ${OBJS} $(LIBPATH) $(OPENCL_LIBS) ${STRIP}

# Instead of linking with libOpenCL, link with libamdocl64
$(BIN)/aevum-amd: ${OBJS}
	$(CXX) $(CXXFLAGS) -o $@ ${OBJS} $(LIBPATH) -lamdocl64 -L/opt/rocm/lib ${STRIP}

.PHONY: test-small-factor-gpu
test-small-factor-gpu: engine-lib
	$(CXX) -O2 -std=c++20 tests/engine_small_factor_gpu_test.cpp $(DL_LIBS) -o build-tests/aevum-engine-small-factor-gpu-test
	build-tests/aevum-engine-small-factor-gpu-test build-engine/libaevum_engine.so $${AEVUM_TEST_DEVICE:-0} .

clean:
	rm -rf build-debug build-release build-cuda build-engine build-tests build-examples

$(BIN)/%.o : src/%.cpp $(DEPDIR)/%.d
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

$(BIN)/%.o : src/cuda/%.cpp $(DEPDIR)/%.d
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

# src/bundle.cpp is just a wrapping of the OpenCL sources (*.cl) as a C string (as well as the CUDA OpenCL translation code)

src/bundle.cpp: genbundle.sh src/cuda/*.cuh src/cl/*.cl
	bash genbundle.sh $^ > src/bundle.cpp

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

src/version.cpp : src/version.inc

src/version.inc: FORCE
	@repo_root=$$(git rev-parse --show-toplevel 2>/dev/null || true); \
	here=$$(pwd -P); \
	if [ -n "$$repo_root" ] && [ "$$(cd "$$repo_root" && pwd -P)" = "$$here" ]; then \
	  desc=$$(git describe --tags --dirty --always --match 'v/aevum/*' --match 'v0.*' 2>/dev/null || echo "$(AEVUM_VERSION)"); \
	else \
	  desc="$(AEVUM_VERSION)"; \
	fi; \
	echo "\"$$desc\"" > $(BIN)/version.new
	@diff -q -N $(BIN)/version.new $@ >/dev/null || mv $(BIN)/version.new $@
	@echo Version: `cat $@`

FORCE:

include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS1))))
# include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS2))))



ENGINE_API_TEST := build-tests/aevum-engine-api-load-test

.PHONY: test-engine-api
test-engine-api: engine-lib $(ENGINE_API_TEST)
	$(ENGINE_API_TEST) $(ENGINE_LIB)

$(ENGINE_API_TEST): tests/engine_api_load_test.cpp
	@mkdir -p $(dir $@)
	$(CXX) -O2 -std=c++20 $< $(DL_LIBS) -o $@

HOST_TEST := build-tests/aevum-host-gf-test
STATE_TEST := build-tests/aevum-state-compact-test

.PHONY: test test-host test-gpu

test: test-host

test-host: $(HOST_TEST) $(STATE_TEST)
	bash tests/source_audit.sh
	$(HOST_TEST)
	$(STATE_TEST)

$(HOST_TEST): tests/host_gf_test.cpp
	@mkdir -p build-tests
	$(CXX) -O3 -std=c++20 -Wall -Wextra $< -o $@

$(STATE_TEST): tests/state_compact_wrap_test.cpp src/state.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 -Wall -Wextra -Isrc $^ -o $@

test-gpu: $(BIN)/aevum
	bash tests/gpu_prp_smoke.sh $(BIN)/aevum

-include $(wildcard $(ENGINE_DEPDIR)/*.d)

EXAMPLE_BIN := build-examples
EXAMPLE_NAMES := fft_plans register_ops power_chain
EXAMPLE_TARGETS := $(addprefix $(EXAMPLE_BIN)/,$(EXAMPLE_NAMES))

.PHONY: examples test-examples-gpu
examples: $(EXAMPLE_TARGETS)

$(EXAMPLE_BIN)/%: examples/%.cpp examples/example_common.h src/EngineApi.h $(ENGINE_LIB)
	@mkdir -p $(EXAMPLE_BIN)
	$(CXX) -O2 -std=c++20 -Isrc $< $(ENGINE_LIB) $(EXAMPLE_RPATH) -o $@

test-examples-gpu: examples
	$(EXAMPLE_BIN)/fft_plans
	$(EXAMPLE_BIN)/register_ops $${AEVUM_TEST_DEVICE:-0} .
	$(EXAMPLE_BIN)/power_chain $${AEVUM_TEST_DEVICE:-0} .
