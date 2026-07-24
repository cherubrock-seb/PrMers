# Use "make CUDA=1" for a CUDA build, use "make DEBUG=1" for a debug build

# The build artifacts are put in the "build-release" subfolder (or "build-debug" for a debug build).

# On Windows invoke with "make exe" or "make all"

# Uncomment below as desired to set a particular compiler or force a debug build:
# CXX = g++-12
# DEBUG = 1
# or export those into environment, or pass on the command line e.g.
# make all DEBUG=1 CXX=g++-12

HOST_OS = $(shell uname -s)
AEVUM_VERSION ?= v0.3.78-workload-plan-policy-audit-fix
MACOSX_DEPLOYMENT_TARGET ?= 12.0

# Use the platform default C++20 compiler.  On macOS, /usr/bin/c++ is
# AppleClang and is fully supported by the engine build.
CXX ?= c++

DL_LIBS = -ldl
ENGINE_LINK_FLAGS = -shared -Wl,-Bsymbolic
EXAMPLE_RPATH = -Wl,-rpath,'$$ORIGIN/../$(ENGINE_BIN)'
TEST_SECTION_FLAGS = -ffunction-sections -fdata-sections
TEST_GC_LINK = -Wl,--gc-sections
ifeq ($(HOST_OS), Darwin)
export MACOSX_DEPLOYMENT_TARGET
DARWIN_MIN_FLAGS = -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET)
DL_LIBS =
ENGINE_LINK_FLAGS = -dynamiclib -Wl,-install_name,@rpath/libaevum_engine.so $(DARWIN_MIN_FLAGS)
EXAMPLE_RPATH = -Wl,-rpath,@loader_path/../$(ENGINE_BIN)
TEST_GC_LINK = -Wl,-dead_strip
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

COMMON_FLAGS = -Wall $(CUDAFLAGS) -std=c++20 $(DARWIN_MIN_FLAGS)
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

SRCS1 = fs.cpp Trig.cpp TuneEntry.cpp Primes.cpp tune.cpp CycleFile.cpp TrigBufCache.cpp Event.cpp Queue.cpp TimeInfo.cpp Profile.cpp bundle.cpp OpenCLSourceBuilder.cpp Saver.cpp KernelCompiler.cpp Kernel.cpp gpuid.cpp File.cpp Proof.cpp log.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp FFTConfig.cpp AllocTrac.cpp sha3.cpp md5.cpp version.cpp EngineApi.cpp

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
ENGINE_FLAGS = -O3 -DNDEBUG -Wall -std=c++20 $(DARWIN_MIN_FLAGS) -fPIC -fvisibility=hidden -DAEVUM_ENGINE_LIBRARY
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
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) tests/engine_small_factor_gpu_test.cpp $(DL_LIBS) -o build-tests/aevum-engine-small-factor-gpu-test
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
WORKLOAD_PLAN_AUDIT := build-tests/aevum-workload-plan-audit

.PHONY: test-engine-api
test-engine-api: engine-lib $(ENGINE_API_TEST)
	$(ENGINE_API_TEST) $(ENGINE_LIB)

$(ENGINE_API_TEST): tests/engine_api_load_test.cpp
	@mkdir -p $(dir $@)
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) $< $(DL_LIBS) -o $@

.PHONY: workload-plan-audit-build
workload-plan-audit-build: engine-lib $(WORKLOAD_PLAN_AUDIT)

$(WORKLOAD_PLAN_AUDIT): tests/workload_plan_audit.cpp
	@mkdir -p $(dir $@)
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $< $(DL_LIBS) -o $@

HOST_TEST := build-tests/aevum-host-gf-test
STATE_TEST := build-tests/aevum-state-compact-test
OPENCL_STANDARD_TEST := build-tests/aevum-opencl-standard-test
MONOLITHIC_SOURCE_TEST := build-tests/aevum-opencl-monolithic-source-test
TYPE4_PLAN_TEST := build-tests/aevum-type4-pfa9-plan-test

.PHONY: test test-host test-gpu test-pfa9-lead-bridge-gpu

test: test-host

$(MONOLITHIC_SOURCE_TEST): tests/opencl_monolithic_source_test.cpp src/OpenCLSourceBuilder.cpp src/OpenCLSourceBuilder.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra tests/opencl_monolithic_source_test.cpp src/OpenCLSourceBuilder.cpp -o $@

test-host: $(MONOLITHIC_SOURCE_TEST) $(HOST_TEST) $(STATE_TEST) $(OPENCL_STANDARD_TEST) $(TYPE4_PLAN_TEST)
	$(MONOLITHIC_SOURCE_TEST)
	bash tests/opencl12_syntax_test.sh
	bash tests/pow2_type4_opencl_syntax.sh
	python3 tests/pfa9_lead_bridge_source_test.py
	bash tests/apple_opencl12_kernel_matrix_syntax.sh
	python3 tests/apple_gf61_ffthin_staging_test.py
	python3 tests/apple_gf31_width_staging_test.py
	python3 tests/apple_fused_fftp_test.py
	python3 tests/apple_gf61_scalar_mapping_test.py
	python3 tests/apple_gf61_width_staging_test.py
	python3 tests/apple_gf61_middlein_staging_test.py
	python3 tests/apple_gf61_middlein_restrict_alias_test.py
	python3 tests/apple_gf61_tail_two_kernel_policy_test.py
	python3 tests/apple_gf61_tailzero_staging_test.py
	python3 tests/apple_gf61_maintail_staging_test.py
	python3 tests/apple_gf61_fftw_staging_test.py
	python3 tests/apple_gf61_tailmul_staging_test.py
	python3 tests/apple_prepared_tailmullow_staging_test.py
	python3 tests/apple_gf61_middleout_staging_test.py
	python3 tests/apple_readchecked_double_sync_test.py
	python3 tests/apple_queue_marker_flush_test.py
	python3 tests/apple_generic_mul_safety_test.py
	python3 tests/apple_global_transpose_test.py
	python3 tests/apple_canonical_set_u32_test.py
	bash tests/source_audit.sh
	python3 tests/gpu_init_order_test.py
	python3 tests/engine_lead_cache_test.py
	$(HOST_TEST)
	$(STATE_TEST)
	$(OPENCL_STANDARD_TEST)
	$(TYPE4_PLAN_TEST)


$(TYPE4_PLAN_TEST): tests/type4_pfa9_plan_test.cpp src/FFTConfig.cpp src/FFTConfig.h src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc \
		tests/type4_pfa9_plan_test.cpp src/FFTConfig.cpp src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp \
		$(TEST_GC_LINK) -o $@

$(HOST_TEST): tests/host_gf_test.cpp
	@mkdir -p build-tests
	$(CXX) -O3 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $< -o $@

$(STATE_TEST): tests/state_compact_wrap_test.cpp src/state.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc $^ -o $@

$(OPENCL_STANDARD_TEST): tests/opencl_standard_test.cpp src/OpenCLStandard.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc $< -o $@

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
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Isrc $< $(ENGINE_LIB) $(EXAMPLE_RPATH) -o $@

test-examples-gpu: examples
	$(EXAMPLE_BIN)/fft_plans
	$(EXAMPLE_BIN)/register_ops $${AEVUM_TEST_DEVICE:-0} .
	$(EXAMPLE_BIN)/power_chain $${AEVUM_TEST_DEVICE:-0} .

.PHONY: native-pfa-host-test native-pfa-build native-pfa-gpu-test
native-pfa-host-test:
	python3 tools/native_pfa_reference_test.py
	python3 tools/native_pfa_source_audit.py
	bash tests/native_pfa_opencl_syntax.sh

native-pfa-build: engine-lib
	python3 tools/native_pfa_plan_test.py build-engine/libaevum_engine.so
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 -Wall -Wextra tests/native_pfa_engine_compare.cpp $(DL_LIBS) -o build-tests/native-pfa-engine-compare

native-pfa-gpu-test: native-pfa-build native-pfa-host-test
	bash scripts/test_native_pfa_gpu.sh $${AEVUM_TEST_DEVICE:-0} $${AEVUM_PFA_TEST_ITERS:-1}


test-pfa9-lead-bridge-gpu: engine-lib
	bash scripts/test_pfa9_lead_bridge_ubuntu.sh $${AEVUM_TEST_DEVICE:-1} $${AEVUM_TEST_EXPONENT:-175000039}
