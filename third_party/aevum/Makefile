# Use "make CUDA=1" for a CUDA build, use "make DEBUG=1" for a debug build

# The build artifacts are put in the "build-release" subfolder (or "build-debug" for a debug build).

# On Windows invoke with "make exe" or "make all"

# Uncomment below as desired to set a particular compiler or force a debug build:
# CXX = g++-12
# DEBUG = 1
# or export those into environment, or pass on the command line e.g.
# make all DEBUG=1 CXX=g++-12

HOST_OS = $(shell uname -s)
AEVUM_VERSION ?= v0.3.110-pass5-prp-max
MACOSX_DEPLOYMENT_TARGET ?= 12.0

# Use the platform default C++20 compiler.  On macOS, /usr/bin/c++ is
# AppleClang and is fully supported by the engine build.
CXX ?= c++

DL_LIBS = -ldl
LOADER_LIBS = $(DL_LIBS)

ifneq ($(findstring _NT,$(HOST_OS)),)
DL_LIBS =
LOADER_LIBS =
endif

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

SRCS1 = fs.cpp Trig.cpp TuneEntry.cpp Primes.cpp tune.cpp CycleFile.cpp TrigBufCache.cpp Event.cpp Queue.cpp TimeInfo.cpp Profile.cpp bundle.cpp OpenCLSourceBuilder.cpp Saver.cpp KernelCompiler.cpp Kernel.cpp gpuid.cpp File.cpp Proof.cpp log.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp FFTConfig.cpp AllocTrac.cpp sha3.cpp md5.cpp version.cpp RuntimeAutotune.cpp PrpUseTune.cpp EngineApi.cpp

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
ENGINE_CUDA_SRCS = $(CUDASRCS1)
ENGINE_CUDA_OBJS = $(ENGINE_CUDA_SRCS:%.cpp=$(ENGINE_BIN)/cuda/%.o)
ENGINE_OBJS = $(ENGINE_CUDA_OBJS) $(ENGINE_SRCS:%.cpp=$(ENGINE_BIN)/%.o)
ENGINE_DEPDIR := $(ENGINE_BIN)/.d
ENGINE_FLAGS = -O3 -DNDEBUG -Wall -std=c++20 $(DARWIN_MIN_FLAGS) -fPIC -fvisibility=hidden -DAEVUM_ENGINE_LIBRARY $(CUDAFLAGS)
ENGINE_LIB = $(ENGINE_BIN)/libaevum_engine.so

.PHONY: engine-lib
engine-lib: $(ENGINE_LIB)

$(ENGINE_LIB): $(ENGINE_OBJS)
	$(CXX) $(ENGINE_LINK_FLAGS) -o $@ $^ $(LIBPATH) $(OPENCL_LIBS)

$(ENGINE_BIN)/%.o: src/%.cpp
	@mkdir -p $(dir $@) $(dir $(ENGINE_DEPDIR)/$*.d)
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

# Device test (OpenCL; PoCL is enough): a faulting PRP -use tuner probe must not take the production engine down.
.PHONY: test-prp-use-isolation-gpu
test-prp-use-isolation-gpu: engine-lib
	bash tests/prp_use_isolation_gpu.sh $(ENGINE_LIB) $${AEVUM_TEST_DEVICE:-0}

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
FFT_EXPLICIT_CAPACITY_TEST := build-tests/aevum-fft-explicit-capacity-test
ARGS_VALUE_FOR_TEST := build-tests/aevum-args-value-for-test
FFT_SIZE_ONLY_SPEC_TEST := build-tests/aevum-fft-size-only-spec-test
FFT_SPEC_ROBUSTNESS_TEST := build-tests/aevum-fft-spec-robustness-test
EVENT_STATUS_TEST := build-tests/aevum-event-status-test
FFT8_SKIP1_TEST := build-tests/aevum-fft8-skip1-test
WMUL_LIMIT_TEST := build-tests/aevum-wmul-limit-test
FFT_VARIANT0_1K_TEST := build-tests/aevum-fft-variant0-1k-test
AUTOTUNE_CACHE_TEST := build-tests/aevum-runtime-autotune-cache-test

.PHONY: test test-host test-fft-variant0-1k test-gpu test-pfa9-lead-bridge-gpu

test: test-host

$(MONOLITHIC_SOURCE_TEST): tests/opencl_monolithic_source_test.cpp src/OpenCLSourceBuilder.cpp src/OpenCLSourceBuilder.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra tests/opencl_monolithic_source_test.cpp src/OpenCLSourceBuilder.cpp -o $@

test-host: $(MONOLITHIC_SOURCE_TEST) $(HOST_TEST) $(STATE_TEST) $(OPENCL_STANDARD_TEST) $(TYPE4_PLAN_TEST) $(FFT_EXPLICIT_CAPACITY_TEST) $(ARGS_VALUE_FOR_TEST) $(FFT_SIZE_ONLY_SPEC_TEST) $(FFT_SPEC_ROBUSTNESS_TEST) $(AUTOTUNE_CACHE_TEST) $(EVENT_STATUS_TEST) $(FFT8_SKIP1_TEST) $(FFT_VARIANT0_1K_TEST) $(WMUL_LIMIT_TEST)
	$(MONOLITHIC_SOURCE_TEST)
	bash tests/opencl12_syntax_test.sh
	bash tests/pow2_type4_opencl_syntax.sh
	bash tests/carryfused_no_generic_as_syntax.sh
	python3 tests/pfa9_lead_bridge_source_test.py
	bash tests/apple_opencl12_kernel_matrix_syntax.sh
	python3 tests/apple_gf61_ffthin_staging_test.py
	python3 tests/apple_ffthin_preserves_input_test.py
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
	python3 tests/readroe_clears_mulroepos_test.py
	python3 tests/tail_kernels_range_source_test.py
	python3 tests/engine_autotune_isolation_test.py
	python3 tests/engine_lead_cache_test.py
	python3 tests/native_tune_compat_source_test.py
	python3 tests/cuda_regcf3161_auto_policy_source_test.py
	python3 tests/nvidia_cc_literal_test.py
	python3 tests/cuda_engine_lib_makefile_source_test.py
	python3 tests/cuda_context_lifecycle_source_test.py
	python3 tests/middlein_gf61_cmul_source_test.py
	python3 tests/subgroup_barrier_lockstep_test.py
	python3 tests/asm_string_braces_test.py
	python3 tests/pfa7_separate_carryb_source_test.py
	python3 tests/pfa7_nuss8_model_test.py
	$(HOST_TEST)
	$(STATE_TEST)
	$(OPENCL_STANDARD_TEST)
	$(TYPE4_PLAN_TEST)
	$(FFT_EXPLICIT_CAPACITY_TEST)
	$(ARGS_VALUE_FOR_TEST)
	$(FFT_SIZE_ONLY_SPEC_TEST)
	$(FFT_SPEC_ROBUSTNESS_TEST)
	$(AUTOTUNE_CACHE_TEST)
	$(EVENT_STATUS_TEST)
	$(FFT8_SKIP1_TEST)
	$(WMUL_LIMIT_TEST)
	$(FFT_VARIANT0_1K_TEST)

$(FFT_VARIANT0_1K_TEST): tests/fft_variant0_1k_test.cpp src/clwrap.cpp src/Trig.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(wildcard src/cl/*.cl)
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/fft_variant0_1k_test.cpp src/clwrap.cpp src/Trig.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(OPENCL_LIBS) -o $@

test-fft-variant0-1k: $(FFT_VARIANT0_1K_TEST)
	$(FFT_VARIANT0_1K_TEST)

# Device test: FP64 radix-4 variant 2 width FFT (partial_tabMul4) against a reference DFT.
FFT_RADIX4_VARIANT2_TEST := build-tests/aevum-fft-radix4-variant2-test
.PHONY: test-fft-radix4-variant2
test-host: test-fft-radix4-variant2
test-fft-radix4-variant2: $(FFT_RADIX4_VARIANT2_TEST)
	$(FFT_RADIX4_VARIANT2_TEST)

$(FFT_RADIX4_VARIANT2_TEST): tests/fft_radix4_variant2_test.cpp src/TrigBufCache.cpp src/clwrap.cpp src/Trig.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(wildcard src/cl/*.cl)
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc tests/fft_radix4_variant2_test.cpp src/TrigBufCache.cpp src/clwrap.cpp src/Trig.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(OPENCL_LIBS) $(TEST_GC_LINK) -o $@

$(WMUL_LIMIT_TEST): tests/wmul_limit_test.cpp src/WmulLimit.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/wmul_limit_test.cpp -o $@

$(FFT8_SKIP1_TEST): tests/fft8_skip1_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(wildcard src/cl/*.cl)
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/fft8_skip1_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(OPENCL_LIBS) -o $@

# KernelCompiler: a failed clLinkProgram is a compile failure; the link log comes from the linked program (device test)
KERNEL_LINK_FAILURE_TEST := build-tests/aevum-kernel-link-failure-test
.PHONY: test-kernel-link-failure
test-host: test-kernel-link-failure
test-kernel-link-failure: $(KERNEL_LINK_FAILURE_TEST)
	$(KERNEL_LINK_FAILURE_TEST)
$(KERNEL_LINK_FAILURE_TEST): tests/kernel_link_failure_test.cpp src/KernelCompiler.cpp src/KernelCompiler.h src/bundle.cpp src/OpenCLSourceBuilder.cpp src/clwrap.cpp src/sha3.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp src/Args.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc \
		tests/kernel_link_failure_test.cpp src/KernelCompiler.cpp src/bundle.cpp src/OpenCLSourceBuilder.cpp src/clwrap.cpp src/sha3.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp src/Args.cpp \
		$(TEST_GC_LINK) $(OPENCL_LIBS) $(DL_LIBS) -o $@

$(EVENT_STATUS_TEST): tests/event_status_test.cpp src/Queue.cpp src/Event.cpp src/TimeInfo.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/timeutil.cpp src/common.cpp src/fs.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -fno-access-control -Wall -Wextra -Isrc $^ $(OPENCL_LIBS) -o $@

$(AUTOTUNE_CACHE_TEST): tests/runtime_autotune_cache_test.cpp src/RuntimeAutotune.cpp src/RuntimeAutotune.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/runtime_autotune_cache_test.cpp src/RuntimeAutotune.cpp -o $@

# tune.txt / config.txt: accept a last line with no final newline (worktodo stays strict)
UNTERMINATED_LINE_TEST := build-tests/aevum-unterminated-last-line-test
.PHONY: test-unterminated-last-line
test-host: test-unterminated-last-line
test-unterminated-last-line: $(UNTERMINATED_LINE_TEST)
	$(UNTERMINATED_LINE_TEST)
$(UNTERMINATED_LINE_TEST): tests/unterminated_last_line_test.cpp src/File.h src/File.cpp src/TuneEntry.cpp src/FFTConfig.cpp src/FFTConfig.h src/Args.cpp src/common.cpp src/fs.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc \
		tests/unterminated_last_line_test.cpp src/PrpUseTune.cpp src/RuntimeAutotune.cpp src/FFTConfig.cpp src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp \
		$(TEST_GC_LINK) -o $@

$(TYPE4_PLAN_TEST): tests/type4_pfa9_plan_test.cpp src/FFTConfig.cpp src/FFTConfig.h src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc \
		tests/type4_pfa9_plan_test.cpp src/PrpUseTune.cpp src/RuntimeAutotune.cpp src/FFTConfig.cpp src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp \
		$(TEST_GC_LINK) -o $@

$(FFT_SIZE_ONLY_SPEC_TEST): tests/fft_size_only_spec_test.cpp src/FFTConfig.cpp src/FFTConfig.h src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc \
		tests/fft_size_only_spec_test.cpp src/PrpUseTune.cpp src/RuntimeAutotune.cpp src/FFTConfig.cpp src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp \
		$(TEST_GC_LINK) -o $@

$(FFT_SPEC_ROBUSTNESS_TEST): tests/fft_spec_robustness_test.cpp src/FFTConfig.cpp src/FFTConfig.h src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc \
		tests/fft_spec_robustness_test.cpp src/PrpUseTune.cpp src/RuntimeAutotune.cpp src/FFTConfig.cpp src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp \
		$(TEST_GC_LINK) -o $@

$(FFT_EXPLICIT_CAPACITY_TEST): tests/fft_explicit_capacity_test.cpp src/FFTConfig.cpp src/FFTConfig.h src/fftbpw.h src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc \
		tests/fft_explicit_capacity_test.cpp src/PrpUseTune.cpp src/RuntimeAutotune.cpp src/FFTConfig.cpp src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp \
		$(TEST_GC_LINK) -o $@

# -verbose [<level>]: the developer self-tests run only at level 99
VERBOSE_LEVEL_TEST := build-tests/aevum-verbose-level-test
.PHONY: test-verbose-level
test-host: test-verbose-level
test-verbose-level: $(VERBOSE_LEVEL_TEST)
	$(VERBOSE_LEVEL_TEST)
$(VERBOSE_LEVEL_TEST): tests/verbose_level_test.cpp src/Args.cpp src/Args.h src/StrictInt.h src/FFTConfig.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc \
		tests/verbose_level_test.cpp src/Args.cpp src/PrpUseTune.cpp src/RuntimeAutotune.cpp src/FFTConfig.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp \
		$(TEST_GC_LINK) -o $@

$(ARGS_VALUE_FOR_TEST): tests/args_value_for_test.cpp src/Args.cpp src/Args.h src/FFTConfig.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc \
		tests/args_value_for_test.cpp src/Args.cpp src/PrpUseTune.cpp src/RuntimeAutotune.cpp src/FFTConfig.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp \
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

.PHONY: test-gpu-pfa-ll
test-gpu-pfa-ll: $(BIN)/aevum
	bash tests/gpu_pfa_ll_test.sh $(BIN)/aevum

# Proof generation failures end to end (slow: one full PRP of 2^400009-1 plus restarts; device test, not in test-host)
.PHONY: test-gpu-proof-retry
test-gpu-proof-retry: $(BIN)/aevum
	bash tests/proof_retry_gpu_test.sh $(BIN)/aevum


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
	$(CXX) -O2 -std=c++20 -Wall -Wextra tests/native_pfa_dense_compare.cpp $(DL_LIBS) -o build-tests/native-pfa-dense-compare

native-pfa-gpu-test: native-pfa-build native-pfa-host-test
	bash scripts/test_native_pfa_gpu.sh $${AEVUM_TEST_DEVICE:-0} $${AEVUM_PFA_TEST_ITERS:-1}


test-pfa9-lead-bridge-gpu: engine-lib
	bash scripts/test_pfa9_lead_bridge_ubuntu.sh $${AEVUM_TEST_DEVICE:-1} $${AEVUM_TEST_EXPONENT:-175000039}

.PHONY: test-prp-use
test-prp-use: $(ENGINE_LIB)
	$(CXX) -O2 -std=c++20 -Isrc tests/prp_use_test.cpp $(ENGINE_OBJS) $(OPENCL_LIBS) $(DL_LIBS) -o $(ENGINE_BIN)/prp-use-test
	$(ENGINE_BIN)/prp-use-test

# PR131 cumulative semantic integration
SHUFL_PERMUTATION_TEST := build-tests/aevum-shufl-permutation-test
.PHONY: test-shufl-permutation
test-host: test-shufl-permutation
test-shufl-permutation: $(SHUFL_PERMUTATION_TEST)
	$(SHUFL_PERMUTATION_TEST)
$(SHUFL_PERMUTATION_TEST): tests/shufl_permutation_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(wildcard src/cl/*.cl)
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/shufl_permutation_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(OPENCL_LIBS) -o $@
# FP32 slowTrig accuracy (trigCoefsFP32 + trig.cl reducedCosSin emulated in float).  -ffp-contract=off: the emulation's
# multiplies and fmas must round exactly as written.
TRIG_FP32_TEST := build-tests/aevum-trig-fp32-accuracy-test
.PHONY: test-trig-fp32-accuracy
test-host: test-trig-fp32-accuracy
test-trig-fp32-accuracy: $(TRIG_FP32_TEST)
	$(TRIG_FP32_TEST)
$(TRIG_FP32_TEST): tests/trig_fp32_accuracy_test.cpp src/Trig.cpp src/Trig.h src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 -ffp-contract=off $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/trig_fp32_accuracy_test.cpp src/Trig.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp -o $@
# PR132 cumulative semantic integration
STATS_SLOT_TEST := build-tests/aevum-stats-slot-test
.PHONY: test-pr132-small-items test-stats-slot
test-host: test-pr132-small-items
test-pr132-small-items: test-stats-slot
	python3 tests/kernel_config_guards_test.py
	python3 tests/carryfused_cs_fence_test.py
	python3 tests/carryfused_weight_shift_modulus_test.py
	python3 tests/carry_gf61_seed_overflow_test.py
	python3 tests/prp_middle1_type_gate_test.py
test-stats-slot: $(STATS_SLOT_TEST)
	$(STATS_SLOT_TEST)
$(STATS_SLOT_TEST): tests/stats_slot_test.cpp src/StatsSlot.h src/common.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/stats_slot_test.cpp -o $@
# STATS: boundCarry(i64) measures 64-bit-only carries against 2^63 (device test; any OpenCL device)
BOUNDCARRY_STATS_TEST := build-tests/aevum-boundcarry-stats-test
.PHONY: test-boundcarry-stats
test-host: test-boundcarry-stats
test-boundcarry-stats: $(BOUNDCARRY_STATS_TEST)
	$(BOUNDCARRY_STATS_TEST)
$(BOUNDCARRY_STATS_TEST): tests/boundcarry_stats_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(wildcard src/cl/*.cl)
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/boundcarry_stats_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(OPENCL_LIBS) -o $@
# PR133 cumulative semantic integration
APPLE_FFTP61_SEED_TEST := build-tests/aevum-apple-fftp61-seed-test
.PHONY: test-pr133-small-items test-apple-fftp61-seed
test-host: test-pr133-small-items
test-pr133-small-items:
	python3 tests/apple_fftp61_seed_overflow_test.py
test-apple-fftp61-seed: $(APPLE_FFTP61_SEED_TEST)
	$(APPLE_FFTP61_SEED_TEST)
$(APPLE_FFTP61_SEED_TEST): tests/apple_fftp61_scalar_seed_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(wildcard src/cl/*.cl)
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/apple_fftp61_scalar_seed_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(OPENCL_LIBS) -o $@
# math.cl: every i96 implementation provides both make_i96(i64, u32) and make_i96(i64, u64) (device test; any OpenCL device)
I96_MAKE_TEST := build-tests/aevum-i96-make-test
.PHONY: test-i96-make
test-host: test-i96-make
test-i96-make: $(I96_MAKE_TEST)
	$(I96_MAKE_TEST)
$(I96_MAKE_TEST): tests/i96_make_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/i96_make_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(OPENCL_LIBS) -o $@
# PR134 cumulative semantic integration
APPLE_TAILZERO_TEST := build-tests/aevum-apple-tailzero-halfline-test
.PHONY: test-pr134-small-items test-apple-tailzero-halfline
test-host: test-pr134-small-items
test-pr134-small-items: test-apple-tailzero-halfline
test-apple-tailzero-halfline: $(APPLE_TAILZERO_TEST)
	$(APPLE_TAILZERO_TEST)

$(APPLE_TAILZERO_TEST): tests/apple_tailzero_halfline_test.cpp src/AppleTailZero.h src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(wildcard src/cl/*.cl)
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/apple_tailzero_halfline_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(OPENCL_LIBS) -o $@

# PR135 cumulative semantic integration
RADIX1K_POLICY_TEST := build-tests/aevum-radix1k-policy-test
RADIX1K_POLICY_APPLE_TEST := build-tests/aevum-radix1k-policy-apple-test
.PHONY: test-pr135-small-items test-radix1k-policy
test-host: test-pr135-small-items
test-pr135-small-items: test-radix1k-policy
test-radix1k-policy: $(RADIX1K_POLICY_TEST) $(RADIX1K_POLICY_APPLE_TEST)
	$(RADIX1K_POLICY_TEST)
	$(RADIX1K_POLICY_APPLE_TEST)

$(RADIX1K_POLICY_TEST): tests/radix1k_policy_test.cpp src/FFTConfig.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/radix1k_policy_test.cpp -o $@

$(RADIX1K_POLICY_APPLE_TEST): tests/radix1k_policy_test.cpp src/FFTConfig.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -DAEVUM_FORCE_RADIX4_1K=1 -Isrc tests/radix1k_policy_test.cpp -o $@

# The CUDA backend reports a sticky CUDA error from clGetEventInfo / clFinish (plain ints, no CUDA needed).
CUDA_EVENT_STATUS_TEST := build-tests/aevum-cuda-event-status-test
.PHONY: test-cuda-event-status
test-host: test-cuda-event-status
test-cuda-event-status: $(CUDA_EVENT_STATUS_TEST)
	$(CUDA_EVENT_STATUS_TEST)
$(CUDA_EVENT_STATUS_TEST): tests/cuda_event_status_test.cpp src/cuda/CudaEventStatus.h src/cuda/CudaErrorMap.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/cuda_event_status_test.cpp -o $@

# PR136 cumulative semantic integration
.PHONY: test-pr136-small-items
test-host: test-pr136-small-items
test-pr136-small-items:
	python3 tests/kernel_inline_linkage_test.py

# PR137 cumulative semantic integration
GF61_LIMB32_CMUL_TEST := build-tests/aevum-gf61-limb32-cmul-test
.PHONY: test-pr137-small-items test-gf61-limb32-cmul
test-host: test-pr137-small-items
test-pr137-small-items: test-gf61-limb32-cmul
test-gf61-limb32-cmul: $(GF61_LIMB32_CMUL_TEST)
	$(GF61_LIMB32_CMUL_TEST)

$(GF61_LIMB32_CMUL_TEST): tests/gf61_limb32_cmul_kernel_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(wildcard src/cl/*.cl)
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/gf61_limb32_cmul_kernel_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(OPENCL_LIBS) -o $@

# fftMiddleOutGF61 direct base table (AEVUM_GF61_MIDDLE_BASE_TABLE) against the stock twiddles
GF61_MIDDLE_BASE_TABLE_TEST := build-tests/aevum-gf61-middle-base-table-test
GF61_MIDDLE_BASE_TABLE_SRCS := src/clwrap.cpp src/log.cpp src/TrigBufCache.cpp src/AllocTrac.cpp src/gpuid.cpp src/Args.cpp src/PrpUseTune.cpp src/RuntimeAutotune.cpp src/FFTConfig.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp
.PHONY: test-gf61-middle-base-table
test-host: test-gf61-middle-base-table
test-gf61-middle-base-table: $(GF61_MIDDLE_BASE_TABLE_TEST)
	$(GF61_MIDDLE_BASE_TABLE_TEST)

$(GF61_MIDDLE_BASE_TABLE_TEST): tests/gf61_middle_base_table_kernel_test.cpp $(GF61_MIDDLE_BASE_TABLE_SRCS) src/TrigBufCache.h $(wildcard src/cl/*.cl)
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc tests/gf61_middle_base_table_kernel_test.cpp $(GF61_MIDDLE_BASE_TABLE_SRCS) $(TEST_GC_LINK) $(OPENCL_LIBS) -o $@

# CUDA clEnqueueFillBuffer planning for all OpenCL pattern sizes.
CUDA_FILL_TEST := build-tests/aevum-cuda-fill-test
.PHONY: test-cuda-fill
test-host: test-cuda-fill
test-cuda-fill: $(CUDA_FILL_TEST)
	$(CUDA_FILL_TEST)

$(CUDA_FILL_TEST): tests/cuda_fill_test.cpp src/cuda/CudaFill.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/cuda_fill_test.cpp -o $@
# The CUDA backend's NVRTC --gpu-architecture choice: own sm_XY, else the newest compute_XY below it (plain ints, no CUDA needed).
CUDA_NVRTC_ARCH_TEST := build-tests/aevum-cuda-nvrtc-arch-test
.PHONY: test-cuda-nvrtc-arch
test-host: test-cuda-nvrtc-arch
test-cuda-nvrtc-arch: $(CUDA_NVRTC_ARCH_TEST)
	$(CUDA_NVRTC_ARCH_TEST)

$(CUDA_NVRTC_ARCH_TEST): tests/cuda_nvrtc_arch_test.cpp src/cuda/NvrtcArch.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/cuda_nvrtc_arch_test.cpp -o $@
# carryFused low-bpw cutoff (long carry below max(10, log2(N)/2 + 0.5) bpw)
FFT_LONG_CARRY_CUTOFF_TEST := build-tests/aevum-fft-long-carry-cutoff-test
.PHONY: test-fft-long-carry-cutoff
test-host: test-fft-long-carry-cutoff
test-fft-long-carry-cutoff: $(FFT_LONG_CARRY_CUTOFF_TEST)
	$(FFT_LONG_CARRY_CUTOFF_TEST)

$(FFT_LONG_CARRY_CUTOFF_TEST): tests/fft_long_carry_cutoff_test.cpp src/Gpu.cpp src/FFTConfig.cpp src/FFTConfig.h src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc \
		tests/fft_long_carry_cutoff_test.cpp src/PrpUseTune.cpp src/RuntimeAutotune.cpp src/FFTConfig.cpp src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp \
		$(TEST_GC_LINK) -o $@
ENGINE_EQUAL_GPU_TEST := build-tests/aevum-engine-equal-canonical-gpu-test

.PHONY: test-engine-equal-gpu
test-engine-equal-gpu: engine-lib $(ENGINE_EQUAL_GPU_TEST)
	$(ENGINE_EQUAL_GPU_TEST) $(ENGINE_LIB) $(if $(AEVUM_TEST_DEVICE),$(AEVUM_TEST_DEVICE),0) . "$(AEVUM_TEST_FFT)"

$(ENGINE_EQUAL_GPU_TEST): tests/engine_equal_canonical_gpu_test.cpp src/EngineApi.h
	@mkdir -p $(dir $@)
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) $< $(DL_LIBS) -o $@
# A reset or lost GPU is reported as such, an allocation failure as out of memory (host only, no device).
GPU_LOST_ERROR_TEST := build-tests/aevum-gpu-lost-error-test
.PHONY: test-gpu-lost-error
test-host: test-gpu-lost-error
test-gpu-lost-error: $(GPU_LOST_ERROR_TEST)
	$(GPU_LOST_ERROR_TEST)
$(GPU_LOST_ERROR_TEST): tests/gpu_lost_error_test.cpp src/GpuLost.h src/clwrap.h src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/gpu_lost_error_test.cpp src/clwrap.cpp src/log.cpp src/File.cpp src/common.cpp src/fs.cpp src/timeutil.cpp $(OPENCL_LIBS) -o $@
# The CUDA backend's CUmodule lifetime: unloaded by the last program/kernel, never after its context died (no CUDA needed).
CUDA_MODULE_LIFE_TEST := build-tests/aevum-cuda-module-life-test
.PHONY: test-cuda-module-life
test-host: test-cuda-module-life
test-cuda-module-life: $(CUDA_MODULE_LIFE_TEST)
	$(CUDA_MODULE_LIFE_TEST)

$(CUDA_MODULE_LIFE_TEST): tests/cuda_module_life_test.cpp src/cuda/CudaModuleLife.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/cuda_module_life_test.cpp -pthread -o $@
# The CUDA shim's clCreateBuffer reports a failed CL_MEM_COPY_HOST_PTR upload: the real clwrap_cuda.cpp built against
# a recording stand-in for the CUDA driver (tests/cuda_stub, no CUDA needed).
CUDA_CREATE_BUFFER_COPY_TEST := build-tests/aevum-cuda-create-buffer-copy-test
.PHONY: test-cuda-create-buffer-copy
test-host: test-cuda-create-buffer-copy
test-cuda-create-buffer-copy: $(CUDA_CREATE_BUFFER_COPY_TEST)
	$(CUDA_CREATE_BUFFER_COPY_TEST)
$(CUDA_CREATE_BUFFER_COPY_TEST): tests/cuda_create_buffer_copy_test.cpp $(wildcard tests/cuda_stub/*) src/cuda/clwrap_cuda.cpp $(wildcard src/cuda/*.h) src/common.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Wno-unused-parameter -DCUDA_BACKEND -Isrc -Itests/cuda_stub tests/cuda_create_buffer_copy_test.cpp tests/cuda_stub/stub_driver.cpp src/cuda/clwrap_cuda.cpp -o $@
# The CUDA backend's mapping of a CUDA driver result to the reported status (plain ints, no CUDA needed).
CUDA_ERROR_MAP_TEST := build-tests/aevum-cuda-error-map-test
.PHONY: test-cuda-error-map
test-host: test-cuda-error-map
test-cuda-error-map: $(CUDA_ERROR_MAP_TEST)
	$(CUDA_ERROR_MAP_TEST)
$(CUDA_ERROR_MAP_TEST): tests/cuda_error_map_test.cpp src/cuda/CudaErrorMap.h src/GpuLost.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/cuda_error_map_test.cpp -o $@
# The CUDA shim frees buffers, streams and events in their own context: the real clwrap_cuda.cpp built against a
# recording stand-in for the CUDA driver (tests/cuda_stub, no CUDA needed).
CUDA_RELEASE_CONTEXT_TEST := build-tests/aevum-cuda-release-context-test
.PHONY: test-cuda-release-context
test-host: test-cuda-release-context
test-cuda-release-context: $(CUDA_RELEASE_CONTEXT_TEST)
	$(CUDA_RELEASE_CONTEXT_TEST)
$(CUDA_RELEASE_CONTEXT_TEST): tests/cuda_release_context_test.cpp $(wildcard tests/cuda_stub/*) src/cuda/clwrap_cuda.cpp $(wildcard src/cuda/*.h) src/common.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Wno-unused-parameter -DCUDA_BACKEND -Isrc -Itests/cuda_stub tests/cuda_release_context_test.cpp tests/cuda_stub/stub_driver.cpp src/cuda/clwrap_cuda.cpp -o $@
# The CUDA shim runs each program, kernel, queue and event operation in its own context (stand-in driver, no GPU).
CUDA_OBJECT_CONTEXT_TEST := build-tests/aevum-cuda-object-context-test
.PHONY: test-cuda-object-context
test-host: test-cuda-object-context
test-cuda-object-context: $(CUDA_OBJECT_CONTEXT_TEST)
	$(CUDA_OBJECT_CONTEXT_TEST)
$(CUDA_OBJECT_CONTEXT_TEST): tests/cuda_object_context_test.cpp $(wildcard tests/cuda_stub/*) src/cuda/clwrap_cuda.cpp $(wildcard src/cuda/*.h) src/common.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -pthread -Wall -Wextra -Wno-unused-parameter -DCUDA_BACKEND -Isrc -Itests/cuda_stub tests/cuda_object_context_test.cpp tests/cuda_stub/stub_driver.cpp src/cuda/clwrap_cuda.cpp -o $@
DATA_LAYOUT_TEST := build-tests/aevum-data-layout-test

.PHONY: test-data-layout
test-host: test-data-layout
test-data-layout: $(DATA_LAYOUT_TEST)
	$(DATA_LAYOUT_TEST)

$(DATA_LAYOUT_TEST): tests/data_layout_test.cpp src/DataLayout.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc $< -o $@
# Explicit 32-bit carry gate (cherubrock-seb/PrMers#104)
FFT_CARRY32_GATE_TEST := build-tests/aevum-fft-carry32-gate-test
.PHONY: test-fft-carry32-gate
test-host: test-fft-carry32-gate
test-fft-carry32-gate: $(FFT_CARRY32_GATE_TEST)
	$(FFT_CARRY32_GATE_TEST)
$(FFT_CARRY32_GATE_TEST): tests/fft_carry32_gate_test.cpp src/FFTConfig.cpp src/FFTConfig.h src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc \
		tests/fft_carry32_gate_test.cpp src/PrpUseTune.cpp src/RuntimeAutotune.cpp src/FFTConfig.cpp src/Args.cpp src/TuneEntry.cpp src/common.cpp src/fs.cpp src/File.cpp src/timeutil.cpp \
		$(TEST_GC_LINK) -o $@
# Trig cache key fields (cherubrock-seb/PrMers#107)
TRIG_CACHE_KEY_TEST := build-tests/aevum-trig-cache-key-test
.PHONY: test-trig-cache-key
test-host: test-trig-cache-key
test-trig-cache-key: $(TRIG_CACHE_KEY_TEST)
	$(TRIG_CACHE_KEY_TEST)

$(TRIG_CACHE_KEY_TEST): tests/trig_cache_key_test.cpp src/TrigBufCache.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/trig_cache_key_test.cpp -o $@
# Strict integer parsing for -use values (cherubrock-seb/PrMers#130)
STRICT_INT_TEST := build-tests/aevum-strict-int-test
.PHONY: test-strict-int
test-host: test-strict-int
test-strict-int: $(STRICT_INT_TEST)
	$(STRICT_INT_TEST)

$(STRICT_INT_TEST): tests/strict_int_test.cpp src/StrictInt.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/strict_int_test.cpp -o $@

# isPrimePRP() waits for the Background thread before computing the proof (source check, no device)
.PHONY: test-proof-background-join
test-host: test-proof-background-join
test-proof-background-join:
	python3 tests/proof_background_join_source_test.py

# isPrimePRP() runs without a ProofSet when the proof power is 0 (source check, no device)
.PHONY: test-proof-power-zero
test-host: test-proof-power-zero
test-proof-power-zero:
	python3 tests/proof_power_zero_source_test.py

# ProofSet::reducePower() rebuilds the checkpoint table (host only, no device)
PROOF_REDUCE_POWER_TEST := build-tests/aevum-proof-reduce-power-test
.PHONY: test-proof-reduce-power
test-host: test-proof-reduce-power
test-proof-reduce-power: $(PROOF_REDUCE_POWER_TEST)
	$(PROOF_REDUCE_POWER_TEST)

$(PROOF_REDUCE_POWER_TEST): tests/proof_reduce_power_test.cpp src/Proof.cpp src/Proof.h src/File.cpp src/log.cpp src/common.cpp src/fs.cpp src/timeutil.cpp src/sha3.cpp src/md5.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra $(TEST_SECTION_FLAGS) -Isrc \
		tests/proof_reduce_power_test.cpp src/Proof.cpp src/File.cpp src/log.cpp src/common.cpp src/fs.cpp src/timeutil.cpp src/sha3.cpp src/md5.cpp \
		$(TEST_GC_LINK) -o $@

# Gpu::saveProof() retries every failure, then lower powers; a failed proof does not lose the result (source check, no device)
.PHONY: test-proof-crc-retry
test-host: test-proof-crc-retry
test-proof-crc-retry:
	python3 tests/proof_crc_retry_source_test.py

# carryFused: warp sync() on both sides of the per-wavefront ready flag
.PHONY: test-carryfused-ready-sync
test-host: test-carryfused-ready-sync
test-carryfused-ready-sync:
	python3 tests/carryfused_ready_sync_source_test.py
	python3 tests/carryfused_ready_per_line_test.py

# CUDA shim: the kernel-cache blob (PTX + CUBIN) and the PTX_VERSION relabeling (plain C++, no CUDA needed).
CUDA_BINARY_BLOB_TEST := build-tests/aevum-cuda-binary-blob-test
.PHONY: test-cuda-binary-blob
test-host: test-cuda-binary-blob
test-cuda-binary-blob: $(CUDA_BINARY_BLOB_TEST)
	$(CUDA_BINARY_BLOB_TEST)
$(CUDA_BINARY_BLOB_TEST): tests/cuda_binary_blob_test.cpp src/cuda/CudaBinaryBlob.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/cuda_binary_blob_test.cpp -o $@

# File errors (CRC/read/write) are std::exceptions that the top-level handlers catch (host only, no device)
FILE_ERRORS_TEST := build-tests/aevum-file-errors-test
.PHONY: test-file-errors-std-exception
test-host: test-file-errors-std-exception
test-file-errors-std-exception: $(FILE_ERRORS_TEST)
	$(FILE_ERRORS_TEST)

$(FILE_ERRORS_TEST): tests/file_errors_std_exception_test.cpp src/File.h src/Background.h src/File.cpp src/log.cpp src/common.cpp src/fs.cpp src/timeutil.cpp
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc -pthread tests/file_errors_std_exception_test.cpp src/File.cpp src/log.cpp src/common.cpp src/fs.cpp src/timeutil.cpp -o $@

# -dir naming a missing directory is reported, not an uncaught exception; needs the binary but no GPU.
.PHONY: test-main-missing-dir
test-host: test-main-missing-dir
test-main-missing-dir: aevum
	python3 tests/main_missing_dir_test.py $(BIN)/aevum

# The CUDA shim answers CL_DEVICE_TOPOLOGY_AMD from the device's PCI bus id (-pci), using the shared host CUDA stub.
CUDA_TOPOLOGY_PCI_TEST := build-tests/aevum-cuda-topology-pci-test
.PHONY: test-cuda-topology-pci
test-host: test-cuda-topology-pci
test-cuda-topology-pci: $(CUDA_TOPOLOGY_PCI_TEST)
	$(CUDA_TOPOLOGY_PCI_TEST)
$(CUDA_TOPOLOGY_PCI_TEST): tests/cuda_topology_pci_test.cpp $(wildcard tests/cuda_stub/*) src/cuda/clwrap_cuda.cpp $(wildcard src/cuda/*.h) src/common.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Wno-unused-parameter -DCUDA_BACKEND -Isrc -Itests/cuda_stub tests/cuda_topology_pci_test.cpp tests/cuda_stub/stub_driver.cpp src/cuda/clwrap_cuda.cpp -o $@

# Signal: SIGTERM and SIGINT request a graceful stop, a second Ctrl-C exits (no GPU).
# Quote-only include search prevents src/Signal.h from shadowing the C library signal.h on case-insensitive macOS filesystems.
SIGNAL_TEST := build-tests/aevum-signal-test
.PHONY: test-signal
test-host: test-signal
test-signal: $(SIGNAL_TEST)
	$(SIGNAL_TEST)
$(SIGNAL_TEST): tests/signal_test.cpp src/Signal.cpp src/Signal.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -iquote src tests/signal_test.cpp src/Signal.cpp -o $@

# The plan search's time budget counts measurement, not engine builds.
TUNE_BUDGET_TEST := build-tests/aevum-tune-budget-test
.PHONY: test-tune-budget
test-host: test-tune-budget
test-tune-budget: $(TUNE_BUDGET_TEST)
	$(TUNE_BUDGET_TEST)
	python3 tests/autotune_budget_source_test.py
$(TUNE_BUDGET_TEST): tests/tune_budget_test.cpp src/TuneBudget.h
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc tests/tune_budget_test.cpp -o $@
# selftest.cl must not pick its inline asm from a guessed wavefront size.
.PHONY: test-selftest-wavefront-guard
test-selftest-wavefront-guard:
	python3 tests/selftest_wavefront_guard_source_test.py

test-host: test-selftest-wavefront-guard

# Finished PRP results survive proof-generation failure (source check, no device)
.PHONY: test-proof-result-survival
test-host: test-proof-result-survival
test-proof-result-survival:
	python3 tests/proof_result_survival_source_test.py src/Gpu.cpp

# timeStr(): reentrant time conversion, since every worker thread logs
.PHONY: test-time-reentrant
test-host: test-time-reentrant
test-time-reentrant:
	python3 tests/time_reentrant_source_test.py

# main() maps how the run ended to the process exit status; needs the binary but no GPU kernels.
.PHONY: test-main-exit-code
test-host: test-main-exit-code
test-main-exit-code: aevum
	python3 tests/main_exit_code_test.py $(BIN)/aevum

# Host regression for per-FFT config parsing. Link against the production engine
# objects so Args::parse is exercised with its real dependencies.
PER_FFT_CONFIG_PARSE_TEST := build-tests/aevum-per-fft-config-parse-test

.PHONY: test-per-fft-config-parse
test-host: test-per-fft-config-parse
test-per-fft-config-parse: $(PER_FFT_CONFIG_PARSE_TEST)
	$(PER_FFT_CONFIG_PARSE_TEST)

$(PER_FFT_CONFIG_PARSE_TEST): tests/per_fft_config_parse_test.cpp $(ENGINE_OBJS)
	@mkdir -p build-tests
	$(CXX) -O2 -std=c++20 $(DARWIN_MIN_FLAGS) -Wall -Wextra -Isrc \
		tests/per_fft_config_parse_test.cpp $(ENGINE_OBJS) \
		$(OPENCL_LIBS) $(DL_LIBS) -o $@
