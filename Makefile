PREFIX      ?= /usr/local
TARGET      := prmers

SRC_DIR     := src
INC_DIR     := include

SRCS        := $(shell find $(SRC_DIR) -type f -name '*.cpp')
OBJS        := $(patsubst $(SRC_DIR)/%.cpp,$(SRC_DIR)/%.o,$(SRCS))
DEPS        := $(OBJS:.o=.d)

UNAME_S := $(shell uname -s)
VERSION := $(shell git describe --tags --always 2>/dev/null || echo 4.20.97-alpha-v100.13-gm-vtrace-aevum-r1)
PACKAGE := prmers-$(VERSION)

WARN        := -Wall -Wextra -Wsign-conversion
CPPFLAGS    := -I$(INC_DIR) -I$(INC_DIR)/marin -DGPU -DAEVUM_ENGINE_DEFAULT_LIB=\"$(PREFIX)/lib/prmers/libaevum_engine.so\" -DAEVUM_ENGINE_DEFAULT_TUNE_DIR=\"$(PREFIX)/share/prmers/aevum\"
MARCH       := native
ifeq ($(UNAME_S),Darwin)
  OPT := -O3 -ffinite-math-only -mcpu=native
  CXX := c++
else
  OPT := -O3 -ffinite-math-only -march=$(MARCH)
  CXX ?= g++
endif
CXXFLAGS    := -std=c++20 $(WARN) $(OPT) -flto=auto
LDFLAGS     := -flto=auto
PLATFORM_CXXFLAGS :=
PLATFORM_LDFLAGS  :=

ifeq ($(UNAME_S),Darwin)
  CXXFLAGS := -std=c++20 $(WARN) $(OPT) -flto
  LDFLAGS := -flto
  MACOSX_DEPLOYMENT_TARGET ?= 12.0
  export MACOSX_DEPLOYMENT_TARGET
  CPPFLAGS += -I/System/Library/Frameworks/OpenCL.framework/Headers
  GMP_PREFIX := $(shell brew --prefix gmp 2>/dev/null)
  ifneq ($(GMP_PREFIX),)
    CPPFLAGS += -I$(GMP_PREFIX)/include
    PLATFORM_LDFLAGS += -L$(GMP_PREFIX)/lib
  endif
  PLATFORM_CXXFLAGS += -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET)
  PLATFORM_LDFLAGS  += -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET) -framework OpenCL
else
  LDFLAGS  += -lOpenCL
  # dlopen on Unix; Windows EngineAevum uses LoadLibrary.
  ifneq ($(shell case $(UNAME_S) in (*_NT*) echo 1;; esac),1)
    LDFLAGS += -ldl
  endif
endif

ifeq ($(shell case $(UNAME_S) in (*_NT*) echo 1;; esac),1)
  LDFLAGS  += -lWs2_32
  KERNEL_PATH ?= ./kernels/
else
  KERNEL_PATH ?= $(PREFIX)/share/$(TARGET)/
endif

LDFLAGS += -lgmpxx -lgmp
CPPFLAGS += -DKERNEL_PATH=\"$(KERNEL_PATH)\"

# v99.97: keep the v99.96 Gaussian factoring source untouched and compile its
# runGaussianMersenneECM method under the legacy symbol.  The new source file
# owns the public method and can fall back to this exact implementation.
$(SRC_DIR)/modes/RunGaussianMersenneFactor.o: CPPFLAGS += -include $(INC_DIR)/core/GmEcmLegacyRename.hpp -include $(INC_DIR)/core/GmPm1LegacyRename.hpp

MARIN_TEST_DEVICE ?= 0

.PHONY: all clean install uninstall package aevum aevum-cuda aevum-engine \
        install-aevum-engine test-aevum-host test-aevum-reg test-aevum-auto test-aevum-default test-aevum-pfa9-bridge test-gui-state test-gui-http test-pm1-bounds test-proof-marin test-ecm-torsion test-marin-ibdwt-bound test-marin-split-aux test-worktodo-manager test-marin-ll-radix5 test-tiny-exponent test-proof-power test-proof-verify test-compact-bits-wrap test-mersenne-reduce test-proof-cpu-fallback test-proof-checkpoint-readback test-proof-fallback-power test-final-carry-digit0 test-marin-invalid-device test-aevum-source test-aevum-auto-gpu test-backend-matrix test-aevum-apple-port-source test-gm clean-all

all: aevum-engine $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(PLATFORM_CXXFLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(PLATFORM_LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(PLATFORM_CXXFLAGS) $(CPPFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

-include $(DEPS)

install: all
	@if [ "$(KERNEL_PATH)" = "./kernels/" ]; then \
		echo "Installation not supported with portable kernel path."; \
		exit 1; \
	fi
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/
	install -d $(DESTDIR)$(KERNEL_PATH)
	install -m 644 kernels/*.cl $(DESTDIR)$(KERNEL_PATH)
	install -d $(DESTDIR)$(PREFIX)/lib/prmers
	install -m 755 third_party/aevum/build-engine/libaevum_engine.so $(DESTDIR)$(PREFIX)/lib/prmers/
	install -d $(DESTDIR)$(PREFIX)/share/prmers/aevum
	install -m 644 third_party/aevum/tune.txt $(DESTDIR)$(PREFIX)/share/prmers/aevum/

package: all
	@if [ "$(KERNEL_PATH)" != "./kernels/" ]; then \
		echo "Packaging only supported with portable kernel path."; \
		exit 1; \
	fi
	mkdir -p package/$(PACKAGE)/third_party/aevum/build-engine
	cp $(TARGET) package/$(PACKAGE)
	cp third_party/aevum/build-engine/libaevum_engine.so package/$(PACKAGE)/third_party/aevum/build-engine/
	cp -r kernels package/$(PACKAGE)
	bsdtar -czvf $(PACKAGE).zip -C package $(PACKAGE)

aevum:
	$(MAKE) -C third_party/aevum

aevum-cuda:
	$(MAKE) -C third_party/aevum CUDA=1

aevum-engine:
	$(MAKE) -C third_party/aevum engine-lib

test-aevum-host:
	$(MAKE) -C third_party/aevum test-host

test-aevum-reg:
	bash tests/test_aevum_reg_adapter.sh

test-aevum-reported-plan:
	bash tests/test_aevum_reported_plan.sh

test-aevum-pfa9-bridge: aevum-engine
	bash third_party/aevum/scripts/test_pfa9_lead_bridge_ubuntu.sh $${AEVUM_TEST_DEVICE:-1} $${AEVUM_TEST_EXPONENT:-175000039}

test-aevum-auto:
	bash tests/test_aevum_auto_policy.sh

test-aevum-default:
	bash tests/test_aevum_default_backend.sh

test-gui-state:
	bash tests/test_web_gui_backend_state.sh

test-gui-http:
	bash tests/test_web_gui_http.sh

test-win-cmdline:
	bash tests/test_win_cmdline_quote.sh

test-pm1-bounds:
	bash tests/test_pm1_bounds.sh

test-pm1-vtrace-small-b1:
	python3 tests/pm1_vtrace_small_b1_test.py

test-proof-marin:
	bash tests/test_proof_marin_padding.sh
	python3 tests/proof_marin_source_regression_test.py

test-ecm-torsion:
	bash tests/test_ecm_torsion_curves.sh

# Marin transform-size bound: exact 128-bit check and OpenCL/GMP device check.
test-marin-ibdwt-bound:
	bash tests/test_marin_ibdwt_size_bound.sh
	bash tests/test_marin_ibdwt_wrap_device.sh $(MARIN_TEST_DEVICE)

# Marin with the split root/weight kernel ABI forced: PRP and GMP prefix checks (OpenCL device, libgmp).
test-marin-split-aux:
	bash tests/test_marin_split_aux_prp_device.sh $(MARIN_TEST_DEVICE)

test-worktodo-manager:
	bash tests/test_worktodo_manager.sh

.PHONY: test-worktodo-doublecheck
test-worktodo-doublecheck:
	bash tests/test_worktodo_doublecheck.sh

.PHONY: test-worktodo-pminus1-factors
test-worktodo-pminus1-factors:
	bash tests/test_worktodo_pminus1_factors.sh

.PHONY: test-worktodo-quoted-factors
test-worktodo-quoted-factors:
	bash tests/test_worktodo_quoted_factors.sh

test-marin-ll-radix5: all
	bash tests/run_marin_ll_radix5_regression.sh $${AEVUM_TEST_DEVICE:-0}

test-tiny-exponent: all
	bash tests/run_tiny_exponent_regression.sh $${AEVUM_TEST_DEVICE:-0}

test-proof-power:
	python3 tests/legacy_proof_power_source_test.py

.PHONY: test-cli-exponent-range
test-cli-exponent-range:
	bash tests/test_cli_exponent_range.sh

.PHONY: test-legacy-enqueue-errors

test-legacy-enqueue-errors:
	python3 tests/legacy_enqueue_errors_source_test.py

.PHONY: test-legacy-pm1-periodic-save

test-legacy-pm1-periodic-save:
	python3 tests/legacy_pm1_periodic_save_source_test.py

test-proof-verify:
	python3 tests/proof_verify_result_source_test.py

.PHONY: test-quick-checker
test-quick-checker:
	bash tests/test_quick_checker.sh

.PHONY: test-json-res64-small-exponent
test-json-res64-small-exponent:
	bash tests/test_json_res64_small_exponent.sh

.PHONY: test-self-exe-restart
test-self-exe-restart:
	bash tests/test_self_exe_restart.sh

.PHONY: test-legacy-small-items

test-legacy-small-items:
	bash tests/test_legacy_small_items.sh

.PHONY: test-legacy-check-equal

test-legacy-check-equal:
	bash tests/test_legacy_check_equal_device.sh $(MARIN_TEST_DEVICE)

test-compact-bits-wrap:
	bash tests/test_compact_bits_wrap.sh

test-mersenne-reduce:
	bash tests/test_mersenne_reduce.sh

test-proof-cpu-fallback:
	python3 tests/proof_cpu_fallback_source_test.py

test-proof-checkpoint-readback:
	bash tests/test_proof_checkpoint_readback.sh

test-proof-fallback-power:
	python3 tests/proof_fallback_power_source_test.py

test-final-carry-digit0:
	bash tests/test_final_carry_digit0.sh

.PHONY: test-worktodo-small-items
test-worktodo-small-items:
	bash tests/test_worktodo_small_items.sh

.PHONY: test-ll-unsafe-zero-residue
test-ll-unsafe-zero-residue:
	python3 tests/ll_unsafe_zero_residue_source_test.py

test-marin-invalid-device:
	mkdir -p tests/build-marin-invalid-device
	$(CXX) -std=c++20 -O2 -Wall -Wextra -Iinclude -Iinclude/marin -DGPU tests/marin_invalid_device_test.cpp -o tests/build-marin-invalid-device/marin-invalid-device-test -lOpenCL
	tests/build-marin-invalid-device/marin-invalid-device-test
	rm -rf tests/build-marin-invalid-device

test-aevum-source:
	python3 tests/aevum_lowrange_prp_safety_source_test.py
	python3 tests/aevum_pow2_type4_source_test.py
	python3 tests/aevum_pass4_source_test.py
	python3 tests/stable_backend_stop_bsgs_apple_source_test.py
	python3 tests/workload_plan_audit_parser_test.py
	bash tests/source_aevum_engine_audit.sh

test-ecm-interrupt-no-result:
	python3 tests/ecm_interrupt_no_result_test.py

test-gm:
	python3 tests/gaussian_mersenne_math_test.py
	python3 tests/gaussian_mersenne_isolation_test.py
	python3 tests/gaussian_mersenne_factor_math_test.py
	python3 tests/gaussian_pm1_vtrace_math_test.py
	python3 tests/gaussian_pm1_vtrace_source_test.py
	python3 tests/gaussian_mersenne_factor_isolation_test.py
	python3 tests/gaussian_mersenne_ecm_seed_regression_test.py
	python3 tests/gaussian_mersenne_ecm_naf_regression_test.py
	python3 tests/gaussian_mersenne_ecm_optimized_regression_test.py
	python3 tests/gaussian_mersenne_ecm_bsgs_small_primes_test.py
	python3 tests/gaussian_mersenne_ecm_special32_regression_test.py
	python3 tests/gaussian_mersenne_ecm_special4096_regression_test.py
	python3 tests/gaussian_mersenne_windows_portability_test.py
	python3 tests/gaussian_mersenne_windows_linkage_test.py
	python3 tests/gaussian_pair_full_pipeline_test.py
	python3 tests/gaussian_pair_backend_policy_test.py
	python3 tests/gaussian_pair_tf_math_test.py
	python3 tests/gaussian_tf_checkpoint_factors_test.py
	python3 tests/test_gaussian_worktodo_generator.py
	bash tests/test_gaussian_worktodo_parser.sh

test-gm-ecm-bsgs-small-b1: all
	bash tests/gm_ecm_bsgs_small_b1_test.sh $${PRMERS_TEST_DEVICE:-0}

test-pm1-stage1-ckpt: all
	bash tests/pm1_stage1_ckpt_b1_test.sh $${PRMERS_TEST_DEVICE:-0}

test-pm1-stage1-checklevel: all
	bash tests/pm1_stage1_checklevel_test.sh $${PRMERS_TEST_DEVICE:-0}

test-pm1-bsgs-resume: all
	bash tests/pm1_bsgs_resume_boundary_test.sh $${PRMERS_TEST_DEVICE:-0}

test-pm1-interrupt-building-e: all
	bash tests/pm1_interrupt_building_e_test.sh $${PRMERS_TEST_DEVICE:-0}

test-pm1-stage2-record-factor: all
	bash tests/pm1_stage2_record_factor_test.sh $${PRMERS_TEST_DEVICE:-0}

test-pm1-bsgs-small-b1: all
	bash tests/pm1_bsgs_small_b1_test.sh $${PRMERS_TEST_DEVICE:-0}

test-pm1-vtrace-low-prime: all
	bash tests/pm1_vtrace_low_prime_test.sh $${PRMERS_TEST_DEVICE:-0}

test-backend-compat: all
	bash tests/test_backend_compatibility_cli.sh

test-aevum-auto-gpu: all
	bash tests/run_aevum_auto_gpu_matrix.sh $${AEVUM_TEST_DEVICE:-0}

test-backend-matrix: all
	bash tests/run_backend_validation_matrix.sh $${PRMERS_TEST_DEVICE:-0} $${PRMERS_MATRIX_PROFILE:-standard}

test-aevum-apple-port-source:
	bash tests/source_v9942_apple_port_audit.sh

install-aevum-engine: aevum-engine
	install -d $(DESTDIR)$(PREFIX)/lib/prmers
	install -m 755 third_party/aevum/build-engine/libaevum_engine.so $(DESTDIR)$(PREFIX)/lib/prmers/
	install -d $(DESTDIR)$(PREFIX)/share/prmers/aevum
	install -m 644 third_party/aevum/tune.txt $(DESTDIR)$(PREFIX)/share/prmers/aevum/

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/$(TARGET)
	rm -f $(DESTDIR)$(PREFIX)/lib/prmers/libaevum_engine.so
	rm -rf $(DESTDIR)$(PREFIX)/share/prmers/aevum
	rm -rf $(DESTDIR)$(KERNEL_PATH)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

clean-all: clean
	$(MAKE) -C third_party/aevum clean
	rm -rf third_party/aevum/build-tests tests/build-aevum-reg package

.PHONY: native-pfa-build native-pfa-host-test native-pfa-gpu-test
native-pfa-build: all
	python3 tests/native_pfa_cli_source_test.py

native-pfa-host-test:
	python3 tests/native_pfa_cli_source_test.py
	$(MAKE) -C third_party/aevum native-pfa-host-test

native-pfa-gpu-test: native-pfa-build
	bash scripts/test_native_pfa_gpu.sh $${PRMERS_TEST_DEVICE:-0} $${AEVUM_PFA_TEST_ITERS:-1}

.PHONY: test-llsafe2-resume
test-llsafe2-resume: all
	bash tests/run_llsafe2_resume_regression.sh $${AEVUM_TEST_DEVICE:-0}

.PHONY: test-llsafe2-result
test-llsafe2-result: all
	bash tests/run_llsafe2_result_regression.sh $${AEVUM_TEST_DEVICE:-0}

.PHONY: test-error-check-retry
test-error-check-retry:
	bash tests/test_error_check_retry.sh

.PHONY: test-llsafe-error-recovery
test-llsafe-error-recovery: all
	bash tests/run_llsafe_error_recovery_regression.sh $${AEVUM_TEST_DEVICE:-0}


.PHONY: test-legacy-prp-resume
test-legacy-prp-resume: all
	bash tests/run_legacy_prp_resume_regression.sh $${AEVUM_TEST_DEVICE:-0}

.PHONY: test-marin-exact-sub

test-marin-exact-sub:
	bash tests/test_marin_exact_subtraction_device.sh
	PRMERS_MARIN_COMPACT_WEIGHT_FORCE=1 bash tests/test_marin_exact_subtraction_device.sh $(MARIN_TEST_DEVICE) 13 1159 4423

# Marin multiply-by-a carry bound: needs an OpenCL device (PoCL works).
test-marin-adc-mul-base:
	bash tests/test_marin_adc_mul_large_base_device.sh $(MARIN_TEST_DEVICE)

.PHONY: test-wagstaff-decode

test-wagstaff-decode:
	python3 tests/wagstaff_decode_source_test.py

.PHONY: test-pm1-ultralowmem-resume

test-pm1-ultralowmem-resume: all
	bash tests/pm1_ultralowmem_resume_test.sh $${PRMERS_TEST_DEVICE:-0}

.PHONY: test-pm1-extend-ckpt

test-pm1-extend-ckpt: all
	bash tests/test_pm1_extend_stale_ckpt.sh

test-pm1-b2start-json:
	bash tests/test_pm1_b2start_json.sh

.PHONY: test-marin-reg-offset-wrap

test-marin-reg-offset-wrap:
	bash tests/test_marin_reg_offset_wrap.sh

.PHONY: test-gm-small-items
test-gm-small-items: all
	bash tests/gm_small_items_test.sh

.PHONY: test-gm-u64-divisor
test-gm-u64-divisor:
	mkdir -p /tmp/prmers-gm-u64-divisor-test
	g++ -std=c++20 -Wall -Wextra -Iinclude tests/gm_u64_divisor_test.cpp -o /tmp/prmers-gm-u64-divisor-test/gm_u64_divisor_test -lgmpxx -lgmp
	/tmp/prmers-gm-u64-divisor-test/gm_u64_divisor_test

.PHONY: test-gm-ecm-resume
test-gm-ecm-resume:
	python3 tests/gm_ecm_resume_source_test.py
	mkdir -p /tmp/prmers-gm-ecm-resume-test
	g++ -std=c++20 -Wall -Wextra -Iinclude tests/gm_ecm_progress_test.cpp -o /tmp/prmers-gm-ecm-resume-test/gm_ecm_progress_test
	/tmp/prmers-gm-ecm-resume-test/gm_ecm_progress_test

test-pm1-prime95-relative-path: all
	bash tests/pm1_prime95_relative_path_test.sh $${PRMERS_TEST_DEVICE:-0}

.PHONY: test-ecm-resume-line-checksum
test-ecm-resume-line-checksum: ; python3 tests/ecm_resume_line_checksum_test.py

.PHONY: test-ecm-resume-curve-index
test-ecm-resume-curve-index: ; python3 tests/ecm_resume_curve_index_source_test.py

.PHONY: test-ecm-stage2-u64-primes
test-ecm-stage2-u64-primes: ; python3 tests/ecm_stage2_u64_primes_source_test.py

.PHONY: test-ecm-te16-construction-factor
test-ecm-te16-construction-factor: ; bash tests/test_ecm_te16_construction_factor.sh

.PHONY: test-ecm-small-items
test-ecm-small-items: ; python3 tests/ecm_small_items_source_test.py

.PHONY: test-ecm-prime95-relative-path
test-ecm-prime95-relative-path: ; bash tests/ecm_prime95_relative_path_test.sh $${PRMERS_TEST_DEVICE:-0}

# PR138 cumulative semantic integration
test-gm-tf-worktodo-queue: all
	bash tests/gmtf_worktodo_queue_test.sh $${PRMERS_TEST_DEVICE:-0}

# PR140 cumulative semantic integration
.PHONY: test-ecm-te-sigma-checkpoint
test-ecm-te-sigma-checkpoint:
	python3 tests/ecm_te_sigma_checkpoint_source_test.py
