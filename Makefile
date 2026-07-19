PREFIX      ?= /usr/local
TARGET      := prmers

SRC_DIR     := src
INC_DIR     := include

SRCS        := $(shell find $(SRC_DIR) -type f -name '*.cpp')
OBJS        := $(patsubst $(SRC_DIR)/%.cpp,$(SRC_DIR)/%.o,$(SRCS))
DEPS        := $(OBJS:.o=.d)

UNAME_S := $(shell uname -s)
VERSION := $(shell git describe --tags --always 2>/dev/null || echo v99.21-aevum-apple-opencl12-conservative-lds)
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
  LDFLAGS  += -lOpenCL -ldl
endif

ifeq ($(shell case $(UNAME_S) in (*_NT*) echo 1;; esac),1)
  LDFLAGS  += -lWs2_32
  KERNEL_PATH ?= ./kernels/
else
  KERNEL_PATH ?= $(PREFIX)/share/$(TARGET)/
endif

LDFLAGS += -lgmpxx -lgmp
CPPFLAGS += -DKERNEL_PATH=\"$(KERNEL_PATH)\"

.PHONY: all clean install uninstall package aevum aevum-cuda aevum-engine \
        install-aevum-engine test-aevum-host test-aevum-reg test-aevum-auto test-aevum-default test-gui-state test-aevum-source test-aevum-auto-gpu test-backend-matrix test-aevum-apple-port-source clean-all

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

test-aevum-auto: aevum-engine
	bash tests/test_aevum_auto_policy.sh

test-aevum-default:
	bash tests/test_aevum_default_backend.sh

test-gui-state:
	bash tests/test_web_gui_backend_state.sh

test-aevum-source:
	bash tests/source_aevum_engine_audit.sh

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
