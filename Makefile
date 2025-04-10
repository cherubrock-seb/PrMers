# Installation path
PREFIX ?= /usr/local
TARGET = prmers
KERNEL_PATH ?= $(PREFIX)/share/$(TARGET)/

# Source and kernel files
SRC = prmers.cpp proof/common.cpp proof/proof.cpp proof/md5.cpp proof/sha3.cpp
KERNELS = prmers.cl

# Compiler
CXX = g++

# Default flags
CXXFLAGS = -std=c++20 -O3 -Wall

# Platform-specific OpenCL flags
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)  # macOS
    OPENCL_CFLAGS  = -I/System/Library/Frameworks/OpenCL.framework/Headers
    OPENCL_LDFLAGS = -framework OpenCL
else
    OPENCL_CFLAGS  =
    OPENCL_LDFLAGS = -lOpenCL
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(OPENCL_CFLAGS) -DKERNEL_PATH="\"$(KERNEL_PATH)\"" -o $(TARGET) $(SRC) $(OPENCL_LDFLAGS)

install: $(TARGET)
	@echo "Installing $(TARGET)..."
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin
	install -d $(DESTDIR)$(KERNEL_PATH)
	install -m 644 $(KERNELS) $(DESTDIR)$(KERNEL_PATH)
	@echo "$(TARGET) installed in $(PREFIX)/bin"
	@echo "Kernels installed in $(KERNEL_PATH)"

uninstall:
	@echo "Uninstalling $(TARGET)..."
	rm -f $(DESTDIR)$(PREFIX)/bin/$(TARGET)
	rm -rf $(DESTDIR)$(KERNEL_PATH)
	@echo "$(TARGET) has been uninstalled."

clean:
	rm -f $(TARGET)

.PHONY: all install uninstall clean
