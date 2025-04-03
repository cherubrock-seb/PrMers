# Installation path
PREFIX ?= /usr/local
KERNEL_PATH ?= $(PREFIX)/share/$(TARGET)/

# Compiler and compilation options
CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall
LDFLAGS = -lOpenCL

# Target executable and source/kernel files
TARGET = prmers
SRC = prmers.cpp proof/common.cpp proof/proof.cpp proof/md5.cpp proof/sha3.cpp
KERNELS = prmers.cl prmers_vload2.cl

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -DKERNEL_PATH="\"$(KERNEL_PATH)\"" -o $(TARGET) $(SRC) $(LDFLAGS)

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