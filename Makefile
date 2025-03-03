# Installation path (modifiable via "make install PREFIX=/your/path")
PREFIX ?= /usr/local

# Compiler and compilation options
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall
LDFLAGS = -lOpenCL

# Target executable and source/kernel files
TARGET = prmers
SRC = prmers.cpp
KERNEL = prmers.cl

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -DKERNEL_PATH="\"$(PREFIX)/share/$(TARGET)/\"" -o $(TARGET) $(SRC) $(LDFLAGS)

install: $(TARGET)
	@echo "Installing $(TARGET)..."
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin
	install -d $(DESTDIR)$(PREFIX)/share/$(TARGET)
	install -m 644 $(KERNEL) $(DESTDIR)$(PREFIX)/share/$(TARGET)
	@echo "$(TARGET) installed in $(PREFIX)/bin"
	@echo "Kernel installed in $(PREFIX)/share/$(TARGET)"

uninstall:
	@echo "Uninstalling $(TARGET)..."
	rm -f $(DESTDIR)$(PREFIX)/bin/$(TARGET)
	rm -rf $(DESTDIR)$(PREFIX)/share/$(TARGET)
	@echo "$(TARGET) has been uninstalled."

clean:
	rm -f $(TARGET)

.PHONY: all install uninstall clean
