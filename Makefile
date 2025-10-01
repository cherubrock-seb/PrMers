PREFIX      ?= /usr/local
TARGET      := prmers

SRC_DIR     := src
INC_DIR     := include

SRCS        := $(shell find $(SRC_DIR) -type f -name '*.cpp')
OBJS        := $(patsubst $(SRC_DIR)/%.cpp,$(SRC_DIR)/%.o,$(SRCS))

CFLAGS      := -Wall -Wextra -Wsign-conversion -ffinite-math-only
FLAGS_CPU   := -O3
FLAGS_GPU   := -O2 -DGPU
MARCH       := native

CXX         := g++
CXXFLAGS    := -std=c++20 -O3 -Wall -I$(INC_DIR) -march=$(MARCH) -flto=auto -I$(INC_DIR)/marin $(CFLAGS) $(FLAGS_GPU) $(FLAGS_CPU)
LDFLAGS     := -flto=auto

UNAME_S := $(shell uname -s)

VERSION := $(shell git describe --tags --always)
PACKAGE := prmers-$(VERSION)

# Mac
ifeq ($(UNAME_S),Darwin)
  CXXFLAGS += -I/System/Library/Frameworks/OpenCL.framework/Headers
  LDFLAGS  += -framework OpenCL
else
  LDFLAGS  += -lOpenCL
endif

# Windows
ifeq ($(shell case $(UNAME_S) in (*_NT*) echo 1;; esac),1)
  LDFLAGS  += -lWs2_32
  KERNEL_PATH ?= ./kernels/
else
  KERNEL_PATH ?= $(PREFIX)/share/$(TARGET)/
endif

LDFLAGS += -lgmpxx -lgmp

USE_CURL ?= 1
ifeq ($(USE_CURL),1)
  CXXFLAGS += -DHAS_CURL=1
  LDFLAGS  += -lcurl
else
  CXXFLAGS += -DNO_CURL=1
endif

CPPFLAGS   := -DKERNEL_PATH=\"$(KERNEL_PATH)\"

.PHONY: all clean install uninstall package

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

install: $(TARGET)
	@if [ "$(KERNEL_PATH)" = "./kernels/" ]; then \
		echo "Installation not supported with portable kernel path."; \
		exit 1; \
	fi
	@echo "Installation de $(TARGET) dans $(PREFIX)/bin"
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/
	@echo "Installation des kernels dans $(KERNEL_PATH)"
	install -d $(DESTDIR)$(KERNEL_PATH)
	install -m 644 kernels/*.cl $(DESTDIR)$(KERNEL_PATH)

package: $(TARGET)
	@if [ "$(KERNEL_PATH)" != "./kernels/" ]; then \
		echo "Packaging only supported with portable kernel path."; \
		exit 1; \
	fi
	mkdir -p package/$(PACKAGE)
	cp $(TARGET) package/$(PACKAGE)
	cp -r kernels package/$(PACKAGE)
	ldd $(TARGET) | grep "=> /.[^/]" | awk '{print $$3}' | xargs -- cp -t package/$(PACKAGE)
	bsdtar -czvf $(PACKAGE).zip -C package $(PACKAGE)

uninstall:
	@echo "Désinstallation de $(TARGET)"
	rm -f $(DESTDIR)$(PREFIX)/bin/$(TARGET)
	rm -rf $(DESTDIR)$(KERNEL_PATH)

clean:
	@echo "Nettoyage des objets et de l'exécutable"
	rm -f $(OBJS) $(TARGET)
