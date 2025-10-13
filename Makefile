PREFIX      ?= /usr/local
TARGET      := prmers
KERNEL_PATH ?= $(PREFIX)/share/$(TARGET)/

SRC_DIR     := src
INC_DIR     := include

SRCS        := $(shell find $(SRC_DIR) -type f -name '*.cpp')
OBJS        := $(patsubst $(SRC_DIR)/%.cpp,$(SRC_DIR)/%.o,$(SRCS))

CFLAGS      := -Wall -Wextra -Wsign-conversion -ffinite-math-only
FLAGS_CPU   := -O3
FLAGS_GPU   := -O2 -DGPU

CXX         := g++
CXXFLAGS    := -std=c++20 -O3 -Wall -I$(INC_DIR) -march=native -flto -I$(INC_DIR)/marin $(CFLAGS) $(FLAGS_GPU) $(FLAGS_CPU)
LDFLAGS     := -flto=8

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXXFLAGS += -I/System/Library/Frameworks/OpenCL.framework/Headers
  LDFLAGS  += -framework OpenCL
else
  LDFLAGS  += -lOpenCL
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

.PHONY: all clean install uninstall

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

install: $(TARGET)
	@echo "Installation de $(TARGET) dans $(PREFIX)/bin"
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/
	@echo "Installation des kernels dans $(KERNEL_PATH)"
	install -d $(DESTDIR)$(KERNEL_PATH)
	install -m 644 kernels/*.cl $(DESTDIR)$(KERNEL_PATH)

uninstall:
	@echo "Désinstallation de $(TARGET)"
	rm -f $(DESTDIR)$(PREFIX)/bin/$(TARGET)
	rm -rf $(DESTDIR)$(KERNEL_PATH)

clean:
	@echo "Nettoyage des objets et de l'exécutable"
	rm -f $(OBJS) $(TARGET)
