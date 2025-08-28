# Makefile pour prmers

PREFIX      ?= /usr/local
TARGET      := prmers
KERNEL_PATH ?= $(PREFIX)/share/$(TARGET)/

SRC_DIR     := src
INC_DIR     := include

# On collecte tous les .cpp sous src/
SRCS        := $(shell find $(SRC_DIR) -type f -name '*.cpp')
# On déduit les .o correspondants
OBJS        := $(patsubst $(SRC_DIR)/%.cpp,$(SRC_DIR)/%.o,$(SRCS))

CXX         := g++
CXXFLAGS    := -std=c++20 -O3 -Wall -I$(INC_DIR) -march=native -flto -I$(INC_DIR)/marin
LDFLAGS     := -flto



# Plateforme OpenCL
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXXFLAGS += -I/System/Library/Frameworks/OpenCL.framework/Headers
  LDFLAGS  += -framework OpenCL
else
  LDFLAGS  += -lOpenCL
endif

# GMP library
LDFLAGS += -lgmpxx -lgmp

# Curl (optionnel)
USE_CURL ?= 1
ifeq ($(USE_CURL),1)
  CXXFLAGS += -DHAS_CURL=1
  LDFLAGS  += -lcurl
else
  CXXFLAGS += -DNO_CURL=1
endif

# Macro pour le chemin des kernels
CPPFLAGS   := -DKERNEL_PATH=\"$(KERNEL_PATH)\"

.PHONY: all clean install uninstall

all: $(TARGET)

# Linker
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS)

# Compilation d'un .cpp en .o
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
