# Chemin d'installation (modifiable via "make install PREFIX=/mon/chemin")
PREFIX ?= /usr/local

# Compilateur et options de compilation
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall
LDFLAGS = -lOpenCL

# Nom de l'exécutable et fichiers sources
TARGET = prmers
SRC = prmers.cpp
KERNEL = prmers.cl

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

install: $(TARGET)
	@echo "Installation de $(TARGET)..."
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin
	install -d $(DESTDIR)$(PREFIX)/share/$(TARGET)
	install -m 644 $(KERNEL) $(DESTDIR)$(PREFIX)/share/$(TARGET)
	@echo "$(TARGET) installé dans $(PREFIX)/bin"
	@echo "Kernel installé dans $(PREFIX)/share/$(TARGET)"

uninstall:
	@echo "Désinstallation de $(TARGET)..."
	rm -f $(DESTDIR)$(PREFIX)/bin/$(TARGET)
	rm -rf $(DESTDIR)$(PREFIX)/share/$(TARGET)
	@echo "$(TARGET) a été désinstallé."

clean:
	rm -f $(TARGET)

.PHONY: all install uninstall clean

