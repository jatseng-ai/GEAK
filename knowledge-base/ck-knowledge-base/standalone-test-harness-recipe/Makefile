# GEAK Test Harness - Kernel .so build rules
# Standalone build (independent of CK's CMake).
#
# Preferred: use compile.py which auto-detects GPU arch from PyTorch.
# Direct usage: make ARCH=gfx950 [target]

HIPCC      ?= /opt/rocm/bin/hipcc
CK_INCLUDE  = ../../../include

ifndef ARCH
  $(error ARCH not set. Use compile.py for auto-detection, or: make ARCH=gfx950)
endif

HIPCC_FLAGS = -shared -fPIC -O3 \
              --offload-arch=$(ARCH) \
              -I$(CK_INCLUDE) \
              -std=c++17

.PHONY: all baseline optimized clean

all: baseline optimized

baseline: libbaseline.so
optimized: liboptimized.so

libbaseline.so: baseline.cpp
	@echo "Building baseline for $(ARCH)..."
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $<

liboptimized.so: optimized.cpp
	@echo "Building optimized for $(ARCH)..."
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $<

clean:
	rm -f libbaseline.so liboptimized.so