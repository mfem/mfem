# Copyright (c) 2010-2025, Lawrence Livermore
# See MFEM LICENSE/NOTICE for details.

# MFEM roots
MFEM_DIR        ?= ../..
MFEM_BUILD_DIR  ?= ../..
MFEM_INSTALL_DIR?= ../../mfem

# Allow out-of-source builds (mirror style from electromagnetics makefile)
PIC_SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/pic/,)

# Pull MFEM build configuration
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

# Link against miniapps/common but don't rebuild it here
COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common
# Add rpath to common when using static layout (same pattern as original)
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(MFEM_XLINKER)-rpath,$(abspath $(MFEM_BUILD_DIR)/miniapps/common))

# Ensure headers from miniapps/electromagnetics are visible
MFEM_FLAGS += -I$(MFEM_DIR)/miniapps/electromagnetics

# Executable(s)
APPS = push_particle_test

all: $(APPS)

# One-file miniapp rule (mirrors the 'lorentz' style: compile then link)
%: $(PIC_SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $@.o $(COMMON_LIB) $(MFEM_LIBS)

# Testing hook (optional; included only if present)
MFEM_TESTS = MINIAPPS
-include $(MFEM_TEST_MK)

# Hard fail if MFEM not built
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

# Cleaning
clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(APPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@true