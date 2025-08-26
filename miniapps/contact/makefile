# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
# See LICENSE and NOTICE for details. LLNL-CODE-806117.

# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/contact/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Build only if TRIBOL is enabled
ifeq ($(MFEM_USE_TRIBOL)$(MFEM_USE_MPI),YESYES)
   MINIAPPS = contact
else
   MINIAPPS =
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in single-file rule (kept for completeness)
%: %.cpp

# ---- Contact miniapp multi-file build ----
# Sources
UTILS_SRC = utils
CONTACT_SRCS = contact.cpp \
               $(UTILS_SRC)/ip.cpp \
               $(UTILS_SRC)/problems.cpp
CONTACT_OBJS = $(CONTACT_SRCS:.cpp=.o)

# Link rule
contact: $(CONTACT_OBJS) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(CONTACT_OBJS) -o $@ $(MFEM_LIBS) 

# Compile objects (add local headers)
$(SRC)%.o: $(SRC)%.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -I$(UTILS_SRC) -c $< -o $@

# Default target
all: $(MINIAPPS)

# ---- MFEM test harness wiring ----
MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
TEST_NAME := Contact miniapp
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), $(TEST_NAME))
%-test-seq: %
	@$(call mfem-test,$<,, $(TEST_NAME))

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f $(CONTACT_OBJS) *~ $(MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf ParaView 
