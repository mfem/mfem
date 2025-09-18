# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
# See LICENSE and NOTICE for details. LLNL-CODE-806117.

# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/contact/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)


ifeq ($(MFEM_USE_TRIBOL)$(MFEM_USE_MPI),YESYES)
   MINIAPPS = contact
   CONTACT_SRC = utils/ip.cpp utils/problems.cpp
else
   MINIAPPS = 
   CONTACT_SRC = 
   .PHONY: contact
   contact:
	@echo "Rebuild MFEM with MFEM_USE_TRIBOL=YES and MFEM_USE_MPI=YES to enable."
endif

CONTACT_OBJ = $(CONTACT_SRC:.cpp=.o)

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp
%.o: %.cpp

%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONTACT_OBJ) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(CONTACT_OBJ) $(MFEM_LIBS) \
		-l$(patsubst lib%,%,$(basename $(notdir $(MFEM_LIB_FILE))))

%.o: $(SRC)%.cpp $(wildcard $(SRC)%.hpp) $(MFEM_LIB_FILE)\
 $(CONFIG_MK) | 
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

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
