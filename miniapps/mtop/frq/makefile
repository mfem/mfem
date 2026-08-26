# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.

# Use the MFEM source/build tree containing miniapps/mtop/frq.
MFEM_DIR ?= ../../..
MFEM_BUILD_DIR ?= ../../..
MFEM_INSTALL_DIR ?= ../../../mfem

CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

TARGET = frequency_domain_preconditioners_regression
OBJECTS = frequency_domain_preconditioners_regression.o \
	frequency_domain_preconditioners.o
PAR_TARGETS = frequency_domain_elasticity_test \
	frequency_domain_elasticity_solver_regression \
	frequency_domain_cantilever \
	frequency_domain_cantilever_mms_regression
ELASTICITY_TEST_OBJECTS = frequency_domain_elasticity_test.o \
	frequency_domain_elasticity.o
SOLVER_REGRESSION_OBJECTS = frequency_domain_elasticity_solver_regression.o \
	frequency_domain_elasticity_solver.o \
	frequency_domain_elasticity.o \
	frequency_domain_preconditioners.o
CANTILEVER_OBJECTS = frequency_domain_cantilever.o \
	frequency_domain_elasticity_solver.o \
	frequency_domain_elasticity.o \
	frequency_domain_preconditioners.o
CANTILEVER_MMS_OBJECTS = frequency_domain_cantilever_mms_regression.o \
	frequency_domain_elasticity_solver.o \
	frequency_domain_elasticity.o \
	frequency_domain_preconditioners.o

.DEFAULT_GOAL := all

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all parallel clean

# Remove built-in rules.
%: %.cpp
%.o: %.cpp

$(TARGET): $(OBJECTS)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

frequency_domain_elasticity_test: $(ELASTICITY_TEST_OBJECTS)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

frequency_domain_elasticity_solver_regression: $(SOLVER_REGRESSION_OBJECTS)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

frequency_domain_cantilever: $(CANTILEVER_OBJECTS)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

frequency_domain_cantilever_mms_regression: $(CANTILEVER_MMS_OBJECTS)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

%.o: %.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(TARGET)

parallel: $(PAR_TARGETS)

MFEM_TESTS = MINIAPPS
MINIAPPS = $(TARGET)
include $(MFEM_TEST_MK)

$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f *.o *~ $(TARGET) $(PAR_TARGETS)
