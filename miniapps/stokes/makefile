.DEFAULT_GOAL := all

MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/stokes/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

RANLIB ?= ranlib
CORE_LIB = libstokes_core.a
SUPPORT_LIB = libstokes_example_support.a
SUPPORT_OBJECTS = examples/support/manufactured.o examples/support/mesh_manufactured.o
SUPPORT_MINIAPPS = stokes_pspg stokes_mms stokes_adjoint_example
BASE_MINIAPPS = stokes_pspg flow_example channel_flow refine_channel_serial \
   stokes_mms stokes_adjoint_example steady_navier_stokes_setup steady_navier_stokes_workflow
MUMPS_MINIAPPS = steady_navier_stokes_gmg_benchmark steady_navier_stokes \
   stokes_brinkman stokes_brinkman_linear_example
ALL_MINIAPPS = $(BASE_MINIAPPS) $(MUMPS_MINIAPPS)
ALIASED_MINIAPPS = flow_example stokes_mms steady_navier_stokes_setup steady_navier_stokes
APPLICATION_OBJECTS = $(patsubst $(SRC)%.cpp,%.o,$(wildcard $(SRC)examples/*.cpp))
OBJECTS = stokes.o $(SUPPORT_OBJECTS) $(APPLICATION_OBJECTS)
HEADERS = $(SRC)stokes.hpp $(wildcard $(SRC)examples/support/*.hpp)
PAR_STOKES =
MINIAPPS =

.SUFFIXES:
.PHONY: all lib check clean clean-build clean-exec

ifeq ($(strip $(CONFIG_MK)),)
.PHONY: $(ALL_MINIAPPS) $(CORE_LIB) $(SUPPORT_LIB)
all lib $(ALL_MINIAPPS) $(CORE_LIB) $(SUPPORT_LIB) test test-par test-ser test-print:
	$(error MFEM is not configured; set MFEM_BUILD_DIR or MFEM_INSTALL_DIR to a configured MFEM build or installation)
else ifeq ($(MFEM_USE_MPI)$(MFEM_USE_DOUBLE),YESYES)
MINIAPPS = $(BASE_MINIAPPS)
PAR_STOKES = flow_example stokes_pspg
ifeq ($(MFEM_USE_MUMPS),YES)
MINIAPPS += $(MUMPS_MINIAPPS)
steady_navier_stokes: examples/steady_navier_stokes_example.o
else
.PHONY: $(MUMPS_MINIAPPS)
$(MUMPS_MINIAPPS):
	$(error $@ requires MFEM_USE_MUMPS=YES)
endif

all: $(MINIAPPS)
lib: $(CORE_LIB)

flow_example: examples/flow.o
stokes_mms: examples/stokes_mms_main.o
steady_navier_stokes_setup: examples/steady_navier_stokes.o
$(filter-out $(ALIASED_MINIAPPS),$(MINIAPPS)): %: examples/%.o
$(SUPPORT_MINIAPPS): $(SUPPORT_LIB)
$(MINIAPPS): $(CORE_LIB) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $(filter %.o,$^) \
	   $(if $(filter $@,$(SUPPORT_MINIAPPS)),$(SUPPORT_LIB)) $(CORE_LIB) -o $@ $(MFEM_LIBS)

$(CORE_LIB): stokes.o
	$(AR) $(ARFLAGS) $@ $^
	$(RANLIB) $@

$(SUPPORT_LIB): $(SUPPORT_OBJECTS)
	$(AR) $(ARFLAGS) $@ $^
	$(RANLIB) $@

$(OBJECTS): %.o: $(SRC)%.cpp $(HEADERS) $(MFEM_LIB_FILE) $(CONFIG_MK)
	@mkdir -p $(@D)
	$(MFEM_CXX) $(MFEM_PICFLAG) $(MFEM_FLAGS) -I$(or $(SRC),.) -c $< -o $@

$(MFEM_LIB_FILE):
	$(error The MFEM library is not built; build MFEM before building the flow package)
else
.PHONY: $(ALL_MINIAPPS) $(CORE_LIB) $(SUPPORT_LIB)
all:
	@printf '%s\n' 'Skipping stokes: MFEM_USE_MPI=YES and MFEM_USE_DOUBLE=YES are required.'
lib $(ALL_MINIAPPS) $(CORE_LIB) $(SUPPORT_LIB):
	$(error The flow package requires MFEM_USE_MPI=YES and MFEM_USE_DOUBLE=YES)
endif

ifneq ($(strip $(CONFIG_MK)),)
MFEM_TESTS = STOKES
include $(MFEM_TEST_MK)
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)

.PHONY: flow_example-test-par stokes_pspg-test-par
flow_example-test-par: flow_example
	@$(call mfem-test,$<,$(RUN_MPI),Linear flow smoke,,SKIP-NO-VIS)
	@$(call mfem-test,$<,$(RUN_MPI),Nonlinear Brinkman flow smoke,--convection --drag 2,SKIP-NO-VIS)

stokes_pspg-test-par: stokes_pspg
	@$(call mfem-test,$<,$(RUN_MPI),Manufactured Stokes smoke,--element quad --order 2 --cells 4 --solver fgmres --no-paraview,SKIP-NO-VIS)
endif

check: test
clean: clean-build clean-exec
clean-build:
	$(RM) $(OBJECTS) $(CORE_LIB) $(SUPPORT_LIB) $(ALL_MINIAPPS)
clean-exec:
	$(RM) flow_example.stderr stokes_pspg.stderr
