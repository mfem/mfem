

# Use the MFEM build directory
MFEM_DIR ?= ..
MFEM_BUILD_DIR ?= ..
SRC = $(if $(MFEM_DIR:..=),$(MFEM_DIR)/ZZ_test/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = zz_test
PAR_EXAMPLES = 
SEQ_DEVICE_EXAMPLES = 
PAR_DEVICE_EXAMPLES = 

ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES) $(SEQ_EXAMPLES)
endif
SUBDIRS =
ifeq ($(MFEM_USE_AMGX),YES)
   SUBDIRS += amgx
endif
ifeq ($(MFEM_USE_GINKGO),YES)
   SUBDIRS += ginkgo
endif
ifeq ($(MFEM_USE_HIOP),YES)
   SUBDIRS += hiop
endif
ifeq ($(MFEM_USE_PETSC),YES)
   SUBDIRS += petsc
endif
ifeq ($(MFEM_USE_PUMI),YES)
   SUBDIRS += pumi
endif
ifeq ($(MFEM_USE_SUNDIALS),YES)
   SUBDIRS += sundials
endif
ifeq ($(MFEM_USE_SUPERLU),YES)
   SUBDIRS += superlu
endif
ifeq ($(MFEM_USE_CALIPER),YES)
   SUBDIRS += caliper
endif

SUBDIRS_ALL = $(addsuffix /all,$(SUBDIRS))
SUBDIRS_TEST = $(addsuffix /test,$(SUBDIRS))
SUBDIRS_CLEAN = $(addsuffix /clean,$(SUBDIRS))
SUBDIRS_TPRINT = $(addsuffix /test-print,$(SUBDIRS))

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(EXAMPLES) $(SUBDIRS_ALL)

.PHONY: $(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_CLEAN) $(SUBDIRS_TPRINT)
$(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_CLEAN):
	$(MAKE) -C $(@D) $(@F)
$(SUBDIRS_TPRINT):
	@$(MAKE) -C $(@D) $(@F)

MFEM_TESTS = EXAMPLES
include $(MFEM_TEST_MK)
test: $(SUBDIRS_TEST)
test-print: $(SUBDIRS_TPRINT)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

# ZZ_test: ZZ_test.o exact.o $(MFEM_LIB_FILE)
# 	$(MFEM_CXX) $(MFEM_FLAGS) $(<) -o $(@) ZZ_test.o exact.o $(MFEM_LIB_FILE)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec $(SUBDIRS_CLEAN)

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints

# clean-exec:
# 	@rm -f refined.mesh displaced.mesh mesh.* ex5.mesh ex6p-checkpoint.*
# 	@rm -rf Example5* Example9* Example15* Example16* Example23* ParaView
# 	@rm -f sphere_refined.* sol.* sol_u.* sol_p.* sol_r.* sol_i.*
# 	@rm -f ex9.mesh ex9-mesh.* ex9-init.* ex9-final.*
# 	@rm -f deformed.* velocity.* elastic_energy.* mode_* flux.*
# 	@rm -f ex5-p-*.bp ex9-p-*.bp ex12-p-*.bp ex16-p-*.bp
# 	@rm -f ex16.mesh ex16-mesh.* ex16-init.* ex16-final.*
# 	@rm -f vortex-mesh.* vortex.mesh vortex-?-init.* vortex-?-final.*
# 	@rm -f deformation.* pressure.*
# 	@rm -f ex20.dat ex20p_?????.dat gnuplot_ex20.inp gnuplot_ex20p.inp
# 	@rm -f ex21*.mesh ex21*.sol ex21p_*.*
# 	@rm -f ex23.mesh ex23-*.gf
# 	@rm -f ex25.mesh ex25-*.gf ex25p-*.*
# 	@rm -rf ex28_* ex28p_*
