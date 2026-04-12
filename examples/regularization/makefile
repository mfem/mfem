# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/examples/regularization/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
	$(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES =
PAR_EXAMPLES = regularp biasedp
ifeq ($(MFEM_USE_MPI),NO)
	EXAMPLES = $(SEQ_EXAMPLES)
else
	EXAMPLES = $(PAR_EXAMPLES) $(SEQ_EXAMPLES)
endif

SFEM_COMMON_SRC = sfem/bilininteg.cpp sfem/lininteg.cpp sfem/coefficient.cpp
SFEM_COMMON_HDR = $(SFEM_COMMON_SRC:.cpp=.hpp)

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(EXAMPLES)

regularp biasedp: %: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< $(SFEM_COMMON_SRC) -o $@ $(MFEM_LIBS)

ifeq ($(MFEM_USE_MPI),YES)
regularp biasedp: $(addprefix $(SRC),$(SFEM_COMMON_SRC) $(SFEM_COMMON_HDR))
endif

MFEM_TESTS = EXAMPLES
include $(MFEM_TEST_MK)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not build)

clean: clean-build

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f refined.mesh sol.gf mesh.* sol.*
