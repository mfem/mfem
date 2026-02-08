# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/examples/signorini/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
	$(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = explicit
PAR_EXAMPLES = explicitp mmsp weakexplicitp groschwheelp regularp
ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES) $(SEQ_EXAMPLES)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(EXAMPLES)

# Replace the default implicit rule for regularp.cpp
regularp: $(SRC)regularp.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< sfem/bilininteg.cpp sfem/lininteg.cpp sfem/coefficient.cpp -o $@ $(MFEM_LIBS)

# Additional dependencies
explicit: $(SRC)signorini.hpp

ifeq ($(MFEM_USE_MPI),YES)
explicitp: $(SRC)signorini.hpp
regularp: $(SRC)sfem/lininteg.hpp $(SRC)sfem/lininteg.cpp \
	$(SRC)sfem/bilininteg.hpp $(SRC)sfem/bilininteg.cpp \
	$(SRC)sfem/coefficient.hpp $(SRC)sfem/coefficient.cpp
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
