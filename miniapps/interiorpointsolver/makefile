# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/interiorpointsolver/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Remove built-in rule
%: %.cpp
%: %.c
%: %.o

%.o: %.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

FRAMEWORK_SRC = Problem.cpp IPsolver.cpp utilities.cpp
FRAMEWORK_OBJ = $(FRAMEWORK_SRC:.cpp=.o)

TestProblem1: TestProblem1.o $(FRAMEWORK_OBJ) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TestProblem1.o $(FRAMEWORK_OBJ) -o $@ $(MFEM_LIBS)

TestProblem2: TestProblem2.o $(FRAMEWORK_OBJ) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TestProblem2.o $(FRAMEWORK_OBJ) -o $@ $(MFEM_LIBS)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)


.PHONY: clean
clean:
	rm -f *.o TestProblem1 TestProblem2
