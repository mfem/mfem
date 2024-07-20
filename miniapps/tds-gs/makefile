# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/tds-gs/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

#define source and object
GS_COMMON_SRC = amr.cpp double_integrals.cpp field.cpp  test.cpp boundary.cpp elliptic_integral.cpp gs.cpp plasma_model.cpp \
				diffusion_term.cpp exact.cpp initial_coefficient.cpp sys_operator.cpp
GS_COMMON_OBJ = $(GS_COMMON_SRC:.cpp=.o)

#define the target executable
MINIAPPS = main

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean
.PRECIOUS: %.o

# Remove built-in rules
%: %.cpp
%.o: %.cpp

%: %.o $(GS_COMMON_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

# this is the older version
#make:
#	mpicxx -O3 -std=c++11 -I$(MFEM_DIR) -I$(MFEM_DIR)/../hypre/src/hypre/include \
#	main.cpp double_integrals.cpp gs.cpp elliptic_integral.cpp test.cpp \
#	initial_coefficient.cpp plasma_model.cpp field.cpp \
#	sys_operator.cpp boundary.cpp diffusion_term.cpp \
#	exact.cpp amr.cpp \
#	-o main -L$(MFEM_DIR) -lmfem -w -L$(MFEM_DIR)/../hypre/src/hypre/lib -lHYPRE -L$(MFEM_DIR)/../metis-4.0 -lmetis

field:
	g++ -g -std=c++11 -I$(MFEM_DIR) \
	make_field.cpp \
	-o make_field.o -L$(MFEM_DIR) -lmfem -w 

test:
	./main -t 1

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f  main
	rm -rf *.o
	rm -rf out/*.gf
	rm -rf out/*.mesh
