MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..

make:
	mpicxx -O3 -std=c++11 -I$(MFEM_DIR) -I$(MFEM_DIR)/../hypre/src/hypre/include \
	main.cpp double_integrals.cpp gs.cpp elliptic_integral.cpp test.cpp \
	initial_coefficient.cpp plasma_model.cpp field.cpp \
	sys_operator.cpp boundary.cpp diffusion_term.cpp \
	exact.cpp amr.cpp \
	-o main -L$(MFEM_DIR) -lmfem -w -L$(MFEM_DIR)/../hypre/src/hypre/lib -lHYPRE -L$(MFEM_DIR)/../metis-4.0 -lmetis

field:
	g++ -g -std=c++11 -I$(MFEM_DIR) \
	make_field.cpp \
	-o make_field.o -L$(MFEM_DIR) -lmfem -w 

test:
	./main -t 1

clean:
	rm -f  main
	rm -rf *.o
	rm -rf out/*.gf
	rm -rf out/*.mesh
