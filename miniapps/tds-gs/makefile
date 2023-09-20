
make:
	mpicxx -std=c++11 -I$(MFEM_DIR) -I/Users/dserino/Documents/mfem/../hypre/src/hypre/include \
	main.cpp double_integrals.cpp gs.cpp elliptic_integral.cpp test.cpp \
	initial_coefficient.cpp plasma_model.cpp field.cpp \
	sys_operator.cpp boundary.cpp diffusion_term.cpp \
	exact.cpp \
	-o main.o -L$(MFEM_DIR) -lmfem -w -L/Users/dserino/Documents/mfem/../hypre/src/hypre/lib -lHYPRE -L/Users/dserino/Documents/mfem/../metis-4.0 -lmetis

field:
	g++ -g -std=c++11 -I$(MFEM_DIR) \
	make_field.cpp \
	-o make_field.o -L$(MFEM_DIR) -lmfem -w 

test:
	./main.o -t 1

clean:
	rm *.o
	rm out/*.gf
	rm out/*.mesh
