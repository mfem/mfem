
make:
	g++ -g -std=c++11 -I$(MFEM_DIR) \
	main.cpp double_integrals.cpp gs.cpp elliptic_integral.cpp test.cpp \
	initial_coefficient.cpp plasma_model.cpp \
	sys_operator.cpp boundary.cpp diffusion_term.cpp \
	exact.cpp \
	-o main.o -L$(MFEM_DIR) -lmfem -w #-lrt


test:
	./main.o -t 1

clean:
	rm *.o
	rm out/*.gf
	rm out/*.mesh
