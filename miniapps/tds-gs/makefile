
make:
	g++ -g -std=c++11 -I$(MFEM_DIR) \
	main.cpp elliptic_integral.cpp test.cpp \
	initial_coefficient.cpp plasma_model.cpp \
	sys_operator.cpp boundary.cpp diffusion_term.cpp \
	exact.cpp \
	-o main.o -L$(MFEM_DIR) -lmfem -lrt

clean:
	rm *.o
