
make:
	g++ -g -std=c++11 -I$(MFEM_DIR) \
	main.cpp elliptic_integral.cpp test.cpp initial_coefficient.cpp \
	-o main.o -L$(MFEM_DIR) -lmfem -lrt

clean:
	rm *.o
