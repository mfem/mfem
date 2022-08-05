
make:
	#g++ -std=c++11 elliptic_integral.cpp
	g++ -g -std=c++11 -I$(MFEM_DIR) main.cpp elliptic_integral.cpp -o main.o -L$(MFEM_DIR) -lmfem -lrt

clean:
	rm *.o
