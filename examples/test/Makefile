EXE = test.x

CXX =  mpicxx -g -O3 -I../../ -I/home/vikram/usr/local/metis/include -I/home/vikram/usr/local/hypre/include 

LDFLAGS = -L../../ -lmfem -L/home/vikram/usr/local/metis/lib -lmetis -L/home/vikram/usr/local/hypre/lib -lHYPRE -lrt

all:  $(EXE)

test2.x: t2.cpp 
	$(CXX) $^  $(LDFLAGS) -o $@ 

cns2d.x: cns2d.cpp 
	$(CXX) $^  $(LDFLAGS) -o $@ 

test4.x: t4.cpp 
	$(CXX) $^  $(LDFLAGS) -o $@ 

test5.x: t5.cpp 
	$(CXX) $^  $(LDFLAGS) -o $@ 

clean: 
	rm *.x 

