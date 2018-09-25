CXXFLAGS = -fno-inline-functions -g -shared -fPIC -Wall 
LIBS = -Wl,-rpath,/home/camier1/home/okmm -Wl,-rpath,/home/camier1/home/stk -L/home/camier1/home/stk

all: okmm.so

%.so: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< -ldl $(LIBS) -lstk

cln:
	\rm -f *.so
