CXXFLAGS = -shared -fPIC -Wall 

all: okmm.so

%.so: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< -ldl

cln:
	\rm -f *.so
