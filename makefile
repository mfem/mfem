CXXLAGS = -shared -fPIC -Wall -Wno-deprecated-declarations

all: okmm.so

%.so: %.cc
	$(CXX) $(CXXLAGS) -o $@ $< -ldl

cln:
	\rm -f *.so
