tgt := mmcheck

CXX = g++
INC = -I$(HOME)/usr/local/include
FLG = -O2 -Wall -g $(INC) -shared -fPIC

.PHONY = clean cln all

mmcheck.so: mmcheck.cpp
	$(CXX) -O2 -Wall -g $(INC) -shared -fPIC -o $@ $< -ldl

clean cln:
	rm -rf mmcheck.*o
