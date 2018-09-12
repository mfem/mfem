CFLAGS = -shared -fPIC -Wall -fno-exceptions

all: okmm.so

%.so: %.cc
	$(CC) $(CFLAGS) -o $@ $< -ldl

cln:
	\rm -f *.so
