#include "dbg.hpp"

extern void backTraceIni(char*);

int f3(void){dbg();return 0;}

int f2(void){dbg();return f3();}

int f1(void){dbg();return f2();}

int f0(void){dbg();return f1();}

int main(int argc, char *argv[]){
  dbgIni(argv[0]);
  dbg();
  return f0();
}
