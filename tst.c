#include "stk.hpp"

//extern void backTraceIni(char*);

int f3(void){stk();stk()<<1;return 0;}

int f2(void){stk();return f3();}

int f1(void){stk();return f2();}

int f0(void){stk();return f1();}

int main(int argc, char *argv[]){
  stkIni(argv[0]);
  stk();
  return f0();
}
