#include <iostream>
#include "mfem.hpp"

using namespace std;
using namespace mfem;

int main()
{
   KnotVector kv(2, 7);

   kv[0] = 0;
   kv[1] = 0;
   kv[2] = 0;
   kv[3] = 0.25;
   kv[4] = 0.5;
   kv[5] = 0.5; // Repeated knot
   kv[6] = 0.75;
   kv[7] = 1;
   kv[8] = 1;
   kv[9] = 1;

   kv.Print(cout);

   // Count number of elements, required for printing of shapes
   kv.GetElements();

   kv.PrintFunctions(cout);
}
