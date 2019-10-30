#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#include "drl4amr.hpp"

int main(int argc, char *argv[])
{
   const int order = 3;
   Drl4Amr sim(order);

   while (sim.GetNorm() > 0.01)
   {
      sim.Compute();
      sim.Refine();
      sim.Update();
   }
   return 0;
}
