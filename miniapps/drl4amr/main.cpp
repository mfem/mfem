#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#include "drl4amr.hpp"

int main(int argc, char *argv[])
{
   Drl4Amr sim(3);
   sim.Compute();
   sim.Refine();
   sim.Update();

   sim.Compute();
   sim.Refine();
   sim.Update();

   sim.Compute();
   sim.Refine();
   sim.Update();

   return 0;
}
