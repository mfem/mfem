#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#include "drl4amr.hpp"

int main(int argc, char *argv[])
{
   const int order = 2;
   Drl4Amr sim(order);
   while (sim.GetNorm() > 0.01)
   {
      const int e = static_cast<int>(drand48()*sim.GetNE());
      sim.Compute();
      sim.Refine(e);
      sim.GetImage();
      //sim.GetImageSize();
      sim.GetElemIdField();
      sim.GetElemDepthField();
   }
   return 0;
}
