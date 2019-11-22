#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#include "drl4amr.hpp"


void ShowImg(socketstream &sock, double *img, int width, int height)
{
   Mesh mesh(width, height, Element::QUADRILATERAL, true, 1.0, 1.0, false);
   L2_FECollection fec(0, 2);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction gf(&fes, img);
   sock << "solution\n" << mesh << gf << flush;
}


int main(int argc, char *argv[])
{
   const int order = 2;
   Drl4Amr sim(order);

   socketstream vis("localhost", 19916);

   while (sim.GetNorm() > 0.01)
   {
#if 0
      const int el = static_cast<int>(drand48()*sim.GetNE());
      sim.Compute();
      sim.Refine(el);
      sim.GetFullImage();
      sim.GetFullWidth();
#else
      double *img = sim.GetLocalImage(sim.GetNE()/2);
      ShowImg(vis, img, sim.GetLocalWidth(), sim.GetLocalHeight());
      break;
#endif
   }
   return 0;
}
