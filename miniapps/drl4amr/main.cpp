#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#include "drl4amr.hpp"


void ShowImg(socketstream &sock, double *img, int width, int height,
             const char *title = NULL)
{
   Mesh mesh(width, height, Element::QUADRILATERAL, true, 1.0, 1.0, false);
   L2_FECollection fec(0, 2);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction gf(&fes, img);
   sock << "solution\n" << mesh << gf;
   sock << "keys Rjlm\n";
   if (title)
   {
      sock << "window_title '" << title << "'\n";
   }
   sock << flush;
}


int main(int argc, char *argv[])
{
   const int order = 2;
   Drl4Amr sim(order);

   socketstream vis1("localhost", 19916);
   socketstream vis2("localhost", 19916);
   socketstream vis3("localhost", 19916);
   socketstream vis4("localhost", 19916);

   //sim.Refine(9);
   sim.RandomRefine();
   sim.RandomRefine();
   sim.RandomRefine();

   sim.Compute();

   while (sim.GetNorm() > 0.01)
   {
#if 0
      const int el = static_cast<int>(drand48()*sim.GetNE());
      sim.Compute();
      sim.Refine(el);
      sim.GetFullImage();
      sim.GetFullWidth();
#else
      double *img = sim.GetLocalImage(142);
      int w = sim.GetLocalWidth();
      int h = sim.GetLocalWidth();
      ShowImg(vis1, img, w, h, "Solution");
      ShowImg(vis2, img + w*h, w, h, "dx");
      ShowImg(vis3, img + 2*w*h, w, h, "dy");
      ShowImg(vis4, img + 3*w*h, w, h, "Depth");
      break;
#endif
   }
   return 0;
}
