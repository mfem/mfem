#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static double r1_ = 0.1;

void trans(const Vector &x0, Vector &x1);

int main(int argc, char *argv[])
{

   int o = 3;
   int nz = 100;
   int nr = 20;

   OptionsParser args(argc, argv);
   args.AddOption(&o, "-o", "--order",
                  "Order of curved mesh");
   args.AddOption(&nz, "-nz", "--num-elem-z",
                  "Number of elements in z-direction");
   args.AddOption(&nr, "-nr", "--num-elem-r",
                  "Number of elements in radial direction");
   args.AddOption(&r1_, "-r1", "--radius at z=1",
                  "Radius of mesh at z=1");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh = Mesh::MakeCartesian2D(nz, nr, Element::QUADRILATERAL, 1, 2.0, 1.0);

   mesh.SetCurvature(o);

   mesh.Transform(trans);

   ofstream ofs("wham_warp.mesh");
   mesh.Print(ofs);
   ofs.close();
}

void trans(const Vector &x0, Vector &x1)
{
   double r = x0[1] * r1_;
   double z = x0[0] - 1.0;

   double phi1 = r * r * (0.5 + 9.5) / 2.0;

   x1.SetSize(2);
   x1[0] = z;
   x1[1] = sqrt(2.0 * phi1 / (0.5 + 9.5 * pow(z, 4)));
}
