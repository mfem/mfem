#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static double ba_ = 0.5;
static double bb_ = 9.5;
static double ra_ = 0.4;
static double rw_ = 0.1;
static double ul_ = 0.5;
static double uw_ = 0.3;

//static double r1_ = 0.1;

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
   args.AddOption(&ra_, "-ra", "--radius",
                  "Maximum Radius of mesh");
   args.AddOption(&rw_, "-rw", "--r-trans-width",
                  "Transition width in r");
   args.AddOption(&ul_, "-ul", "--u-trans-loc",
                  "Location of transition in u");
   args.AddOption(&uw_, "-uw", "--u-trans-width",
                  "Transition width in u");
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

   ofstream ofs("wham_slab.mesh");
   mesh.Print(ofs);
   ofs.close();
}

void trans(const Vector &x0, Vector &x1)
{
   double z = x0[0] - 1.0;
   double r = ra_;

   double bz = ba_ + bb_ * pow(z, 4);
   double rl = sqrt(0.01 / bz);
   double rw = rw_ * rl / ra_;

   double u = x0[1];
   if (u < ul_ - 0.5 * uw_)
   {
      r = u * (rl - 0.5 * rw) / (ul_ - 0.5 * uw_);
   }
   else if (u < ul_ + 0.5 * uw_)
   {
      r = u * rw / uw_ + rl - rw * ul_ / uw_;
   }
   else
   {
      r = u * (ra_ - (rl + 0.5 * rw)) / (1.0 - ul_ - 0.5 * uw_) +
          (rl +0.5 * rw - ra_ * (ul_ + 0.5 * uw_)) / (1.0 - ul_ - 0.5 * uw_);
   }

   x1.SetSize(2);
   x1[0] = z;
   x1[1] = r;
}
