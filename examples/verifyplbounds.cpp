// Sample runs:
// make verifyplbounds -j4 && mpirun -np 1 verifyplbounds -nr 5 -mr 9 -no-gll


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "gslib.h"
#include "bounds.hpp"

using namespace std;
using namespace mfem;

// Piecewise linear values given by (x,y), interpolate at xv
double GetPLValue(const Vector &x, const Vector &y, double xv)
{
   int n = x.Size();
   if (xv < x(0))
   {
      return y(0);
   }
   else if (xv > x(n-1))
   {
      return y(n-1);
   }

   double yv = 0.0;
   for (int i = 0; i < n-1; i++)
   {
      double xm = x(i),
             xp = x(i+1),
             ym = y(i),
             yp = y(i+1);
      if (xv >= xm && xv <= xp)
      {
         yv = ym + (xv-xm)/(xp-xm)*(yp-ym);
         break;
      }
   }
   return yv;
}

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command line options.
   int nr = 3;
   int mr = 8;
   int nbrute = 100000;
   int seed = 5;
   int outsuffix = 0;
   string mesh_file = "semi-invert.mesh";
   bool gll = true;

   OptionsParser args(argc, argv);
   args.AddOption(&nr, "-nr", "--nr", "Finite element polynomial degree");
   args.AddOption(&mr, "-mr", "--mr", "Finite element polynomial degree");
   args.AddOption(&nbrute, "-nh", "--nh", "Finite element polynomial degree");
   args.AddOption(&seed, "-seed", "--seed", "Seed");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&outsuffix, "-out", "--out", "out suffix");
   args.AddOption(&gll, "-gll", "--gll", "-no-gll",
                  "--no-gll", "GLL or GL nodes");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // const int dim = pmesh->Dimension();
   if (myid == 0)
   {
      std::cout << "The function order is: " << nr-1 << std::endl;
   }

   std::string filename = "bnddata_" + std::to_string(nr) + "_" + std::to_string(
                             mr) + "_opt.txt";

   auto fileExists = [](const std::string& filepath) -> bool
   {
      std::ifstream file(filepath);
      return file.good();
   };

   PLBound plb;
   if (fileExists(filename) && gll)
   {
      plb.Setup(filename);
   }
   else
   {
      std::cout << nr << " " << mr << " Setup gslib type bounds\n";
      plb.Setup(nr, mr, gll);
   }

   const Vector gllX = plb.GetGLLX();
   const Vector intX = plb.GetIntX();
   const DenseMatrix lbound = plb.GetLBound();
   const DenseMatrix ubound = plb.GetUBound();
   Poly_1D::Basis &basis1d(poly1d.GetBasis(nr-1, gll ?
                                                 BasisType::GaussLobatto :
                                                 BasisType::GaussLegendre));

   Vector bv(nr); //basis value

   Vector lb_i, ub_i; // lower piecewise bound
   for (int j = 0; j < nbrute; j++)
   {
      double xv = (j)*1.0/(nbrute-1.0); //[0,1]
      bv = 0.0;
      if (gll)
      {
         if (j == 0)
         {
            bv(0) = 1.0;
         }
         else if (j  == nbrute-1)
         {
            bv(nr-1) = 1.0;
         }
         else
         {
            basis1d.Eval(xv, bv);
         }
      }
      else
      {
         basis1d.Eval(xv, bv);
      }
      for (int i = 0; i < nr; i++) //ith basis
      {
         double bv_xv = bv(i); // basis value at this point
         lbound.GetRow(i, lb_i);
         ubound.GetRow(i, ub_i);
         double lb_xv = GetPLValue(intX, lb_i, xv);
         double ub_xv = GetPLValue(intX, ub_i, xv);
         // if (lb_xv > bv_xv)
         // {
         //    std::cout << lb_xv << " " << bv_xv << " " << lb_xv-bv_xv << " k101\n";
         //    std::cout << " k10this" << std::endl;
         // }
         // if (ub_xv < bv_xv)
         // {
         //    std::cout << " k10this2" << std::endl;
         // }
         if (lb_xv > bv_xv || ub_xv < bv_xv)
         {
            // set precision to 14
            std::setprecision(10);

            std::cout << i << " " << j << " " << xv << " "<<
            bv_xv << " " << lb_xv << " " << ub_xv << " k101" << std::endl;
            MFEM_ABORT("PL bounds no good.");
         }
      }
   }

   return 0;
}
