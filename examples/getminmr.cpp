// Sample runs:
// 2D
// make getminmr -j && mpirun -np 1 getminmr -nr 15


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

bool ArePLBoundsGood(int nbrute, int nr, int mr, bool gll, bool print)
{
   PLBound plb;
   plb.Setup(nr, mr, gll);
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
         if (lb_xv > bv_xv || ub_xv < bv_xv)
         {
            // std::cout << i << " "<< j << " " << xv << " " << bv_xv << " " <<
            // lb_xv << " " << ub_xv << std::endl;
            // MFEM_ABORT(" ");
            return false;
         }
      }
   }

   if (print)
   {
      std::ostringstream filename;
      filename << "minmr_PL_" << (gll ? "GLL_" : "GL_") << std::to_string(nr)
                  << ".txt";
      ofstream myfile;
      myfile.open(filename.str());
      myfile << nr << std::endl;
      myfile << mr << std::endl;
      myfile << nbrute << std::endl;
      for (int i = 0; i < gllX.Size(); i++)
      {
         myfile << gllX(i) << std::endl;
      }
      for (int i = 0; i < intX.Size(); i++)
      {
         myfile << intX(i) << std::endl;
      }
      DenseMatrix lboundT;
      lboundT.Transpose(lbound);
      DenseMatrix uboundT;
      uboundT.Transpose(ubound);
      double *ld = lboundT.GetData();
      for (int i = 0; i < lboundT.Height()*lboundT.Width(); i++)
      {
         myfile << ld[i] << std::endl;
      }
      double *ud = uboundT.GetData();
      for (int i = 0; i < lboundT.Height()*lboundT.Width(); i++)
      {
         myfile << ud[i] << std::endl;
      }
      myfile.close();
   }
   return true;
}

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command line options.
   int nrmax = 10;
   int mr = 8;
   int nbrute = 100000;
   int seed = 5;
   int outsuffix = 0;
   string mesh_file = "semi-invert.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&nrmax, "-nr", "--nr", "Finite element polynomial degree");
   args.AddOption(&mr, "-mr", "--mr", "Finite element polynomial degree");
   args.AddOption(&nbrute, "-nh", "--nh", "Finite element polynomial degree");
   args.AddOption(&seed, "-seed", "--seed", "Seed");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&outsuffix, "-out", "--out", "out suffix");

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
      std::cout << "The max function order is: " << nrmax-1 << std::endl;
   }

   Array<int> nrvals, minmr_GLL, minmr_GL;

   for (int i = 2; i < nrmax+1; i++)
   {
      int nr = i;
      nrvals.Append(nr);
      for (int j = i+1; j < 4*i; j++)
      {
         int mr = j;
         bool bounds_good = ArePLBoundsGood(nbrute, nr, mr, true, true);
         if (bounds_good)
         {
            std::cout << mr << " points good for " << nr << " GLL points" << std::endl;
            minmr_GLL.Append(mr);
            break;
         }
      }

      for (int j = i+1; j < 4*i; j++)
      {
         int mr = j;
         bool bounds_good = ArePLBoundsGood(nbrute, nr, mr, false, true);
         if (bounds_good)
         {
            std::cout << mr << " points good for " << nr << " GL points" << std::endl;
            minmr_GL.Append(mr);
            break;
         }
      }
   }

   std::cout << std::setw(15) << std::left << "N"
                  << std::setw(15) << std::left << std::fixed << std::setprecision(2) << "M(GLL)"
                  << std::setw(15) << std::left << std::fixed << std::setprecision(2) << "M(GL)" << std::endl;
   for (int i = 0; i < nrvals.Size() ;i++)
   {
      std::cout << std::setw(15) << std::left << nrvals[i]
                  << std::setw(15) << std::left << std::fixed << std::setprecision(2) << minmr_GLL[i]
                  << std::setw(15) << std::left << std::fixed << std::setprecision(2) << minmr_GL[i] << std::endl;

   }

   return 0;
}
