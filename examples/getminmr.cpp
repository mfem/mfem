// Sample runs:
// 2D
//  make getminmr -j && ./getminmr -nrmax 7 -nrmin 3
// plot using python3 plotminmrbndscomp.py
//                    plotminmrbnds.py


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

bool ArePLBoundsGood(int nbrute, int nr, int mr, int nodetype, int intptype, bool print, Vector &errors, Vector &dxvals)
{
   PLBound plb;
   std::string fname = "../scripts/bnddata_spts_";
   if (nodetype == 1)
   {
      fname += "lobatto_" + std::to_string(nr);
   }
   else if (nodetype == 0)
   {
      fname += "legendre_" + std::to_string(nr);
   }
   else
   {
      std::cout << "Invalid nodetype\n";
      MFEM_ABORT(" ");
   }
   // intp = 0,1,2 - we use default bounds
   // 3,4,5 we use optimized bounds
   // std::cout << intptype << " " << fname << std::endl;
   double read_eps = 0.0;
   if (intptype == 3)
   {
      fname += "_bpts_legendre_" + std::to_string(mr)+ ".txt";
      plb.Setup(fname, read_eps);
   }
   else if (intptype == 4)
   {
      fname += "_bpts_lobatto_" + std::to_string(mr)+ ".txt";
      plb.Setup(fname, read_eps);
   }
   else if (intptype == 5)
   {
      fname += "_bpts_chebyshev_" + std::to_string(mr)+ ".txt";
      plb.Setup(fname, read_eps);
   }
   else if (intptype == 6)
   {
      fname += "_bpts_equispaced_" + std::to_string(mr)+ ".txt";
      plb.Setup(fname, read_eps);
   }
   else if (intptype == 7)
   {
      fname += "_bpts_opt_" + std::to_string(mr)+ ".txt";
      plb.Setup(fname, read_eps);
   }
   else
   {
      plb.Setup2(nr, mr, nodetype, intptype);
   }
   const Vector gllX = plb.GetGLLX();
   const Vector intX = plb.GetIntX();
   const DenseMatrix lbound = plb.GetLBound();
   const DenseMatrix ubound = plb.GetUBound();
   // gllX.Print();
   // intX.Print();
   MFEM_VERIFY(nodetype <= 1, "Only GL (0) or GLL (1) nodes supported\n");
   Poly_1D::Basis &basis1d(poly1d.GetBasis(nr-1, nodetype == 0 ?
                                                 BasisType::GaussLegendre :
                                                 BasisType::GaussLobatto));

   Vector bv(nr); //basis value
   int error_type = 2; // 0 - area, 1 - Linf, 2 - L2
   errors.SetSize(3);
   errors = 0.0;
   Vector opterror(nr);
   opterror = 0.0;

   dxvals.SetSize(3); // holds min/max/avg dx between interval points
   const Vector &intx = plb.GetIntX();
   dxvals(0) = std::numeric_limits<float>::max();
   dxvals(1) = -std::numeric_limits<float>::max();
   dxvals(2) = 0.0;
   for (int i = 0; i < intx.Size()-1; i++)
   {
      double x0 = intx(i);
      double x1 = intx(i+1);
      double dx = x1-x0;
      dxvals(0) = std::min(dxvals(0), dx);
      dxvals(1) = std::max(dxvals(1), dx);
      dxvals(2) += dx;
   }
   dxvals(2) *= 1.0/(intx.Size()-1);


   Vector lb_i, ub_i; // lower piecewise bound
   for (int j = 0; j < nbrute; j++)
   {
      double xv = (j)*1.0/(nbrute-1.0); //[0,1]
      bv = 0.0;
      basis1d.Eval(xv, bv);
      double h = 1.0/(nbrute-1.0);
      if (j == 0)
      {
         xv = 0.0;
      }
      else if (j == nbrute-1)
      {
         xv = 1.0;
      }
      for (int i = 0; i < nr; i++) //ith basis
      {
         double bv_xv = bv(i); // basis value at this point
         lbound.GetRow(i, lb_i);
         ubound.GetRow(i, ub_i);
         double lb_xv = GetPLValue(intX, lb_i, xv);
         double ub_xv = GetPLValue(intX, ub_i, xv);
         if (j != 0 && j != nbrute-1)
         {
            if ( lb_xv > bv_xv || ub_xv < bv_xv)
            {
               std::cout << setprecision(14);
               std::cout << j << " " << i << " " << xv << " " <<
               bv_xv << " " << lb_xv << " " << ub_xv << " intervalpt,basis(0index),xv,basisval,lowerboundval,upperboundval\n";
               errors = -1.0;
               return false;
            }
         }

         errors(1) = std::max(errors(1), ub_xv-bv_xv);
         errors(1) = std::max(errors(1), bv_xv-lb_xv);

         errors(2) += std::pow(ub_xv-bv_xv, 2.0) + std::pow(bv_xv-lb_xv, 2.0);
         opterror(i) += std::pow(ub_xv-bv_xv, 2.0) + std::pow(bv_xv-lb_xv, 2.0);
         if (j == 0 || j == nbrute-1)
         {
            errors(0) += 0.5*h*(ub_xv-lb_xv);
         }
         else
         {
            errors(0) += h*(ub_xv-lb_xv);
         }
      }
   }
   errors *= 1.0/nbrute;
   errors(2) = std::sqrt(errors(2));

   //  optimized bound use this
   for (int i = 0; i < nr; i++)
   {
      opterror(i) = std::sqrt(opterror(i)/nbrute);
   }
   errors(2) = opterror.Sum();


   if (print)
   {
      std::ostringstream filename;
      filename << "minmr_PL_" << (nodetype == 1 ? "GLL_" : "GL_") << std::to_string(nr) << "_Int_" << std::to_string(intptype)
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
   int nbrute = 100000;
   int nodetype = 1;
   int nrmin = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&nrmax, "-nrmax", "--nrmax", "Finite element polynomial degree");
   args.AddOption(&nbrute, "-nh", "--nh", "Finite element polynomial degree");
   args.AddOption(&nodetype, "-ntype", "--ntype", "0 = GL, 1 = GLL");
   args.AddOption(&nrmin, "-nrmin", "--nrmin", "Finite element polynomial degree");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   if (myid == 0)
   {
      std::cout << "The max function order is: " << nrmax-1 << std::endl;
   }

   Array<Array<int>*> minmra;
   minmra.SetSize(8); //set to 7 if you want to exclude optimal point location set, 8 if you want to include.
   Array<int> nrva;
   double area;
   Vector errors;
   Vector dxv;

   for (int kk = 0; kk < minmra.Size(); kk++)
   {
      Array<int> nrvals, minmr;
      int inttype = kk;
      for (int i = nrmin; i < nrmax+1; i++)
      {
         int nr = i;
         nrvals.Append(nr);
         if (kk == 0) { nrva.Append(nr); }
         for (int j = (kk > 2 ? 3 : i); j < 4*i; j++)
         {
            int mr = j;
            bool bounds_good = ArePLBoundsGood(nbrute, nr, mr, nodetype, inttype, true, errors, dxv);
            if (bounds_good)
            {
               std::cout << mr << " points good for " << nr << " GLL points" << std::endl;
               minmr.Append(mr);
               break;
            }
            if ( j == 4*i-1) {
               std::cout << "Could not find any useful points for nr=" << i << std::endl;
               minmr.Append(-1);
            }
         }
         minmra[kk] = new Array<int>();
         minmra[kk]->Append(minmr);
      }

      std::string nt = nodetype == 0 ? "N_{GL}" : "N_{GLL}";
      std::string itp = (inttype == 2 || inttype == 5)  ? "M_{Cheb}" :
                        (inttype == 0 || inttype == 3) ? "M_{GL+End}" :
                        (inttype == 1 || inttype == 4)  ? "M_{GLL}" :
                        (inttype == 6)  ? "M_{Uniform}" : "M_{opt}";
      std::cout << std::setw(15) << std::left << nt
                     << std::setw(15) << std::left << std::fixed << std::setprecision(2) << itp << std::endl;

      for (int i = 0; i < nrvals.Size() ;i++)
      {
         std::cout << std::setw(15) << std::left << nrvals[i]
                     << std::setw(15) << std::left << std::fixed << std::setprecision(2) << minmr[i] << std::endl;

      }
   }

   std::string nt = nodetype == 0 ? "N_{GL}" : "N_{GLL}";
   std::cout << std::setw(15) << std::left << nt
                  << std::setw(15) << std::left << std::fixed << "M_{GL+End}"
                  << std::setw(15) << std::left << std::fixed << "M_{GLL}"
                  << std::setw(15) << std::left << std::fixed << "M_{Cheb}"
                  << std::setw(15) << std::left << std::fixed << "M_{OPT,GL+End}"
                  << std::setw(15) << std::left << std::fixed << "M_{OPT,GLL}"
                  << std::setw(15) << std::left << std::fixed << "M_{OPT,Cheb}"
                  << std::setw(15) << std::left << std::fixed << "M_{OPT,Uniform}"
                  << std::setw(15) << std::left << std::fixed << "M_{OPT}"
                  << std::endl;
   for (int i = 0; i < nrva.Size() ;i++)
   {
      std::cout << std::setw(15) << std::left << nrva[i]
                  << std::setw(15) << std::left << std::fixed << std::setprecision(2) << (*minmra[0])[i];
      for (int j = 1; j < minmra.Size(); j++)
      {
         std::cout << std::setw(15) << std::left << std::fixed << std::setprecision(2) << (*minmra[j])[i];
      }
      std::cout << std::endl;
   }
   // MFEM_ABORT(" ");

   Array<Array<int>*> info_nmk;
   Array<double> info_error1;
   Array<double> info_error2;
   Array<double> info_error3;
   Array<double> info_size1;
   Array<double> info_size2;
   Array<double> info_size3;

   // Get compactness of bounds from the minimum computed MR to max MR for that N.
   for (int i = 0; i < nrva.Size(); i++)
   {
      int nr = nrva[i];
      Array<int> mra(3);
      mra[0] = (*minmra[0])[i];
      mra[1] = (*minmra[1])[i];
      mra[2] = (*minmra[2])[i];
      int mrmax = mra.Max();
      for (int kk = 0; kk < minmra.Size(); kk++)
      {
         int inttype = kk;
         int mrme = (*minmra[kk])[i];
         if (mrme == -1) { continue; }
         // for (int j = mrme; j < std::min(39,mrmax+1+2); j++)
         for (int j = mrme; j < std::min(20,20); j++)
         {
            int mr = j;
            bool good =  ArePLBoundsGood(nbrute, nr, mr, nodetype, inttype, false, errors, dxv);
            // MFEM_VERIFY(good, "Bounds not good. Something seriously wrong!");
            info_nmk.Append(new Array<int>());
            int ns = info_nmk.Size();
            info_nmk[ns-1]->Append(nr);
            info_nmk[ns-1]->Append(mr);
            info_nmk[ns-1]->Append(kk);
            info_error1.Append(errors(0));
            info_error2.Append(errors(1));
            info_error3.Append(errors(2));
            info_size1.Append(dxv(0));
            info_size2.Append(dxv(1));
            info_size3.Append(dxv(2));
         }
      }
   }

   std::setprecision(12);

   std::ostringstream filename;
   filename << "minmr_bnd_comp.txt";
   ofstream myfile;
   myfile.open(filename.str());

   for (int i = 0; i < info_nmk.Size(); i++)
   {
      myfile << (*info_nmk[i])[0] << " " << std::endl;
      myfile << (*info_nmk[i])[1] << " " << std::endl;
      myfile << (*info_nmk[i])[2] << " " << std::endl;
      myfile << std::setprecision(12) << info_error1[i] << std::endl;
      myfile << std::setprecision(12) << info_error2[i] << std::endl;
      myfile << std::setprecision(12) << info_error3[i] << std::endl;
      myfile << std::setprecision(12) << info_size1[i] << std::endl;
      myfile << std::setprecision(12) << info_size2[i] << std::endl;
      myfile << std::setprecision(12) << info_size3[i] << std::endl;
      std::cout << (*info_nmk[i])[0] << " " <<
                   (*info_nmk[i])[1] << " " <<
                   (*info_nmk[i])[2] << " " <<
                  std::setprecision(12) <<
                   info_error1[i] << " " <<
                   info_error2[i] << " " <<
                   info_error3[i] << " " <<  " k10info\n";
   }

   return 0;
}
