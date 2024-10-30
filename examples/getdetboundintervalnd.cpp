// Sample runs:
// 1D
//////// First run this file to generate data
//   make getdetboundintervalnd -j && ./getdetboundintervalnd -o 2 -mr 6 -m ../data/inline-segment.mesh
//////// Plot in Python
// Then run python3 plot1Dbndsrecursive.py --N 5 --M 6

// 2D
///// Run this file to get data for determinant of a single element mesh that is inverted somewhere between the mesh nodes.
// make getdetboundintervalnd -j && ./getdetboundintervalnd -o 2 -mr 6 -m semi-invert.mesh
// Now plot it in Python (todo: Tarik to provide the script)
//


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "gslib.h"
#include "bounds.hpp"

using namespace std;
using namespace mfem;

double GenerateRandomized1D(int n1D, Poly_1D::Basis &basis1d,
                           Vector &solcoeff, int seed = 0, int nbrute = 1000)
{
   solcoeff.SetSize(n1D);
   solcoeff.Randomize(seed);

    double noise_min = 1.0;
   double noise_max = 2.75;
   solcoeff *= noise_max - noise_min;
   solcoeff += noise_min;
   // Set boundary values such that minima is not at end points.
   solcoeff(0) = std::max(noise_max, solcoeff.Max());
   solcoeff(n1D-1) = std::max(noise_max, solcoeff.Max());

   // Brute force to find the minimum of this function
   double minval = std::numeric_limits<double>::infinity();
   double maxval = -std::numeric_limits<double>::infinity();
   Vector basisvals(n1D);
   for (int i = 0; i < nbrute; i++)
   {
      double ipx = i/(nbrute-1.0);
      basis1d.Eval(ipx, basisvals);
      double val = basisvals*solcoeff;
      minval = std::min(minval, val);
      maxval = std::max(maxval, val);
   }

   // offset the minimum to make it barely negative
   while (minval > 0)
   {
      solcoeff -= (solcoeff.Min()-0.001*rand_real());

      minval = std::numeric_limits<double>::infinity();
      maxval = -std::numeric_limits<double>::infinity();
      for (int i = 0; i < nbrute; i++)
      {
         double ipx = i/(nbrute-1.0);
         basis1d.Eval(ipx, basisvals);
         double val = basisvals*solcoeff;
         minval = std::min(minval, val);
         maxval = std::max(maxval, val);
      }
      minval = std::min(minval, solcoeff.Min());
      maxval = std::max(maxval, solcoeff.Max());
   }

   MFEM_VERIFY(minval < 0 && solcoeff.Min() > 0, "The minimum value of the function is not negative or the coefficient has become negative");
   return minval;
}

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   int order = 3;
   int mr = 8;
   int nbrute = 1000;
   int seed = 5;
   int outsuffix = 0;
   string mesh_file = "semi-invert.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&mr, "-mr", "--mr", "Finite element polynomial degree");
   args.AddOption(&nbrute, "-nh", "--nh", "Finite element polynomial degree");
   args.AddOption(&seed, "-seed", "--seed", "Seed");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&outsuffix, "-out", "--out", "out suffix");
   args.ParseCheck();

   Mesh mesh(mesh_file);
   GridFunction *x = mesh.GetNodes();
   if (!x)
   {
      mesh.SetCurvature(order, false, -1, Ordering::byNODES);
      x = mesh.GetNodes();
   }
   int mesh_order = x->FESpace()->GetOrder(0);
   std::cout << "The mesh order is: " << mesh_order << std::endl;

   int det_order = 2*mesh_order;
   const int dim = mesh.Dimension();
   std::cout << "The determinant order is: " << det_order << std::endl;

   int n1D = det_order + 1;
   int b_type = BasisType::GaussLobatto;
   Poly_1D poly1d;
   Poly_1D::Basis basis1d(poly1d.GetBasis(det_order, b_type));
   std::cout << "N and M are: " << n1D << " " << mr << std::endl;

   Vector ref_nodes(n1D);
   const double *temp = poly1d.GetPoints(det_order, b_type);
   ref_nodes = temp;

   Vector chebX = GetChebyshevNodes(mr); // [-1, 1]
   ScaleNodes(chebX, 0.0, 1.0, chebX); // [0, 1]

   DenseMatrix lbound, ubound;
   GetGSLIBBasisBounds(n1D, mr, lbound, ubound);

   IntegrationRule irule(n1D);
   QuadratureFunctions1D::GaussLobatto(n1D, &irule);
   Vector gllW(n1D), gllX(n1D);

   for (int i = 0; i < n1D; i++)
   {
      gllW(i) = irule.IntPoint(i).weight;
      gllX(i) = irule.IntPoint(i).x;
   }

   // 1D stuff
   if (dim == 1)
   {
      // Read custom bounds
      std::string filename = "bnddata_" + std::to_string(n1D) + "_" + std::to_string(mr) + "_chebyshev.txt";

      DenseMatrix lboundT, uboundT;
      Vector gllT, intT;
      ReadCustomBounds(gllT, intT, lboundT, uboundT, filename);

      // Generate a random solution that is -ve at some point.
      Vector randomsol(n1D);
      double brutemin = GenerateRandomized1D(n1D, basis1d, randomsol, seed, nbrute);

      std::cout << "The minimum value of the function is: " << brutemin << std::endl;

      Vector intmin, intmax;
      Get1DBounds(gllX, chebX, gllW, lbound, ubound, randomsol, intmin, intmax, true);
      std::cout << "The minimum bounded value of the function using GSLIB is between: " << intmin.Min() << " " << intmax.Min() << std::endl;

      Vector intminT, intmaxT;
      Get1DBounds(gllX, intT, gllW, lboundT, uboundT, randomsol, intminT, intmaxT, true);
      std::cout << "The minimum bounded value of the function using custom bounds is between: " << intminT.Min() << " " << intmaxT.Min() << std::endl;

      // Write the bounds to file
      ofstream myfile;
      {
         myfile.open ("bnd_comp_gslib_"+ std::to_string(n1D) + "_" + std::to_string(mr) + "_out=" + std::to_string(outsuffix) + ".txt");
         myfile << n1D << std::endl;
         gllX.Print(myfile,1);
         myfile << mr << std::endl;
         chebX.Print(myfile,1);
         randomsol.Print(myfile,1);
         intmin.Print(myfile,1);
         intmax.Print(myfile,1);
         myfile.close();
      }
      {
         myfile.open ("bnd_comp_tarik_"+ std::to_string(n1D) + "_" + std::to_string(mr) + "_out=" + std::to_string(outsuffix) + ".txt");
         myfile << n1D << std::endl;
         gllX.Print(myfile,1);
         myfile << mr << std::endl;
         chebX.Print(myfile,1);
         randomsol.Print(myfile,1);
         intminT.Print(myfile,1);
         intmaxT.Print(myfile,1);
         myfile.close();
      }

      // Recursively get extrema and write results to file
      int maxdepth = 10;
      Array<double> intptsO,intminO,intmaxO;
      Array<int> intdepthO;
      GetRecursiveExtrema1D(1, maxdepth, randomsol, gllX, gllW,
                           intT, lboundT, uboundT,
                           basis1d, 0.0, 1.0,
                           intptsO, intminO, intmaxO, intdepthO);

      {
         myfile.open ("recursive_bnd_"+ std::to_string(n1D) + "_" + std::to_string(mr) + "_out=" + std::to_string(outsuffix) + ".txt");
         myfile << n1D << std::endl;
         gllX.Print(myfile,1);
         randomsol.Print(myfile,1);
         intptsO.Print(myfile,1);
         intminO.Print(myfile,1);
         intmaxO.Print(myfile,1);
         intdepthO.Print(myfile,1);
         myfile.close();
      }

      std::cout << "The minimum value determined from recursion is at-least: " << intmaxO.Min() << std::endl;
   }
   else if (dim == 2)
   {

      L2_FECollection fec(det_order, dim, BasisType::GaussLobatto);
      FiniteElementSpace fespace(&mesh, &fec);
      GridFunction detgf(&fespace);
      Array<int> dofs;
      Array<double> intptsx, intptsy, intmin,intmax, intdepth;
      double brute_min_det = std::numeric_limits<double>::infinity();
      double brute_max_det = -std::numeric_limits<double>::infinity();

      std::string filename = "bnddata_" + std::to_string(n1D) + "_" + std::to_string(mr) + "_opt.txt";

      DenseMatrix lboundT, uboundT;
      Vector gllT, intT;
      ReadCustomBounds(gllT, intT, lboundT, uboundT, filename);

      for (int e = 0; e < mesh.GetNE(); e++)
      {
         const FiniteElement *fe = fespace.GetFE(e);
         const IntegrationRule ir = fe->GetNodes();
         ElementTransformation *transf = mesh.GetElementTransformation(e);
         DenseMatrix Jac(fe->GetDim());
         const NodalFiniteElement *nfe = dynamic_cast<const NodalFiniteElement*>
                                       (fe);
         const Array<int> &irordering = nfe->GetLexicographicOrdering();
         IntegrationRule ir2 = irordering.Size() ?
                              PermuteIR(&ir, irordering) :
                              ir;

         Vector detvals(ir2.GetNPoints());
         Vector loc(dim);
         for (int q = 0; q < ir2.GetNPoints(); q++)
         {
            IntegrationPoint ip = ir2.IntPoint(q);
            transf->SetIntPoint(&ip);
            transf->Transform(ip, loc);
            Jac = transf->Jacobian();
            detvals(q) = Jac.Det();
         }

         fespace.GetElementDofs(e, dofs);
         if (irordering.Size())
         {
            for (int i = 0; i < dofs.Size(); i++)
            {
               detgf(dofs[i]) = detvals(irordering[i]);
            }
         }
         else
         {
            detgf.SetSubVector(dofs, detvals);
         }

         // GSLIBBound detb(n1D, mr/n1D);
         // Vector qpmin;
         // Vector qpmax;
         // detb.GetBounds(detvals, qpmin, qpmax, dim);
         // std::cout << "Element: " << e << " ,the minimum determinant using GSLIB is between: " << qpmin.Min() << " " << qpmax.Min() << std::endl;

         Vector qpminCus, qpmaxCus;
         Get2DBounds(gllX, intT, gllW, lboundT, uboundT, detvals, qpminCus, qpmaxCus, true);
         // my implementation of gslib
         // Get2DBounds(gllX, chebX, gllW, lbound, ubound, randomsol, qpminCus, qpmaxCus, true);
         std::cout << "Element " << e << ": the minimum determinant using custom bounds is between: " << qpminCus.Min() << " " << qpmaxCus.Min() << std::endl;


         // Brute force also
         IntegrationPoint ip;
         double brute_el_min = std::numeric_limits<double>::infinity();
         double brute_el_max = -std::numeric_limits<double>::infinity();

         for (int j = 0; j < nbrute; j++)
         {
            for (int i = 0; i < nbrute; i++)
            {
               ip.x = i/(nbrute-1.0);
               ip.y = j/(nbrute-1.0);
               transf->SetIntPoint(&ip);
               Jac = transf->Jacobian();
               double detj = Jac.Det();
               brute_el_min = std::min(brute_el_min, detj);
               brute_el_max = std::max(brute_el_max, detj);
            }
         }
         brute_min_det = std::min(brute_el_min, brute_min_det);
         brute_max_det = std::max(brute_el_max, brute_max_det);

         std::cout << "Element " << e << ": the minimum determinant using brute force is: " << brute_el_min << std::endl;

         // Recursion
         int maxdepth = 5;
         if (qpminCus.Min() < 0 && qpmaxCus.Min() > 0)
         {
            GetRecursiveExtrema2D(1, maxdepth, e, detgf, gllX, intT, gllW,
                                  lboundT, uboundT, 0.0, 1.0, 0.0, 1.0, intptsx, intptsy, intmin, intmax, intdepth);
         }
      }

      {
         ofstream myfile;
         myfile.open ("2DcustomboundinfoM" + std::to_string(mr) + ".txt");
         for (int i = 0; i < intptsx.Size(); i++)
         {
            myfile << i << " " << intptsx[i] << " " << intptsy[i] << " " <<
                     intmin[i] << " " << intmax[i] << " " << intdepth[i] << std::endl;
         }
         myfile.close();
      }
   }

   return 0;
}
