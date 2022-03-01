//                      MFEM Example 31
//
// Compile with: make ex42
//
// Sample runs:  ex31 -m ../data/square-disc.mesh -alpha 0.33 -o 2
//               ex31 -m ../data/star.mesh -alpha 0.99 -o 3
//               ex31 -m ../data/inline-quad.mesh -alpha 0.2 -o 3
//               ex31 -m ../data/disc-nurbs.mesh -alpha 0.33 -o 3
//
//
// Description:
//
//  In this example we solve the following fractional PDE with MFEM:
//
//    ( - Δ )^α u = f  in Ω,      u = 0  on ∂Ω,      0 < α < 1,
//
//  To solve this FPDE, we rely on a rational approximation [2] of the normal
//  linear operator A^{-α}, where A = - Δ (with associated homogenous
//  boundary conditions). Namely, we first approximate the operator
//
//    A^{-α} ≈ Σ_{i=0}^N c_i (A + d_i I)^{-1},      d_0 = 0,   d_i > 0,
//
//  where I is the L2-identity operator and the coefficients c_i and d_i
//  are generated offline to a prescribed accuracy in a pre-processing step.
//  We use the triple-A algorithm [1] to generate the rational approximation
//  that this partial fractional expansion derives from. We then solve N+1
//  independent integer-order PDEs,
//
//    A u_i + d_i u_i = c_i f  in Ω,      u_i = 0  on ∂Ω,      i=0,...,N,
//
//  using MFEM and sum u_i to arrive at an approximate solution of the FPDE
//
//    u ≈ Σ_{i=0}^N u_i.
//
//
// References:
//
// [1] Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). The AAA algorithm
//     for rational approximation. SIAM Journal on Scientific Computing, 40(3),
//     A1494-A1522.
//
// [2] Harizanov, S., Lazarov, R., Margenov, S., Marinov, P., & Pasciak, J.
//     (2020). Analysis of numerical methods for spectral fractional elliptic
//     equations based on the best uniform rational approximation. Journal of
//     Computational Physics, 408, 109285.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ex31.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = true;
   double alpha = 0.2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Fractional exponent");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Array<double> coeffs, poles;

   ComputePartialFractionApproximation(alpha,coeffs,poles);

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   mesh.UniformRefinement();
   mesh.UniformRefinement();
   mesh.UniformRefinement();

   FiniteElementCollection *fec = new H1_FECollection(order, dim);

   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }


   ConstantCoefficient f(1.0);
   ConstantCoefficient one(1.0);
   GridFunction u(&fespace);
   u = 0.;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream xout;
   socketstream uout;
   if (visualization)
   {
      xout.open(vishost, visport);
      xout.precision(8);
      uout.open(vishost, visport);
      uout.precision(8);
   }

   for (int i = 0; i<coeffs.Size(); i++)
   {
      LinearForm b(&fespace);
      ProductCoefficient cf(coeffs[i], f);
      b.AddDomainIntegrator(new DomainLFIntegrator(cf));
      b.Assemble();

      GridFunction x(&fespace);
      x = 0.0;

      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      ConstantCoefficient c2(-poles[i]);
      a.AddDomainIntegrator(new MassIntegrator(c2));
      a.Assemble();

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cout << "Size of linear system: " << A->Height() << endl;

      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);

      // 12. Recover the solution as a finite element grid function.
      a.RecoverFEMSolution(X, b, x);

      u+=x;

      // 14. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         xout << "solution\n" << mesh << x << flush;
         uout << "solution\n" << mesh << u << flush;
      }
   }
   return 0;
}
