//                                MFEM Example 33
//
// Compile with: make ex33
//
// Sample runs:  ex33 -m ../data/square-disc.mesh -alpha 0.33 -o 2
//               ex33 -m ../data/star.mesh -alpha 0.99 -o 3
//               ex33 -m ../data/inline-quad.mesh -alpha 0.5 -o 3
//               ex33 -m ../data/disc-nurbs.mesh -alpha 0.33 -o 3
//               ex33 -m ../data/l-shape.mesh -alpha 0.33 -o 3 -r 4
//
// Description:
//
//  In this example we solve the following fractional PDE with MFEM:
//
//    ( - Δ )^α u = f  in Ω,      u = 0  on ∂Ω,      0 < α < 1,
//
//  To solve this FPDE, we rely on a rational approximation [2] of the normal
//  linear operator A^{-α}, where A = - Δ (with associated homogeneous
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

#include "ex33.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int num_refs = 3;
   bool visualization = true;
   double alpha = 0.5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&num_refs, "-r", "--refs",
                  "Number of uniform refinements");
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

   // 2. Compute the coefficients that define the integer-order PDEs.
   ComputePartialFractionApproximation(alpha,coeffs,poles);

   // 3. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution.
   for (int i = 0; i < num_refs; i++)
   {
      mesh.UniformRefinement();
   }

   // 5. Define a finite element space on the mesh.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 7. Define diffusion coefficient, load, and solution GridFunction.
   ConstantCoefficient f(1.0);
   ConstantCoefficient one(1.0);
   GridFunction u(&fespace);
   u = 0.;

   // 8. Prepare for visualization.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream xout, uout;
   ostringstream oss_x, oss_u;
   if (visualization)
   {
      xout.open(vishost, visport);
      xout.precision(8);
      uout.open(vishost, visport);
      uout.precision(8);
   }

   for (int i = 0; i < coeffs.Size(); i++)
   {
      // 9. Set up the linear form b(.) for integer-order PDE solve.
      LinearForm b(&fespace);
      ProductCoefficient cf(coeffs[i], f);
      b.AddDomainIntegrator(new DomainLFIntegrator(cf));
      b.Assemble();

      // 10. Define GridFunction for integer-order PDE solve.
      GridFunction x(&fespace);
      x = 0.0;

      // 11. Set up the bilinear form a(.,.) for integer-order PDE solve.
      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      ConstantCoefficient c2(-poles[i]);
      a.AddDomainIntegrator(new MassIntegrator(c2));
      a.Assemble();

      // 12. Assemble the bilinear form and the corresponding linear system.
      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      // 13. Solve the linear system A X = B.
      GSSmoother M((SparseMatrix&)(*A));

      mfem::out << "\nSolving PDE -Δ u + " << -poles[i]
                << " u = " << coeffs[i] << " f " << endl;
      PCG(*A, M, B, X, 3, 200, 1e-12, 0.0);

      // 14. Recover the solution as a finite element grid function.
      a.RecoverFEMSolution(X, b, x);

      // 15. Accumulate integer-order PDE solutions.
      u+=x;

      // 16. Send the solutions by socket to a GLVis server.
      if (visualization)
      {
         oss_x.str(""); oss_x.clear();
         oss_x << "Solution of PDE -Δ u + " << -poles[i]
               << " u = " << coeffs[i] << " f";
         xout << "solution\n" << mesh << x
              << "window_title '" << oss_x.str() << "'" << flush;

         oss_u.str(""); oss_u.clear();
         oss_u << "Solution of fractional PDE -Δ^" << alpha
               << " u = f";
         uout << "solution\n" << mesh << u
              << "window_title '" << oss_u.str() << "'" << flush;
      }
   }

   // 17. Free the used memory.
   delete fec;
   return 0;
}
