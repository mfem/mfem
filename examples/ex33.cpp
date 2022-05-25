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
// Verification runs:
//    ex33 -m ../data/inline-quad.mesh -ver -alpha <arbitrary> -o 2 -r 4
//    Note: the analytic solution to this problem is u(x) = sin(pi x) sin(pi y)
//          for all alpha.
//
// Description:
//
//  In this example we solve the following fractional PDE with MFEM:
//
//    ( - Δ )^α u = f  in Ω,      u = 0  on ∂Ω,      0 < α,
//
//  To solve this FPDE, we multiply with ( - Δ )^(-N) where the integer
//  N is given by floor(α). We obtain
//
//    ( - Δ )^(α-N) u = ( - Δ )^(-N) f  in Ω,      u = 0  on ∂Ω,      0 < α.
//
//  We first compute the right hand side by solving the integer order PDE
//
//   ( - Δ )^N g = f  in Ω,      g = 0  on ∂Ω,
//
//  The remaining FPDE is then given by
//
//  ( - Δ )^(α-N) u = g  in Ω,      u = 0  on ∂Ω.
//
//  We rely on a rational approximation [2] of the normal linear operator
//  A^{-α + N}, where A = - Δ (with associated homogeneous boundary conditions)
//  and (a-N) in (0,1). We approximate the operator
//
//    A^{-α+N} ≈ Σ_{i=0}^M c_i (A + d_i I)^{-1},      d_0 = 0,   d_i > 0,
//
//  where I is the L2-identity operator and the coefficients c_i and d_i
//  are generated offline to a prescribed accuracy in a pre-processing step.
//  We use the triple-A algorithm [1] to generate the rational approximation
//  that this partial fractional expansion derives from. We then solve M+1
//  independent integer-order PDEs,
//
//    A u_i + d_i u_i = c_i g  in Ω,      u_i = 0  on ∂Ω,      i=0,...,M,
//
//  using MFEM and sum u_i to arrive at an approximate solution of the FPDE
//
//    u ≈ Σ_{i=0}^M u_i.
//
//  (If alpha is an integer, we stop after the first PDE was solved.)
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
#include <math.h>

#include "ex33.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int num_refs = 3;
   double alpha = 0.5;
   bool visualization = true;
   bool verification = false;

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
   args.AddOption(&verification, "-ver", "--verification", "-no-ver",
                  "--no-verification",
                  "Use sinusoidal function (rhs) for analytic comparison.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Array<double> coeffs, poles;
   int progress_steps = 1;

   // 2. Compute the rational expansion coefficients that define the
   //    integer-order PDEs.
   int power_of_laplace = floor(alpha);
   double exponent_to_approximate = alpha - power_of_laplace;
   bool integer_order = false;
   // Check if alpha is an integer or not.
   if (alpha - power_of_laplace !=0)
   {
      mfem::out << "Approximating the fractional exponent "
                << exponent_to_approximate
                << endl;
      ComputePartialFractionApproximation(exponent_to_approximate, coeffs,
                                          poles);
   }
   else
   {
      integer_order = true;
      mfem::out << "Treating integer order PDE." << endl;
   }

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
   auto func = [&alpha](const Vector &x)
   {
      return pow(2*pow(M_PI,2), alpha) * sin(M_PI * x[0]) * sin(M_PI * x[1]);
   };
   FunctionCoefficient f(func);
   ConstantCoefficient one(1.0);
   GridFunction u(&fespace);
   u = 0.;

   // 8. Prepare for visualization.
   char vishost[] = "localhost";
   int  visport   = 19916;

   // 9 Compute Right Hand Side
   LinearForm rhs(&fespace);
   if (verification)
   {
      // This statement is only relevant for the verification of the code. It
      // uses a different RHS such that an analytic solution is known and easy
      // to compare with the numerical one. The FPDE becomes:
      // (-Δ)^α u = (2\pi ^2)^α sin(\pi x) sin(\pi y) on [0,1]^2
      // -> u(x,y) = sin(\pi x) sin(\pi y)
      rhs.AddDomainIntegrator(new DomainLFIntegrator(f));
   }
   else
   {
      rhs.AddDomainIntegrator(new DomainLFIntegrator(one));
   }
   rhs.Assemble();

   // ------------------------------------------------------------------------
   // 10. Solve the PDE -Δ ^ N g = f, i.e. compute g = (-Δ)^{-1}^N f.
   // ------------------------------------------------------------------------

   if (power_of_laplace > 0)
   {
      // 10.1 Compute Stiffnes Matrix
      BilinearForm k(&fespace);
      k.AddDomainIntegrator(new DiffusionIntegrator(one));
      k.Assemble();

      // 10.2 Compute Mass Matrix
      BilinearForm m(&fespace);
      m.AddDomainIntegrator(new MassIntegrator(one));
      m.Assemble();
      SparseMatrix mass;
      Array<int> empty;
      m.FormSystemMatrix(empty, mass);

      // 10.3 from the system of equations
      Vector B, X;
      GridFunction g(&fespace);
      OperatorPtr Op;
      k.FormLinearSystem(ess_tdof_list, g, rhs, Op, X, B);
      GSSmoother M((SparseMatrix&)(*Op));

      mfem::out << "\nComputing -Δ ^ -" << power_of_laplace
                << " ( f ) " << endl;
      for (int i = 0; i < power_of_laplace; i++)
      {
         // 10.4 Solve the linear system A X = B (N times).
         PCG(*Op, M, B, X, 3, 200, 1e-12, 0.0);
         // 10.5 Visualize the solution g of -Δ ^ N g = f in the last step
         if (visualization && i == power_of_laplace - 1)
         {
            socketstream fout;
            ostringstream oss_f;
            fout.open(vishost, visport);
            fout.precision(8);
            k.RecoverFEMSolution(X, rhs, g);
            oss_f.str(""); oss_f.clear();
            oss_f << "Step " << progress_steps++ << ": Solution of PDE -Δ ^ "
                  << power_of_laplace
                  << " g = f";
            fout << "solution\n" << mesh << g
                 << "window_title '" << oss_f.str() << "'" << flush;
         }
         mass.Mult(X, B);
         X.SetSubVectorComplement(ess_tdof_list,0.0);
      }
      // 10.6 Extract solution for the next step.
      rhs = B;
   }

   // ------------------------------------------------------------------------
   // 11. Solve the fractional PDE by solving M integer order PDEs and adding
   //     up the solutions.
   // ------------------------------------------------------------------------
   if (!integer_order)
   {
      // Setup visualization.
      socketstream xout, uout;
      ostringstream oss_x, oss_u;
      if (visualization)
      {
         xout.open(vishost, visport);
         xout.precision(8);
         uout.open(vishost, visport);
         uout.precision(8);
      }
      // Iterate over all expansion coefficient that contribute to the
      // solution.
      for (int i = 0; i < coeffs.Size(); i++)
      {
         mfem::out << "\nSolving PDE -Δ u + " << -poles[i]
                   << " u = " << coeffs[i] << " f " << endl;

         // 11.1 Set up the linear form b(.) for integer-order PDE solve.
         //      (c_i * g)
         Vector b (rhs);
         b *= coeffs[i];

         // 11.2 Define GridFunction for integer-order PDE solve.
         GridFunction x(&fespace);
         x = 0.0;

         // 11.3 Set up the bilinear form a(.,.) for integer-order PDE solve.
         BilinearForm a(&fespace);
         a.AddDomainIntegrator(new DiffusionIntegrator(one));
         ConstantCoefficient c2(-poles[i]);
         a.AddDomainIntegrator(new MassIntegrator(c2));
         a.Assemble();

         // 11.4 Assemble the bilinear form and the corresponding linear system.
         OperatorPtr A;
         Vector B, X;
         a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

         // 11.5 Solve the linear system A X = B.
         GSSmoother M((SparseMatrix&)(*A));

         PCG(*A, M, B, X, 3, 200, 1e-12, 0.0);

         // 11.6 Recover the solution as a finite element grid function.
         a.RecoverFEMSolution(X, b, x);

         // 11.7 Accumulate integer-order PDE solutions.
         u+=x;

         // 11.8 Send the solutions by socket to a GLVis server.
         if (visualization)
         {
            oss_x.str(""); oss_x.clear();
            oss_x << "Step " << progress_steps
                  << ": Solution of PDE -Δ u + " << -poles[i]
                  << " u = " << coeffs[i] << " f";
            xout << "solution\n" << mesh << x
                 << "window_title '" << oss_x.str() << "'" << flush;

            oss_u.str(""); oss_u.clear();
            oss_u << "Step " << progress_steps + 1
                  << ": Solution of fractional PDE -Δ^" << alpha
                  << " u = f";
            uout << "solution\n" << mesh << u
                 << "window_title '" << oss_u.str() << "'" << flush;
         }
      }
   }
   // 12. Free the used memory.
   delete fec;
   return 0;
}
