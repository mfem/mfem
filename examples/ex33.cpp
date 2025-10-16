//                                MFEM Example 33
//
// Compile with: make ex33
//
// Sample runs:  ex33 -m ../data/square-disc.mesh -alpha 0.33 -o 2
//               ex33 -m ../data/square-disc.mesh -alpha 4.5 -o 3
//               ex33 -m ../data/star.mesh -alpha 1.4 -o 3
//               ex33 -m ../data/star.mesh -alpha 0.99 -o 3
//               ex33 -m ../data/inline-quad.mesh -alpha 0.5 -o 3
//               ex33 -m ../data/amr-quad.mesh -alpha 1.5 -o 3
//               ex33 -m ../data/disc-nurbs.mesh -alpha 0.33 -o 3
//               ex33 -m ../data/disc-nurbs.mesh -alpha 2.4 -o 3 -r 4
//               ex33 -m ../data/l-shape.mesh -alpha 0.33 -o 3 -r 4
//               ex33 -m ../data/l-shape.mesh -alpha 1.7 -o 3 -r 5
//
// Verification runs:
//    ex33 -m ../data/inline-segment.mesh -ver -alpha 1.7 -o 2 -r 2
//    ex33 -m ../data/inline-quad.mesh -ver -alpha 1.2 -o 2 -r 2
//    ex33 -m ../data/amr-quad.mesh -ver -alpha 2.6 -o 2 -r 2
//    ex33 -m ../data/inline-hex.mesh -ver -alpha 0.3 -o 2 -r 1
//
//  Note: The manufactured solution used in this problem is
//
//            u = ∏_{i=0}^{dim-1} sin(π x_i) ,
//
//        regardless of the value of alpha.
//
// Description:
//
//  In this example we solve the following fractional PDE with MFEM:
//
//    ( - Δ )^α u = f  in Ω,      u = 0  on ∂Ω,      0 < α,
//
//  To solve this FPDE, we apply the operator ( - Δ )^(-N), where the integer
//  N is given by floor(α). By doing so, we obtain
//
//    ( - Δ )^(α-N) u = ( - Δ )^(-N) f  in Ω,      u = 0  on ∂Ω,      0 < α.
//
//  We first compute the right hand side by solving the integer order PDE
//
//   ( - Δ )^N g = f  in Ω, g = ( - Δ )^k g = 0 on ∂Ω, k = 1,..,N-1
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
#include <string>

#include "ex33.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_SINGLE
   cout << "This example is not supported in single precision.\n\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int num_refs = 3;
   real_t alpha = 0.5;
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
                  "Use sinusoidal function (f) for manufactured "
                  "solution test.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Array<real_t> coeffs, poles;
   int progress_steps = 1;

   // 2. Compute the rational expansion coefficients that define the
   //    integer-order PDEs.
   const int power_of_laplace = (int)floor(alpha);
   real_t exponent_to_approximate = alpha - power_of_laplace;
   bool integer_order = false;
   // Check if alpha is an integer or not.
   if (abs(exponent_to_approximate) > 1e-12)
   {
      mfem::out << "Approximating the fractional exponent "
                << exponent_to_approximate
                << endl;
      ComputePartialFractionApproximation(exponent_to_approximate, coeffs,
                                          poles);

      // If the example is built without LAPACK, the exponent_to_approximate
      // might be modified by the function call above.
      alpha = exponent_to_approximate + power_of_laplace;
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
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of degrees of freedom: "
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
      real_t val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(M_PI*x(i));
      }
      return pow(x.Size()*pow(M_PI,2), alpha) * val;
   };
   FunctionCoefficient f(func);
   ConstantCoefficient one(1.0);
   GridFunction u(&fespace);
   GridFunction x(&fespace);
   GridFunction g(&fespace);
   u = 0.0;
   x = 0.0;
   g = 0.0;

   // 8. Prepare for visualization.
   char vishost[] = "localhost";
   int  visport   = 19916;

   // 9. Set up the linear form b(.) for integer-order PDE solves.
   LinearForm b(&fespace);
   if (verification)
   {
      // This statement is only relevant for the verification of the code. It
      // uses a different f such that an analytic solution is known and easy
      // to compare with the numerical one. The FPDE becomes:
      // (-Δ)^α u = (2\pi ^2)^α sin(\pi x) sin(\pi y) on [0,1]^2
      // -> u(x,y) = sin(\pi x) sin(\pi y)
      b.AddDomainIntegrator(new DomainLFIntegrator(f));
   }
   else
   {
      b.AddDomainIntegrator(new DomainLFIntegrator(one));
   }
   b.Assemble();

   // ------------------------------------------------------------------------
   // 10. Solve the PDE (-Δ)^N g = f, i.e. compute g = (-Δ)^{-1}^N f.
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

      // 10.3 Form the system of equations
      Vector B, X;
      OperatorPtr Op;
      k.FormLinearSystem(ess_tdof_list, g, b, Op, X, B);
      GSSmoother M((SparseMatrix&)(*Op));

      mfem::out << "\nComputing (-Δ) ^ -" << power_of_laplace
                << " ( f ) " << endl;
      for (int i = 0; i < power_of_laplace; i++)
      {
         // 10.4 Solve the linear system Op X = B (N times).
         PCG(*Op, M, B, X, 3, 300, 1e-12, 0.0);

         // 10.5 Visualize the solution g of -Δ ^ N g = f in the last step
         if (i == power_of_laplace - 1)
         {
            // Needed for visualization and solution verification.
            k.RecoverFEMSolution(X, b, g);
            if (integer_order && verification)
            {
               // For an integer order PDE, g is also our solution u.
               u+=g;
            }
            if (visualization)
            {
               socketstream fout;
               ostringstream oss_f;
               fout.open(vishost, visport);
               fout.precision(8);
               oss_f.str(""); oss_f.clear();
               oss_f << "Step " << progress_steps++ << ": Solution of PDE -Δ ^ "
                     << power_of_laplace
                     << " g = f";
               fout << "solution\n" << mesh << g
                    << "window_title '" << oss_f.str() << "'" << flush;
            }
         }

         // 10.6 Prepare for next iteration (primal / dual space)
         mass.Mult(X, B);
         X.SetSubVectorComplement(ess_tdof_list,0.0);
      }

      // 10.7 Extract solution for the next step. The b now corresponds to the
      //      function g in the PDE.
      const SparseMatrix * R = fespace.GetRestrictionMatrix();
      if (R)
      {
         R->MultTranspose(B,b);
      }
      else
      {
         b = B;
      }
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
                   << " u = " << coeffs[i] << " g " << endl;


         // 11.1 Reset GridFunction for integer-order PDE solve.
         x = 0.0;

         // 11.2 Set up the bilinear form a(.,.) for integer-order PDE solve.
         BilinearForm a(&fespace);
         a.AddDomainIntegrator(new DiffusionIntegrator(one));
         ConstantCoefficient d_i(-poles[i]);
         a.AddDomainIntegrator(new MassIntegrator(d_i));
         a.Assemble();

         // 11.3 Assemble the bilinear form and the corresponding linear system.
         OperatorPtr A;
         Vector B, X;
         a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

         // 11.4 Solve the linear system A X = B.
         GSSmoother M((SparseMatrix&)(*A));

         PCG(*A, M, B, X, 3, 300, 1e-12, 0.0);

         // 11.5 Recover the solution as a finite element grid function.
         a.RecoverFEMSolution(X, b, x);

         // 11.6 Accumulate integer-order PDE solutions.
         x *= coeffs[i];
         u += x;

         // 11.7 Send fractional PDE solution to a GLVis server.
         if (visualization)
         {
            oss_x.str(""); oss_x.clear();
            oss_x << "Step " << progress_steps
                  << ": Solution of PDE -Δ u + " << -poles[i]
                  << " u = " << coeffs[i] << " g";
            xout << "solution\n" << mesh << x
                 << "window_title '" << oss_x.str() << "'" << flush;

            oss_u.str(""); oss_u.clear();
            oss_u << "Step " << progress_steps + 1
                  << ": Solution of fractional PDE (-Δ)^" << alpha
                  << " u = f";
            uout << "solution\n" << mesh << u
                 << "window_title '" << oss_u.str() << "'"
                 << flush;
         }
      }
   }

   // ------------------------------------------------------------------------
   // 12. (optional) Verify the solution.
   // ------------------------------------------------------------------------
   if (verification)
   {
      auto solution = [] (const Vector &x)
      {
         real_t val = 1.0;
         for (int i=0; i<x.Size(); i++)
         {
            val *= sin(M_PI*x(i));
         }
         return val;
      };
      FunctionCoefficient sol(solution);
      real_t l2_error = u.ComputeL2Error(sol);

      string manufactured_solution,expected_mesh;
      switch (dim)
      {
         case 1:
            manufactured_solution = "sin(π x)";
            expected_mesh = "inline_segment.mesh";
            break;
         case 2:
            manufactured_solution = "sin(π x) sin(π y)";
            expected_mesh = "inline_quad.mesh";
            break;
         default:
            manufactured_solution = "sin(π x) sin(π y) sin(π z)";
            expected_mesh = "inline_hex.mesh";
            break;
      }

      mfem::out << "\n" << string(80,'=')
                << "\n\nSolution Verification in "<< dim << "D \n\n"
                << "Manufactured solution : " << manufactured_solution << "\n"
                << "Expected mesh         : " << expected_mesh <<"\n"
                << "Your mesh             : " << mesh_file << "\n"
                << "L2 error              : " << l2_error << "\n\n"
                << string(80,'=') << endl;
   }

   return 0;
}
