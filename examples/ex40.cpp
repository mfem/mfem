//                                MFEM Example 40
//
// Compile with: make ex40
//
// Sample runs: ex40 -o 2
//              ex40 -o 2 -r 4
//
// Description: This example code demonstrates to how to use MFEM to solve
//              the Monge–Ampère equation
//
//                      det(∇²u) = f in Ω,   u = 0 on ∂Ω.
//
//              This example highlights the ExponentialMatrixCoefficient
//              class, which is used in Newton's method to solve the
//              variational formulation
//
//               Find M ∈ H₀(div,Ω)ⁿ and u ∈ H₀¹(Ω) such that
//                 (exp(M), N) + (∇u, ∇⋅N) = 0           ∀ N ∈ H₀(div,Ω)ⁿ
//                 (tr(M), v)              = (ln f, v)   ∀ v ∈ H₀¹(Ω)
//
//              where n is the spatial dimension of the domain Ω.
//
//
//              The linearized subproblem is
//
//               Find δM ∈ H₀(div,Ω)ⁿ and u ∈ H₀¹(Ω) such that
//                 (exp(M) δM, N) + (∇u, ∇⋅N) = -(exp(M), N)        ∀ N ∈ H₀(div,Ω)ⁿ
//                 (tr(δM), v)                = (ln f - tr(M), v)   ∀ v ∈ H₀¹(Ω)
//
//
//              (exp(M) δM, N)    :::   VectorFEMassIntegrator
//              (∇u, ∇⋅N)         :::   MixedGradDivIntegrator
//              (tr(δM), v)       :::   MixedDotProductIntegrator
//              (exp(M), N)       :::   VectorFEDomainLFIntegrator
//              (ln f - tr(M), v) :::   DomainLFIntegrator
//
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t exact_solution(const Vector &pt);
void exact_solution_gradient(const Vector &pt, Vector &grad);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 3;
   int max_it = 10;
   int ref_levels = 1;
   real_t tol = 1e-5;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&tol, "-tol", "--tol",
                  "Stopping criteria based on the difference between"
                  "successive solution updates");
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

   // 2. Read the mesh from the mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   if (dim != 2)
   {
      MFEM_ABORT("Example 40 currently only supports 2D problems")
   }

   // 3. Postprocess the mesh.
   // 3A. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 3B. Interpolate the geometry after refinement to control geometry error.
   // NOTE: Minimum second-order interpolation is used to improve the accuracy.
   int curvature_order = max(order,2);
   mesh.SetCurvature(curvature_order);

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection H1fec(order, dim);
   FiniteElementSpace H1fes(&mesh, &H1fec);

   RT_FECollection RTfec(order, dim);
   FiniteElementSpace RTfes(&mesh, &RTfec);

   cout << "Number of H¹ degrees of freedom: "
        << H1fes.GetTrueVSize() << endl;
   cout << "Number of H(div) degrees of freedom: "
        << RTfes.GetTrueVSize() * dim << endl;

   Array<int> offsets(4);
   offsets[0] = 0;
   offsets[1] = RTfes.GetVSize();
   offsets[2] = RTfes.GetVSize();
   offsets[3] = H1fes.GetVSize();
   offsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   // 5. Determine the list of true (i.e., conforming) essential boundary dofs.
   Array<int> ess_bdr;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
   }

   // 6. Define constants to be used later.
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   Vector V1(2), V2(2);
   V1(0) = 1.0; V1(1) = 0.0;
   V2(0) = 0.0; V2(1) = 1.0;
   VectorConstantCoefficient onezero(V1);
   VectorConstantCoefficient zeroone(V2);

   // 7. Define the solution vectors as finite element grid functions
   //    corresponding to the fespaces.
   GridFunction delta_M1_gf, delta_M2_gf, u_gf;

   delta_M1_gf.MakeRef(&RTfes,x,offsets[0]);
   delta_M2_gf.MakeRef(&RTfes,x,offsets[1]);
   u_gf.MakeRef(&H1fes,x,offsets[2]);

   GridFunction M1_gf(&RTfes);
   GridFunction M2_gf(&RTfes);
   GridFunction u_prvs_gf(&H1fes);

   // 8. Define the function coefficients for the solution and use them to
   //    initialize the initial guess
   FunctionCoefficient exact_coef(exact_solution);
   VectorFunctionCoefficient exact_grad_coef(dim,exact_solution_gradient);
   ConstantCoefficient ln_rhs_coef(0.0);
   u_gf.ProjectCoefficient(zero);
   u_prvs_gf = u_gf;


   // 9. Initialize the Lie algebra variable M
   M1_gf = 0.0;
   M2_gf = 0.0;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost,visport);
      sol_sock.precision(8);
   }

   // 10. Iterate
   int k;
   for (k = 0; k < max_it; k++)
   {
      mfem::out << "\nITERATION " << k+1 << endl;

      LinearForm b0,b1,b2;
      b0.Update(&RTfes,rhs.GetBlock(0),0);
      b1.Update(&RTfes,rhs.GetBlock(1),0);
      b2.Update(&H1fes,rhs.GetBlock(2),0);

      VectorGridFunctionCoefficient M1_coeff(&M1_gf);
      VectorGridFunctionCoefficient M2_coeff(&M2_gf);

      MatrixArrayVectorCoefficient exp_M(dim);
      exp_M.Set(0, &M1_coeff);
      exp_M.Set(1, &M2_coeff);

      MatrixVectorProductCoefficient exp_M1(exp_M, onezero);
      MatrixVectorProductCoefficient exp_M2(exp_M, zeroone);
      InnerProductCoefficient exp_M11(exp_M1, onezero);
      InnerProductCoefficient exp_M12(exp_M1, zeroone);
      InnerProductCoefficient exp_M21(exp_M2, onezero);
      InnerProductCoefficient exp_M22(exp_M2, zeroone);

      ScalarVectorProductCoefficient neg_exp_M1(-1.0, exp_M1);
      b0.AddDomainIntegrator(new VectorFEDomainLFIntegrator(neg_exp_M1));
      b0.Assemble();

      ScalarVectorProductCoefficient neg_exp_M2(-1.0, exp_M2);
      b1.AddDomainIntegrator(new VectorFEDomainLFIntegrator(neg_exp_M2));
      b1.Assemble();

      InnerProductCoefficient M11(M1_coeff, onezero);
      InnerProductCoefficient M22(M2_coeff, zeroone);
      SumCoefficient trace_M(M11, M22);
      SumCoefficient rhs2(ln_rhs_coef, trace_M, 1.0, -1.0);
      b2.AddDomainIntegrator(new DomainLFIntegrator(rhs2));
      b2.Assemble();

      BilinearForm a00(&RTfes);
      a00.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
      a00.AddDomainIntegrator(new VectorFEMassIntegrator(exp_M11));
      a00.Assemble();
      a00.EliminateEssentialBC(ess_bdr,x.GetBlock(0),rhs.GetBlock(0),
                               mfem::Operator::DIAG_ONE);
      a00.Finalize();
      SparseMatrix &A00 = a00.SpMat();

      BilinearForm a01(&RTfes);
      a01.AddDomainIntegrator(new VectorFEMassIntegrator(exp_M12));
      a01.Assemble();
      SparseMatrix &A01 = a01.SpMat();

      MixedBilinearForm a02(&RTfes,&H1fes);
      a02.AddDomainIntegrator(new TransposeIntegrator(new MixedGradDivIntegrator(
                                                         onezero)));
      a02.Assemble();
      SparseMatrix &A02 = a02.SpMat();

      BilinearForm a10(&RTfes);
      a10.AddDomainIntegrator(new VectorFEMassIntegrator(exp_M21));
      a10.Assemble();
      SparseMatrix &A10 = a10.SpMat();

      BilinearForm a11(&RTfes);
      a11.AddDomainIntegrator(new VectorFEMassIntegrator(exp_M22));
      a11.Assemble();
      SparseMatrix &A11 = a11.SpMat();

      MixedBilinearForm a12(&RTfes,&H1fes);
      a12.AddDomainIntegrator(new TransposeIntegrator(new MixedGradDivIntegrator(
                                                         zeroone)));
      a12.Assemble();
      SparseMatrix &A12 = a12.SpMat();

      MixedBilinearForm a20(&H1fes,&RTfes);
      a20.AddDomainIntegrator(new TransposeIntegrator(new MixedDotProductIntegrator(
                                                         onezero)));
      a20.Assemble();
      SparseMatrix &A20 = a20.SpMat();

      MixedBilinearForm a21(&H1fes,&RTfes);
      a21.AddDomainIntegrator(new TransposeIntegrator(new MixedDotProductIntegrator(
                                                         zeroone)));
      a21.Assemble();
      SparseMatrix &A21 = a21.SpMat();

      BilinearForm a22(&H1fes);
      // TODO (zero)
      a22.Finalize();
      SparseMatrix &A22 = a22.SpMat();

      BlockOperator A(offsets);
      A.SetBlock(0,0,&A00);
      A.SetBlock(0,1,&A01);
      A.SetBlock(0,2,&A02);
      A.SetBlock(1,0,&A10);
      A.SetBlock(1,1,&A11);
      A.SetBlock(1,2,&A12);
      A.SetBlock(2,0,&A20);
      A.SetBlock(2,1,&A21);
      A.SetBlock(2,2,&A22);

      BlockDiagonalPreconditioner prec(offsets);
      prec.SetDiagonalBlock(0,new GSSmoother(A00));
      prec.SetDiagonalBlock(1,new GSSmoother(A11));
      prec.SetDiagonalBlock(1,new GSSmoother(A22));
      prec.owns_blocks = 1;

      GMRES(A,prec,rhs,x,0,10000,500,1e-12,0.0);

      delta_M1_gf.MakeRef(&RTfes, x.GetBlock(0), 0);
      delta_M2_gf.MakeRef(&RTfes, x.GetBlock(1), 0);
      u_gf.MakeRef(&H1fes, x.GetBlock(2), 0);

      u_prvs_gf -= u_gf;
      real_t Newton_update_size = u_prvs_gf.ComputeL2Error(zero);
      u_prvs_gf = u_gf;

      M1_gf += delta_M1_gf;
      M2_gf += delta_M2_gf;

      if (visualization)
      {
         sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
                  << flush;
         mfem::out << "Increment (|| uₕ - uₕ_prvs||) = " << Newton_update_size <<
                   endl;
      }

      if (Newton_update_size < tol || k == max_it-1)
      {
         break;
      }

      real_t H1_error = u_gf.ComputeH1Error(&exact_coef,&exact_grad_coef);
      mfem::out << "H1-error  (|| u - uₕᵏ||)       = " << H1_error << endl;

   }

   mfem::out << "\n Total iterations: " << k+1
             << "\n Total dofs:       " << RTfes.GetTrueVSize() * 2 + H1fes.GetTrueVSize()
             << endl;

   // 11. Exact solution.
   if (visualization)
   {
      socketstream err_sock(vishost, visport);
      err_sock.precision(8);

      GridFunction error_gf(&H1fes);
      error_gf.ProjectCoefficient(exact_coef);
      error_gf -= u_gf;

      err_sock << "solution\n" << mesh << error_gf << "window_title 'Error'"  <<
               flush;
   }

   return 0;
}

real_t exact_solution(const Vector &pt)
{
   real_t x = pt(0), y = pt(1);
   return (x*x + y*y) / 2.0;
}

void exact_solution_gradient(const Vector &pt, Vector &grad)
{
   real_t x = pt(0), y = pt(1);

   grad(0) = x;
   grad(1) = y;
}
