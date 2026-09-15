//                             MFEM Example 43 HX
//
// Compile with: make ex43_hx
//
// Description: Solve a div-div plus mass problem for a symmetric matrix
// field using the lowest-order 2D Johnson--Mercier element and the auxiliary
// space preconditioner
//
//                 R + Pi B_1 Pi^t + J B_2 J^t.
//
// Here R is a vertex-patch Schwarz smoother, Pi is the canonical interpolant
// from continuous piecewise-linear symmetric matrices, B_1 is the inverse of
// the matrix H1 operator, J is the Airy map from the HCT space, and B_2 is the
// inverse of the HCT biharmonic operator.

#include "ex43.hpp"
#include <iostream>
#include <memory>

using namespace mfem;
using namespace std;

class HXPreconditioner : public Solver
{
private:
   H1_FECollection h1_fec;
   FiniteElementSpace h1_fespace;
   HCT_FECollection hct_fec;
   FiniteElementSpace hct_fespace;
   MatrixConstantCoefficient matrix_h1_coefficient;
   BilinearForm matrix_h1_form;
   BilinearForm biharmonic_form;
   DiscreteLinearOperator pi;
   DiscreteLinearOperator airy;
   VertexPatchSmoother patch_smoother;
   Array<int> hct_gauge_dofs;
   unique_ptr<Solver> matrix_h1_inverse;
   unique_ptr<Solver> biharmonic_inverse;
   mutable Vector matrix_h1_rhs, matrix_h1_solution;
   mutable Vector biharmonic_rhs, biharmonic_solution;
   mutable Vector auxiliary_correction;

   static DenseMatrix MatrixH1Weight()
   {
      DenseMatrix weight(3);
      weight = 0.0;
      weight(0,0) = 1.0;
      weight(1,1) = 2.0;
      weight(2,2) = 1.0;
      return weight;
   }

   static unique_ptr<Solver> MakeInverse(SparseMatrix &op)
   {
#ifdef MFEM_USE_SUITESPARSE
      return unique_ptr<Solver>(new UMFPackSolver(op));
#else
      CGSolver *inverse = new CGSolver;
      inverse->iterative_mode = false;
      inverse->SetOperator(op);
      inverse->SetRelTol(1e-12);
      inverse->SetAbsTol(0.0);
      inverse->SetMaxIter(20000);
      inverse->SetPrintLevel(0);
      return unique_ptr<Solver>(inverse);
#endif
   }

public:
   HXPreconditioner(const SparseMatrix &op, FiniteElementSpace &jm_fespace)
      : Solver(op.Height()),
        h1_fec(1, 2),
        h1_fespace(jm_fespace.GetMesh(), &h1_fec, 3, Ordering::byVDIM),
        hct_fespace(jm_fespace.GetMesh(), &hct_fec),
        matrix_h1_coefficient(MatrixH1Weight()),
        matrix_h1_form(&h1_fespace),
        biharmonic_form(&hct_fespace),
        pi(&h1_fespace, &jm_fespace),
        airy(&hct_fespace, &jm_fespace),
        patch_smoother(op, jm_fespace)
   {
      MFEM_VERIFY(jm_fespace.GetTrueVSize() == jm_fespace.GetVSize(),
                  "HXPreconditioner currently requires a conforming mesh");

      pi.AddDomainInterpolator(new IdentityInterpolator);
      pi.Assemble();
      pi.Finalize();

      airy.AddDomainInterpolator(new AiryInterpolator);
      airy.Assemble();
      airy.Finalize();

      matrix_h1_form.AddDomainIntegrator(
         new VectorMassIntegrator(matrix_h1_coefficient));
      matrix_h1_form.AddDomainIntegrator(
         new VectorDiffusionIntegrator(matrix_h1_coefficient));
      matrix_h1_form.Assemble();
      matrix_h1_form.Finalize();
      matrix_h1_inverse = MakeInverse(matrix_h1_form.SpMat());

      biharmonic_form.AddDomainIntegrator(new HessianIntegrator);
      biharmonic_form.Assemble();
      biharmonic_form.Finalize();

      // Hessians annihilate affine functions. Fix value and both first
      // derivatives at one vertex to select a representative of HCT/P1.
      hct_fespace.GetVertexDofs(0, hct_gauge_dofs);
      MFEM_VERIFY(hct_gauge_dofs.Size() == 3,
                  "expected three HCT degrees of freedom at a vertex");
      for (int i = 0; i < hct_gauge_dofs.Size(); i++)
      {
         hct_gauge_dofs[i] = UnsignIndex(hct_gauge_dofs[i]);
         biharmonic_form.SpMat().EliminateRowCol(hct_gauge_dofs[i]);
      }
      biharmonic_inverse = MakeInverse(biharmonic_form.SpMat());
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      patch_smoother.Mult(x, y);

      matrix_h1_rhs.SetSize(pi.Width());
      pi.MultTranspose(x, matrix_h1_rhs);
      matrix_h1_solution.SetSize(matrix_h1_rhs.Size());
      matrix_h1_inverse->Mult(matrix_h1_rhs, matrix_h1_solution);
      auxiliary_correction.SetSize(pi.Height());
      pi.Mult(matrix_h1_solution, auxiliary_correction);
      y += auxiliary_correction;

      biharmonic_rhs.SetSize(airy.Width());
      airy.MultTranspose(x, biharmonic_rhs);
      for (int i = 0; i < hct_gauge_dofs.Size(); i++)
      {
         biharmonic_rhs(hct_gauge_dofs[i]) = 0.0;
      }
      biharmonic_solution.SetSize(biharmonic_rhs.Size());
      biharmonic_inverse->Mult(biharmonic_rhs, biharmonic_solution);
      auxiliary_correction.SetSize(airy.Height());
      airy.Mult(biharmonic_solution, auxiliary_correction);
      y += auxiliary_correction;
   }

   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
   void SetOperator(const Operator &) override
   { MFEM_ABORT("HXPreconditioner does not support SetOperator"); }
};

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/ref-triangle.mesh";
   int refinements = 2;
   bool visualization = false;
   bool random_rhs = false;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Input triangle mesh.");
   args.AddOption(&refinements, "-r", "--refinements",
                  "Number of uniform refinements.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable or disable visualization (accepted for consistency).");
   args.AddOption(&random_rhs, "-random-rhs", "--random-rhs",
                  "-constant-rhs", "--constant-rhs",
                  "Use a reproducible random algebraic right-hand side.");
   args.ParseCheck();

   Mesh mesh(mesh_file);
   MFEM_VERIFY(mesh.Dimension() == 2, "");
   for (int level = 0; level < refinements; level++)
   {
      mesh.UniformRefinement();
   }

   JohnsonMercierFECollection fec;
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "\nJohnson--Mercier space: " << fespace.GetNE()
        << " elements, " << fespace.GetTrueVSize() << " unknowns\n";

   DenseMatrix identity(2);
   identity = 0.0;
   identity(0,0) = identity(1,1) = 1.0;
   MatrixConstantCoefficient rhs(identity);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new MatrixFEDomainLFIntegrator(rhs));
   b.Assemble();
   GridFunction solution(&fespace);
   solution = 0.0;

   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new MatrixDivDivIntegrator);
   a.AddDomainIntegrator(new MatrixFEMassIntegrator);
   a.Assemble();
   Array<int> ess_tdof_list;
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, solution, b, A, X, B);
   if (random_rhs) { B.Randomize(1); }

   HXPreconditioner hx(A, fespace);
   CGSolver solver;
   solver.SetOperator(A);
   solver.SetPreconditioner(hx);
   solver.SetRelTol(1e-10);
   solver.SetAbsTol(0.0);
   solver.SetMaxIter(500);
   solver.SetPrintLevel(1);
   solver.Mult(B, X);
   a.RecoverFEMSolution(X, b, solution);
   cout << "PCG iterations: " << solver.GetNumIterations() << '\n'
        << "Final residual norm: " << solver.GetFinalNorm() << '\n';
   if (!solver.GetConverged())
   {
      cerr << "PCG did not converge.\n";
      return 3;
   }
   return 0;
}
