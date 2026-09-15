//                             MFEM Example 43 HX
//
// Compile with: make ex43_hx
//
// Description: Solve a div-div plus mass problem for a symmetric matrix
// field using lowest-order 2D Johnson--Mercier, Arnold--Winther, or Hu--Zhang elements
// and the auxiliary space preconditioner
//
//                 R + Pi B_1 Pi^t + J B_2 J^t.
//
// Here R is a vertex-patch, Jacobi, or symmetric Gauss-Seidel smoother.
// Pi is the canonical interpolant
// from continuous piecewise-linear symmetric matrices, B_1 is the inverse of
// the matrix H1 operator, J is the Airy map from the HCT, Argyris, or Bell space, and B_2 is the
// inverse of the corresponding biharmonic operator.

#include "ex43.hpp"
#include <cstring>
#include <iostream>
#include <memory>

using namespace mfem;
using namespace std;

enum class HXSmoother { VERTEX_PATCH, JACOBI, GAUSS_SEIDEL };

class HXPreconditioner : public Solver
{
private:
   H1_FECollection h1_fec;
   FiniteElementSpace h1_fespace;
   unique_ptr<FiniteElementCollection> potential_fec;
   FiniteElementSpace potential_fespace;
   MatrixConstantCoefficient matrix_h1_coefficient;
   BilinearForm matrix_h1_form;
   BilinearForm biharmonic_form;
   DiscreteLinearOperator pi;
   DiscreteLinearOperator airy;
   unique_ptr<Solver> smoother;
   Array<int> potential_gauge_dofs;
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
   HXPreconditioner(const SparseMatrix &op, FiniteElementSpace &stress_fespace,
                    const char *potential_name, HXSmoother smoother_type)
      : Solver(op.Height()),
        h1_fec(1, 2),
        h1_fespace(stress_fespace.GetMesh(), &h1_fec, 3, Ordering::byVDIM),
        potential_fec(FiniteElementCollection::New(
                         potential_name)),
        potential_fespace(stress_fespace.GetMesh(), potential_fec.get()),
        matrix_h1_coefficient(MatrixH1Weight()),
        matrix_h1_form(&h1_fespace),
        biharmonic_form(&potential_fespace),
        pi(&h1_fespace, &stress_fespace),
        airy(&potential_fespace, &stress_fespace)
   {
      MFEM_VERIFY(stress_fespace.GetTrueVSize() == stress_fespace.GetVSize(),
                  "HXPreconditioner currently requires a conforming mesh");

      switch (smoother_type)
      {
         case HXSmoother::VERTEX_PATCH:
            smoother.reset(new VertexPatchSmoother(op, stress_fespace));
            break;
         case HXSmoother::JACOBI:
            smoother.reset(new DSmoother(op, DSmoother::JACOBI));
            break;
         case HXSmoother::GAUSS_SEIDEL:
            // Forward/backward sweeps keep the HX preconditioner symmetric
            // for the outer conjugate-gradient solver.
            smoother.reset(new GSSmoother(op, GSSmoother::SYMMETRIC));
            break;
      }

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
      // derivatives at one vertex to select a representative modulo P1.
      potential_fespace.GetVertexDofs(0, potential_gauge_dofs);
      // Argyris and Bell also have three second derivatives at each vertex; those
      // are not in the affine kernel and must remain unconstrained.
      potential_gauge_dofs.SetSize(3);
      for (int i = 0; i < potential_gauge_dofs.Size(); i++)
      {
         potential_gauge_dofs[i] = UnsignIndex(potential_gauge_dofs[i]);
         biharmonic_form.SpMat().EliminateRowCol(potential_gauge_dofs[i]);
      }
      biharmonic_inverse = MakeInverse(biharmonic_form.SpMat());
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      smoother->Mult(x, y);

      matrix_h1_rhs.SetSize(pi.Width());
      pi.MultTranspose(x, matrix_h1_rhs);
      matrix_h1_solution.SetSize(matrix_h1_rhs.Size());
      matrix_h1_inverse->Mult(matrix_h1_rhs, matrix_h1_solution);
      auxiliary_correction.SetSize(pi.Height());
      pi.Mult(matrix_h1_solution, auxiliary_correction);
      y += auxiliary_correction;

      biharmonic_rhs.SetSize(airy.Width());
      airy.MultTranspose(x, biharmonic_rhs);
      for (int i = 0; i < potential_gauge_dofs.Size(); i++)
      {
         biharmonic_rhs(potential_gauge_dofs[i]) = 0.0;
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
   const char *smoother_name = "vertex-patch";
   bool visualization = false;
   bool random_rhs = false;
   bool use_aw = false;
   bool use_hz = false;
   bool use_hzzz = false;
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
   args.AddOption(&use_aw, "-aw", "--arnold-winther", "-jm", "--johnson-mercier",
                  "Use Arnold--Winther or Johnson--Mercier elements.");
   args.AddOption(&use_hz, "-hz", "--hu-zhang", "-no-hz", "--no-hu-zhang",
                  "Use cubic Hu--Zhang stress elements (overrides -aw/-jm).");
   args.AddOption(&smoother_name, "-s", "--smoother",
                  "HX smoother: vertex-patch (default), jacobi, or gauss-seidel "
                  "(symmetric forward/backward sweeps).");
   args.AddOption(&use_hzzz, "-hzzz", "--huang-zhang-zhou-zhu",
                  "-no-hzzz", "--no-huang-zhang-zhou-zhu",
                  "Use 21-DOF HZZZ stress elements (overrides -hz/-aw/-jm).");
   args.ParseCheck();

   HXSmoother smoother_type;
   if (!strcmp(smoother_name, "vertex-patch"))
   {
      smoother_type = HXSmoother::VERTEX_PATCH;
   }
   else if (!strcmp(smoother_name, "jacobi"))
   {
      smoother_type = HXSmoother::JACOBI;
   }
   else if (!strcmp(smoother_name, "gauss-seidel"))
   {
      smoother_type = HXSmoother::GAUSS_SEIDEL;
   }
   else
   {
      cerr << "Unknown HX smoother '" << smoother_name
           << "'. Choose vertex-patch, jacobi, or gauss-seidel.\n";
      return 1;
   }

   Mesh mesh(mesh_file);
   MFEM_VERIFY(mesh.Dimension() == 2, "");
   for (int level = 0; level < refinements; level++)
   {
      mesh.UniformRefinement();
   }

   const char *fec_name = use_hzzz ? "HZZZ_2D_P3" :
                          (use_hz ? "HZ_2D_P3" :
                           (use_aw ? "AW_2D_P3" : "JM_2D_P1"));
   unique_ptr<FiniteElementCollection> fec(FiniteElementCollection::New(fec_name));
   FiniteElementSpace fespace(&mesh, fec.get());
   cout << "\n" << fec->Name() << " space: " << fespace.GetNE()
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

   const char *potential_name = use_hzzz ? "Bell_2D_P5" :
                                (use_aw || use_hz ? "Argyris_2D_P5" : "HCT_2D_P3");
   HXPreconditioner hx(A, fespace, potential_name, smoother_type);
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
