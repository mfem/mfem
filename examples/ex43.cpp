//                                MFEM Example 43
//
// Compile with: make ex43
//
// Description: Solve a div-div plus mass problem for a symmetric matrix
// field using lowest-order 2D Johnson--Mercier or Arnold--Winther elements
// and geometric multigrid with vertex-patch Schwarz smoothers.

#include "ex43.hpp"
#include <iostream>

using namespace mfem;
using namespace std;

class SymmetricMatrixMultigrid : public GeometricMultigrid
{
   std::unique_ptr<Solver> coarse_prec;
public:
   SymmetricMatrixMultigrid(FiniteElementSpaceHierarchy &fes_hierarchy,
                            const Array<int> &ess_bdr)
      : GeometricMultigrid(fes_hierarchy, ess_bdr)
   {
      const int num_levels = fes_hierarchy.GetNumLevels();
      for (int level = 0; level < num_levels; level++)
      {
         FiniteElementSpace &fespace = fes_hierarchy.GetFESpaceAtLevel(level);
         BilinearForm *form = new BilinearForm(&fespace);
         form->AddDomainIntegrator(new MatrixDivDivIntegrator);
         form->AddDomainIntegrator(new MatrixFEMassIntegrator);
         form->Assemble();
         bfs.Append(form);

         OperatorPtr level_operator;
         level_operator.SetType(Operator::ANY_TYPE);
         form->FormSystemMatrix(*essentialTrueDofs[level], level_operator);
         level_operator.SetOperatorOwner(false);

         SparseMatrix *sparse_operator =
            dynamic_cast<SparseMatrix *>(level_operator.Ptr());
         MFEM_VERIFY(sparse_operator,
                     "expected an assembled sparse level operator");

         Solver *level_solver;
         if (level == 0)
         {
#ifdef MFEM_USE_SUITESPARSE
            level_solver = new UMFPackSolver(*sparse_operator);
#else
            coarse_prec.reset(new GSSmoother(*sparse_operator));
            CGSolver *cg_solver = new CGSolver;
            cg_solver->SetOperator(*sparse_operator);
            cg_solver->SetPreconditioner(*coarse_prec);
            cg_solver->SetRelTol(1e-10);
            cg_solver->SetAbsTol(1e-10);
            cg_solver->SetMaxIter(500);
            level_solver = cg_solver;
#endif
         }
         else
         {
            level_solver = new VertexPatchSmoother(*sparse_operator, fespace);
         }
         // The assembled matrix is owned by the bilinear form in bfs.
         AddLevel(level_operator.Ptr(), level_solver, false, true);
      }
   }
};

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/ref-triangle.mesh";
   int geometric_refinements = 2;
   bool visualization = false;
   bool random_rhs = false;
   bool use_aw = false;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Input triangle mesh.");
   args.AddOption(&geometric_refinements, "-r", "--refinements",
                  "Number of uniform geometric refinements.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable or disable visualization (accepted for consistency).");
   args.AddOption(&random_rhs, "-random-rhs", "--random-rhs",
                  "-constant-rhs", "--constant-rhs",
                  "Use a reproducible random algebraic right-hand side.");
   args.AddOption(&use_aw, "-aw", "--arnold-winther", "-jm", "--johnson-mercier",
                  "Use Arnold--Winther or Johnson--Mercier elements.");
   args.ParseCheck();

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   MFEM_VERIFY(mesh->Dimension() == 2, "");

   unique_ptr<FiniteElementCollection> fec(FiniteElementCollection::New(
                                              use_aw ? "AW_2D_P3" : "JM_2D_P1"));
   FiniteElementSpace *coarse_fespace = new FiniteElementSpace(mesh, fec.get());
   FiniteElementSpaceHierarchy fes_hierarchy(mesh, coarse_fespace, true, true);
   for (int level = 0; level < geometric_refinements; level++)
   {
      fes_hierarchy.AddUniformlyRefinedLevel(
         1, Ordering::byVDIM, Operator::MFEM_SPARSEMAT);
   }

   cout << "\nGeometric multigrid hierarchy:\n";
   for (int level = 0; level < fes_hierarchy.GetNumLevels(); level++)
   {
      const FiniteElementSpace &fespace = fes_hierarchy.GetFESpaceAtLevel(level);
      cout << "  level " << level << ": " << fespace.GetNE()
           << " elements, " << fespace.GetTrueVSize() << " unknowns\n";
   }
   FiniteElementSpace &fine_fespace = fes_hierarchy.GetFinestFESpace();
   DenseMatrix identity(2); identity = 0.0; identity(0,0) = identity(1,1) = 1.0;
   MatrixConstantCoefficient rhs(identity);
   LinearForm b(&fine_fespace);
   b.AddDomainIntegrator(new MatrixFEDomainLFIntegrator(rhs)); b.Assemble();
   GridFunction solution(&fine_fespace);
   solution = 0.0;
   Array<int> ess_bdr;
   SymmetricMatrixMultigrid multigrid(fes_hierarchy, ess_bdr);
   multigrid.SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);
   OperatorHandle A; Vector B, X;
   multigrid.FormFineLinearSystem(solution, b, A, X, B);
   if (random_rhs) { B.Randomize(1); }

   CGSolver solver;
   solver.SetOperator(*A);
   solver.SetPreconditioner(multigrid);
   solver.SetRelTol(1e-10);
   solver.SetAbsTol(0.0);
   solver.SetMaxIter(500);
   solver.SetPrintLevel(1);
   solver.Mult(B, X);

   cout << "PCG iterations: " << solver.GetNumIterations() << '\n'
        << "Final residual norm: " << solver.GetFinalNorm() << '\n';
   if (!solver.GetConverged()) { cerr << "PCG did not converge.\n"; }

   return 0;
}
