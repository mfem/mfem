//                                MFEM Example 43
//
// Compile with: make ex43
//
// Description: Solve a div-div plus mass problem for a symmetric matrix
// field using the lowest-order 2D Johnson--Mercier element and geometric
// multigrid with vertex-patch Schwarz smoothers.

#include "mfem.hpp"
#include <iostream>

using namespace mfem;
using namespace std;

class VertexPatchSmoother : public Solver
{
private:
   Array<Array<int> *> patch_dofs;
   Array<DenseMatrix *> patch_inverses;
   mutable Vector patch_rhs, patch_solution;

public:
   VertexPatchSmoother(const SparseMatrix &op, FiniteElementSpace &fespace)
      : Solver(op.Height())
   {
      Mesh *mesh = fespace.GetMesh();
      Table *vertex_to_element = mesh->GetVertexToElementTable();
      Array<int> marker(fespace.GetVSize());
      marker = -1;
      Array<int> dofs_on_entity, element_edges, edge_orientations;
      Array<int> edge_vertices;
      for (int vertex = 0; vertex < mesh->GetNV(); vertex++)
      {
         Array<int> *dofs = new Array<int>;
         const int *incident_elements = vertex_to_element->GetRow(vertex);
         const int num_incident_elements = vertex_to_element->RowSize(vertex);
         for (int i = 0; i < num_incident_elements; i++)
         {
            const int element = incident_elements[i];
            fespace.GetElementInteriorDofs(element, dofs_on_entity);
            for (int j = 0; j < dofs_on_entity.Size(); j++)
            {
               const int dof = UnsignIndex(dofs_on_entity[j]);
               if (marker[dof] != vertex) { marker[dof] = vertex; dofs->Append(dof); }
            }
            mesh->GetElementEdges(element, element_edges, edge_orientations);
            for (int j = 0; j < element_edges.Size(); j++)
            {
               const int edge = element_edges[j];
               mesh->GetEdgeVertices(edge, edge_vertices);
               if (edge_vertices.Find(vertex) < 0) { continue; }
               fespace.GetEdgeDofs(edge, dofs_on_entity);
               for (int k = 0; k < dofs_on_entity.Size(); k++)
               {
                  const int dof = UnsignIndex(dofs_on_entity[k]);
                  if (marker[dof] != vertex) { marker[dof] = vertex; dofs->Append(dof); }
               }
            }
         }
         DenseMatrix patch_matrix(dofs->Size());
         op.GetSubMatrix(*dofs, *dofs, patch_matrix);
         DenseMatrix *patch_inverse = new DenseMatrix;
         DenseMatrixInverse inverse(patch_matrix, true);
         inverse.GetInverseMatrix(*patch_inverse);
         patch_dofs.Append(dofs);
         patch_inverses.Append(patch_inverse);
      }
      delete vertex_to_element;
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      y = 0.0;
      for (int patch = 0; patch < patch_dofs.Size(); patch++)
      {
         const Array<int> &dofs = *patch_dofs[patch];
         x.GetSubVector(dofs, patch_rhs);
         patch_solution.SetSize(dofs.Size());
         patch_inverses[patch]->Mult(patch_rhs, patch_solution);
         y.AddElementVector(dofs, patch_solution);
      }
      y *= 0.33;
   }

   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
   void SetOperator(const Operator &) override
   { MFEM_ABORT("VertexPatchSmoother does not support SetOperator"); }
   ~VertexPatchSmoother() override
   {
      for (int patch = 0; patch < patch_dofs.Size(); patch++)
      { delete patch_dofs[patch]; delete patch_inverses[patch]; }
   }
};

class JohnsonMercierMultigrid : public GeometricMultigrid
{
   std::unique_ptr<Solver> coarse_prec;
public:
   JohnsonMercierMultigrid(FiniteElementSpaceHierarchy &fespaces,
                           const Array<int> &ess_bdr,
                           bool exact_smoother)
      : GeometricMultigrid(fespaces, ess_bdr)
   {
      const int num_levels = fespaces.GetNumLevels();
      for (int level = 0; level < num_levels; level++)
      {
         FiniteElementSpace &fespace = fespaces.GetFESpaceAtLevel(level);
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
         if (level == 0 || (exact_smoother && level + 1 < num_levels))
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
   bool exact_smoother = false;
   bool patch_only = false;
   bool random_rhs = false;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Input triangle mesh.");
   args.AddOption(&geometric_refinements, "-r", "--refinements",
                  "Number of uniform geometric refinements.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable or disable visualization (accepted for consistency).");
   args.AddOption(&exact_smoother, "-exact-smoother", "--exact-smoother",
                  "-patch-smoother", "--patch-smoother",
                  "Use exact solves on all multigrid levels.");
   args.AddOption(&patch_only, "-patch-only", "--patch-only",
                  "-multigrid", "--multigrid",
                  "Use the vertex-patch smoother directly as a preconditioner.");
   args.AddOption(&random_rhs, "-random-rhs", "--random-rhs",
                  "-constant-rhs", "--constant-rhs",
                  "Use a reproducible random algebraic right-hand side.");
   args.ParseCheck();

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   MFEM_VERIFY(mesh->Dimension() == 2, "");

   JohnsonMercierFECollection fec;
   FiniteElementSpace *coarse_fespace = new FiniteElementSpace(mesh, &fec);
   FiniteElementSpaceHierarchy fespaces(mesh, coarse_fespace, true, true);
   for (int level = 0; level < geometric_refinements; level++)
   { fespaces.AddUniformlyRefinedLevel(); }

   cout << "\nGeometric multigrid hierarchy:\n";
   for (int level = 0; level < fespaces.GetNumLevels(); level++)
   {
      const FiniteElementSpace &fespace = fespaces.GetFESpaceAtLevel(level);
      cout << "  level " << level << ": " << fespace.GetNE()
           << " elements, " << fespace.GetTrueVSize() << " unknowns\n";
   }
   FiniteElementSpace &fine_fespace = fespaces.GetFinestFESpace();
   DenseMatrix identity(2); identity = 0.0; identity(0,0) = identity(1,1) = 1.0;
   MatrixConstantCoefficient rhs(identity);
   LinearForm b(&fine_fespace);
   b.AddDomainIntegrator(new MatrixFEDomainLFIntegrator(rhs)); b.Assemble();
   GridFunction solution(&fine_fespace); solution = 0.0;
   Array<int> ess_bdr;
   JohnsonMercierMultigrid multigrid(fespaces, ess_bdr, exact_smoother);
   multigrid.SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);
   // multigrid.SetCycleType(Multigrid::CycleType::WCYCLE, 1, 1);
   OperatorHandle A; Vector B, X;
   multigrid.FormFineLinearSystem(solution, b, A, X, B);
   if (random_rhs) { B.Randomize(1); }
   CGSolver solver;
   solver.SetOperator(*A);
   VertexPatchSmoother patch_smoother(*A.As<SparseMatrix>(), fine_fespace);
   if (patch_only) { solver.SetPreconditioner(patch_smoother); }
   else { solver.SetPreconditioner(multigrid); }
   solver.SetRelTol(1e-10); solver.SetAbsTol(0.0); solver.SetMaxIter(500);
   solver.SetPrintLevel(0); solver.Mult(B, X);
   multigrid.RecoverFineFEMSolution(X, b, solution);
   cout << "PCG iterations: " << solver.GetNumIterations() << '\n'
        << "Final residual norm: " << solver.GetFinalNorm() << '\n';
   if (!solver.GetConverged()) { cerr << "PCG did not converge.\n"; return 3; }
   return 0;
}
