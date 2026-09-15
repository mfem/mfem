//                                MFEM Example 43p
//
// Compile with: make ex43p
//
// Sample runs: mpirun -np 4 ex43p -r 2
//              mpirun -np 4 ex43p -m ../data/inline-tri.mesh -rs 1 -r 3 -random-rhs
//
// Description: Parallel div-div plus mass problem for a symmetric matrix
// field using lowest-order 2D Johnson--Mercier or Arnold--Winther elements
// and geometric multigrid with vertex-patch Schwarz smoothers. The coarse solver is PCG
// preconditioned by l1 hybrid Gauss-Seidel through HypreSmoother.

#include "ex43.hpp"
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

using namespace mfem;
using namespace std;

// Assign each complete vertex patch to the rank owning that vertex. A sparse
// gather operator imports all dofs needed by the rank's patches; its transpose
// sums patch corrections back onto the distributed true-dof vector.
class ParVertexPatchSmoother : public Solver
{
   unique_ptr<HypreParMatrix> gather;
   unique_ptr<VertexPatchSmoother> local_smoother;
   mutable Vector local_rhs, local_solution;

public:
   ParVertexPatchSmoother(const HypreParMatrix &op,
                          ParFiniteElementSpace &fespace)
      : Solver(op.Height())
   {
      ParMesh &mesh = *fespace.GetParMesh();
      const MPI_Comm comm = mesh.GetComm();
      const int nranks = fespace.GetNRanks();
      H1_FECollection vertex_fec(1, 2);
      ParFiniteElementSpace vertices(&mesh, &vertex_fec);
      Array<HYPRE_BigInt> vertex_offsets(nranks + 1);
      HYPRE_BigInt vertex_start = vertices.GetMyTDofOffset();
      MPI_Allgather(&vertex_start, 1, HYPRE_MPI_BIG_INT,
                    vertex_offsets.GetData(), 1, HYPRE_MPI_BIG_INT, comm);
      vertex_offsets[nranks] = vertices.GlobalTrueVSize();

      // Send (global vertex, global dof) pairs to the vertex owner. Include
      // element interiors and only those edges that contain the vertex,
      // matching the patches in the serial example.
      vector<vector<HYPRE_BigInt>> outgoing(nranks);
      unique_ptr<Table> vertex_elements(mesh.GetVertexToElementTable());
      Array<int> dofs, edges, orientations, edge_vertices;
      for (int vertex = 0; vertex < mesh.GetNV(); vertex++)
      {
         const HYPRE_BigInt global_vertex = vertices.GetGlobalTDofNumber(vertex);
         const int owner = upper_bound(vertex_offsets.begin(),
                                       vertex_offsets.end(), global_vertex)
                           - vertex_offsets.begin() - 1;
         set<HYPRE_BigInt> patch;
         fespace.GetVertexDofs(vertex, dofs);
         for (int dof : dofs)
         { patch.insert(fespace.GetGlobalTDofNumber(UnsignIndex(dof))); }
         const int *elements = vertex_elements->GetRow(vertex);
         for (int i = 0; i < vertex_elements->RowSize(vertex); i++)
         {
            const int element = elements[i];
            fespace.GetElementInteriorDofs(element, dofs);
            for (int dof : dofs)
            { patch.insert(fespace.GetGlobalTDofNumber(UnsignIndex(dof))); }
            mesh.GetElementEdges(element, edges, orientations);
            for (int edge : edges)
            {
               mesh.GetEdgeVertices(edge, edge_vertices);
               if (edge_vertices.Find(vertex) < 0) { continue; }
               fespace.GetEdgeInteriorDofs(edge, dofs);
               for (int dof : dofs)
               { patch.insert(fespace.GetGlobalTDofNumber(UnsignIndex(dof))); }
            }
         }
         for (HYPRE_BigInt dof : patch)
         {
            outgoing[owner].push_back(global_vertex);
            outgoing[owner].push_back(dof);
         }
      }

      Array<int> send_counts(nranks), recv_counts(nranks);
      Array<int> send_offsets(nranks + 1), recv_offsets(nranks + 1);
      for (int rank = 0; rank < nranks; rank++)
      { send_counts[rank] = outgoing[rank].size(); }
      MPI_Alltoall(send_counts.GetData(), 1, MPI_INT,
                   recv_counts.GetData(), 1, MPI_INT, comm);
      send_offsets[0] = recv_offsets[0] = 0;
      for (int rank = 0; rank < nranks; rank++)
      {
         send_offsets[rank+1] = send_offsets[rank] + send_counts[rank];
         recv_offsets[rank+1] = recv_offsets[rank] + recv_counts[rank];
      }
      Array<HYPRE_BigInt> send_dofs(send_offsets.Last());
      Array<HYPRE_BigInt> recv_dofs(recv_offsets.Last());
      for (int rank = 0; rank < nranks; rank++)
      {
         copy(outgoing[rank].begin(), outgoing[rank].end(),
              send_dofs.begin() + send_offsets[rank]);
      }
      MPI_Alltoallv(send_dofs.GetData(), send_counts.GetData(),
                    send_offsets.GetData(), HYPRE_MPI_BIG_INT,
                    recv_dofs.GetData(), recv_counts.GetData(),
                    recv_offsets.GetData(), HYPRE_MPI_BIG_INT, comm);

      map<HYPRE_BigInt, set<HYPRE_BigInt>> patches;
      map<HYPRE_BigInt, int> local_dofs;
      for (int i = 0; i < recv_dofs.Size(); i += 2)
      {
         patches[recv_dofs[i]].insert(recv_dofs[i+1]);
         local_dofs.emplace(recv_dofs[i+1], 0);
      }
      const int ndofs = local_dofs.size();
      Array<int> rows(ndofs + 1);
      Array<HYPRE_BigInt> columns(ndofs);
      Vector entries(ndofs);
      entries = 1.0;
      int index = 0;
      for (auto &dof : local_dofs)
      {
         dof.second = index;
         rows[index] = index;
         columns[index++] = dof.first;
      }
      rows[ndofs] = ndofs;
      HYPRE_BigInt local_size = ndofs;
      Array<HYPRE_BigInt> row_offsets;
      Array<HYPRE_BigInt> *offsets[] = { &row_offsets };
      mesh.GenerateOffsets(1, &local_size, offsets);
      gather.reset(new HypreParMatrix(comm, ndofs, row_offsets.Last(),
                                      fespace.GlobalTrueVSize(), rows.GetData(),
                                      columns.GetData(), entries.GetData(),
                                      row_offsets.GetData(),
                                      fespace.GetTrueDofOffsets()));

      // The diagonal block of G A G^T contains the global principal submatrix
      // needed for every patch assigned to this rank, including remote dofs.
      unique_ptr<HypreParMatrix> Gt(gather->Transpose());
      unique_ptr<HypreParMatrix> local_op(RAP(&op, Gt.get()));
      SparseMatrix local_matrix;
      local_op->GetDiag(local_matrix);
      Table patch_dofs;
      patch_dofs.MakeI(patches.size());
      index = 0;
      for (const auto &patch : patches)
      { patch_dofs.AddColumnsInRow(index++, patch.second.size()); }
      patch_dofs.MakeJ();
      index = 0;
      for (const auto &patch : patches)
      {
         for (HYPRE_BigInt dof : patch.second)
         { patch_dofs.AddConnection(index, local_dofs.at(dof)); }
         index++;
      }
      patch_dofs.ShiftUpI();
      local_smoother.reset(new VertexPatchSmoother(local_matrix, patch_dofs));
      local_rhs.SetSize(ndofs);
      local_solution.SetSize(ndofs);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      gather->Mult(x, local_rhs);
      local_smoother->Mult(local_rhs, local_solution);
      gather->MultTranspose(local_solution, y);
   }

   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
   void SetOperator(const Operator &) override
   { MFEM_ABORT("ParVertexPatchSmoother does not support SetOperator"); }
};

class SymmetricMatrixMultigrid : public GeometricMultigrid
{
   HypreSmoother coarse_preconditioner;

public:
   SymmetricMatrixMultigrid(ParFiniteElementSpaceHierarchy &fes_hierarchy,
                            const Array<int> &ess_bdr)
      : GeometricMultigrid(fes_hierarchy, ess_bdr)
   {
      const int num_levels = fes_hierarchy.GetNumLevels();
      for (int level = 0; level < num_levels; level++)
      {
         ParFiniteElementSpace &fespace = fes_hierarchy.GetFESpaceAtLevel(level);
         ParBilinearForm *form = new ParBilinearForm(&fespace);
         form->AddDomainIntegrator(new MatrixDivDivIntegrator);
         form->AddDomainIntegrator(new MatrixFEMassIntegrator);
         form->Assemble();
         bfs.Append(form);

         OperatorPtr level_operator(Operator::Hypre_ParCSR);
         form->FormSystemMatrix(*essentialTrueDofs[level], level_operator);
         HypreParMatrix *matrix = level_operator.As<HypreParMatrix>();
         level_operator.SetOperatorOwner(false);

         Solver *level_solver;
         if (level == 0)
         {
            coarse_preconditioner.SetType(HypreSmoother::l1GS);
            coarse_preconditioner.SetOperator(*matrix);
            coarse_preconditioner.SetOperatorSymmetry(true);
            CGSolver *coarse_solver = new CGSolver(matrix->GetComm());
            coarse_solver->SetOperator(*matrix);
            coarse_solver->SetPreconditioner(coarse_preconditioner);
            coarse_solver->SetRelTol(1e-10);
            coarse_solver->SetAbsTol(0.0);
            coarse_solver->SetMaxIter(500);
            coarse_solver->SetPrintLevel(-1);
            coarse_solver->iterative_mode = false;
            level_solver = coarse_solver;
         }
         else
         {
            level_solver = new ParVertexPatchSmoother(*matrix, fespace);
         }
         AddLevel(matrix, level_solver, false, true);
      }
   }
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../data/ref-triangle.mesh";
   int serial_refinements = 0;
   int geometric_refinements = 2;
   bool visualization = false;
   bool random_rhs = false;
   bool use_aw = false;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Input triangle mesh.");
   args.AddOption(&serial_refinements, "-rs", "--serial-refinements",
                  "Serial refinements before partitioning; additional refinements "
                  "ensure at least one coarse element per MPI rank.");
   args.AddOption(&geometric_refinements, "-r", "--refinements",
                  "Number of uniform parallel geometric refinements.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable or disable visualization (accepted for consistency).");
   args.AddOption(&random_rhs, "-random-rhs", "--random-rhs",
                  "-constant-rhs", "--constant-rhs",
                  "Use a random algebraic right-hand side reproducible for a fixed "
                  "MPI partition.");
   args.AddOption(&use_aw, "-aw", "--arnold-winther", "-jm", "--johnson-mercier",
                  "Use Arnold--Winther or Johnson--Mercier elements.");
   args.ParseCheck();

   MFEM_VERIFY(serial_refinements >= 0 && geometric_refinements >= 0,
               "Refinement counts must be nonnegative.");

   ParMesh pmesh = [&]()
   {
      Mesh mesh(mesh_file);
      MFEM_VERIFY(mesh.Dimension() == 2 && mesh.SpaceDimension() == 2 &&
                  !mesh.Nonconforming() && mesh.MeshGenerator() == 1, "");

      for (int i = 0; i < serial_refinements; i++) { mesh.UniformRefinement(); }
      if (mesh.GetNE() < Mpi::WorldSize() && Mpi::Root())
      {
         mfem::out << "WARNING: The requested serial refinement leaves only "
                   << mesh.GetNE() << " coarse elements for " << Mpi::WorldSize()
                   << " MPI ranks. Refining the serial mesh until it has at least "
                   "one element per rank; this changes the requested coarse "
                   "multigrid level.\n";
      }
      while (mesh.GetNE() < Mpi::WorldSize()) { mesh.UniformRefinement(); }

      return ParMesh(MPI_COMM_WORLD, mesh);
   }();

   tic();

   unique_ptr<FiniteElementCollection> fec(FiniteElementCollection::New(
                                              use_aw ? "AW_2D_P3" : "JM_2D_P1"));
   ParFiniteElementSpace coarse_fespace(&pmesh, fec.get());
   ParFiniteElementSpaceHierarchy fes_hierarchy(
      &pmesh, &coarse_fespace, false, false);
   for (int level = 0; level < geometric_refinements; level++)
   {
      fes_hierarchy.AddUniformlyRefinedLevel(
         1, Ordering::byVDIM, Operator::Hypre_ParCSR);
   }

   if (Mpi::Root()) { cout << "\nGeometric multigrid hierarchy:\n"; }
   for (int level = 0; level < fes_hierarchy.GetNumLevels(); level++)
   {
      ParFiniteElementSpace &fespace = fes_hierarchy.GetFESpaceAtLevel(level);
      const auto elements = fespace.GetParMesh()->GetGlobalNE();
      const auto unknowns = fespace.GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << "  level " << level << ": " << elements
              << " elements, " << unknowns << " unknowns\n";
      }
   }
   ParFiniteElementSpace &fine_fespace = fes_hierarchy.GetFinestFESpace();
   DenseMatrix identity(2);
   identity = 0.0;
   identity(0,0) = identity(1,1) = 1.0;
   MatrixConstantCoefficient rhs(identity);
   ParLinearForm b(&fine_fespace);
   b.AddDomainIntegrator(new MatrixFEDomainLFIntegrator(rhs));
   b.Assemble();
   ParGridFunction solution(&fine_fespace);
   solution = 0.0;
   Array<int> ess_bdr;
   SymmetricMatrixMultigrid multigrid(fes_hierarchy, ess_bdr);
   multigrid.SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);
   OperatorHandle A;
   Vector B, X;
   multigrid.FormFineLinearSystem(solution, b, A, X, B);
   if (random_rhs) { B.Randomize(1 + Mpi::WorldRank()); }

   if (Mpi::Root())
   {
      std::cout << "Setup elapsed: " << toc() << std::endl;
   }
   tic();

   CGSolver solver(MPI_COMM_WORLD);
   solver.SetOperator(*A);
   solver.SetPreconditioner(multigrid);
   solver.SetRelTol(1e-10);
   solver.SetAbsTol(0.0);
   solver.SetMaxIter(500);
   solver.SetPrintLevel(1);
   solver.Mult(B, X);

   if (Mpi::Root())
   {
      cout << "PCG iterations: " << solver.GetNumIterations() << '\n'
           << "Final residual norm: " << solver.GetFinalNorm() << '\n';
      if (!solver.GetConverged()) { cerr << "PCG did not converge.\n"; }
      std::cout << "Solve elapsed: " << toc() << std::endl;
   }

   return 0;
}
