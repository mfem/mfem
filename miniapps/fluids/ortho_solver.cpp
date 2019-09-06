#include "ortho_solver.hpp"

using namespace mfem;
using namespace flow;

OrthoSolver::OrthoSolver(MPI_Comm comm) : Solver(0, true), comm(comm) {}

void OrthoSolver::SetOperator(const Operator &op)
{
   oper = &op;
}

void OrthoSolver::Mult(const Vector &b, Vector &x) const
{
   // Apply operator.
   oper->Mult(b, x);

   // Orthoganlize residual.
   Orthoganalize(x);
}

void OrthoSolver::Orthoganalize(Vector &v) const
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   v -= global_sum / static_cast<double>(global_size);
}