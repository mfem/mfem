#include "ortho_solver.hpp"

using namespace mfem;
using namespace navier;

OrthoSolver::OrthoSolver() : Solver(0, true) {}

void OrthoSolver::SetOperator(const Operator &op)
{
   oper = &op;
}

void OrthoSolver::Mult(const Vector &b, Vector &x) const
{
   // Orthoganlize input.
   Orthoganalize(b, b_ortho);

   // Apply operator.
   oper->Mult(b_ortho, x);

   // Orthoganlize output.
   Orthoganalize(x, x);
}

void OrthoSolver::Orthoganalize(const Vector &v, Vector &v_ortho) const
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   double ratio = global_sum / static_cast<double>(global_size);
   v_ortho.SetSize(v.Size());
   for (int i = 0; i < v_ortho.Size(); ++i)
   {
      v_ortho(i) = v(i) - ratio;
   }
}