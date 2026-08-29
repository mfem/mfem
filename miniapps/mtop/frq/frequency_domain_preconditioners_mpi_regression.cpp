// MPI regression coverage for distributed two-level coarse projections.

#include "frequency_domain_preconditioners.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

using namespace mfem;

namespace
{

class DistributedDiagonalOperator : public Operator
{
public:
   DistributedDiagonalOperator(const int local_size, const int offset)
      : Operator(local_size), offset_(offset) { }

   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == Width(), "Distributed input size mismatch.");
      y.SetSize(Height());
      for (int i = 0; i < Height(); ++i)
      {
         y(i) = (2.0 + offset_ + i)*x(i);
      }
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      Mult(x, y);
   }

private:
   int offset_;
};

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);

#ifndef MFEM_USE_LAPACK
   if (Mpi::Root())
   {
      std::cout << "SKIP  two-level MPI regression requires LAPACK\n";
   }
   return EXIT_SUCCESS;
#else
   MPI_Comm communicator = MPI_COMM_WORLD;
   int rank = 0;
   MPI_Comm_rank(communicator, &rank);

   constexpr int local_size = 2;
   const int offset = rank*local_size;
   DistributedDiagonalOperator op(local_size, offset);
   TwoLevelPreconditioner preconditioner(communicator, op, 2);

   Vector z0(local_size), z1(local_size), rhs(local_size);
   for (int i = 0; i < local_size; ++i)
   {
      const real_t global_index = offset + i + 1.0;
      z0(i) = 1.0;
      z1(i) = global_index;
      rhs(i) = 0.25 - 0.1*global_index;
   }
   preconditioner.AddCoarseVector(z0);
   preconditioner.AddCoarseVector(z1);

   real_t local_values[4] = {0.0, 0.0, 0.0, 0.0};
   for (int i = 0; i < local_size; ++i)
   {
      const real_t diagonal = 2.0 + offset + i;
      local_values[0] += diagonal*z0(i)*z0(i);
      local_values[1] += diagonal*z0(i)*z1(i);
      local_values[2] += diagonal*z1(i)*z0(i);
      local_values[3] += diagonal*z1(i)*z1(i);
   }
   MPI_Allreduce(MPI_IN_PLACE, local_values, 4, MFEM_MPI_REAL_T, MPI_SUM,
                 communicator);

   const real_t determinant = local_values[0]*local_values[3]
                              - local_values[1]*local_values[2];
   MFEM_VERIFY(std::abs(determinant) > 0.0,
               "Distributed reference coarse matrix is singular.");
   auto ApplyReferenceCoarse = [&](const Vector &input)
   {
      real_t coarse_rhs[2] = {z0*input, z1*input};
      MPI_Allreduce(MPI_IN_PLACE, coarse_rhs, 2, MFEM_MPI_REAL_T, MPI_SUM,
                    communicator);
      const real_t coefficient0 =
         (local_values[3]*coarse_rhs[0]
          - local_values[1]*coarse_rhs[1])/determinant;
      const real_t coefficient1 =
         (-local_values[2]*coarse_rhs[0]
          + local_values[0]*coarse_rhs[1])/determinant;
      Vector output(local_size);
      for (int i = 0; i < local_size; ++i)
      {
         output(i) = coefficient0*z0(i) + coefficient1*z1(i);
      }
      return output;
   };

   const Vector expected = ApplyReferenceCoarse(rhs);
   Vector actual;
   preconditioner.MultCoarse(rhs, actual);
   actual -= expected;
   real_t local_error = actual.Normlinf();
   real_t global_error = 0.0;
   MPI_Allreduce(&local_error, &global_error, 1, MFEM_MPI_REAL_T, MPI_MAX,
                 communicator);

   const real_t tolerance =
      1000.0*std::numeric_limits<real_t>::epsilon();
   if (Mpi::Root())
   {
      std::cout << (global_error <= tolerance ? "PASS  " : "FAIL  ")
                << "distributed two-level coarse action: |error|_inf = "
                << global_error << '\n';
   }

   Vector expected_deflated(rhs);
   for (int i = 0; i < local_size; ++i)
   {
      expected_deflated(i) -= (2.0 + offset + i)*expected(i);
   }
   preconditioner.MultLeftDeflation(rhs, actual);
   actual -= expected_deflated;
   local_error = actual.Normlinf();
   real_t deflation_error = 0.0;
   MPI_Allreduce(&local_error, &deflation_error, 1, MFEM_MPI_REAL_T,
                 MPI_MAX, communicator);
   if (Mpi::Root())
   {
      std::cout << (deflation_error <= tolerance ? "PASS  " : "FAIL  ")
                << "distributed left deflation: |error|_inf = "
                << deflation_error << '\n';
   }

   Vector operator_rhs(local_size);
   op.Mult(rhs, operator_rhs);
   const Vector coarse_operator_rhs = ApplyReferenceCoarse(operator_rhs);
   Vector expected_right(rhs);
   expected_right -= coarse_operator_rhs;
   preconditioner.MultRightDeflation(rhs, actual);
   actual -= expected_right;
   local_error = actual.Normlinf();
   real_t right_error = 0.0;
   MPI_Allreduce(&local_error, &right_error, 1, MFEM_MPI_REAL_T, MPI_MAX,
                 communicator);

   Vector expected_operator(operator_rhs);
   Vector operator_coarse_operator_rhs(local_size);
   op.Mult(coarse_operator_rhs, operator_coarse_operator_rhs);
   expected_operator -= operator_coarse_operator_rhs;
   preconditioner.MultDeflatedOperator(rhs, actual);
   actual -= expected_operator;
   local_error = actual.Normlinf();
   real_t operator_error = 0.0;
   MPI_Allreduce(&local_error, &operator_error, 1, MFEM_MPI_REAL_T, MPI_MAX,
                 communicator);

   Vector expected_recovery(expected);
   expected_recovery += expected_right;
   preconditioner.RecoverDeflatedSolution(rhs, rhs, actual);
   actual -= expected_recovery;
   local_error = actual.Normlinf();
   real_t recovery_error = 0.0;
   MPI_Allreduce(&local_error, &recovery_error, 1, MFEM_MPI_REAL_T, MPI_MAX,
                 communicator);

   if (Mpi::Root())
   {
      std::cout << (right_error <= tolerance ? "PASS  " : "FAIL  ")
                << "distributed right deflation: |error|_inf = "
                << right_error << '\n'
                << (operator_error <= tolerance ? "PASS  " : "FAIL  ")
                << "distributed deflated operator: |error|_inf = "
                << operator_error << '\n'
                << (recovery_error <= tolerance ? "PASS  " : "FAIL  ")
                << "distributed deflated recovery: |error|_inf = "
                << recovery_error << '\n';
   }
   const bool passed = global_error <= tolerance &&
                       deflation_error <= tolerance &&
                       right_error <= tolerance &&
                       operator_error <= tolerance &&
                       recovery_error <= tolerance;
   return passed ? EXIT_SUCCESS : EXIT_FAILURE;
#endif
}
