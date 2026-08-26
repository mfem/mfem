// Algebraic regression tests for the problem-independent PRESB and two-block
// diagonal preconditioners. No mesh or PDE-specific operator is used here.

#include "frequency_domain_preconditioners.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace mfem;

namespace
{

class DenseTestOperator : public Operator
{
public:
   DenseTestOperator(const int size, std::vector<real_t> entries)
      : Operator(size), entries_(std::move(entries))
   {
      MFEM_VERIFY(static_cast<int>(entries_.size()) == size*size,
                  "Dense test matrix has the wrong number of entries.");
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == Width(), "Dense test input size mismatch.");
      y.SetSize(Height());
      for (int i = 0; i < Height(); ++i)
      {
         y(i) = 0.0;
         for (int j = 0; j < Width(); ++j)
         {
            y(i) += entries_[i*Width() + j]*x(j);
         }
      }
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == Height(), "Dense test input size mismatch.");
      y.SetSize(Width());
      for (int j = 0; j < Width(); ++j)
      {
         y(j) = 0.0;
         for (int i = 0; i < Height(); ++i)
         {
            y(j) += entries_[i*Width() + j]*x(i);
         }
      }
   }

private:
   std::vector<real_t> entries_;
};

Vector SolveDense(std::vector<real_t> matrix, Vector rhs,
                  const bool transpose = false)
{
   const int n = rhs.Size();
   MFEM_VERIFY(static_cast<int>(matrix.size()) == n*n,
               "Dense solve matrix size mismatch.");
   if (transpose)
   {
      for (int i = 0; i < n; ++i)
      {
         for (int j = i + 1; j < n; ++j)
         {
            std::swap(matrix[i*n + j], matrix[j*n + i]);
         }
      }
   }

   for (int pivot = 0; pivot < n; ++pivot)
   {
      int pivot_row = pivot;
      for (int row = pivot + 1; row < n; ++row)
      {
         if (std::abs(matrix[row*n + pivot]) >
             std::abs(matrix[pivot_row*n + pivot]))
         {
            pivot_row = row;
         }
      }
      MFEM_VERIFY(std::abs(matrix[pivot_row*n + pivot]) > 1.0e-14,
                  "Singular dense test matrix.");
      if (pivot_row != pivot)
      {
         for (int column = pivot; column < n; ++column)
         {
            std::swap(matrix[pivot*n + column],
                      matrix[pivot_row*n + column]);
         }
         std::swap(rhs(pivot), rhs(pivot_row));
      }

      const real_t diagonal = matrix[pivot*n + pivot];
      for (int row = pivot + 1; row < n; ++row)
      {
         const real_t factor = matrix[row*n + pivot]/diagonal;
         matrix[row*n + pivot] = 0.0;
         for (int column = pivot + 1; column < n; ++column)
         {
            matrix[row*n + column] -=
               factor*matrix[pivot*n + column];
         }
         rhs(row) -= factor*rhs(pivot);
      }
   }

   Vector solution(n);
   for (int row = n - 1; row >= 0; --row)
   {
      real_t value = rhs(row);
      for (int column = row + 1; column < n; ++column)
      {
         value -= matrix[row*n + column]*solution(column);
      }
      solution(row) = value/matrix[row*n + row];
   }
   return solution;
}

std::vector<real_t> MakePRESBMatrix(const std::vector<real_t> &W,
                                    const std::vector<real_t> &T,
                                    const int n, const int sign)
{
   std::vector<real_t> P(4*n*n, 0.0);
   const int size = 2*n;
   for (int i = 0; i < n; ++i)
   {
      for (int j = 0; j < n; ++j)
      {
         P[i*size + j] = W[i*n + j] + 2.0*T[i*n + j];
         P[i*size + n + j] = -sign*T[i*n + j];
         P[(n + i)*size + j] = sign*T[i*n + j];
         P[(n + i)*size + n + j] = W[i*n + j];
      }
   }
   return P;
}

std::vector<real_t> MakeBlockDiagonalMatrix(
   const std::vector<real_t> &H, const int n)
{
   std::vector<real_t> matrix(4*n*n, 0.0);
   const int size = 2*n;
   for (int i = 0; i < n; ++i)
   {
      for (int j = 0; j < n; ++j)
      {
         matrix[i*size + j] = H[i*n + j];
         matrix[(n + i)*size + n + j] = H[i*n + j];
      }
   }
   return matrix;
}

bool Check(const std::string &name, const Vector &actual,
           const Vector &expected, const real_t tolerance)
{
   Vector difference(actual);
   difference -= expected;
   const real_t error = difference.Normlinf();
   const bool passed = error <= tolerance;
   std::cout << (passed ? "PASS  " : "FAIL  ") << name
             << ": |error|_inf = " << error << '\n';
   return passed;
}

} // namespace

int main()
{
   constexpr int n = 2;
   // Deliberately nonsymmetric matrices ensure that the transpose tests
   // distinguish MultTranspose() from Mult(). The PRESB factorization remains
   // algebraically valid; symmetry is needed only for its spectral bounds.
   const std::vector<real_t> H = {4.0, 1.0,
                                  0.5, 3.0};
   const std::vector<real_t> H_inverse_entries = {3.0/11.5, -1.0/11.5,
                                                  -0.5/11.5, 4.0/11.5};
   const std::vector<real_t> T_entries = {1.0, 0.2,
                                          -0.1, 0.5};
   std::vector<real_t> W_entries(H.size());
   for (int i = 0; i < static_cast<int>(H.size()); ++i)
   {
      W_entries[i] = H[i] - T_entries[i];
   }

   DenseTestOperator T(n, T_entries);
   DenseTestOperator H_inverse(n, H_inverse_entries);
   Vector rhs(2*n);
   rhs(0) = 0.7;
   rhs(1) = -1.2;
   rhs(2) = 2.1;
   rhs(3) = 0.4;

   int failures = 0;
   const real_t tolerance = 100.0*std::numeric_limits<real_t>::epsilon();

   for (const int sign : {1, -1})
   {
      PRESBPreconditioner presb(T, H_inverse, sign);
      const std::vector<real_t> P =
         MakePRESBMatrix(W_entries, T_entries, n, sign);

      Vector actual;
      presb.Mult(rhs, actual);
      const Vector expected = SolveDense(P, rhs);
      failures += !Check("PRESB inverse, sign " + std::to_string(sign),
                         actual, expected, tolerance);

      presb.MultTranspose(rhs, actual);
      const Vector expected_transpose = SolveDense(P, rhs, true);
      failures += !Check("PRESB inverse transpose, sign " +
                         std::to_string(sign), actual,
                         expected_transpose, tolerance);

      Vector in_place(rhs);
      presb.Mult(in_place, in_place);
      failures += !Check("PRESB in-place application, sign " +
                         std::to_string(sign), in_place,
                         expected, tolerance);
   }

   RealBlockDiagonalPreconditioner block_diagonal(H_inverse);
   const std::vector<real_t> P_block = MakeBlockDiagonalMatrix(H, n);
   Vector actual;
   block_diagonal.Mult(rhs, actual);
   const Vector expected = SolveDense(P_block, rhs);
   failures += !Check("two-block diagonal inverse", actual, expected,
                      tolerance);

   block_diagonal.MultTranspose(rhs, actual);
   const Vector expected_transpose = SolveDense(P_block, rhs, true);
   failures += !Check("two-block diagonal inverse transpose", actual,
                      expected_transpose, tolerance);

   Vector block_in_place(rhs);
   block_diagonal.Mult(block_in_place, block_in_place);
   failures += !Check("two-block diagonal in-place application",
                      block_in_place, expected, tolerance);

   Vector block_transpose_in_place(rhs);
   block_diagonal.MultTranspose(block_transpose_in_place,
                                block_transpose_in_place);
   failures += !Check("two-block diagonal in-place transpose application",
                      block_transpose_in_place, expected_transpose, tolerance);

   std::cout << (failures == 0 ? "ALL TESTS PASSED\n" : "TESTS FAILED\n");
   return failures == 0 ? 0 : 2;
}
