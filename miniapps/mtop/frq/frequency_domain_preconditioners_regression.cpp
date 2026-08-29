// Algebraic regression tests for the problem-independent frequency-domain
// and two-level preconditioners. No mesh or PDE-specific operator is used.

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

#ifdef MFEM_USE_LAPACK
Vector ApplyDense(const std::vector<real_t> &matrix, const Vector &input,
                  const bool transpose = false)
{
   const int n = input.Size();
   Vector output(n);
   output = 0.0;
   for (int i = 0; i < n; ++i)
   {
      for (int j = 0; j < n; ++j)
      {
         output(i) += (transpose ? matrix[j*n + i] : matrix[i*n + j])*
                      input(j);
      }
   }
   return output;
}

Vector ApplyCoarseReference(const std::vector<real_t> &matrix,
                            const std::vector<Vector> &basis,
                            const Vector &input)
{
   const int n = input.Size();
   const int coarse_size = static_cast<int>(basis.size());
   std::vector<real_t> reduced(coarse_size*coarse_size, 0.0);
   Vector coarse_rhs(coarse_size);
   for (int j = 0; j < coarse_size; ++j)
   {
      const Vector image = ApplyDense(matrix, basis[j]);
      for (int i = 0; i < coarse_size; ++i)
      {
         reduced[i*coarse_size + j] = basis[i]*image;
      }
      coarse_rhs(j) = basis[j]*input;
   }

   const Vector coarse_solution = SolveDense(reduced, coarse_rhs);
   Vector output(n);
   output = 0.0;
   for (int i = 0; i < coarse_size; ++i)
   {
      output.Add(coarse_solution(i), basis[i]);
   }
   return output;
}

std::vector<real_t> TransposeDense(const std::vector<real_t> &matrix,
                                   const int n)
{
   std::vector<real_t> transpose(n*n);
   for (int i = 0; i < n; ++i)
   {
      for (int j = 0; j < n; ++j)
      {
         transpose[i*n + j] = matrix[j*n + i];
      }
   }
   return transpose;
}

Vector ApplyTwoLevelReference(const std::vector<real_t> &matrix,
                              const std::vector<real_t> &smoother,
                              const std::vector<Vector> &basis,
                              const Vector &input)
{
   Vector output = ApplyDense(smoother, input);
   Vector residual(input);
   residual -= ApplyDense(matrix, output);

   const Vector correction = ApplyCoarseReference(matrix, basis, residual);
   output += correction;
   residual -= ApplyDense(matrix, correction);
   output += ApplyDense(smoother, residual, true);
   return output;
}

Vector ApplyTwoLevelReference(const std::vector<real_t> &matrix,
                              const std::vector<real_t> &pre_smoother,
                              const std::vector<real_t> &post_smoother,
                              const std::vector<Vector> &basis,
                              const Vector &input)
{
   Vector output = ApplyDense(pre_smoother, input);
   Vector residual(input);
   residual -= ApplyDense(matrix, output);

   const Vector correction = ApplyCoarseReference(matrix, basis, residual);
   output += correction;
   residual -= ApplyDense(matrix, correction);
   output += ApplyDense(post_smoother, residual);
   return output;
}

std::vector<real_t> MakeTwoLevelReferenceMatrix(
   const std::vector<real_t> &matrix,
   const std::vector<real_t> &smoother,
   const std::vector<Vector> &basis)
{
   const int n = basis[0].Size();
   std::vector<real_t> result(n*n, 0.0);
   for (int j = 0; j < n; ++j)
   {
      Vector unit(n);
      unit = 0.0;
      unit(j) = 1.0;
      const Vector column =
         ApplyTwoLevelReference(matrix, smoother, basis, unit);
      for (int i = 0; i < n; ++i)
      {
         result[i*n + j] = column(i);
      }
   }
   return result;
}

std::vector<real_t> MakeTwoLevelReferenceMatrix(
   const std::vector<real_t> &matrix,
   const std::vector<real_t> &pre_smoother,
   const std::vector<real_t> &post_smoother,
   const std::vector<Vector> &basis)
{
   const int n = basis[0].Size();
   std::vector<real_t> result(n*n, 0.0);
   for (int j = 0; j < n; ++j)
   {
      Vector unit(n);
      unit = 0.0;
      unit(j) = 1.0;
      const Vector column = ApplyTwoLevelReference(
         matrix, pre_smoother, post_smoother, basis, unit);
      for (int i = 0; i < n; ++i)
      {
         result[i*n + j] = column(i);
      }
   }
   return result;
}
#endif

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

   // Distinct Vector objects may still alias the same underlying storage.
   Vector aliased_storage(rhs);
   Vector aliased_input, aliased_output;
   aliased_input.MakeRef(aliased_storage, 0, aliased_storage.Size());
   aliased_output.MakeRef(aliased_storage, 0, aliased_storage.Size());
   block_diagonal.Mult(aliased_input, aliased_output);
   failures += !Check("two-block diagonal shared-storage alias",
                      aliased_output, expected, tolerance);

   aliased_storage = rhs;
   block_diagonal.MultTranspose(aliased_input, aliased_output);
   failures += !Check("two-block diagonal transpose shared-storage alias",
                      aliased_output, expected_transpose, tolerance);

#ifdef MFEM_USE_LAPACK
   const std::vector<real_t> two_level_matrix = {
      4.0, 1.0, 0.5,
      0.0, 3.0, 1.0,
      0.2, 0.0, 2.0
   };
   const std::vector<real_t> smoother_entries = {
      0.20, 0.03, 0.00,
      0.01, 0.25, 0.02,
      0.00, 0.04, 0.30
   };
   const std::vector<real_t> post_smoother_entries = {
      0.18, 0.00, 0.02,
      0.03, 0.22, 0.00,
      0.01, 0.05, 0.27
   };
   DenseTestOperator two_level_operator(3, two_level_matrix);
   DenseTestOperator smoother(3, smoother_entries);
   DenseTestOperator post_smoother(3, post_smoother_entries);
   Vector z0(3), z1(3), z2(3), two_level_rhs(3);
   z0 = 0.0;
   z1 = 0.0;
   z2 = 0.0;
   z0(0) = 1.0;
   z1(1) = 1.0;
   z2(2) = 1.0;
   two_level_rhs(0) = 0.7;
   two_level_rhs(1) = -1.1;
   two_level_rhs(2) = 0.4;

   TwoLevelPreconditioner two_level(two_level_operator, 2);
   failures += two_level.GetMaxCoarseVectors() != 2;
   failures += two_level.GetNumCoarseVectors() != 0;
   failures += two_level.AddCoarseVector(z0) != 0;
   failures += two_level.AddCoarseVector(z1) != 1;
   failures += two_level.GetNumCoarseVectors() != 2;
   const std::vector<Vector> first_basis = {z0, z1};

   Vector two_level_actual;
   two_level.Assemble();
   two_level.Mult(two_level_rhs, two_level_actual);
   const Vector coarse_expected =
      ApplyCoarseReference(two_level_matrix, first_basis, two_level_rhs);
   failures += !Check("two-level coarse-only action", two_level_actual,
                      coarse_expected, 500.0*tolerance);

   two_level.MultCoarse(two_level_rhs, two_level_actual);
   failures += !Check("two-level explicit coarse action", two_level_actual,
                      coarse_expected, 500.0*tolerance);

   Vector left_deflation_expected(two_level_rhs);
   left_deflation_expected -=
      ApplyDense(two_level_matrix, coarse_expected);
   two_level.MultLeftDeflation(two_level_rhs, two_level_actual);
   failures += !Check("two-level left deflation", two_level_actual,
                      left_deflation_expected, 1000.0*tolerance);
   two_level.FormDeflatedRHS(two_level_rhs, two_level_actual);
   failures += !Check("two-level deflated right-hand side", two_level_actual,
                      left_deflation_expected, 1000.0*tolerance);
   Vector deflation_in_place(two_level_rhs);
   two_level.MultLeftDeflation(deflation_in_place, deflation_in_place);
   failures += !Check("two-level in-place left deflation",
                      deflation_in_place, left_deflation_expected,
                      1000.0*tolerance);

   const std::vector<real_t> transpose_matrix =
      TransposeDense(two_level_matrix, 3);
   const Vector transpose_image =
      ApplyDense(two_level_matrix, two_level_rhs, true);
   Vector right_deflation_expected(two_level_rhs);
   right_deflation_expected -=
      ApplyCoarseReference(transpose_matrix, first_basis, transpose_image);
   two_level.MultRightDeflation(two_level_rhs, two_level_actual);
   failures += !Check("two-level right deflation", two_level_actual,
                      right_deflation_expected, 1000.0*tolerance);

   const Vector operator_image =
      ApplyDense(two_level_matrix, two_level_rhs);
   const Vector coarse_operator_image =
      ApplyCoarseReference(two_level_matrix, first_basis, operator_image);
   Vector deflated_operator_expected(operator_image);
   deflated_operator_expected -=
      ApplyDense(two_level_matrix, coarse_operator_image);
   two_level.MultDeflatedOperator(two_level_rhs, two_level_actual);
   failures += !Check("two-level deflated operator", two_level_actual,
                      deflated_operator_expected, 1000.0*tolerance);

   Vector complementary(3);
   complementary(0) = -0.2;
   complementary(1) = 0.6;
   complementary(2) = 1.1;
   Vector recovered_expected(coarse_expected);
   Vector projected_complementary(complementary);
   const Vector transpose_complementary =
      ApplyDense(two_level_matrix, complementary, true);
   projected_complementary -= ApplyCoarseReference(
                                 transpose_matrix, first_basis,
                                 transpose_complementary);
   recovered_expected += projected_complementary;
   two_level.RecoverDeflatedSolution(two_level_rhs, complementary,
                                     two_level_actual);
   failures += !Check("two-level deflated solution recovery",
                      two_level_actual, recovered_expected,
                      1000.0*tolerance);
   Vector recovery_in_place(two_level_rhs);
   two_level.RecoverDeflatedSolution(recovery_in_place, complementary,
                                     recovery_in_place);
   failures += !Check("two-level in-place deflated recovery",
                      recovery_in_place, recovered_expected,
                      1000.0*tolerance);

   two_level.SetSmoother(smoother);
   failures += !two_level.PostSmootherUsesTranspose();
   const Vector smoothed_expected = ApplyTwoLevelReference(
      two_level_matrix, smoother_entries, first_basis, two_level_rhs);
   two_level.Mult(two_level_rhs, two_level_actual);
   failures += !Check("two-level smoothed action", two_level_actual,
                      smoothed_expected, 500.0*tolerance);

   const std::vector<real_t> two_level_reference =
      MakeTwoLevelReferenceMatrix(two_level_matrix, smoother_entries,
                                  first_basis);
   two_level.MultTranspose(two_level_rhs, two_level_actual);
   const Vector two_level_transpose_expected =
      ApplyDense(two_level_reference, two_level_rhs, true);
   failures += !Check("two-level transpose action", two_level_actual,
                      two_level_transpose_expected, 1000.0*tolerance);

   Vector two_level_in_place(two_level_rhs);
   two_level.Mult(two_level_in_place, two_level_in_place);
   failures += !Check("two-level in-place action", two_level_in_place,
                      smoothed_expected, 500.0*tolerance);

   Vector two_level_alias_storage(two_level_rhs);
   Vector two_level_alias_input, two_level_alias_output;
   two_level_alias_input.MakeRef(two_level_alias_storage, 0, 3);
   two_level_alias_output.MakeRef(two_level_alias_storage, 0, 3);
   two_level.Mult(two_level_alias_input, two_level_alias_output);
   failures += !Check("two-level shared-storage alias",
                      two_level_alias_output, smoothed_expected,
                      500.0*tolerance);

   two_level.SetPreSmoother(smoother);
   two_level.SetPostSmoother(post_smoother);
   const Vector independent_smoothers_expected = ApplyTwoLevelReference(
      two_level_matrix, smoother_entries, post_smoother_entries,
      first_basis, two_level_rhs);
   two_level.Mult(two_level_rhs, two_level_actual);
   failures += !Check("two-level independent smoothers", two_level_actual,
                      independent_smoothers_expected, 500.0*tolerance);
   const std::vector<real_t> independent_reference =
      MakeTwoLevelReferenceMatrix(two_level_matrix, smoother_entries,
                                  post_smoother_entries, first_basis);
   two_level.MultTranspose(two_level_rhs, two_level_actual);
   const Vector independent_transpose_expected =
      ApplyDense(independent_reference, two_level_rhs, true);
   failures += !Check("two-level independent-smoother transpose",
                      two_level_actual, independent_transpose_expected,
                      1000.0*tolerance);
   failures += two_level.GetPreSmoother() != &smoother;
   failures += two_level.GetPostSmoother() != &post_smoother;
   failures += two_level.GetSmoother() != nullptr;
   failures += two_level.PostSmootherUsesTranspose();

   two_level.SetSmoother(nullptr);
   two_level.Mult(two_level_rhs, two_level_actual);
   failures += !Check("two-level removed smoother", two_level_actual,
                      coarse_expected, 500.0*tolerance);

   failures += two_level.AddCoarseVector(z2) != 0;
   Vector extracted;
   two_level.GetCoarseVector(0, extracted);
   failures += !Check("two-level cyclic overwrite", extracted, z2,
                      tolerance);
   two_level.SetCoarseVector(1, z0);
   two_level.GetCoarseVector(1, extracted);
   failures += !Check("two-level indexed replacement", extracted, z0,
                      tolerance);
   const std::vector<Vector> replaced_basis = {z2, z0};
   two_level.Mult(two_level_rhs, two_level_actual);
   const Vector replaced_expected = ApplyCoarseReference(
      two_level_matrix, replaced_basis, two_level_rhs);
   failures += !Check("two-level lazy reassembly", two_level_actual,
                      replaced_expected, 500.0*tolerance);

   const std::vector<real_t> identity_entries = {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0
   };
   DenseTestOperator identity(3, identity_entries);
   TwoLevelPreconditioner rank_deficient(identity, 2);
   rank_deficient.AddCoarseVector(z0);
   rank_deficient.AddCoarseVector(z0);
   rank_deficient.Mult(two_level_rhs, two_level_actual);
   Vector rank_deficient_expected(3);
   rank_deficient_expected = 0.0;
   rank_deficient_expected(0) = two_level_rhs(0);
   failures += !Check("two-level rank-deficient basis", two_level_actual,
                      rank_deficient_expected, 1000.0*tolerance);

   Vector scaled_z1(z1);
   scaled_z1 *= 1.0e-4;
   TwoLevelPreconditioner filtered(identity, 2);
   filtered.AddCoarseVector(z0);
   filtered.AddCoarseVector(scaled_z1);
   filtered.SetSVDRelativeTolerance(1.0e-6);
   filtered.Mult(two_level_rhs, two_level_actual);
   failures += !Check("two-level SVD filtering", two_level_actual,
                      rank_deficient_expected, 1000.0*tolerance);

   filtered.SetSVDRelativeTolerance(0.0);
   filtered.Mult(two_level_rhs, two_level_actual);
   Vector unfiltered_expected(two_level_rhs);
   unfiltered_expected(2) = 0.0;
   failures += !Check("two-level configurable SVD tolerance",
                      two_level_actual, unfiltered_expected,
                      1000.0*tolerance);
#endif

   std::cout << (failures == 0 ? "ALL TESTS PASSED\n" : "TESTS FAILED\n");
   return failures == 0 ? 0 : 2;
}
