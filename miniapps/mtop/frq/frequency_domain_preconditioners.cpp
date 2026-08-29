// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.

#include "frequency_domain_preconditioners.hpp"

#include <climits>
#include <limits>

// This implementation depends only on MFEM's linear algebra interfaces.

namespace mfem
{

namespace
{

/// Construct offsets for two contiguous blocks of equal size.
///
/// The returned array is [0,n,2n], as required by mfem::BlockVector. Keeping
/// this layout local to the implementation prevents either preconditioner from
/// depending on a PDE-specific finite element space.
Array<int> MakeTwoBlockOffsets(const int block_size)
{
   MFEM_VERIFY(block_size >= 0 && block_size <= INT_MAX / 2,
               "Block size out of valid range.");
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = block_size;
   offsets[2] = 2*block_size;
   return offsets;
}

/// Verify the square-block requirement in References [1,2].
///
/// PRESB and the associated block-diagonal method are defined for two-by-two
/// systems whose four blocks have the same order.
void VerifySquareOperator(const Operator &op, const char *name)
{
   MFEM_VERIFY(op.Height() == op.Width(), name << " must be square.");
}

} // namespace

/// Initialize a PRESB inverse action from T and H^{-1}, where H = W+T.
///
/// Only these two constituent actions are retained, so this constructor is
/// independent of how W, T, and H were assembled. The two operators are
/// non-owning. The full real-block size exposed through mfem::Solver is 2n.
/// Work vectors are configured to use device memory when MFEM device support
/// is enabled.
PRESBPreconditioner::PRESBPreconditioner(const Operator &T,
                                         const Operator &H_inverse,
                                         const int imaginary_sign)
   : Solver(2*T.Height()),
     T_(&T),
     H_inverse_(&H_inverse),
     imaginary_sign_(imaginary_sign),
     block_size_(T.Height()),
     first_rhs_(block_size_),
     second_rhs_(block_size_),
     first_solution_(block_size_),
     second_solution_(block_size_)
{
   VerifySquareOperator(T, "T");
   VerifySquareOperator(H_inverse, "H inverse");
   MFEM_VERIFY(T.Width() == block_size_ && H_inverse.Width() == block_size_,
               "T and H inverse dimensions are inconsistent.");
   MFEM_VERIFY(H_inverse.Height() == block_size_,
               "T and H inverse sizes do not match.");
   MFEM_VERIFY(imaginary_sign_ == 1 || imaginary_sign_ == -1,
               "The imaginary sign must be +1 or -1.");

   // Enable device execution for all work vectors. This allows GPU operation
   // when MFEM is configured with device support and the input vectors and
   // operators are device-capable.
   first_rhs_.UseDevice(true);
   second_rhs_.UseDevice(true);
   first_solution_.UseDevice(true);
   second_solution_.UseDevice(true);
}

/// Check the [real; imaginary] input length before accessing its blocks.
void PRESBPreconditioner::ValidateInput(const Vector &b) const
{
   MFEM_VERIFY(b.Size() == Width(),
               "PRESB input has an incompatible size.");
}

/// Check compatibility with an outer real-block system.
///
/// mfem::Solver requires this method, but PRESB cannot recover T or H^{-1}
/// from an arbitrary Operator. Consequently SetOperator validates dimensions
/// without changing the constituent operators supplied to the constructor.
/// This is intentional: the preconditioner is defined by its constructor
/// arguments and uses duck typing for operator compatibility.
void PRESBPreconditioner::SetOperator(const Operator &op)
{
   MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
               "The real block operator is incompatible with PRESB.");
}

/// Apply P_s^{-1} using the block factorization from References [1,2].
///
/// With H = W+T and b = [b_r;b_i], the implementation computes
///
///   z   = H^{-1}(b_r - s b_i),
///   q   = H^{-1}(s b_i - T z),
///   x_r = z + q,  x_i = s q.
///
/// Optimized to avoid unnecessary vector copies by detecting aliasing and using
/// block views directly when possible.
void PRESBPreconditioner::Mult(const Vector &b, Vector &x) const
{
   ValidateInput(b);

   // Create block views directly from input to avoid full vector copy
   b.Read();
   Vector b_real_view, b_imag_view;
   b_real_view.MakeRef(const_cast<Vector&>(b), 0, block_size_);
   b_imag_view.MakeRef(const_cast<Vector&>(b), block_size_, block_size_);

   // Use local sign variable for cleaner code.
   const real_t s = imaginary_sign_;

   // First solve: H z = b_r - s*b_i
   // The sign conjugates the formulation (e.g., s=-1 for W-iT becomes W+iT).
   first_rhs_ = b_real_view;
   first_rhs_.Add(-s, b_imag_view);
   H_inverse_->Mult(first_rhs_, first_solution_);

   // Second solve: H q = s*b_i - T*z
   // Optimize: reuse first_rhs_ as work vector to reduce memory footprint
   T_->Mult(first_solution_, first_rhs_);
   second_rhs_ = b_imag_view;
   second_rhs_ *= s;
   second_rhs_ -= first_rhs_;
   H_inverse_->Mult(second_rhs_, second_solution_);

   // Assemble result directly into output x
   // Create block views for output
   x.SetSize(Height());
   x.UseDevice(true);
   Vector x_real, x_imag;
   x_real.MakeRef(x, 0, block_size_);
   x_imag.MakeRef(x, block_size_, block_size_);

   // x_r = z + q, x_i = s*q
   x_real = first_solution_;
   x_real += second_solution_;
   x_imag = second_solution_;
   x_imag *= s;

   // Sync memory views
   x_real.SyncAliasMemory(x);
   x_imag.SyncAliasMemory(x);
}

/// Apply P_s^{-T} by transposing the PRESB block factorization.
///
/// After conjugating the second block by s, the two triangular solves are
///
///   q_2 = H^{-T}(b_r + s b_i),
///   q_1 = H^{-T}(b_r - T^T q_2),
///
/// followed by x_r = q_1 and x_i = s(q_2 - q_1). This method therefore
/// requires transpose actions from both constituent operators.
/// Optimized to avoid unnecessary vector copies.
void PRESBPreconditioner::MultTranspose(const Vector &b, Vector &x) const
{
   ValidateInput(b);

   // Create block views directly from input
   b.Read();
   Vector b_real_view, b_imag_view;
   b_real_view.MakeRef(const_cast<Vector&>(b), 0, block_size_);
   b_imag_view.MakeRef(const_cast<Vector&>(b), block_size_, block_size_);

   // Use local sign variable for cleaner code.
   const real_t s = imaginary_sign_;

   // First solve: H^T q_2 = b_r + s*b_i
   // P_s^{-T} = D_s P_+^{-T} D_s for sign conjugation.
   second_rhs_ = b_real_view;
   second_rhs_.Add(s, b_imag_view);
   H_inverse_->MultTranspose(second_rhs_, second_solution_);

   // Second solve: H^T q_1 = b_r - T^T*q_2
   // Optimize: reuse second_rhs_ as work vector
   T_->MultTranspose(second_solution_, second_rhs_);
   first_rhs_ = b_real_view;
   first_rhs_ -= second_rhs_;
   H_inverse_->MultTranspose(first_rhs_, first_solution_);

   // Assemble result directly into output x
   x.SetSize(Height());
   x.UseDevice(true);
   Vector x_real, x_imag;
   x_real.MakeRef(x, 0, block_size_);
   x_imag.MakeRef(x, block_size_, block_size_);

   // x_r = q_1, x_i = s*(q_2 - q_1)
   x_real = first_solution_;
   x_imag = second_solution_;
   x_imag -= first_solution_;
   x_imag *= s;

   // Sync memory views
   x_real.SyncAliasMemory(x);
   x_imag.SyncAliasMemory(x);
}

/// Initialize P_BD^{-1} = diag(H^{-1},H^{-1}).
///
/// H_inverse is non-owning and may represent an exact solve or a fixed
/// approximate inverse. The full real-block size exposed by Solver is 2n.
/// Work vectors are configured to use device memory when MFEM device support
/// is enabled.
RealBlockDiagonalPreconditioner::RealBlockDiagonalPreconditioner(
   const Operator &H_inverse)
   : Solver(2*H_inverse.Height()),
     H_inverse_(&H_inverse),
     block_size_(H_inverse.Height()),
     block_offsets_(MakeTwoBlockOffsets(block_size_)),
     input_(block_offsets_),
     output_(block_offsets_)
{
   VerifySquareOperator(H_inverse, "H inverse");

   // Enable device execution for work vectors.
   input_.UseDevice(true);
   output_.UseDevice(true);
   input_.SyncToBlocks();
   output_.SyncToBlocks();
}

/// Check the two-block input length before accessing its blocks.
void RealBlockDiagonalPreconditioner::ValidateInput(const Vector &b) const
{
   MFEM_VERIFY(b.Size() == Width(),
               "Block-diagonal input has an incompatible size.");
}

/// Check compatibility with an outer symmetric real-block system.
///
/// The supplied H^{-1} action cannot be inferred from a general Operator, so
/// this method validates dimensions without replacing H_inverse_. This is
/// intentional: the preconditioner is defined by its constructor arguments
/// and uses duck typing for operator compatibility.
void RealBlockDiagonalPreconditioner::SetOperator(const Operator &op)
{
   MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
               "The real block operator is incompatible with the "
               "two-block diagonal preconditioner.");
}

/// Apply H^{-1} independently to the two blocks of @a b.
///
/// Stage the input and output through owned block vectors so that the input may
/// safely alias the output even when H_inverse does not support in-place use.
void RealBlockDiagonalPreconditioner::Mult(const Vector &b, Vector &x) const
{
   ValidateInput(b);

   // Always stage the input because distinct Vector objects can still refer to
   // the same memory. Object-address comparison is not sufficient to detect
   // that form of aliasing, and H_inverse need not support in-place use.
   input_.Set(1.0, b);
   input_.SyncToBlocks();
   H_inverse_->Mult(input_.GetBlock(0), output_.GetBlock(0));
   H_inverse_->Mult(input_.GetBlock(1), output_.GetBlock(1));
   output_.SyncFromBlocks();
   x.UseDevice(true);
   x = output_;
}

/// Apply H^{-T} independently to the two blocks of @a b.
///
/// MINRES normally calls Mult(), not this method; MultTranspose() is supplied
/// to make the inverse operator complete for other MFEM compositions.
/// Staging also preserves the documented in-place behavior for this action.
void RealBlockDiagonalPreconditioner::MultTranspose(const Vector &b,
                                                    Vector &x) const
{
   ValidateInput(b);

   // See Mult(): staging is required for aliases represented by different
   // Vector objects as well as for the exact same object.
   input_.Set(1.0, b);
   input_.SyncToBlocks();
   H_inverse_->MultTranspose(input_.GetBlock(0), output_.GetBlock(0));
   H_inverse_->MultTranspose(input_.GetBlock(1), output_.GetBlock(1));
   output_.SyncFromBlocks();
   x.UseDevice(true);
   x = output_;
}

/// Construct a two-level preconditioner using rank-local inner products.
TwoLevelPreconditioner::TwoLevelPreconditioner(
   const Operator &op, const int max_coarse_vectors,
   const Operator *smoother)
   : Solver(op.Height(), op.Width()),
     operator_(&op),
     pre_smoother_(nullptr),
     post_smoother_(nullptr),
     max_coarse_vectors_(max_coarse_vectors)
{
   ValidateOperator(op, "Two-level operator");
   MFEM_VERIFY(max_coarse_vectors_ > 0,
               "The coarse-vector capacity must be positive.");
   coarse_vectors_.resize(max_coarse_vectors_);

   SetSmoother(smoother);
   pre_smoothed_.SetSize(Height());
   residual_.SetSize(Height());
   coarse_correction_.SetSize(Height());
   operator_work_.SetSize(Height());
   post_smoothed_.SetSize(Height());
   pre_smoothed_.UseDevice(true);
   residual_.UseDevice(true);
   coarse_correction_.UseDevice(true);
   operator_work_.UseDevice(true);
   post_smoothed_.UseDevice(true);
}

#ifdef MFEM_USE_MPI
/// Construct a two-level preconditioner using communicator-wide products.
TwoLevelPreconditioner::TwoLevelPreconditioner(
   MPI_Comm communicator, const Operator &op,
   const int max_coarse_vectors, const Operator *smoother)
   : TwoLevelPreconditioner(op, max_coarse_vectors, smoother)
{
   MFEM_VERIFY(communicator != MPI_COMM_NULL,
               "The two-level communicator must not be MPI_COMM_NULL.");
   communicator_ = communicator;
   use_global_inner_products_ = true;
}
#endif

/// Verify a square fine-grid operator with the fixed local dimensions.
void TwoLevelPreconditioner::ValidateOperator(const Operator &op,
                                              const char *name) const
{
   MFEM_VERIFY(op.Height() == op.Width(), name << " must be square.");
   MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
               name << " has incompatible dimensions.");
}

/// Verify the local segment of a coarse or fine-grid vector.
void TwoLevelPreconditioner::ValidateVector(const Vector &vector) const
{
   MFEM_VERIFY(vector.Size() == Width(),
               "Two-level vector has an incompatible local size.");
}

/// Append a vector until full, then replace the next cyclic physical slot.
int TwoLevelPreconditioner::AddCoarseVector(const Vector &vector)
{
   ValidateVector(vector);

   int slot;
   if (num_coarse_vectors_ < max_coarse_vectors_)
   {
      slot = num_coarse_vectors_++;
      if (num_coarse_vectors_ == max_coarse_vectors_)
      {
         next_coarse_slot_ = 0;
      }
   }
   else
   {
      slot = next_coarse_slot_;
      next_coarse_slot_ = (next_coarse_slot_ + 1)%max_coarse_vectors_;
   }

   coarse_vectors_[slot].SetSize(Width());
   coarse_vectors_[slot].UseDevice(true);
   coarse_vectors_[slot] = vector;
   assembled_ = false;
   return slot;
}

/// Replace one occupied slot without changing the cyclic insertion cursor.
void TwoLevelPreconditioner::SetCoarseVector(const int slot,
                                             const Vector &vector)
{
   MFEM_VERIFY(slot >= 0 && slot < num_coarse_vectors_,
               "The coarse-vector slot is not occupied.");
   ValidateVector(vector);
   coarse_vectors_[slot] = vector;
   assembled_ = false;
}

/// Extract a copy of one rank-local coarse-vector segment.
void TwoLevelPreconditioner::GetCoarseVector(const int slot,
                                             Vector &vector) const
{
   MFEM_VERIFY(slot >= 0 && slot < num_coarse_vectors_,
               "The coarse-vector slot is not occupied.");
   vector = coarse_vectors_[slot];
}

/// Set one smoother as the pre-action and its transpose as the post-action.
void TwoLevelPreconditioner::SetSmoother(const Operator *smoother)
{
   if (smoother)
   {
      ValidateOperator(*smoother, "Two-level smoother");
   }
   pre_smoother_ = smoother;
   post_smoother_ = smoother;
   post_smoother_uses_transpose_ = smoother != nullptr;
}

/// Replace or remove the independent pre-smoothing action.
void TwoLevelPreconditioner::SetPreSmoother(const Operator *smoother)
{
   if (smoother)
   {
      ValidateOperator(*smoother, "Two-level pre-smoother");
   }
   pre_smoother_ = smoother;
}

/// Replace or remove the independent post-smoothing action.
void TwoLevelPreconditioner::SetPostSmoother(const Operator *smoother)
{
   if (smoother)
   {
      ValidateOperator(*smoother, "Two-level post-smoother");
   }
   post_smoother_ = smoother;
   post_smoother_uses_transpose_ = false;
}

/// Select the relative singular-value threshold used by the next assembly.
void TwoLevelPreconditioner::SetSVDRelativeTolerance(const real_t tolerance)
{
   MFEM_VERIFY(tolerance >= 0.0 || tolerance == -1.0,
               "The relative SVD tolerance must be nonnegative, or -1 for "
               "the automatic default.");
   svd_relative_tolerance_ = tolerance;
   assembled_ = false;
}

/// Replace the fine-grid operator while retaining the current coarse vectors.
void TwoLevelPreconditioner::SetOperator(const Operator &op)
{
   ValidateOperator(op, "Two-level operator");
   operator_ = &op;
   assembled_ = false;
}

/// Sum a small coefficient vector over the configured communicator.
void TwoLevelPreconditioner::Reduce(Vector &values) const
{
#ifdef MFEM_USE_MPI
   if (use_global_inner_products_ && values.Size() > 0)
   {
      MPI_Allreduce(MPI_IN_PLACE, values.HostReadWrite(), values.Size(),
                    MFEM_MPI_REAL_T, MPI_SUM, communicator_);
   }
#else
   (void)values;
#endif
}

/// Sum a small dense matrix over the configured communicator.
void TwoLevelPreconditioner::Reduce(DenseMatrix &values) const
{
#ifdef MFEM_USE_MPI
   if (use_global_inner_products_ && values.TotalSize() > 0)
   {
      MPI_Allreduce(MPI_IN_PLACE, values.HostReadWrite(), values.TotalSize(),
                    MFEM_MPI_REAL_T, MPI_SUM, communicator_);
   }
#else
   (void)values;
#endif
}

/// Project a distributed fine-grid vector onto a distributed basis.
void TwoLevelPreconditioner::Project(const std::vector<Vector> &basis,
                                     const Vector &input,
                                     Vector &coefficients) const
{
   coefficients.SetSize(num_coarse_vectors_);
   for (int i = 0; i < num_coarse_vectors_; ++i)
   {
      coefficients(i) = InnerProduct(basis[i], input);
   }
   Reduce(coefficients);
}

/// Form Z^T first_input - (A Z)^T second_input with one reduction.
void TwoLevelPreconditioner::ProjectDifference(
   const Vector &first_input, const Vector &second_input,
   Vector &coefficients) const
{
   coefficients.SetSize(num_coarse_vectors_);
   for (int i = 0; i < num_coarse_vectors_; ++i)
   {
      coefficients(i) = InnerProduct(coarse_vectors_[i], first_input)
                        - InnerProduct(operator_coarse_vectors_[i],
                                       second_input);
   }
   Reduce(coefficients);
}

/// Expand a replicated coarse coefficient vector in a distributed basis.
void TwoLevelPreconditioner::Combine(const std::vector<Vector> &basis,
                                     const Vector &coefficients,
                                     Vector &result) const
{
   result.SetSize(Height());
   result.UseDevice(true);
   result = 0.0;
   for (int i = 0; i < num_coarse_vectors_; ++i)
   {
      result.Add(coefficients(i), basis[i]);
   }
}

/// Rebuild the projected operator and its SVD pseudoinverse.
void TwoLevelPreconditioner::Assemble() const
{
   const int coarse_size = num_coarse_vectors_;
   operator_coarse_vectors_.resize(coarse_size);
   coarse_pseudoinverse_.SetSize(coarse_size, coarse_size);

   if (coarse_size == 0)
   {
      assembled_ = true;
      return;
   }

   for (int j = 0; j < coarse_size; ++j)
   {
      Vector &operator_vector = operator_coarse_vectors_[j];
      operator_vector.SetSize(Height());
      operator_vector.UseDevice(true);
      operator_->Mult(coarse_vectors_[j], operator_vector);
   }

   DenseMatrix reduced_operator(coarse_size);
   for (int j = 0; j < coarse_size; ++j)
   {
      for (int i = 0; i < coarse_size; ++i)
      {
         reduced_operator(i, j) =
            InnerProduct(coarse_vectors_[i],
                         operator_coarse_vectors_[j]);
      }
   }
   Reduce(reduced_operator);

#ifdef MFEM_USE_LAPACK
   DenseMatrixSVD svd(reduced_operator, 'S', 'S');
   svd.Eval(reduced_operator);
   const Vector &singular_values = svd.Singularvalues();
   const real_t relative_tolerance = svd_relative_tolerance_ >= 0.0 ?
                                     svd_relative_tolerance_ :
                                     coarse_size*
                                     std::numeric_limits<real_t>::epsilon();
   const real_t cutoff = relative_tolerance*singular_values(0);
   const DenseMatrix &left = svd.LeftSingularvectors();
   const DenseMatrix &right_transpose = svd.RightSingularvectors();

   coarse_pseudoinverse_ = 0.0;
   for (int k = 0; k < coarse_size; ++k)
   {
      if (singular_values(k) <= cutoff) { continue; }
      const real_t inverse_singular_value = 1.0/singular_values(k);
      for (int j = 0; j < coarse_size; ++j)
      {
         for (int i = 0; i < coarse_size; ++i)
         {
            coarse_pseudoinverse_(i, j) +=
               right_transpose(k, i)*inverse_singular_value*left(j, k);
         }
      }
   }
#else
   MFEM_ABORT("TwoLevelPreconditioner::Assemble requires LAPACK support.");
#endif

   assembled_ = true;
}

/// Assemble stale coarse data on first use after a configuration change.
void TwoLevelPreconditioner::EnsureAssembled() const
{
   if (!assembled_) { Assemble(); }
}

/// Apply Q or Q^T to a fine-grid vector.
void TwoLevelPreconditioner::ApplyCoarse(const Vector &input,
                                         Vector &result,
                                         const bool transpose) const
{
   if (num_coarse_vectors_ == 0)
   {
      coarse_rhs_.SetSize(0);
      coarse_solution_.SetSize(0);
      result.SetSize(Height());
      result.UseDevice(true);
      result = 0.0;
      return;
   }

   Project(coarse_vectors_, input, coarse_rhs_);
   coarse_solution_.SetSize(num_coarse_vectors_);
   if (transpose)
   {
      coarse_pseudoinverse_.MultTranspose(coarse_rhs_, coarse_solution_);
   }
   else
   {
      coarse_pseudoinverse_.Mult(coarse_rhs_, coarse_solution_);
   }
   Combine(coarse_vectors_, coarse_solution_, result);
}

/// Apply the coarse inverse independently of the configured smoothers.
void TwoLevelPreconditioner::MultCoarse(const Vector &b, Vector &x) const
{
   ValidateVector(b);
   EnsureAssembled();
   ApplyCoarse(b, x, false);
}

/// Apply P = I-AQ, reusing the cached columns A Z.
void TwoLevelPreconditioner::MultLeftDeflation(const Vector &b,
                                               Vector &x) const
{
   ValidateVector(b);
   EnsureAssembled();

   residual_ = b;
   ApplyCoarse(b, coarse_correction_, false);
   Combine(operator_coarse_vectors_, coarse_solution_, operator_work_);
   residual_ -= operator_work_;
   x = residual_;
}

/// Apply P^T = I-Q^T A^T.
void TwoLevelPreconditioner::MultRightDeflation(const Vector &b,
                                                Vector &x) const
{
   ValidateVector(b);
   EnsureAssembled();

   residual_ = b;
   operator_->MultTranspose(b, operator_work_);
   ApplyCoarse(operator_work_, coarse_correction_, true);
   residual_ -= coarse_correction_;
   x = residual_;
}

/// Apply P A = A-AQA for a deflated Krylov solve.
void TwoLevelPreconditioner::MultDeflatedOperator(const Vector &b,
                                                  Vector &x) const
{
   ValidateVector(b);
   EnsureAssembled();

   operator_->Mult(b, residual_);
   ApplyCoarse(residual_, coarse_correction_, false);
   Combine(operator_coarse_vectors_, coarse_solution_, operator_work_);
   residual_ -= operator_work_;
   x = residual_;
}

/// Add the coarse solution Qb to the projected complementary solution.
void TwoLevelPreconditioner::RecoverDeflatedSolution(
   const Vector &b, const Vector &x_hat, Vector &x) const
{
   ValidateVector(b);
   ValidateVector(x_hat);
   EnsureAssembled();

   ApplyCoarse(b, pre_smoothed_, false);
   MultRightDeflation(x_hat, post_smoothed_);
   residual_ = pre_smoothed_;
   residual_ += post_smoothed_;
   x = residual_;
}

/// Apply the configured post-smoothing action.
void TwoLevelPreconditioner::ApplyPostSmoother(const Vector &input,
                                               Vector &result) const
{
   MFEM_ASSERT(post_smoother_, "The post-smoother is not configured.");
   if (post_smoother_uses_transpose_)
   {
      post_smoother_->MultTranspose(input, result);
   }
   else
   {
      post_smoother_->Mult(input, result);
   }
}

/// Apply the transpose of the configured post-smoothing action.
void TwoLevelPreconditioner::ApplyPostSmootherTranspose(
   const Vector &input, Vector &result) const
{
   MFEM_ASSERT(post_smoother_, "The post-smoother is not configured.");
   if (post_smoother_uses_transpose_)
   {
      post_smoother_->Mult(input, result);
   }
   else
   {
      post_smoother_->MultTranspose(input, result);
   }
}

/// Apply a coarse correction with optional symmetric pre/post smoothing.
void TwoLevelPreconditioner::Mult(const Vector &b, Vector &x) const
{
   ValidateVector(b);
   EnsureAssembled();

   if (!pre_smoother_ && !post_smoother_)
   {
      ApplyCoarse(b, x, false);
      return;
   }

   pre_smoothed_ = 0.0;
   residual_ = b;
   if (pre_smoother_)
   {
      pre_smoother_->Mult(b, pre_smoothed_);
      operator_->Mult(pre_smoothed_, operator_work_);
      residual_ -= operator_work_;
   }

   ApplyCoarse(residual_, coarse_correction_, false);
   Combine(operator_coarse_vectors_, coarse_solution_, operator_work_);
   residual_ -= operator_work_;

   x.SetSize(Height());
   x.UseDevice(true);
   x = pre_smoothed_;
   x += coarse_correction_;
   if (post_smoother_)
   {
      ApplyPostSmoother(residual_, post_smoothed_);
      x += post_smoothed_;
   }
}

/// Apply the exact transpose of the multiplicative two-level cycle.
void TwoLevelPreconditioner::MultTranspose(const Vector &b, Vector &x) const
{
   ValidateVector(b);
   EnsureAssembled();

   if (!pre_smoother_ && !post_smoother_)
   {
      ApplyCoarse(b, x, true);
      return;
   }

   // Let t=R^T b. Then
   // r_bar=t+Q^T(b-A^T t) and B^T b=r_bar+L^T(b-A^T r_bar).
   // Z^T A^T t=(A Z)^T t avoids the first explicit A^T application.
   pre_smoothed_ = 0.0;
   if (post_smoother_)
   {
      ApplyPostSmootherTranspose(b, pre_smoothed_);
   }
   if (num_coarse_vectors_ == 0)
   {
      coarse_correction_.SetSize(Height());
      coarse_correction_ = 0.0;
   }
   else
   {
      ProjectDifference(b, pre_smoothed_, coarse_rhs_);
      coarse_solution_.SetSize(num_coarse_vectors_);
      coarse_pseudoinverse_.MultTranspose(coarse_rhs_, coarse_solution_);
      Combine(coarse_vectors_, coarse_solution_, coarse_correction_);
   }
   residual_ = pre_smoothed_;
   residual_ += coarse_correction_;

   x.SetSize(Height());
   x.UseDevice(true);
   x = residual_;
   if (pre_smoother_)
   {
      operator_->MultTranspose(residual_, operator_work_);
      operator_work_ *= -1.0;
      operator_work_ += b;
      pre_smoother_->MultTranspose(operator_work_, post_smoothed_);
      x += post_smoothed_;
   }
}

} // namespace mfem
