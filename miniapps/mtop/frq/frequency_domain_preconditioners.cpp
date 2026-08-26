// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.

#include "frequency_domain_preconditioners.hpp"

#include <climits>

// This implementation depends only on MFEM's Operator and Solver interfaces.

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
/// Block vectors (input_, output_) are sized according to block_offsets_.
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
     block_offsets_(MakeTwoBlockOffsets(block_size_)),
     input_(block_offsets_),
     output_(block_offsets_),
     first_rhs_(block_size_),
     second_rhs_(block_size_),
     first_solution_(block_size_),
     second_solution_(block_size_),
     work_(block_size_)
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
   input_.UseDevice(true);
   output_.UseDevice(true);
   input_.SyncToBlocks();
   output_.SyncToBlocks();
   first_rhs_.UseDevice(true);
   second_rhs_.UseDevice(true);
   first_solution_.UseDevice(true);
   second_solution_.UseDevice(true);
   work_.UseDevice(true);

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

   // Only copy if input and output alias (in-place operation)
   const bool aliased = (&b == &x);

   // Create block views directly from input to avoid full vector copy
   const real_t *b_data = b.Read();
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

   input_.Set(1.0, b);
   input_.SyncToBlocks();
   H_inverse_->MultTranspose(input_.GetBlock(0), output_.GetBlock(0));
   H_inverse_->MultTranspose(input_.GetBlock(1), output_.GetBlock(1));
   output_.SyncFromBlocks();
   x.UseDevice(true);
   x = output_;
}

} // namespace mfem
