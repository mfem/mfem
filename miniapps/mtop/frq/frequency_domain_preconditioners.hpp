// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
//
// Problem-independent real-block preconditioners for systems obtained from
// (W + i T) x = b. The classes in this file are independent of the finite
// element discretization and of the way the inverse of H = W + T is built.
//
// References:
// [1] O. Axelsson, D. K. Salkuyeh, "A new version of a preconditioning
//     method for certain two-by-two block matrices with square blocks,"
//     BIT Numerical Mathematics 59 (2019), 321-342.
//     https://doi.org/10.1007/s10543-018-0741-x
// [2] O. Axelsson, J. Karatson, "Superior properties of the PRESB
//     preconditioner for operators on two-by-two block form with square
//     blocks," Numerische Mathematik 146 (2020), 335-368.
//     https://doi.org/10.1007/s00211-020-01143-x

#ifndef MFEM_MTOP_FREQUENCY_DOMAIN_PRECONDITIONERS_HPP
#define MFEM_MTOP_FREQUENCY_DOMAIN_PRECONDITIONERS_HPP

#include "mfem.hpp"

namespace mfem
{

/// Apply the preconditioned square-block (PRESB) inverse.
///
/// Consider a complex system
///
///     (W + i s T)(x_r + i x_i) = b_r + i b_i,  s in {-1,+1},
///
/// whose standard real form is
///
///     A_s [x_r; x_i] = [b_r; b_i],
///
///             [ W   -s T ]
///     A_s  =  [          ].
///             [ s T    W ]
///
/// PRESB replaces A_s by
///
///             [ W + 2 T  -s T ]
///     P_s  =  [               ].
///             [   s T       W ]
///
/// Set H = W+T. The action x=P_s^{-1}b is evaluated without assembling P_s:
///
///     H z   = b_r - s b_i,
///     H q   = s b_i - T z,
///     x_r   = z + q,
///     x_i   = s q.
///
/// Thus one application needs one multiplication by T and two applications
/// of the supplied H_inverse operator. W is not needed separately. H_inverse
/// may be an exact solver or a fixed approximate inverse such as one AMG
/// cycle. Both supplied operators are non-owning and must outlive this object.
///
/// @par Computational cost per application:
/// - 2 applications of H_inverse
/// - 1 multiplication by T
/// - O(n) vector operations
///
/// If W and T are symmetric positive semidefinite and H is positive definite,
/// exact inversion of H gives eigenvalues of P_s^{-1}A_s in [1/2,1]; see
/// References [1,2]. PRESB is generally paired with GMRES because P_s and A_s
/// are nonsymmetric. A variable H inverse requires a flexible outer method.
///
/// Input and output vectors use contiguous MFEM block ordering
/// [real block; imaginary block], with equally sized blocks.
///
/// @warning Thread safety: Internal mutable work vectors make one instance
/// unsafe for simultaneous calls from multiple host threads. Create separate
/// instances for concurrent use.
///
/// @warning GPU execution: Work vectors follow the device configuration set
/// at construction. For GPU execution, T and H_inverse must support device
/// operations. This preconditioner is NOT safe for concurrent use across
/// multiple GPU streams; create separate instances for each stream.
class PRESBPreconditioner : public Solver
{
public:
   /// Construct the inverse action for P_s.
   ///
   /// @param T Square coupling operator in the imaginary part of the complex
   /// system. It must have the same size as @a H_inverse.
   /// @param H_inverse Exact or approximate inverse action for H=W+T.
   /// @param imaginary_sign Sign s of the imaginary term; it must be +1 for
   /// W+iT or -1 for W-iT.
   PRESBPreconditioner(const Operator &T, const Operator &H_inverse,
                       int imaginary_sign = 1);

   /// Compute @a x = P_s^{-1} @a b using the four-step algorithm above.
   ///
   /// The input may alias the output. Both vectors have size 2n, where n is
   /// the order of T and H_inverse.
   void Mult(const Vector &b, Vector &x) const override;

   /// Compute @a x = P_s^{-T} @a b.
   ///
   /// This operation calls MultTranspose() on both T and H_inverse. It is
   /// available for algorithms that explicitly require the transpose of the
   /// preconditioner; an adjoint complex system should normally instead use a
   /// PRESBPreconditioner constructed with imaginary_sign=-1.
   void MultTranspose(const Vector &b, Vector &x) const override;

   /// Validate the dimensions of a real two-by-two block system operator.
   ///
   /// The constituent T and H_inverse operators are fixed by
   /// the constructor and are not inferred from @a op.
   void SetOperator(const Operator &op) override;

   /// Return +1 for W+iT or -1 for W-iT.
   int GetImaginarySign() const { return imaginary_sign_; }

   /// Return the non-owning coupling operator T.
   ///
   /// The returned operator is valid only while this preconditioner exists and
   /// the original T operator passed to the constructor remains valid.
   const Operator &GetCouplingOperator() const { return *T_; }

   /// Return the non-owning exact or approximate inverse action for H=W+T.
   ///
   /// The returned operator is valid only while this preconditioner exists and
   /// the original H_inverse operator passed to the constructor remains valid.
   const Operator &GetDiagonalInverse() const { return *H_inverse_; }

private:
   /// Verify that @a b has the required two-block size.
   void ValidateInput(const Vector &b) const;

   const Operator *T_;
   const Operator *H_inverse_;
   int imaginary_sign_;
   int block_size_;

   mutable Vector first_rhs_;
   mutable Vector second_rhs_;
   mutable Vector first_solution_;
   mutable Vector second_solution_;
};

/// Apply a block-diagonal inverse to the symmetric real formulation.
///
/// For the complex system
///
///     (W + i s T)(x_r + i x_i) = b_r + i b_i,  s in {-1,+1},
///
/// Two equivalent symmetric formulations are obtained with
/// D=diag(I,-I):
///
///     D A_s = [ W  -sT ]       and       A_s D = [ W   sT ].
///             [-sT  -W ]                         [ sT  -W ]
///
/// MFEM's BLOCK_SYMMETRIC convention uses D A_s, transforms the right-hand
/// side to [b_r;-b_i], and leaves [x_r;x_i] unchanged. The right-scaled form
/// instead solves for [x_r;-x_i] with the original right-hand side. Both use
/// P_BD=diag(H,H), H=W+T. This inverse action is independent of s; the caller
/// is responsible for the transformation associated with its chosen form.
///
/// This class applies P_BD^{-1} = diag(H^{-1},H^{-1}). It does not construct
/// W, T, or H. The supplied exact or approximate H_inverse remains owned by
/// the caller and must outlive this object.
///
/// @par Computational cost per application:
/// - 2 independent applications of H_inverse (can be parallelized)
/// - O(n) vector copy operations
///
/// If W and T are symmetric positive semidefinite and H is positive definite,
/// the spectrum of P_BD^{-1}A is contained in
/// [-1,-1/sqrt(2)] union [1/sqrt(2),1]; see Reference [2]. The symmetric
/// formulation can therefore use MINRES, provided H_inverse is itself a fixed,
/// symmetric positive-definite operation.
///
/// The name deliberately differs from mfem::BlockDiagonalPreconditioner,
/// which is a general container for unrelated diagonal blocks.
///
/// @warning Thread safety: Internal mutable work vectors make one instance
/// unsafe for simultaneous calls from multiple host threads. Create separate
/// instances for concurrent use.
///
/// @warning GPU execution: Work vectors follow the device configuration set
/// at construction. For GPU execution, H_inverse must support device
/// operations. This preconditioner is NOT safe for concurrent use across
/// multiple GPU streams; create separate instances for each stream.
class RealBlockDiagonalPreconditioner : public Solver
{
public:
   /// Construct diag(H^{-1},H^{-1}) from a square inverse action.
   ///
   /// @param H_inverse Exact or approximate inverse action for H=W+T. The
   /// operator is non-owning and must remain valid for this object's lifetime.
   explicit RealBlockDiagonalPreconditioner(const Operator &H_inverse);

   /// Compute @a x = diag(H^{-1},H^{-1}) @a b.
   ///
   /// The input may alias the output. Both vectors contain two contiguous,
   /// equally sized blocks.
   void Mult(const Vector &b, Vector &x) const override;

   /// Compute @a x = diag(H^{-T},H^{-T}) @a b.
   ///
   /// This operation calls MultTranspose() on H_inverse.
   void MultTranspose(const Vector &b, Vector &x) const override;

   /// Validate the dimensions of a real two-by-two block system operator.
   ///
   /// H_inverse is fixed by the constructor and is not inferred from @a op.
   void SetOperator(const Operator &op) override;

   /// Return the non-owning exact or approximate inverse action for H=W+T.
   ///
   /// The returned operator is valid only while this preconditioner exists and
   /// the original H_inverse operator passed to the constructor remains valid.
   const Operator &GetDiagonalInverse() const { return *H_inverse_; }

private:
   /// Verify that @a b has the required two-block size.
   void ValidateInput(const Vector &b) const;

   const Operator *H_inverse_;
   int block_size_;
   Array<int> block_offsets_;
   mutable BlockVector input_;
   mutable BlockVector output_;
};

} // namespace mfem

#endif
