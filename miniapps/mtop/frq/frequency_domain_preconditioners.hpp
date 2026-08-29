// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
//
// Problem-independent preconditioners used by the frequency-domain miniapps.
// These classes are independent of the finite element discretization and
// include real-block inverses and a user-populated two-level coarse space.
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

#include <vector>

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

/// Apply a multiplicative two-level inverse with a user-managed coarse space.
///
/// Let the columns of Z be coarse vectors and define
///
///     E = Z^T A Z,             Q = Z E^\dagger Z^T,
///
/// where E^\dagger is formed with an SVD. With pre-smoother L and post-smoother
/// R, Mult() applies the multiplicative cycle
///
///     x = L b,
///     r = b - A x,
///     x = x + Q r,
///     r = r - A Q r,
///     x = x + R r.
///
/// Missing pre- or post-smoothers omit their respective steps. The cached
/// products A Z make the second residual update inexpensive. If neither
/// smoother is configured, Mult() applies only Q. In particular, an empty
/// coarse space with no smoothers produces the zero vector.
///
/// Coarse vectors occupy stable physical slots. AddCoarseVector() initially
/// fills slots [0,capacity), then overwrites them cyclically. SetCoarseVector()
/// replaces an occupied slot without changing the next insertion slot.
///
/// The operator, optional smoother, and communicator are non-owning and must
/// remain valid for this object's lifetime. SetPreSmoother() and
/// SetPostSmoother() configure independent actions whose Mult() methods are
/// used directly. SetSmoother(S) is the symmetric convenience operation: it
/// sets L=S and R=S^T. Passing nullptr to any setter disables the corresponding
/// action without changing or reassembling the coarse operator.
///
/// In the MPI constructor, every stored Vector is the rank-local part of a
/// global vector. All ranks must use the same capacity, coarse-space state,
/// SVD tolerance, and collective call order. Assemble(), Mult(),
/// MultTranspose(), and the explicit coarse and deflation operations perform
/// communicator-wide reductions. The serial constructor performs ordinary
/// local inner products.
///
/// Assembly is lazy after an operator, vector, or SVD-tolerance change. Calling
/// Assemble() explicitly always refreshes A Z, E, and E^\dagger. SVD assembly
/// requires an MFEM build with LAPACK.
///
/// @par Computational cost:
/// - assembly: one A application per active vector, one coarse-matrix global
///   reduction, and an O(m^3) dense SVD for coarse dimension m;
/// - fully smoothed application: one A application, one pre-smoother action,
///   one post-smoother action, one global reduction, and O(nm) coarse work;
/// - coarse-only application: one global reduction and O(nm) work.
///
/// The explicit deflation operations use the same Q without changing Mult().
/// With P=I-AQ, MultLeftDeflation() applies P, MultRightDeflation() applies
/// P^T, and MultDeflatedOperator() applies PA. A compatible deflated solve
/// PA*x_hat=P*b is recovered with RecoverDeflatedSolution(). For symmetric
/// positive-definite A and symmetric Q, PA=A-AQA is symmetric positive
/// semidefinite and its null space contains the coarse space.
///
/// @warning For the cycle to be symmetric, A and Q must be symmetric and the
/// post-smoother must be the transpose of the pre-smoother. SetSmoother()
/// establishes the latter relationship automatically.
///
/// @warning Thread safety: Lazy assembly and mutable work vectors make one
/// instance unsafe for simultaneous calls from multiple host threads.
///
/// @warning GPU execution: Fine-grid vectors may use device memory, but the
/// small reduced matrix and its SVD are processed on the host. MPI reductions
/// communicate host-resident coarse coefficients.
class TwoLevelPreconditioner : public Solver
{
public:
   /// Construct locally; @a smoother supplies L through Mult and R through
   /// MultTranspose.
   TwoLevelPreconditioner(const Operator &op, int max_coarse_vectors,
                          const Operator *smoother = nullptr);

#ifdef MFEM_USE_MPI
   /// Construct for distributed vectors; @a smoother supplies L through Mult
   /// and R through MultTranspose.
   TwoLevelPreconditioner(MPI_Comm communicator, const Operator &op,
                          int max_coarse_vectors,
                          const Operator *smoother = nullptr);
#endif

   /// Add a coarse vector and return the stable physical slot written.
   int AddCoarseVector(const Vector &vector);

   /// Replace one occupied coarse-vector slot.
   void SetCoarseVector(int slot, const Vector &vector);

   /// Copy one occupied rank-local coarse-vector slot into @a vector.
   void GetCoarseVector(int slot, Vector &vector) const;

   /// Return the number of occupied coarse-vector slots.
   int GetNumCoarseVectors() const { return num_coarse_vectors_; }

   /// Return the maximum number of coarse vectors retained.
   int GetMaxCoarseVectors() const { return max_coarse_vectors_; }

   /// Set or replace the non-owning smoother; nullptr disables smoothing.
   void SetSmoother(const Operator *smoother);

   /// Set or replace the non-owning smoother.
   void SetSmoother(const Operator &smoother) { SetSmoother(&smoother); }

   /// Set or disable the independent non-owning pre-smoother.
   void SetPreSmoother(const Operator *smoother);

   /// Set the independent non-owning pre-smoother.
   void SetPreSmoother(const Operator &smoother)
   { SetPreSmoother(&smoother); }

   /// Set or disable the independent non-owning post-smoother.
   ///
   /// Unlike SetSmoother(), this action is applied with Mult(), not
   /// MultTranspose().
   void SetPostSmoother(const Operator *smoother);

   /// Set the independent non-owning post-smoother.
   void SetPostSmoother(const Operator &smoother)
   { SetPostSmoother(&smoother); }

   /// Return the configured pre-smoother, or nullptr when disabled.
   const Operator *GetPreSmoother() const { return pre_smoother_; }

   /// Return the post-smoother's underlying operator, or nullptr when disabled.
   const Operator *GetPostSmoother() const { return post_smoother_; }

   /// Return whether the post step uses its operator's transpose action.
   bool PostSmootherUsesTranspose() const
   { return post_smoother_uses_transpose_; }

   /// Return the shared symmetric smoother, or nullptr for independent setup.
   const Operator *GetSmoother() const
   {
      return post_smoother_uses_transpose_ &&
             pre_smoother_ == post_smoother_ ? pre_smoother_ : nullptr;
   }

   /// Set the relative SVD cutoff. The value -1 restores the default.
   void SetSVDRelativeTolerance(real_t tolerance);

   /// Return the configured cutoff, or a negative value for the default.
   real_t GetSVDRelativeTolerance() const
   { return svd_relative_tolerance_; }

   /// Rebuild A Z, Z^T A Z, and the reduced pseudoinverse immediately.
   void Assemble() const;

   /// Apply the coarse inverse Q = Z (Z^T A Z)^\dagger Z^T.
   ///
   /// This action is independent of the configured smoothers. The input may
   /// alias the output.
   void MultCoarse(const Vector &b, Vector &x) const;

   /// Apply the left deflation projector P = I - A Q.
   ///
   /// This action is independent of the configured smoothers. The input may
   /// alias the output.
   void MultLeftDeflation(const Vector &b, Vector &x) const;

   /// Apply P^T = I - Q^T A^T.
   ///
   /// For symmetric A and Q this is the right deflation projector I-QA used
   /// to reconstruct a solution of a deflated system. The input may alias the
   /// output.
   void MultRightDeflation(const Vector &b, Vector &x) const;

   /// Apply the deflated operator P A = A - A Q A.
   ///
   /// For symmetric positive-definite A and a full-rank coarse basis this is
   /// symmetric positive semidefinite and annihilates the coarse space. It is
   /// intended for a compatible deflated CG solve, not as a preconditioner.
   void MultDeflatedOperator(const Vector &b, Vector &x) const;

   /// Form the compatible deflated right-hand side P b.
   void FormDeflatedRHS(const Vector &b, Vector &deflated_b) const
   { MultLeftDeflation(b, deflated_b); }

   /// Recover x = Q b + P^T x_hat after solving P A x_hat = P b.
   ///
   /// Deflated CG requires symmetric positive-definite A and symmetric Q.
   /// Both input vectors may alias @a x.
   void RecoverDeflatedSolution(const Vector &b, const Vector &x_hat,
                                 Vector &x) const;

   /// Apply the coarse-only or multiplicative two-level inverse action.
   void Mult(const Vector &b, Vector &x) const override;

   /// Apply the algebraic transpose of Mult().
   void MultTranspose(const Vector &b, Vector &x) const override;

   /// Replace A with a same-sized non-owning operator and invalidate assembly.
   void SetOperator(const Operator &op) override;

private:
   void ValidateOperator(const Operator &op, const char *name) const;
   void ValidateVector(const Vector &vector) const;
   void EnsureAssembled() const;
   void Reduce(Vector &values) const;
   void Reduce(DenseMatrix &values) const;
   void Project(const std::vector<Vector> &basis, const Vector &input,
                Vector &coefficients) const;
   void ProjectDifference(const Vector &first_input,
                          const Vector &second_input,
                          Vector &coefficients) const;
   void Combine(const std::vector<Vector> &basis,
                const Vector &coefficients, Vector &result) const;
   void ApplyCoarse(const Vector &input, Vector &result,
                    bool transpose) const;
   void ApplyPostSmoother(const Vector &input, Vector &result) const;
   void ApplyPostSmootherTranspose(const Vector &input,
                                   Vector &result) const;

   const Operator *operator_;
   const Operator *pre_smoother_;
   const Operator *post_smoother_;
   bool post_smoother_uses_transpose_ = false;
   int max_coarse_vectors_;
   int num_coarse_vectors_ = 0;
   int next_coarse_slot_ = 0;
   real_t svd_relative_tolerance_ = -1.0;
   std::vector<Vector> coarse_vectors_;

#ifdef MFEM_USE_MPI
   MPI_Comm communicator_ = MPI_COMM_NULL;
   bool use_global_inner_products_ = false;
#endif

   mutable bool assembled_ = false;
   mutable std::vector<Vector> operator_coarse_vectors_;
   mutable DenseMatrix coarse_pseudoinverse_;
   mutable Vector coarse_rhs_;
   mutable Vector coarse_solution_;
   mutable Vector pre_smoothed_;
   mutable Vector residual_;
   mutable Vector coarse_correction_;
   mutable Vector operator_work_;
   mutable Vector post_smoothed_;
};

} // namespace mfem

#endif
