// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PBILINEARFORM
#define MFEM_PBILINEARFORM

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <mpi.h>
#include "pfespace.hpp"
#include "pgridfunc.hpp"
#include "bilinearform.hpp"

namespace mfem
{

/// Class for parallel bilinear form
class ParBilinearForm : public BilinearForm
{
   friend FABilinearFormExtension;
protected:
   ParFiniteElementSpace *pfes; ///< Points to the same object as #fes

   /// Auxiliary vectors used in TrueAddMult(): L-, L-, and T-vector, resp.
   mutable Vector Xaux, Yaux, Ytmp;

   OperatorHandle p_mat, p_mat_e;

   bool keep_nbr_block;

   // Allocate mat - called when (mat == NULL && fbfi.Size() > 0)
   void pAllocMat();

   void AssembleSharedFaces(int skip_zeros = 1);

private:
   /// Copy construction is not supported; body is undefined.
   ParBilinearForm(const ParBilinearForm &);

   /// Copy assignment is not supported; body is undefined.
   ParBilinearForm &operator=(const ParBilinearForm &);

public:
   /// Creates parallel bilinear form associated with the FE space @a *pf.
   /** The pointer @a pf is not owned by the newly constructed object. */
   ParBilinearForm(ParFiniteElementSpace *pf)
      : BilinearForm(pf), pfes(pf),
        p_mat(Operator::Hypre_ParCSR), p_mat_e(Operator::Hypre_ParCSR)
   { keep_nbr_block = false; }

   /** @brief Create a ParBilinearForm on the ParFiniteElementSpace @a *pf,
       using the same integrators as the ParBilinearForm @a *bf.

       The pointer @a pf is not owned by the newly constructed object.

       The integrators in @a bf are copied as pointers and they are not owned by
       the newly constructed ParBilinearForm. */
   ParBilinearForm(ParFiniteElementSpace *pf, ParBilinearForm *bf)
      : BilinearForm(pf, bf), pfes(pf),
        p_mat(Operator::Hypre_ParCSR), p_mat_e(Operator::Hypre_ParCSR)
   { keep_nbr_block = false; }

   /** When set to true and the ParBilinearForm has interior face integrators,
       the local SparseMatrix will include the rows (in addition to the columns)
       corresponding to face-neighbor dofs. The default behavior is to disregard
       those rows. Must be called before the first Assemble call. */
   void KeepNbrBlock(bool knb = true) { keep_nbr_block = knb; }

   /** @brief Set the operator type id for the parallel matrix/operator when
       using AssemblyLevel::LEGACY. */
   /** If using static condensation or hybridization, call this method *after*
       enabling it. */
   void SetOperatorType(Operator::Type tid)
   {
      p_mat.SetType(tid); p_mat_e.SetType(tid);
      if (hybridization) { hybridization->SetOperatorType(tid); }
      if (static_cond) { static_cond->SetOperatorType(tid); }
   }

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   /** @brief Assemble the diagonal of the bilinear form into @a diag. Note that
       @a diag is a true-dof Vector.

       When the AssemblyLevel is not LEGACY, and the mesh is nonconforming,
       this method returns |P^T| d_l, where d_l is the local diagonal of the
       form before applying parallel/conforming assembly, P^T is the transpose
       of the parallel/conforming prolongation, and |.| denotes the entry-wise
       absolute value. In general, this is just an approximation of the exact
       diagonal for this case. */
   void AssembleDiagonal(Vector &diag) const override;

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   HypreParMatrix *ParallelAssemble() { return ParallelAssemble(mat); }

   /// Returns the eliminated matrix assembled on the true dofs, i.e. P^t A_e P.
   /** The returned matrix has to be deleted by the caller. */
   HypreParMatrix *ParallelAssembleElim() { return ParallelAssemble(mat_e); }

   /// Return the matrix @a m assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   HypreParMatrix *ParallelAssemble(SparseMatrix *m);

   /** @brief Compute parallel RAP operator and store it in @a A as a HypreParMatrix.

       @param[in] loc_A The rank-local `SparseMatrix`.
       @param[out] A The `OperatorHandle` containing the global `HypreParMatrix`.
       @param[in] steal_loc_A Have the `HypreParMatrix` in @a A take ownership of
                              the memory objects in @a loc_A.
       */
   void ParallelRAP(SparseMatrix &loc_A,
                    OperatorHandle &A,
                    bool steal_loc_A = false);

   /** @brief Returns the matrix assembled on the true dofs, i.e.
       @a A = P^t A_local P, in the format (type id) specified by @a A. */
   void ParallelAssemble(OperatorHandle &A) { ParallelAssemble(A, mat); }

   /** Returns the eliminated matrix assembled on the true dofs, i.e.
       @a A_elim = P^t A_elim_local P in the format (type id) specified by @a A.
    */
   void ParallelAssembleElim(OperatorHandle &A_elim)
   { ParallelAssemble(A_elim, mat_e); }

   /** Returns the matrix @a A_local assembled on the true dofs, i.e.
       @a A = P^t A_local P in the format (type id) specified by @a A. */
   void ParallelAssemble(OperatorHandle &A, SparseMatrix *A_local);

   /// Eliminate essential boundary DOFs from a parallel assembled system.
   /** The array @a bdr_attr_is_ess marks boundary attributes that constitute
       the essential part of the boundary. */
   void ParallelEliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                                     HypreParMatrix &A,
                                     const HypreParVector &X,
                                     HypreParVector &B) const;

   /// Eliminate essential boundary DOFs from a parallel assembled matrix @a A.
   /** The array @a bdr_attr_is_ess marks boundary attributes that constitute
       the essential part of the boundary. The eliminated part is stored in a
       matrix A_elim such that A_original = A_new + A_elim. Returns a pointer to
       the newly allocated matrix A_elim which should be deleted by the caller.
       The matrices @a A and A_elim can be used to eliminate boundary conditions
       in multiple right-hand sides, by calling the function EliminateBC() (from
       hypre.hpp). */
   HypreParMatrix *ParallelEliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                                                HypreParMatrix &A) const;

   /// Eliminate essential true DOFs from a parallel assembled matrix @a A.
   /** Given a list of essential true dofs and the parallel assembled matrix
       @a A, eliminate the true dofs from the matrix, storing the eliminated
       part in a matrix A_elim such that A_original = A_new + A_elim. Returns a
       pointer to the newly allocated matrix A_elim which should be deleted by
       the caller. The matrices @a A and A_elim can be used to eliminate
       boundary conditions in multiple right-hand sides, by calling the function
       EliminateBC() (from hypre.hpp). */
   HypreParMatrix *ParallelEliminateTDofs(const Array<int> &tdofs_list,
                                          HypreParMatrix &A) const
   { return A.EliminateRowsCols(tdofs_list); }

   /** @brief Compute @a y += @a a (P^t A P) @a x, where @a x and @a y are
       vectors on the true dofs. */
   void TrueAddMult(const Vector &x, Vector &y, const real_t a = 1.0) const;

   /// Compute $ y^T M x $
   /** @warning The calculation is performed on local dofs, assuming that
       the local vectors are consistent with the prolongations of the true
       vectors (see ParGridFunction::Distribute()). If this is not the case,
       use TrueInnerProduct(const ParGridFunction &, const ParGridFunction &)
       instead.
       @note It is assumed that the local matrix is assembled and it has
       not been replaced by the parallel matrix through FormSystemMatrix().
       @see TrueInnerProduct(const ParGridFunction&, const ParGridFunction&) */
   real_t ParInnerProduct(const ParGridFunction &x,
                          const ParGridFunction &y) const;

   /// Compute $ y^T M x $ on true dofs (grid function version)
   /** @note The ParGridFunction%s are restricted to the true-vectors for
       for calculation.
       @note It is assumed that the parallel system matrix is assembled,
       see FormSystemMatrix().
       @see ParInnerProduct(const ParGridFunction&, const ParGridFunction&) */
   real_t TrueInnerProduct(const ParGridFunction &x,
                           const ParGridFunction &y) const;

   /// Compute $ y^T M x $ on true dofs (Hypre vector version)
   /** @note It is assumed that the parallel system matrix is assembled,
       see FormSystemMatrix(). */
   real_t TrueInnerProduct(HypreParVector &x, HypreParVector &y) const;

   /// Compute $ y^T M x $ on true dofs (true-vector version)
   /** @note It is assumed that the parallel system matrix is assembled,
       see FormSystemMatrix(). */
   real_t TrueInnerProduct(const Vector &x, const Vector &y) const;

   /// Return the parallel FE space associated with the ParBilinearForm.
   ParFiniteElementSpace *ParFESpace() const { return pfes; }

   /// Return the parallel trace FE space associated with static condensation.
   ParFiniteElementSpace *SCParFESpace() const
   { return static_cond ? static_cond->GetParTraceFESpace() : NULL; }

   /// Get the parallel finite element space prolongation matrix
   const Operator *GetProlongation() const override
   { return pfes->GetProlongationMatrix(); }
   /// Get the transpose of GetRestriction, useful for matrix-free RAP
   virtual const Operator *GetRestrictionTranspose() const
   { return pfes->GetRestrictionTransposeOperator(); }
   /// Get the parallel finite element space restriction matrix
   const Operator *GetRestriction() const override
   { return pfes->GetRestrictionMatrix(); }

   using BilinearForm::FormLinearSystem;
   using BilinearForm::FormSystemMatrix;

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x,
                         Vector &b, OperatorHandle &A, Vector &X,
                         Vector &B, int copy_interior = 0) override;

   void FormSystemMatrix(const Array<int> &ess_tdof_list,
                         OperatorHandle &A) override;

   /** Call this method after solving a linear system constructed using the
       FormLinearSystem method to recover the solution as a ParGridFunction-size
       vector in x. Use the same arguments as in the FormLinearSystem call. */
   void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x) override;

   void Update(FiniteElementSpace *nfes = NULL) override;

   void EliminateVDofsInRHS(const Array<int> &vdofs, const Vector &x, Vector &b);

   virtual ~ParBilinearForm() { }
};

/// Class for parallel bilinear form using different test and trial FE spaces.
class ParMixedBilinearForm : public MixedBilinearForm
{
protected:
   /// Points to the same object as #trial_fes
   ParFiniteElementSpace *trial_pfes;
   /// Points to the same object as #test_fes
   ParFiniteElementSpace *test_pfes;
   /// Auxiliary objects used in TrueAddMult().
   mutable ParGridFunction Xaux, Yaux;

   /// Matrix and eliminated matrix
   OperatorHandle p_mat, p_mat_e;

private:
   /// Copy construction is not supported; body is undefined.
   ParMixedBilinearForm(const ParMixedBilinearForm &);

   /// Copy assignment is not supported; body is undefined.
   ParMixedBilinearForm &operator=(const ParMixedBilinearForm &);

public:
   /** @brief Construct a ParMixedBilinearForm on the given FiniteElementSpace%s
       @a trial_fes and @a test_fes. */
   /** The pointers @a trial_fes and @a test_fes are not owned by the newly
       constructed object. */
   ParMixedBilinearForm(ParFiniteElementSpace *trial_fes,
                        ParFiniteElementSpace *test_fes)
      : MixedBilinearForm(trial_fes, test_fes),
        p_mat(Operator::Hypre_ParCSR), p_mat_e(Operator::Hypre_ParCSR)
   {
      trial_pfes = trial_fes;
      test_pfes  = test_fes;
   }

   /** @brief Create a ParMixedBilinearForm on the given FiniteElementSpace%s
       @a trial_fes and @a test_fes, using the same integrators as the
       ParMixedBilinearForm @a mbf.

       The pointers @a trial_fes and @a test_fes are not owned by the newly
       constructed object.

       The integrators in @a mbf are copied as pointers and they are not owned
       by the newly constructed ParMixedBilinearForm. */
   ParMixedBilinearForm(ParFiniteElementSpace *trial_fes,
                        ParFiniteElementSpace *test_fes,
                        ParMixedBilinearForm * mbf)
      : MixedBilinearForm(trial_fes, test_fes, mbf),
        p_mat(Operator::Hypre_ParCSR), p_mat_e(Operator::Hypre_ParCSR)
   {
      trial_pfes = trial_fes;
      test_pfes  = test_fes;
   }

   /// Returns the matrix assembled on the true dofs, i.e. P_test^t A P_trial.
   HypreParMatrix *ParallelAssemble();

   /** @brief Returns the matrix assembled on the true dofs, i.e.
       @a A = P_test^t A_local P_trial, in the format (type id) specified by
       @a A. */
   void ParallelAssemble(OperatorHandle &A);

   using MixedBilinearForm::FormRectangularSystemMatrix;
   using MixedBilinearForm::FormRectangularLinearSystem;

   /** @brief Return in @a A a parallel (on truedofs) version of this operator.

       This returns the same operator as FormRectangularLinearSystem(), but does
       without the transformations of the right-hand side. */
   void FormRectangularSystemMatrix(const Array<int> &trial_tdof_list,
                                    const Array<int> &test_tdof_list,
                                    OperatorHandle &A) override;

   /** @brief Form the parallel linear system A X = B, corresponding to this mixed
       bilinear form and the linear form @a b(.).

       Return in @a A a *reference* to the system matrix that is column-constrained.
       The reference will be invalidated when SetOperatorType(), Update(), or the
       destructor is called. */
   void FormRectangularLinearSystem(const Array<int> &trial_tdof_list,
                                    const Array<int> &test_tdof_list, Vector &x,
                                    Vector &b, OperatorHandle &A, Vector &X,
                                    Vector &B) override;

   /// Compute y += a (P^t A P) x, where x and y are vectors on the true dofs
   void TrueAddMult(const Vector &x, Vector &y, const real_t a = 1.0) const;

   virtual ~ParMixedBilinearForm() { }
};

/** The parallel matrix representation a linear operator between parallel finite
    element spaces */
class ParDiscreteLinearOperator : public DiscreteLinearOperator
{
protected:
   /// Points to the same object as #trial_fes
   ParFiniteElementSpace *domain_fes;
   /// Points to the same object as #test_fes
   ParFiniteElementSpace *range_fes;

private:
   /// Copy construction is not supported; body is undefined.
   ParDiscreteLinearOperator(const ParDiscreteLinearOperator &);

   /// Copy assignment is not supported; body is undefined.
   ParDiscreteLinearOperator &operator=(const ParDiscreteLinearOperator &);

public:
   /** @brief Construct a ParDiscreteLinearOperator on the given
       FiniteElementSpace%s @a dfes (domain FE space) and @a rfes (range FE
       space). */
   /** The pointers @a dfes and @a rfes are not owned by the newly constructed
       object. */
   ParDiscreteLinearOperator(ParFiniteElementSpace *dfes,
                             ParFiniteElementSpace *rfes)
      : DiscreteLinearOperator(dfes, rfes) { domain_fes=dfes; range_fes=rfes; }

   /// Returns the matrix "assembled" on the true dofs
   HypreParMatrix *ParallelAssemble() const;

   /** @brief Returns the matrix assembled on the true dofs, i.e.
       @a A = R_test A_local P_trial, in the format (type id) specified by
       @a A. */
   void ParallelAssemble(OperatorHandle &A);

   /** Extract the parallel blocks corresponding to the vector dimensions of the
       domain and range parallel finite element spaces */
   void GetParBlocks(Array2D<HypreParMatrix *> &blocks) const;

   using MixedBilinearForm::FormRectangularSystemMatrix;

   /** @brief Return in @a A a parallel (on truedofs) version of this operator. */
   virtual void FormRectangularSystemMatrix(OperatorHandle &A);

   virtual ~ParDiscreteLinearOperator() { }
};

}

#endif // MFEM_USE_MPI

#endif
