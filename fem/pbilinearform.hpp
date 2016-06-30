// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_PBILINEARFORM
#define MFEM_PBILINEARFORM

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <mpi.h>
#include "../linalg/hypre.hpp"
#include "pfespace.hpp"
#include "pgridfunc.hpp"
#include "bilinearform.hpp"

namespace mfem
{

/// Class for parallel bilinear form
class ParBilinearForm : public BilinearForm
{
protected:
   ParFiniteElementSpace *pfes;
   mutable ParGridFunction X, Y; // used in TrueAddMult

   HypreParMatrix *p_mat, *p_mat_e;

   bool keep_nbr_block;

   // called when (mat == NULL && fbfi.Size() > 0)
   void pAllocMat();

   void AssembleSharedFaces(int skip_zeros = 1);

public:
   ParBilinearForm(ParFiniteElementSpace *pf)
      : BilinearForm(pf), pfes(pf), p_mat(NULL), p_mat_e(NULL)
   { keep_nbr_block = false; }

   ParBilinearForm(ParFiniteElementSpace *pf, ParBilinearForm *bf)
      : BilinearForm(pf, bf), pfes(pf), p_mat(NULL), p_mat_e(NULL)
   { keep_nbr_block = false; }

   /** When set to true and the ParBilinearForm has interior face integrators,
       the local SparseMatrix will include the rows (in addition to the columns)
       corresponding to face-neighbor dofs. The default behavior is to disregard
       those rows. Must be called before the first Assemble call. */
   void KeepNbrBlock(bool knb = true) { keep_nbr_block = knb; }

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   HypreParMatrix *ParallelAssemble() { return ParallelAssemble(mat); }

   /// Returns the eliminated matrix assembled on the true dofs, i.e. P^t A_e P.
   HypreParMatrix *ParallelAssembleElim() { return ParallelAssemble(mat_e); }

   /// Return the matrix m assembled on the true dofs, i.e. P^t A P
   HypreParMatrix *ParallelAssemble(SparseMatrix *m);

   /** Eliminate essential boundary DOFs from a parallel assembled system.
       The array 'bdr_attr_is_ess' marks boundary attributes that constitute
       the essential part of the boundary. */
   void ParallelEliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                                     HypreParMatrix &A,
                                     const HypreParVector &X,
                                     HypreParVector &B) const;

   /** Eliminate essential boundary DOFs from a parallel assembled matrix A.
       The array 'bdr_attr_is_ess' marks boundary attributes that constitute the
       essential part of the boundary. The eliminated part is stored in a matrix
       A_elim such that A_new = A_original + A_elim. Returns a pointer to the
       newly allocated matrix A_elim which should be deleted by the caller. The
       matrices A and A_elim can be used to eliminate boundary conditions in
       multiple right-hand sides, by calling the function EliminateBC (from
       hypre.hpp).*/
   HypreParMatrix *ParallelEliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                                                HypreParMatrix &A) const;

   /** Given a list of essential true dofs and the parallel assembled matrix A,
       eliminate the true dofs from the matrix storing the eliminated part in a
       matrix A_elim such that A_new = A_original + A_elim. Returns a pointer to
       the newly allocated matrix A_elim which should be deleted by the
       caller. The matrices A and A_elim can be used to eliminate boundary
       conditions in multiple right-hand sides, by calling the function
       EliminateBC (from hypre.hpp). */
   HypreParMatrix *ParallelEliminateTDofs(const Array<int> &tdofs_list,
                                          HypreParMatrix &A) const
   { return A.EliminateRowsCols(tdofs_list); }

   /// Compute y += a (P^t A P) x, where x and y are vectors on the true dofs
   void TrueAddMult(const Vector &x, Vector &y, const double a = 1.0) const;

   /// Return the parallel FE space associated with the ParBilinearForm.
   ParFiniteElementSpace *ParFESpace() const { return pfes; }

   /// Return the parallel trace FE space associated with static condensation.
   ParFiniteElementSpace *SCParFESpace() const
   { return static_cond ? static_cond->GetParTraceFESpace() : NULL; }

   /** Form the linear system A X = B, corresponding to the current bilinear
       form and b(.), by applying any necessary transformations such as:
       eliminating boundary conditions; applying conforming constraints for
       non-conforming AMR; parallel assembly; static condensation;
       hybridization.

       The ParGridFunction-size vector x must contain the essential b.c. The
       ParBilinearForm and the ParLinearForm-size vector b must be assembled.

       The vector X is initialized with a suitable initial guess: when using
       hybridization, the vector X is set to zero; otherwise, the essential
       entries of X are set to the corresponding b.c. and all other entries are
       set to zero (copy_interior == 0) or copied from x (copy_interior != 0).

       This method can be called multiple times (with the same ess_tdof_list
       array) to initialize different right-hand sides and boundary condition
       values.

       After solving the linear system, the finite element solution x can be
       recovered by calling RecoverFEMSolution (with the same vectors X, b, and
       x). */
   void FormLinearSystem(Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         HypreParMatrix &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   /** Call this method after solving a linear system constructed using the
       FormLinearSystem method to recover the solution as a ParGridFunction-size
       vector in x. Use the same arguments as in the FormLinearSystem call. */
   void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   virtual void Update(FiniteElementSpace *nfes = NULL);

   virtual ~ParBilinearForm() { delete p_mat_e; delete p_mat; }
};

/// Class for parallel bilinear form
class ParMixedBilinearForm : public MixedBilinearForm
{
protected:
   ParFiniteElementSpace *trial_pfes;
   ParFiniteElementSpace *test_pfes;
   mutable ParGridFunction X, Y; // used in TrueAddMult

public:
   ParMixedBilinearForm(ParFiniteElementSpace *trial_fes,
                        ParFiniteElementSpace *test_fes)
      : MixedBilinearForm(trial_fes, test_fes)
   {
      trial_pfes = trial_fes;
      test_pfes  = test_fes;
   }

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   HypreParMatrix *ParallelAssemble();

   /// Compute y += a (P^t A P) x, where x and y are vectors on the true dofs
   void TrueAddMult(const Vector &x, Vector &y, const double a = 1.0) const;

   virtual ~ParMixedBilinearForm() { }
};

/** The parallel matrix representation a linear operator between parallel finite
    element spaces */
class ParDiscreteLinearOperator : public DiscreteLinearOperator
{
protected:
   ParFiniteElementSpace *domain_fes;
   ParFiniteElementSpace *range_fes;

public:
   ParDiscreteLinearOperator(ParFiniteElementSpace *dfes,
                             ParFiniteElementSpace *rfes)
      : DiscreteLinearOperator(dfes, rfes) { domain_fes=dfes; range_fes=rfes; }

   /// Returns the matrix "assembled" on the true dofs
   HypreParMatrix *ParallelAssemble() const;

   /** Extract the parallel blocks corresponding to the vector dimensions of the
       domain and range parallel finite element spaces */
   void GetParBlocks(Array2D<HypreParMatrix *> &blocks) const;

   virtual ~ParDiscreteLinearOperator() { }
};

}

#endif // MFEM_USE_MPI

#endif
