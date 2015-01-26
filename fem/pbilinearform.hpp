// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
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
#include "bilinearform.hpp"

namespace mfem
{

/// Class for parallel bilinear form
class ParBilinearForm : public BilinearForm
{
protected:
   ParFiniteElementSpace *pfes;
   mutable ParGridFunction X, Y; // used in TrueAddMult

   bool keep_nbr_block;

   // called when (mat == NULL && fbfi.Size() > 0)
   void pAllocMat();

   void AssembleSharedFaces(int skip_zeros = 1);

public:
   ParBilinearForm(ParFiniteElementSpace *pf)
      : BilinearForm(pf), pfes(pf)
   { keep_nbr_block = false; }

   ParBilinearForm(ParFiniteElementSpace *pf, ParBilinearForm *bf)
      : BilinearForm(pf, bf) { pfes = pf; keep_nbr_block = false; }

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

   /// Compute y += a (P^t A P) x, where x and y are vectors on the true dofs
   void TrueAddMult(const Vector &x, Vector &y, const double a = 1.0) const;

   ParFiniteElementSpace *ParFESpace() const { return pfes; }

   virtual ~ParBilinearForm() { }
};

/// Class for parallel bilinear form
class ParMixedBilinearForm : public MixedBilinearForm
{
protected:
   ParFiniteElementSpace *trial_pfes;
   ParFiniteElementSpace *test_pfes;

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

   virtual ~ParMixedBilinearForm() { }
};

/** The parallel matrix representation a linear operator between parallel finite
    element spaces */
class ParDiscreteLinearOperator : public DiscreteLinearOperator
{
protected:
   ParFiniteElementSpace *domain_fes;
   ParFiniteElementSpace *range_fes;

   HypreParMatrix *ParallelAssemble(SparseMatrix *m);

public:
   ParDiscreteLinearOperator(ParFiniteElementSpace *dfes,
                             ParFiniteElementSpace *rfes)
      : DiscreteLinearOperator(dfes, rfes) { domain_fes=dfes; range_fes=rfes; }

   /// Returns the matrix "assembled" on the true dofs
   HypreParMatrix *ParallelAssemble() { return ParallelAssemble(mat); }

   /** Extract the parallel blocks corresponding to the vector dimensions of the
       domain and range parallel finite element spaces */
   void GetParBlocks(Array2D<HypreParMatrix *> &blocks) const;

   virtual ~ParDiscreteLinearOperator() { }
};

}

#endif // MFEM_USE_MPI

#endif
