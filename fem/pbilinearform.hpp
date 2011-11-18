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

/// Class for parallel bilinear form
class ParBilinearForm : public BilinearForm
{
protected:
   ParFiniteElementSpace *pfes;

   HypreParMatrix *ParallelAssemble(SparseMatrix *m);

public:
   ParBilinearForm(ParFiniteElementSpace *pf)
      : BilinearForm(pf) { pfes = pf; }

   ParBilinearForm(ParFiniteElementSpace *pf, ParBilinearForm *bf)
      : BilinearForm(pf, bf) { pfes = pf; }

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   HypreParMatrix *ParallelAssemble() { return ParallelAssemble(mat); }

   /// Returns the eliminated matrix assembled on the true dofs, i.e. P^t A_e P.
   HypreParMatrix *ParallelAssembleElim() { return ParallelAssemble(mat_e); }

   virtual ~ParBilinearForm() { }
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

#endif
