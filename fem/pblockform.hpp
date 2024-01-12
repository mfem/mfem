// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PBLOCKFORM
#define MFEM_PBLOCKFORM

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <mpi.h>
#include "pfespace.hpp"
#include "pgridfunc.hpp"
#include "pbilinearform.hpp"

namespace mfem
{
// square block forms
class ParBlockForm
{

private:
   // BilinearForms
   Array2D<ParBilinearForm * > bforms;
   Array2D<ParMixedBilinearForm * > mforms;
   Array<ParFiniteElementSpace *> pfes;
   Array<int> dof_offsets;
   Array<int> tdof_offsets;
   int nblocks;

   // ess_tdof list for each space
   Array<Array<int> *> ess_tdofs;

   // Block operator of HypreParMatrix
   BlockOperator * P = nullptr; // Block Prolongation
   BlockMatrix * R = nullptr; // Block Restriction

   BlockMatrix * sp_mat = nullptr;
   // Block operator of HypreParMatrix
   BlockOperator * p_mat = nullptr;
   BlockOperator * p_mat_e = nullptr;

   void FillEssTdofLists(const Array<int> & ess_tdof_list);
   void BuildProlongation();
   void ParallelAssemble(BlockMatrix *m);

public:

   ParBlockForm(const Array<ParFiniteElementSpace*> pfes_ );

   void SetBlock(ParBilinearForm * bform, int row_idx, int col_idx);
   void SetBlock(ParMixedBilinearForm * mform, int row_idx, int col_idx);

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector & b,
                         OperatorHandle &A, Vector &X,
                         Vector &B, int copy_interior = 0);

   void FormSystemMatrix(const Array<int> &ess_tdof_list,
                         OperatorHandle &A);

   void RecoverFEMSolution(const Vector &X, Vector &x);

   /// Destroys bilinear form.
   ~ParBlockForm();
};

}

#endif // MFEM_USE_MPI

#endif
