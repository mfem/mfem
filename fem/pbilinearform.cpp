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

#ifdef MFEM_USE_MPI

#include "fem.hpp"

HypreParMatrix *ParBilinearForm::ParallelAssemble()
{
   int  nproc   = pfes -> GetNRanks();
   int *dof_off = pfes -> GetDofOffsets();

   // construct the block-diagonal matrix A
   HypreParMatrix *A;
   if (HYPRE_AssumedPartitionCheck())
      A = new HypreParMatrix(dof_off[2], dof_off, mat);
   else
      A = new HypreParMatrix(dof_off[nproc], dof_off, mat);

   HypreParMatrix *rap = RAP(A, pfes -> Dof_TrueDof_Matrix());

   delete A;

   return rap;
}

#endif
