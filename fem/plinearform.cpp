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

void ParLinearForm::Update(ParFiniteElementSpace *pf)
{
   if (pf) pfes = pf;

   LinearForm::Update(pfes);
}

HypreParVector *ParLinearForm::ParallelAssemble()
{
   int  nproc    = pfes -> GetNRanks();
   int *dof_off  = pfes -> GetDofOffsets();
   int *tdof_off = pfes -> GetTrueDofOffsets();

   // vector on (all) dofs
   HypreParVector *v;
   if (HYPRE_AssumedPartitionCheck())
      v = new HypreParVector(dof_off[2], data, dof_off);
   else
      v = new HypreParVector(dof_off[nproc], data, dof_off);

   // vector on true dofs
   HypreParVector *tv;
   if (HYPRE_AssumedPartitionCheck())
      tv = new HypreParVector(tdof_off[2], tdof_off);
   else
      tv = new HypreParVector(tdof_off[nproc], tdof_off);

   pfes -> Dof_TrueDof_Matrix() -> MultTranspose(*v,*tv);

   delete v;

   return tv;
}

#endif
