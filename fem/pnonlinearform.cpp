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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"

namespace mfem
{

void ParNonlinearForm::SetEssentialBC(const Array<int> &bdr_attr_is_ess,
                                      Vector *rhs)
{
   ParFiniteElementSpace *pfes = ParFESpace();

   NonlinearForm::SetEssentialBC(bdr_attr_is_ess);

   // ess_vdofs is a list of local vdofs
   if (rhs)
      for (int i = 0; i < ess_vdofs.Size(); i++)
      {
         int tdof = pfes->GetLocalTDofNumber(ess_vdofs[i]);
         if (tdof >= 0)
         {
            (*rhs)(tdof) = 0.0;
         }
      }
}

double ParNonlinearForm::GetEnergy(const ParGridFunction &x) const
{
   double loc_energy, glob_energy;

   loc_energy = NonlinearForm::GetEnergy(x);

   MPI_Allreduce(&loc_energy, &glob_energy, 1, MPI_DOUBLE, MPI_SUM,
                 ParFESpace()->GetComm());

   return glob_energy;
}

double ParNonlinearForm::GetEnergy(const Vector &x) const
{
   X.Distribute(&x);
   return GetEnergy(X);
}

void ParNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   X.Distribute(&x);

   NonlinearForm::Mult(X, Y);

   ParFESpace()->GroupComm().Reduce<double>(Y, GroupCommunicator::Sum);

   Y.GetTrueDofs(y);
}

const SparseMatrix &ParNonlinearForm::GetLocalGradient(const Vector &x) const
{
   X.Distribute(&x);

   NonlinearForm::GetGradient(X); // (re)assemble Grad

   return *Grad;
}

Operator &ParNonlinearForm::GetGradient(const Vector &x) const
{
   ParFiniteElementSpace *pfes = ParFESpace();

   delete pGrad;

   X.Distribute(&x);

   NonlinearForm::GetGradient(X); // (re)assemble Grad

   // construct a parallel block-diagonal wrapper matrix A based on Grad
   HypreParMatrix *A =
      new HypreParMatrix(pfes->GetComm(),
                         pfes->GlobalVSize(), pfes->GetDofOffsets(), Grad);

   pGrad = RAP(A, pfes->Dof_TrueDof_Matrix());

   delete A;

   return *pGrad;
}

}

#endif
