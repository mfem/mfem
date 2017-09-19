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
   {
      for (int i = 0; i < ess_vdofs.Size(); i++)
      {
         int tdof = pfes->GetLocalTDofNumber(ess_vdofs[i]);
         if (tdof >= 0)
         {
            (*rhs)(tdof) = 0.0;
         }
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
   const Operator *P = ParFESpace()->GetProlongationMatrix();

   P->Mult(x, X);

   NonlinearForm::Mult(X, Y);

   if (fbfi.Size())
   {
      // Still need to add integrals over shared faces.

      // First exchange data in X with neighbors.
      X.ExchangeFaceNbrData();

      ParFiniteElementSpace *pfes = ParFESpace();
      ParMesh *pmesh = pfes->GetParMesh();
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs1, vdofs2;
      Vector el_x, el_y;

      for (int i = 0; i < pmesh->GetNSharedFaces(); i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i, true);

         fe1 = pfes->GetFE(tr->Elem1No);
         fe2 = pfes->GetFaceNbrFE(tr->Elem2No);

         pfes->GetElementVDofs(tr->Elem1No, vdofs1);
         pfes->GetFaceNbrElementVDofs(tr->Elem2No, vdofs2);

         el_x.SetSize(vdofs1.Size() + vdofs2.Size());
         X.GetSubVector(vdofs1, el_x.GetData());
         X.FaceNbrData().GetSubVector(vdofs2, el_x.GetData() + vdofs1.Size());

         for (int k = 0; k < fbfi.Size(); k++)
         {
            fbfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
            Y.AddElementVector(vdofs1, el_y.GetData());
         }
      }
   }

   P->MultTranspose(Y, y);
}

const SparseMatrix &ParNonlinearForm::GetLocalGradient(const Vector &x) const
{
   X.Distribute(&x);

   NonlinearForm::GetGradient(X); // (re)assemble Grad with b.c.

   return *Grad;
}

Operator &ParNonlinearForm::GetGradient(const Vector &x) const
{
   ParFiniteElementSpace *pfes = ParFESpace();

   pGrad.Clear();

   X.Distribute(&x);

   NonlinearForm::GetGradient(X); // (re)assemble Grad with b.c.

   OperatorHandle dA(pGrad.Type()), Ph(pGrad.Type());
   dA.MakeSquareBlockDiag(pfes->GetComm(), pfes->GlobalVSize(),
                          pfes->GetDofOffsets(), Grad);
   // TODO - construct Dof_TrueDof_Matrix directly in the pGrad format
   Ph.ConvertFrom(pfes->Dof_TrueDof_Matrix());
   pGrad.MakePtAP(dA, Ph);

   return *pGrad.Ptr();
}

}

#endif
