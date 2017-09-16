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
   X.Distribute(&x);

   NonlinearForm::Mult(X, Y);

   if (fbfi.Size())
   {
      // Still need to add integrals over shared faces.
      ParFiniteElementSpace *pfes = ParFESpace();
      ParMesh *pmesh = pfes->GetParMesh();
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs1, vdofs2, vdofs_all;
      Vector el_x, el_y;

      for (int i = 0; i < pmesh->GetNSharedFaces(); i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i, true);
         if (tr != NULL)
         {
            fe1 = pfes->GetFE(tr->Elem1No);
            fe2 = pfes->GetFaceNbrFE(tr->Elem2No);

            pfes->GetElementVDofs(tr->Elem1No, vdofs1);
            pfes->GetFaceNbrElementVDofs(tr->Elem2No, vdofs2);

            vdofs1.Copy(vdofs_all);
            for (int j = 0; j < vdofs2.Size(); j++)
            {
               vdofs2[j] += height;
            }
            vdofs_all.Append(vdofs2);

            X.GetSubVector(vdofs_all, el_x);
            if (pfes->GetMyRank() == 0)
               {
                  std::cout << X.Size() << std::endl;
                  const int dof1 = fe1->GetDof();
                  const int dof2 = fe2->GetDof();
                  DenseMatrix elfun1_mat(el_x.GetData(), dof1, 4);
                  DenseMatrix elfun2_mat(el_x.GetData() + dof1 * 4, dof2, 4);
                  // el_x.Print();
                  elfun1_mat.Print();
                  elfun2_mat.Print();
               }

            for (int k = 0; k < fbfi.Size(); k++)
            {
               fbfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
               Y.AddElementVector(vdofs_all, el_y);
            }
         }
      }
   }

   ParFESpace()->GroupComm().Reduce<double>(Y, GroupCommunicator::Sum);

   Y.GetTrueDofs(y);
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
