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

ParNonlinearForm::ParNonlinearForm(ParFiniteElementSpace *pf)
   : NonlinearForm(pf), pGrad(Operator::Hypre_ParCSR)
{
   X.MakeRef(pf, NULL);
   Y.MakeRef(pf, NULL);
   MFEM_VERIFY(!Serial(), "internal MFEM error");
}

double ParNonlinearForm::GetParGridFunctionEnergy(const Vector &x) const
{
   double loc_energy, glob_energy;

   loc_energy = GetGridFunctionEnergy(x);

   if (fnfi.Size())
   {
      MFEM_ABORT("TODO: add energy contribution from shared faces");
   }

   MPI_Allreduce(&loc_energy, &glob_energy, 1, MPI_DOUBLE, MPI_SUM,
                 ParFESpace()->GetComm());

   return glob_energy;
}

void ParNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   NonlinearForm::Mult(x, y); // x --(P)--> aux1 --(A_local)--> aux2
   Y.SetData(aux2.GetData()); // aux2 contains A_local.P.x

   if (fnfi.Size())
   {
      // Terms over shared interior faces in parallel.
      ParFiniteElementSpace *pfes = ParFESpace();
      ParMesh *pmesh = pfes->GetParMesh();
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs1, vdofs2;
      Vector el_x, el_y;

      X.SetData(aux1.GetData()); // aux1 contains P.x
      X.ExchangeFaceNbrData();
      const int n_shared_faces = pmesh->GetNSharedFaces();
      for (int i = 0; i < n_shared_faces; i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i, true);

         fe1 = pfes->GetFE(tr->Elem1No);
         fe2 = pfes->GetFaceNbrFE(tr->Elem2No);

         pfes->GetElementVDofs(tr->Elem1No, vdofs1);
         pfes->GetFaceNbrElementVDofs(tr->Elem2No, vdofs2);

         el_x.SetSize(vdofs1.Size() + vdofs2.Size());
         X.GetSubVector(vdofs1, el_x.GetData());
         X.FaceNbrData().GetSubVector(vdofs2, el_x.GetData() + vdofs1.Size());

         for (int k = 0; k < fnfi.Size(); k++)
         {
            fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
            Y.AddElementVector(vdofs1, el_y.GetData());
         }
      }
   }

   P->MultTranspose(Y, y);
}

const SparseMatrix &ParNonlinearForm::GetLocalGradient(const Vector &x) const
{
   NonlinearForm::GetGradient(x); // (re)assemble Grad, no b.c.

   return *Grad;
}

Operator &ParNonlinearForm::GetGradient(const Vector &x) const
{
   ParFiniteElementSpace *pfes = ParFESpace();

   pGrad.Clear();

   NonlinearForm::GetGradient(x); // (re)assemble Grad, no b.c.

   OperatorHandle dA(pGrad.Type()), Ph(pGrad.Type());

   if (fnfi.Size() == 0)
   {
      dA.MakeSquareBlockDiag(pfes->GetComm(), pfes->GlobalVSize(),
                             pfes->GetDofOffsets(), Grad);
   }
   else
   {
      MFEM_ABORT("TODO: assemble contributions from shared face terms");
   }

   // TODO - construct Dof_TrueDof_Matrix directly in the pGrad format
   Ph.ConvertFrom(pfes->Dof_TrueDof_Matrix());
   pGrad.MakePtAP(dA, Ph);

   // Impose b.c. on pGrad
   OperatorHandle pGrad_e;
   pGrad_e.EliminateRowsCols(pGrad, ess_tdof_list);

   return *pGrad.Ptr();
}

void ParNonlinearForm::Update()
{
   Y.MakeRef(ParFESpace(), NULL);
   X.MakeRef(ParFESpace(), NULL);
   pGrad.Clear();
   NonlinearForm::Update();
}

}

#endif
