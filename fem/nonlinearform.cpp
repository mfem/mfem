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

#include "fem.hpp"

namespace mfem
{

void NonlinearForm::SetEssentialBC(const Array<int> &bdr_attr_is_ess,
                                   Vector *rhs)
{
   int i, j, vsize, nv;
   vsize = fes->GetVSize();
   Array<int> vdof_marker(vsize);

   // virtual call, works in parallel too
   fes->GetEssentialVDofs(bdr_attr_is_ess, vdof_marker);
   nv = 0;
   for (i = 0; i < vsize; i++)
      if (vdof_marker[i])
      {
         nv++;
      }

   ess_vdofs.SetSize(nv);

   for (i = j = 0; i < vsize; i++)
      if (vdof_marker[i])
      {
         ess_vdofs[j++] = i;
      }

   if (rhs)
      for (i = 0; i < nv; i++)
      {
         (*rhs)(ess_vdofs[i]) = 0.0;
      }
}

double NonlinearForm::GetEnergy(const Vector &x) const
{
   Array<int> vdofs;
   Vector el_x;
   const FiniteElement *fe;
   ElementTransformation *T;
   double energy = 0.0;

   if (dfi.Size())
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         x.GetSubVector(vdofs, el_x);
         for (int k = 0; k < dfi.Size(); k++)
         {
            energy += dfi[k]->GetElementEnergy(*fe, *T, el_x);
         }
      }

   return energy;
}

void NonlinearForm::Mult(const Vector &x, Vector &y) const
{
   Array<int> vdofs;
   Vector el_x, el_y;
   const FiniteElement *fe;
   ElementTransformation *T;

   y = 0.0;

   if (dfi.Size())
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         x.GetSubVector(vdofs, el_x);
         for (int k = 0; k < dfi.Size(); k++)
         {
            dfi[k]->AssembleElementVector(*fe, *T, el_x, el_y);
            y.AddElementVector(vdofs, el_y);
         }
      }

   for (int i = 0; i < ess_vdofs.Size(); i++)
   {
      y(ess_vdofs[i]) = 0.0;
   }
   // y(ess_vdofs[i]) = x(ess_vdofs[i]);
}

Operator &NonlinearForm::GetGradient(const Vector &x) const
{
   const int skip_zeros = 0;
   Array<int> vdofs;
   Vector el_x;
   DenseMatrix elmat;
   const FiniteElement *fe;
   ElementTransformation *T;

   if (Grad == NULL)
   {
      Grad = new SparseMatrix(fes->GetVSize());
   }
   else
   {
      *Grad = 0.0;
   }

   if (dfi.Size())
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         x.GetSubVector(vdofs, el_x);
         for (int k = 0; k < dfi.Size(); k++)
         {
            dfi[k]->AssembleElementGrad(*fe, *T, el_x, elmat);
            Grad->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
            // Grad->AddSubMatrix(vdofs, vdofs, elmat, 1);
         }
      }

   for (int i = 0; i < ess_vdofs.Size(); i++)
   {
      Grad->EliminateRowCol(ess_vdofs[i]);
   }

   if (!Grad->Finalized())
   {
      Grad->Finalize(skip_zeros);
   }

   return *Grad;
}

NonlinearForm::~NonlinearForm()
{
   delete Grad;
   for (int i = 0; i < dfi.Size(); i++)
   {
      delete dfi[i];
   }
}

}
