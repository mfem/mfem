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

// Implementation of class LinearForm

#include "fem.hpp"

namespace mfem
{

void LinearForm::AddDomainIntegrator (LinearFormIntegrator * lfi)
{
   dlfi.Append (lfi);
}

void LinearForm::AddBoundaryIntegrator (LinearFormIntegrator * lfi)
{
   blfi.Append (lfi);
}

void LinearForm::AddBdrFaceIntegrator (LinearFormIntegrator * lfi)
{
   flfi.Append (lfi);
}

void LinearForm::Assemble()
{
   Array<int> vdofs;
   ElementTransformation *eltrans;
   Vector elemvect;

   int i;

   Vector::operator=(0.0);

   if (dlfi.Size())
      for (i = 0; i < fes -> GetNE(); i++)
      {
         fes -> GetElementVDofs (i, vdofs);
         eltrans = fes -> GetElementTransformation (i);
         for (int k=0; k < dlfi.Size(); k++)
         {
            dlfi[k]->AssembleRHSElementVect(*fes->GetFE(i), *eltrans, elemvect);
            AddElementVector (vdofs, elemvect);
         }
      }

   if (blfi.Size())
      for (i = 0; i < fes -> GetNBE(); i++)
      {
         fes -> GetBdrElementVDofs (i, vdofs);
         eltrans = fes -> GetBdrElementTransformation (i);
         for (int k=0; k < blfi.Size(); k++)
         {
            blfi[k]->AssembleRHSElementVect(*fes->GetBE(i), *eltrans, elemvect);
            AddElementVector (vdofs, elemvect);
         }
      }

   if (flfi.Size())
   {
      FaceElementTransformations *tr;
      Mesh *mesh = fes -> GetMesh();
      for (i = 0; i < mesh -> GetNBE(); i++)
      {
         tr = mesh -> GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            fes -> GetElementVDofs (tr -> Elem1No, vdofs);
            for (int k = 0; k < flfi.Size(); k++)
            {
               flfi[k] -> AssembleRHSElementVect (*fes->GetFE(tr -> Elem1No),
                                                  *tr, elemvect);
               AddElementVector (vdofs, elemvect);
            }
         }
      }
   }
}

void LinearForm::ConformingAssemble(Vector &b) const
{
   const SparseMatrix *P = fes->GetConformingProlongation();
   if (P)
   {
      b.SetSize(P->Width());
      P->MultTranspose(*this, b);
      return;
   }

   b = *this;
}

void LinearForm::ConformingAssemble()
{
   if (fes->Nonconforming())
   {
      Vector b;
      ConformingAssemble(b);
      static_cast<Vector&>(*this) = b;
   }
}

void LinearForm::Update(FiniteElementSpace *f, Vector &v, int v_offset)
{
   fes = f;
   NewDataAndSize((double *)v + v_offset, fes->GetVSize());
}

LinearForm::~LinearForm()
{
   int k;
   for (k=0; k < dlfi.Size(); k++) { delete dlfi[k]; }
   for (k=0; k < blfi.Size(); k++) { delete blfi[k]; }
   for (k=0; k < flfi.Size(); k++) { delete flfi[k]; }
}

}
