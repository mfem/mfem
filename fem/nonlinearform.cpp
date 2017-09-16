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

   if (fbfi.Size())
   {
      MFEM_WARNING("WARNING: Skipping interior faces "
                   "in NonlinearForm::GetEnergy()!");
   }

   if (bfbfi.Size())
   {
      MFEM_WARNING("WARNING: Skipping boundary faces "
                   "in NonlinearForm::GetEnergy()!");
   }

   return energy;
}

void NonlinearForm::Mult(const Vector &x, Vector &y) const
{
   Array<int> vdofs;
   Vector el_x, el_y;
   const FiniteElement *fe;
   ElementTransformation *T;
   Mesh *mesh = fes->GetMesh();

   y = 0.0;

   if (dfi.Size())
   {
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
   }

   if (fbfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs2;

      for (int i = 0; i < mesh->GetNumFaces(); i++)
      {
         tr = mesh->GetInteriorFaceTransformations(i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            fes->GetElementVDofs(tr->Elem2No, vdofs2);
            vdofs.Append (vdofs2);

            x.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            fe2 = fes->GetFE(tr->Elem2No);

            for (int k = 0; k < fbfi.Size(); k++)
            {
               fbfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
               y.AddElementVector(vdofs, el_y);
            }
         }
      }
   }

   if (bfbfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < bfbfi.Size(); k++)
      {
         if (bfbfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *bfbfi_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            x.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            // The fe2 object is really a dummy and not used on the boundaries,
            // but we can't dereference a NULL pointer, and we don't want to
            // actually make a fake element.
            fe2 = fe1;
            for (int k = 0; k < bfbfi.Size(); k++)
            {
               if (bfbfi_marker[k] &&
                   (*bfbfi_marker[k])[bdr_attr-1] == 0) { continue; }

               bfbfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
               y.AddElementVector(vdofs, el_y);
            }
         }
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
   Mesh *mesh = fes->GetMesh();

   if (Grad == NULL)
   {
      Grad = new SparseMatrix(fes->GetVSize());
   }
   else
   {
      *Grad = 0.0;
   }

   if (dfi.Size())
   {
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
   }

   if (fbfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs2;

      for (int i = 0; i < mesh->GetNumFaces(); i++)
      {
         tr = mesh->GetInteriorFaceTransformations(i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            fes->GetElementVDofs(tr->Elem2No, vdofs2);
            vdofs.Append (vdofs2);

            x.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            fe2 = fes->GetFE(tr->Elem2No);

            for (int k = 0; k < fbfi.Size(); k++)
            {
               fbfi[k]->AssembleFaceGrad(*fe1, *fe2, *tr, el_x, elmat);
               Grad->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
            }
         }
      }
   }

   if (bfbfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < bfbfi.Size(); k++)
      {
         if (bfbfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *bfbfi_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            x.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            // The fe2 object is really a dummy and not used on the boundaries,
            // but we can't dereference a NULL pointer, and we don't want to
            // actually make a fake element.
            fe2 = fe1;
            for (int k = 0; k < bfbfi.Size(); k++)
            {
               if (bfbfi_marker[k] &&
                   (*bfbfi_marker[k])[bdr_attr-1] == 0) { continue; }

               bfbfi[k]->AssembleFaceGrad(*fe1, *fe2, *tr, el_x, elmat);
               Grad->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
            }
         }
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
