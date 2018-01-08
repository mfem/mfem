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

NonlinearForm::NonlinearForm(FiniteElementSpace *f) : Operator(f->GetVSize())
{
   fes = f;
   Grad = NULL;

   needs_gs = false;
   X = Y = NULL;
   if (dynamic_cast<const L2_FECollection *>(fes->FEColl()))
   {
      needs_gs = false;
   }
   else
   {
      X = new Vector(fes->GetLocalVSize());
      Y = new Vector(fes->GetLocalVSize());
   }
}


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

   if (dnfi.Size())
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         x.GetSubVector(vdofs, el_x);
         for (int k = 0; k < dnfi.Size(); k++)
         {
            energy += dnfi[k]->GetElementEnergy(*fe, *T, el_x);
         }
      }

   if (fnfi.Size())
   {
      MFEM_ABORT("TODO: add energy contribution from interior face terms");
   }

   if (bfnfi.Size())
   {
      MFEM_ABORT("TODO: add energy contribution from boundary face terms");
   }

   return energy;
}

void NonlinearForm::MultGeneral(const Vector &x, Vector &y) const
{
   Array<int> vdofs;
   Vector el_x, el_y;
   const FiniteElement *fe;
   ElementTransformation *T;
   Mesh *mesh = fes->GetMesh();

   y = 0.0;

   if (dnfi.Size())
   {
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         x.GetSubVector(vdofs, el_x);
         for (int k = 0; k < dnfi.Size(); k++)
         {
            dnfi[k]->AssembleElementVector(*fe, *T, el_x, el_y);
            y.AddElementVector(vdofs, el_y);
         }
      }
   }

   if (fnfi.Size())
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

            for (int k = 0; k < fnfi.Size(); k++)
            {
               fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
               y.AddElementVector(vdofs, el_y);
            }
         }
      }
   }

   if (bfnfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < bfnfi.Size(); k++)
      {
         if (bfnfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *bfnfi_marker[k];
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
            for (int k = 0; k < bfnfi.Size(); k++)
            {
               if (bfnfi_marker[k] &&
                   (*bfnfi_marker[k])[bdr_attr-1] == 0) { continue; }

               bfnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
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

static inline bool CanTensorizeAssembly(const FiniteElementSpace *fes)
{
   const Mesh *mesh = fes->GetMesh();

   const int BaseGeom = mesh->GetElementBaseGeometry(mesh->GetNE());
   // Would have to dynamic_cast every element if mesh were mixed.
   if (BaseGeom < 0) return false;

   // Dynamic cast only the first fe to check if it's supported
   const FiniteElement *fe = fes->GetFE(0);
   return (dynamic_cast<const TensorBasisElement *>(fe) != NULL);
}

void NonlinearForm::Mult(const Vector &x, Vector &y) const
{
   if (CanTensorizeAssembly(fes) && (fesi.Size() > 0))
   {
      if (needs_gs)
      {
         fes->ToLocalVector(x, *X);
         *Y = 0.0;
      }
      else
      {
         X = const_cast<Vector *>(&x);
         Y = &y;
      }

      for (int i = 0; i < fesi.Size(); i++)
      {
         fesi[i]->Assemble(fes, fes, *X);
         fesi[i]->FormVector(*Y);
      }

      if (needs_gs)
      {
         fes->ToGlobalVector(*Y, y);
      }
   }
   else
   {
      MultGeneral(x, y);
   }
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

   if (dnfi.Size())
   {
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         x.GetSubVector(vdofs, el_x);
         for (int k = 0; k < dnfi.Size(); k++)
         {
            dnfi[k]->AssembleElementGrad(*fe, *T, el_x, elmat);
            Grad->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
            // Grad->AddSubMatrix(vdofs, vdofs, elmat, 1);
         }
      }
   }

   if (fnfi.Size())
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

            for (int k = 0; k < fnfi.Size(); k++)
            {
               fnfi[k]->AssembleFaceGrad(*fe1, *fe2, *tr, el_x, elmat);
               Grad->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
            }
         }
      }
   }

   if (bfnfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < bfnfi.Size(); k++)
      {
         if (bfnfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *bfnfi_marker[k];
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
            for (int k = 0; k < bfnfi.Size(); k++)
            {
               if (bfnfi_marker[k] &&
                   (*bfnfi_marker[k])[bdr_attr-1] == 0) { continue; }

               bfnfi[k]->AssembleFaceGrad(*fe1, *fe2, *tr, el_x, elmat);
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
   if (needs_gs) delete Y;
   for (int i = 0; i <  dnfi.Size(); i++) { delete  dnfi[i]; }
   for (int i = 0; i <  fnfi.Size(); i++) { delete  fnfi[i]; }
   for (int i = 0; i < bfnfi.Size(); i++) { delete bfnfi[i]; }
}

}
