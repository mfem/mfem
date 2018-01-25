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
   // virtual call, works in parallel too
   fes->GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);

   if (rhs)
   {
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         (*rhs)(ess_tdof_list[i]) = 0.0;
      }
   }
}

void NonlinearForm::SetEssentialVDofs(const Array<int> &ess_vdofs_list)
{
   if (!P)
   {
      ess_vdofs_list.Copy(ess_tdof_list); // ess_vdofs_list --> ess_tdof_list
   }
   else
   {
      Array<int> ess_vdof_marker, ess_tdof_marker;
      FiniteElementSpace::ListToMarker(ess_vdofs_list, fes->GetVSize(),
                                       ess_vdof_marker);
      if (Serial())
      {
         fes->ConvertToConformingVDofs(ess_vdof_marker, ess_tdof_marker);
      }
      else
      {
#ifdef MFEM_USE_MPI
         ParFiniteElementSpace *pf = dynamic_cast<ParFiniteElementSpace*>(fes);
         ess_tdof_marker.SetSize(pf->GetTrueVSize());
         pf->Dof_TrueDof_Matrix()->BooleanMultTranspose(1, ess_vdof_marker,
                                                        0, ess_tdof_marker);
#else
         MFEM_ABORT("internal MFEM error");
#endif
      }
      FiniteElementSpace::MarkerToList(ess_tdof_marker, ess_tdof_list);
   }
}

double NonlinearForm::GetGridFunctionEnergy(const Vector &x) const
{
   Array<int> vdofs;
   Vector el_x;
   const FiniteElement *fe;
   ElementTransformation *T;
   double energy = 0.0;

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
            energy += dnfi[k]->GetElementEnergy(*fe, *T, el_x);
         }
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

const Vector &NonlinearForm::Prolongate(const Vector &x) const
{
   MFEM_VERIFY(x.Size() == Width(), "invalid input Vector size");
   if (P)
   {
      aux1.SetSize(P->Height());
      P->Mult(x, aux1);
      return aux1;
   }
   return x;
}

void NonlinearForm::Mult(const Vector &x, Vector &y) const
{
   Array<int> vdofs;
   Vector el_x, el_y;
   const FiniteElement *fe;
   ElementTransformation *T;
   Mesh *mesh = fes->GetMesh();
   const Vector &px = Prolongate(x);
   Vector &py = P ? aux2.SetSize(P->Height()), aux2 : y;

   py = 0.0;

   if (dnfi.Size())
   {
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         px.GetSubVector(vdofs, el_x);
         for (int k = 0; k < dnfi.Size(); k++)
         {
            dnfi[k]->AssembleElementVector(*fe, *T, el_x, el_y);
            py.AddElementVector(vdofs, el_y);
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

            px.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            fe2 = fes->GetFE(tr->Elem2No);

            for (int k = 0; k < fnfi.Size(); k++)
            {
               fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
               py.AddElementVector(vdofs, el_y);
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
            px.GetSubVector(vdofs, el_x);

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
               py.AddElementVector(vdofs, el_y);
            }
         }
      }
   }

   if (Serial())
   {
      if (cP) { cP->MultTranspose(py, y); }

      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         y(ess_tdof_list[i]) = 0.0;
      }
      // y(ess_tdof_list[i]) = x(ess_tdof_list[i]);
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
   const Vector &px = Prolongate(x);

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
         px.GetSubVector(vdofs, el_x);
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

            px.GetSubVector(vdofs, el_x);

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
            px.GetSubVector(vdofs, el_x);

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

   if (!Grad->Finalized())
   {
      Grad->Finalize(skip_zeros);
   }

   SparseMatrix *mGrad = Grad;
   if (Serial())
   {
      if (cP)
      {
         delete cGrad;
         cGrad = RAP(*cP, *Grad, *cP);
         mGrad = cGrad;
      }
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         mGrad->EliminateRowCol(ess_tdof_list[i]);
      }
   }

   return *mGrad;
}

void NonlinearForm::Update()
{
   if (sequence == fes->GetSequence()) { return; }

   height = width = fes->GetTrueVSize();
   delete cGrad; cGrad = NULL;
   delete Grad; Grad = NULL;
   ess_tdof_list.SetSize(0); // essential b.c. will need to be set again
   sequence = fes->GetSequence();
   // Do not modify aux1 and aux2, their size will be set before use.
   P = fes->GetProlongationMatrix();
   cP = dynamic_cast<const SparseMatrix*>(P);
}

NonlinearForm::~NonlinearForm()
{
   delete cGrad;
   delete Grad;
   for (int i = 0; i <  dnfi.Size(); i++) { delete  dnfi[i]; }
   for (int i = 0; i <  fnfi.Size(); i++) { delete  fnfi[i]; }
   for (int i = 0; i < bfnfi.Size(); i++) { delete bfnfi[i]; }
}

}
