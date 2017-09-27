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

BlockNonlinearForm::BlockNonlinearForm() :
   fes(0), BlockGrad(NULL)
{
   height = 0;
   width = 0;
   
}

void BlockNonlinearForm::SetSpaces(Array<FiniteElementSpace *> &f)
{
   height = 0;
   width = 0;
   f.Copy(fes);
   block_offsets.SetSize(f.Size() + 1);
   block_trueOffsets.SetSize(f.Size() + 1);
   block_offsets[0] = 0;
   block_trueOffsets[0] = 0;

   for (int i=0; i<fes.Size(); i++) {
      block_offsets[i+1] = fes[i]->GetVSize();
      block_trueOffsets[i+1] = fes[i]->GetTrueVSize();
   }

   block_offsets.PartialSum();
   block_trueOffsets.PartialSum();

   height = block_trueOffsets[fes.Size()];
   width = block_trueOffsets[fes.Size()];

   Grads.SetSize(fes.Size(), fes.Size());
   for (int i=0; i<fes.Size(); i++) {
      for (int j=0; j<fes.Size(); j++) {
         Grads(i,j) = NULL;
      }
   }

   ess_vdofs.SetSize(fes.Size());

}


BlockNonlinearForm::BlockNonlinearForm(Array<FiniteElementSpace *> &f)
{
   SetSpaces(f);
}

void BlockNonlinearForm::AddBdrFaceIntegrator(BlockNonlinearFormIntegrator *fi,
                                              Array<int> &bdr_attr_marker)
{
   ffi.Append(fi);
   ffi_marker.Append(&bdr_attr_marker);
}

   
void BlockNonlinearForm::SetEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                                        Array<Vector *> &rhs)
{
   int i, j, vsize, nv;

   for (int s=0; s<fes.Size(); s++) {
      // First, set u variables
      vsize = fes[s]->GetVSize();
      Array<int> vdof_marker(vsize);

      // virtual call, works in parallel too
      fes[s]->GetEssentialVDofs(*(bdr_attr_is_ess[s]), vdof_marker);
      nv = 0;
      for (i = 0; i < vsize; i++) {
         if (vdof_marker[i]) {
            nv++;
         }
      }
      
      ess_vdofs[s] = new Array<int>(nv);

      for (i = j = 0; i < vsize; i++) {
         if (vdof_marker[i]) {
            (*ess_vdofs[s])[j++] = i;
         }
      }

      if (rhs[s]) {
         for (i = 0; i < nv; i++) {
            (*rhs[s])[(*ess_vdofs[s])[i]] = 0.0;
         }
      }
   }
}
   
void BlockNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   Array<Array<int> *>vdofs(fes.Size());
   Array<Vector *> el_x(fes.Size());
   Array<Vector *> el_y(fes.Size());
   Array<const FiniteElement *> fe(fes.Size());
   ElementTransformation *T;
   
   Array<Vector *> xs(fes.Size()), ys(fes.Size());

   for (int i=0; i<fes.Size(); i++) {
      xs[i] = new Vector(x.GetData() + block_offsets[i], fes[i]->GetVSize());
      ys[i] = new Vector(y.GetData() + block_offsets[i], fes[i]->GetVSize());
      *ys[i] = 0.0;
      el_x[i] = new Vector();
      el_y[i] = new Vector();
      vdofs[i] = new Array<int>;
   }

   if (dfi.Size()) {
      for (int i = 0; i < fes[0]->GetNE(); i++) {
         T = fes[0]->GetElementTransformation(i);
         for (int s = 0; s < fes.Size(); s++) {
            fes[s]->GetElementVDofs(i, *(vdofs[s]));            
            fe[s] = fes[s]->GetFE(i);
            xs[s]->GetSubVector(*(vdofs[s]), *el_x[s]);
         }
         
         for (int k = 0; k < dfi.Size(); k++)
         {
            dfi[k]->AssembleElementVector(fe, *T, 
                                          el_x, el_y);

            for (int s=0; s<fes.Size(); s++) {
               ys[s]->AddElementVector(*(vdofs[s]), *el_y[s]);
            }
         }
      }
   }
   
   if (bfi.Size()) {
      for (int i = 0; i < fes[0]->GetNBE(); i++) {
         T = fes[0]->GetBdrElementTransformation(i);
         for (int s=0; s<fes.Size(); s++) {
            fe[s] = fes[s]->GetBE(i);
            fes[s]->GetBdrElementVDofs(i, *(vdofs[s]));
            xs[s]->GetSubVector(*(vdofs[s]), *el_x[s]);
         }
         
         for (int k = 0; k < bfi.Size(); k++) {
            bfi[k]->AssembleElementVector(fe, *T, 
                                          el_x, el_y);

            for (int s = 0; s < fes.Size(); s++) { 
               ys[s]->AddElementVector(*(vdofs[s]), *el_y[s]);
            }
         }
      }
   }
   
   if (ffi.Size()) {
      Mesh *mesh = fes[0]->GetMesh();
      FaceElementTransformations *tr;
      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < ffi.Size(); k++)
      {
         if (ffi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *ffi_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
            {
               bdr_attr_marker[i] |= bdr_marker[i];
            }
      }      
      
      for (int i = 0; i < mesh->GetNBE(); i++) {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations(i);
         if (tr != NULL) {
            for (int s=0; s<fes.Size(); s++) {
               fe[s] = fes[s]->GetFE(tr->Elem1No);
               fes[s]->GetElementVDofs(tr->Elem1No, *(vdofs[s]));
               xs[s]->GetSubVector(*(vdofs[s]), *el_x[s]);
            }

            for (int k = 0; k < ffi.Size(); k++) {
               if (ffi_marker[k] &&
                   (*ffi_marker[k])[bdr_attr-1] == 0) { continue; }

               ffi[k]->AssembleRHSElementVector(fe, *tr, 
                                                el_x, el_y);

               for (int s=0; s<fes.Size(); s++) {
                  ys[s]->AddElementVector(*(vdofs[s]), *el_y[s]);
               }
            }
         }
      }
   }
      

   for (int s=0; s<fes.Size(); s++) {
      delete vdofs[s];
      for (int i = 0; i < ess_vdofs[s]->Size(); i++) {
         (*ys[s])((*ess_vdofs[s])[i]) = 0.0;
      }
   }
}
 
Operator &BlockNonlinearForm::GetGradient(const Vector &x) const
{
   const int skip_zeros = 0;
   Array<Array<int> *> vdofs(fes.Size());
   Array<Vector *> el_x(fes.Size());
   Array2D<DenseMatrix *> elmats(fes.Size(), fes.Size());
   Array<const FiniteElement *>fe(fes.Size());
   ElementTransformation * T;
   Array<Vector *> xs(fes.Size()), ys(fes.Size());

   if (BlockGrad != NULL) {
      delete BlockGrad;
   }

   BlockGrad = new BlockOperator(block_offsets);

   for (int i=0; i<fes.Size(); i++) {
      xs[i] = new Vector(x.GetData() + block_offsets[i], fes[i]->GetVSize());
      el_x[i] = new Vector();
      vdofs[i] = new Array<int>;
      for (int j=0; j<fes.Size(); j++) {
         elmats(i,j) = new DenseMatrix();
      }
   }

   for (int i=0; i<fes.Size(); i++) {
      for (int j=0; j<fes.Size(); j++) {
         if (Grads(i,j) != NULL) {
            delete Grads(i,j);
         }
         Grads(i,j) = new SparseMatrix(fes[i]->GetVSize(), fes[j]->GetVSize());
      }
   }

   if (dfi.Size()) {
      for (int i = 0; i < fes[0]->GetNE(); i++) {
         T = fes[0]->GetElementTransformation(i);
         for (int s = 0; s < fes.Size(); s++) {
            fe[s] = fes[s]->GetFE(i);
            fes[s]->GetElementVDofs(i, *vdofs[s]);
            xs[s]->GetSubVector(*vdofs[s], *el_x[s]);
         }
         
         for (int k = 0; k < dfi.Size(); k++) {
            dfi[k]->AssembleElementGrad(fe, *T, 
                                        el_x, elmats);
            for (int j=0; j<fes.Size(); j++) {
               for (int l=0; l<fes.Size(); l++) {
                  Grads(j,l)->AddSubMatrix(*vdofs[j], *vdofs[l], *elmats(j,l), skip_zeros);
               }
            }
         }
      }
   }
   if (bfi.Size()) {
      for (int i = 0; i < fes[0]->GetNBE(); i++) {
         T = fes[0]->GetBdrElementTransformation(i);
         for (int s=0; s < fes.Size(); s++) {
            fe[s] = fes[s]->GetBE(i);
            fes[s]->GetBdrElementVDofs(i, *vdofs[s]);
            xs[s]->GetSubVector(*vdofs[s], *el_x[s]);
         }

         for (int k = 0; k < dfi.Size(); k++) {
            bfi[k]->AssembleElementGrad(fe, *T, 
                                        el_x, elmats);
            for (int j=0; j<fes.Size(); j++) {
               for (int l=0; l<fes.Size(); l++) {
                  Grads(j,l)->AddSubMatrix(*vdofs[j], *vdofs[l], *elmats(j,l), skip_zeros);
               }
            }
         }
      }
   }         
         
   if (ffi.Size()) {
      FaceElementTransformations *tr;
      Mesh *mesh = fes[0]->GetMesh();

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < ffi.Size(); k++)
      {
         if (ffi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *ffi_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
            {
               bdr_attr_marker[i] |= bdr_marker[i];
            }
      }      
   
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations(i);
         if (tr != NULL) {
            T = fes[0]->GetElementTransformation(i);
            for (int s = 0; s < fes.Size(); s++) {
               fe[s] = fes[s]->GetFE(i);
               fes[s]->GetElementVDofs(i, *vdofs[s]);
               xs[s]->GetSubVector(*vdofs[s], *el_x[s]);
            }

            for (int k = 0; k < dfi.Size(); k++) {
               ffi[k]->AssembleElementGrad(fe, *T, 
                                           el_x, elmats);
               for (int l=0; l<fes.Size(); l++) {
                  for (int j=0; j<fes.Size(); j++) {
                     Grads(j,l)->AddSubMatrix(*vdofs[j], *vdofs[l], *elmats(j,l), skip_zeros);
                  }
               }
            }
         }
      }
   }

   for (int s=0; s<fes.Size(); s++) {
      for (int i = 0; i < ess_vdofs[s]->Size(); i++)
      {
         for (int j=0; j<fes.Size(); j++) {
            if (s==j) {
               Grads(s,s)->EliminateRowCol((*ess_vdofs[s])[i], 1);
            }
            else {
               Grads(s,j)->EliminateRow((*ess_vdofs[s])[i]);
               Grads(j,s)->EliminateCol((*ess_vdofs[s])[i]);
            }
         }
      }
   }

   if (!Grads(0,0)->Finalized()) {
      for (int i=0; i<fes.Size(); i++) {
         for (int j=0; j<fes.Size(); j++) {
            Grads(i,j)->Finalize(skip_zeros);
         }
      }
   }
  

   for (int i=0; i<fes.Size(); i++) {
      for (int j=0; j<fes.Size(); j++) {
         BlockGrad->SetBlock(i,j,Grads(i,j));
         delete elmats(i,j);
      }
   }

   return *BlockGrad;
}

BlockNonlinearForm::~BlockNonlinearForm()
{
   for (int i=0; i<fes.Size(); i++) {
      for (int j=0; j<fes.Size(); j++) {
         delete Grads(i,j);
      }
      delete ess_vdofs[i];
   }

   for (int i = 0; i < dfi.Size(); i++)
      {
         delete dfi[i];
      }

   for (int i = 0; i < bfi.Size(); i++)
      {
         delete bfi[i];
      }

   for (int i = 0; i < bfi.Size(); i++)
      {
         delete ffi[i];
      }

}

}
