// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of class BlockForm

#include "../config/config.hpp"

#include "fem.hpp"

namespace mfem
{

void BlockForm::BuildProlongation()
{
   P = new BlockMatrix(dof_offsets, tdof_offsets);
   R = new BlockMatrix(tdof_offsets, dof_offsets);
   P->owns_blocks = 0;
   R->owns_blocks = 0;

   for (int i = 0; i<nblocks; i++)
   {
      const SparseMatrix * P_ = fes[i]->GetConformingProlongation();
      P->SetBlock(i,i,const_cast<SparseMatrix*>(P_));
      const SparseMatrix * R_ = fes[i]->GetRestrictionMatrix();
      R->SetBlock(i,i,const_cast<SparseMatrix*>(R_));
   }
}

void BlockForm::Finalize(int skip_zeros)
{
   if (mat) { mat->Finalize(skip_zeros); }
   if (mat_e) { mat_e->Finalize(skip_zeros); }
}

void BlockForm::ConformingAssemble()
{
   Finalize(0);
   if (!P) { BuildProlongation(); }

   BlockMatrix * Pt = Transpose(*P);
   BlockMatrix * PtA = mfem::Mult(*Pt, *mat);
   // mat->owns_blocks = 0;
   for (int i = 0; i<nblocks; i++)
   {
      for (int j = 0; j<nblocks; j++)
      {
         if (mat->IsZeroBlock(i,j)) { continue; }
         if (Pt->IsZeroBlock(i,i))
         {
            PtA->SetBlock(i,j,&mat->GetBlock(i,j));
         }
      }
   }
   delete mat;
   if (mat_e)
   {
      BlockMatrix *PtAe = mfem::Mult(*Pt, *mat_e);
      mat_e->owns_blocks = 0;
      for (int i = 0; i<nblocks; i++)
      {
         for (int j = 0; j<nblocks; j++)
         {
            if (mat_e->IsZeroBlock(i,j)) { continue; }
            SparseMatrix * tmp = &mat_e->GetBlock(i,j);
            if (Pt->IsZeroBlock(i,i))
            {
               PtAe->SetBlock(i,j,tmp);
            }
            else
            {
               delete tmp;
            }
         }
      }
      delete mat_e;
      mat_e = PtAe;
      mat_e->owns_blocks = 1;
   }
   delete Pt;

   mat = mfem::Mult(*PtA, *P);
   PtA->owns_blocks = 0;
   for (int i = 0; i<nblocks; i++)
   {
      for (int j = 0; j<nblocks; j++)
      {
         if (PtA->IsZeroBlock(j,i)) { continue; }
         SparseMatrix * tmp = &PtA->GetBlock(j,i);
         if (P->IsZeroBlock(i,i))
         {
            mat->SetBlock(j,i,tmp);
         }
         else
         {
            delete tmp;
         }
      }
   }
   delete PtA;

   if (mat_e)
   {
      BlockMatrix *PtAeP = mfem::Mult(*mat_e, *P);
      mat_e->owns_blocks = 0;
      for (int i = 0; i<nblocks; i++)
      {
         for (int j = 0; j<nblocks; j++)
         {
            if (mat_e->IsZeroBlock(j,i)) { continue; }
            SparseMatrix * tmp = &mat_e->GetBlock(j,i);
            if (P->IsZeroBlock(i,i))
            {
               PtAeP->SetBlock(j,i,tmp);
            }
            else
            {
               delete tmp;
            }
         }
      }

      delete mat_e;
      mat_e = PtAeP;
   }
   height = mat->Height();
   width = mat->Width();
}


BlockForm::BlockForm(const Array<FiniteElementSpace*> fes_ ): fes(
      fes_)
{
   nblocks = fes.Size();
   bforms.SetSize(nblocks,nblocks);
   mforms.SetSize(nblocks,nblocks);
   dof_offsets.Append(0);
   tdof_offsets.Append(0);
   for (int i = 0; i<nblocks; i++)
   {
      dof_offsets.Append(fes[i]->GetVSize());
      tdof_offsets.Append(fes[i]->GetTrueVSize());
      for (int j = 0; j<nblocks; j++)
      {
         bforms(i,j) = nullptr;
         mforms(i,j) = nullptr;
      }
   }
   dof_offsets.PartialSum();
   tdof_offsets.PartialSum();
   diag_policy = mfem::Operator::DIAG_ONE;

}

void BlockForm::SetBlock(BilinearForm * bform, int row_idx, int col_idx)
{
   MFEM_VERIFY((row_idx >=0 && row_idx < nblocks), "row index out of bounds");
   MFEM_VERIFY((col_idx >=0 && col_idx < nblocks), "col index out of bounds");
   MFEM_VERIFY(!mforms(row_idx,col_idx), "Entry has already been set");
   MFEM_VERIFY(!bforms(row_idx,col_idx), "Entry has already been set");
   bforms(row_idx,col_idx) = bform;
}
void BlockForm::SetBlock(MixedBilinearForm * mform, int row_idx,
                         int col_idx)
{
   MFEM_VERIFY((row_idx >=0 && row_idx < nblocks), "row index out of bounds");
   MFEM_VERIFY((col_idx >=0 && col_idx < nblocks), "col index out of bounds");
   MFEM_VERIFY(!mforms(row_idx,col_idx), "Entry has already been set");
   MFEM_VERIFY(!bforms(row_idx,col_idx), "Entry has already been set");
   mforms(row_idx,col_idx) = mform;
}

/// Assemble the local matrix
void BlockForm::Assemble(int skip_zeros)
{
   mat = new BlockMatrix(dof_offsets);
   for (int i = 0; i<nblocks; i++)
   {
      int h = dof_offsets[i+1]-dof_offsets[i];
      for (int j = 0; j<nblocks; j++)
      {
         int w = dof_offsets[j+1]-dof_offsets[j];
         if (bforms(i,j))
         {
            bforms(i,j)->Assemble(skip_zeros);
            MFEM_VERIFY(h = bforms(i,j)->Height(), "inconsistent height of bilinear form");
            MFEM_VERIFY(w = bforms(i,j)->Width(), "inconsistent width of bilinear form");
            mat->SetBlock(i,j,&bforms(i,j)->SpMat());
         }
         else if (mforms(i,j))
         {
            mforms(i,j)->Assemble(skip_zeros);
            MFEM_VERIFY(h = mforms(i,j)->Height(),
                        "inconsistent height of MixedBilinear form");
            MFEM_VERIFY(w = mforms(i,j)->Width(),
                        "inconsistent width of Mixedbilinear form");
            mat->SetBlock(i,j,&mforms(i,j)->SpMat());
         }
         else
         {
            mat->SetBlock(i,j,nullptr);
         }
      }
   }
}

void BlockForm::FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x,
                                 Vector & b,
                                 OperatorHandle &A, Vector &X,
                                 Vector &B, int copy_interior)
{
   FormSystemMatrix(ess_tdof_list, A);

   if (!P)
   {
      EliminateVDofsInRHS(ess_tdof_list, x, b);
      X.MakeRef(x, 0, x.Size());
      B.MakeRef(b, 0, b.Size());
      if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
   }
   else // non conforming space
   {
      B.SetSize(P->Width());

      P->MultTranspose(b, B);
      real_t *data = b.GetData();
      Vector tmp;
      for (int i = 0; i<nblocks; i++)
      {
         if (P->IsZeroBlock(i,i))
         {
            int offset = tdof_offsets[i];
            tmp.SetDataAndSize(&data[offset],tdof_offsets[i+1]-tdof_offsets[i]);
            B.SetVector(tmp,offset);
         }
      }

      X.SetSize(R->Height());

      R->Mult(x, X);
      data = x.GetData();
      for (int i = 0; i<nblocks; i++)
      {
         if (R->IsZeroBlock(i,i))
         {
            int offset = tdof_offsets[i];
            tmp.SetDataAndSize(&data[offset],tdof_offsets[i+1]-tdof_offsets[i]);
            X.SetVector(tmp,offset);
         }
      }

      EliminateVDofsInRHS(ess_tdof_list, X, B);
      if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
   }
}

void BlockForm::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                 OperatorHandle &A)
{
   if (!mat_e)
   {
      bool conforming = true;
      for (int i = 0; i<nblocks; i++)
      {
         const SparseMatrix *P_ = fes[i]->GetConformingProlongation();
         if (P_)
         {
            conforming = false;
            break;
         }
      }
      if (!conforming) { ConformingAssemble(); }
      const int remove_zeros = 0;
      EliminateVDofs(ess_tdof_list, diag_policy);
      Finalize(remove_zeros);
   }
   A.Reset(mat, false);
}

void BlockForm::RecoverFEMSolution(const Vector &X, Vector &x)
{
   if (!P)
   {
      x.SyncMemory(X);
   }
   else
   {
      x.SetSize(P->Height());
      P->Mult(X, x);
      real_t *data = X.GetData();
      Vector tmp;
      for (int i = 0; i<nblocks; i++)
      {
         if (P->IsZeroBlock(i,i))
         {
            int offset = tdof_offsets[i];
            tmp.SetDataAndSize(&data[offset],tdof_offsets[i+1]-tdof_offsets[i]);
            x.SetVector(tmp,offset);
         }
      }
   }
}

void BlockForm::EliminateVDofs(const Array<int> &vdofs,
                               Operator::DiagonalPolicy dpolicy)
{
   if (mat_e == NULL)
   {
      Array<int> offsets;

      offsets.MakeRef( (P) ? tdof_offsets : dof_offsets);

      mat_e = new BlockMatrix(offsets);
      mat_e->owns_blocks = 1;
      for (int i = 0; i<mat_e->NumRowBlocks(); i++)
      {
         int h = offsets[i+1] - offsets[i];
         for (int j = 0; j<mat_e->NumColBlocks(); j++)
         {
            int w = offsets[j+1] - offsets[j];
            mat_e->SetBlock(i,j,new SparseMatrix(h, w));
         }
      }
   }

   mat->EliminateRowCols(vdofs,mat_e,diag_policy);
}

void BlockForm::EliminateVDofsInRHS(
   const Array<int> &vdofs, const Vector &x, Vector &b)
{
   mat_e->AddMult(x,b,-1.);
   mat->PartMult(vdofs,x,b);
}

BlockForm::~BlockForm()
{
   delete mat_e;
   mat_e = nullptr;
   delete mat;
   mat = nullptr;
   delete P;
   delete R;
}

};
