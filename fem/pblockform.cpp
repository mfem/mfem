// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of class ParBlockForm

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"

namespace mfem
{

void ParBlockForm::FillEssTdofLists(const Array<int> & ess_tdof_list)
{
   int j;
   for (int i = 0; i<ess_tdof_list.Size(); i++)
   {
      int tdof = ess_tdof_list[i];
      for (j = 0; j < nblocks; j++)
      {
         if (tdof_offsets[j+1] > tdof) { break; }
      }
      ess_tdofs[j]->Append(tdof-tdof_offsets[j]);
   }
}

void ParBlockForm::BuildProlongation()
{
   P = new BlockOperator(dof_offsets, tdof_offsets);
   R = new BlockMatrix(tdof_offsets, dof_offsets);
   P->owns_blocks = 0;
   R->owns_blocks = 0;

   for (int i = 0; i<nblocks; i++)
   {
      HypreParMatrix * P_ = pfes[i]->Dof_TrueDof_Matrix();
      P->SetBlock(i,i,P_);
      const SparseMatrix * R_ = pfes[i]->GetRestrictionMatrix();
      R->SetBlock(i,i,const_cast<SparseMatrix*>(R_));
   }
}

void ParBlockForm::ParallelAssemble(BlockMatrix *m)
{
   if (!P) { BuildProlongation(); }

   p_mat = new BlockOperator(tdof_offsets);
   p_mat_e = new BlockOperator(tdof_offsets);
   p_mat->owns_blocks = 1;
   p_mat_e->owns_blocks = 1;
   HypreParMatrix * A = nullptr;
   HypreParMatrix * PtAP = nullptr;
   for (int i = 0; i<nblocks; i++)
   {
      HypreParMatrix * Pi = (HypreParMatrix*)(&P->GetBlock(i,i));
      HypreParMatrix * Pit = Pi->Transpose();
      for (int j = 0; j<nblocks; j++)
      {
         if (m->IsZeroBlock(i,j)) { continue; }
         if (i == j)
         {
            // Make block diagonal square hypre matrix
            A = new HypreParMatrix(pfes[i]->GetComm(), pfes[i]->GlobalVSize(),
                                   pfes[i]->GetDofOffsets(),&m->GetBlock(i,i));
            PtAP = RAP(A,Pi);
            delete A;
            p_mat_e->SetBlock(i,i,PtAP->EliminateRowsCols(*ess_tdofs[i]));
         }
         else
         {
            HypreParMatrix * Pj = (HypreParMatrix*)(&P->GetBlock(j,j));
            A = new HypreParMatrix(pfes[i]->GetComm(), pfes[i]->GlobalVSize(),
                                   pfes[j]->GlobalVSize(), pfes[i]->GetDofOffsets(),
                                   pfes[j]->GetDofOffsets(), &m->GetBlock(i,j));
            HypreParMatrix * APj = ParMult(A, Pj,true);
            delete A;
            PtAP = ParMult(Pit,APj,true);
            delete APj;
            p_mat_e->SetBlock(i,j,PtAP->EliminateCols(*ess_tdofs[j]));
            PtAP->EliminateRows(*ess_tdofs[i]);
         }
         p_mat->SetBlock(i,j,PtAP);
      }
      delete Pit;
   }
}


ParBlockForm::ParBlockForm(const Array<ParFiniteElementSpace*> pfes_ ): pfes(
      pfes_)
{
   nblocks = pfes.Size();
   bforms.SetSize(nblocks,nblocks);
   mforms.SetSize(nblocks,nblocks);
   ess_tdofs.SetSize(nblocks);
   dof_offsets.Append(0);
   tdof_offsets.Append(0);
   for (int i = 0; i<nblocks; i++)
   {
      dof_offsets.Append(pfes[i]->GetVSize());
      tdof_offsets.Append(pfes[i]->TrueVSize());
      ess_tdofs[i] = new Array<int>();
      for (int j = 0; j<nblocks; j++)
      {
         bforms(i,j) = nullptr;
         mforms(i,j) = nullptr;
      }
   }
   dof_offsets.PartialSum();
   tdof_offsets.PartialSum();
}

void ParBlockForm::SetBlock(ParBilinearForm * bform, int row_idx, int col_idx)
{
   MFEM_VERIFY((row_idx >=0 && row_idx < nblocks), "row index out of bounds");
   MFEM_VERIFY((col_idx >=0 && col_idx < nblocks), "col index out of bounds");
   MFEM_VERIFY(!mforms(row_idx,col_idx), "Entry has already been set");
   MFEM_VERIFY(!bforms(row_idx,col_idx), "Entry has already been set");
   bforms(row_idx,col_idx) = bform;
}
void ParBlockForm::SetBlock(ParMixedBilinearForm * mform, int row_idx,
                            int col_idx)
{
   MFEM_VERIFY((row_idx >=0 && row_idx < nblocks), "row index out of bounds");
   MFEM_VERIFY((col_idx >=0 && col_idx < nblocks), "col index out of bounds");
   MFEM_VERIFY(!mforms(row_idx,col_idx), "Entry has already been set");
   MFEM_VERIFY(!bforms(row_idx,col_idx), "Entry has already been set");
   mforms(row_idx,col_idx) = mform;
}

/// Assemble the local matrix
void ParBlockForm::Assemble(int skip_zeros)
{
   sp_mat = new BlockMatrix(dof_offsets);
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
            sp_mat->SetBlock(i,j,&bforms(i,j)->SpMat());
         }
         else if (mforms(i,j))
         {
            mforms(i,j)->Assemble(skip_zeros);
            MFEM_VERIFY(h = mforms(i,j)->Height(),
                        "inconsistent height of MixedBilinear form");
            MFEM_VERIFY(w = mforms(i,j)->Width(),
                        "inconsistent width of Mixedbilinear form");
            sp_mat->SetBlock(i,j,&mforms(i,j)->SpMat());
         }
         else
         {
            sp_mat->SetBlock(i,j,nullptr);
         }
      }
   }
}

void ParBlockForm::FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x,
                                    Vector & b,
                                    OperatorHandle &A, Vector &X,
                                    Vector &B, int copy_interior)
{
   FormSystemMatrix(ess_tdof_list, A);
   B.SetSize(P->Width());
   P->MultTranspose(b,B);
   X.SetSize(R->Height());
   R->Mult(x,X);

   // eliminate tdof in RHS
   // B -= Ae*X
   Vector tmp(B.Size());
   p_mat_e->Mult(X,tmp);
   B-=tmp;

   for (int j = 0; j<nblocks; j++)
   {
      if (!ess_tdofs[j]->Size()) { continue; }
      for (int i = 0; i < ess_tdofs[j]->Size(); i++)
      {
         int tdof = (*ess_tdofs[j])[i];
         int gdof = tdof + tdof_offsets[j];
         B(gdof) = X(gdof); // diagonal policy in always one in parallel
      }
   }
   if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
}

void ParBlockForm::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                    OperatorHandle &A)
{
   FillEssTdofLists(ess_tdof_list);
   if (sp_mat)
   {
      sp_mat->Finalize();
      ParallelAssemble(sp_mat);
      delete sp_mat;
      sp_mat = nullptr;
   }
   A.Reset(p_mat,false);
}

void ParBlockForm::RecoverFEMSolution(const Vector &X, Vector &x)
{
   x.SetSize(P->Height());
   P->Mult(X, x);
}

ParBlockForm::~ParBlockForm()
{
   delete p_mat_e;
   p_mat_e = nullptr;
   delete p_mat;
   p_mat = nullptr;
   for (int i = 0; i<nblocks; i++)
   {
      delete ess_tdofs[i];
   }
   delete P;
   delete R;
}

};

#endif