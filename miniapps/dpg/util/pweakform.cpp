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

#include "pweakform.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

void ParDPGWeakForm::FillEssTdofLists(const Array<int> & ess_tdof_list)
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

void ParDPGWeakForm::Assemble(int skip_zeros)
{
   DPGWeakForm::Assemble(skip_zeros);
}

void ParDPGWeakForm::ParallelAssemble(BlockMatrix *m)
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
            A = new HypreParMatrix(trial_pfes[i]->GetComm(), trial_pfes[i]->GlobalVSize(),
                                   trial_pfes[i]->GetDofOffsets(),&m->GetBlock(i,i));
            PtAP = RAP(A,Pi);
            delete A;
            p_mat_e->SetBlock(i,i,PtAP->EliminateRowsCols(*ess_tdofs[i]));
         }
         else
         {
            HypreParMatrix * Pj = (HypreParMatrix*)(&P->GetBlock(j,j));
            A = new HypreParMatrix(trial_pfes[i]->GetComm(), trial_pfes[i]->GlobalVSize(),
                                   trial_pfes[j]->GlobalVSize(), trial_pfes[i]->GetDofOffsets(),
                                   trial_pfes[j]->GetDofOffsets(), &m->GetBlock(i,j));
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


void ParDPGWeakForm::BuildProlongation()
{
   P = new BlockOperator(dof_offsets, tdof_offsets);
   R = new BlockMatrix(tdof_offsets, dof_offsets);
   P->owns_blocks = 0;
   R->owns_blocks = 0;

   for (int i = 0; i<nblocks; i++)
   {
      HypreParMatrix * P_ = trial_pfes[i]->Dof_TrueDof_Matrix();
      P->SetBlock(i,i,P_);
      const SparseMatrix * R_ = trial_pfes[i]->GetRestrictionMatrix();
      R->SetBlock(i,i,const_cast<SparseMatrix*>(R_));
   }
}

void ParDPGWeakForm::FormLinearSystem(const Array<int>
                                      &ess_tdof_list,
                                      Vector &x,
                                      OperatorHandle &A, Vector &X,
                                      Vector &B, int copy_interior)
{
   FormSystemMatrix(ess_tdof_list, A);


   if (static_cond)
   {
      static_cond->ReduceSystem(x, X, B, copy_interior);
   }
   else
   {
      B.SetSize(P->Width());
      P->MultTranspose(*y,B);
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
}

void ParDPGWeakForm::FormSystemMatrix(const Array<int>
                                      &ess_tdof_list,
                                      OperatorHandle &A)
{
   if (static_cond)
   {
      if (!static_cond->HasEliminatedBC())
      {
         static_cond->SetEssentialTrueDofs(ess_tdof_list);
         static_cond->FormSystemMatrix(Operator::DiagonalPolicy::DIAG_ONE);
      }
      A.Reset(&static_cond->GetParallelSchurMatrix(), false);
   }
   else
   {
      FillEssTdofLists(ess_tdof_list);
      if (mat)
      {
         const int remove_zeros = 0;
         Finalize(remove_zeros);
         ParallelAssemble(mat);
         delete mat;
         mat = nullptr;
         delete mat_e;
         mat_e = nullptr;
      }
      A.Reset(p_mat,false);
   }

}



void ParDPGWeakForm::RecoverFEMSolution(const Vector &X,
                                        Vector &x)
{

   if (static_cond)
   {
      static_cond->ComputeSolution(X, x);
   }
   else
   {
      x.SetSize(P->Height());
      P->Mult(X, x);
   }
}

void ParDPGWeakForm::Update()
{
   DPGWeakForm::Update();
   delete p_mat_e;
   p_mat_e = nullptr;
   delete p_mat;
   p_mat = nullptr;
   for (int i = 0; i<nblocks; i++)
   {
      delete ess_tdofs[i];
      ess_tdofs[i] = new Array<int>();
   }
   delete P;
   P = nullptr;
   delete R;
   R = nullptr;
}

ParDPGWeakForm::~ParDPGWeakForm()
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

} // namespace mfem

#endif
