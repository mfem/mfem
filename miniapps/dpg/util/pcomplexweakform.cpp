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

#include "pcomplexweakform.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

void ParComplexDPGWeakForm::FillEssTdofLists(const Array<int> &
                                             ess_tdof_list)
{
   int j;
   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      int tdof = ess_tdof_list[i];
      for (j = 0; j < nblocks; j++)
      {
         if (tdof_offsets[j+1] > tdof) { break; }
      }
      ess_tdofs[j]->Append(tdof-tdof_offsets[j]);
   }
}

void ParComplexDPGWeakForm::Assemble(int skip_zeros)
{
   ComplexDPGWeakForm::Assemble(skip_zeros);
}

void ParComplexDPGWeakForm::ParallelAssemble(BlockMatrix *m_r,
                                             BlockMatrix *m_i)
{
   if (!P) { BuildProlongation(); }

   p_mat_r = new BlockOperator(tdof_offsets);
   p_mat_i = new BlockOperator(tdof_offsets);
   p_mat_e_r = new BlockOperator(tdof_offsets);
   p_mat_e_i = new BlockOperator(tdof_offsets);
   p_mat_r->owns_blocks = 1;
   p_mat_i->owns_blocks = 1;
   p_mat_e_r->owns_blocks = 1;
   p_mat_e_i->owns_blocks = 1;
   HypreParMatrix * A_r = nullptr;
   HypreParMatrix * A_i = nullptr;
   HypreParMatrix * PtAP_r = nullptr;
   HypreParMatrix * PtAP_i = nullptr;
   for (int i = 0; i < nblocks; i++)
   {
      HypreParMatrix * Pi = (HypreParMatrix*)(&P->GetBlock(i,i));
      for (int j = 0; j<nblocks; j++)
      {
         if (m_r->IsZeroBlock(i,j)) { continue; }
         if (i == j)
         {
            // Make block diagonal square hypre matrix
            A_r = new HypreParMatrix(trial_pfes[i]->GetComm(), trial_pfes[i]->GlobalVSize(),
                                     trial_pfes[i]->GetDofOffsets(), &m_r->GetBlock(i,i));
            PtAP_r = RAP(A_r,Pi);
            delete A_r;
            p_mat_e_r->SetBlock(i, i, PtAP_r->EliminateRowsCols(*ess_tdofs[i]));

            A_i = new HypreParMatrix(trial_pfes[i]->GetComm(), trial_pfes[i]->GlobalVSize(),
                                     trial_pfes[i]->GetDofOffsets(), &m_i->GetBlock(i,i));

            PtAP_i = RAP(A_i,Pi);
            delete A_i;
            p_mat_e_i->SetBlock(i, i, PtAP_i->EliminateCols(*ess_tdofs[i]));
            PtAP_i->EliminateRows(*ess_tdofs[i]);
         }
         else
         {
            HypreParMatrix * Pj = (HypreParMatrix*)(&P->GetBlock(j,j));
            A_r = new HypreParMatrix(trial_pfes[i]->GetComm(), trial_pfes[i]->GlobalVSize(),
                                     trial_pfes[j]->GlobalVSize(), trial_pfes[i]->GetDofOffsets(),
                                     trial_pfes[j]->GetDofOffsets(), &m_r->GetBlock(i,j));
            PtAP_r = RAP(Pi,A_r,Pj);
            delete A_r;
            p_mat_e_r->SetBlock(i, j, PtAP_r->EliminateCols(*ess_tdofs[j]));
            PtAP_r->EliminateRows(*ess_tdofs[i]);

            A_i = new HypreParMatrix(trial_pfes[i]->GetComm(), trial_pfes[i]->GlobalVSize(),
                                     trial_pfes[j]->GlobalVSize(), trial_pfes[i]->GetDofOffsets(),
                                     trial_pfes[j]->GetDofOffsets(), &m_i->GetBlock(i,j));
            PtAP_i = RAP(Pi,A_i,Pj);
            delete A_i;
            p_mat_e_i->SetBlock(i, j, PtAP_i->EliminateCols(*ess_tdofs[j]));
            PtAP_i->EliminateRows(*ess_tdofs[i]);
         }
         p_mat_r->SetBlock(i, j, PtAP_r);
         p_mat_i->SetBlock(i, j, PtAP_i);
      }
   }
}


void ParComplexDPGWeakForm::BuildProlongation()
{
   P = new BlockOperator(dof_offsets, tdof_offsets);
   R = new BlockMatrix(tdof_offsets, dof_offsets);
   P->owns_blocks = 0;
   R->owns_blocks = 0;

   for (int i = 0; i < nblocks; i++)
   {
      HypreParMatrix * P_ = trial_pfes[i]->Dof_TrueDof_Matrix();
      P->SetBlock(i,i,P_);
      const SparseMatrix * R_ = trial_pfes[i]->GetRestrictionMatrix();
      R->SetBlock(i, i, const_cast<SparseMatrix*>(R_));
   }
}

void ParComplexDPGWeakForm::FormLinearSystem(const Array<int>
                                             &ess_tdof_list,
                                             Vector &x,
                                             OperatorHandle &A,
                                             Vector &X, Vector &B,
                                             int copy_interior)
{
   FormSystemMatrix(ess_tdof_list, A);
   if (static_cond)
   {
      static_cond->ReduceSystem(x, X, B, copy_interior);
   }
   else
   {
      int n = P->Width();
      B.SetSize(2*n);
      Vector B_r(B, 0, n);
      Vector B_i(B, n, n);
      P->MultTranspose(*y_r, B_r);
      P->MultTranspose(*y_i, B_i);

      int m = R->Height();
      X.SetSize(2*m);

      Vector X_r(X, 0, m);
      Vector X_i(X, m, m);

      Vector x_r(x, 0, x.Size()/2);
      Vector x_i(x, x.Size()/2, x.Size()/2);

      R->Mult(x_r, X_r);
      R->Mult(x_i, X_i);

      // eliminate tdof is RHS
      // B_r -= Ae_r*X_r + Ae_i X_i
      // B_i -= Ae_i*X_r + Ae_r X_i
      Vector tmp(B_r.Size());
      p_mat_e_r->Mult(X_r, tmp); B_r-=tmp;
      p_mat_e_i->Mult(X_i, tmp); B_r+=tmp;

      p_mat_e_i->Mult(X_r, tmp); B_i-=tmp;
      p_mat_e_r->Mult(X_i, tmp); B_i-=tmp;

      for (int j = 0; j < nblocks; j++)
      {
         if (!ess_tdofs[j]->Size()) { continue; }
         for (int i = 0; i < ess_tdofs[j]->Size(); i++)
         {
            int tdof = (*ess_tdofs[j])[i];
            int gdof = tdof + tdof_offsets[j];
            B_r(gdof) = X_r(gdof); // diagonal policy is always one in parallel
            B_i(gdof) = X_i(gdof); // diagonal policy is always one in parallel
         }
      }
      if (!copy_interior)
      {
         X_r.SetSubVectorComplement(ess_tdof_list, 0.0);
         X_i.SetSubVectorComplement(ess_tdof_list, 0.0);
      }
   }
}

void ParComplexDPGWeakForm::FormSystemMatrix(const Array<int>
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
      A.Reset(&static_cond->GetSchurComplexOperator(), false);
   }
   else
   {
      FillEssTdofLists(ess_tdof_list);
      if (mat_r)
      {
         const int remove_zeros = 0;
         Finalize(remove_zeros);
         ParallelAssemble(mat_r, mat_i);
         delete mat_r;
         delete mat_i;
         mat_r = nullptr;
         mat_i = nullptr;
         delete mat_e_r;
         delete mat_e_i;
         mat_e_r = nullptr;
         mat_e_i = nullptr;
      }
      p_mat = new ComplexOperator(p_mat_r, p_mat_i, false, false);
      A.Reset(p_mat, false);
   }
}

void ParComplexDPGWeakForm::RecoverFEMSolution(const Vector &X,
                                               Vector &x)
{
   if (static_cond)
   {
      static_cond->ComputeSolution(X, x);
   }
   else
   {
      int n = P->Height();
      int m = P->Width();
      x.SetSize(2*n);

      Vector x_r(x,0,n);
      Vector x_i(x,n,n);
      Vector X_r(const_cast<Vector&>(X), 0, m);
      Vector X_i(const_cast<Vector&>(X), m, m);

      P->Mult(X_r, x_r);
      P->Mult(X_i, x_i);
   }
}

void ParComplexDPGWeakForm::Update()
{
   ComplexDPGWeakForm::Update();
   delete p_mat_e_r;
   delete p_mat_e_i;
   p_mat_e_r = nullptr;
   p_mat_e_i = nullptr;
   delete p_mat_r;
   delete p_mat_i;
   p_mat_r = nullptr;
   p_mat_i = nullptr;
   delete p_mat;
   p_mat = nullptr;
   for (int i = 0; i < nblocks; i++)
   {
      delete ess_tdofs[i];
      ess_tdofs[i] = new Array<int>();
   }
   delete P;
   P = nullptr;
   delete R;
   R = nullptr;
}

ParComplexDPGWeakForm::~ParComplexDPGWeakForm()
{
   delete p_mat_e_r;
   delete p_mat_e_i;
   p_mat_e_r = nullptr;
   p_mat_e_i = nullptr;
   delete p_mat_r;
   delete p_mat_i;
   p_mat_r = nullptr;
   p_mat_i = nullptr;
   delete p_mat;
   p_mat = nullptr;
   for (int i = 0; i < nblocks; i++)
   {
      delete ess_tdofs[i];
   }
   delete P;
   delete R;
}

} // namespace mfem

#endif
