// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "blockcomplexhypremat.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

void ParBlockComplexSystem::FillEssTdofLists(const Array<int> &
                                             ess_tdof_list)
{
   for (int i = 0; i < ess_tdofs.Size(); i++)
   {
      delete ess_tdofs[i];
      ess_tdofs[i] = new Array<int>();
   }
   int j;
   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      int tdof = ess_tdof_list[i];
      for (j = 0; j < nblocks; j++)
      {
         if (toffsets[j+1] > tdof) { break; }
      }
      ess_tdofs[j]->Append(tdof-toffsets[j]);
   }
}

ComplexOperator * ParBlockComplexSystem::EliminateBC(const Array<int>
                                                     ess_tdof_list, Vector &X, Vector & B)
{
   FillEssTdofLists(ess_tdof_list);
   delete op_e_r;
   delete op_e_i;
   op_e_r = new BlockOperator(toffsets);
   op_e_i = new BlockOperator(toffsets);
   op_e_r->owns_blocks = 1;
   op_e_i->owns_blocks = 1;

   for (int i = 0; i < nblocks; i++)
   {
      for (int j = 0; j < nblocks; j++)
      {
         if (op_r->IsZeroBlock(i,j)) { continue; }
         if (i == j)
         {
            auto mat_r = &(HypreParMatrix &)op_r->GetBlock(i,i);
            op_e_r->SetBlock(i, i, mat_r->EliminateRowsCols(*ess_tdofs[i]));
            auto mat_i = &(HypreParMatrix &)op_i->GetBlock(i,i);
            op_e_i->SetBlock(i, i, mat_i->EliminateCols(*ess_tdofs[i]));
            mat_i->EliminateRows(*ess_tdofs[i]);
         }
         else
         {
            auto mat_r = &(HypreParMatrix &)op_r->GetBlock(i,j);
            op_e_r->SetBlock(i, j, mat_r->EliminateCols(*ess_tdofs[j]));
            mat_r->EliminateRows(*ess_tdofs[i]);
            auto mat_i = &(HypreParMatrix &)op_i->GetBlock(i,j);
            op_e_i->SetBlock(i, j, mat_i->EliminateCols(*ess_tdofs[j]));
            mat_i->EliminateRows(*ess_tdofs[i]);
         }
      }
   }

   int n = B.Size()/2;
   Vector B_r(B, 0, n);
   Vector B_i(B, n, n);

   Vector X_r(X, 0, n);
   Vector X_i(X, n, n);

   // eliminate tdof is RHS
   // B_r -= Ae_r*X_r + Ae_i X_i
   // B_i -= Ae_i*X_r + Ae_r X_i
   Vector tmp(B_r.Size());
   op_e_r->Mult(X_r, tmp); B_r-=tmp;
   op_e_i->Mult(X_i, tmp); B_r+=tmp;

   op_e_i->Mult(X_r, tmp); B_i-=tmp;
   op_e_r->Mult(X_i, tmp); B_i-=tmp;

   for (int j = 0; j < nblocks; j++)
   {
      if (!ess_tdofs[j]->Size()) { continue; }
      for (int i = 0; i < ess_tdofs[j]->Size(); i++)
      {
         int tdof = (*ess_tdofs[j])[i];
         int gdof = tdof + toffsets[j];
         B_r(gdof) = X_r(gdof); // diagonal policy is always one in parallel
         B_i(gdof) = X_i(gdof); // diagonal policy is always one in parallel
      }
   }

   X_r.SetSubVectorComplement(ess_tdof_list, 0.0);
   X_i.SetSubVectorComplement(ess_tdof_list, 0.0);

   return op;
}



} // namespace mfem

#endif
