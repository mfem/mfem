// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"

namespace mfem
{

void ParBilinearForm::pAllocMat()
{
   int nbr_size = pfes->GetFaceNbrVSize();

   if (precompute_sparsity == 0 || fes->GetVDim() > 1)
   {
      if (keep_nbr_block)
         mat = new SparseMatrix(height + nbr_size, width + nbr_size);
      else
         mat = new SparseMatrix(height, width + nbr_size);
      return;
   }

   // the sparsity pattern is defined from the map: face->element->dof
   fes->BuildElementToDofTable();
   const Table &lelem_ldof = fes->GetElementToDofTable(); // <-- dofs
   const Table &nelem_ndof = pfes->face_nbr_element_dof; // <-- vdofs
   Table elem_dof; // element + nbr-element <---> dof
   if (nbr_size > 0)
   {
      // merge lelem_ldof and nelem_ndof into elem_dof
      int s1 = lelem_ldof.Size(), s2 = nelem_ndof.Size();
      const int *I1 = lelem_ldof.GetI(), *J1 = lelem_ldof.GetJ();
      const int *I2 = nelem_ndof.GetI(), *J2 = nelem_ndof.GetJ();
      const int nnz1 = I1[s1], nnz2 = I2[s2];

      elem_dof.SetDims(s1 + s2, nnz1 + nnz2);

      int *I = elem_dof.GetI(), *J = elem_dof.GetJ();
      for (int i = 0; i <= s1; i++)
         I[i] = I1[i];
      for (int j = 0; j < nnz1; j++)
         J[j] = J1[j];
      for (int i = 0; i <= s2; i++)
         I[s1+i] = I2[i] + nnz1;
      for (int j = 0; j < nnz2; j++)
         J[nnz1+j] = J2[j] + height;
   }
   //   dof_elem x  elem_face x face_elem x elem_dof  (keep_nbr_block = true)
   // ldof_lelem x lelem_face x face_elem x elem_dof  (keep_nbr_block = false)
   Table dof_dof;
   {
      Table face_dof; // face_elem x elem_dof
      {
         Table *face_elem = pfes->GetParMesh()->GetFaceToAllElementTable();
         if (nbr_size > 0)
            mfem::Mult(*face_elem, elem_dof, face_dof);
         else
            mfem::Mult(*face_elem, lelem_ldof, face_dof);
         delete face_elem;
         if (nbr_size > 0)
            elem_dof.Clear();
      }

      if (keep_nbr_block)
      {
         Table dof_face;
         Transpose(face_dof, dof_face, height + nbr_size);
         mfem::Mult(dof_face, face_dof, dof_dof);
      }
      else
      {
         Table ldof_face;
         {
            Table face_ldof;
            Table *face_lelem = fes->GetMesh()->GetFaceToElementTable();
            mfem::Mult(*face_lelem, lelem_ldof, face_ldof);
            delete face_lelem;
            Transpose(face_ldof, ldof_face, height);
         }
         mfem::Mult(ldof_face, face_dof, dof_dof);
      }
   }

   int *I = dof_dof.GetI();
   int *J = dof_dof.GetJ();
   int nrows = dof_dof.Size();
   double *data = new double[I[nrows]];

   mat = new SparseMatrix(I, J, data, nrows, height + nbr_size);
   *mat = 0.0;

   dof_dof.LoseData();
}

HypreParMatrix *ParBilinearForm::ParallelAssemble(SparseMatrix *m)
{
   if (m == NULL)
      return NULL;

   HypreParMatrix *A;
   if (fbfi.Size() == 0)
   {
      // construct a parallel block-diagonal wrapper matrix A based on m
      A = new HypreParMatrix(pfes->GetComm(),
                             pfes->GlobalVSize(), pfes->GetDofOffsets(), m);
   }
   else
   {
      // handle the case when 'm' contains offdiagonal
      int  lvsize = pfes->GetVSize();
      int *face_nbr_glob_ldof = pfes->GetFaceNbrGlobalDofMap();
      int ldof_offset = pfes->GetMyDofOffset();

      Array<int> glob_J(m->NumNonZeroElems());
      int *J = m->GetJ();
      for (int i = 0; i < glob_J.Size(); i++)
         if (J[i] < lvsize)
            glob_J[i] = J[i] + ldof_offset;
         else
            glob_J[i] = face_nbr_glob_ldof[J[i] - lvsize];

      A = new HypreParMatrix(pfes->GetComm(), lvsize, pfes->GlobalVSize(),
                             pfes->GlobalVSize(), m->GetI(), glob_J,
                             m->GetData(), pfes->GetDofOffsets(),
                             pfes->GetDofOffsets());
   }

   HypreParMatrix *rap = RAP(A, pfes->Dof_TrueDof_Matrix());

   delete A;

   return rap;
}

void ParBilinearForm::AssembleSharedFaces(int skip_zeros)
{
   ParMesh *pmesh = pfes->GetParMesh();
   FaceElementTransformations *T;
   Array<int> vdofs1, vdofs2, vdofs_all;
   DenseMatrix elemmat;

   int nfaces = pmesh->GetNSharedFaces();
   for (int i = 0; i < nfaces; i++)
   {
      T = pmesh->GetSharedFaceTransformations(i);
      pfes->GetElementVDofs(T->Elem1No, vdofs1);
      pfes->GetFaceNbrElementVDofs(T->Elem2No, vdofs2);
      vdofs1.Copy(vdofs_all);
      for (int j = 0; j < vdofs2.Size(); j++)
         vdofs2[j] += height;
      vdofs_all.Append(vdofs2);
      for (int k = 0; k < fbfi.Size(); k++)
      {
         fbfi[k]->AssembleFaceMatrix(*pfes->GetFE(T->Elem1No),
                                     *pfes->GetFaceNbrFE(T->Elem2No),
                                     *T, elemmat);
         if (keep_nbr_block)
            mat->AddSubMatrix(vdofs_all, vdofs_all, elemmat, skip_zeros);
         else
            mat->AddSubMatrix(vdofs1, vdofs_all, elemmat, skip_zeros);
      }
   }
}

void ParBilinearForm::Assemble(int skip_zeros)
{
   if (mat == NULL && fbfi.Size() > 0)
   {
      pfes->ExchangeFaceNbrData();
      pAllocMat();
   }

   BilinearForm::Assemble(skip_zeros);

   if (fbfi.Size() > 0)
      AssembleSharedFaces(skip_zeros);
}

void ParBilinearForm::TrueAddMult(const Vector &x, Vector &y, const double a)
   const
{
   MFEM_VERIFY(fbfi.Size() == 0, "the case of interior face integrators is not"
               " implemented");

   if (X.ParFESpace() != pfes)
   {
      X.Update(pfes);
      Y.Update(pfes);
   }

   X.Distribute(&x);
   mat->Mult(X, Y);
   pfes->Dof_TrueDof_Matrix()->MultTranspose(a, Y, 1.0, y);
}


HypreParMatrix *ParDiscreteLinearOperator::ParallelAssemble(SparseMatrix *m)
{
   if (m == NULL)
      return NULL;

   int *I = m->GetI();
   int *J = m->GetJ();
   double *data = m->GetData();

   // remap to tdof local row and tdof global column indices
   SparseMatrix local(range_fes->TrueVSize(), domain_fes->GlobalTrueVSize());
   for (int i = 0; i < m->Height(); i++)
   {
      int lti = range_fes->GetLocalTDofNumber(i);
      if (lti >= 0)
         for (int j = I[i]; j < I[i+1]; j++)
            local.Set(lti, domain_fes->GetGlobalTDofNumber(J[j]), data[j]);
   }
   local.Finalize();

   // construct and return a global ParCSR matrix by splitting the local matrix
   // into diag and offd parts
   return new HypreParMatrix(range_fes->GetComm(),
                             range_fes->TrueVSize(),
                             range_fes->GlobalTrueVSize(),
                             domain_fes->GlobalTrueVSize(),
                             local.GetI(), local.GetJ(), local.GetData(),
                             range_fes->GetTrueDofOffsets(),
                             domain_fes->GetTrueDofOffsets());
}

void ParDiscreteLinearOperator::GetParBlocks(Array2D<HypreParMatrix *> &blocks) const
{
   int rdim = range_fes->GetVDim();
   int ddim = domain_fes->GetVDim();

   blocks.SetSize(rdim, ddim);

   int i, j, n;

   // construct the scalar versions of the row/coll offset arrays
   int *row_starts, *col_starts;
   if (HYPRE_AssumedPartitionCheck())
      n = 2;
   else
      n = range_fes->GetNRanks()+1;
   row_starts = new int[n];
   col_starts = new int[n];
   for (i = 0; i < n; i++)
   {
      row_starts[i] = (range_fes->GetTrueDofOffsets())[i] / rdim;
      col_starts[i] = (domain_fes->GetTrueDofOffsets())[i] / ddim;
   }

   Array2D<SparseMatrix *> lblocks;
   GetBlocks(lblocks);

   for (int bi = 0; bi < rdim; bi++)
      for (int bj = 0; bj < ddim; bj++)
      {
         int *I = lblocks(bi,bj)->GetI();
         int *J = lblocks(bi,bj)->GetJ();
         double *data = lblocks(bi,bj)->GetData();

         // remap to tdof local row and tdof global column indices
         SparseMatrix local(range_fes->TrueVSize()/rdim,
                            domain_fes->GlobalTrueVSize()/ddim);
         for (i = 0; i < lblocks(bi,bj)->Height(); i++)
         {
            int lti = range_fes->GetLocalTDofNumber(i);
            if (lti >= 0)
               for (j = I[i]; j < I[i+1]; j++)
                  local.Set(lti,
                            domain_fes->GetGlobalScalarTDofNumber(J[j]),
                            data[j]);
         }
         local.Finalize();

         delete lblocks(bi,bj);

         // construct and return a global ParCSR matrix by splitting the local
         // matrix into diag and offd parts
         blocks(bi,bj) = new HypreParMatrix(range_fes->GetComm(),
                                            range_fes->TrueVSize()/rdim,
                                            range_fes->GlobalTrueVSize()/rdim,
                                            domain_fes->GlobalTrueVSize()/ddim,
                                            local.GetI(), local.GetJ(), local.GetData(),
                                            row_starts, col_starts);
      }

   delete [] row_starts;
   delete [] col_starts;
}

HypreParMatrix *ParMixedBilinearForm::ParallelAssemble()
{
   int  nproc   = trial_pfes -> GetNRanks();
   int *trial_dof_off = trial_pfes -> GetDofOffsets();
   int *test_dof_off  = test_pfes -> GetDofOffsets();

   // construct the block-diagonal matrix A
   HypreParMatrix *A;
   if (HYPRE_AssumedPartitionCheck())
      A = new HypreParMatrix(trial_pfes->GetComm(), test_dof_off[2], trial_dof_off[2], test_dof_off, trial_dof_off, mat);
   else
      A = new HypreParMatrix(trial_pfes->GetComm(), test_dof_off[nproc], trial_dof_off[nproc], test_dof_off, trial_dof_off, mat);

   HypreParMatrix *rap = RAP(test_pfes -> Dof_TrueDof_Matrix(), A, trial_pfes -> Dof_TrueDof_Matrix());

   delete A;

   return rap;
}

}

#endif
