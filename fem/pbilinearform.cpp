// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"
#include "../general/sort_pairs.hpp"

namespace mfem
{

void ParBilinearForm::pAllocMat()
{
   int nbr_size = pfes->GetFaceNbrVSize();

   if (precompute_sparsity == 0 || fes->GetVDim() > 1)
   {
      if (keep_nbr_block)
      {
         mat = new SparseMatrix(height + nbr_size, width + nbr_size);
      }
      else
      {
         mat = new SparseMatrix(height, width + nbr_size);
      }
      return;
   }

   // the sparsity pattern is defined from the map: face->element->dof
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
      {
         I[i] = I1[i];
      }
      for (int j = 0; j < nnz1; j++)
      {
         J[j] = J1[j];
      }
      for (int i = 0; i <= s2; i++)
      {
         I[s1+i] = I2[i] + nnz1;
      }
      for (int j = 0; j < nnz2; j++)
      {
         J[nnz1+j] = J2[j] + height;
      }
   }
   //   dof_elem x  elem_face x face_elem x elem_dof  (keep_nbr_block = true)
   // ldof_lelem x lelem_face x face_elem x elem_dof  (keep_nbr_block = false)
   Table dof_dof;
   {
      Table face_dof; // face_elem x elem_dof
      {
         Table *face_elem = pfes->GetParMesh()->GetFaceToAllElementTable();
         if (nbr_size > 0)
         {
            mfem::Mult(*face_elem, elem_dof, face_dof);
         }
         else
         {
            mfem::Mult(*face_elem, lelem_ldof, face_dof);
         }
         delete face_elem;
         if (nbr_size > 0)
         {
            elem_dof.Clear();
         }
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
   double *data = Memory<double>(I[nrows]);

   mat = new SparseMatrix(I, J, data, nrows, height + nbr_size);
   *mat = 0.0;

   dof_dof.LoseData();
}

/// @brief Make @a A_hyp steal ownership of its diagonal part @a A_diag.
///
/// If @a A_hyp does not own I and J, then they are aliases pointing to the I
/// and J arrays in @a A_diag. In that case, this function swaps the memory
/// objects. Similarly for the data array.
///
/// After this function is called, @a A_hyp will own all of the arrays of its
/// diagonal part.
///
/// @note I and J can only be aliases when HYPRE_BIGINT is disabled.
static void HypreStealOwnership(HypreParMatrix &A_hyp, SparseMatrix &A_diag)
{
#ifndef HYPRE_BIGINT
   bool own_i = A_hyp.GetDiagMemoryI().OwnsHostPtr();
   bool own_j = A_hyp.GetDiagMemoryJ().OwnsHostPtr();
   MFEM_CONTRACT_VAR(own_j);
   MFEM_ASSERT(own_i == own_j, "Inconsistent ownership");
   if (!own_i)
   {
      std::swap(A_diag.GetMemoryI(), A_hyp.GetDiagMemoryI());
      std::swap(A_diag.GetMemoryJ(), A_hyp.GetDiagMemoryJ());
   }
#endif
   if (!A_hyp.GetDiagMemoryData().OwnsHostPtr())
   {
      std::swap(A_diag.GetMemoryData(), A_hyp.GetDiagMemoryData());
   }
   A_hyp.SetOwnerFlags(3, A_hyp.OwnsOffd(), A_hyp.OwnsColMap());
}

void ParBilinearForm::ParallelRAP(SparseMatrix &loc_sp_mat, OperatorHandle &A)
{
   ParFiniteElementSpace &pfespace = *ParFESpace();

   // Create a block diagonal parallel matrix
   OperatorHandle A_diag(Operator::Hypre_ParCSR);
   A_diag.MakeSquareBlockDiag(pfespace.GetComm(),
                              pfespace.GlobalVSize(),
                              pfespace.GetDofOffsets(),
                              &loc_sp_mat);

   // Parallel matrix assembly using P^t A P (if needed)
   if (IsIdentityProlongation(pfespace.GetProlongationMatrix()))
   {
      A_diag.SetOperatorOwner(false);
      A.Reset(A_diag.Ptr(), false);
      HypreStealOwnership(*A.As<HypreParMatrix>(), loc_sp_mat);
   }
   else
   {
      OperatorHandle P(Operator::Hypre_ParCSR);
      P.ConvertFrom(pfespace.Dof_TrueDof_Matrix());
      A.MakePtAP(A_diag, P);
   }
}

void ParBilinearForm::EliminateBC(const Array<int> &ess_dofs,
                                  DiagonalPolicy diag_policy,
                                  HypreParMatrix &A)
{
   hypre_ParCSRMatrix *A_hypre = A;
   A.HypreReadWrite();

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A_hypre);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A_hypre);

   HYPRE_Int diag_nrows = hypre_CSRMatrixNumRows(diag);
   HYPRE_Int offd_ncols = hypre_CSRMatrixNumCols(offd);

   const int n_ess_dofs = ess_dofs.Size();
   const auto ess_dofs_d = ess_dofs.GetMemory().Read(
                              GetHypreMemoryClass(), n_ess_dofs);

   // Start communication to figure out which columns need to be eliminated in
   // the off-diagonal block
   hypre_ParCSRCommHandle *comm_handle;
   HYPRE_Int *int_buf_data, *eliminate_row, *eliminate_col;
   {
      eliminate_row = hypre_CTAlloc(HYPRE_Int, diag_nrows, HYPRE_MEMORY_DEVICE);
      eliminate_col = hypre_CTAlloc(HYPRE_Int, offd_ncols, HYPRE_MEMORY_DEVICE);

      // Make sure A has a communication package
      hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A_hypre);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);
      }

      // Which of the local rows are to be eliminated?
      MFEM_HYPRE_FORALL(i, diag_nrows, eliminate_row[i] = 0; );
      MFEM_HYPRE_FORALL(i, n_ess_dofs, eliminate_row[ess_dofs_d[i]] = 1; );

      // Use a matvec communication pattern to find (in eliminate_col) which of
      // the local offd columns are to be eliminated

      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      HYPRE_Int int_buf_sz = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      int_buf_data = hypre_CTAlloc(HYPRE_Int, int_buf_sz, HYPRE_MEMORY_DEVICE);

      HYPRE_Int *send_map_elmts;
#if defined(HYPRE_USING_GPU)
      hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
      send_map_elmts = hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);
#else
      send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
#endif
      MFEM_HYPRE_FORALL(i, int_buf_sz,
      {
         int k = send_map_elmts[i];
         int_buf_data[i] = eliminate_row[k];
      });

      // Try to use device-aware MPI for the communication if available
      comm_handle = hypre_ParCSRCommHandleCreate_v2(
                       11, comm_pkg, HYPRE_MEMORY_DEVICE, int_buf_data,
                       HYPRE_MEMORY_DEVICE, eliminate_col);
   }

   // Eliminate rows and columns in the diagonal block
   {
      const auto I = diag->i;
      const auto J = diag->j;
      auto data = diag->data;

      MFEM_HYPRE_FORALL(i, n_ess_dofs,
      {
         const int idof = ess_dofs_d[i];
         for (int j=I[idof]; j<I[idof+1]; ++j)
         {
            const int jdof = J[j];
            if (jdof == idof)
            {
               if (diag_policy == DiagonalPolicy::DIAG_ONE)
               {
                  data[j] = 1.0;
               }
               else if (diag_policy == DiagonalPolicy::DIAG_ZERO)
               {
                  data[j] = 0.0;
               }
               // else (diag_policy == DiagonalPolicy::DIAG_KEEP)
            }
            else
            {
               data[j] = 0.0;
               for (int k=I[jdof]; k<I[jdof+1]; ++k)
               {
                  if (J[k] == idof)
                  {
                     data[k] = 0.0;
                     break;
                  }
               }
            }
         }
      });
   }

   // Eliminate rows in the off-diagonal block
   {
      const auto I = offd->i;
      auto data = offd->data;
      MFEM_HYPRE_FORALL(i, n_ess_dofs,
      {
         const int idof = ess_dofs_d[i];
         for (int j=I[idof]; j<I[idof+1]; ++j)
         {
            data[j] = 0.0;
         }
      });
   }

   // Wait for MPI communication to finish
   hypre_ParCSRCommHandleDestroy(comm_handle);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_DEVICE);
   hypre_TFree(eliminate_row, HYPRE_MEMORY_DEVICE);

   // Eliminate columns in the off-diagonal block
   {
      const int nrows_offd = hypre_CSRMatrixNumRows(offd);
      const auto I = offd->i;
      const auto J = offd->j;
      auto data = offd->data;
      MFEM_HYPRE_FORALL(i, nrows_offd,
      {
         for (int j=I[i]; j<I[i+1]; ++j)
         {
            data[j] *= 1 - eliminate_col[J[j]];
         }
      });
   }

   hypre_TFree(eliminate_col, HYPRE_MEMORY_DEVICE);
}

void ParBilinearForm::ParallelAssemble(OperatorHandle &A, SparseMatrix *A_local)
{
   A.Clear();

   if (A_local == NULL) { return; }
   MFEM_VERIFY(A_local->Finalized(), "the local matrix must be finalized");

   OperatorHandle dA(A.Type()), Ph(A.Type()), hdA;

   if (interior_face_integs.Size() == 0)
   {
      // construct a parallel block-diagonal matrix 'A' based on 'a'
      dA.MakeSquareBlockDiag(pfes->GetComm(), pfes->GlobalVSize(),
                             pfes->GetDofOffsets(), A_local);
   }
   else
   {
      // handle the case when 'a' contains off-diagonal
      int lvsize = pfes->GetVSize();
      const HYPRE_BigInt *face_nbr_glob_ldof = pfes->GetFaceNbrGlobalDofMap();
      HYPRE_BigInt ldof_offset = pfes->GetMyDofOffset();

      Array<HYPRE_BigInt> glob_J(A_local->NumNonZeroElems());
      int *J = A_local->GetJ();
      for (int i = 0; i < glob_J.Size(); i++)
      {
         if (J[i] < lvsize)
         {
            glob_J[i] = J[i] + ldof_offset;
         }
         else
         {
            glob_J[i] = face_nbr_glob_ldof[J[i] - lvsize];
         }
      }

      // TODO - construct dA directly in the A format
      hdA.Reset(
         new HypreParMatrix(pfes->GetComm(), lvsize, pfes->GlobalVSize(),
                            pfes->GlobalVSize(), A_local->GetI(), glob_J,
                            A_local->GetData(), pfes->GetDofOffsets(),
                            pfes->GetDofOffsets()));
      // - hdA owns the new HypreParMatrix
      // - the above constructor copies all input arrays
      glob_J.DeleteAll();
      dA.ConvertFrom(hdA);
   }

   // TODO - assemble the Dof_TrueDof_Matrix directly in the required format?
   Ph.ConvertFrom(pfes->Dof_TrueDof_Matrix());
   // TODO: When Ph.Type() == Operator::ANY_TYPE we want to use the Operator
   // returned by pfes->GetProlongationMatrix(), however that Operator is a
   // const Operator, so we cannot store it in OperatorHandle. We need a const
   // version of class OperatorHandle, e.g. ConstOperatorHandle.

   A.MakePtAP(dA, Ph);
}

HypreParMatrix *ParBilinearForm::ParallelAssemble(SparseMatrix *m)
{
   OperatorHandle Mh(Operator::Hypre_ParCSR);
   ParallelAssemble(Mh, m);
   Mh.SetOperatorOwner(false);
   return Mh.As<HypreParMatrix>();
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
      int Elem2NbrNo = T->Elem2No - pmesh->GetNE();
      pfes->GetElementVDofs(T->Elem1No, vdofs1);
      pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);
      vdofs1.Copy(vdofs_all);
      for (int j = 0; j < vdofs2.Size(); j++)
      {
         if (vdofs2[j] >= 0)
         {
            vdofs2[j] += height;
         }
         else
         {
            vdofs2[j] -= height;
         }
      }
      vdofs_all.Append(vdofs2);
      for (int k = 0; k < interior_face_integs.Size(); k++)
      {
         interior_face_integs[k]->
         AssembleFaceMatrix(*pfes->GetFE(T->Elem1No),
                            *pfes->GetFaceNbrFE(Elem2NbrNo),
                            *T, elemmat);
         if (keep_nbr_block)
         {
            mat->AddSubMatrix(vdofs_all, vdofs_all, elemmat, skip_zeros);
         }
         else
         {
            mat->AddSubMatrix(vdofs1, vdofs_all, elemmat, skip_zeros);
         }
      }
   }
}

void ParBilinearForm::Assemble(int skip_zeros)
{
   if (interior_face_integs.Size())
   {
      pfes->ExchangeFaceNbrData();
      if (!ext && mat == NULL)
      {
         pAllocMat();
      }
   }

   BilinearForm::Assemble(skip_zeros);

   if (!ext && interior_face_integs.Size() > 0)
   {
      AssembleSharedFaces(skip_zeros);
   }
}

void ParBilinearForm::AssembleDiagonal(Vector &diag) const
{
   MFEM_ASSERT(diag.Size() == fes->GetTrueVSize(),
               "Vector for holding diagonal has wrong size!");
   const Operator *P = fes->GetProlongationMatrix();
   if (!ext)
   {
      MFEM_ASSERT(p_mat.Ptr(), "the ParBilinearForm is not assembled!");
      p_mat->AssembleDiagonal(diag); // TODO: add support for PETSc matrices
      return;
   }
   // Here, we have extension, ext.
   if (IsIdentityProlongation(P))
   {
      ext->AssembleDiagonal(diag);
      return;
   }
   // Here, we have extension, ext, and parallel/conforming prolongation, P.
   Vector local_diag(P->Height());
   ext->AssembleDiagonal(local_diag);
   if (fes->Conforming())
   {
      P->MultTranspose(local_diag, diag);
      return;
   }
   // For an AMR mesh, a convergent diagonal is assembled with |P^T| d_l,
   // where |P^T| has the entry-wise absolute values of the conforming
   // prolongation transpose operator.
   const HypreParMatrix *HP = dynamic_cast<const HypreParMatrix*>(P);
   if (HP)
   {
      HP->AbsMultTranspose(1.0, local_diag, 0.0, diag);
   }
   else
   {
      MFEM_ABORT("unsupported prolongation matrix type.");
   }
}

void ParBilinearForm
::ParallelEliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                               HypreParMatrix &A, const HypreParVector &X,
                               HypreParVector &B) const
{
   Array<int> dof_list;

   pfes->GetEssentialTrueDofs(bdr_attr_is_ess, dof_list);

   // do the parallel elimination
   A.EliminateRowsCols(dof_list, X, B);
}

HypreParMatrix *ParBilinearForm::
ParallelEliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                             HypreParMatrix &A) const
{
   Array<int> dof_list;

   pfes->GetEssentialTrueDofs(bdr_attr_is_ess, dof_list);

   return A.EliminateRowsCols(dof_list);
}

void ParBilinearForm::TrueAddMult(const Vector &x, Vector &y, const double a)
const
{
   if (Xaux.ParFESpace() != pfes)
   {
      Xaux.SetSpace(pfes);
      Yaux.SetSpace(pfes);
      Ytmp.SetSize(pfes->GetTrueVSize());
   }

   Xaux.Distribute(&x);
   if (ext)
   {
      ext->Mult(Xaux, Yaux);
   }
   else
   {
      MFEM_VERIFY(interior_face_integs.Size() == 0,
                  "the case of interior face integrators is not"
                  " implemented");
      mat->Mult(Xaux, Yaux);
   }
   pfes->GetProlongationMatrix()->MultTranspose(Yaux, Ytmp);
   y.Add(a,Ytmp);
}

void ParBilinearForm::FormLinearSystem(
   const Array<int> &ess_tdof_list, Vector &x, Vector &b,
   OperatorHandle &A, Vector &X, Vector &B, int copy_interior)
{
   if (ext)
   {
      ext->FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);
      return;
   }

   // Finish the matrix assembly and perform BC elimination, storing the
   // eliminated part of the matrix.
   FormSystemMatrix(ess_tdof_list, A);

   const Operator &P = *pfes->GetProlongationMatrix();
   const SparseMatrix &R = *pfes->GetRestrictionMatrix();

   // Transform the system and perform the elimination in B, based on the
   // essential BC values from x. Restrict the BC part of x in X, and set the
   // non-BC part to zero. Since there is no good initial guess for the Lagrange
   // multipliers, set X = 0.0 for hybridization.
   if (static_cond)
   {
      // Schur complement reduction to the exposed dofs
      static_cond->ReduceSystem(x, b, X, B, copy_interior);
   }
   else if (hybridization)
   {
      // Reduction to the Lagrange multipliers system
      HypreParVector true_X(pfes), true_B(pfes);
      P.MultTranspose(b, true_B);
      R.Mult(x, true_X);
      p_mat.EliminateBC(p_mat_e, ess_tdof_list, true_X, true_B);
      R.EnsureMultTranspose();
      R.MultTranspose(true_B, b);
      hybridization->ReduceRHS(true_B, B);
      X.SetSize(B.Size());
      X = 0.0;
   }
   else
   {
      // Variational restriction with P
      X.SetSize(P.Width());
      B.SetSize(X.Size());
      P.MultTranspose(b, B);
      R.Mult(x, X);
      p_mat.EliminateBC(p_mat_e, ess_tdof_list, X, B);
      if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
   }
}

void ParBilinearForm::EliminateVDofsInRHS(
   const Array<int> &vdofs, const Vector &x, Vector &b)
{
   p_mat.EliminateBC(p_mat_e, vdofs, x, b);
}

void ParBilinearForm::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                       OperatorHandle &A)
{
   if (ext)
   {
      ext->FormSystemMatrix(ess_tdof_list, A);
      return;
   }

   // Finish the matrix assembly and perform BC elimination, storing the
   // eliminated part of the matrix.
   if (static_cond)
   {
      if (!static_cond->HasEliminatedBC())
      {
         static_cond->SetEssentialTrueDofs(ess_tdof_list);
         static_cond->Finalize();
         static_cond->EliminateReducedTrueDofs(Matrix::DIAG_ONE);
      }
      static_cond->GetParallelMatrix(A);
   }
   else
   {
      if (mat)
      {
         const int remove_zeros = 0;
         Finalize(remove_zeros);
         MFEM_VERIFY(p_mat.Ptr() == NULL && p_mat_e.Ptr() == NULL,
                     "The ParBilinearForm must be updated with Update() before "
                     "re-assembling the ParBilinearForm.");
         ParallelAssemble(p_mat, mat);
         delete mat;
         mat = NULL;
         delete mat_e;
         mat_e = NULL;
         p_mat_e.EliminateRowsCols(p_mat, ess_tdof_list);
      }
      if (hybridization)
      {
         hybridization->GetParallelMatrix(A);
      }
      else
      {
         A = p_mat;
      }
   }
}

void ParBilinearForm::RecoverFEMSolution(
   const Vector &X, const Vector &b, Vector &x)
{
   if (ext)
   {
      ext->RecoverFEMSolution(X, b, x);
      return;
   }

   const Operator &P = *pfes->GetProlongationMatrix();

   if (static_cond)
   {
      // Private dofs back solve
      static_cond->ComputeSolution(b, X, x);
   }
   else if (hybridization)
   {
      // Primal unknowns recovery
      HypreParVector true_X(pfes), true_B(pfes);
      P.MultTranspose(b, true_B);
      const SparseMatrix &R = *pfes->GetRestrictionMatrix();
      R.Mult(x, true_X); // get essential b.c. from x
      hybridization->ComputeSolution(true_B, X, true_X);
      x.SetSize(P.Height());
      P.Mult(true_X, x);
   }
   else
   {
      // Apply conforming prolongation
      x.SetSize(P.Height(), GetHypreMemoryType());
      P.Mult(X, x);
   }
}

void ParBilinearForm::Update(FiniteElementSpace *nfes)
{
   BilinearForm::Update(nfes);

   if (nfes)
   {
      pfes = dynamic_cast<ParFiniteElementSpace *>(nfes);
      MFEM_VERIFY(pfes != NULL, "nfes must be a ParFiniteElementSpace!");
   }

   p_mat.Clear();
   p_mat_e.Clear();
}


HypreParMatrix *ParMixedBilinearForm::ParallelAssemble()
{
   // construct the block-diagonal matrix A
   HypreParMatrix *A =
      new HypreParMatrix(trial_pfes->GetComm(),
                         test_pfes->GlobalVSize(),
                         trial_pfes->GlobalVSize(),
                         test_pfes->GetDofOffsets(),
                         trial_pfes->GetDofOffsets(),
                         mat);

   HypreParMatrix *rap = RAP(test_pfes->Dof_TrueDof_Matrix(), A,
                             trial_pfes->Dof_TrueDof_Matrix());

   delete A;

   return rap;
}

void ParMixedBilinearForm::ParallelAssemble(OperatorHandle &A)
{
   // construct the rectangular block-diagonal matrix dA
   OperatorHandle dA(A.Type());
   dA.MakeRectangularBlockDiag(trial_pfes->GetComm(),
                               test_pfes->GlobalVSize(),
                               trial_pfes->GlobalVSize(),
                               test_pfes->GetDofOffsets(),
                               trial_pfes->GetDofOffsets(),
                               mat);

   OperatorHandle P_test(A.Type()), P_trial(A.Type());

   // TODO - construct the Dof_TrueDof_Matrix directly in the required format.
   P_test.ConvertFrom(test_pfes->Dof_TrueDof_Matrix());
   P_trial.ConvertFrom(trial_pfes->Dof_TrueDof_Matrix());

   A.MakeRAP(P_test, dA, P_trial);
}

/// Compute y += a (P^t A P) x, where x and y are vectors on the true dofs
void ParMixedBilinearForm::TrueAddMult(const Vector &x, Vector &y,
                                       const double a) const
{
   if (Xaux.ParFESpace() != trial_pfes)
   {
      Xaux.SetSpace(trial_pfes);
      Yaux.SetSpace(test_pfes);
   }

   Xaux.Distribute(&x);
   mat->Mult(Xaux, Yaux);
   test_pfes->Dof_TrueDof_Matrix()->MultTranspose(a, Yaux, 1.0, y);
}

void ParMixedBilinearForm::FormRectangularSystemMatrix(
   const Array<int>
   &trial_tdof_list,
   const Array<int> &test_tdof_list,
   OperatorHandle &A)
{
   if (ext)
   {
      ext->FormRectangularSystemOperator(trial_tdof_list, test_tdof_list, A);
      return;
   }

   if (mat)
   {
      Finalize();
      ParallelAssemble(p_mat);
      delete mat;
      mat = NULL;
      delete mat_e;
      mat_e = NULL;
      HypreParMatrix *temp =
         p_mat.As<HypreParMatrix>()->EliminateCols(trial_tdof_list);
      p_mat.As<HypreParMatrix>()->EliminateRows(test_tdof_list);
      p_mat_e.Reset(temp, true);
   }

   A = p_mat;
}

void ParMixedBilinearForm::FormRectangularLinearSystem(
   const Array<int>
   &trial_tdof_list,
   const Array<int> &test_tdof_list, Vector &x,
   Vector &b, OperatorHandle &A, Vector &X,
   Vector &B)
{
   if (ext)
   {
      ext->FormRectangularLinearSystem(trial_tdof_list, test_tdof_list,
                                       x, b, A, X, B);
      return;
   }

   FormRectangularSystemMatrix(trial_tdof_list, test_tdof_list, A);

   const Operator *test_P = test_pfes->GetProlongationMatrix();
   const SparseMatrix *trial_R = trial_pfes->GetRestrictionMatrix();

   X.SetSize(trial_pfes->TrueVSize());
   B.SetSize(test_pfes->TrueVSize());
   test_P->MultTranspose(b, B);
   trial_R->Mult(x, X);

   p_mat_e.As<HypreParMatrix>()->Mult(-1.0, X, 1.0, B);
   B.SetSubVector(test_tdof_list, 0.0);
}

HypreParMatrix* ParDiscreteLinearOperator::ParallelAssemble() const
{
   MFEM_ASSERT(mat, "Matrix is not assembled");
   MFEM_ASSERT(mat->Finalized(), "Matrix is not finalized");
   SparseMatrix* RA = mfem::Mult(*range_fes->GetRestrictionMatrix(), *mat);
   HypreParMatrix* P = domain_fes->Dof_TrueDof_Matrix();
   HypreParMatrix* RAP = P->LeftDiagMult(*RA, range_fes->GetTrueDofOffsets());
   delete RA;
   return RAP;
}

void ParDiscreteLinearOperator::ParallelAssemble(OperatorHandle &A)
{
   // construct the rectangular block-diagonal matrix dA
   OperatorHandle dA(A.Type());
   dA.MakeRectangularBlockDiag(domain_fes->GetComm(),
                               range_fes->GlobalVSize(),
                               domain_fes->GlobalVSize(),
                               range_fes->GetDofOffsets(),
                               domain_fes->GetDofOffsets(),
                               mat);

   SparseMatrix *Rt = Transpose(*range_fes->GetRestrictionMatrix());
   OperatorHandle R_test_transpose(A.Type());
   R_test_transpose.MakeRectangularBlockDiag(range_fes->GetComm(),
                                             range_fes->GlobalVSize(),
                                             range_fes->GlobalTrueVSize(),
                                             range_fes->GetDofOffsets(),
                                             range_fes->GetTrueDofOffsets(),
                                             Rt);

   // TODO - construct the Dof_TrueDof_Matrix directly in the required format.
   OperatorHandle P_trial(A.Type());
   P_trial.ConvertFrom(domain_fes->Dof_TrueDof_Matrix());

   A.MakeRAP(R_test_transpose, dA, P_trial);
   delete Rt;
}

void ParDiscreteLinearOperator::FormRectangularSystemMatrix(OperatorHandle &A)
{
   if (ext)
   {
      Array<int> empty;
      ext->FormRectangularSystemOperator(empty, empty, A);
      return;
   }

   mfem_error("not implemented!");
}

void ParDiscreteLinearOperator::GetParBlocks(Array2D<HypreParMatrix *> &blocks)
const
{
   MFEM_VERIFY(mat->Finalized(), "Local matrix needs to be finalized for "
               "GetParBlocks");

   HypreParMatrix* RLP = ParallelAssemble();

   blocks.SetSize(range_fes->GetVDim(), domain_fes->GetVDim());

   RLP->GetBlocks(blocks,
                  range_fes->GetOrdering() == Ordering::byVDIM,
                  domain_fes->GetOrdering() == Ordering::byVDIM);

   delete RLP;
}

}

#endif
