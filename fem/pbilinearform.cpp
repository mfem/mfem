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
   double *data = new double[I[nrows]];

   mat = new SparseMatrix(I, J, data, nrows, height + nbr_size);
   *mat = 0.0;

   dof_dof.LoseData();
}

void ParBilinearForm::ParallelAssemble(OperatorHandle &A, SparseMatrix *A_local)
{
   A.Clear();

   if (A_local == NULL) { return; }
   MFEM_VERIFY(A_local->Finalized(), "the local matrix must be finalized");

   OperatorHandle dA(A.Type()), Ph(A.Type()), hdA;

   if (fbfi.Size() == 0)
   {
      // construct a parallel block-diagonal matrix 'A' based on 'a'
      dA.MakeSquareBlockDiag(pfes->GetComm(), pfes->GlobalVSize(),
                             pfes->GetDofOffsets(), A_local);
   }
   else
   {
      // handle the case when 'a' contains offdiagonal
      int lvsize = pfes->GetVSize();
      const HYPRE_Int *face_nbr_glob_ldof = pfes->GetFaceNbrGlobalDofMap();
      HYPRE_Int ldof_offset = pfes->GetMyDofOffset();

      Array<HYPRE_Int> glob_J(A_local->NumNonZeroElems());
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
      pfes->GetElementVDofs(T->Elem1No, vdofs1);
      pfes->GetFaceNbrElementVDofs(T->Elem2No, vdofs2);
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
      for (int k = 0; k < fbfi.Size(); k++)
      {
         fbfi[k]->AssembleFaceMatrix(*pfes->GetFE(T->Elem1No),
                                     *pfes->GetFaceNbrFE(T->Elem2No),
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
   if (mat == NULL && fbfi.Size() > 0)
   {
      pfes->ExchangeFaceNbrData();
      pAllocMat();
   }

   BilinearForm::Assemble(skip_zeros);

   if (fbfi.Size() > 0)
   {
      AssembleSharedFaces(skip_zeros);
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
   MFEM_VERIFY(fbfi.Size() == 0, "the case of interior face integrators is not"
               " implemented");

   if (X.ParFESpace() != pfes)
   {
      X.SetSpace(pfes);
      Y.SetSpace(pfes);
   }

   X.Distribute(&x);
   mat->Mult(X, Y);
   pfes->Dof_TrueDof_Matrix()->MultTranspose(a, Y, 1.0, y);
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
      R.MultTranspose(true_B, b);
      hybridization->ReduceRHS(true_B, B);
      X.SetSize(B.Size());
      X = 0.0;
   }
   else
   {
      // Variational restriction with P
      X.SetSize(pfes->TrueVSize());
      B.SetSize(X.Size());
      P.MultTranspose(b, B);
      R.Mult(x, X);
      p_mat.EliminateBC(p_mat_e, ess_tdof_list, X, B);
      if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
   }
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
      x.SetSize(P.Height());
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
   if (X.ParFESpace() != trial_pfes)
   {
      X.SetSpace(trial_pfes);
      Y.SetSpace(test_pfes);
   }

   X.Distribute(&x);
   mat->Mult(X, Y);
   test_pfes->Dof_TrueDof_Matrix()->MultTranspose(a, Y, 1.0, y);
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
