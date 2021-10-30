// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem.hpp"

namespace mfem
{

BlockBilinearForm::BlockBilinearForm(Array<FiniteElementSpace *> & fespaces_) :
   Matrix(0), fespaces(fespaces_)
{
   height = 0;
   int nblocks = fespaces.Size();
   for (int i =0; i<nblocks; i++)
   {
      height += fespaces[i]->GetVSize();
   }
   width = height;
   mat = mat_e = NULL;
   extern_bfs = 0;
   element_matrices = NULL;
   diag_policy = DIAG_KEEP;
}


// Allocate appropriate SparseMatrix and assign it to mat
void BlockBilinearForm::AllocMat()
{
   mat = new SparseMatrix(height);
}

void BlockBilinearForm::ConformingAssemble()
{
   // TODO
   // Finalize(0);
   // MFEM_ASSERT(mat, "the BilinearForm is not assembled");

}

void BlockBilinearForm::Mult(const Vector &x, Vector &y) const
{
   // TODO
}


double& BlockBilinearForm::Elem (int i, int j)
{
   return mat -> Elem(i,j);
}

const double& BlockBilinearForm::Elem (int i, int j) const
{
   return mat -> Elem(i,j);
}

MatrixInverse * BlockBilinearForm::Inverse() const
{
   return mat -> Inverse();
}

void BlockBilinearForm::Finalize(int skip_zeros)
{
   mat->Finalize(skip_zeros);
   if (mat_e) { mat_e->Finalize(skip_zeros); }
}

/// Adds new Block Domain Integrator. Assumes ownership of @a bfi.
void BlockBilinearForm::AddDomainIntegrator(BlockBilinearFormIntegrator *bfi)
{
   domain_integs.Append(bfi);
}

/// Assembles the form i.e. sums over all domain integrators.
void BlockBilinearForm::Assemble(int skip_zeros)
{
   ElementTransformation *eltrans;
   DofTransformation * doftrans_j, *doftrans_k;
   Mesh *mesh = fespaces[0] -> GetMesh();
   DenseMatrix elmat, *elmat_p;
   int nblocks = fespaces.Size();
   Array<const FiniteElement *> fe(nblocks);
   Array<int> vdofs_j, vdofs_k;
   Array<int> offsetvdofs_j, offsetvdofs_k;
   Array<int> elementblockoffsets(nblocks+1);
   elementblockoffsets[0] = 0;
   Array<int> blockoffsets(nblocks+1);
   blockoffsets[0] = 0;
   for (int i =0; i<nblocks; i++)
   {
      blockoffsets[i+1] = fespaces[i]->GetVSize();
   }
   blockoffsets.PartialSum();
   // mfem::out << "blockoffsets = " ; blockoffsets.Print();

   if (mat == NULL)
   {
      AllocMat();
   }

   if (domain_integs.Size())
   {
      // loop through elements
      for (int i = 0; i < mesh -> GetNE(); i++)
      {
         if (element_matrices)
         {
            elmat_p = &(*element_matrices)(i);
         }
         else
         {
            elmat.SetSize(0);
            for (int k = 0; k < domain_integs.Size(); k++)
            {
               for (int j = 0; j<nblocks; j++)
               {
                  fe[j] = fespaces[j]->GetFE(i);
                  elementblockoffsets[j+1] = fe[j]->GetDof();
               }
               elementblockoffsets.PartialSum();
               eltrans = mesh->GetElementTransformation(i);
               domain_integs[k]->AssembleElementMatrix(fe, *eltrans, elemmat);
               if (elmat.Size() == 0)
               {
                  elmat = elemmat;
               }
               else
               {
                  elmat += elemmat;
               }
            }
         }
         if (elmat.Size() == 0)
         {
            continue;
         }
         else
         {
            elmat_p = &elmat;
         }
         for (int j = 0; j<nblocks; j++)
         {
            doftrans_j = fespaces[j]->GetElementVDofs(i, vdofs_j);
            int offset_j = blockoffsets[j];
            // mfem::out << "offset_j = " << offset_j << std::endl;
            offsetvdofs_j.SetSize(vdofs_j.Size());
            for (int l = 0; l<vdofs_j.Size(); l++)
            {
               MFEM_WARNING("need to modify matrix dofs according to vdofs sign");
               int vdof_idx = vdofs_j[l] < 0 ? -1 -vdofs_j[l] : vdofs_j[l];
               offsetvdofs_j[l] = offset_j + vdof_idx;
            }
            for (int k = 0; k<nblocks; k++)
            {
               doftrans_k = fespaces[k]->GetElementVDofs(i, vdofs_k);
               int offset_k = blockoffsets[k];
               mfem::out << "offset_k = " << offset_k << std::endl;
               offsetvdofs_k.SetSize(vdofs_k.Size());
               for (int l = 0; l<vdofs_k.Size(); l++)
               {
                  MFEM_WARNING("need to modify matrix dofs according to vdofs sign");
                  int vdof_idx = vdofs_k[l] < 0 ? -1 - vdofs_k[l] : vdofs_k[l];
                  offsetvdofs_k[l] = offset_k + vdof_idx;
               }
               // extract sub matrix (using elementblockoffsets)
               DenseMatrix A;
               int jbeg = elementblockoffsets[j];
               int jend = elementblockoffsets[j+1]-1;
               int kbeg = elementblockoffsets[k];
               int kend = elementblockoffsets[k+1]-1;
               elmat_p->GetSubMatrix(jbeg,jend,kbeg, kend, A);
               if (doftrans_k || doftrans_j)
               {
                  TransformDual(doftrans_j, doftrans_k, A);
               }
               // offsetvdofs_j.Print();
               // offsetvdofs_k.Print();
               mat->AddSubMatrix(offsetvdofs_j, offsetvdofs_k, A, skip_zeros);
            }
         }
      }
   }
}


void BlockBilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list,
                                         Vector &x,
                                         Vector &b, OperatorHandle &A, Vector &X,
                                         Vector &B, int copy_interior)
{
   const SparseMatrix *P = fespaces[0]->GetConformingProlongation();
   if (P)
   {
      MFEM_ABORT("BlockBilinearForm::FormLinearSystem:: Non-conforming not implemented yet")
   }
   FormSystemMatrix(ess_tdof_list, A);

   EliminateVDofsInRHS(ess_tdof_list, x, b);
   X.MakeRef(x, 0, x.Size());
   B.MakeRef(b, 0, b.Size());
   if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }

}

void BlockBilinearForm::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                         OperatorHandle &A)
{
   if (!mat_e)
   {
      const SparseMatrix *P = fespaces[0]->GetConformingProlongation();
      if (P)
      {
         MFEM_ABORT("BlockBilinearForm::FormSystemMatrix:: Non-conforming not implemented yet")
      }
      EliminateVDofs(ess_tdof_list, diag_policy);
      const int remove_zeros = 0;
      Finalize(remove_zeros);
   }
   A.Reset(mat, false);
}

void BlockBilinearForm::RecoverFEMSolution(const Vector &X, const Vector &b,
                                           Vector &x)
{
   const SparseMatrix *P = fespaces[0]->GetConformingProlongation();
   if (P)
   {
      MFEM_ABORT("BlockBilinearForm::RecoverFEMSolution:: Non-conforming not implemented yet")
   }
   x.SyncMemory(X);
}



void BlockBilinearForm::ComputeElementMatrices()
{
   MFEM_ABORT("BlockBilinearForm::ComputeElementMatrices:not implemented yet")
}

void BlockBilinearForm::ComputeElementMatrix(int i, DenseMatrix &elmat)
{
   if (element_matrices)
   {
      elmat.SetSize(element_matrices->SizeI(), element_matrices->SizeJ());
      elmat = element_matrices->GetData(i);
      return;
   }

   int nblocks = fespaces.Size();
   Array<const FiniteElement *> fe(nblocks);
   ElementTransformation *eltrans;

   elmat.SetSize(0);
   if (domain_integs.Size())
   {
      for (int j = 0; j<nblocks; j++)
      {
         fe[j] = fespaces[j]->GetFE(i);
      }
      eltrans = fespaces[0]->GetElementTransformation(i);
      domain_integs[0]->AssembleElementMatrix(fe, *eltrans, elmat);
      for (int k = 1; k < domain_integs.Size(); k++)
      {
         domain_integs[k]->AssembleElementMatrix(fe, *eltrans, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      int matsize = 0;
      for (int j = 0; j<nblocks; j++)
      {
         matsize += fespaces[j]->GetFE(i)->GetDof();
      }
      elmat.SetSize(matsize);
      elmat = 0.0;
   }
}

void BlockBilinearForm::EliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                                             const Vector &sol, Vector &rhs,
                                             DiagonalPolicy dpolicy)
{
   MFEM_ABORT("BlockBilinearForm::EliminateEssentialBC: not implemented yet");
   // Array<int> ess_dofs, conf_ess_dofs;
   // fes->GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   // if (fes->GetVSize() == height)
   // {
   //    EliminateEssentialBCFromDofs(ess_dofs, sol, rhs, dpolicy);
   // }
   // else
   // {
   //    fes->GetRestrictionMatrix()->BooleanMult(ess_dofs, conf_ess_dofs);
   //    EliminateEssentialBCFromDofs(conf_ess_dofs, sol, rhs, dpolicy);
   // }
}

void BlockBilinearForm::EliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                                             DiagonalPolicy dpolicy)
{
   MFEM_ABORT("BlockBilinearForm::EliminateEssentialBC: not implemented yet");
   // Array<int> ess_dofs, conf_ess_dofs;
   // fes->GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   // if (fes->GetVSize() == height)
   // {
   //    EliminateEssentialBCFromDofs(ess_dofs, dpolicy);
   // }
   // else
   // {
   //    fes->GetRestrictionMatrix()->BooleanMult(ess_dofs, conf_ess_dofs);
   //    EliminateEssentialBCFromDofs(conf_ess_dofs, dpolicy);
   // }
}

void BlockBilinearForm::EliminateEssentialBCDiag (const Array<int>
                                                  &bdr_attr_is_ess,
                                                  double value)
{
   MFEM_ABORT("BlockBilinearForm::EliminateEssentialBCDiag: not implemented yet");
   // Array<int> ess_dofs, conf_ess_dofs;
   // fes->GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   // if (fes->GetVSize() == height)
   // {
   //    EliminateEssentialBCFromDofsDiag(ess_dofs, value);
   // }
   // else
   // {
   //    fes->GetRestrictionMatrix()->BooleanMult(ess_dofs, conf_ess_dofs);
   //    EliminateEssentialBCFromDofsDiag(conf_ess_dofs, value);
   // }
}

void BlockBilinearForm::EliminateVDofs(const Array<int> &vdofs,
                                       const Vector &sol, Vector &rhs,
                                       DiagonalPolicy dpolicy)
{
   vdofs.HostRead();
   for (int i = 0; i < vdofs.Size(); i++)
   {
      int vdof = vdofs[i];
      if ( vdof >= 0 )
      {
         mat -> EliminateRowCol (vdof, sol(vdof), rhs, dpolicy);
      }
      else
      {
         mat -> EliminateRowCol (-1-vdof, sol(-1-vdof), rhs, dpolicy);
      }
   }
}

void BlockBilinearForm::EliminateVDofs(const Array<int> &vdofs,
                                       DiagonalPolicy dpolicy)
{
   if (mat_e == NULL)
   {
      mat_e = new SparseMatrix(height);
   }

   for (int i = 0; i < vdofs.Size(); i++)
   {
      int vdof = vdofs[i];
      if ( vdof >= 0 )
      {
         mat -> EliminateRowCol (vdof, *mat_e, dpolicy);
      }
      else
      {
         mat -> EliminateRowCol (-1-vdof, *mat_e, dpolicy);
      }
   }
}

void BlockBilinearForm::EliminateEssentialBCFromDofs(
   const Array<int> &ess_dofs, const Vector &sol, Vector &rhs,
   DiagonalPolicy dpolicy)
{
   MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");
   MFEM_ASSERT(sol.Size() == height, "incorrect sol Vector size");
   MFEM_ASSERT(rhs.Size() == height, "incorrect rhs Vector size");

   for (int i = 0; i < ess_dofs.Size(); i++)
   {
      if (ess_dofs[i] < 0)
      {
         mat -> EliminateRowCol (i, sol(i), rhs, dpolicy);
      }
   }
}

void BlockBilinearForm::EliminateEssentialBCFromDofs (const Array<int>
                                                      &ess_dofs,
                                                      DiagonalPolicy dpolicy)
{
   MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");

   for (int i = 0; i < ess_dofs.Size(); i++)
   {
      if (ess_dofs[i] < 0)
      {
         mat -> EliminateRowCol (i, dpolicy);
      }
   }
}

void BlockBilinearForm::EliminateEssentialBCFromDofsDiag (
   const Array<int> &ess_dofs,
   double value)
{
   MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");

   for (int i = 0; i < ess_dofs.Size(); i++)
   {
      if (ess_dofs[i] < 0)
      {
         mat -> EliminateRowColDiag (i, value);
      }
   }
}

void BlockBilinearForm::EliminateVDofsInRHS(
   const Array<int> &vdofs, const Vector &x, Vector &b)
{
   mat_e->AddMult(x, b, -1.);
   mat->PartMult(vdofs, x, b);
}



BlockBilinearForm::~BlockBilinearForm()
{
   delete mat_e;
   delete mat;
   delete element_matrices;

   if (!extern_bfs)
   {
      int k;
      for (k=0; k < domain_integs.Size(); k++) { delete domain_integs[k]; }
   }
}




} // namespace mfem
