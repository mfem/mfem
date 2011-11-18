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

// Implementation of class BilinearForm

#include "fem.hpp"
#include <math.h>

BilinearForm::BilinearForm (FiniteElementSpace * f)
   : Matrix (f->GetVSize())
{
   fes = f;
   mat = mat_e = NULL;
   extern_bfs = 0;
   element_matrices = NULL;
}

BilinearForm::BilinearForm (FiniteElementSpace * f, BilinearForm * bf)
   : Matrix (f->GetVSize())
{
   int i;
   Array<BilinearFormIntegrator*> *bfi;

   fes = f;
   mat = new SparseMatrix (size);
   mat_e = NULL;
   extern_bfs = 1;
   element_matrices = NULL;

   bfi = bf->GetDBFI();
   dbfi.SetSize (bfi->Size());
   for (i = 0; i < bfi->Size(); i++)
      dbfi[i] = (*bfi)[i];

   bfi = bf->GetBBFI();
   bbfi.SetSize (bfi->Size());
   for (i = 0; i < bfi->Size(); i++)
      bbfi[i] = (*bfi)[i];

   bfi = bf->GetFBFI();
   fbfi.SetSize (bfi->Size());
   for (i = 0; i < bfi->Size(); i++)
      fbfi[i] = (*bfi)[i];

   bfi = bf->GetBFBFI();
   bfbfi.SetSize (bfi->Size());
   for (i = 0; i < bfi->Size(); i++)
      bfbfi[i] = (*bfi)[i];
}

double& BilinearForm::Elem (int i, int j)
{
   return mat -> Elem(i,j);
}

const double& BilinearForm::Elem (int i, int j) const
{
   return mat -> Elem(i,j);
}

void BilinearForm::Mult (const Vector & x, Vector & y) const
{
   mat -> Mult (x, y);
}

MatrixInverse * BilinearForm::Inverse() const
{
   return mat -> Inverse();
}

void BilinearForm::Finalize (int skip_zeros)
{
   mat -> Finalize (skip_zeros);
   if (mat_e)
      mat_e -> Finalize (skip_zeros);
}

void BilinearForm::AddDomainIntegrator (BilinearFormIntegrator * bfi)
{
   dbfi.Append (bfi);
}

void BilinearForm::AddBoundaryIntegrator (BilinearFormIntegrator * bfi)
{
   bbfi.Append (bfi);
}

void BilinearForm::AddInteriorFaceIntegrator (BilinearFormIntegrator * bfi)
{
   fbfi.Append (bfi);
}

void BilinearForm::AddBdrFaceIntegrator (BilinearFormIntegrator * bfi)
{
   bfbfi.Append (bfi);
}

void BilinearForm::ComputeElementMatrix(int i, DenseMatrix &elmat)
{
   if (element_matrices)
   {
      DenseMatrix tmp(element_matrices->GetData(i),
                      element_matrices->SizeI(),
                      element_matrices->SizeJ());
      elmat = tmp;
      return;
   }

   if (dbfi.Size())
   {
      const FiniteElement &fe = *fes->GetFE(i);
      ElementTransformation *eltrans = fes->GetElementTransformation(i);
      dbfi[0]->AssembleElementMatrix(fe, *eltrans, elmat);
      for (int k = 1; k < dbfi.Size(); k++)
      {
         dbfi[k]->AssembleElementMatrix(fe, *eltrans, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      fes->GetElementVDofs(i, vdofs);
      elmat.SetSize(vdofs.Size());
      elmat = 0.0;
   }
}

void BilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, Array<int> &vdofs, int skip_zeros)
{
   if (mat == NULL)
      mat = new SparseMatrix(size);
   fes->GetElementVDofs(i, vdofs);
   mat->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
}

void BilinearForm::Assemble (int skip_zeros)
{
   ElementTransformation *eltrans;
   Mesh *mesh = fes -> GetMesh();

   int i;

   if (mat == NULL)
      mat = new SparseMatrix(size);

#ifdef MFEM_USE_OPENMP
   int free_element_matrices = 0;
   if (!element_matrices)
   {
      ComputeElementMatrices();
      free_element_matrices = 1;
   }
#endif

   if (dbfi.Size())
      for (i = 0; i < fes -> GetNE(); i++)
      {
         fes->GetElementVDofs(i, vdofs);
         if (element_matrices)
         {
            mat->AddSubMatrix(vdofs, vdofs, (*element_matrices)(i), skip_zeros);
         }
         else
         {
            const FiniteElement &fe = *fes->GetFE(i);
            eltrans = fes->GetElementTransformation(i);
            for (int k = 0; k < dbfi.Size(); k++)
            {
               dbfi[k]->AssembleElementMatrix(fe, *eltrans, elemmat);
               mat->AddSubMatrix(vdofs, vdofs, elemmat, skip_zeros);
            }
         }
      }

   if (bbfi.Size())
      for (i = 0; i < fes -> GetNBE(); i++)
      {
         const FiniteElement &be = *fes->GetBE(i);
         fes -> GetBdrElementVDofs (i, vdofs);
         eltrans = fes -> GetBdrElementTransformation (i);
         for (int k=0; k < bbfi.Size(); k++)
         {
            bbfi[k] -> AssembleElementMatrix(be, *eltrans, elemmat);
            mat -> AddSubMatrix (vdofs, vdofs, elemmat, skip_zeros);
         }
      }

   if (fbfi.Size())
   {
      FaceElementTransformations *tr;
      Array<int> vdofs2;

      int nfaces;
      if (mesh -> Dimension() == 2)
         nfaces = mesh -> GetNEdges();
      else
         nfaces = mesh -> GetNFaces();
      for (i = 0; i < nfaces; i++)
      {
         tr = mesh -> GetInteriorFaceTransformations (i);
         if (tr != NULL)
         {
            fes -> GetElementVDofs (tr -> Elem1No, vdofs);
            fes -> GetElementVDofs (tr -> Elem2No, vdofs2);
            vdofs.Append (vdofs2);
            for (int k = 0; k < fbfi.Size(); k++)
            {
               fbfi[k] -> AssembleFaceMatrix (*fes -> GetFE (tr -> Elem1No),
                                              *fes -> GetFE (tr -> Elem2No),
                                              *tr, elemmat);
               mat -> AddSubMatrix (vdofs, vdofs, elemmat, skip_zeros);
            }
         }
      }
   }

   if (bfbfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *nfe = NULL;

      for (i = 0; i < fes -> GetNBE(); i++)
      {
         tr = mesh -> GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            fes -> GetElementVDofs (tr -> Elem1No, vdofs);
            for (int k = 0; k < bfbfi.Size(); k++)
            {
               bfbfi[k] -> AssembleFaceMatrix (*fes -> GetFE (tr -> Elem1No),
                                               *nfe, *tr, elemmat);
               mat -> AddSubMatrix (vdofs, vdofs, elemmat, skip_zeros);
            }
         }
      }
   }

#ifdef MFEM_USE_OPENMP
   if (free_element_matrices)
      FreeElementMatrices();
#endif
}

void BilinearForm::ComputeElementMatrices()
{
   if (element_matrices || dbfi.Size() == 0)
      return;

   int num_elements = fes->GetNE();
   int num_dofs_per_el = fes->GetFE(0)->GetDof() * fes->GetVDim();

   element_matrices = new DenseTensor(num_dofs_per_el, num_dofs_per_el,
                                      num_elements);

   DenseMatrix tmp;
   IsoparametricTransformation eltrans;

#ifdef MFEM_USE_OPENMP
#pragma omp parallel for private(tmp,eltrans)
#endif
   for (int i = 0; i < num_elements; i++)
   {
      DenseMatrix elmat(element_matrices->GetData(i),
                        num_dofs_per_el, num_dofs_per_el);
      const FiniteElement &fe = *fes->GetFE(i);
#ifdef MFEM_DEBUG
      if (num_dofs_per_el != fe.GetDof()*fes->GetVDim())
         mfem_error("BilinearForm::ComputeElementMatrices:"
                    " all elements must have same number of dofs");
#endif
      fes->GetElementTransformation(i, &eltrans);

      dbfi[0]->AssembleElementMatrix(fe, eltrans, elmat);
      for (int k = 1; k < dbfi.Size(); k++)
      {
         // note: some integrators may not be thread-safe
         dbfi[k]->AssembleElementMatrix(fe, eltrans, tmp);
         elmat += tmp;
      }
      elmat.ClearExternalData();
   }
}

void BilinearForm::EliminateEssentialBC (
   Array<int> &bdr_attr_is_ess, Vector &sol, Vector &rhs, int d )
{
   int i, j, k;

   for (i = 0; i < fes -> GetNBE(); i++)
      if (bdr_attr_is_ess[fes -> GetBdrAttribute (i)-1])
      {
         fes -> GetBdrElementVDofs (i, vdofs);
         for (j = 0; j < vdofs.Size(); j++)
            if ( (k = vdofs[j]) >= 0 )
               mat -> EliminateRowCol (k, sol(k), rhs, d);
            else
               mat -> EliminateRowCol (-1-k, sol(-1-k), rhs, d);
      }
}

void BilinearForm::EliminateVDofs (
   Array<int> &vdofs, Vector &sol, Vector &rhs, int d)
{
   for (int i = 0; i < vdofs.Size(); i++)
   {
      int vdof = vdofs[i];
      if ( vdof >= 0 )
         mat -> EliminateRowCol (vdof, sol(vdof), rhs, d);
      else
         mat -> EliminateRowCol (-1-vdof, sol(-1-vdof), rhs, d);
   }
}

void BilinearForm::EliminateVDofs(Array<int> &vdofs, int d)
{
   if (mat_e == NULL)
      mat_e = new SparseMatrix(size);

   for (int i = 0; i < vdofs.Size(); i++)
   {
      int vdof = vdofs[i];
      if ( vdof >= 0 )
         mat -> EliminateRowCol (vdof, *mat_e, d);
      else
         mat -> EliminateRowCol (-1-vdof, *mat_e, d);
   }
}

void BilinearForm::EliminateVDofsInRHS(
   Array<int> &vdofs, const Vector &x, Vector &b)
{
   mat_e->AddMult(x, b, -1.);
   mat->PartMult(vdofs, x, b);
}

void BilinearForm::EliminateEssentialBC (Array<int> &bdr_attr_is_ess, int d)
{
   int i, j, k;
   Array<int> vdofs;

   for (i = 0; i < fes -> GetNBE(); i++)
      if (bdr_attr_is_ess[fes -> GetBdrAttribute (i)-1])
      {
         fes -> GetBdrElementVDofs (i, vdofs);
         for (j = 0; j < vdofs.Size(); j++)
            if ( (k = vdofs[j]) >= 0 )
               mat -> EliminateRowCol (k, d);
            else
               mat -> EliminateRowCol (-1-k, d);
      }
}

void BilinearForm::EliminateEssentialBCFromDofs (
   Array<int> &ess_dofs, Vector &sol, Vector &rhs, int d )
{
   for (int i = 0; i < ess_dofs.Size(); i++)
      if (ess_dofs[i] < 0)
         mat -> EliminateRowCol (i, sol(i), rhs, d);
}

void BilinearForm::EliminateEssentialBCFromDofs (Array<int> &ess_dofs, int d)
{
   for (int i = 0; i < ess_dofs.Size(); i++)
      if (ess_dofs[i] < 0)
         mat -> EliminateRowCol (i, d);
}

void BilinearForm::Update (FiniteElementSpace *nfes)
{
   if (nfes)  fes = nfes;

   delete mat_e;
   delete mat;
   FreeElementMatrices();

   size = fes->GetVSize();

   mat = mat_e = NULL;
}

BilinearForm::~BilinearForm()
{
   delete mat_e;
   delete mat;
   delete element_matrices;

   if (!extern_bfs)
   {
      int k;
      for (k=0; k < dbfi.Size(); k++) delete dbfi[k];
      for (k=0; k < bbfi.Size(); k++) delete bbfi[k];
      for (k=0; k < fbfi.Size(); k++) delete fbfi[k];
      for (k=0; k < bfbfi.Size(); k++) delete bfbfi[k];
   }
}


MixedBilinearForm::MixedBilinearForm (FiniteElementSpace *tr_fes,
                                      FiniteElementSpace *te_fes)
   : Matrix (te_fes->GetVSize())
{
   width = tr_fes->GetVSize();
   trial_fes = tr_fes;
   test_fes = te_fes;
   mat = NULL;
}

double & MixedBilinearForm::Elem (int i, int j)
{
   return (*mat)(i, j);
}

const double & MixedBilinearForm::Elem (int i, int j) const
{
   return (*mat)(i, j);
}

void MixedBilinearForm::Mult (const Vector & x, Vector & y) const
{
   mat -> Mult (x, y);
}

void MixedBilinearForm::AddMult (const Vector & x, Vector & y,
                                 const double a) const
{
   mat -> AddMult (x, y, a);
}

void MixedBilinearForm::AddMultTranspose (const Vector & x, Vector & y,
                                          const double a) const
{
   mat -> AddMultTranspose (x, y, a);
}

MatrixInverse * MixedBilinearForm::Inverse() const
{
   return mat -> Inverse ();
}

void MixedBilinearForm::Finalize (int skip_zeros)
{
   mat -> Finalize (skip_zeros);
}

void MixedBilinearForm::GetBlocks(Array2D<SparseMatrix *> &blocks) const
{
   if (trial_fes->GetOrdering() != Ordering::byNODES ||
       test_fes->GetOrdering() != Ordering::byNODES)
      mfem_error("MixedBilinearForm::GetBlocks :\n"
                 " Both trial and test spaces must use Ordering::byNODES!");

   blocks.SetSize(test_fes->GetVDim(), trial_fes->GetVDim());

   mat->GetBlocks(blocks);
}

void MixedBilinearForm::AddDomainIntegrator (BilinearFormIntegrator * bfi)
{
   dom.Append (bfi);
}

void MixedBilinearForm::AddBoundaryIntegrator (BilinearFormIntegrator * bfi)
{
   bdr.Append (bfi);
}

void MixedBilinearForm::Assemble (int skip_zeros)
{
   int i, k;
   Array<int> tr_vdofs, te_vdofs;
   ElementTransformation *eltrans;
   DenseMatrix elemmat;

   if (mat == NULL)
      mat = new SparseMatrix(size, width);

   if (dom.Size())
      for (i = 0; i < test_fes -> GetNE(); i++)
      {
         trial_fes -> GetElementVDofs (i, tr_vdofs);
         test_fes  -> GetElementVDofs (i, te_vdofs);
         eltrans = test_fes -> GetElementTransformation (i);
         for (k = 0; k < dom.Size(); k++)
         {
            dom[k] -> AssembleElementMatrix2 (*trial_fes -> GetFE(i),
                                              *test_fes  -> GetFE(i),
                                              *eltrans, elemmat);
            mat -> AddSubMatrix (te_vdofs, tr_vdofs, elemmat, skip_zeros);
         }
      }

   if (bdr.Size())
      for (i = 0; i < test_fes -> GetNBE(); i++)
      {
         trial_fes -> GetBdrElementVDofs (i, tr_vdofs);
         test_fes  -> GetBdrElementVDofs (i, te_vdofs);
         eltrans = test_fes -> GetBdrElementTransformation (i);
         for (k = 0; k < bdr.Size(); k++)
         {
            bdr[k] -> AssembleElementMatrix2 (*trial_fes -> GetBE(i),
                                              *test_fes  -> GetBE(i),
                                              *eltrans, elemmat);
            mat -> AddSubMatrix (te_vdofs, tr_vdofs, elemmat, skip_zeros);
         }
      }
}

void MixedBilinearForm::EliminateTrialDofs (
   Array<int> &bdr_attr_is_ess, Vector &sol, Vector &rhs )
{
   int i, j, k;
   Array<int> tr_vdofs, cols_marker (trial_fes -> GetVSize());

   cols_marker = 0;
   for (i = 0; i < trial_fes -> GetNBE(); i++)
      if (bdr_attr_is_ess[trial_fes -> GetBdrAttribute (i)-1])
      {
         trial_fes -> GetBdrElementVDofs (i, tr_vdofs);
         for (j = 0; j < tr_vdofs.Size(); j++)
         {
            if ( (k = tr_vdofs[j]) < 0 )
               k = -1-k;
            cols_marker[k] = 1;
         }
      }
   mat -> EliminateCols (cols_marker, &sol, &rhs);
}

void MixedBilinearForm::EliminateTestDofs (Array<int> &bdr_attr_is_ess)
{
   int i, j, k;
   Array<int> te_vdofs;

   for (i = 0; i < test_fes -> GetNBE(); i++)
      if (bdr_attr_is_ess[test_fes -> GetBdrAttribute (i)-1])
      {
         test_fes -> GetBdrElementVDofs (i, te_vdofs);
         for (j = 0; j < te_vdofs.Size(); j++)
         {
            if ( (k = te_vdofs[j]) < 0 )
               k = -1-k;
            mat -> EliminateRow (k);
         }
      }
}

void MixedBilinearForm::Update()
{
   delete mat;
   mat = NULL;
   size = test_fes->GetVSize();
   width = trial_fes->GetVSize();
}

MixedBilinearForm::~MixedBilinearForm()
{
   int i;

   if (mat)  delete mat;
   for (i = 0; i < dom.Size(); i++)  delete dom[i];
   for (i = 0; i < bdr.Size(); i++)  delete bdr[i];
}


void DiscreteLinearOperator::Assemble(int skip_zeros)
{
   Array<int> dom_vdofs, ran_vdofs;
   ElementTransformation *T;
   const FiniteElement *dom_fe, *ran_fe;
   DenseMatrix totelmat, elmat;

   if (mat == NULL)
      mat = new SparseMatrix(size, width);

   if (dom.Size() > 0)
      for (int i = 0; i < test_fes->GetNE(); i++)
      {
         trial_fes->GetElementVDofs(i, dom_vdofs);
         test_fes->GetElementVDofs(i, ran_vdofs);
         T = test_fes->GetElementTransformation(i);
         dom_fe = trial_fes->GetFE(i);
         ran_fe = test_fes->GetFE(i);

         dom[0]->AssembleElementMatrix2(*dom_fe, *ran_fe, *T, totelmat);
         for (int j = 1; j < dom.Size(); j++)
         {
            dom[j]->AssembleElementMatrix2(*dom_fe, *ran_fe, *T, elmat);
            totelmat += elmat;
         }
         mat->SetSubMatrix(ran_vdofs, dom_vdofs, totelmat, skip_zeros);
      }
}
