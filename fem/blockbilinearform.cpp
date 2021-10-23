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

BlockBilinearForm::BlockBilinearForm(Array<FiniteElementSpace *> & fespaces) :
   Matrix(0)
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
   // TODO
}

/// Adds new Block Domain Integrator. Assumes ownership of @a bfi.
void BlockBilinearForm::AddDomainIntegrator(BlockBilinearFormIntegrator *bfi)
{
   // TODO
}

/// Assembles the form i.e. sums over all domain integrators.
void BlockBilinearForm::Assemble(int skip_zeros)
{
   // TODO
   ElementTransformation *eltrans;
   DofTransformation * doftrans_j, *doftrans_k;
   Mesh *mesh = fespaces[0] -> GetMesh();
   DenseMatrix elmat, *elmat_p;
   int nblocks = fespaces.Size();
   Array<const FiniteElement *> fe(nblocks);
   Array<int> vdofs_j, vdofs_k;

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
               }
               eltrans = fespaces[0]->GetElementTransformation(i);
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
            for (int k = 0; k<nblocks; k++)
            {
               doftrans_k = fespaces[k]->GetElementVDofs(i, vdofs_k);
               // extract sub matrix
               DenseMatrix temp(vdofs_j.Size(), vdofs_k.Size());
               MFEM_ABORT("BlockBilinearForm::extract dense submatrix");
               if (doftrans_k || doftrans_j)
               {
                  TransformDual(doftrans_j, doftrans_k, temp);
               }
               // extract temp matrix
               mat->AddSubMatrix(vdofs_j, vdofs_k, temp, skip_zeros);
            }
         }
      }
   }
}

void BlockBilinearForm::ComputeElementMatrices()
{
   // TODO
}

void BlockBilinearForm::ComputeElementMatrix(int i, DenseMatrix &elmat)
{
   // TODO
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
