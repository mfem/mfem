// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

void DofTransformation::TransformPrimal(real_t *v) const
{
   if (IsIdentity()) { return; }
   int size = dof_trans_->Size();

   if (vdim_ == 1 || (Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         dof_trans_->TransformPrimal(Fo_, &v[i*size]);
      }
   }
   else
   {
      Vector vec(size);
      for (int i=0; i<vdim_; i++)
      {
         for (int j=0; j<size; j++)
         {
            vec(j) = v[j*vdim_+i];
         }
         dof_trans_->TransformPrimal(Fo_, vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void DofTransformation::InvTransformPrimal(real_t *v) const
{
   if (IsIdentity()) { return; }
   int size = dof_trans_->Height();

   if (vdim_ == 1 || (Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         dof_trans_->InvTransformPrimal(Fo_, &v[i*size]);
      }
   }
   else
   {
      Vector vec(size);
      for (int i=0; i<vdim_; i++)
      {
         for (int j=0; j<size; j++)
         {
            vec(j) = v[j*vdim_+i];
         }
         dof_trans_->InvTransformPrimal(Fo_, vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void DofTransformation::TransformDual(real_t *v) const
{
   if (IsIdentity()) { return; }
   int size = dof_trans_->Size();

   if (vdim_ == 1 || (Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         dof_trans_->TransformDual(Fo_, &v[i*size]);
      }
   }
   else
   {
      Vector vec(size);
      for (int i=0; i<vdim_; i++)
      {
         for (int j=0; j<size; j++)
         {
            vec(j) = v[j*vdim_+i];
         }
         dof_trans_->TransformDual(Fo_, vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void DofTransformation::InvTransformDual(real_t *v) const
{
   if (IsIdentity()) { return; }
   int size = dof_trans_->Size();

   if (vdim_ == 1 || (Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         dof_trans_->InvTransformDual(Fo_, &v[i*size]);
      }
   }
   else
   {
      Vector vec(size);
      for (int i=0; i<vdim_; i++)
      {
         for (int j=0; j<size; j++)
         {
            vec(j) = v[j*vdim_+i];
         }
         dof_trans_->InvTransformDual(Fo_, vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void TransformPrimal(const DofTransformation &ran_dof_trans,
                     const DofTransformation &dom_dof_trans,
                     DenseMatrix &elmat)
{
   // No action if both transformations are NULL
   if (!ran_dof_trans.IsIdentity())
   {
      ran_dof_trans.TransformPrimalCols(elmat);
   }
   if (!dom_dof_trans.IsIdentity())
   {
      dom_dof_trans.TransformDualRows(elmat);
   }
}

void TransformDual(const DofTransformation &ran_dof_trans,
                   const DofTransformation &dom_dof_trans,
                   DenseMatrix &elmat)
{
   // No action if both transformations are NULL
   if (!ran_dof_trans.IsIdentity())
   {
      ran_dof_trans.TransformDualCols(elmat);
   }
   if (!dom_dof_trans.IsIdentity())
   {
      dom_dof_trans.TransformDualRows(elmat);
   }
}

// ordering (i0j0, i1j0, i0j1, i1j1), each row is a column major matrix
const real_t ND_DofTransformation::T_data[24] =
{
   1.0,  0.0,  0.0,  1.0,
   -1.0, -1.0,  0.0,  1.0,
   0.0,  1.0, -1.0, -1.0,
   1.0,  0.0, -1.0, -1.0,
   -1.0, -1.0,  1.0,  0.0,
   0.0,  1.0,  1.0,  0.0
};

const DenseTensor ND_DofTransformation
::T(const_cast<real_t *>(ND_DofTransformation::T_data), 2, 2, 6);

// ordering (i0j0, i1j0, i0j1, i1j1), each row is a column major matrix
const real_t ND_DofTransformation::TInv_data[24] =
{
   1.0,  0.0,  0.0,  1.0,
   -1.0, -1.0,  0.0,  1.0,
   -1.0, -1.0,  1.0,  0.0,
   1.0,  0.0, -1.0, -1.0,
   0.0,  1.0, -1.0, -1.0,
   0.0,  1.0,  1.0,  0.0
};

const DenseTensor ND_DofTransformation
::TInv(const_cast<real_t *>(TInv_data), 2, 2, 6);

ND_DofTransformation::ND_DofTransformation(int size, int p, int num_edges,
                                           int num_faces,
                                           int face_types[])
   : StatelessDofTransformation(size)
   , order(p)
   , nedofs(p)
   , ntdofs(p*(p-1))
   , nqdofs(2*p*(p-1))
   , nedges(num_edges)
   , nfaces(num_faces)
   , ftypes(face_types)
{
}

void ND_DofTransformation::TransformPrimal(const Array<int> & Fo,
                                           real_t *v) const
{
   // Return immediately when no face DoFs are present
   if (IsIdentity()) { return; }

   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is shorter than the number of faces in "
               "ND_DofTransformation");

   int of = 0;
   real_t data[2];
   Vector v2(data, 2);
   DenseMatrix T2;

   // Transform face DoFs
   for (int f=0; f<nfaces; f++)
   {
      if (ftypes[f] == Geometry::TRIANGLE)
      {
         for (int i=0; i<ntdofs/2; i++)
         {
            v2 = &v[nedges*nedofs + of + 2*i];
            T2.UseExternalData(const_cast<real_t *>(T.GetData(Fo[f])), 2, 2);
            T2.Mult(v2, &v[nedges*nedofs + of + 2*i]);
         }
         of += ntdofs;
      }
      else
      {
         of += nqdofs;
      }
   }
}

void ND_DofTransformation::InvTransformPrimal(const Array<int> & Fo,
                                              real_t *v) const
{
   // Return immediately when no face DoFs are present
   if (IsIdentity()) { return; }

   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is shorter than the number of faces in "
               "ND_DofTransformation");

   int of = 0;
   real_t data[2];
   Vector v2(data, 2);
   DenseMatrix T2Inv;

   // Transform face DoFs
   for (int f=0; f<nfaces; f++)
   {
      if (ftypes[f] == Geometry::TRIANGLE)
      {
         for (int i=0; i<ntdofs/2; i++)
         {
            v2 = &v[nedges*nedofs + of + 2*i];
            T2Inv.UseExternalData(const_cast<real_t *>(TInv.GetData(Fo[f])), 2, 2);
            T2Inv.Mult(v2, &v[nedges*nedofs + of + 2*i]);
         }
         of += ntdofs;
      }
      else
      {
         of += nqdofs;
      }
   }
}

void ND_DofTransformation::TransformDual(const Array<int> & Fo, real_t *v) const
{
   // Return immediately when no face DoFs are present
   if (IsIdentity()) { return; }

   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is shorter than the number of faces in "
               "ND_DofTransformation");

   int of = 0;
   real_t data[2];
   Vector v2(data, 2);
   DenseMatrix T2Inv;

   // Transform face DoFs
   for (int f=0; f<nfaces; f++)
   {
      if (ftypes[f] == Geometry::TRIANGLE)
      {
         for (int i=0; i<ntdofs/2; i++)
         {
            v2 = &v[nedges*nedofs + of + 2*i];
            T2Inv.UseExternalData(const_cast<real_t *>(TInv.GetData(Fo[f])), 2, 2);
            T2Inv.MultTranspose(v2, &v[nedges*nedofs + of + 2*i]);
         }
         of += ntdofs;
      }
      else
      {
         of += nqdofs;
      }

   }
}

void ND_DofTransformation::InvTransformDual(const Array<int> & Fo,
                                            real_t *v) const
{
   // Return immediately when no face DoFs are present
   if (IsIdentity()) { return; }

   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is shorter than the number of faces in "
               "ND_DofTransformation");

   int of = 0;
   real_t data[2];
   Vector v2(data, 2);
   DenseMatrix T2;

   // Transform face DoFs
   for (int f=0; f<nfaces; f++)
   {
      if (ftypes[f] == Geometry::TRIANGLE)
      {
         for (int i=0; i<ntdofs/2; i++)
         {
            v2 = &v[nedges*nedofs + of + 2*i];
            T2.UseExternalData(const_cast<real_t *>(T.GetData(Fo[f])), 2, 2);
            T2.MultTranspose(v2, &v[nedges*nedofs + of + 2*i]);
         }
         of += ntdofs;
      }
      else
      {
         of += nqdofs;
      }
   }
}

} // namespace mfem
