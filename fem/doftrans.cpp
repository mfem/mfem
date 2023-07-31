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

#include "fem.hpp"

namespace mfem
{

void TransformPrimal(const DofTransformation *ran_dof_trans,
                     const DofTransformation *dom_dof_trans,
                     DenseMatrix &elmat)
{
   if (ran_dof_trans && dom_dof_trans)
   {
      ran_dof_trans->TransformPrimalCols(elmat);
      dom_dof_trans->TransformDualRows(elmat);
   }
   else if (ran_dof_trans)
   {
      ran_dof_trans->TransformPrimalCols(elmat);
   }
   else if (dom_dof_trans)
   {
      dom_dof_trans->TransformDualRows(elmat);
   }
   else
   {
      // If both transformations are NULL this function should not be called
   }
}

void TransformDual(const DofTransformation *ran_dof_trans,
                   const DofTransformation *dom_dof_trans,
                   DenseMatrix &elmat)
{
   if (ran_dof_trans && dom_dof_trans)
   {
      ran_dof_trans->TransformDualCols(elmat);
      dom_dof_trans->TransformDualRows(elmat);
   }
   else if (ran_dof_trans)
   {
      ran_dof_trans->TransformDualCols(elmat);
   }
   else if (dom_dof_trans)
   {
      dom_dof_trans->TransformDualRows(elmat);
   }
   else
   {
      // If both transformations are NULL this function should not be called
   }
}

void StatelessVDofTransformation::TransformPrimal(const Array<int> & face_ori,
                                                  double *v) const
{
   int size = sdoftrans_->Size();

   if ((Ordering::Type)ordering_ == Ordering::byNODES || vdim_ == 1)
   {
      for (int i=0; i<vdim_; i++)
      {
         sdoftrans_->TransformPrimal(face_ori, &v[i*size]);
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
         sdoftrans_->TransformPrimal(face_ori, vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void StatelessVDofTransformation::InvTransformPrimal(
   const Array<int> & face_ori,
   double *v) const
{
   int size = sdoftrans_->Height();

   if ((Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         sdoftrans_->InvTransformPrimal(face_ori, &v[i*size]);
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
         sdoftrans_->InvTransformPrimal(face_ori, vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void StatelessVDofTransformation::TransformDual(const Array<int> & face_ori,
                                                double *v) const
{
   int size = sdoftrans_->Size();

   if ((Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         sdoftrans_->TransformDual(face_ori, &v[i*size]);
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
         sdoftrans_->TransformDual(face_ori, vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void StatelessVDofTransformation::InvTransformDual(const Array<int> & face_ori,
                                                   double *v) const
{
   int size = sdoftrans_->Size();

   if ((Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         sdoftrans_->InvTransformDual(face_ori, &v[i*size]);
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
         sdoftrans_->InvTransformDual(face_ori, vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

// ordering (i0j0, i1j0, i0j1, i1j1), each row is a column major matrix
const double ND_StatelessDofTransformation::T_data[24] =
{
   1.0,  0.0,  0.0,  1.0,
   -1.0, -1.0,  0.0,  1.0,
   0.0,  1.0, -1.0, -1.0,
   1.0,  0.0, -1.0, -1.0,
   -1.0, -1.0,  1.0,  0.0,
   0.0,  1.0,  1.0,  0.0
};

const DenseTensor ND_StatelessDofTransformation
::T(const_cast<double*>(ND_StatelessDofTransformation::T_data), 2, 2, 6);

// ordering (i0j0, i1j0, i0j1, i1j1), each row is a column major matrix
const double ND_StatelessDofTransformation::TInv_data[24] =
{
   1.0,  0.0,  0.0,  1.0,
   -1.0, -1.0,  0.0,  1.0,
   -1.0, -1.0,  1.0,  0.0,
   1.0,  0.0, -1.0, -1.0,
   0.0,  1.0, -1.0, -1.0,
   0.0,  1.0,  1.0,  0.0
};

const DenseTensor ND_StatelessDofTransformation
::TInv(const_cast<double*>(TInv_data), 2, 2, 6);

ND_StatelessDofTransformation::ND_StatelessDofTransformation(int size, int p,
                                                             int num_edges,
                                                             int num_tri_faces)
   : StatelessDofTransformation(size)
   , order(p)
   , nedofs(p)
   , nfdofs(p*(p-1))
   , nedges(num_edges)
   , nfaces(num_tri_faces)
{
}

void ND_StatelessDofTransformation::TransformPrimal(const Array<int> & Fo,
                                                    double *v) const
{
   // Return immediately when no face DoFs are present
   if (nfdofs < 2) { return; }

   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is shorter than the number of faces in "
               "ND_StatelessDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform face DoFs
   for (int f=0; f<nfaces; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[nedges*nedofs + f*nfdofs + 2*i];
         T(Fo[f]).Mult(v2, &v[nedges*nedofs + f*nfdofs + 2*i]);
      }
   }
}

void ND_StatelessDofTransformation::InvTransformPrimal(const Array<int> & Fo,
                                                       double *v) const
{
   // Return immediately when no face DoFs are present
   if (nfdofs < 2) { return; }

   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is shorter than the number of faces in "
               "ND_StatelessDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform face DoFs
   for (int f=0; f<nfaces; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[nedges*nedofs + f*nfdofs + 2*i];
         TInv(Fo[f]).Mult(v2, &v[nedges*nedofs + f*nfdofs + 2*i]);
      }
   }
}

void ND_StatelessDofTransformation::TransformDual(const Array<int> & Fo,
                                                  double *v) const
{
   // Return immediately when no face DoFs are present
   if (nfdofs < 2) { return; }

   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is shorter than the number of faces in "
               "ND_StatelessDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform face DoFs
   for (int f=0; f<nfaces; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[nedges*nedofs + f*nfdofs + 2*i];
         TInv(Fo[f]).MultTranspose(v2, &v[nedges*nedofs + f*nfdofs + 2*i]);
      }
   }
}

void ND_StatelessDofTransformation::InvTransformDual(const Array<int> & Fo,
                                                     double *v) const
{
   // Return immediately when no face DoFs are present
   if (nfdofs < 2) { return; }

   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is shorter than the number of faces in "
               "ND_StatelessDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform face DoFs
   for (int f=0; f<nfaces; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[nedges*nedofs + f*nfdofs + 2*i];
         T(Fo[f]).MultTranspose(v2, &v[nedges*nedofs + f*nfdofs + 2*i]);
      }
   }
}

} // namespace mfem
