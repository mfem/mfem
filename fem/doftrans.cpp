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

void DofTransformation::TransformPrimal(Vector &v) const
{
   TransformPrimal(v.GetData());
}

void DofTransformation::TransformPrimalCols(DenseMatrix &V) const
{
   for (int c=0; c<V.Width(); c++)
   {
      TransformPrimal(V.GetColumn(c));
   }
}

void DofTransformation::TransformDual(Vector &v) const
{
   TransformDual(v.GetData());
}

void DofTransformation::TransformDual(DenseMatrix &V) const
{
   TransformDualCols(V);
   TransformDualRows(V);
}

void DofTransformation::TransformDualRows(DenseMatrix &V) const
{
   Vector row;
   for (int r=0; r<V.Height(); r++)
   {
      V.GetRow(r, row);
      TransformDual(row);
      V.SetRow(r, row);
   }
}

void DofTransformation::TransformDualCols(DenseMatrix &V) const
{
   for (int c=0; c<V.Width(); c++)
   {
      TransformDual(V.GetColumn(c));
   }
}

void DofTransformation::InvTransformPrimal(Vector &v) const
{
   InvTransformPrimal(v.GetData());
}

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

void VDofTransformation::TransformPrimal(double *v) const
{
   int size = doftrans_->Size();

   if ((Ordering::Type)ordering_ == Ordering::byNODES || vdim_ == 1)
   {
      for (int i=0; i<vdim_; i++)
      {
         doftrans_->TransformPrimal(&v[i*size]);
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
         doftrans_->TransformPrimal(vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void VDofTransformation::InvTransformPrimal(double *v) const
{
   int size = doftrans_->Height();

   if ((Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         doftrans_->InvTransformPrimal(&v[i*size]);
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
         doftrans_->InvTransformPrimal(vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void VDofTransformation::TransformDual(double *v) const
{
   int size = doftrans_->Size();

   if ((Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         doftrans_->TransformDual(&v[i*size]);
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
         doftrans_->TransformDual(vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

const double ND_DofTransformation::T_data[24] =
{
   1.0,  0.0,  0.0,  1.0,
   -1.0, -1.0,  0.0,  1.0,
   0.0,  1.0, -1.0, -1.0,
   1.0,  0.0, -1.0, -1.0,
   -1.0, -1.0,  1.0,  0.0,
   0.0,  1.0,  1.0,  0.0
};

const DenseTensor ND_DofTransformation
::T(const_cast<double*>(ND_DofTransformation::T_data), 2, 2, 6);

const double ND_DofTransformation::TInv_data[24] =
{
   1.0,  0.0,  0.0,  1.0,
   -1.0, -1.0,  0.0,  1.0,
   -1.0, -1.0,  1.0,  0.0,
   1.0,  0.0, -1.0, -1.0,
   0.0,  1.0, -1.0, -1.0,
   0.0,  1.0,  1.0,  0.0
};

const DenseTensor ND_DofTransformation
::TInv(const_cast<double*>(TInv_data), 2, 2, 6);

ND_DofTransformation::ND_DofTransformation(int size, int p)
   : DofTransformation(size)
   , order(p)
   , nedofs(p)
   , nfdofs(p*(p-1))
{
}

ND_TriDofTransformation::ND_TriDofTransformation(int p)
   : ND_DofTransformation(p*(p + 2), p)
{
}

void ND_TriDofTransformation::TransformPrimal(double *v) const
{
   MFEM_VERIFY(Fo.Size() >= 1,
               "Face orientations are unset in ND_TriDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform face DoFs
   for (int f=0; f<1; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[3*nedofs + f*nfdofs + 2*i];
         T(Fo[f]).Mult(v2, &v[3*nedofs + f*nfdofs + 2*i]);
      }
   }
}

void
ND_TriDofTransformation::InvTransformPrimal(double *v) const
{
   MFEM_VERIFY(Fo.Size() >= 1,
               "Face orientations are unset in ND_TriDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform face DoFs
   for (int f=0; f<1; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[3*nedofs + f*nfdofs + 2*i];
         TInv(Fo[f]).Mult(v2, &v[3*nedofs + f*nfdofs + 2*i]);
      }
   }
}

void
ND_TriDofTransformation::TransformDual(double *v) const
{
   MFEM_VERIFY(Fo.Size() >= 1,
               "Face orientations are unset in ND_TriDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform face DoFs
   for (int f=0; f<1; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[3*nedofs + f*nfdofs + 2*i];
         TInv(Fo[f]).MultTranspose(v2, &v[3*nedofs + f*nfdofs + 2*i]);
      }
   }
}

ND_TetDofTransformation::ND_TetDofTransformation(int p)
   : ND_DofTransformation(p*(p + 2)*(p + 3)/2, p)
{
}

void ND_TetDofTransformation::TransformPrimal(double *v) const
{
   MFEM_VERIFY(Fo.Size() >= 4,
               "Face orientations are unset in ND_TetDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform face DoFs
   for (int f=0; f<4; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[6*nedofs + f*nfdofs + 2*i];
         T(Fo[f]).Mult(v2, &v[6*nedofs + f*nfdofs + 2*i]);
      }
   }
}

void
ND_TetDofTransformation::InvTransformPrimal(double *v) const
{
   MFEM_VERIFY(Fo.Size() >= 4,
               "Face orientations are unset in ND_TetDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform face DoFs
   for (int f=0; f<4; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[6*nedofs + f*nfdofs + 2*i];
         TInv(Fo[f]).Mult(v2, &v[6*nedofs + f*nfdofs + 2*i]);
      }
   }
}

void
ND_TetDofTransformation::TransformDual(double *v) const
{
   MFEM_VERIFY(Fo.Size() >= 4,
               "Face orientations are unset in ND_TetDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform face DoFs
   for (int f=0; f<4; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[6*nedofs + f*nfdofs + 2*i];
         TInv(Fo[f]).MultTranspose(v2, &v[6*nedofs + f*nfdofs + 2*i]);
      }
   }
}


ND_WedgeDofTransformation::ND_WedgeDofTransformation(int p)
   : ND_DofTransformation(3 * p * ((p + 1) * (p + 2))/2, p)
{
}

void ND_WedgeDofTransformation::TransformPrimal(double *v) const
{
   MFEM_VERIFY(Fo.Size() >= 2,
               "Face orientations are unset in ND_WedgeDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform triangular face DoFs
   for (int f=0; f<2; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[9*nedofs + f*nfdofs + 2*i];
         T(Fo[f]).Mult(v2, &v[9*nedofs + f*nfdofs + 2*i]);
      }
   }
}

void
ND_WedgeDofTransformation::InvTransformPrimal(double *v) const
{
   MFEM_VERIFY(Fo.Size() >= 2,
               "Face orientations are unset in ND_WedgeDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform triangular face DoFs
   for (int f=0; f<2; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[9*nedofs + f*nfdofs + 2*i];
         TInv(Fo[f]).Mult(v2, &v[9*nedofs + f*nfdofs + 2*i]);
      }
   }
}

void
ND_WedgeDofTransformation::TransformDual(double *v) const
{
   MFEM_VERIFY(Fo.Size() >= 2,
               "Face orientations are unset in ND_WedgeDofTransformation");

   double data[2];
   Vector v2(data, 2);

   // Transform triangular face DoFs
   for (int f=0; f<2; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         v2 = &v[9*nedofs + f*nfdofs + 2*i];
         TInv(Fo[f]).MultTranspose(v2, &v[9*nedofs + f*nfdofs + 2*i]);
      }
   }
}

} // namespace mfem
