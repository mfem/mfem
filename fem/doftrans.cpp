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

void DofTransformation::TransformPrimal(const Vector &v, Vector &v_trans) const
{
   v_trans.SetSize(height_);
   TransformPrimal(v.GetData(), v_trans.GetData());
}

void DofTransformation::TransformPrimalCols(const DenseMatrix &V,
                                            DenseMatrix &V_trans) const
{
   V_trans.SetSize(height_, V.Width());
   Vector col_trans;
   for (int c=0; c<V.Width(); c++)
   {
      V_trans.GetColumnReference(c, col_trans);
      TransformPrimal(V.GetColumn(c), col_trans);
   }
}

void DofTransformation::TransformDual(const Vector &v, Vector &v_trans) const
{
   v_trans.SetSize(width_);
   TransformDual(v.GetData(), v_trans.GetData());
}

void DofTransformation::TransformDual(const DenseMatrix &V,
                                      DenseMatrix &V_trans) const
{
   DenseMatrix V_col_trans;
   TransformDualCols(V, V_col_trans);
   TransformDualRows(V_col_trans, V_trans);
}

void DofTransformation::TransformDualRows(const DenseMatrix &V,
                                          DenseMatrix &V_trans) const
{
   V_trans.SetSize(V.Height(), width_);
   Vector row;
   Vector row_trans;
   for (int r=0; r<V.Height(); r++)
   {
      V.GetRow(r, row);
      TransformDual(row, row_trans);
      V_trans.SetRow(r, row_trans);
   }
}

void DofTransformation::TransformDualCols(const DenseMatrix &V,
                                          DenseMatrix &V_trans) const
{
   V_trans.SetSize(height_, V.Width());
   Vector col_trans;
   for (int c=0; c<V.Width(); c++)
   {
      V_trans.GetColumnReference(c, col_trans);
      TransformDual(V.GetColumn(c), col_trans);
   }
}

void DofTransformation::InvTransformPrimal(const Vector &v_trans,
                                           Vector &v) const
{
   v.SetSize(width_);
   InvTransformPrimal(v_trans.GetData(), v.GetData());
}

void TransformPrimal(const DofTransformation *ran_dof_trans,
                     const DofTransformation *dom_dof_trans,
                     const DenseMatrix &elmat, DenseMatrix &elmat_trans)
{
   if (ran_dof_trans && dom_dof_trans)
   {
      DenseMatrix elmat_tmp;
      ran_dof_trans->TransformPrimalCols(elmat, elmat_tmp);
      dom_dof_trans->TransformDualRows(elmat_tmp, elmat_trans);
   }
   else if (ran_dof_trans)
   {
      ran_dof_trans->TransformPrimalCols(elmat, elmat_trans);
   }
   else if (dom_dof_trans)
   {
      dom_dof_trans->TransformDualRows(elmat, elmat_trans);
   }
   else
   {
      // If both transformations are NULL this function should not be called
      elmat_trans = elmat;
   }
}

void TransformDual(const DofTransformation *ran_dof_trans,
                   const DofTransformation *dom_dof_trans,
                   const DenseMatrix &elmat, DenseMatrix &elmat_trans)
{
   if (ran_dof_trans && dom_dof_trans)
   {
      DenseMatrix elmat_tmp;
      ran_dof_trans->TransformDualCols(elmat, elmat_tmp);
      dom_dof_trans->TransformDualRows(elmat_tmp, elmat_trans);
   }
   else if (ran_dof_trans)
   {
      ran_dof_trans->TransformDualCols(elmat, elmat_trans);
   }
   else if (dom_dof_trans)
   {
      dom_dof_trans->TransformDualRows(elmat, elmat_trans);
   }
   else
   {
      // If both transformations are NULL this function should not be called
      elmat_trans = elmat;
   }
}

void VDofTransformation::TransformPrimal(const double *v, double *v_trans) const
{
   int height = doftrans_->Height();
   int width  = doftrans_->Width();

   if ((Ordering::Type)ordering_ == Ordering::byNODES || vdim_ == 1)
   {
      for (int i=0; i<vdim_; i++)
      {
         doftrans_->TransformPrimal(&v[i*width], &v_trans[i*height]);
      }
   }
   else
   {
      Vector vec(width);
      Vector vec_trans(height);
      for (int i=0; i<vdim_; i++)
      {
         for (int j=0; j<width; j++)
         {
            vec(j) = v[j*vdim_+i];
         }
         doftrans_->TransformPrimal(vec, vec_trans);
         for (int j=0; j<height; j++)
         {
            v_trans[j*vdim_+i] = vec_trans(j);
         }
      }
   }
}

void VDofTransformation::InvTransformPrimal(const double *v_trans,
                                            double *v) const
{
   int height = doftrans_->Height();
   int width  = doftrans_->Width();

   if ((Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         doftrans_->InvTransformPrimal(&v_trans[i*height], &v[i*width]);
      }
   }
   else
   {
      Vector vec_trans(height);
      Vector vec(width);
      for (int i=0; i<vdim_; i++)
      {
         for (int j=0; j<height; j++)
         {
            vec_trans(j) = v_trans[j*vdim_+i];
         }
         doftrans_->InvTransformPrimal(vec_trans, vec);
         for (int j=0; j<width; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void VDofTransformation::TransformDual(const double *v, double *v_trans) const
{
   int height = doftrans_->Height();
   int width  = doftrans_->Width();

   if ((Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         doftrans_->TransformDual(&v[i*height], &v_trans[i*width]);
      }
   }
   else
   {
      Vector vec_trans(height);
      Vector vec(width);
      for (int i=0; i<vdim_; i++)
      {
         for (int j=0; j<width; j++)
         {
            vec(j) = v[j*vdim_+i];
         }
         doftrans_->TransformDual(vec, vec_trans);
         for (int j=0; j<height; j++)
         {
            v_trans[j*vdim_+i] = vec_trans(j);
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

ND_DofTransformation::ND_DofTransformation(int height, int width, int p)
   : DofTransformation(height, width),
     order(p)
{
}

ND_TriDofTransformation::ND_TriDofTransformation(int p)
   : ND_DofTransformation(p*(p + 2), p*(p + 2), p)
   , nedofs(order)
   , nfdofs(order*(order-1))
{
}

void ND_TriDofTransformation::TransformPrimal(const double *v,
                                              double *v_trans) const
{
   MFEM_VERIFY(Fo.Size() >= 1,
               "Face orientations are unset in ND_TriDofTransformation");

   // Copy edge DoFs
   for (int i=0; i<3*nedofs; i++)
   {
      v_trans[i] = v[i];
   }

   // Transform face DoFs
   for (int f=0; f<1; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         T(Fo[f]).Mult(&v[3*nedofs + f*nfdofs + 2*i],
                       &v_trans[3*nedofs + f*nfdofs + 2*i]);
      }
   }
}

void
ND_TriDofTransformation::InvTransformPrimal(const double *v_trans,
                                            double *v) const
{
   MFEM_VERIFY(Fo.Size() >= 1,
               "Face orientations are unset in ND_TriDofTransformation");

   // Copy edge DoFs
   for (int i=0; i<3*nedofs; i++)
   {
      v[i] = v_trans[i];
   }

   // Transform face DoFs
   for (int f=0; f<1; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         TInv(Fo[f]).Mult(&v_trans[3*nedofs + f*nfdofs + 2*i],
                          &v[3*nedofs + f*nfdofs + 2*i]);
      }
   }
}

void
ND_TriDofTransformation::TransformDual(const double *v, double *v_trans) const
{
   MFEM_VERIFY(Fo.Size() >= 1,
               "Face orientations are unset in ND_TriDofTransformation");

   // Copy edge DoFs
   for (int i=0; i<3*nedofs; i++)
   {
      v_trans[i] = v[i];
   }

   // Transform face DoFs
   for (int f=0; f<1; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         TInv(Fo[f]).MultTranspose(&v[3*nedofs + f*nfdofs + 2*i],
                                   &v_trans[3*nedofs + f*nfdofs + 2*i]);
      }
   }
}

ND_TetDofTransformation::ND_TetDofTransformation(int p)
   : ND_DofTransformation(p*(p + 2)*(p + 3)/2, p*(p + 2)*(p + 3)/2, p)
   , nedofs(order)
   , nfdofs(order*(order-1))
{
}

void ND_TetDofTransformation::TransformPrimal(const double *v,
                                              double *v_trans) const
{
   MFEM_VERIFY(Fo.Size() >= 4,
               "Face orientations are unset in ND_TetDofTransformation");

   // Copy edge DoFs
   for (int i=0; i<6*nedofs; i++)
   {
      v_trans[i] = v[i];
   }

   // Transform face DoFs
   for (int f=0; f<4; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         T(Fo[f]).Mult(&v[6*nedofs + f*nfdofs + 2*i],
                       &v_trans[6*nedofs + f*nfdofs + 2*i]);
      }
   }

   // Copy interior DoFs
   for (int i=6*nedofs + 4*nfdofs; i<height_; i++)
   {
      v_trans[i] = v[i];
   }
}

void
ND_TetDofTransformation::InvTransformPrimal(const double *v_trans,
                                            double *v) const
{
   MFEM_VERIFY(Fo.Size() >= 4,
               "Face orientations are unset in ND_TetDofTransformation");

   // Copy edge DoFs
   for (int i=0; i<6*nedofs; i++)
   {
      v[i] = v_trans[i];
   }

   // Transform face DoFs
   for (int f=0; f<4; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         TInv(Fo[f]).Mult(&v_trans[6*nedofs + f*nfdofs + 2*i],
                          &v[6*nedofs + f*nfdofs + 2*i]);
      }
   }

   // Copy interior DoFs
   for (int i=6*nedofs + 4*nfdofs; i<height_; i++)
   {
      v[i] = v_trans[i];
   }
}

void
ND_TetDofTransformation::TransformDual(const double *v, double *v_trans) const
{
   MFEM_VERIFY(Fo.Size() >= 4,
               "Face orientations are unset in ND_TetDofTransformation");

   // Copy edge DoFs
   for (int i=0; i<6*nedofs; i++)
   {
      v_trans[i] = v[i];
   }

   // Transform face DoFs
   for (int f=0; f<4; f++)
   {
      for (int i=0; i<nfdofs/2; i++)
      {
         TInv(Fo[f]).MultTranspose(&v[6*nedofs + f*nfdofs + 2*i],
                                   &v_trans[6*nedofs + f*nfdofs + 2*i]);
      }
   }

   // Copy interior DoFs
   for (int i=6*nedofs + 4*nfdofs; i<height_; i++)
   {
      v_trans[i] = v[i];
   }
}


ND_WedgeDofTransformation::ND_WedgeDofTransformation(int p)
   : ND_DofTransformation(3 * p * ((p + 1) * (p + 2))/2,
                          3 * p * ((p + 1) * (p + 2))/2, p)
   , nedofs(order)
   , ntdofs(order*(order-1))
   , nqdofs(2*order*(order-1))
{
}

void ND_WedgeDofTransformation::TransformPrimal(const double *v,
                                                double *v_trans) const
{
   MFEM_VERIFY(Fo.Size() >= 2,
               "Face orientations are unset in ND_WedgeDofTransformation");

   // Copy edge DoFs
   for (int i=0; i<9*nedofs; i++)
   {
      v_trans[i] = v[i];
   }

   // Transform triangular face DoFs
   for (int f=0; f<2; f++)
   {
      for (int i=0; i<ntdofs/2; i++)
      {
         T(Fo[f]).Mult(&v[9*nedofs + f*ntdofs + 2*i],
                       &v_trans[9*nedofs + f*ntdofs + 2*i]);
      }
   }

   // Transform quadrilateral face DoFs
   for (int i=9*nedofs + 2*ntdofs; i<9*nedofs + 2*ntdofs + 3*nqdofs; i++)
   {
      v_trans[i] = v[i];
   }

   // Copy interior DoFs
   for (int i=9*nedofs + 2*ntdofs + 3*nqdofs; i<height_; i++)
   {
      v_trans[i] = v[i];
   }
}

void
ND_WedgeDofTransformation::InvTransformPrimal(const double *v_trans,
                                              double *v) const
{
   MFEM_VERIFY(Fo.Size() >= 2,
               "Face orientations are unset in ND_WedgeDofTransformation");

   // Copy edge DoFs
   for (int i=0; i<9*nedofs; i++)
   {
      v[i] = v_trans[i];
   }

   // Transform triangular face DoFs
   for (int f=0; f<2; f++)
   {
      for (int i=0; i<ntdofs/2; i++)
      {
         TInv(Fo[f]).Mult(&v_trans[9*nedofs + f*ntdofs + 2*i],
                          &v[9*nedofs + f*ntdofs + 2*i]);
      }
   }

   // Transform quadrilateral face DoFs
   for (int i=9*nedofs + 2*ntdofs; i<9*nedofs + 2*ntdofs + 3*nqdofs; i++)
   {
      v[i] = v_trans[i];
   }

   // Copy interior DoFs
   for (int i=9*nedofs + 2*ntdofs + 3*nqdofs; i<height_; i++)
   {
      v[i] = v_trans[i];
   }
}

void
ND_WedgeDofTransformation::TransformDual(const double *v, double *v_trans) const
{
   MFEM_VERIFY(Fo.Size() >= 2,
               "Face orientations are unset in ND_WedgeDofTransformation");

   // Copy edge DoFs
   for (int i=0; i<9*nedofs; i++)
   {
      v_trans[i] = v[i];
   }

   // Transform triangular face DoFs
   for (int f=0; f<2; f++)
   {
      for (int i=0; i<ntdofs/2; i++)
      {
         TInv(Fo[f]).MultTranspose(&v[9*nedofs + f*ntdofs + 2*i],
                                   &v_trans[9*nedofs + f*ntdofs + 2*i]);
      }
   }

   // Transform quadrilateral face DoFs
   for (int i=9*nedofs + 2*ntdofs; i<9*nedofs + 2*ntdofs + 3*nqdofs; i++)
   {
      v_trans[i] = v[i];
   }

   // Copy interior DoFs
   for (int i=9*nedofs + 2*ntdofs + 3*nqdofs; i<height_; i++)
   {
      v_trans[i] = v[i];
   }
}

} // namespace mfem
