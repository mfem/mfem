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

#include "fem.hpp"

namespace mfem
{

void DofTransformation::Transform(const Vector &v, Vector &v_trans) const
{
   v_trans.SetSize(height_);
   Transform(v.GetData(), v_trans.GetData());
}

void DofTransformation::Transform(const DenseMatrix &A,
                                  DenseMatrix &A_trans) const
{
   DenseMatrix A_col_trans;
   TransformCols(A, A_col_trans);
   TransformRows(A_col_trans, A_trans);
}

void DofTransformation::TransformRows(const DenseMatrix &A,
                                      DenseMatrix &A_trans) const
{
   A_trans.SetSize(A.Height(), width_);
   Vector row;
   Vector row_trans(width_);
   for (int r=0; r<A.Height(); r++)
   {
      A.GetRow(r, row);
      Transform(row, row_trans);
      A_trans.SetRow(r, row_trans);
   }
}

void DofTransformation::TransformCols(const DenseMatrix &A,
                                      DenseMatrix &A_trans) const
{
   A_trans.SetSize(height_, A.Width());
   Vector col_trans;
   for (int c=0; c<A.Width(); c++)
   {
      A_trans.GetColumnReference(c, col_trans);
      Transform(A.GetColumn(c), col_trans);
   }
}

void DofTransformation::TransformBack(const Vector &v_trans, Vector &v) const
{
   v.SetSize(width_);
   TransformBack(v_trans.GetData(), v.GetData());
}

void VDofTransformation::Transform(const double *v, double *v_trans) const
{
  int height = doftrans_->Height();
  int width  = doftrans_->Width();

  if ((Ordering::Type)ordering_ == Ordering::byNODES)
  {
    for (int i=0; i<vdim_; i++)
    {
      doftrans_->Transform(&v[i*width], &v_trans[i*height]);
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
      doftrans_->Transform(vec, vec_trans);
      for (int j=0; j<height; j++)
      {
	v_trans[j*vdim_+i] = vec_trans(j);
      }
    }
  }
}

void VDofTransformation::TransformBack(const double *v_trans, double *v) const
{
  int height = doftrans_->Height();
  int width  = doftrans_->Width();

  if ((Ordering::Type)ordering_ == Ordering::byNODES)
  {
    for (int i=0; i<vdim_; i++)
    {
      doftrans_->TransformBack(&v_trans[i*height], &v[i*width]);
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
      doftrans_->TransformBack(vec_trans, vec);
      for (int j=0; j<width; j++)
      {
	v[j*vdim_+i] = vec(j);
      }
    }
  }
}

const double ND_TetDofTransformation::T_data[24] =
{
   1.0,  0.0,  0.0,  1.0,
  -1.0,  0.0, -1.0,  1.0,
  -1.0,  1.0, -1.0,  0.0,
   1.0, -1.0,  0.0, -1.0,
   0.0, -1.0,  1.0, -1.0,
   0.0,  1.0,  1.0,  0.0
};

const double ND_TetDofTransformation::TInv_data[24] =
{
   1.0,  0.0,  0.0,  1.0,
  -1.0,  0.0, -1.0,  1.0,
   0.0, -1.0,  1.0, -1.0,
   1.0, -1.0,  0.0, -1.0,
  -1.0,  1.0, -1.0,  0.0,
   0.0,  1.0,  1.0,  0.0
};

ND_TetDofTransformation::ND_TetDofTransformation(int p)
  : DofTransformation(p*(p + 2)*(p + 3)/2, p*(p + 2)*(p + 3)/2),
    T(const_cast<double*>(T_data), 2, 2, 6),
    TInv(const_cast<double*>(TInv_data), 2, 2, 6),
    order(p)
{
}
  
void ND_TetDofTransformation::Transform(const double *v, double *v_trans) const
{
  int nedofs = order; // number of DoFs per edge
  int nfdofs = order*(order-1); // number of DoFs per face
  int ndofs  = order*(order+2)*(order+3)/2; // total number of DoFs
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
	  T(Fo[f]).Mult(&v[6*nedofs + 2*i], &v_trans[6*nedofs + 2*i]);
	}
    }

  // Copy interior DoFs
  for (int i=6*nedofs + 4*nfdofs; i<ndofs; i++)
  {
     v_trans[i] = v[i];
  }
}

void ND_TetDofTransformation::TransformBack(const double *, double *) const
{}

} // namespace mfem
