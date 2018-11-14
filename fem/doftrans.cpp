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

void VDofTransformation::Transform(const double *, double *) const
{}

void VDofTransformation::TransformBack(const Vector &, Vector &) const
{}

} // namespace mfem
