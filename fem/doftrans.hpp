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

#ifndef MFEM_DOFTRANSFORM
#define MFEM_DOFTRANSFORM

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "intrules.hpp"
#include "fe.hpp"

namespace mfem
{

class DofTransformation
{
protected:
   int height_;
   int width_;

public:
   DofTransformation(int height, int width)
      : height_(height), width_(width) {}

   inline int Height() const { return height_; }
   inline int NumRows() const { return height_; }
   inline int Width() const { return width_; }
   inline int NumCols() const { return width_; }

   virtual void Transform(const double *, double *) const = 0;

   virtual void Transform(const Vector &, Vector &) const;

   virtual void Transform(const DenseMatrix &, DenseMatrix &) const;

   virtual void TransformRows(const DenseMatrix &, DenseMatrix &) const;

   virtual void TransformCols(const DenseMatrix &, DenseMatrix &) const;

   // virtual void TransformRow(const Vector &, Vector &) const;

   // virtual void TransformCol(const Vector &, Vector &) const;

   virtual void TransformRowCol(const double *, double *) const = 0;

   virtual void TransformBack(const double *, double *) const = 0;

   virtual void TransformBack(const Vector &, Vector &) const;

   virtual ~DofTransformation() {}
};

class VDofTransformation : public DofTransformation
{
private:
   int vdim_;
   int ordering_;
   DofTransformation * doftrans_;

public:
   VDofTransformation(int vdim = 1, int ordering = 0)
      : DofTransformation(0,0),
        vdim_(vdim), ordering_(ordering),
        doftrans_(NULL) {}

   VDofTransformation(DofTransformation & doftrans, int vdim = 1,
                      int ordering = 0)
      : DofTransformation(vdim * doftrans.Height(), vdim * doftrans.Width()),
        vdim_(vdim), ordering_(ordering),
        doftrans_(&doftrans) {}

   inline void SetDofTransformation(DofTransformation & doftrans)
   {
      height_ = vdim_ * doftrans.Height();
      width_  = vdim_ * doftrans.Width();
      doftrans_ = &doftrans;
   }

   void Transform(const double *, double *) const;
   void TransformBack(const double *, double *) const;
   void TransformRowCol(const double *, double *) const;
};

class ND_TetDofTransformation : public DofTransformation
{
private:
   static const double T_data[24];
   static const double TInv_data[24];
   const DenseTensor T, TInv;
   Array<int> Fo;
   int order;

public:
   ND_TetDofTransformation(int order);

   inline void SetFaceOrientation(const Array<int> & face_orientation)
   { Fo = face_orientation; }

   void Transform(const double *, double *) const;

   void TransformBack(const double *, double *) const;

   void TransformRowCol(const double *, double *) const;
};

class ND_WedgeDofTransformation : public DofTransformation
{
public:
   ND_WedgeDofTransformation();

   void Transform(const double *, double *) const;

   void TransformBack(const double *, double *) const;
};

} // namespace mfem

#endif // MFEM_DOFTRANSFORM
