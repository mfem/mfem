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

   Array<int> Fo;

public:
   DofTransformation(int height, int width)
      : height_(height), width_(width) {}

   inline int Height() const { return height_; }
   inline int NumRows() const { return height_; }
   inline int Width() const { return width_; }
   inline int NumCols() const { return width_; }

   inline void SetFaceOrientations(const Array<int> & face_orientation)
   { Fo = face_orientation; }

   virtual void TransformPrimal(const double *, double *) const = 0;

   virtual void TransformPrimal(const Vector &, Vector &) const;

   virtual void TransformPrimalRows(const DenseMatrix &, DenseMatrix &) const;

   virtual void TransformPrimalCols(const DenseMatrix &, DenseMatrix &) const;

   virtual void InvTransformPrimal(const double *, double *) const = 0;

   virtual void InvTransformPrimal(const Vector &, Vector &) const;

   virtual void TransformDual(const double *, double *) const = 0;

   virtual void TransformDual(const DenseMatrix &, DenseMatrix &) const;

   virtual void TransformDualRows(const DenseMatrix &, DenseMatrix &) const;

   virtual void TransformDualCols(const DenseMatrix &, DenseMatrix &) const;

   virtual ~DofTransformation() {}
};

void TransformPrimal(const DofTransformation *, const DofTransformation *,
                     const DenseMatrix &, DenseMatrix &);

void TransformDual(const DofTransformation *, const DofTransformation *,
                   const DenseMatrix &, DenseMatrix &);

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

   inline void SetVDim(int vdim)
   {
      vdim_ = vdim;
      if (doftrans_)
      {
         height_ = vdim_ * doftrans_->Height();
         width_  = vdim_ * doftrans_->Width();
      }
   }

   inline void SetDofTransformation(DofTransformation & doftrans)
   {
      height_ = vdim_ * doftrans.Height();
      width_  = vdim_ * doftrans.Width();
      doftrans_ = &doftrans;
   }

   inline void SetFaceOrientation(const Array<int> & face_orientation)
   { Fo = face_orientation; doftrans_->SetFaceOrientations(face_orientation);}

   using DofTransformation::TransformPrimal;
   using DofTransformation::InvTransformPrimal;
   using DofTransformation::TransformDual;

   void TransformPrimal(const double *, double *) const;
   void InvTransformPrimal(const double *, double *) const;
   void TransformDual(const double *, double *) const;
};

class ND_DofTransformation : public DofTransformation
{
protected:
   static const double T_data[24];
   static const double TInv_data[24];
   static const DenseTensor T, TInv;
   int order;

   ND_DofTransformation(int height, int width, int order);

public:
   static const DenseMatrix & GetFaceTransform(int ori) { return T(ori); }
   static const DenseMatrix & GetFaceInverseTransform(int ori)
   { return TInv(ori); }
};

class ND_TetDofTransformation : public ND_DofTransformation
{
public:
   ND_TetDofTransformation(int order);

   using DofTransformation::TransformPrimal;
   using DofTransformation::InvTransformPrimal;
   using DofTransformation::TransformDual;

   void TransformPrimal(const double *, double *) const;

   void InvTransformPrimal(const double *, double *) const;

   void TransformDual(const double *, double *) const;
};

class ND_WedgeDofTransformation : public ND_DofTransformation
{
public:
   ND_WedgeDofTransformation(int order);

   using DofTransformation::TransformPrimal;
   using DofTransformation::InvTransformPrimal;
   using DofTransformation::TransformDual;

   void TransformPrimal(const double *, double *) const;

   void InvTransformPrimal(const double *, double *) const;

   void TransformDual(const double *, double *) const;
};

} // namespace mfem

#endif // MFEM_DOFTRANSFORM
