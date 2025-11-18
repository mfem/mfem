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

#ifndef MFEM_FE_H1_BUBBLE
#define MFEM_FE_H1_BUBBLE

#include "fe_base.hpp"

namespace mfem
{

/// Arbitrary order H1 plus bubble elements in 2D on a triangle
class H1Bubble_TriangleElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
   mutable DenseMatrix du;
#endif
   int bubble_order;
   DenseMatrix T_pinv;

public:
   // Degree-p polynomials, enriched with degree-q bubbles.
   H1Bubble_TriangleElement(int p, int q, int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

/// Arbitrary order H1 plus bubble elements in 2D on a quadrilateral
class H1Bubble_QuadrilateralElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y, u;
   mutable DenseMatrix du;
#endif
   int bubble_order;
   DenseMatrix T_pinv;

public:
   // Degree-p polynomials, enriched with degree-q bubbles.
   H1Bubble_QuadrilateralElement(
      int p, int q, int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

} // namespace mfem

#endif
