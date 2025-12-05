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
   int base_order;
   int bubble_order;
   DenseMatrix T_pinv;

public:
   /// @brief Construct the triangular bubble element with degree-p polynomials,
   /// enriched with cubic bubble times degree q polynomial.
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
   int base_order;
   int bubble_order;
   DenseMatrix T_pinv;

public:
   /// @brief Construct the quadrilateral bubble element with degree-p
   /// polynomials, enriched with biquadratic bubble times degree q polynomial.
   H1Bubble_QuadrilateralElement(
      int p, int q, int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

/// Arbitrary order H1 plus bubble elements in 3D on a tetrahedron
class H1Bubble_TetrahedronElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l, u;
   mutable DenseMatrix du;
#endif
   int base_order;
   int bubble_order;
   DenseMatrix T_pinv;

public:
   /// @brief Construct the tetrahedral bubble element with degree-p
   /// polynomials, enriched with quartic bubble times degree q polynomial.
   H1Bubble_TetrahedronElement(int p, int q, int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

/// Arbitrary order H1 plus bubble elements in 3D on a hexahedron
class H1Bubble_HexahedronElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z;
   mutable Vector dshape_x, dshape_y, dshape_z, u;
   mutable DenseMatrix du;
#endif
   int base_order;
   int bubble_order;
   DenseMatrix T_pinv;

public:
   /// @brief Construct the hexahedral bubble element with degree-p polynomials,
   /// enriched with triquadratic bubble times degree q polynomial.
   H1Bubble_HexahedronElement(int p, int q, int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

} // namespace mfem

#endif
