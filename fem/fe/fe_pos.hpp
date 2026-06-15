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

#ifndef MFEM_FE_POS
#define MFEM_FE_POS

#include "fe_base.hpp"
#include "fe_pyramid.hpp"

namespace mfem
{

/** @brief Class for finite elements utilizing the
    always positive Bernstein basis. */
class PositiveFiniteElement : public ScalarFiniteElement
{
public:
   /** @brief Construct PositiveFiniteElement with given
       @param D    Reference space dimension
       @param G    Geometry type (of type Geometry::Type)
       @param Do   Number of degrees of freedom in the FiniteElement
       @param O    Order/degree of the FiniteElement
       @param F    FunctionSpace type of the FiniteElement
   */
   PositiveFiniteElement(int D, Geometry::Type G, int Do, int O,
                         int F = FunctionSpace::Pk) :
      ScalarFiniteElement(D, G, Do, O, F)
   { }

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { ScalarLocalInterpolation(Trans, I, *this); }

   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { ScalarLocalL2Restriction(Trans, R, *this); }

   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override
   { CheckScalarFE(fe).ScalarLocalInterpolation(Trans, I, *this); }

   using FiniteElement::Project;

   // Low-order monotone "projection" (actually it is not a projection): the
   // dofs are set to be the Coefficient values at the nodes.
   void Project(Coefficient &coeff,
                ElementTransformation &Trans, Vector &dofs) const override;

   void Project (VectorCoefficient &vc,
                 ElementTransformation &Trans, Vector &dofs) const override;

   void Project(const FiniteElement &fe, ElementTransformation &Trans,
                DenseMatrix &I) const override;
};


class PositiveTensorFiniteElement : public PositiveFiniteElement,
   public TensorBasisElement
{
public:
   PositiveTensorFiniteElement(const int dims, const int p,
                               const DofMapType dmtype);

   const DofToQuad &GetDofToQuad(const IntegrationRule &ir,
                                 DofToQuad::Mode mode) const override
   {
      return (mode == DofToQuad::FULL) ?
             FiniteElement::GetDofToQuad(ir, mode) :
             GetTensorDofToQuad(*this, ir, mode, basis1d, true, dof2quad_array);
   }

   void GetFaceMap(const int face_id, Array<int> &face_map) const override;
};


/// A 2D positive bi-quadratic element on a square utilizing the 2nd order
/// Bernstein basis
class BiQuadPos2DFiniteElement : public PositiveFiniteElement
{
public:
   /// Construct the BiQuadPos2DFiniteElement
   BiQuadPos2DFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;
   using FiniteElement::Project;
   void Project(Coefficient &coeff, ElementTransformation &Trans,
                Vector &dofs) const override;
   void Project(VectorCoefficient &vc, ElementTransformation &Trans,
                Vector &dofs) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override
   { dofs = 0.; dofs(vertex) = 1.; }
};


/// A 1D quadratic positive element utilizing the 2nd order Bernstein basis
class QuadPos1DFiniteElement : public PositiveFiniteElement
{
public:
   /// Construct the QuadPos1DFiniteElement
   QuadPos1DFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};


/// Arbitrary order H1 elements in 1D utilizing the Bernstein basis
class H1Pos_SegmentElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   // This is to share scratch space between invocations, which helps speed
   // things up, but with OpenMP, we need one copy per thread. Right now, we
   // solve this by allocating this space within each function call every time
   // we call it. Alternatively, we should do some sort thread private thing.
   // Brunner, Jan 2014
   mutable Vector shape_x, dshape_x;
#endif

public:
   /// Construct the H1Pos_SegmentElement of order @a p
   H1Pos_SegmentElement(const int p);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order H1 elements in 2D utilizing the Bernstein basis on a square
class H1Pos_QuadrilateralElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   // See comment in H1Pos_SegmentElement
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif

public:
   /// Construct the H1Pos_QuadrilateralElement of order @a p
   H1Pos_QuadrilateralElement(const int p);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order H1 elements in 3D utilizing the Bernstein basis on a cube
class H1Pos_HexahedronElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   // See comment in H1Pos_SegmentElement.
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif

public:
   /// Construct the H1Pos_HexahedronElement of order @a p
   H1Pos_HexahedronElement(const int p);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order H1 elements in 2D utilizing the Bernstein basis on a triangle
class H1Pos_TriangleElement : public PositiveFiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector m_shape, dshape_1d;
   mutable DenseMatrix m_dshape;
#endif
   Array<int> dof_map;

public:
   /// Construct the H1Pos_TriangleElement of order @a p
   H1Pos_TriangleElement(const int p);

   // The size of shape is (p+1)(p+2)/2 (dof).
   static void CalcShape(const int p, const real_t x, const real_t y,
                         real_t *shape);

   // The size of dshape_1d is p+1; the size of dshape is (dof x dim).
   static void CalcDShape(const int p, const real_t x, const real_t y,
                          real_t *dshape_1d, real_t *dshape);

   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};


/// Arbitrary order H1 elements in 3D utilizing the Bernstein basis on a
/// tetrahedron
class H1Pos_TetrahedronElement : public PositiveFiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector m_shape, dshape_1d;
   mutable DenseMatrix m_dshape;
#endif
   Array<int> dof_map;

public:
   /// Construct the H1Pos_TetrahedronElement of order @a p
   H1Pos_TetrahedronElement(const int p);

   // The size of shape is (p+1)(p+2)(p+3)/6 (dof).
   static void CalcShape(const int p, const real_t x, const real_t y,
                         const real_t z, real_t *shape);

   // The size of dshape_1d is p+1; the size of dshape is (dof x dim).
   static void CalcDShape(const int p, const real_t x, const real_t y,
                          const real_t z, real_t *dshape_1d, real_t *dshape);

   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};


/// Arbitrary order H1 elements in 3D utilizing the Bernstein basis on a wedge
class H1Pos_WedgeElement : public PositiveFiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   H1Pos_TriangleElement TriangleFE;
   H1Pos_SegmentElement  SegmentFE;

public:
   /// Construct the H1Pos_WedgeElement of order @a p
   H1Pos_WedgeElement(const int p);

   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};


/// Arbitrary order H1 elements in 3D utilizing the Bernstein basis on a pyramid
///
/// The pyramid affine-related coordinates $\lambda_i$ for $i=1,\ldots,5$ can
/// be used to define a positive H1 basis by noting that $\lambda_i \ge 0$
/// inside the pyramid for all $i$ and that $\sum_{i=1}^5\lambda_i=1$. This
/// leads to $1 = (\sum_{i=1}^5\lambda_i)^p$. The terms of this product,
/// expanded as a polynomial in the $\lambda_i$, can be used as a Bernstein
/// basis of order $p$ on a pyramid.
class H1Pos_PyramidElement : public PositiveFiniteElement, FuentesPyramid
{
protected:
   const int nterms;
#ifndef MFEM_THREAD_SAFE
   mutable Vector m_shape_1d;
   mutable Vector m_shape;
   mutable DenseMatrix m_dshape;
#endif
   std::map<int,int> dof_map;

   struct Index
   {
      Index() = default;
      int operator()(int i1, int i2, int i3, int i4, int i5)
      {
         const int p = i1 + i2 + i3 + i4 + i5;
         const int min24 = std::min(i2,i4);
         i1 += min24;
         i2 -= min24;
         i3 += min24;
         i4 -= min24;
         return i2 + i3 * (p - i4 - i5) - i4 * i5 * (p + 2)
                - ((i3 - 3) * i3) / 2 + i4 * ((p + 1) * (p + 2)) / 2
                + (i4 * i5 * (i4 + i5)) / 2 + ((i4 - 1) * i4 * (i4 +1)) / 6
                - (p + 2) * ((i4 - 1) * i4) / 2
                + (i5 * (5 + 2 * p - i5) * (i5 * i5 - i5 * (5 + 2 * p)
                                            + 2 * (5 + 5 * p + p * p))) / 24;
      }
   };

public:
   /// Construct the H1Pos_PyramidElement of order @a p
   H1Pos_PyramidElement(const int p);

   // The size of shape is (p+1)(p+2)(p+3)(p+4)/24.
   // The size of shape_1d should be at least p+1.
   static void CalcShape(const int p, const real_t x, const real_t y,
                         const real_t z, real_t *shape_1d, real_t *shape);

   // The size of dshape is (p+1)(p+2)(p+3)(p+4)/24 by 3.
   // The size of dshape_1d should be at least p+1.
   static void CalcDShape(const int p, const real_t x, const real_t y,
                          const real_t z, real_t *dshape_1d, real_t *dshape);

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   // Returns (p+1)(p+2)(p+3)(p+4)/24 which is the size of the temporary arrays
   // needed above
   int GetNumTerms() const { return nterms; }
};


/// Arbitrary order L2 elements in 1D utilizing the Bernstein basis on a segment
class L2Pos_SegmentElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x;
#endif

public:
   /// Construct the L2Pos_SegmentElement of order @a p
   L2Pos_SegmentElement(const int p);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order L2 elements in 2D utilizing the Bernstein basis on a square
class L2Pos_QuadrilateralElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif

public:
   /// Construct the L2Pos_QuadrilateralElement of order @a p
   L2Pos_QuadrilateralElement(const int p);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order L2 elements in 3D utilizing the Bernstein basis on a cube
class L2Pos_HexahedronElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif

public:
   /// Construct the L2Pos_HexahedronElement of order @a p
   L2Pos_HexahedronElement(const int p);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order L2 elements in 2D utilizing the Bernstein basis on a triangle
class L2Pos_TriangleElement : public PositiveFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector dshape_1d;
#endif

public:
   /// Construct the L2Pos_TriangleElement of order @a p
   L2Pos_TriangleElement(const int p);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order L2 elements in 3D utilizing the Bernstein basis on a
/// tetrahedron
class L2Pos_TetrahedronElement : public PositiveFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector dshape_1d;
#endif

public:
   /// Construct the L2Pos_TetrahedronElement of order @a p
   L2Pos_TetrahedronElement(const int p);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order L2 elements in 3D utilizing the Bernstein basis on a wedge
class L2Pos_WedgeElement : public PositiveFiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   L2Pos_TriangleElement TriangleFE;
   L2Pos_SegmentElement  SegmentFE;

public:
   /// Construct the L2Pos_WedgeElement of order @a p
   L2Pos_WedgeElement(const int p);

   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

/// Arbitrary order L2 elements in 3D utilizing the Bernstein basis on a pyramid
class L2Pos_PyramidElement : public PositiveFiniteElement, FuentesPyramid
{
protected:
   const int nterms;
#ifndef MFEM_THREAD_SAFE
   mutable Vector m_shape_1d;
   mutable Vector m_shape;
   mutable DenseMatrix m_dshape;
#endif
   std::map<int,int> dof_map;

   struct Index
   {
      Index() = default;
      int operator()(int i1, int i2, int i3, int i4, int i5)
      {
         const int p = i1 + i2 + i3 + i4 + i5;
         const int min24 = std::min(i2,i4);
         i1 += min24;
         i2 -= min24;
         i3 += min24;
         i4 -= min24;
         return i2 + i3 * (p - i4 - i5) - i4 * i5 * (p + 2)
                - ((i3 - 3) * i3) / 2 + i4 * ((p + 1) * (p + 2)) / 2
                + (i4 * i5 * (i4 + i5)) / 2 + ((i4 - 1) * i4 * (i4 +1)) / 6
                - (p + 2) * ((i4 - 1) * i4) / 2
                + (i5 * (5 + 2 * p - i5) * (i5 * i5 - i5 * (5 + 2 * p)
                                            + 2 * (5 + 5 * p + p * p))) / 24;
      }
   };

   // Returns (p+1)(p+2)(p+3)(p+4)/24 which is the size of the temporary arrays
   // needed below
   int GetNumTerms() const { return nterms; }

   // The size of shape is (p+1)(p+2)(p+3)(p+4)/24.
   // The size of shape_1d should be at least p+1.
   static void CalcShape(const int p, const real_t x, const real_t y,
                         const real_t z, real_t *shape_1d, real_t *shape);

   // The size of dshape is (p+1)(p+2)(p+3)(p+4)/24 by 3.
   // The size of dshape_1d should be at least p+1.
   static void CalcDShape(const int p, const real_t x, const real_t y,
                          const real_t z, real_t *dshape_1d, real_t *dshape);

public:
   /// Construct the L2Pos_PyramidElement of order @a p
   L2Pos_PyramidElement(const int p);

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

} // namespace mfem

#endif
