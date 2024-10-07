// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FE_L2
#define MFEM_FE_L2

#include "fe_base.hpp"

namespace mfem
{

/// Arbitrary order L2 elements in 1D on a segment
class L2_SegmentElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x;
#endif

public:
   /// Construct the L2_SegmentElement of order @a p and BasisType @a btype
   L2_SegmentElement(const int p, const int btype = BasisType::GaussLegendre);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;

   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { ScalarLocalL2Restriction(Trans, R, *this); }

};


/// Arbitrary order L2 elements in 2D on a square
class L2_QuadrilateralElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif

public:
   /// Construct the L2_QuadrilateralElement of order @a p and BasisType @a btype
   L2_QuadrilateralElement(const int p,
                           const int btype = BasisType::GaussLegendre);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
   void ProjectCurl(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &curl) const override
   { ProjectCurl_2D(fe, Trans, curl); }

   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { ScalarLocalL2Restriction(Trans, R, *this); }

   using FiniteElement::Project;
   void ProjectDiv(const FiniteElement &fe,
                   ElementTransformation &Trans,
                   DenseMatrix &div) const override;
   void Project(Coefficient &coeff,
                ElementTransformation &Trans, Vector &dofs) const override;
};


/// Arbitrary order L2 elements in 3D on a cube
class L2_HexahedronElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif

public:
   /// Construct the L2_HexahedronElement of order @a p and BasisType @a btype
   L2_HexahedronElement(const int p,
                        const int btype = BasisType::GaussLegendre);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;

   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { ScalarLocalL2Restriction(Trans, R, *this); }

   using FiniteElement::Project;
   void ProjectDiv(const FiniteElement &fe,
                   ElementTransformation &Trans,
                   DenseMatrix &div) const override;
   void Project(Coefficient &coeff,
                ElementTransformation &Trans, Vector &dofs) const override;
};


/// Arbitrary order L2 elements in 2D on a triangle
class L2_TriangleElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
   mutable DenseMatrix du;
#endif
   DenseMatrixInverse Ti;

public:
   /// Construct the L2_TriangleElement of order @a p and BasisType @a btype
   L2_TriangleElement(const int p,
                      const int btype = BasisType::GaussLegendre);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
   void ProjectCurl(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &curl) const override
   { ProjectCurl_2D(fe, Trans, curl); }

   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { ScalarLocalL2Restriction(Trans, R, *this); }

};


/// Arbitrary order L2 elements in 3D on a tetrahedron
class L2_TetrahedronElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l, u;
   mutable DenseMatrix du;
#endif
   DenseMatrixInverse Ti;

public:
   /// Construct the L2_TetrahedronElement of order @a p and BasisType @a btype
   L2_TetrahedronElement(const int p,
                         const int btype = BasisType::GaussLegendre);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;

   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { ScalarLocalL2Restriction(Trans, R, *this); }

};


/// Arbitrary order L2 elements in 3D on a wedge
class L2_WedgeElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   L2_TriangleElement TriangleFE;
   L2_SegmentElement  SegmentFE;

public:
   /// Construct the L2_WedgeElement of order @a p and BasisType @a btype
   L2_WedgeElement(const int p,
                   const int btype = BasisType::GaussLegendre);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

} // namespace mfem

#endif
