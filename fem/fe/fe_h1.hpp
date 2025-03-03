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

#ifndef MFEM_FE_H1
#define MFEM_FE_H1

#include "fe_base.hpp"
#include "fe_pyramid.hpp"

namespace mfem
{

/// Arbitrary order H1 elements in 1D
class H1_SegmentElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x, d2shape_x;
#endif

public:
   /// Construct the H1_SegmentElement of order @a p and BasisType @a btype
   H1_SegmentElement(const int p, const int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &Hessian) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order H1 elements in 2D on a square
class H1_QuadrilateralElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y, d2shape_x, d2shape_y;
#endif

public:
   /// Construct the H1_QuadrilateralElement of order @a p and BasisType @a btype
   H1_QuadrilateralElement(const int p,
                           const int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &Hessian) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order H1 elements in 3D on a cube
class H1_HexahedronElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z,
           d2shape_x, d2shape_y, d2shape_z;
#endif

public:
   /// Construct the H1_HexahedronElement of order @a p and BasisType @a btype
   H1_HexahedronElement(const int p, const int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &Hessian) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// Arbitrary order H1 elements in 2D on a triangle
class H1_TriangleElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
   mutable Vector ddshape_x, ddshape_y, ddshape_l;
   mutable DenseMatrix du, ddu;
#endif
   DenseMatrixInverse Ti;

public:
   /// Construct the H1_TriangleElement of order @a p and BasisType @a btype
   H1_TriangleElement(const int p, const int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &ddshape) const override;
};


/// Arbitrary order H1 elements in 3D  on a tetrahedron
class H1_TetrahedronElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l, u;
   mutable Vector ddshape_x, ddshape_y, ddshape_z, ddshape_l;
   mutable DenseMatrix du, ddu;
#endif
   DenseMatrixInverse Ti;

public:
   /// Construct the H1_TetrahedronElement of order @a p and BasisType @a btype
   H1_TetrahedronElement(const int p,
                         const int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &ddshape) const override;
};



/// Arbitrary order H1 elements in 3D on a wedge
class H1_WedgeElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   H1_TriangleElement TriangleFE;
   H1_SegmentElement  SegmentFE;

public:
   /// Construct the H1_WedgeElement of order @a p and BasisType @a btype
   H1_WedgeElement(const int p,
                   const int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

/** Arbitrary order H1 basis functions defined on pyramid-shaped elements

  This implementation is closely based on the finite elements
  described in section 9.1 of the paper "Orientation embedded high
  order shape functions for the exact sequence elements of all shapes"
  by Federico Fuentes, Brendan Keith, Leszek Demkowicz, and Sriram
  Nagaraj, see https://doi.org/10.1016/j.camwa.2015.04.027.
 */
class H1_FuentesPyramidElement
   : public NodalFiniteElement, public FuentesPyramid
{
private:
   mutable real_t zmax;

#ifndef MFEM_THREAD_SAFE
   mutable Vector tmp_i, tmp_u;
   mutable DenseMatrix tmp1_ij, tmp2_ij, tmp_du;
   mutable DenseTensor tmp_ijk;
#endif
   DenseMatrixInverse Ti;

   void calcBasis(const int p, const IntegrationPoint &ip,
                  Vector &phi_i, DenseMatrix &phi_ij, Vector &u) const;
   void calcGradBasis(const int p, const IntegrationPoint &ip,
                      Vector &phi_i, DenseMatrix &dphi_i,
                      DenseMatrix &phi_ij, DenseTensor &dphi_ij,
                      DenseMatrix &du) const;

public:
   H1_FuentesPyramidElement(const int p,
                            const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   void CalcRawShape(const IntegrationPoint &ip, Vector &shape) const;
   void CalcRawDShape(const IntegrationPoint &ip,
                      DenseMatrix &dshape) const;

   real_t GetZetaMax() const { return zmax; }
};

/** Arbitrary order H1 basis functions defined on pyramid-shaped elements

  This implementation is based on the finite elements described in the
  2010 paper "Higher-Order Finite Elements for Hybrid Meshes Using New
  Nodal Pyramidal Elements" by Morgane Bergot, Gary Cohen, and Marc
  Durufle, see https://hal.archives-ouvertes.fr/hal-00454261.
 */
class H1_BergotPyramidElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_z_dt, u;
   mutable Vector ddshape_x, ddshape_y, ddshape_z;
   mutable DenseMatrix du, ddu;
#endif
   DenseMatrixInverse Ti;

public:
   H1_BergotPyramidElement(const int p,
                           const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

} // namespace mfem

#endif
