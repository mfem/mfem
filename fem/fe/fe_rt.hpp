// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FE_RT
#define MFEM_FE_RT

#include "fe_base.hpp"
#include "fe_h1.hpp"
#include "fe_l2.hpp"

namespace mfem
{

/// Arbitrary order Raviart-Thomas elements in 2D on a square
class RT_QuadrilateralElement : public VectorTensorFiniteElement
{
private:
   static const double nk[8];

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
   mutable Vector dshape_cx, dshape_cy;
#endif
   Array<int> dof2nk;
   const double *cp;

public:
   /** @brief Construct the RT_QuadrilateralElement of order @a p and closed and
       open BasisType @a cb_type and @a ob_type */
   RT_QuadrilateralElement(const int p,
                           const int cb_type = BasisType::GaussLobatto,
                           const int ob_type = BasisType::GaussLegendre);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   {
      if (obasis1d.IsIntegratedType()) { ProjectIntegrated(vc, Trans, dofs); }
      else { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   }
   virtual void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                                 Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   // Gradient + rotation = Curl: H1 -> H(div)
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, grad); }
   // Curl = Gradient + rotation: H1 -> H(div)
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, curl); }

   virtual void GetFaceMap(const int face_id, Array<int> &face_map) const;

protected:
   void ProjectIntegrated(VectorCoefficient &vc, ElementTransformation &Trans,
                          Vector &dofs) const;
};


/// Arbitrary order Raviart-Thomas elements in 3D on a cube
class RT_HexahedronElement : public VectorTensorFiniteElement
{
   static const double nk[18];

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy, shape_cz, shape_oz;
   mutable Vector dshape_cx, dshape_cy, dshape_cz;
#endif
   Array<int> dof2nk;
   const double *cp;

public:
   /** @brief Construct the RT_HexahedronElement of order @a p and closed and
       open BasisType @a cb_type and @a ob_type */
   RT_HexahedronElement(const int p,
                        const int cb_type = BasisType::GaussLobatto,
                        const int ob_type = BasisType::GaussLegendre);

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   {
      if (obasis1d.IsIntegratedType()) { ProjectIntegrated(vc, Trans, dofs); }
      else { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   }
   virtual void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                                 Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_RT(nk, dof2nk, fe, Trans, curl); }

   /// @brief Return the mapping from lexicographically ordered face DOFs to
   /// lexicographically ordered element DOFs corresponding to local face
   /// @a face_id.
   virtual void GetFaceMap(const int face_id, Array<int> &face_map) const;

protected:
   void ProjectIntegrated(VectorCoefficient &vc,
                          ElementTransformation &Trans,
                          Vector &dofs) const;
};


/// Arbitrary order Raviart-Thomas elements in 2D on a triangle
class RT_TriangleElement : public VectorFiniteElement
{
   static const double nk[6], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_l;
   mutable DenseMatrix u;
   mutable Vector divu;
#endif
   Array<int> dof2nk;
   DenseMatrixInverse Ti;

public:
   /// Construct the RT_TriangleElement of order @a p
   RT_TriangleElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                                 Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   // Gradient + rotation = Curl: H1 -> H(div)
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, grad); }
   // Curl = Gradient + rotation: H1 -> H(div)
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, curl); }
};


/// Arbitrary order Raviart-Thomas elements in 3D on a tetrahedron
class RT_TetrahedronElement : public VectorFiniteElement
{
   static const double nk[12], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l;
   mutable DenseMatrix u;
   mutable Vector divu;
#endif
   Array<int> dof2nk;
   DenseMatrixInverse Ti;

public:
   /// Construct the RT_TetrahedronElement of order @a p
   RT_TetrahedronElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                                 Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_RT(nk, dof2nk, fe, Trans, curl); }
};

class RT_WedgeElement : public VectorFiniteElement
{
   static const double nk[15];

#ifndef MFEM_THREAD_SAFE
   mutable Vector      tl2_shape;
   mutable Vector      sh1_shape;
   mutable DenseMatrix trt_shape;
   mutable Vector      sl2_shape;
   mutable DenseMatrix sh1_dshape;
   mutable Vector      trt_dshape;
#endif
   Array<int> dof2nk, t_dof, s_dof;

   // The RT_Wedge is implemented as the sum of tensor products of
   // lower dimensional basis funcgtions.
   // Specifically: L2TriangleFE x H1SegmentFE + RTTriangle x L2SegmentFE
   L2_TriangleElement L2TriangleFE;
   RT_TriangleElement RTTriangleFE;
   H1_SegmentElement  H1SegmentFE;
   L2_SegmentElement  L2SegmentFE;

public:
   RT_WedgeElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_RT(nk, dof2nk, fe, Trans, curl); }
};


/// Arbitrary order, three component, Raviart-Thomas elements in 1D on a segment
/** RT_R1D_SegmentElement provides a representation of a three component
    Raviart-Thomas basis where the vector components vary along only one
    dimension.
*/
class RT_R1D_SegmentElement : public VectorFiniteElement
{
   static const double nk[9];
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox;
   mutable Vector dshape_cx;
#endif
   Array<int> dof_map, dof2nk;

   Poly_1D::Basis &cbasis1d, &obasis1d;

public:
   /** @brief Construct the RT_R1D_SegmentElement of order @a p and closed and
       open BasisType @a cb_type and @a ob_type */
   RT_R1D_SegmentElement(const int p,
                         const int cb_type = BasisType::GaussLobatto,
                         const int ob_type = BasisType::GaussLegendre);

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const;

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   using FiniteElement::Project;

   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const;

   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const;

   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const;
};


/** RT_R2D_SegmentElement provides a representation of a 3D Raviart-Thomas
    basis where the vector field is assumed constant in the third dimension.
*/
class RT_R2D_SegmentElement : public VectorFiniteElement
{
   static const double nk[2];
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_ox;
#endif
   Array<int> dof_map, dof2nk;

   Poly_1D::Basis &obasis1d;

private:
   void LocalInterpolation(const VectorFiniteElement &cfe,
                           ElementTransformation &Trans,
                           DenseMatrix &I) const;

public:
   /** @brief Construct the RT_R2D_SegmentElement of order @a p and open
       BasisType @a ob_type */
   RT_R2D_SegmentElement(const int p,
                         const int ob_type = BasisType::GaussLegendre);

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const;

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &div_shape) const;

   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation(*this, Trans, I); }

   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { MFEM_ABORT("method is not overloaded"); }

   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation(CheckVectorFE(fe), Trans, I); }
};

class RT_R2D_FiniteElement : public VectorFiniteElement
{
protected:
   const double *nk;
   Array<int> dof_map, dof2nk;

   RT_R2D_FiniteElement(int p, Geometry::Type G, int Do, const double *nk_fe);

private:
   void LocalInterpolation(const VectorFiniteElement &cfe,
                           ElementTransformation &Trans,
                           DenseMatrix &I) const;

public:
   using FiniteElement::CalcVShape;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const;

   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation(*this, Trans, I); }

   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const;

   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation(CheckVectorFE(fe), Trans, I); }

   using FiniteElement::Project;

   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const;

   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const;

   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const;
};

/// Arbitrary order Raviart-Thomas 3D elements in 2D on a triangle
class RT_R2D_TriangleElement : public RT_R2D_FiniteElement
{
private:
   static const double nk_t[12];

#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix rt_shape;
   mutable Vector      l2_shape;
   mutable Vector      rt_dshape;
#endif

   RT_TriangleElement RT_FE;
   L2_TriangleElement L2_FE;

public:
   /** @brief Construct the RT_R2D_TriangleElement of order @a p */
   RT_R2D_TriangleElement(const int p);

   using RT_R2D_FiniteElement::CalcVShape;

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
};

/// Arbitrary order Raviart-Thomas 3D elements in 2D on a square
class RT_R2D_QuadrilateralElement : public RT_R2D_FiniteElement
{
private:
   static const double nk_q[15];

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
   mutable Vector dshape_cx, dshape_cy;
#endif

   Poly_1D::Basis &cbasis1d, &obasis1d;

public:
   /** @brief Construct the RT_QuadrilateralElement of order @a p and closed and
       open BasisType @a cb_type and @a ob_type */
   RT_R2D_QuadrilateralElement(const int p,
                               const int cb_type = BasisType::GaussLobatto,
                               const int ob_type = BasisType::GaussLegendre);

   using RT_R2D_FiniteElement::CalcVShape;

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
};


} // namespace mfem

#endif
