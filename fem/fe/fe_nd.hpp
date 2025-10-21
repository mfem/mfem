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

#ifndef MFEM_FE_ND
#define MFEM_FE_ND

#include "fe_base.hpp"
#include "fe_h1.hpp"
#include "fe_pyramid.hpp"

namespace mfem
{

/// Arbitrary order Nedelec elements in 3D on a cube
class ND_HexahedronElement : public VectorTensorFiniteElement
{
   static const real_t tk[18];
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy, shape_cz, shape_oz;
   mutable Vector dshape_cx, dshape_cy, dshape_cz;
#endif
   Array<int> dof2tk;
   const real_t *cp;

public:
   /** @brief Construct the ND_HexahedronElement of order @a p and closed and
       open BasisType @a cb_type and @a ob_type */
   ND_HexahedronElement(const int p,
                        const int cb_type = BasisType::GaussLobatto,
                        const int ob_type = BasisType::GaussLegendre);

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_ND(Trans, shape); }

   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }

   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }

   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override
   {
      if (obasis1d.IsIntegratedType()) { ProjectIntegrated(vc, Trans, dofs); }
      else { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   }

   void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                         Vector &dofs) const override
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }

   void ProjectMatrixCoefficient(MatrixCoefficient &mc,
                                 ElementTransformation &T,
                                 Vector &dofs) const override
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }

   void Project(const FiniteElement &fe,
                ElementTransformation &Trans,
                DenseMatrix &I) const override
   { Project_ND(tk, dof2tk, fe, Trans, I); }

   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }

   void ProjectCurl(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &curl) const override
   { ProjectCurl_ND(tk, dof2tk, fe, Trans, curl); }

   void GetFaceMap(const int face_id, Array<int> &face_map) const override;

protected:
   void ProjectIntegrated(VectorCoefficient &vc,
                          ElementTransformation &Trans,
                          Vector &dofs) const;
};


/// Arbitrary order Nedelec elements in 2D on a square
class ND_QuadrilateralElement : public VectorTensorFiniteElement
{
   static const real_t tk[8];

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
   mutable Vector dshape_cx, dshape_cy;
#endif
   Array<int> dof2tk;
   const real_t *cp;

public:
   /** @brief Construct the ND_QuadrilateralElement of order @a p and closed and
       open BasisType @a cb_type and @a ob_type */
   ND_QuadrilateralElement(const int p,
                           const int cb_type = BasisType::GaussLobatto,
                           const int ob_type = BasisType::GaussLegendre);
   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;
   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_ND(Trans, shape); }
   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;
   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override
   {
      if (obasis1d.IsIntegratedType()) { ProjectIntegrated(vc, Trans, dofs); }
      else { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   }
   void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                         Vector &dofs) const override
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   void ProjectMatrixCoefficient(MatrixCoefficient &mc,
                                 ElementTransformation &T,
                                 Vector &dofs) const override
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   void Project(const FiniteElement &fe,
                ElementTransformation &Trans,
                DenseMatrix &I) const override
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }

   void GetFaceMap(const int face_id, Array<int> &face_map) const override;

protected:
   void ProjectIntegrated(VectorCoefficient &vc,
                          ElementTransformation &Trans,
                          Vector &dofs) const;
};


/// Arbitrary order Nedelec elements in 3D on a tetrahedron
class ND_TetrahedronElement : public VectorFiniteElement
{
   static const real_t tk[18], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l;
   mutable DenseMatrix u;
#endif
   Array<int> dof2tk;
   DenseMatrixInverse Ti;

   ND_TetDofTransformation doftrans;

public:
   /// Construct the ND_TetrahedronElement of order @a p
   ND_TetrahedronElement(const int p);
   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;
   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_ND(Trans, shape); }
   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;
   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }
   const StatelessDofTransformation *GetDofTransformation() const override
   { return &doftrans; }
   using FiniteElement::Project;
   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                         Vector &dofs) const override
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   void ProjectMatrixCoefficient(MatrixCoefficient &mc,
                                 ElementTransformation &T,
                                 Vector &dofs) const override
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   void Project(const FiniteElement &fe,
                ElementTransformation &Trans,
                DenseMatrix &I) const override
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }

   void ProjectCurl(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &curl) const override
   { ProjectCurl_ND(tk, dof2tk, fe, Trans, curl); }
};

/// Arbitrary order Nedelec elements in 2D on a triangle
class ND_TriangleElement : public VectorFiniteElement
{
   static const real_t tk[8], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_l;
   mutable DenseMatrix u;
   mutable Vector curlu;
#endif
   Array<int> dof2tk;
   DenseMatrixInverse Ti;

   ND_TriDofTransformation doftrans;

public:
   /// Construct the ND_TriangleElement of order @a p
   ND_TriangleElement(const int p);
   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;
   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_ND(Trans, shape); }
   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;
   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }
   const StatelessDofTransformation *GetDofTransformation() const override
   { return &doftrans; }
   using FiniteElement::Project;
   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                         Vector &dofs) const override
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   void ProjectMatrixCoefficient(MatrixCoefficient &mc,
                                 ElementTransformation &T,
                                 Vector &dofs) const override
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   void Project(const FiniteElement &fe,
                ElementTransformation &Trans,
                DenseMatrix &I) const override
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }
};


/// Arbitrary order Nedelec elements in 1D on a segment
class ND_SegmentElement : public VectorTensorFiniteElement
{
   static const real_t tk[1];

   Array<int> dof2tk;

public:
   /** @brief Construct the ND_SegmentElement of order @a p and open
       BasisType @a ob_type */
   ND_SegmentElement(const int p, const int ob_type = BasisType::GaussLegendre);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override
   { obasis1d.Eval(ip.x, shape); }
   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;
   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_ND(Trans, shape); }
   // void CalcCurlShape(const IntegrationPoint &ip,
   //                            DenseMatrix &curl_shape) const;
   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   void ProjectMatrixCoefficient(MatrixCoefficient &mc,
                                 ElementTransformation &T,
                                 Vector &dofs) const override
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   void Project(const FiniteElement &fe,
                ElementTransformation &Trans,
                DenseMatrix &I) const override
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }
};

class ND_WedgeElement : public VectorFiniteElement
{
private:
   static const real_t tk[15];

#ifndef MFEM_THREAD_SAFE
   mutable Vector      t1_shape, s1_shape;
   mutable DenseMatrix tn_shape, sn_shape;
   mutable DenseMatrix t1_dshape, s1_dshape, tn_dshape;
#endif
   Array<int> dof2tk, t_dof, s_dof;

   ND_WedgeDofTransformation doftrans;

   H1_TriangleElement H1TriangleFE;
   ND_TriangleElement NDTriangleFE;
   H1_SegmentElement  H1SegmentFE;
   ND_SegmentElement  NDSegmentFE;

public:
   ND_WedgeElement(const int p,
                   const int cb_type = BasisType::GaussLobatto,
                   const int ob_type = BasisType::GaussLegendre);

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_ND(Trans, shape); }

   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }

   const StatelessDofTransformation *GetDofTransformation() const override
   { return &doftrans; }

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }

   void ProjectMatrixCoefficient(MatrixCoefficient &mc,
                                 ElementTransformation &T,
                                 Vector &dofs) const override
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }

   void Project(const FiniteElement &fe,
                ElementTransformation &Trans,
                DenseMatrix &I) const override
   { Project_ND(tk, dof2tk, fe, Trans, I); }

   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }

   void ProjectCurl(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &curl) const override
   { ProjectCurl_ND(tk, dof2tk, fe, Trans, curl); }
};


/** Arbitrary order H(Curl) basis functions defined on pyramid-shaped elements

  This implementation is closely based on the finite elements
  described in section 9.2 of the paper "Orientation embedded high
  order shape functions for the exact sequence elements of all shapes"
  by Federico Fuentes, Brendan Keith, Leszek Demkowicz, and Sriram
  Nagaraj, see https://doi.org/10.1016/j.camwa.2015.04.027.
 */
class ND_FuentesPyramidElement
   : public VectorFiniteElement, public FuentesPyramid
{
private:
   static const real_t tk[27];

   mutable real_t zmax;

#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix tmp_E_E_ij;
   mutable DenseMatrix tmp_dE_E_ij;
   mutable DenseTensor tmp_E_Q1_ijk;
   mutable DenseTensor tmp_dE_Q1_ijk;
   mutable DenseTensor tmp_E_Q2_ijk;
   mutable DenseTensor tmp_dE_Q2_ijk;
   mutable DenseTensor tmp_E_T_ijk;
   mutable DenseTensor tmp_dE_T_ijk;
   mutable DenseMatrix tmp_phi_Q1_ij;
   mutable DenseTensor tmp_dphi_Q1_ij;
   mutable DenseMatrix tmp_phi_Q2_ij;
   mutable DenseTensor tmp_dphi_Q2_ij;
   mutable Vector      tmp_phi_E_i;
   mutable DenseMatrix tmp_dphi_E_i;
   mutable DenseMatrix u;
   mutable DenseMatrix curlu;
#endif
   Array<int> dof2tk;
   DenseMatrixInverse Ti;

   ND_PyramidDofTransformation doftrans;

   void calcBasis(const int p, const IntegrationPoint & ip,
                  DenseMatrix & E_E_ik,
                  DenseTensor & E_Q1_ijk,
                  DenseTensor & E_Q2_ijk,
                  DenseTensor & E_T_ijk,
                  DenseMatrix & phi_Q1_ij,
                  DenseTensor & dphi_Q1_ij,
                  DenseMatrix & phi_Q2_ij,
                  Vector      & phi_E_k,
                  DenseMatrix & dphi_E_k,
                  DenseMatrix & W) const;

   void calcCurlBasis(const int p, const IntegrationPoint & ip,
                      DenseMatrix & E_E_ik,
                      DenseMatrix & dE_E_ik,
                      DenseTensor & E_Q1_ijk,
                      DenseTensor & dE_Q1_ijk,
                      DenseTensor & E_Q2_ijk,
                      DenseTensor & dE_Q2_ijk,
                      DenseTensor & E_T_ijk,
                      DenseTensor & dE_T_ijk,
                      DenseMatrix & phi_Q2_ij,
                      DenseTensor & dphi_Q2_ij,
                      Vector      & phi_E_k,
                      DenseMatrix & dphi_E_k,
                      DenseMatrix & dW) const;

public:
   ND_FuentesPyramidElement(const int p,
                            const int cb_type = BasisType::GaussLobatto,
                            const int ob_type = BasisType::GaussLegendre);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }

   using FiniteElement::Project;

   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }

   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }

   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_ND(tk, dof2tk, fe, Trans, curl); }

   void CalcRawVShape(const IntegrationPoint &ip,
                      DenseMatrix &shape) const;

   void CalcRawCurlShape(const IntegrationPoint &ip,
                         DenseMatrix &dshape) const;

   real_t GetZetaMax() const { return zmax; }
};


/// A 0D Nedelec finite element for the boundary of a 1D domain
/** ND_R1D_PointElement provides a representation of the trace of a three
    component Nedelec basis restricted to 1D.
*/
class ND_R1D_PointElement : public VectorFiniteElement
{
   static const real_t tk[9];

public:
   /** @brief Construct the ND_R1D_PointElement */
   ND_R1D_PointElement(int p);

   using FiniteElement::CalcVShape;

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override;
};

/// Arbitrary order, three component, Nedelec elements in 1D on a segment
/** ND_R1D_SegmentElement provides a representation of a three component Nedelec
    basis where the vector components vary along only one dimension.
*/
class ND_R1D_SegmentElement : public VectorFiniteElement
{
   static const real_t tk[9];
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox;
   mutable Vector dshape_cx;
#endif
   Array<int> dof_map, dof2tk;

   Poly_1D::Basis &cbasis1d, &obasis1d;

public:
   /** @brief Construct the ND_R1D_SegmentElement of order @a p and closed and
       open BasisType @a cb_type and @a ob_type */
   ND_R1D_SegmentElement(const int p,
                         const int cb_type = BasisType::GaussLobatto,
                         const int ob_type = BasisType::GaussLegendre);

   using FiniteElement::CalcVShape;
   using FiniteElement::CalcPhysCurlShape;

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override;

   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;

   void CalcPhysCurlShape(ElementTransformation &Trans,
                          DenseMatrix &curl_shape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }

   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }

   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;

   void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                         Vector &dofs) const override
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }

   void ProjectMatrixCoefficient(MatrixCoefficient &mc,
                                 ElementTransformation &T,
                                 Vector &dofs) const override
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }

   void Project(const FiniteElement &fe,
                ElementTransformation &Trans,
                DenseMatrix &I) const override;

   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }

   void ProjectCurl(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &curl) const override
   { ProjectCurl_ND(tk, dof2tk, fe, Trans, curl); }
};


/** ND_R2D_SegmentElement provides a representation of a 3D Nedelec
    basis where the vector field is assumed constant in the third dimension.
*/
class ND_R2D_SegmentElement : public VectorFiniteElement
{
   static const real_t tk[4];
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox;
   mutable Vector dshape_cx;
#endif
   Array<int> dof_map, dof2tk;

   Poly_1D::Basis &cbasis1d, &obasis1d;

private:
   void LocalInterpolation(const VectorFiniteElement &cfe,
                           ElementTransformation &Trans,
                           DenseMatrix &I) const;

public:
   /** @brief Construct the ND_R2D_SegmentElement of order @a p and closed and
       open BasisType @a cb_type and @a ob_type */
   ND_R2D_SegmentElement(const int p,
                         const int cb_type = BasisType::GaussLobatto,
                         const int ob_type = BasisType::GaussLegendre);

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override;

   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { LocalInterpolation(*this, Trans, I); }

   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override
   { MFEM_ABORT("method is not overloaded"); }

   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override
   { LocalInterpolation(CheckVectorFE(fe), Trans, I); }

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;
};

class ND_R2D_FiniteElement : public VectorFiniteElement
{
protected:
   const real_t *tk;
   Array<int> dof_map, dof2tk;

   ND_R2D_FiniteElement(int p, Geometry::Type G, int Do, const real_t *tk_fe);

private:
   void LocalInterpolation(const VectorFiniteElement &cfe,
                           ElementTransformation &Trans,
                           DenseMatrix &I) const;

public:
   using FiniteElement::CalcVShape;
   using FiniteElement::CalcPhysCurlShape;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override;

   void CalcPhysCurlShape(ElementTransformation &Trans,
                          DenseMatrix &curl_shape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { LocalInterpolation(*this, Trans, I); }

   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override;

   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override
   { LocalInterpolation(CheckVectorFE(fe), Trans, I); }

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;

   void Project(const FiniteElement &fe, ElementTransformation &Trans,
                DenseMatrix &I) const override;

   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override;
};

/// Arbitrary order Nedelec 3D elements in 2D on a triangle
class ND_R2D_TriangleElement : public ND_R2D_FiniteElement
{
private:
   static const real_t tk_t[15];

#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix nd_shape;
   mutable Vector      h1_shape;
   mutable DenseMatrix nd_dshape;
   mutable DenseMatrix h1_dshape;
#endif

   ND_TriangleElement ND_FE;
   H1_TriangleElement H1_FE;

public:
   /// Construct the ND_R2D_TriangleElement of order @a p
   ND_R2D_TriangleElement(const int p,
                          const int cb_type = BasisType::GaussLobatto);

   using ND_R2D_FiniteElement::CalcVShape;
   using ND_R2D_FiniteElement::CalcPhysCurlShape;

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;
   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;
};


/// Arbitrary order Nedelec 3D elements in 2D on a square
class ND_R2D_QuadrilateralElement : public ND_R2D_FiniteElement
{
   static const real_t tk_q[15];

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
   mutable Vector dshape_cx, dshape_cy;
#endif

   Poly_1D::Basis &cbasis1d, &obasis1d;

public:
   /** @brief Construct the ND_R2D_QuadrilateralElement of order @a p and
       closed and open BasisType @a cb_type and @a ob_type */
   ND_R2D_QuadrilateralElement(const int p,
                               const int cb_type = BasisType::GaussLobatto,
                               const int ob_type = BasisType::GaussLegendre);

   using ND_R2D_FiniteElement::CalcVShape;
   using ND_R2D_FiniteElement::CalcPhysCurlShape;

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;
   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;
};


} // namespace mfem

#endif
