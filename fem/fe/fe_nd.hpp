// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

/// Arbitrary order Nedelec elements in 3D on a cube
class ND_HexahedronElement : public VectorTensorFiniteElement
{
   static const double tk[18];
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy, shape_cz, shape_oz;
   mutable Vector dshape_cx, dshape_cy, dshape_cz;
#endif
   Array<int> dof2tk;
   const double *cp;

public:
   /** @brief Construct the ND_HexahedronElement of order @a p and closed and
       open BasisType @a cb_type and @a ob_type */
   ND_HexahedronElement(const int p,
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
   {
      if (obasis1d.IsIntegratedType()) { ProjectIntegrated(vc, Trans, dofs); }
      else { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   }

   virtual void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                                 Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }

   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }

   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
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

protected:
   void ProjectIntegrated(VectorCoefficient &vc,
                          ElementTransformation &Trans,
                          Vector &dofs) const;
};


/// Arbitrary order Nedelec elements in 2D on a square
class ND_QuadrilateralElement : public VectorTensorFiniteElement
{
   static const double tk[8];

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
   mutable Vector dshape_cx, dshape_cy;
#endif
   Array<int> dof2tk;
   const double *cp;

public:
   /** @brief Construct the ND_QuadrilateralElement of order @a p and closed and
       open BasisType @a cb_type and @a ob_type */
   ND_QuadrilateralElement(const int p,
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
   {
      if (obasis1d.IsIntegratedType()) { ProjectIntegrated(vc, Trans, dofs); }
      else { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   }
   virtual void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                                 Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }

protected:
   void ProjectIntegrated(VectorCoefficient &vc,
                          ElementTransformation &Trans,
                          Vector &dofs) const;
};


/// Arbitrary order Nedelec elements in 3D on a tetrahedron
class ND_TetrahedronElement : public VectorFiniteElement
{
   static const double tk[18], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l;
   mutable DenseMatrix u;
#endif
   Array<int> dof2tk;
   DenseMatrixInverse Ti;

public:
   /// Construct the ND_TetrahedronElement of order @a p
   ND_TetrahedronElement(const int p);
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
   virtual void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                                 Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
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
};

/// Arbitrary order Nedelec elements in 2D on a triangle
class ND_TriangleElement : public VectorFiniteElement
{
   static const double tk[8], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_l;
   mutable DenseMatrix u;
   mutable Vector curlu;
#endif
   Array<int> dof2tk;
   DenseMatrixInverse Ti;

public:
   /// Construct the ND_TriangleElement of order @a p
   ND_TriangleElement(const int p);
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
   virtual void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                                 Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }
};


/// Arbitrary order Nedelec elements in 1D on a segment
class ND_SegmentElement : public VectorTensorFiniteElement
{
   static const double tk[1];

   Array<int> dof2tk;

public:
   /** @brief Construct the ND_SegmentElement of order @a p and open
       BasisType @a ob_type */
   ND_SegmentElement(const int p, const int ob_type = BasisType::GaussLegendre);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const
   { obasis1d.Eval(ip.x, shape); }
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   // virtual void CalcCurlShape(const IntegrationPoint &ip,
   //                            DenseMatrix &curl_shape) const;
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
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }
};

} // namespace mfem

#endif
