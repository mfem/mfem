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

#ifndef MFEM_FE_RT
#define MFEM_FE_RT

#include "fe_base.hpp"

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

} // namespace mfem

#endif
