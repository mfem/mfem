// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#ifndef MFEM_FE_ARGYRIS
#define MFEM_FE_ARGYRIS

#include "fe_base.hpp"

namespace mfem
{

/** Quintic Argyris triangle with 21 degrees of freedom: value, Cartesian
    first derivatives and second derivatives (xx,xy,yy) at each vertex,
    followed by the oriented, unnormalized normal derivative at each edge
    midpoint. Physical shape evaluation requires an affine 2D map. */
class ArgyrisTriangleFiniteElement : public FiniteElement
{
private:
   DenseMatrix basis;
   void GetPhysicalDofMatrix(ElementTransformation &Trans, DenseMatrix &M) const;

public:
   ArgyrisTriangleFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip, DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &hessian) const override;
   void CalcPhysShape(ElementTransformation &Trans, Vector &shape) const override;
   void CalcPhysDShape(ElementTransformation &Trans,
                       DenseMatrix &dshape) const override;
   void CalcPhysHessian(ElementTransformation &Trans,
                        DenseMatrix &hessian) const override;
};

} // namespace mfem

#endif
