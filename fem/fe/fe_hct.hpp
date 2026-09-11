// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#ifndef MFEM_FE_HCT
#define MFEM_FE_HCT

#include "fe_base.hpp"

namespace mfem
{

/** The cubic Hsieh--Clough--Tocher macroelement on a triangle.

    The element is piecewise cubic on the barycentric (Clough--Tocher) split.
    Its twelve degrees of freedom are the value and two physical Cartesian
    derivatives at each vertex, followed by one oriented, unnormalized normal
    derivative at the midpoint of each edge. The physical basis implementation
    assumes an affine element transformation. */
class HCTTriangleFiniteElement : public FiniteElement
{
private:
   // Rows are the twelve nodal basis functions; columns are the thirty
   // monomials of the broken P3 space (ten on each subtriangle).
   DenseMatrix basis;

   static int GetSubTriangle(const IntegrationPoint &ip);
   static void CalcRawShape(const IntegrationPoint &ip, Vector &raw);
   static void CalcRawDShape(const IntegrationPoint &ip, DenseMatrix &raw);
   static void CalcRawHessian(const IntegrationPoint &ip, DenseMatrix &raw);
   void GetPhysicalDofMatrix(ElementTransformation &Trans,
                             DenseMatrix &M) const;

public:
   HCTTriangleFiniteElement();

   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &hessian) const override;

   void CalcPhysShape(ElementTransformation &Trans,
                      Vector &shape) const override;
   void CalcPhysDShape(ElementTransformation &Trans,
                       DenseMatrix &dshape) const override;
   void CalcPhysHessian(ElementTransformation &Trans,
                        DenseMatrix &hessian) const override;
};

} // namespace mfem

#endif
