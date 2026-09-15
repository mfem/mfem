// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.


#ifndef MFEM_FE_BELL
#define MFEM_FE_BELL

#include "fe_argyris.hpp"

namespace mfem
{

/** Bell triangle: quintics with cubic normal derivative on every physical
    edge. The 18 DOFs are value, Cartesian gradient, and Hessian (xx,xy,yy)
    at each vertex. Requires affine 2D maps, but the polynomial subspace is
    geometry dependent. Refinement uses nodal interpolation, since the
    spaces on successive meshes need not be nested. */
class BellTriangleFiniteElement : public FiniteElement
{
private:
   ArgyrisTriangleFiniteElement argyris;
   DenseMatrix reference_embedding;
   void GetEmbedding(const DenseMatrix &J, DenseMatrix &E) const;
   void Interpolate(ElementTransformation &child,
                    ElementTransformation &coarse, DenseMatrix &I) const;

public:
   BellTriangleFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip, DenseMatrix &shape) const override;
   void CalcHessian(const IntegrationPoint &ip, DenseMatrix &shape) const override;
   void CalcPhysShape(ElementTransformation &T, Vector &shape) const override;
   void CalcPhysDShape(ElementTransformation &T,
                       DenseMatrix &shape) const override;
   void CalcPhysHessian(ElementTransformation &T,
                        DenseMatrix &shape) const override;
   void GetTransferMatrix(const FiniteElement &fe, ElementTransformation &T,
                          DenseMatrix &I) const override;
   void GetLocalInterpolation(ElementTransformation &T,
                              DenseMatrix &I) const override
   { GetTransferMatrix(*this, T, I); }
   bool RequiresPhysicalTransfer() const override { return true; }
   void GetPhysicalTransferMatrix(const DenseMatrix &reference_transfer,
                                  ElementTransformation &child,
                                  ElementTransformation &fine,
                                  DenseMatrix &I) const override;
   void GetFaceDofs(int face, int **dofs, int *ndofs) const override;
};

} // namespace mfem

#endif
