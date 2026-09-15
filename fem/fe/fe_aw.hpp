// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#ifndef MFEM_FE_AW
#define MFEM_FE_AW

#include "fe_base.hpp"

namespace mfem
{

/** Lowest-order conforming Arnold--Winther triangle:
    symmetric cubic tensors whose divergence is linear. Its 24 DOFs are
    Cartesian tensor components (xx,xy,yy) at each vertex, four traction
    moments per edge (constant nn, nt, then linear nn, nt), and three cell
    moments of the double-Piola pullback. Physical evaluation and transfer
    require affine 2D transformations. */
class ArnoldWintherTriangleFiniteElement : public FiniteElement
{
private:
   // Rows contain the 24 nodal basis functions in symmetric P3 (30 columns).
   DenseMatrix basis;

   void GetBasisTransform(const DenseMatrix &J, DenseMatrix &A) const;

   void CalcShape(const IntegrationPoint &, Vector &) const override
   { MFEM_ABORT("Arnold-Winther shape functions are matrix-valued"); }
   void CalcDShape(const IntegrationPoint &, DenseMatrix &) const override
   { MFEM_ABORT("use CalcDivShape for the Arnold-Winther element"); }

public:
   ArnoldWintherTriangleFiniteElement();

   void CalcMShape(const IntegrationPoint &ip,
                   DenseTensor &shape) const override;
   void CalcMShape(ElementTransformation &Trans,
                   DenseTensor &shape) const override;

   void CalcDivShape(const IntegrationPoint &ip,
                     DenseMatrix &divshape) const override;
   void CalcPhysDivShape(ElementTransformation &Trans,
                         DenseMatrix &divshape) const override;

   /** Project a scalar H1 element with three byVDIM components representing
       (s00,s01,s11) using the canonical Arnold--Winther moments. */
   void Project(const FiniteElement &fe, ElementTransformation &Trans,
                DenseMatrix &I) const override;

   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override;

   bool RequiresPhysicalTransfer() const override { return true; }

   /// Convert reference-child interpolation to the physical moment bases.
   void GetPhysicalTransferMatrix(const DenseMatrix &reference_transfer,
                                  ElementTransformation &child,
                                  ElementTransformation &fine,
                                  DenseMatrix &I) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { GetTransferMatrix(*this, Trans, I); }

   void GetFaceDofs(int face, int **dofs, int *ndofs) const override;
};

} // namespace mfem

#endif
