// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#ifndef MFEM_FE_JM
#define MFEM_FE_JM

#include "fe_base.hpp"

namespace mfem
{

/** The lowest-order, two-dimensional Johnson--Mercier element.

    The element consists of symmetric, piecewise-linear matrix fields on the
    Alfeld split of a triangle. Its 15 degrees of freedom are four traction
    moments on each edge and three element moments. Matrix fields are mapped
    using the double contravariant Piola transformation. The physical
    divergence implementation assumes an affine element transformation. */
class JohnsonMercierTriangleFiniteElement : public FiniteElement
{
private:
   // Rows contain the coefficients of the 15 nodal basis functions in the
   // 27-dimensional broken, symmetric P1 basis on the Alfeld split.
   DenseMatrix basis;

   static int GetSubTriangle(const IntegrationPoint &ip);
   static void CalcRawShape(const IntegrationPoint &ip, Vector &raw);
   static void CalcRawDivShape(const IntegrationPoint &ip,
                               DenseMatrix &raw_div);
   void GetFacetTransform(const DenseMatrix &J, DenseMatrix &A) const;

   void CalcShape(const IntegrationPoint &, Vector &) const override
   { MFEM_ABORT("Johnson-Mercier shape functions are matrix-valued"); }
   void CalcDShape(const IntegrationPoint &, DenseMatrix &) const override
   { MFEM_ABORT("use CalcDivShape for the Johnson-Mercier element"); }

public:
   JohnsonMercierTriangleFiniteElement();

   void CalcMShape(const IntegrationPoint &ip,
                   DenseTensor &shape) const override;
   void CalcMShape(ElementTransformation &Trans,
                   DenseTensor &shape) const override;

   void CalcDivShape(const IntegrationPoint &ip,
                     DenseMatrix &divshape) const override;
   void CalcPhysDivShape(ElementTransformation &Trans,
                         DenseMatrix &divshape) const override;

   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override;

   /// Convert reference-child interpolation to the physical moment bases.
   void GetPhysicalTransferMatrix(const DenseMatrix &reference_transfer,
                                  ElementTransformation &child,
                                  ElementTransformation &fine,
                                  DenseMatrix &I) const;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { GetTransferMatrix(*this, Trans, I); }

   void GetFaceDofs(int face, int **dofs, int *ndofs) const override;
};

} // namespace mfem

#endif
