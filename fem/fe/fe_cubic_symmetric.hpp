// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#ifndef MFEM_FE_CUBIC_SYMMETRIC
#define MFEM_FE_CUBIC_SYMMETRIC

#include "fe_base.hpp"

namespace mfem
{

/** Shared basis, mapping, and interpolation for cubic symmetric stress
    triangles. Both variants use Cartesian vertex values and constant/linear
    nn and nt edge moments. Cell DOFs are moments of the double-Piola pullback
    against constant tensors (AW) or linear tensors (Hu--Zhang).

    Physical evaluation and transfer require affine 2D transformations. */
class CubicSymmetricTriangleFiniteElement : public FiniteElement
{
private:
   // Rows contain the nodal basis functions in symmetric P3 (30 columns).
   const int cell_modes;
   DenseMatrix basis;

   void GetBasisTransform(const DenseMatrix &J, DenseMatrix &A) const;

   void CalcShape(const IntegrationPoint &, Vector &) const override
   { MFEM_ABORT("Cubic symmetric stress shape functions are matrix-valued"); }
   void CalcDShape(const IntegrationPoint &, DenseMatrix &) const override
   { MFEM_ABORT("use CalcDivShape for the Cubic symmetric stress element"); }

protected:
   /// Restrict divergence to P1 for AW; otherwise use all of symmetric P3.
   explicit CubicSymmetricTriangleFiniteElement(bool linear_divergence);

public:

   void CalcMShape(const IntegrationPoint &ip,
                   DenseTensor &shape) const override;
   void CalcMShape(ElementTransformation &Trans,
                   DenseTensor &shape) const override;

   void CalcDivShape(const IntegrationPoint &ip,
                     DenseMatrix &divshape) const override;
   void CalcPhysDivShape(ElementTransformation &Trans,
                         DenseMatrix &divshape) const override;

   /** Project a scalar H1 element with three byVDIM components representing
       (s00,s01,s11) using the canonical symmetric stress moments. */
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
