// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.


#ifndef MFEM_FE_HZZZ
#define MFEM_FE_HZZZ

#include "fe_aw.hpp"

namespace mfem
{

/** The 21-DOF Huang--Zhang--Zhou--Zhu stress triangle from Section 2 of
    https://arxiv.org/abs/2310.13920. Symmetric cubics have linear divergence
    and quadratic tangential-normal traction on every physical edge.

    DOFs: Cartesian (xx,xy,yy) values at the vertices; constant nn, constant
    nt, and linear nn moments on each edge; three constant cell moments of
    the double-Piola pullback, as in AW. Requires affine 2D maps. The physical
    space is geometry dependent and refinement uses canonical interpolation,
    not an inclusion of nested spaces. Its Airy potential is the Bell element. */
class HuangZhangZhouZhuTriangleFiniteElement : public FiniteElement
{
private:
   ArnoldWintherTriangleFiniteElement aw;
   DenseMatrix reference_embedding;
   void GetEmbedding(const DenseMatrix &J, DenseMatrix &E) const;
   void ReduceShape(const DenseMatrix &E, const DenseTensor &full,
                    DenseTensor &shape) const;
   void SelectDofs(const DenseMatrix &full, DenseMatrix &I) const;
   void CalcShape(const IntegrationPoint &, Vector &) const override
   { MFEM_ABORT("HZZZ shape functions are matrix-valued"); }
   void CalcDShape(const IntegrationPoint &, DenseMatrix &) const override
   { MFEM_ABORT("use CalcDivShape for the HZZZ element"); }

public:
   HuangZhangZhouZhuTriangleFiniteElement();
   void CalcMShape(const IntegrationPoint &ip, DenseTensor &shape) const override;
   void CalcMShape(ElementTransformation &T, DenseTensor &shape) const override;
   void CalcDivShape(const IntegrationPoint &ip,
                     DenseMatrix &shape) const override;
   void CalcPhysDivShape(ElementTransformation &T,
                         DenseMatrix &shape) const override;
   void Project(const FiniteElement &fe, ElementTransformation &T,
                DenseMatrix &I) const override;
   void GetTransferMatrix(const FiniteElement &fe, ElementTransformation &T,
                          DenseMatrix &I) const override;
   void GetLocalInterpolation(ElementTransformation &T,
                              DenseMatrix &I) const override
   { GetTransferMatrix(*this,T,I); }
   bool RequiresPhysicalTransfer() const override { return true; }
   void GetPhysicalTransferMatrix(const DenseMatrix &reference_transfer,
                                  ElementTransformation &child,
                                  ElementTransformation &fine,
                                  DenseMatrix &I) const override;
   void GetFaceDofs(int face, int **dofs, int *ndofs) const override;
};

} // namespace mfem

#endif
