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

#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "bilininteg_hdiv_kernels.hpp"

namespace mfem
{

void
VectorFEDivergenceIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                         const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with a vector test space and
   // scalar trial space.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetTypicalFE();
   const FiniteElement *test_fel = test_fes.GetTypicalFE();

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const NodalTensorFiniteElement *test_el =
      dynamic_cast<const NodalTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only NodalTensorFiniteElement is supported!");

   const IntegrationRule *ir = IntRule ? IntRule : &MassIntegrator::GetRule(
                                  *trial_el, *trial_el,
                                  *mesh->GetTypicalElementTransformation());

   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder() + 1, "");

   ne = trial_fes.GetNE();
   mapsC = &trial_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &trial_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   L2mapsO = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   L2dofs1D = L2mapsO->ndof;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");
   if (dim == 2)
   {
      MFEM_VERIFY(nq == quad1D * quad1D, "");
   }
   else
   {
      MFEM_VERIFY(nq == quad1D * quad1D * quad1D, "");
   }

   pa_data.SetSize(nq * ne, Device::GetMemoryType());

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::FULL);

   if (test_el->GetMapType() == FiniteElement::INTEGRAL)
   {
      const GeometricFactors *geom =
         mesh->GetGeometricFactors(*ir, GeometricFactors::DETERMINANTS);
      coeff /= geom->detJ;
   }

   if (trial_el->GetDerivType() == mfem::FiniteElement::DIV && dim == 3)
   {
      internal::PAHdivL2Setup3D(quad1D, ne, ir->GetWeights(), coeff, pa_data);
   }
   else if (trial_el->GetDerivType() == mfem::FiniteElement::DIV && dim == 2)
   {
      internal::PAHdivL2Setup2D(quad1D, ne, ir->GetWeights(), coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void VectorFEDivergenceIntegrator::AssembleDiagonalPA_ADAt(const Vector &D,
                                                           Vector &diag)
{
   if (dim == 3)
   {
      internal::PAHdivL2AssembleDiagonal_ADAt_3D(dofs1D, quad1D, L2dofs1D, ne,
                                                 L2mapsO->B,
                                                 mapsC->Gt, mapsO->Bt, pa_data, D, diag);
   }
   else if (dim == 2)
   {
      internal::PAHdivL2AssembleDiagonal_ADAt_2D(dofs1D, quad1D, L2dofs1D, ne,
                                                 L2mapsO->B,
                                                 mapsC->Gt, mapsO->Bt, pa_data, D, diag);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void VectorFEDivergenceIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      internal::PAHdivL2Apply3D(dofs1D, quad1D, L2dofs1D, ne, mapsO->B, mapsC->G,
                                L2mapsO->Bt, pa_data, x, y);
   }
   else if (dim == 2)
   {
      internal::PAHdivL2Apply2D(dofs1D, quad1D, L2dofs1D, ne, mapsO->B, mapsC->G,
                                L2mapsO->Bt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void VectorFEDivergenceIntegrator::AddMultTransposePA(const Vector &x,
                                                      Vector &y) const
{
   if (dim == 3)
   {
      internal::PAHdivL2ApplyTranspose3D(dofs1D, quad1D, L2dofs1D, ne, L2mapsO->B,
                                         mapsC->Gt, mapsO->Bt, pa_data, x, y);
   }
   else if (dim == 2)
   {
      internal::PAHdivL2ApplyTranspose2D(dofs1D, quad1D, L2dofs1D, ne, L2mapsO->B,
                                         mapsC->Gt, mapsO->Bt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

} // namespace mfem
