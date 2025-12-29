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

void DivDivIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fel = fes.GetTypicalFE();

   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir = IntRule ? IntRule : &MassIntegrator::GetRule
                               (*el, *el, *mesh->GetTypicalElementTransformation());

   const int dims = el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   pa_data.SetSize(nq * ne, Device::GetMemoryType());

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::FULL);

   if (el->GetDerivType() == mfem::FiniteElement::DIV && dim == 3)
   {
      internal::PADivDivSetup3D(quad1D, ne, ir->GetWeights(), geom->J, coeff,
                                pa_data);
   }
   else if (el->GetDerivType() == mfem::FiniteElement::DIV && dim == 2)
   {
      internal::PADivDivSetup2D(quad1D, ne, ir->GetWeights(), geom->J, coeff,
                                pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void DivDivIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if (dim == 3)
   {
      internal::PADivDivAssembleDiagonal3D(dofs1D, quad1D, ne,
                                           mapsO->B, mapsC->G, pa_data, diag);
   }
   else
   {
      internal::PADivDivAssembleDiagonal2D(dofs1D, quad1D, ne,
                                           mapsO->B, mapsC->G, pa_data, diag);
   }
}

void DivDivIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
      internal::PADivDivApply3D(dofs1D, quad1D, ne, mapsO->B, mapsC->G,
                                mapsO->Bt, mapsC->Gt, pa_data, x, y);
   else if (dim == 2)
      internal::PADivDivApply2D(dofs1D, quad1D, ne, mapsO->B, mapsC->G,
                                mapsO->Bt, mapsC->Gt, pa_data, x, y);
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

} // namespace mfem
