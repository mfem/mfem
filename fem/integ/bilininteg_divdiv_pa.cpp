// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include "../ceed/integrators/divdiv/divdiv.hpp"
#include "bilininteg_hdiv_kernels.hpp"

namespace mfem
{

void DivDivIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      ceedOp = new ceed::PADivDivIntegrator(*this, fes, Q);
      return;
   }

   // Assumes tensor-product elements
   const FiniteElement *fel = fes.GetFE(0);
   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");
   ElementTransformation &T = *mesh->GetElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(*el, T);
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

void DivDivIntegrator::AssemblePABoundary(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNBE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      ceedOp = new ceed::PADivDivIntegrator(*this, fes, Q, true);
      return;
   }

   // Assumes tensor-product elements
   // const FiniteElement &el = *fes.GetBE(0);
   // ElementTransformation &T = *mesh->GetBdrElementTransformation(0);
   // const IntegrationRule *ir = IntRule ? IntRule : &GetRule(*el, T);
   MFEM_ABORT("Error: DivDivIntegrator::AssemblePABoundary only implemented with"
              " libCEED");
}

void DivDivIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else
   {
      if (dim == 3)
      {
         internal::PADivDivAssembleDiagonal3D(dofs1D, quad1D, ne,
                                              mapsO->B, mapsC->G, pa_data, diag);
      }
      else if (dim == 2)
      {
         internal::PADivDivAssembleDiagonal2D(dofs1D, quad1D, ne,
                                              mapsO->B, mapsC->G, pa_data, diag);
      }
      else
      {
         MFEM_ABORT("Unsupported dimension!");
      }
   }
}

void DivDivIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      if (dim == 3)
      {
         internal::PADivDivApply3D(dofs1D, quad1D, ne, mapsO->B, mapsC->G,
                                   mapsO->Bt, mapsC->Gt, pa_data, x, y);
      }
      else if (dim == 2)
      {
         internal::PADivDivApply2D(dofs1D, quad1D, ne, mapsO->B, mapsC->G,
                                   mapsO->Bt, mapsC->Gt, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT("Unsupported dimension!");
      }
   }
}

} // namespace mfem
