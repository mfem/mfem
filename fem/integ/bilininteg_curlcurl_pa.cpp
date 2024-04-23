// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../qfunction.hpp"
#include "bilininteg_hcurl_kernels.hpp"

namespace mfem
{

void CurlCurlIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fel = fes.GetTypicalFE();

   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*el, *el,
                                                     *mesh->GetTypicalElementTransformation());

   const int dims = el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::SYMMETRIC);
   if (Q) { coeff.Project(*Q); }
   else if (MQ) { coeff.ProjectTranspose(*MQ); }
   else if (DQ) { coeff.Project(*DQ); }
   else { coeff.SetConstant(1.0); }

   const int coeff_dim = coeff.GetVDim();
   symmetric = (coeff_dim != dim*dim);
   const int sym_dims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int ndata = (dim == 2) ? 1 : (symmetric ? sym_dims : dim*dim);
   pa_data.SetSize(ndata * nq * ne, Device::GetMemoryType());

   if (el->GetDerivType() != mfem::FiniteElement::CURL)
   {
      MFEM_ABORT("Unknown kernel.");
   }

   if (dim == 3)
   {
      internal::PACurlCurlSetup3D(quad1D, coeff_dim, ne, ir->GetWeights(), geom->J,
                                  coeff, pa_data);
   }
   else
   {
      internal::PACurlCurlSetup2D(quad1D, ne, ir->GetWeights(), geom->J, coeff,
                                  pa_data);
   }
}

void CurlCurlIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if (dim == 3)
   {
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         const int ID = (dofs1D << 4) | quad1D;
         switch (ID)
         {
            case 0x23:
               return internal::SmemPACurlCurlAssembleDiagonal3D<2,3>(
                         dofs1D,
                         quad1D,
                         symmetric, ne,
                         mapsO->B, mapsC->B,
                         mapsO->G, mapsC->G,
                         pa_data, diag);
            case 0x34:
               return internal::SmemPACurlCurlAssembleDiagonal3D<3,4>(
                         dofs1D,
                         quad1D,
                         symmetric, ne,
                         mapsO->B, mapsC->B,
                         mapsO->G, mapsC->G,
                         pa_data, diag);
            case 0x45:
               return internal::SmemPACurlCurlAssembleDiagonal3D<4,5>(
                         dofs1D,
                         quad1D,
                         symmetric, ne,
                         mapsO->B, mapsC->B,
                         mapsO->G, mapsC->G,
                         pa_data, diag);
            case 0x56:
               return internal::SmemPACurlCurlAssembleDiagonal3D<5,6>(
                         dofs1D,
                         quad1D,
                         symmetric, ne,
                         mapsO->B, mapsC->B,
                         mapsO->G, mapsC->G,
                         pa_data, diag);
            default:
               return internal::SmemPACurlCurlAssembleDiagonal3D(
                         dofs1D, quad1D,
                         symmetric, ne,
                         mapsO->B, mapsC->B,
                         mapsO->G, mapsC->G,
                         pa_data, diag);
         }
      }
      else
      {
         internal::PACurlCurlAssembleDiagonal3D(dofs1D, quad1D, symmetric, ne,
                                                mapsO->B, mapsC->B,
                                                mapsO->G, mapsC->G,
                                                pa_data, diag);
      }
   }
   else if (dim == 2)
   {
      internal::PACurlCurlAssembleDiagonal2D(dofs1D, quad1D, ne,
                                             mapsO->B, mapsC->G, pa_data, diag);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void CurlCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         const int ID = (dofs1D << 4) | quad1D;
         switch (ID)
         {
            case 0x23:
               return internal::SmemPACurlCurlApply3D<2,3>(
                         dofs1D, quad1D,
                         symmetric, ne,
                         mapsO->B, mapsC->B, mapsO->Bt, mapsC->Bt,
                         mapsC->G, mapsC->Gt, pa_data, x, y);
            case 0x34:
               return internal::SmemPACurlCurlApply3D<3,4>(
                         dofs1D, quad1D,
                         symmetric, ne,
                         mapsO->B, mapsC->B, mapsO->Bt, mapsC->Bt,
                         mapsC->G, mapsC->Gt, pa_data, x, y);
            case 0x45:
               return internal::SmemPACurlCurlApply3D<4,5>(
                         dofs1D, quad1D,
                         symmetric, ne,
                         mapsO->B, mapsC->B, mapsO->Bt, mapsC->Bt,
                         mapsC->G, mapsC->Gt, pa_data, x, y);
            case 0x56:
               return internal::SmemPACurlCurlApply3D<5,6>(
                         dofs1D, quad1D,
                         symmetric, ne,
                         mapsO->B, mapsC->B, mapsO->Bt, mapsC->Bt,
                         mapsC->G, mapsC->Gt, pa_data, x, y);
            default:
               return internal::SmemPACurlCurlApply3D(
                         dofs1D, quad1D, symmetric, ne,
                         mapsO->B, mapsC->B, mapsO->Bt, mapsC->Bt,
                         mapsC->G, mapsC->Gt, pa_data, x, y);
         }
      }
      else
      {
         internal::PACurlCurlApply3D(dofs1D, quad1D, symmetric, ne, mapsO->B, mapsC->B,
                                     mapsO->Bt, mapsC->Bt, mapsC->G, mapsC->Gt,
                                     pa_data, x, y);
      }
   }
   else if (dim == 2)
   {
      internal::PACurlCurlApply2D(dofs1D, quad1D, ne, mapsO->B, mapsO->Bt,
                                  mapsC->G, mapsC->Gt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

} // namespace mfem
