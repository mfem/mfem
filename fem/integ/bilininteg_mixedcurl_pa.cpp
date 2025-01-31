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

#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "bilininteg_hcurl_kernels.hpp"
#include "bilininteg_hcurlhdiv_kernels.hpp"

namespace mfem
{

void MixedScalarCurlIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                           const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *fel = trial_fes.GetTypicalFE(); // In H(curl)
   const FiniteElement *eltest = test_fes.GetTypicalFE(); // In scalar space

   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");

   if (el->GetDerivType() != mfem::FiniteElement::CURL)
   {
      MFEM_ABORT("Unknown kernel.");
   }

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*eltest, *eltest,
                                                     *mesh->GetTypicalElementTransformation());

   const int dims = el->GetDim();
   MFEM_VERIFY(dims == 2, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2, "");

   ne = test_fes.GetNE();
   mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   if (el->GetOrder() == eltest->GetOrder())
   {
      dofs1Dtest = dofs1D;
   }
   else
   {
      dofs1Dtest = dofs1D - 1;
   }

   pa_data.SetSize(nq * ne, Device::GetMemoryType());

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::FULL);

   if (dim == 2)
   {
      internal::PAHcurlL2Setup2D(quad1D, ne, ir->GetWeights(), coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void MixedScalarCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 2)
   {
      internal::PAHcurlL2Apply2D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                                 mapsO->Bt, mapsC->Bt, mapsC->G, pa_data,
                                 x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void MixedScalarCurlIntegrator::AddMultTransposePA(const Vector &x,
                                                   Vector &y) const
{
   if (dim == 2)
   {
      internal::PAHcurlL2ApplyTranspose2D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                                          mapsO->Bt, mapsC->B, mapsC->Gt, pa_data,
                                          x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void MixedVectorCurlIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                           const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with vector test and trial spaces.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetTypicalFE();
   const FiniteElement *test_fel = test_fes.GetTypicalFE();

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetTypicalElementTransformation());
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &trial_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &trial_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   mapsCtest = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsOtest = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;
   dofs1Dtest = mapsCtest->ndof;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   testType = test_el->GetDerivType();
   trialType = trial_el->GetDerivType();

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   coeffDim = (DQ ? 3 : 1);

   const bool curlSpaces = (testType == mfem::FiniteElement::CURL &&
                            trialType == mfem::FiniteElement::CURL);

   const int ndata = curlSpaces ? (coeffDim == 1 ? 1 : 9) : symmDims;
   pa_data.SetSize(ndata * nq * ne, Device::GetMemoryType());

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::FULL);
   if (Q) { coeff.Project(*Q); }
   else if (DQ) { coeff.Project(*DQ); }
   else { coeff.SetConstant(1.0); }

   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      if (coeffDim == 1)
      {
         internal::PAHcurlL2Setup3D(nq, coeffDim, ne, ir->GetWeights(), coeff, pa_data);
      }
      else
      {
         internal::PAHcurlHdivMassSetup3D(quad1D, coeffDim, ne, false, ir->GetWeights(),
                                          geom->J, coeff, pa_data);
      }
   }
   else if (testType == mfem::FiniteElement::DIV &&
            trialType == mfem::FiniteElement::CURL && dim == 3 &&
            test_fel->GetOrder() == trial_fel->GetOrder())
   {
      internal::PACurlCurlSetup3D(quad1D, coeffDim, ne, ir->GetWeights(), geom->J,
                                  coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void MixedVectorCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      const int ndata = coeffDim == 1 ? 1 : 9;

      if (Device::Allows(Backend::DEVICE_MASK))
      {
         const int ID = (dofs1D << 4) | quad1D;
         switch (ID)
         {
            case 0x23:
               return internal::SmemPAHcurlL2Apply3D<2,3>(
                         dofs1D, quad1D, ndata, ne,
                         mapsO->B, mapsC->B, mapsC->G,
                         pa_data, x, y);
            case 0x34:
               return internal::SmemPAHcurlL2Apply3D<3,4>(
                         dofs1D, quad1D, ndata, ne,
                         mapsO->B, mapsC->B, mapsC->G,
                         pa_data, x, y);
            case 0x45:
               return internal::SmemPAHcurlL2Apply3D<4,5>(
                         dofs1D, quad1D, ndata, ne,
                         mapsO->B, mapsC->B, mapsC->G,
                         pa_data, x, y);
            case 0x56:
               return internal::SmemPAHcurlL2Apply3D<5,6>(
                         dofs1D, quad1D, ndata, ne,
                         mapsO->B, mapsC->B, mapsC->G,
                         pa_data, x, y);
            default:
               return internal::SmemPAHcurlL2Apply3D(
                         dofs1D, quad1D, ndata, ne,
                         mapsO->B, mapsC->B, mapsC->G,
                         pa_data, x, y);
         }
      }
      else
      {
         internal::PAHcurlL2Apply3D(dofs1D, quad1D, ndata, ne, mapsO->B, mapsC->B,
                                    mapsO->Bt, mapsC->Bt, mapsC->G, pa_data, x, y);
      }
   }
   else if (testType == mfem::FiniteElement::DIV &&
            trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      internal::PAHcurlHdivApply3D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                                   mapsC->B, mapsOtest->Bt, mapsCtest->Bt, mapsC->G,
                                   pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

void MixedVectorCurlIntegrator::AddMultTransposePA(const Vector &x,
                                                   Vector &y) const
{
   if (testType == mfem::FiniteElement::DIV &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      internal::PAHcurlHdivApplyTranspose3D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                                            mapsC->B, mapsOtest->Bt, mapsCtest->Bt,
                                            mapsC->Gt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

void MixedVectorWeakCurlIntegrator::AssemblePA(const FiniteElementSpace
                                               &trial_fes,
                                               const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with vector test and trial spaces.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetTypicalFE();
   const FiniteElement *test_fel = test_fes.GetTypicalFE();

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetTypicalElementTransformation());
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   testType = test_el->GetDerivType();
   trialType = trial_el->GetDerivType();

   const bool curlSpaces = (testType == mfem::FiniteElement::CURL &&
                            trialType == mfem::FiniteElement::CURL);

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6

   coeffDim = DQ ? 3 : 1;
   const int ndata = curlSpaces ? (DQ ? 9 : 1) : symmDims;

   pa_data.SetSize(ndata * nq * ne, Device::GetMemoryType());

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::FULL);
   if (Q) { coeff.Project(*Q); }
   else if (DQ) { coeff.Project(*DQ); }
   else { coeff.SetConstant(1.0); }

   if (trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      if (coeffDim == 1)
      {
         internal::PAHcurlL2Setup3D(nq, coeffDim, ne, ir->GetWeights(), coeff, pa_data);
      }
      else
      {
         internal::PAHcurlHdivMassSetup3D(quad1D, coeffDim, ne, false, ir->GetWeights(),
                                          geom->J, coeff, pa_data);
      }
   }
   else if (trialType == mfem::FiniteElement::DIV && dim == 3 &&
            test_el->GetOrder() == trial_el->GetOrder())
   {
      internal::PACurlCurlSetup3D(quad1D, coeffDim, ne, ir->GetWeights(), geom->J,
                                  coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void MixedVectorWeakCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      const int ndata = coeffDim == 1 ? 1 : 9;
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         const int ID = (dofs1D << 4) | quad1D;
         switch (ID)
         {
            case 0x23:
               return internal::SmemPAHcurlL2ApplyTranspose3D<2,3>(
                         dofs1D, quad1D, ndata,
                         ne, mapsO->B, mapsC->B,
                         mapsC->G, pa_data, x, y);
            case 0x34:
               return internal::SmemPAHcurlL2ApplyTranspose3D<3,4>(
                         dofs1D, quad1D, ndata,
                         ne, mapsO->B, mapsC->B,
                         mapsC->G, pa_data, x, y);
            case 0x45:
               return internal::SmemPAHcurlL2ApplyTranspose3D<4,5>(
                         dofs1D, quad1D, ndata,
                         ne, mapsO->B, mapsC->B,
                         mapsC->G, pa_data, x, y);
            case 0x56:
               return internal::SmemPAHcurlL2ApplyTranspose3D<5,6>(
                         dofs1D, quad1D, ndata,
                         ne, mapsO->B, mapsC->B,
                         mapsC->G, pa_data, x, y);
            default:
               return internal::SmemPAHcurlL2ApplyTranspose3D(
                         dofs1D, quad1D, ndata, ne,
                         mapsO->B, mapsC->B,
                         mapsC->G, pa_data, x, y);
         }
      }
      else
      {
         internal::PAHcurlL2ApplyTranspose3D(dofs1D, quad1D, ndata, ne, mapsO->B,
                                             mapsC->B, mapsO->Bt, mapsC->Bt, mapsC->Gt,
                                             pa_data, x, y);
      }
   }
   else if (testType == mfem::FiniteElement::CURL &&
            trialType == mfem::FiniteElement::DIV && dim == 3)
   {
      internal::PAHcurlHdivApplyTranspose3D(dofs1D, dofs1D, quad1D, ne, mapsO->B,
                                            mapsC->B, mapsO->Bt, mapsC->Bt,
                                            mapsC->Gt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

void MixedVectorWeakCurlIntegrator::AddMultTransposePA(const Vector &x,
                                                       Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::DIV && dim == 3)
   {
      internal::PAHcurlHdivApply3D(dofs1D, dofs1D, quad1D, ne, mapsO->B,
                                   mapsC->B, mapsO->Bt, mapsC->Bt, mapsC->G,
                                   pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

} // namespace mfem
