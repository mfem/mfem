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
#include "bilininteg_diffusion_kernels.hpp"
#include "bilininteg_hcurl_kernels.hpp"
#include "bilininteg_hdiv_kernels.hpp"
#include "bilininteg_hcurlhdiv_kernels.hpp"

namespace mfem
{

void VectorFEMassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   AssemblePA(fes, fes);
}

void VectorFEMassIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                        const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = trial_fes.GetMesh();

   const FiniteElement *trial_fel = trial_fes.GetTypicalFE();
   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const FiniteElement *test_fel = test_fes.GetTypicalFE();
   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetTypicalElementTransformation());
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   ne = trial_fes.GetNE();
   MFEM_VERIFY(ne == test_fes.GetNE(),
               "Different meshes for test and trial spaces");
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &trial_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &trial_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   mapsCtest = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsOtest = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1Dtest = mapsCtest->ndof;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   trial_fetype = trial_el->GetDerivType();
   test_fetype = test_el->GetDerivType();

   const bool trial_curl = (trial_fetype == mfem::FiniteElement::CURL);
   const bool trial_div = (trial_fetype == mfem::FiniteElement::DIV);
   const bool test_curl = (test_fetype == mfem::FiniteElement::CURL);
   const bool test_div = (test_fetype == mfem::FiniteElement::DIV);

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::SYMMETRIC);
   if (Q) { coeff.Project(*Q); }
   else if (MQ) { coeff.ProjectTranspose(*MQ); }
   else if (DQ) { coeff.Project(*DQ); }
   else { coeff.SetConstant(1.0); }

   const int coeff_dim = coeff.GetVDim();
   symmetric = (coeff_dim != dim*dim);

   if ((trial_curl && test_div) || (trial_div && test_curl))
   {
      pa_data.SetSize((coeff_dim == 1 ? 1 : dim*dim) * nq * ne,
                      Device::GetMemoryType());
   }
   else
   {
      pa_data.SetSize((symmetric ? symmDims : dims*dims) * nq * ne,
                      Device::GetMemoryType());
   }
   if (trial_curl && test_curl && dim == 3)
   {
      internal::PADiffusionSetup3D(quad1D, coeff_dim, ne, ir->GetWeights(), geom->J,
                                   coeff, pa_data);
   }
   else if (trial_curl && test_curl && dim == 2)
   {
      internal::PADiffusionSetup2D<2>(quad1D, coeff_dim, ne, ir->GetWeights(),
                                      geom->J, coeff, pa_data);
   }
   else if (trial_div && test_div && dim == 3)
   {
      internal::PAHdivMassSetup3D(quad1D, coeff_dim, ne, ir->GetWeights(), geom->J,
                                  coeff, pa_data);
   }
   else if (trial_div && test_div && dim == 2)
   {
      internal::PAHdivMassSetup2D(quad1D, coeff_dim, ne, ir->GetWeights(), geom->J,
                                  coeff, pa_data);
   }
   else if (((trial_curl && test_div) || (trial_div && test_curl)) &&
            test_fel->GetOrder() == trial_fel->GetOrder())
   {
      if (coeff_dim == 1)
      {
         internal::PAHcurlL2Setup3D(nq, coeff_dim, ne, ir->GetWeights(), coeff, pa_data);
      }
      else
      {
         const bool tr = (trial_div && test_curl);
         if (dim == 3)
         {
            internal::PAHcurlHdivMassSetup3D(quad1D, coeff_dim, ne, tr, ir->GetWeights(),
                                             geom->J, coeff, pa_data);
         }
         else
         {
            internal::PAHcurlHdivMassSetup2D(quad1D, coeff_dim, ne, tr, ir->GetWeights(),
                                             geom->J, coeff, pa_data);
         }
      }
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void VectorFEMassIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if (dim == 3)
   {
      if (trial_fetype == mfem::FiniteElement::CURL && test_fetype == trial_fetype)
      {
         if (Device::Allows(Backend::DEVICE_MASK))
         {
            const int ID = (dofs1D << 4) | quad1D;
            switch (ID)
            {
               case 0x23:
                  return internal::SmemPAHcurlMassAssembleDiagonal3D<2,3>(
                            dofs1D, quad1D, ne, symmetric,
                            mapsO->B, mapsC->B, pa_data, diag);
               case 0x34:
                  return internal::SmemPAHcurlMassAssembleDiagonal3D<3,4>(
                            dofs1D, quad1D, ne, symmetric,
                            mapsO->B, mapsC->B, pa_data, diag);
               case 0x45:
                  return internal::SmemPAHcurlMassAssembleDiagonal3D<4,5>(
                            dofs1D, quad1D, ne, symmetric,
                            mapsO->B, mapsC->B, pa_data, diag);
               case 0x56:
                  return internal::SmemPAHcurlMassAssembleDiagonal3D<5,6>(
                            dofs1D, quad1D, ne, symmetric,
                            mapsO->B, mapsC->B, pa_data, diag);
               default:
                  return internal::SmemPAHcurlMassAssembleDiagonal3D(
                            dofs1D, quad1D, ne, symmetric,
                            mapsO->B, mapsC->B, pa_data, diag);
            }
         }
         else
         {
            internal::PAHcurlMassAssembleDiagonal3D(dofs1D, quad1D, ne, symmetric,
                                                    mapsO->B, mapsC->B, pa_data, diag);
         }
      }
      else if (trial_fetype == mfem::FiniteElement::DIV &&
               test_fetype == trial_fetype)
      {
         internal::PAHdivMassAssembleDiagonal3D(dofs1D, quad1D, ne, symmetric,
                                                mapsO->B, mapsC->B, pa_data, diag);
      }
      else
      {
         MFEM_ABORT("Unknown kernel.");
      }
   }
   else // 2D
   {
      if (trial_fetype == mfem::FiniteElement::CURL && test_fetype == trial_fetype)
      {
         internal::PAHcurlMassAssembleDiagonal2D(dofs1D, quad1D, ne, symmetric,
                                                 mapsO->B, mapsC->B, pa_data, diag);
      }
      else if (trial_fetype == mfem::FiniteElement::DIV &&
               test_fetype == trial_fetype)
      {
         internal::PAHdivMassAssembleDiagonal2D(dofs1D, quad1D, ne, symmetric,
                                                mapsO->B, mapsC->B, pa_data, diag);
      }
      else
      {
         MFEM_ABORT("Unknown kernel.");
      }
   }
}

void VectorFEMassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   const bool trial_curl = (trial_fetype == mfem::FiniteElement::CURL);
   const bool trial_div = (trial_fetype == mfem::FiniteElement::DIV);
   const bool test_curl = (test_fetype == mfem::FiniteElement::CURL);
   const bool test_div = (test_fetype == mfem::FiniteElement::DIV);

   if (dim == 3)
   {
      if (trial_curl && test_curl)
      {
         if (Device::Allows(Backend::DEVICE_MASK))
         {
            const int ID = (dofs1D << 4) | quad1D;
            switch (ID)
            {
               case 0x23:
                  return internal::SmemPAHcurlMassApply3D<2,3>(
                            dofs1D, quad1D, ne, symmetric,
                            mapsO->B, mapsC->B, mapsO->Bt,
                            mapsC->Bt, pa_data, x, y);
               case 0x34:
                  return internal::SmemPAHcurlMassApply3D<3,4>(
                            dofs1D, quad1D, ne, symmetric,
                            mapsO->B, mapsC->B, mapsO->Bt,
                            mapsC->Bt, pa_data, x, y);
               case 0x45:
                  return internal::SmemPAHcurlMassApply3D<4,5>(
                            dofs1D, quad1D, ne, symmetric,
                            mapsO->B, mapsC->B, mapsO->Bt,
                            mapsC->Bt, pa_data, x, y);
               case 0x56:
                  return internal::SmemPAHcurlMassApply3D<5,6>(
                            dofs1D, quad1D, ne, symmetric,
                            mapsO->B, mapsC->B, mapsO->Bt,
                            mapsC->Bt, pa_data, x, y);
               default:
                  return internal::SmemPAHcurlMassApply3D(
                            dofs1D, quad1D, ne, symmetric,
                            mapsO->B, mapsC->B, mapsO->Bt,
                            mapsC->Bt, pa_data, x, y);
            }
         }
         else
         {
            internal::PAHcurlMassApply3D(dofs1D, quad1D, ne, symmetric, mapsO->B, mapsC->B,
                                         mapsO->Bt, mapsC->Bt, pa_data, x, y);
         }
      }
      else if (trial_div && test_div)
      {
         internal::PAHdivMassApply(3, dofs1D, quad1D, ne, symmetric, mapsO->B, mapsC->B,
                                   mapsO->Bt, mapsC->Bt, pa_data, x, y);
      }
      else if (trial_curl && test_div)
      {
         const bool scalarCoeff = !(DQ || MQ);
         internal::PAHcurlHdivMassApply3D(dofs1D, dofs1Dtest, quad1D, ne, scalarCoeff,
                                          true, false, mapsO->B, mapsC->B, mapsOtest->Bt,
                                          mapsCtest->Bt, pa_data, x, y);
      }
      else if (trial_div && test_curl)
      {
         const bool scalarCoeff = !(DQ || MQ);
         internal::PAHcurlHdivMassApply3D(dofs1D, dofs1Dtest, quad1D, ne, scalarCoeff,
                                          false, false, mapsO->B, mapsC->B, mapsOtest->Bt,
                                          mapsCtest->Bt, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT("Unknown kernel.");
      }
   }
   else // 2D
   {
      if (trial_curl && test_curl)
      {
         internal::PAHcurlMassApply2D(dofs1D, quad1D, ne, symmetric, mapsO->B, mapsC->B,
                                      mapsO->Bt, mapsC->Bt, pa_data, x, y);
      }
      else if (trial_div && test_div)
      {
         internal::PAHdivMassApply(2, dofs1D, quad1D, ne, symmetric, mapsO->B, mapsC->B,
                                   mapsO->Bt,
                                   mapsC->Bt, pa_data, x, y);
      }
      else if ((trial_curl && test_div) || (trial_div && test_curl))
      {
         const bool scalarCoeff = !(DQ || MQ);
         internal::PAHcurlHdivMassApply2D(dofs1D, dofs1Dtest, quad1D, ne, scalarCoeff,
                                          trial_curl, false, mapsO->B, mapsC->B,
                                          mapsOtest->Bt, mapsCtest->Bt, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT("Unknown kernel.");
      }
   }
}

void VectorFEMassIntegrator::AddMultTransposePA(const Vector &x,
                                                Vector &y) const
{
   const bool trial_curl = (trial_fetype == mfem::FiniteElement::CURL);
   const bool trial_div = (trial_fetype == mfem::FiniteElement::DIV);
   const bool test_curl = (test_fetype == mfem::FiniteElement::CURL);
   const bool test_div = (test_fetype == mfem::FiniteElement::DIV);

   bool symmetricSpaces = true;
   if (dim == 3 && ((trial_div && test_curl) || (trial_curl && test_div)))
   {
      const bool scalarCoeff = !(DQ || MQ);
      internal::PAHcurlHdivMassApply3D(dofs1D, dofs1Dtest, quad1D, ne, scalarCoeff,
                                       trial_div, true, mapsO->B, mapsC->B,
                                       mapsOtest->Bt, mapsCtest->Bt, pa_data, x, y);
      symmetricSpaces = false;
   }
   else if (dim == 2 && ((trial_curl && test_div) || (trial_div && test_curl)))
   {
      const bool scalarCoeff = !(DQ || MQ);
      internal::PAHcurlHdivMassApply2D(dofs1D, dofs1Dtest, quad1D, ne, scalarCoeff,
                                       !trial_curl, true, mapsO->B, mapsC->B,
                                       mapsOtest->Bt, mapsCtest->Bt, pa_data, x, y);
      symmetricSpaces = false;
   }
   if (symmetricSpaces)
   {
      if (MQ && dynamic_cast<SymmetricMatrixCoefficient*>(MQ) == NULL)
      {
         MFEM_ABORT("VectorFEMassIntegrator transpose not implemented for asymmetric MatrixCoefficient");
      }
      AddMultPA(x, y);
   }
}

} // namespace mfem
