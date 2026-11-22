// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
#include "bilininteg_vectorfemass_kernels.hpp"

namespace mfem
{
/// \cond DO_NOT_DOCUMENT
VectorFEMassIntegrator::ApplyKernelType
VectorFEMassIntegrator::ApplyPAKernels::Fallback(
   FiniteElement::DerivType TrialType, FiniteElement::DerivType TestType,
   int dim, int, int, int)
{
   const bool trial_curl = (TrialType == mfem::FiniteElement::CURL);
   const bool trial_div = (TrialType == mfem::FiniteElement::DIV);
   const bool test_curl = (TestType == mfem::FiniteElement::CURL);
   const bool test_div = (TestType == mfem::FiniteElement::DIV);

   if (dim == 3)
   {
      if (trial_curl && test_curl)
      {
         return internal::PAHcurlMassApply3D;
      }
      else if (trial_div && test_div)
      {
         return internal::PAHdivMassApply3D;
      }
      else if (trial_curl && test_div)
      {
         return internal::PAHdivHcurlMassApply3D;
      }
      else if (trial_div && test_curl)
      {
         return internal::PAHcurlHdivMassApply3D;
      }
   }
   else if (dim == 2) // 2D
   {
      if (trial_curl && test_curl)
      {
         return internal::PAHcurlMassApply2D;
      }
      else if (trial_div && test_div)
      {
         return internal::PAHdivMassApply2D;
      }
      else if (trial_curl && test_div)
      {
         return internal::PAHdivHcurlMassApply2D;
      }
      else if (trial_div && test_curl)
      {
         return internal::PAHcurlHdivMassApply2D;
      }
   }
   MFEM_ABORT("Unknown kernel.");
}
/// \endcond DO_NOT_DOCUMENT

VectorFEMassIntegrator::Kernels::Kernels()
{
   // h(curl), h(curl)
   // Q = P + 1 (3D)
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::CURL,
                          FiniteElement::CURL, 3, 2, 2, 3>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::CURL,
                          FiniteElement::CURL, 3, 3, 3, 4>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::CURL,
                          FiniteElement::CURL, 3, 4, 4, 5>();
   // Q = P + 2 (3D)
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::CURL,
                          FiniteElement::CURL, 3, 2, 2, 4>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::CURL,
                          FiniteElement::CURL, 3, 3, 3, 5>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::CURL,
                          FiniteElement::CURL, 3, 4, 4, 6>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::CURL,
                          FiniteElement::CURL, 3, 5, 5, 7>();
   // Q = P + 4 (3D)
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::CURL,
                          FiniteElement::CURL, 3, 2, 2, 6>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::CURL,
                          FiniteElement::CURL, 3, 3, 3, 7>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::CURL,
                          FiniteElement::CURL, 3, 4, 4, 8>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::CURL,
                          FiniteElement::CURL, 3, 5, 5, 9>();
   // h(div), h(div)
   // Q = P (2D)
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::DIV,
                          FiniteElement::DIV, 2, 2, 2, 2>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::DIV,
                          FiniteElement::DIV, 2, 3, 3, 3>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::DIV,
                          FiniteElement::DIV, 2, 4, 4, 4>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::DIV,
                          FiniteElement::DIV, 2, 5, 5, 5>();

   // Q = P + 1 (3D)
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::DIV,
                          FiniteElement::DIV, 3, 2, 2, 3>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::DIV,
                          FiniteElement::DIV, 3, 3, 3, 4>();
   VectorFEMassIntegrator::AddSpecialization<FiniteElement::DIV,
                          FiniteElement::DIV, 3, 4, 4, 5>();
}

void VectorFEMassIntegrator::Init(Coefficient *q, DiagonalMatrixCoefficient *dq,
                                  MatrixCoefficient *mq)
{
   static Kernels kernels{};
   Q = q;
   DQ = dq;
   MQ = mq;
}

void VectorFEMassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   AssemblePA(fes, fes);
}

void VectorFEMassIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                        const FiniteElementSpace &test_fes)
{
   var_order = trial_fes.IsVariableOrder() || test_fes.IsVariableOrder();
   pa_groups.clear();

   // Assumes tensor-product elements
   Mesh *mesh = trial_fes.GetMesh();

   if (var_order)
   {
      MFEM_VERIFY(&trial_fes == &test_fes,
                  "Variable-order VectorFEMassIntegrator PA requires identical "
                  "trial and test spaces.");
      MFEM_VERIFY(!Device::Allows(Backend::CEED_MASK),
                  "Variable-order VectorFEMassIntegrator PA does not support CEED backends.");
      MFEM_VERIFY(trial_fes.GetVDim() == 1,
                  "Variable-order VectorFEMassIntegrator PA requires vdim == 1.");
      dim = mesh->Dimension();
      MFEM_VERIFY(dim == 2 || dim == 3, "");
      MFEM_VERIFY(mesh->SpaceDimension() == dim,
                  "Variable-order VectorFEMassIntegrator PA requires mesh dimension == space dimension.");

      const FiniteElement *typical_fel = trial_fes.GetTypicalFE();
      const auto *typical_el = dynamic_cast<const VectorTensorFiniteElement*>(typical_fel);
      MFEM_VERIFY(typical_el != nullptr, "Only VectorTensorFiniteElement is supported!");
      trial_fetype = static_cast<FiniteElement::DerivType>(typical_el->GetDerivType());
      test_fetype = trial_fetype;
      const bool is_curl = (trial_fetype == FiniteElement::CURL);
      const bool is_div = (trial_fetype == FiniteElement::DIV);
      MFEM_VERIFY(is_curl || is_div, "Unknown kernel.");

      symmetric = !(MQ && dynamic_cast<SymmetricMatrixCoefficient*>(MQ) == nullptr);
      ne = trial_fes.GetNE();

      const ElementDofOrdering ordering = GetEVectorOrdering(trial_fes);
      MFEM_VERIFY(ordering == ElementDofOrdering::LEXICOGRAPHIC,
                  "Variable-order PA requires tensor-product elements.");
      const auto *vr = dynamic_cast<const VarOrderElementRestriction*>(
                          trial_fes.GetElementRestriction(ordering));
      MFEM_VERIFY(vr != nullptr,
                  "Expected VarOrderElementRestriction for variable-order space.");

      const int symm_dims = (dim * (dim + 1)) / 2;
      const int ndata = symmetric ? symm_dims : dim * dim;

      pa_groups.resize(vr->GetNumGroups());
      for (int gi = 0; gi < vr->GetNumGroups(); ++gi)
      {
         const Array<int> &elems = vr->GetGroupElements(gi);
         PAGroup &g = pa_groups[gi];
         g.ne = elems.Size();
         g.evec_offset = vr->GetGroupOffset(gi);
         g.evec_size = vr->GetGroupSize(gi);
         if (g.ne == 0) { continue; }

         const FiniteElement *fel = trial_fes.GetFE(elems[0]);
         const auto *el = dynamic_cast<const VectorTensorFiniteElement*>(fel);
         MFEM_VERIFY(el != nullptr, "Only VectorTensorFiniteElement is supported!");
         MFEM_VERIFY(el->GetDim() == dim, "");
         MFEM_VERIFY(el->GetDerivType() == trial_fetype,
                     "Inconsistent finite element DerivType in VectorFEMassIntegrator PA.");

         const IntegrationRule *ir = IntRule ? IntRule
                                     : &MassIntegrator::GetRule(*el, *el,
                                                                *mesh->GetElementTransformation(elems[0]));
         g.mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
         g.mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
         g.dofs1D = g.mapsC->ndof;
         g.quad1D = g.mapsC->nqpt;
         MFEM_VERIFY(g.dofs1D == g.mapsO->ndof + 1 &&
                     g.quad1D == g.mapsO->nqpt, "");
         const int expected_nq = (dim == 3)
                                 ? (g.quad1D * g.quad1D * g.quad1D)
                                 : (g.quad1D * g.quad1D);
         MFEM_VERIFY(ir->GetNPoints() == expected_nq,
                     "Variable-order VectorFEMassIntegrator PA requires isotropic tensor-product quadrature.");
         const int max_d1d = is_curl ? DeviceDofQuadLimits::Get().HCURL_MAX_D1D
                                     : DeviceDofQuadLimits::Get().HDIV_MAX_D1D;
         const int max_q1d = is_curl ? DeviceDofQuadLimits::Get().HCURL_MAX_Q1D
                                     : DeviceDofQuadLimits::Get().HDIV_MAX_Q1D;
         MFEM_VERIFY(g.dofs1D <= max_d1d && g.quad1D <= max_q1d,
                     "Variable-order VectorFEMassIntegrator PA exceeds MAX_D1D / MAX_Q1D.");

         Vector J_g, C_g;
         int coeff_dim = 0;
         bool grp_sym = true;
         internal::ComputeGroupJacobiansAndCoeffs(*mesh, elems, *ir, dim,
                                                  Q, DQ, MQ, J_g, C_g,
                                                  coeff_dim, grp_sym);
         g.pa_data.SetSize(ndata * ir->GetNPoints() * g.ne,
                           Device::GetMemoryType());

         if (is_curl && dim == 3)
         {
            internal::PADiffusionSetup3D(g.quad1D, coeff_dim, g.ne,
                                         ir->GetWeights(), J_g, C_g, g.pa_data);
         }
         else if (is_curl && dim == 2)
         {
            internal::PADiffusionSetup2D<2>(g.quad1D, coeff_dim, g.ne,
                                            ir->GetWeights(), J_g, C_g, g.pa_data);
         }
         else if (is_div && dim == 3)
         {
            internal::PAHdivMassSetup3D(g.quad1D, coeff_dim, g.ne,
                                        ir->GetWeights(), J_g, C_g, g.pa_data);
         }
         else
         {
            internal::PAHdivMassSetup2D(g.quad1D, coeff_dim, g.ne,
                                        ir->GetWeights(), J_g, C_g, g.pa_data);
         }
      }
      return;
   }

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

   trial_fetype = static_cast<FiniteElement::DerivType>(trial_el->GetDerivType());
   test_fetype = static_cast<FiniteElement::DerivType>(test_el->GetDerivType());

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
   if (var_order)
   {
      for (const PAGroup &g : pa_groups)
      {
         if (g.ne == 0) { continue; }
         Vector dg;
         dg.MakeRef(diag, g.evec_offset, g.evec_size);
         if (dim == 3)
         {
            if (trial_fetype == FiniteElement::CURL)
            {
               if (Device::Allows(Backend::DEVICE_MASK))
               {
                  const int ID = (g.dofs1D << 4) | g.quad1D;
                  switch (ID)
                  {
                     case 0x23:
                        internal::SmemPAHcurlMassAssembleDiagonal3D<2,3>(
                           g.dofs1D, g.quad1D, g.ne, symmetric,
                           g.mapsO->B, g.mapsC->B, g.pa_data, dg);
                        break;
                     case 0x34:
                        internal::SmemPAHcurlMassAssembleDiagonal3D<3,4>(
                           g.dofs1D, g.quad1D, g.ne, symmetric,
                           g.mapsO->B, g.mapsC->B, g.pa_data, dg);
                        break;
                     case 0x45:
                        internal::SmemPAHcurlMassAssembleDiagonal3D<4,5>(
                           g.dofs1D, g.quad1D, g.ne, symmetric,
                           g.mapsO->B, g.mapsC->B, g.pa_data, dg);
                        break;
                     case 0x56:
                        internal::SmemPAHcurlMassAssembleDiagonal3D<5,6>(
                           g.dofs1D, g.quad1D, g.ne, symmetric,
                           g.mapsO->B, g.mapsC->B, g.pa_data, dg);
                        break;
                     default:
                        internal::SmemPAHcurlMassAssembleDiagonal3D(
                           g.dofs1D, g.quad1D, g.ne, symmetric,
                           g.mapsO->B, g.mapsC->B, g.pa_data, dg);
                        break;
                  }
               }
               else
               {
                  internal::PAHcurlMassAssembleDiagonal3D(
                     g.dofs1D, g.quad1D, g.ne, symmetric,
                     g.mapsO->B, g.mapsC->B, g.pa_data, dg);
               }
            }
            else if (trial_fetype == FiniteElement::DIV)
            {
               internal::PAHdivMassAssembleDiagonal3D(
                  g.dofs1D, g.quad1D, g.ne, symmetric,
                  g.mapsO->B, g.mapsC->B, g.pa_data, dg);
            }
            else
            {
               MFEM_ABORT("Unknown kernel.");
            }
         }
         else
         {
            if (trial_fetype == FiniteElement::CURL)
            {
               internal::PAHcurlMassAssembleDiagonal2D(
                  g.dofs1D, g.quad1D, g.ne, symmetric,
                  g.mapsO->B, g.mapsC->B, g.pa_data, dg);
            }
            else if (trial_fetype == FiniteElement::DIV)
            {
               internal::PAHdivMassAssembleDiagonal2D(
                  g.dofs1D, g.quad1D, g.ne, symmetric,
                  g.mapsO->B, g.mapsC->B, g.pa_data, dg);
            }
            else
            {
               MFEM_ABORT("Unknown kernel.");
            }
         }
         dg.SyncAliasMemory(diag);
      }
      return;
   }
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
   const bool scalar_coeff = !(DQ || MQ);
   if (var_order)
   {
      x.Read();
      for (const PAGroup &g : pa_groups)
      {
         if (g.ne == 0) { continue; }
         Vector xg, yg;
         xg.MakeRef(const_cast<Vector&>(x), g.evec_offset, g.evec_size);
         yg.MakeRef(y, g.evec_offset, g.evec_size);
         ApplyPAKernels::Run(trial_fetype, test_fetype, dim,
                             g.dofs1D, g.dofs1D, g.quad1D, g.ne, symmetric,
                             scalar_coeff, g.mapsO->B, g.mapsC->B,
                             g.mapsO->Bt, g.mapsC->Bt, g.pa_data, xg, yg,
                             g.dofs1D, g.dofs1D, g.quad1D);
         yg.SyncAliasMemory(y);
      }
      return;
   }
   ApplyPAKernels::Run(trial_fetype, test_fetype, dim, dofs1D, dofs1Dtest,
                       quad1D, ne, symmetric, scalar_coeff, mapsO->B, mapsC->B,
                       mapsOtest->Bt, mapsCtest->Bt, pa_data, x, y, dofs1D,
                       dofs1Dtest, quad1D);
}

void VectorFEMassIntegrator::AddAbsMultPA(const Vector &x, Vector &y) const
{
   const bool scalar_coeff = !(DQ || MQ);
   if (var_order)
   {
      x.Read();
      for (const PAGroup &g : pa_groups)
      {
         if (g.ne == 0) { continue; }
         Vector abs_pa_data(g.pa_data);
         abs_pa_data.Abs();
         Array<real_t> absBo(g.mapsO->B);
         Array<real_t> absBc(g.mapsC->B);
         Array<real_t> absBto_t(g.mapsO->Bt);
         Array<real_t> absBtc_t(g.mapsC->Bt);
         absBo.Abs();
         absBc.Abs();
         absBto_t.Abs();
         absBtc_t.Abs();

         Vector xg, yg;
         xg.MakeRef(const_cast<Vector&>(x), g.evec_offset, g.evec_size);
         yg.MakeRef(y, g.evec_offset, g.evec_size);
         ApplyPAKernels::Run(trial_fetype, test_fetype, dim,
                             g.dofs1D, g.dofs1D, g.quad1D, g.ne, symmetric,
                             scalar_coeff, absBo, absBc, absBto_t, absBtc_t,
                             abs_pa_data, xg, yg,
                             g.dofs1D, g.dofs1D, g.quad1D);
         yg.SyncAliasMemory(y);
      }
      return;
   }

   Vector abs_pa_data(pa_data);
   abs_pa_data.Abs();

   Array<real_t> absBo(mapsO->B);
   Array<real_t> absBc(mapsC->B);
   Array<real_t> absBto_t(mapsOtest->Bt);
   Array<real_t> absBtc_t(mapsCtest->Bt);

   absBo.Abs();
   absBc.Abs();
   absBto_t.Abs();
   absBtc_t.Abs();

   ApplyPAKernels::Run(trial_fetype, test_fetype, dim, dofs1D, dofs1Dtest,
                       quad1D, ne, symmetric, scalar_coeff, absBo, absBc,
                       absBto_t, absBtc_t, abs_pa_data, x, y, dofs1D,
                       dofs1Dtest, quad1D);
}

void VectorFEMassIntegrator::AddMultTransposePA(const Vector &x,
                                                Vector &y) const
{
   if (var_order)
   {
      if (MQ && dynamic_cast<SymmetricMatrixCoefficient*>(MQ) == nullptr)
      {
         MFEM_ABORT("VectorFEMassIntegrator transpose not implemented for asymmetric MatrixCoefficient");
      }
      AddMultPA(x, y);
      return;
   }
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
