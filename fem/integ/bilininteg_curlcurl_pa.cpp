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

#include "../qfunction.hpp"
#include "bilininteg_hcurl_kernels.hpp"

namespace mfem
{

CurlCurlIntegrator::CurlCurlIntegrator() : Q(nullptr), DQ(nullptr), MQ(nullptr)
{
   static Kernels kernels;
}

CurlCurlIntegrator::CurlCurlIntegrator(Coefficient &q,
                                       const IntegrationRule *ir)
   : BilinearFormIntegrator(ir), Q(&q), DQ(nullptr), MQ(nullptr)
{
   static Kernels kernels;
}

CurlCurlIntegrator::CurlCurlIntegrator(DiagonalMatrixCoefficient &dq,
                                       const IntegrationRule *ir)
   : BilinearFormIntegrator(ir), Q(nullptr), DQ(&dq), MQ(nullptr)
{
   static Kernels kernels;
}

CurlCurlIntegrator::CurlCurlIntegrator(MatrixCoefficient &mq,
                                       const IntegrationRule *ir)
   : BilinearFormIntegrator(ir), Q(nullptr), DQ(nullptr), MQ(&mq)
{
   static Kernels kernels;
}

/// \cond DO_NOT_DOCUMENT

CurlCurlIntegrator::Kernels::Kernels()
{
   CurlCurlIntegrator::AddSpecialization<3, 2, 3>();
   CurlCurlIntegrator::AddSpecialization<3, 3, 4>();
   CurlCurlIntegrator::AddSpecialization<3, 4, 5>();
   CurlCurlIntegrator::AddSpecialization<3, 5, 6>();
}

CurlCurlIntegrator::ApplyKernelType
CurlCurlIntegrator::ApplyPAKernels::Fallback(int DIM, int, int)
{
   if (DIM == 2) { return internal::PACurlCurlApply2D; }
   else if (DIM == 3)
   {
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         return internal::SmemPACurlCurlApply3D;
      }
      else
      {
         return internal::PACurlCurlApply3D;
      }
   }
   else { MFEM_ABORT(""); }
}

CurlCurlIntegrator::DiagonalKernelType
CurlCurlIntegrator::DiagonalPAKernels::Fallback(int DIM, int, int)
{
   if (DIM == 2)
   {
      return internal::PACurlCurlAssembleDiagonal2D;
   }
   else if (DIM == 3)
   {
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         return internal::SmemPACurlCurlAssembleDiagonal3D;
      }
      else
      {
         return internal::PACurlCurlAssembleDiagonal3D;
      }
   }
   else
   {
      MFEM_ABORT("");
   }
}

/// \endcond DO_NOT_DOCUMENT

void CurlCurlIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   var_order = fes.IsVariableOrder();
   pa_groups.clear();

   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();

   if (var_order)
   {
      MFEM_VERIFY(!Device::Allows(Backend::CEED_MASK),
                  "Variable-order CurlCurlIntegrator PA does not support CEED backends.");
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "Variable-order CurlCurlIntegrator PA requires vdim == 1.");
      dim = mesh->Dimension();
      MFEM_VERIFY(dim == 2 || dim == 3, "");
      MFEM_VERIFY(mesh->SpaceDimension() == dim,
                  "Variable-order CurlCurlIntegrator PA requires mesh dimension == space dimension.");
      if (dim == 2)
      {
         MFEM_VERIFY(!DQ && !MQ,
                     "2D CurlCurlIntegrator PA requires scalar coefficient.");
      }

      symmetric = (dim == 2) ||
                  !(MQ && dynamic_cast<SymmetricMatrixCoefficient*>(MQ) == nullptr);
      ne = fes.GetNE();

      const ElementDofOrdering ordering = GetEVectorOrdering(fes);
      MFEM_VERIFY(ordering == ElementDofOrdering::LEXICOGRAPHIC,
                  "Variable-order PA requires tensor-product elements.");
      const auto *vr = dynamic_cast<const VarOrderElementRestriction*>(
                          fes.GetElementRestriction(ordering));
      MFEM_VERIFY(vr != nullptr,
                  "Expected VarOrderElementRestriction for variable-order space.");

      const int sym_dims = (dim * (dim + 1)) / 2;
      const int ndata = (dim == 2) ? 1 : (symmetric ? sym_dims : dim * dim);

      pa_groups.resize(vr->GetNumGroups());
      for (int gi = 0; gi < vr->GetNumGroups(); ++gi)
      {
         const Array<int> &elems = vr->GetGroupElements(gi);
         PAGroup &g = pa_groups[gi];
         g.ne = elems.Size();
         g.evec_offset = vr->GetGroupOffset(gi);
         g.evec_size = vr->GetGroupSize(gi);
         if (g.ne == 0) { continue; }

         const FiniteElement *fel = fes.GetFE(elems[0]);
         const auto *el = dynamic_cast<const VectorTensorFiniteElement*>(fel);
         MFEM_VERIFY(el != nullptr, "Only VectorTensorFiniteElement is supported!");
         MFEM_VERIFY(el->GetDim() == dim, "");
         MFEM_VERIFY(el->GetDerivType() == FiniteElement::CURL,
                     "Unknown kernel.");

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
                     "Variable-order CurlCurlIntegrator PA requires isotropic tensor-product quadrature.");
         MFEM_VERIFY(g.dofs1D <= DeviceDofQuadLimits::Get().HCURL_MAX_D1D &&
                     g.quad1D <= DeviceDofQuadLimits::Get().HCURL_MAX_Q1D,
                     "Variable-order CurlCurlIntegrator PA exceeds HCURL_MAX_D1D / HCURL_MAX_Q1D.");

         Vector J_g, C_g;
         int coeff_dim = 0;
         bool grp_sym = true;
         internal::ComputeGroupJacobiansAndCoeffs(*mesh, elems, *ir, dim,
                                                  Q, DQ, MQ, J_g, C_g,
                                                  coeff_dim, grp_sym);
         g.pa_data.SetSize(ndata * ir->GetNPoints() * g.ne,
                           Device::GetMemoryType());

         if (dim == 3)
         {
            internal::PACurlCurlSetup3D(g.quad1D, coeff_dim, g.ne,
                                        ir->GetWeights(), J_g, C_g, g.pa_data);
         }
         else
         {
            internal::PACurlCurlSetup2D(g.quad1D, g.ne, ir->GetWeights(),
                                        J_g, C_g, g.pa_data);
         }
      }
      return;
   }

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
   if (var_order)
   {
      for (const PAGroup &g : pa_groups)
      {
         if (g.ne == 0) { continue; }
         Vector dg;
         dg.MakeRef(diag, g.evec_offset, g.evec_size);
         DiagonalPAKernels::Run(dim, g.dofs1D, g.quad1D, g.dofs1D, g.quad1D,
                                symmetric, g.ne, g.mapsO->B, g.mapsC->B,
                                g.mapsO->G, g.mapsC->G, g.pa_data, dg);
         dg.SyncAliasMemory(diag);
      }
      return;
   }
   DiagonalPAKernels::Run(dim, dofs1D, quad1D, dofs1D, quad1D, symmetric, ne,
                          mapsO->B, mapsC->B, mapsO->G, mapsC->G, pa_data,
                          diag);
}

void CurlCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (var_order)
   {
      x.Read();
      for (const PAGroup &g : pa_groups)
      {
         if (g.ne == 0) { continue; }
         Vector xg, yg;
         xg.MakeRef(const_cast<Vector&>(x), g.evec_offset, g.evec_size);
         yg.MakeRef(y, g.evec_offset, g.evec_size);
         ApplyPAKernels::Run(dim, g.dofs1D, g.quad1D, g.dofs1D, g.quad1D,
                             symmetric, g.ne, g.mapsO->B, g.mapsC->B,
                             g.mapsO->Bt, g.mapsC->Bt, g.mapsC->G, g.mapsC->Gt,
                             g.pa_data, xg, yg, false);
         yg.SyncAliasMemory(y);
      }
      return;
   }
   ApplyPAKernels::Run(dim, dofs1D, quad1D, dofs1D, quad1D, symmetric, ne,
                       mapsO->B, mapsC->B, mapsO->Bt, mapsC->Bt, mapsC->G,
                       mapsC->Gt, pa_data, x, y, false);
}

void CurlCurlIntegrator::AddAbsMultPA(const Vector &x, Vector &y) const
{
   if (var_order)
   {
      x.Read();
      for (const PAGroup &g : pa_groups)
      {
         if (g.ne == 0) { continue; }
         Vector abs_pa_data(g.pa_data);
         abs_pa_data.Abs();
         auto absO = g.mapsO->Abs();
         auto absC = g.mapsC->Abs();
         Vector xg, yg;
         xg.MakeRef(const_cast<Vector&>(x), g.evec_offset, g.evec_size);
         yg.MakeRef(y, g.evec_offset, g.evec_size);
         ApplyPAKernels::Run(dim, g.dofs1D, g.quad1D, g.dofs1D, g.quad1D,
                             symmetric, g.ne, absO.B, absC.B, absO.Bt, absC.Bt,
                             absC.G, absC.Gt, abs_pa_data, xg, yg, true);
         yg.SyncAliasMemory(y);
      }
      return;
   }
   Vector abs_pa_data(pa_data);
   abs_pa_data.Abs();
   auto absO = mapsO->Abs();
   auto absC = mapsC->Abs();

   ApplyPAKernels::Run(dim, dofs1D, quad1D, dofs1D, quad1D, symmetric, ne,
                       absO.B, absC.B, absO.Bt, absC.Bt, absC.G, absC.Gt,
                       abs_pa_data, x, y, true);
}

void CurlCurlIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(symmetric,
               "CurlCurlIntegrator::AddMultTransposePA does not support "
               "nonsymmetric MatrixCoefficient.");
   AddMultPA(x, y);
}

} // namespace mfem
