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
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "bilininteg_hdiv_kernels.hpp"

namespace mfem
{

DivDivIntegrator::DivDivIntegrator() : Q(nullptr)
{
   static Kernels kernels;
}

DivDivIntegrator::DivDivIntegrator(Coefficient &q, const IntegrationRule *ir)
   : BilinearFormIntegrator(ir), Q(&q)
{
   static Kernels kernels;
}

/// \cond DO_NOT_DOCUMENT

DivDivIntegrator::Kernels::Kernels()
{
   // 2D: Q1D = D1D
   // RT0 (D1D = Q1D = 2) is not specialized because the fallback is faster
   DivDivIntegrator::AddSpecialization<2, 3, 3>();
   DivDivIntegrator::AddSpecialization<2, 4, 4>();
   DivDivIntegrator::AddSpecialization<2, 5, 5>();
   // 3D: Q1D = D1D + 1
   DivDivIntegrator::AddSpecialization<3, 2, 3>();
   DivDivIntegrator::AddSpecialization<3, 3, 4>();
   DivDivIntegrator::AddSpecialization<3, 4, 5>();
   DivDivIntegrator::AddSpecialization<3, 5, 6>();
}

DivDivIntegrator::ApplyKernelType
DivDivIntegrator::ApplyPAKernels::Fallback(int DIM, int, int)
{
   if (DIM == 2) { return internal::PADivDivApply2D; }
   else if (DIM == 3) { return internal::PADivDivApply3D; }
   else { MFEM_ABORT("Unsupported dimension!"); }
}

/// \endcond DO_NOT_DOCUMENT

void DivDivIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   var_order = fes.IsVariableOrder();
   pa_groups.clear();

   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();

   if (var_order)
   {
      MFEM_VERIFY(!Device::Allows(Backend::CEED_MASK),
                  "Variable-order DivDivIntegrator PA does not support CEED backends.");
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "Variable-order DivDivIntegrator PA requires vdim == 1.");
      dim = mesh->Dimension();
      MFEM_VERIFY(dim == 2 || dim == 3, "");
      MFEM_VERIFY(mesh->SpaceDimension() == dim,
                  "Variable-order DivDivIntegrator PA requires mesh dimension == space dimension.");

      ne = fes.GetNE();

      const ElementDofOrdering ordering = GetEVectorOrdering(fes);
      MFEM_VERIFY(ordering == ElementDofOrdering::LEXICOGRAPHIC,
                  "Variable-order PA requires tensor-product elements.");
      const auto *vr = dynamic_cast<const VarOrderElementRestriction*>(
                          fes.GetElementRestriction(ordering));
      MFEM_VERIFY(vr != nullptr,
                  "Expected VarOrderElementRestriction for variable-order space.");

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
         MFEM_VERIFY(el->GetDerivType() == FiniteElement::DIV,
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
                     "Variable-order DivDivIntegrator PA requires isotropic tensor-product quadrature.");
         MFEM_VERIFY(g.dofs1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D &&
                     g.quad1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
                     "Variable-order DivDivIntegrator PA exceeds HDIV_MAX_D1D / HDIV_MAX_Q1D.");

         Vector J_g, C_g;
         int coeff_dim = 0;
         bool grp_sym = true;
         internal::ComputeGroupJacobiansAndCoeffs(*mesh, elems, *ir, dim,
                                                  Q, nullptr, nullptr, J_g, C_g,
                                                  coeff_dim, grp_sym);
         g.pa_data.SetSize(ir->GetNPoints() * g.ne, Device::GetMemoryType());

         if (dim == 3)
         {
            internal::PADivDivSetup3D(g.quad1D, g.ne, ir->GetWeights(),
                                      J_g, C_g, g.pa_data);
         }
         else
         {
            internal::PADivDivSetup2D(g.quad1D, g.ne, ir->GetWeights(),
                                      J_g, C_g, g.pa_data);
         }
      }
      return;
   }

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
   if (var_order)
   {
      for (const PAGroup &g : pa_groups)
      {
         if (g.ne == 0) { continue; }
         Vector dg;
         dg.MakeRef(diag, g.evec_offset, g.evec_size);
         if (dim == 3)
         {
            internal::PADivDivAssembleDiagonal3D(g.dofs1D, g.quad1D, g.ne,
                                                 g.mapsO->B, g.mapsC->G,
                                                 g.pa_data, dg);
         }
         else
         {
            internal::PADivDivAssembleDiagonal2D(g.dofs1D, g.quad1D, g.ne,
                                                 g.mapsO->B, g.mapsC->G,
                                                 g.pa_data, dg);
         }
         dg.SyncAliasMemory(diag);
      }
      return;
   }
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
   if (var_order)
   {
      x.Read();
      for (const PAGroup &g : pa_groups)
      {
         if (g.ne == 0) { continue; }
         Vector xg, yg;
         xg.MakeRef(const_cast<Vector&>(x), g.evec_offset, g.evec_size);
         yg.MakeRef(y, g.evec_offset, g.evec_size);
         ApplyPAKernels::Run(dim, g.dofs1D, g.quad1D, g.dofs1D, g.quad1D, g.ne,
                             g.mapsO->B, g.mapsC->G, g.mapsO->Bt, g.mapsC->Gt,
                             g.pa_data, xg, yg);
         yg.SyncAliasMemory(y);
      }
      return;
   }
   ApplyPAKernels::Run(dim, dofs1D, quad1D, dofs1D, quad1D, ne, mapsO->B,
                       mapsC->G, mapsO->Bt, mapsC->Gt, pa_data, x, y);
}

void DivDivIntegrator::AddAbsMultPA(const Vector &x, Vector &y) const
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
         ApplyPAKernels::Run(dim, g.dofs1D, g.quad1D, g.dofs1D, g.quad1D, g.ne,
                             absO.B, absC.G, absO.Bt, absC.Gt,
                             abs_pa_data, xg, yg);
         yg.SyncAliasMemory(y);
      }
      return;
   }
   Vector abs_pa_data(pa_data);
   abs_pa_data.Abs();
   auto absO = mapsO->Abs();
   auto absC = mapsC->Abs();
   ApplyPAKernels::Run(dim, dofs1D, quad1D, dofs1D, quad1D, ne, absO.B, absC.G,
                       absO.Bt, absC.Gt, abs_pa_data, x, y);
}

void DivDivIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   AddMultPA(x, y);
}

} // namespace mfem
