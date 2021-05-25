// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../tmop.hpp"
#include "../linearform.hpp"
#include "../pgridfunc.hpp"
#include "../tmop_tools.hpp"
#include "../quadinterpolator.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

void TMOP_Integrator::AssembleGradPA(const Vector &xe,
                                     const FiniteElementSpace &fes)
{
   MFEM_VERIFY(PA.enabled, "PA extension setup has not been done!");
   MFEM_VERIFY(PA.fes == &fes, "");
   // TODO: we need a more robust way to check that the 'fes' used when
   // AssemblePA() was called has not been modified or completely destroyed and
   // a new object created at the same address.

   if (PA.Jtr_needs_update || targetC->UsesPhysicalCoordinates())
   {
      ComputeAllElementTargets(xe);
      PA.Jtr_debug_grad = true;
   }

   if (PA.dim == 2)
   {
      AssembleGradPA_2D(xe);
      if (coeff0) { AssembleGradPA_C0_2D(xe); }
   }

   if (PA.dim == 3)
   {
      AssembleGradPA_3D(xe);
      if (coeff0) { AssembleGradPA_C0_3D(xe); }
   }
}

void TMOP_Integrator::AssemblePA_Limiting()
{
   // Return immediately if limiting is not enabled
   if (coeff0 == nullptr) { return; }
   MFEM_VERIFY(nodes0, "internal error");

   MFEM_VERIFY(PA.enabled, "AssemblePA_Limiting but PA is not enabled!");
   MFEM_VERIFY(lim_func, "No TMOP_LimiterFunction specification!")
   MFEM_VERIFY(dynamic_cast<TMOP_QuadraticLimiter*>(lim_func),
               "Only TMOP_QuadraticLimiter is supported");

   const FiniteElementSpace *fes = PA.fes;
   const int NE = PA.ne;
   if (NE == 0) { return; }  // Quick return for empty processors
   const IntegrationRule &ir = *PA.ir;

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;

   // H0 for coeff0, (dim x dim) Q-vector
   PA.H0.UseDevice(true);
   PA.H0.SetSize(PA.dim * PA.dim * PA.nq * NE, Device::GetDeviceMemoryType());

   // coeff0 -> PA.C0 (Q-vector)
   PA.C0.UseDevice(true);
   if (ConstantCoefficient* cQ =
          dynamic_cast<ConstantCoefficient*>(coeff0))
   {
      PA.C0.SetSize(1, Device::GetMemoryType());
      PA.C0.HostWrite();
      PA.C0(0) = cQ->constant;
   }
   else
   {
      PA.C0.SetSize(PA.nq * PA.ne, Device::GetMemoryType());
      auto C0 = Reshape(PA.C0.HostWrite(), PA.nq, PA.ne);
      for (int e = 0; e < NE; ++e)
      {
         ElementTransformation& T = *fes->GetElementTransformation(e);
         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            C0(q,e) = coeff0->Eval(T, ir.IntPoint(q));
         }
      }
   }

   // nodes0 -> PA.X0 (E-vector)
   MFEM_VERIFY(nodes0->FESpace() == fes, "");
   const Operator *n0_R = fes->GetElementRestriction(ordering);
   PA.X0.SetSize(n0_R->Height(), Device::GetMemoryType());
   PA.X0.UseDevice(true);
   n0_R->Mult(*nodes0, PA.X0);

   // Limiting distances: lim_dist -> PA.LD (E-vector)
   // TODO: remove the hack for the case lim_dist == NULL.
   const FiniteElementSpace *limfes = (lim_dist) ? lim_dist->FESpace() : fes;
   const FiniteElement &lim_fe = *limfes->GetFE(0);
   PA.maps_lim = &lim_fe.GetDofToQuad(ir, DofToQuad::TENSOR);
   PA.LD.SetSize(NE*lim_fe.GetDof(), Device::GetMemoryType());
   PA.LD.UseDevice(true);
   if (lim_dist)
   {
      const Operator *ld_R = limfes->GetElementRestriction(ordering);
      ld_R->Mult(*lim_dist, PA.LD);
   }
   else
   {
      PA.LD = 1.0;
   }
}

void TargetConstructor::ComputeAllElementTargets(const FiniteElementSpace &fes,
                                                 const IntegrationRule &ir,
                                                 const Vector &xe,
                                                 DenseTensor &Jtr) const
{
   MFEM_VERIFY(Jtr.SizeI() == Jtr.SizeJ() && Jtr.SizeI() > 1, "");
   const int dim = Jtr.SizeI();
   bool done = false;
   if (dim == 2) { done = ComputeAllElementTargets<2>(fes, ir, xe, Jtr); }
   if (dim == 3) { done = ComputeAllElementTargets<3>(fes, ir, xe, Jtr); }

   if (!done) { ComputeAllElementTargets_Fallback(fes, ir, xe, Jtr); }
}

void AnalyticAdaptTC::ComputeAllElementTargets(const FiniteElementSpace &fes,
                                               const IntegrationRule &ir,
                                               const Vector &xe,
                                               DenseTensor &Jtr) const
{
   ComputeAllElementTargets_Fallback(fes, ir, xe, Jtr);
}


// Code paths leading to ComputeElementTargets:
// - GetElementEnergy(elfun) which is done through GetGridFunctionEnergyPA(x)
// - AssembleElementVectorExact(elfun)
// - AssembleElementGradExact(elfun)
// - EnableNormalization(x) -> ComputeNormalizationEnergies(x)
// - (AssembleElementVectorFD(elfun))
// - (AssembleElementGradFD(elfun))
// ============================================================================
//      - TargetConstructor():
//          - IDEAL_SHAPE_UNIT_SIZE: Wideal
//          - IDEAL_SHAPE_EQUAL_SIZE: α * Wideal
//          - IDEAL_SHAPE_GIVEN_SIZE: β * Wideal
//          - GIVEN_SHAPE_AND_SIZE:   β * Wideal
//      - AnalyticAdaptTC(elfun):
//          - GIVEN_FULL: matrix_tspec->Eval(Jtr(elfun))
//      - DiscreteAdaptTC():
//          - IDEAL_SHAPE_GIVEN_SIZE: size^{1.0/dim} * Jtr(i)  (size)
//          - GIVEN_SHAPE_AND_SIZE:   Jtr(i) *= D_rho          (ratio)
//                                    Jtr(i) *= Q_phi          (skew)
//                                    Jtr(i) *= R_theta        (orientation)
void TMOP_Integrator::ComputeAllElementTargets(const Vector &xe) const
{
   PA.Jtr_needs_update = false;
   PA.Jtr_debug_grad = false;
   const FiniteElementSpace *fes = PA.fes;
   if (PA.ne == 0) { return; }  // Quick return for empty processors
   const IntegrationRule &ir = *PA.ir;

   // Compute PA.Jtr for all elements
   targetC->ComputeAllElementTargets(*fes, ir, xe, PA.Jtr);
}

void TMOP_Integrator::AssemblePA(const FiniteElementSpace &fes)
{
   PA.enabled = true;
   PA.fes = &fes;
   Mesh *mesh = fes.GetMesh();
   const int ne = PA.ne = mesh->GetNE();
   if (ne == 0) { return; }  // Quick return for empty processors
   const int dim = PA.dim = mesh->Dimension();
   MFEM_VERIFY(PA.dim == 2 || PA.dim == 3, "Not yet implemented!");
   MFEM_VERIFY(mesh->GetNumGeometries(dim) <= 1,
               "mixed meshes are not supported");
   MFEM_VERIFY(!fes.IsVariableOrder(), "variable orders are not supported");
   const FiniteElement &fe = *fes.GetFE(0);
   PA.ir = &EnergyIntegrationRule(fe);
   const IntegrationRule &ir = *PA.ir;
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");

   const int nq = PA.nq = ir.GetNPoints();
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   PA.maps = &fe.GetDofToQuad(ir, mode);
   PA.geom = mesh->GetGeometricFactors(ir, GeometricFactors::JACOBIANS);

   // Energy vector, scalar Q-vector
   PA.E.UseDevice(true);
   PA.E.SetSize(ne*nq, Device::GetDeviceMemoryType());

   // H for Grad, (dim x dim) Q-vector
   PA.H.UseDevice(true);
   PA.H.SetSize(dim*dim * dim*dim * nq*ne, Device::GetDeviceMemoryType());

   // Scalar Q-vector of '1', used to compute sums via dot product
   PA.O.SetSize(ne*nq, Device::GetDeviceMemoryType());
   PA.O = 1.0;

   // Setup ref->target Jacobians, PA.Jtr, (dim x dim) Q-vector, DenseTensor
   PA.Jtr.SetSize(dim, dim, PA.ne*PA.nq);
   PA.Jtr_needs_update = true;
   PA.Jtr_debug_grad = false;

   // Limiting: coeff0 -> PA.C0, nodes0 -> PA.X0, lim_dist -> PA.LD, PA.H0
   if (coeff0) { AssemblePA_Limiting(); }
}

void TMOP_Integrator::AssembleGradDiagonalPA(Vector &de) const
{
   // This method must be called after AssembleGradPA().

   MFEM_VERIFY(PA.Jtr_needs_update == false, "");

   if (targetC->UsesPhysicalCoordinates())
   {
      MFEM_VERIFY(PA.Jtr_debug_grad == true, "AssembleGradPA() was not called"
                  " or Jtr was overwritten by another method!");
   }

   if (PA.dim == 2)
   {
      AssembleDiagonalPA_2D(de);
      if (coeff0) { AssembleDiagonalPA_C0_2D(de); }
   }

   if (PA.dim == 3)
   {
      AssembleDiagonalPA_3D(de);
      if (coeff0) { AssembleDiagonalPA_C0_3D(de); }
   }
}

void TMOP_Integrator::AddMultPA(const Vector &xe, Vector &ye) const
{
   // This method must be called after AssemblePA().

   if (PA.Jtr_needs_update || targetC->UsesPhysicalCoordinates())
   {
      ComputeAllElementTargets(xe);
   }

   if (PA.dim == 2)
   {
      AddMultPA_2D(xe,ye);
      if (coeff0) { AddMultPA_C0_2D(xe,ye); }
   }

   if (PA.dim == 3)
   {
      AddMultPA_3D(xe,ye);
      if (coeff0) { AddMultPA_C0_3D(xe,ye); }
   }
}

void TMOP_Integrator::AddMultGradPA(const Vector &re, Vector &ce) const
{
   // This method must be called after AssembleGradPA().

   MFEM_VERIFY(PA.Jtr_needs_update == false, "");

   if (targetC->UsesPhysicalCoordinates())
   {
      MFEM_VERIFY(PA.Jtr_debug_grad == true, "AssembleGradPA() was not called or "
                  "Jtr was overwritten by another method!");
   }

   if (PA.dim == 2)
   {
      AddMultGradPA_2D(re,ce);
      if (coeff0) { AddMultGradPA_C0_2D(re,ce); }
   }

   if (PA.dim == 3)
   {
      AddMultGradPA_3D(re,ce);
      if (coeff0) { AddMultGradPA_C0_3D(re,ce); }
   }
}

double TMOP_Integrator::GetLocalStateEnergyPA(const Vector &xe) const
{
   // This method must be called after AssemblePA().

   double energy = 0.0;

   if (PA.Jtr_needs_update || targetC->UsesPhysicalCoordinates())
   {
      ComputeAllElementTargets(xe);
   }

   if (PA.dim == 2)
   {
      energy = GetLocalStateEnergyPA_2D(xe);
      if (coeff0) { energy += GetLocalStateEnergyPA_C0_2D(xe); }
   }

   if (PA.dim == 3)
   {
      energy = GetLocalStateEnergyPA_3D(xe);
      if (coeff0) { energy += GetLocalStateEnergyPA_C0_3D(xe); }
   }

   return energy;
}

} // namespace mfem
