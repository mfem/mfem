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

#include "tmop.hpp"
#include "linearform.hpp"
#include "pgridfunc.hpp"
#include "tmop_tools.hpp"
#include "quadinterpolator.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

void TMOP_Integrator::AssembleGradPA(const Vector &xe,
                                     const FiniteElementSpace &fes)
{
   MFEM_VERIFY(PA.R, "PA extension setup has not been done!");
   PA.setup_Grad = true;

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

// We might come here w/o knowing that PA will be used.
// It is the case when EnableLimiting is called before the Setup => AssemblePA.
void TMOP_Integrator::EnableLimitingPA(const GridFunction &n0)
{
   MFEM_VERIFY(PA.enabled, "EnableLimitingPA but PA is not enabled!");
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;

   // Nodes0
   const FiniteElementSpace *n0_fes = n0.FESpace();
   const Operator *n0_R = n0_fes->GetElementRestriction(ordering);
   PA.X0.SetSize(n0_R->Height(), Device::GetMemoryType());
   PA.X0.UseDevice(true);
   n0_R->Mult(n0, PA.X0);

   // Get the 1D maps for the distance FE space.
   const IntegrationRule &ir = EnergyIntegrationRule(*n0_fes->GetFE(0));
   PA.maps_lim =
      &lim_dist->FESpace()->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);

   // lim_dist & lim_func checks
   MFEM_VERIFY(lim_dist, "No lim_dist!")
   const FiniteElementSpace *ld_fes = lim_dist->FESpace();
   const Operator *ld_R = ld_fes->GetElementRestriction(ordering);
   MFEM_VERIFY(ld_R, "No lim_dist restriction operator found!");
   PA.LD.SetSize(ld_R->Height(), Device::GetMemoryType());
   PA.LD.UseDevice(true);
   ld_R->Mult(*lim_dist, PA.LD);

   // Only TMOP_QuadraticLimiter is supported
   MFEM_VERIFY(lim_func, "No lim_func!")
   MFEM_VERIFY(dynamic_cast<TMOP_QuadraticLimiter*>(lim_func),
               "Only TMOP_QuadraticLimiter is supported");
}

bool TargetConstructor::ComputeElementTargetsPA(const FiniteElementSpace *fes,
                                                const IntegrationRule *ir,
                                                DenseTensor &Jtr,
                                                const Vector &xe) const
{
   MFEM_VERIFY(Jtr.SizeI() == Jtr.SizeJ() && Jtr.SizeI() > 1, "");
   const int dim = Jtr.SizeI();
   if (dim == 2) { return ComputeElementTargetsPA<2>(fes, ir, Jtr, xe); }
   if (dim == 3) { return ComputeElementTargetsPA<3>(fes, ir, Jtr, xe); }
   return false;
}

bool AnalyticAdaptTC::ComputeElementTargetsPA(const FiniteElementSpace *fes,
                                              const IntegrationRule *ir,
                                              DenseTensor &Jtr,
                                              const Vector &xe) const
{
   return false;
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
void TMOP_Integrator::ComputeElementTargetsPA(const Vector &xe) const
{
   PA.setup_Jtr = false;
   const FiniteElementSpace *fes = PA.fes;
   const IntegrationRule &ir = EnergyIntegrationRule(*fes->GetFE(0));

   const TargetConstructor::TargetType &target_type = targetC->Type();
   const DiscreteAdaptTC *discr_tc = GetDiscreteAdaptTC();

   // Skip when TargetConstructor needs the nodes but have not been set
   const bool use_nodes =
      target_type == TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE ||
      target_type == TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE ||
      target_type == TargetConstructor::GIVEN_SHAPE_AND_SIZE;
   if (targetC && !discr_tc && use_nodes && !targetC->GetNodes()) { return; }

   // Try to use the TargetConstructor ComputeElementTargetsPA
   PA.setup_Jtr = targetC->ComputeElementTargetsPA(fes, &ir, PA.Jtr);
   if (PA.setup_Jtr) { return; }

   // Defaulting to host version
   PA.Jtr.HostWrite();

   const int NE = PA.ne;
   const int NQ = PA.nq;
   const int dim = PA.dim;
   DenseTensor &Jtr = PA.Jtr;

   Vector x;
   const bool useable_input_vector = xe.Size() > 0;
   const bool use_input_vector = target_type == TargetConstructor::GIVEN_FULL;

   if (use_input_vector && !useable_input_vector) { return; }

   if (discr_tc && !discr_tc->GetTspecFesv()) { return; }

   if (use_input_vector)
   {
      x.SetSize(PA.R->Width(), Device::GetMemoryType());
      x.UseDevice(true);
      PA.R->MultTranspose(xe, x);
      // Scale by weights
      const int N = PA.W.Size();
      const auto W = Reshape(PA.W.Read(), N);
      auto X = Reshape(x.ReadWrite(), N);
      MFEM_FORALL(i, N, X(i) /= W(i););
   }

   // Use TargetConstructor::ComputeElementTargets to fill the PA.Jtr
   Vector elfun;
   Array<int> vdofs;
   DenseTensor J;
   for (int e = 0; e < NE; e++)
   {
      const FiniteElement &fe = *fes->GetFE(e);
      if (use_input_vector)
      {
         fes->GetElementVDofs(e, vdofs);
         x.GetSubVector(vdofs, elfun);
      }
      J.UseExternalData(Jtr(e*NQ).Data(), dim, dim, NQ);
      targetC->ComputeElementTargets(e, fe, ir, elfun, J);
   }
   PA.setup_Jtr = true;
}

void TMOP_Integrator::AssemblePA(const FiniteElementSpace &fes)
{
   PA.enabled = true;
   MFEM_ASSERT(fes.GetMesh()->GetNE() > 0, "");
   PA.ir = &EnergyIntegrationRule(*fes.GetFE(0));
   const IntegrationRule &ir = *PA.ir;
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");

   PA.fes = &fes;
   Mesh *mesh = fes.GetMesh();
   const int nq = PA.nq = ir.GetNPoints();
   const int ne = PA.ne = fes.GetMesh()->GetNE();
   const int dim = PA.dim = mesh->Dimension();
   MFEM_VERIFY(PA.dim == 2 || PA.dim == 3, "Not yet implemented!");

   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   PA.maps = &fes.GetFE(0)->GetDofToQuad(ir, mode);
   PA.geom = mesh->GetGeometricFactors(ir, GeometricFactors::JACOBIANS, mode);

   // Energy vector
   PA.E.UseDevice(true);
   PA.E.SetSize(ne*nq, Device::GetDeviceMemoryType());

   // Setup initialization
   PA.setup_Jtr = false;
   PA.setup_Grad = false;

   // H for Grad
   PA.H.UseDevice(true);
   PA.H.SetSize(dim*dim * dim*dim * nq*ne, Device::GetDeviceMemoryType());
   // H0 for coeff0
   PA.H0.UseDevice(true);
   PA.H0.SetSize(dim * dim * nq*ne, Device::GetDeviceMemoryType());

   // Restriction setup
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   PA.R = fes.GetElementRestriction(ordering);
   MFEM_VERIFY(PA.R, "Not yet implemented!");

   // Weight of the R^t
   PA.W.SetSize(PA.R->Width(), Device::GetDeviceMemoryType());
   PA.W.UseDevice(true);
   PA.O.SetSize(dim*ne*nq, Device::GetDeviceMemoryType());
   PA.O.UseDevice(true);
   PA.O = 1.0;
   PA.R->MultTranspose(PA.O, PA.W);

   // Scalar vector of '1'
   PA.O.SetSize(ne*nq, Device::GetDeviceMemoryType());
   PA.O = 1.0;

   // TargetConstructor TargetType setup
   PA.Jtr.SetSize(dim, dim, PA.ne*PA.nq);
   ComputeElementTargetsPA();

   // Coeff0 PA.C0
   PA.C0.UseDevice(true);
   if (coeff0 == nullptr)
   {
      PA.C0.SetSize(1, Device::GetMemoryType());
      PA.C0.HostWrite();
      PA.C0(0) = 0.0;
   }
   else if (ConstantCoefficient* cQ =
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
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation& T = *fes.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            C0(q,e) = coeff0->Eval(T, ir.IntPoint(q));
         }
      }
   }

   if (coeff0)
   {
      MFEM_VERIFY(nodes0, "nodes0 has not been set!");
      EnableLimitingPA(*nodes0);
   }
}

void TMOP_Integrator::AssembleGradDiagonalPA(Vector &de) const
{
   MFEM_VERIFY(PA.R, "PA extension setup has not been done!");

   if (PA.dim == 2)
   {
      AssembleDiagonalPA_2D(de);
      if (coeff0) { AssembleDiagonalPA_C0_2D(de); }
   }
   else if (PA.dim == 3)
   {
      AssembleDiagonalPA_3D(de);
      if (coeff0) { AssembleDiagonalPA_C0_3D(de); }
   }
   else
   {
      MFEM_ABORT("3D diagonal computation is WIP.");
   }
}

void TMOP_Integrator::AddMultPA(const Vector &xe, Vector &ye) const
{
   if (!PA.setup_Jtr) { ComputeElementTargetsPA(); }

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
   if (!PA.setup_Jtr) { ComputeElementTargetsPA(); }

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
   double energy = 0.0;

   ComputeElementTargetsPA(xe);

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
