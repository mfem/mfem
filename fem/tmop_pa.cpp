// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

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
void TMOP_Integrator::ComputeElementTargetsPA(const Vector &x) const
{
   PA.Jtr.HostWrite();

   const int NE = PA.ne;
   const int NQ = PA.nq;
   const int dim = PA.dim;
   DenseTensor &Jtr = PA.Jtr;
   const FiniteElementSpace *fes = PA.fes;

   const TargetConstructor::TargetType &target_type = targetC->Type();
   const IntegrationRule &ir = *EnergyIntegrationRule(*PA.fes->GetFE(0));

   Vector xe;
   const bool useable_input_vector = x.Size() > 0;
   const bool use_input_vector = target_type == TargetConstructor::GIVEN_FULL;

   if (use_input_vector && !useable_input_vector) { return; }

   if (use_input_vector)
   {
      xe.SetSize(PA.R->Width(), Device::GetMemoryType());
      xe.UseDevice(true);
      PA.R->MultTranspose(x, xe);
      // Scale by weights
      const int N = PA.W.Size();
      const auto W = Reshape(PA.W.Read(), N);
      auto X = Reshape(xe.ReadWrite(), N);
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
         xe.GetSubVector(vdofs, elfun);
      }
      J.UseExternalData(Jtr(e*NQ).Data(), dim, dim, NQ);
      targetC->ComputeElementTargets(e, fe, ir, elfun, J);
   }
   PA.setup_Jtr = true;
}

void TMOP_Integrator::AssemblePA(const FiniteElementSpace &fes)
{
   const IntegrationRule *ir = EnergyIntegrationRule(*fes.GetFE(0));
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");

   PA.fes = &fes;
   Mesh *mesh = fes.GetMesh();
   const int nq = PA.nq = ir->GetNPoints();
   const int ne = PA.ne = fes.GetMesh()->GetNE();
   const int dim = PA.dim = mesh->Dimension();
   MFEM_VERIFY(PA.dim == 2 || PA.dim == 3, "Not yet implemented!");
   PA.maps = &fes.GetFE(0)->GetDofToQuad(*ir, DofToQuad::TENSOR);
   PA.geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);

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
            C0(q,e) = coeff0->Eval(T, ir->IntPoint(q));
         }
      }
   }

   // Coeff0 PA.X0
   if (coeff0)
   {
      // Nodes0
      MFEM_VERIFY(nodes0, "No nodes0!")
      PA.X0.SetSize(PA.R->Height(), Device::GetMemoryType());
      PA.X0.UseDevice(true);
      PA.R->Mult(*nodes0, PA.X0);

      // lim_dist & lim_func checks
      MFEM_VERIFY(lim_dist, "No lim_dist!")
      PA.LD.SetSize(PA.R->Height(), Device::GetMemoryType());
      PA.LD.UseDevice(true);
      PA.R->Mult(*lim_dist, PA.LD);

      // Only TMOP_QuadraticLimiter is supported
      MFEM_VERIFY(lim_func, "No lim_func!")
      MFEM_VERIFY(dynamic_cast<TMOP_QuadraticLimiter*>(lim_func),
                  "Only TMOP_QuadraticLimiter is supported");
   }
}

void TMOP_Integrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (!PA.setup_Jtr) { ComputeElementTargetsPA(); }

   if (PA.dim == 2)
   {
      AddMultPA_2D(x,y);
      if (coeff0) { AddMultPA_C0_2D(x,y); }
   }

   if (PA.dim == 3)
   {
      AddMultPA_3D(x,y);
      if (coeff0) { AddMultPA_C0_3D(x,y); }
   }
}

void TMOP_Integrator::AddMultGradPA(const Vector &x,
                                    const Vector &r, Vector &c) const
{
   if (!PA.setup_Jtr) { ComputeElementTargetsPA(x); }

   if (!PA.setup_Grad)
   {
      PA.setup_Grad = true;

      if (PA.dim == 2)
      {
         AssembleGradPA_2D(x);
         if (coeff0) { AssembleGradPA_C0_2D(x); }
      }

      if (PA.dim == 3)
      {
         AssembleGradPA_3D(x);
         if (coeff0) { AssembleGradPA_C0_3D(x); }
      }
   }

   if (PA.dim == 2)
   {
      AddMultGradPA_2D(r,c);
      if (coeff0) { AddMultGradPA_C0_2D(x,r,c); }
   }

   if (PA.dim == 3)
   {
      AddMultGradPA_3D(x,r,c);
      if (coeff0) { AddMultGradPA_C0_3D(x,r,c); }
   }
}

double TMOP_Integrator::GetGridFunctionEnergyPA(const Vector &x) const
{
   double energy = 0.0;

   ComputeElementTargetsPA(x);

   if (PA.dim == 2)
   {
      energy = GetGridFunctionEnergyPA_2D(x);
      if (coeff0) { energy += GetGridFunctionEnergyPA_C0_2D(x); }
   }

   if (PA.dim == 3)
   {
      energy = GetGridFunctionEnergyPA_3D(x);
      if (coeff0) { energy += GetGridFunctionEnergyPA_C0_3D(x); }
   }

   return energy;
}

} // namespace mfem
