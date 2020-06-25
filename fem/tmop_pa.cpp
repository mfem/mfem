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

static void ScaleByWeight(const Vector &w, Vector &x)
{
   MFEM_VERIFY(w.Size() == x.Size(), "Size error!");
   const int N = w.Size();
   const auto W = Reshape(w.Read(), N);
   auto X = Reshape(x.ReadWrite(), N);
   MFEM_FORALL(i, N, X(i) /= W(i););
}

// Use TargetConstructor::ComputeElementTargets to fill the PA.Jtr
static void SetupJtr(const int dim, const int NE, const int NQ,
                     const FiniteElementSpace *fes, const IntegrationRule *ir,
                     const TargetConstructor::TargetType &target_type,
                     const TargetConstructor *targetC,
                     const Vector &x, DenseTensor &Jtr)
{
   Vector elfun;
   Array<int> vdofs;
   for (int e = 0; e < NE; e++)
   {
      const FiniteElement *fe = fes->GetFE(e);
      if (target_type == TargetConstructor::GIVEN_FULL)
      {
         fes->GetElementVDofs(e, vdofs);
         x.GetSubVector(vdofs, elfun);
      }
      DenseTensor J(dim, dim, NQ);
      targetC->ComputeElementTargets(e, *fe, *ir, elfun, J);
      for (int q = 0; q < NQ; q++) { Jtr(e*NQ+q) = J(q); }
   }

}

void TMOP_Integrator::AssemblePA(const FiniteElementSpace &fes)
{
   const IntegrationRule *ir = EnergyIntegrationRule(*fes.GetFE(0));
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");

   PA.fes = &fes;
   Mesh *mesh = fes.GetMesh();
   const int dim = PA.dim = mesh->Dimension();
   const int nq = PA.nq = ir->GetNPoints();
   const int ne = PA.ne = fes.GetMesh()->GetNE();
   PA.maps = &fes.GetFE(0)->GetDofToQuad(*ir, DofToQuad::TENSOR);
   PA.geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);

   // Energy vector
   PA.E.UseDevice(true);
   PA.E.SetSize(ne*nq, Device::GetDeviceMemoryType());

   // Setup initialization
   PA.setup_Grad = false;
   PA.setup_Jtr = false;

   // H for Grad
   PA.H.UseDevice(true);
   PA.H.SetSize(dim*dim * dim*dim * nq*ne, Device::GetDeviceMemoryType());
   // H0 for coeff0
   PA.H0.UseDevice(true);
   PA.H0.SetSize(dim*dim * dim*dim * nq*ne, Device::GetDeviceMemoryType());

   // Restriction setup
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   PA.R = fes.GetElementRestriction(ordering);
   MFEM_VERIFY(PA.R, "Not yet implemented!");

   // Weight of the R^t
   PA.W.SetSize(PA.R->Width(), Device::GetDeviceMemoryType());
   PA.W.UseDevice(true);
   PA.O.UseDevice(true);
   PA.O.SetSize(dim*ne*nq, Device::GetDeviceMemoryType());
   PA.O = 1.0;
   PA.R->MultTranspose(PA.O, PA.W);

   // Scalar vector of '1'
   PA.O.SetSize(ne*nq, Device::GetDeviceMemoryType());
   PA.O = 1.0;

   // TargetConstructor TargetType setup
   const TargetConstructor::TargetType &target_type = targetC->Type();
   MFEM_VERIFY(target_type == TargetConstructor::IDEAL_SHAPE_UNIT_SIZE ||
               target_type == TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE ||
               target_type == TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE ||
               target_type == TargetConstructor::GIVEN_SHAPE_AND_SIZE ||
               target_type == TargetConstructor::GIVEN_FULL, "");
   const int NE = mesh->GetNE();
   const int NQ = ir->GetNPoints();
   PA.Jtr.SetSize(dim, dim, NE*NQ);
   PA.Jtr.HostWrite();
   const bool datc = dynamic_cast<const DiscreteAdaptTC*>(targetC) != nullptr;
   if (!datc && target_type < TargetConstructor::GIVEN_FULL)
   {
      PA.setup_Jtr = true;
      Vector x;
      SetupJtr(dim, NE, NQ, PA.fes, ir, target_type, targetC, x, PA.Jtr);
   }

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
      PA.C0.SetSize(NQ * NE, Device::GetMemoryType());
      auto C0 = Reshape(PA.C0.HostWrite(), NQ, NE);
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

      // lim_dist
      MFEM_VERIFY(lim_dist, "No lim_dist!")
   }
}

void TMOP_Integrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (!PA.setup_Jtr)
   {
      MFEM_ABORT("Should be inited!");
      Vector X(PA.R->Width(), Device::GetMemoryType());
      X.UseDevice(true);
      PA.R->MultTranspose(x,X);
      ScaleByWeight(PA.W, X);
      const TargetConstructor::TargetType &target_type = targetC->Type();
      MFEM_VERIFY(target_type == TargetConstructor::GIVEN_FULL, "");
      const int dim = PA.dim;
      const int NE = PA.ne;
      const int NQ = PA.nq;
      PA.Jtr.SetSize(dim, dim, NE*NQ);
      PA.Jtr.HostWrite();
      const IntegrationRule *ir = EnergyIntegrationRule(*PA.fes->GetFE(0));
      SetupJtr(dim, NE, NQ, PA.fes, ir, target_type, targetC, X, PA.Jtr);
      PA.setup_Jtr = true;
   }

   if (PA.dim == 2)
   {
      AddMultPA_2D(x,y);
      if (coeff0) { AddMultPA_C0_2D(x,y); }
      return;
   }
   if (PA.dim == 3)
   {
      AddMultPA_3D(x,y);
      if (coeff0) { AddMultPA_C0_3D(x,y); }
      return;
   }
   MFEM_ABORT("Not yet implemented!");
}

void TMOP_Integrator::AddMultGradPA(const Vector &x,
                                    const Vector &r, Vector &c) const
{
   if (!PA.setup_Jtr)
   {
      Vector X(PA.R->Width(), Device::GetMemoryType());
      X.UseDevice(true);
      PA.R->MultTranspose(x,X);
      ScaleByWeight(PA.W, X);
      const TargetConstructor::TargetType &target_type = targetC->Type();
      const int dim = PA.dim;
      const int NE = PA.ne;
      const int NQ = PA.nq;
      PA.Jtr.SetSize(dim, dim, NE*NQ);
      PA.Jtr.HostWrite();
      const IntegrationRule *ir = EnergyIntegrationRule(*PA.fes->GetFE(0));
      SetupJtr(dim, NE, NQ, PA.fes, ir, target_type, targetC, X, PA.Jtr);
      PA.setup_Jtr = true;
   }

   if (!PA.setup_Grad)
   {
      PA.setup_Grad = true;
      if (PA.dim == 2)
      {
         AssembleGradPA_2D(x);
         //if (coeff0) { AssembleGradPA_C0_2D(x); }
      }
      if (PA.dim == 3)
      {
         AssembleGradPA_3D(x);
         //if (coeff0) { AssembleGradPA_C0_3D(x); }
      }
   }

   if (PA.dim == 2)
   {
      AddMultGradPA_2D(r,c);
      if (coeff0) { AddMultGradPA_C0_2D(x,r,c); }
      return;
   }

   if (PA.dim == 3)
   {
      AddMultGradPA_3D(x,r,c);
      if (coeff0) { AddMultGradPA_C0_3D(x,r,c); }
      return;
   }
   MFEM_ABORT("Not yet implemented!");
}

double TMOP_Integrator::GetGridFunctionEnergyPA(const Vector &x) const
{
   MFEM_VERIFY(PA.dim == 2 || PA.dim == 3, "PA setup has not been done!");

   const bool datc = dynamic_cast<const DiscreteAdaptTC*>(targetC) != nullptr;
   if (datc || targetC->Type() == TargetConstructor::GIVEN_FULL)
   {
      Vector X(PA.R->Width(), Device::GetMemoryType());
      X.UseDevice(true);
      PA.R->MultTranspose(x, X);
      ScaleByWeight(PA.W, X);
      const TargetConstructor::TargetType &target_type = targetC->Type();
      const int dim = PA.dim;
      const int NE = PA.ne;
      const int NQ = PA.nq;
      PA.Jtr.SetSize(dim, dim, NE*NQ);
      PA.Jtr.HostWrite();
      const IntegrationRule *ir = EnergyIntegrationRule(*PA.fes->GetFE(0));
      SetupJtr(dim, NE, NQ, PA.fes, ir, target_type, targetC, X, PA.Jtr);
   }

   double energy = 0.0;

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

   PA.setup_Jtr = true;
   return energy;
}

} // namespace mfem
