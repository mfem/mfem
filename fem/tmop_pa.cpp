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

void TMOP_Integrator::AssemblePA(const FiniteElementSpace &fes)
{
   MFEM_VERIFY(IntRule,"");
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");

   PA.fes = &fes;
   Mesh *mesh = fes.GetMesh();
   const int dim = PA.dim = mesh->Dimension();
   const int nq = PA.nq = IntRule->GetNPoints();
   const int ne = PA.ne = fes.GetMesh()->GetNE();
   const IntegrationRule &ir = *IntRule;
   PA.maps = &fes.GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   PA.geom = mesh->GetGeometricFactors(ir, GeometricFactors::JACOBIANS);

   // Energy & One vectors
   PA.E.UseDevice(true);
   PA.E.SetSize(ne*nq, Device::GetDeviceMemoryType());

   PA.O.UseDevice(true);
   PA.O.SetSize(ne*nq, Device::GetDeviceMemoryType());
   PA.O = 1.0;

   PA.setup = false;
   PA.H.UseDevice(true);
   PA.H.SetSize(dim*dim * dim*dim * nq*ne, Device::GetDeviceMemoryType());
   // H0 for coeff0
   PA.H0.UseDevice(true);
   PA.H0.SetSize(dim*dim * dim*dim * nq*ne, Device::GetDeviceMemoryType());

   // X gradient vector
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   PA.R = fes.GetElementRestriction(ordering);
   MFEM_VERIFY(PA.R, "Not yet implemented!");

   // TargetConstructor TargetType setup
   const TargetConstructor::TargetType &target_type = targetC->Type();
   MFEM_VERIFY(target_type == TargetConstructor::IDEAL_SHAPE_UNIT_SIZE ||
               target_type == TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE ||
               target_type == TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE, "");
   const int NE = mesh->GetNE();
   const int NQ = ir.GetNPoints();
   PA.Jtr.SetSize(dim, dim, NE*NQ);
   PA.Jtr.HostWrite();
   for (int e = 0; e < NE; e++)
   {
      const FiniteElement *fe = fes.GetFE(e);
      DenseTensor Jtr(dim, dim, NQ);
      targetC->ComputeElementTargets(e, *fe, ir, Vector(), Jtr);
      for (int q = 0; q < NQ; q++) { PA.Jtr(e*NQ+q) = Jtr(q); }
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
            C0(q,e) = coeff0->Eval(T, ir.IntPoint(q));
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
   if (!PA.setup)
   {
      PA.setup = true;
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
   MFEM_VERIFY(PA.dim == 2 || PA.dim ==3, "PA setup has not been done!");

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
   return energy;
}

} // namespace mfem
