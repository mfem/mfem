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

   // P (AddMultPA) & A (AddMultGradPA) vectors
   PA.P.UseDevice(true);
   PA.P.SetSize(dim*dim * nq*ne, Device::GetDeviceMemoryType());

   PA.setup = false;
   PA.A.UseDevice(true);
   PA.A.SetSize(dim*dim * dim*dim * nq*ne, Device::GetDeviceMemoryType());

   // X gradient vector
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   PA.elem_restrict_lex = fes.GetElementRestriction(ordering);
   MFEM_VERIFY(PA.elem_restrict_lex, "Not yet implemented!");
   PA.X.SetSize(PA.elem_restrict_lex->Height(), Device::GetMemoryType());
   PA.X.UseDevice(true);

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
}

void TMOP_Integrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (PA.dim == 2) { return AddMultPA_2D(x,y); }
   if (PA.dim == 3) { return AddMultPA_3D(x,y); }
   MFEM_ABORT("Not yet implemented!");
}

void TMOP_Integrator::AssembleGradPA(const Vector &x) const
{
   if (PA.dim == 2) { return AssembleGradPA_2D(x); }
   if (PA.dim == 3) { return AssembleGradPA_3D(x); }
   MFEM_ABORT("Not yet implemented!");
}

void TMOP_Integrator::AddMultGradPA(const Vector &x, const Vector &r,
                                    Vector &c) const
{
   if (PA.dim == 2) { return AddMultGradPA_2D(x,r,c); }
   if (PA.dim == 3) { return AddMultGradPA_3D(x,r,c); }
   MFEM_ABORT("Not yet implemented!");
}

double TMOP_Integrator::GetGridFunctionEnergyPA(const Vector &x) const
{
   if (PA.dim == 2) { return GetGridFunctionEnergyPA_2D(x); }
   if (PA.dim == 3) { return GetGridFunctionEnergyPA_3D(x); }
   MFEM_ABORT("Not yet implemented!");
   return 0.0;
}

} // namespace mfem
