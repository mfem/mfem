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
#define MFEM_DBG_COLOR 123
#include "../general/dbg.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

void TMOP_Integrator::AssemblePA(const FiniteElementSpace &fes)
{
   PA.fes = &fes;
   MFEM_ASSERT(fes->GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");
   Mesh *mesh = fes.GetMesh();
   const int dim = PA.dim = mesh->Dimension();
   MFEM_VERIFY(IntRule,"");
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   const int nq = PA.nq = IntRule->GetNPoints();
   const int ne = PA.ne = fes.GetMesh()->GetNE();
   const IntegrationRule &ir = *IntRule;
   PA.maps = &fes.GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   PA.geom = mesh->GetGeometricFactors(ir, GeometricFactors::JACOBIANS);

   // Energy, One & X vectors
   PA.E.UseDevice(true);
   PA.E.SetSize(ne * nq, Device::GetDeviceMemoryType());

   PA.O.UseDevice(true);
   PA.O.SetSize(ne * nq, Device::GetDeviceMemoryType());

   PA.X.UseDevice(true);
   PA.X.SetSize(dim*dim * nq * ne, Device::GetDeviceMemoryType());
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   PA.elem_restrict_lex = fes.GetElementRestriction(ordering);
   if (PA.elem_restrict_lex)
   {
      PA.X.SetSize(PA.elem_restrict_lex->Height(), Device::GetMemoryType());
   }
   else
   {
      MFEM_ABORT("Not yet implemented!");
   }

   PA.P.UseDevice(true);
   PA.P.SetSize(dim*dim * nq * ne, Device::GetDeviceMemoryType());

   PA.setup = false;
   PA.A.UseDevice(true);
   PA.A.SetSize(dim*dim * dim*dim * nq * ne, Device::GetDeviceMemoryType());
}

void TMOP_Integrator::AddMultPA(const Vector &Xe, Vector &Ye) const
{
   if (PA.dim == 2) { return AddMultPA_2D(Xe,Ye); }
   if (PA.dim == 3) { return AddMultPA_3D(Xe,Ye); }
   MFEM_ABORT("Not yet implemented!");
}

void TMOP_Integrator::AssembleGradPA(const DenseMatrix &Jtr,
                                     const Vector &Xe) const
{
   if (PA.dim == 2) { return AssembleGradPA_2D(Jtr,Xe); }
   if (PA.dim == 3) { return AssembleGradPA_3D(Jtr,Xe); }
   MFEM_ABORT("Not yet implemented!");
}

void TMOP_Integrator::AddMultGradPA(const Vector &Xe, const Vector &Re,
                                    Vector &Ce) const
{
   if (PA.dim == 2) { return AddMultGradPA_2D(Xe,Re,Ce); }
   if (PA.dim == 3) { return AddMultGradPA_3D(Xe,Re,Ce); }
   MFEM_ABORT("Not yet implemented!");
}

double TMOP_Integrator::GetGridFunctionEnergyPA(const FiniteElementSpace &fes,
                                                const Vector &x) const
{
   if (PA.dim == 2) { return GetGridFunctionEnergyPA_2D(fes,x); }
   if (PA.dim == 3) { return GetGridFunctionEnergyPA_3D(fes,x); }
   MFEM_ABORT("Not yet implemented!");
   return 0.0;
}

} // namespace mfem
