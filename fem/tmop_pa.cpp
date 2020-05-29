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

void TMOP_Integrator::AssemblePA(const FiniteElementSpace &fespace)
{
   fes = &fespace;
   MFEM_ASSERT(fes->GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");
   Mesh *mesh = fes->GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(IntRule,"");
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   nq = IntRule->GetNPoints();
   ne = fes->GetMesh()->GetNE();
   const IntegrationRule &ir = *IntRule;
   maps = &fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   geom = mesh->GetGeometricFactors(ir, GeometricFactors::JACOBIANS);

   // Energy, One & X vectors
   Epa.UseDevice(true);
   Epa.SetSize(ne * nq, Device::GetDeviceMemoryType());

   Opa.UseDevice(true);
   Opa.SetSize(ne * nq, Device::GetDeviceMemoryType());

   const int dim_d = dim == 2 ? dim*dim :
                     dim == 3 ? dim*dim*dim : -1;
   Xpa.UseDevice(true);
   Xpa.SetSize(dim_d * nq * ne, Device::GetDeviceMemoryType());
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   elem_restrict_lex = fes->GetElementRestriction(ordering);
   if (elem_restrict_lex)
   {
      Xpa.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
   }
   else
   {
      MFEM_ABORT("Not yet implemented!");
   }

   Dpa.UseDevice(true);
   Dpa.SetSize(dim_d * nq * ne, Device::GetDeviceMemoryType());

   setup = false;
   dPpa.UseDevice(true);
   dPpa.SetSize(dim_d * dim_d * nq * ne, Device::GetDeviceMemoryType());
}

void TMOP_Integrator::AddMultPA(const Vector &X, Vector &Y) const
{
   if (dim == 2) { return AddMultPA_2D(X,Y); }
   if (dim == 3) { return AddMultPA_3D(X,Y); }
   MFEM_ABORT("Not yet implemented!");
}

void TMOP_Integrator::AssembleGradPA(const DenseMatrix &Jtr,
                                     const Vector &Xe) const
{
   if (dim == 2) { return AssembleGradPA_2D(Jtr,Xe); }
   if (dim == 3) { return AssembleGradPA_3D(Jtr,Xe); }
   MFEM_ABORT("Not yet implemented!");
}

void TMOP_Integrator::AddMultGradPA(const Vector &Xe, const Vector &Re,
                                    Vector &Ce) const
{
   if (dim == 2) { return AddMultGradPA_2D(Xe,Re,Ce); }
   if (dim == 3) { return AddMultGradPA_3D(Xe,Re,Ce); }
   MFEM_ABORT("Not yet implemented!");
}

double TMOP_Integrator::GetGridFunctionEnergyPA(const FiniteElementSpace &fes,
                                                const Vector &x) const
{
   dbg("");
   if (dim == 2) { return GetGridFunctionEnergyPA_2D(fes,x); }
   if (dim == 3) { return GetGridFunctionEnergyPA_3D(fes,x); }
   MFEM_ABORT("Not yet implemented!");
   return 0.0;
}

} // namespace mfem
