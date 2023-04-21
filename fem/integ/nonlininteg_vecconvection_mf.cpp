// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../nonlininteg.hpp"
#include "../ceed/integrators/nlconvection/nlconvection.hpp"

namespace mfem
{

void VectorConvectionNLFIntegrator::AssembleMF(const FiniteElementSpace &fes)
{
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES,
               "MF only supports Ordering::byNODES!");
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      ceedOp = new ceed::MFVectorConvectionNLIntegrator(*this, fes, Q);
      return;
   }

   // Assuming the same element type
   // const FiniteElement &el = *fes.GetFE(0);
   // ElementTransformation &T = *mesh->GetElementTransformation(0);
   // const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   MFEM_ABORT("Error: VectorConvectionNLFIntegrator::AssembleMF only"
              " implemented with libCEED");
}

void VectorConvectionNLFIntegrator::AddMultMF(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      MFEM_ABORT("Not yet implemented!");
   }
}

} // namespace mfem
