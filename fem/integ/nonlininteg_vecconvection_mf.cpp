// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
               "PA Only supports Ordering::byNODES!");
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation &T = *mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed)
      {
         ceedOp = new ceed::MixedMFVectorConvectionNLIntegrator(*this, fes, Q);
      }
      else
      {
         ceedOp = new ceed::MFVectorConvectionNLFIntegrator(fes, *ir, Q);
      }
      return;
   }
   MFEM_ABORT("Not yet implemented.");
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
