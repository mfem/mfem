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

#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../ceed/integrators/diffusion/diffusion.hpp"

namespace mfem
{

void VectorDiffusionIntegrator::AssembleMF(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      MFEM_VERIFY(!VQ && !MQ,
                  "Only scalar coefficient is supported for matrix-free assembly for VectorDiffusionIntegrator");
      delete ceedOp;
      ceedOp = new ceed::MFDiffusionIntegrator(*this, fes, Q);
      return;
   }

   // Assumes tensor-product elements
   // const FiniteElement &el = *fes.GetFE(0);
   // ElementTransformation &T = *mesh->GetElementTransformation(0);
   // const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   MFEM_ABORT("Error: VectorDiffusionIntegrator::AssembleMF only implemented"
              " with libCEED");
}

void VectorDiffusionIntegrator::AssembleMFBoundary(
   const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNBE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      MFEM_VERIFY(!VQ && !MQ,
                  "Only scalar coefficient is supported for matrix-free assembly for VectorDiffusionIntegrator");
      delete ceedOp;
      ceedOp = new ceed::MFDiffusionIntegrator(*this, fes, Q, true);
      return;
   }

   // Assumes tensor-product elements
   // const FiniteElement &el = *fes.GetBE(0);
   // ElementTransformation &T = *mesh->GetBdrElementTransformation(0);
   // const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   MFEM_ABORT("Error: VectorDiffusionIntegrator::AssembleMFBoundary only implemented"
              " with libCEED");
}

void VectorDiffusionIntegrator::AssembleDiagonalMF(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      if (ceedOp) { ceedOp->GetDiagonal(diag); }
   }
   else
   {
      MFEM_ABORT("Error: VectorDiffusionIntegrator::AssembleDiagonalMF only"
                 " implemented with libCEED");
   }
}

void VectorDiffusionIntegrator::AddMultMF(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      if (ceedOp) { ceedOp->AddMult(x, y); }
   }
   else
   {
      MFEM_ABORT("Error: VectorDiffusionIntegrator::AddMultMF only implemented"
                 " with libCEED");
   }
}

} // namespace mfem
