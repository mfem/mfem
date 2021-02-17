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

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "ceed/diffusion.hpp"

using namespace std;

namespace mfem
{

void VectorDiffusionIntegrator::AssembleMF(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir
      = IntRule ? IntRule : &DiffusionIntegrator::GetRule(el, el);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      ceedOp = new ceed::MFDiffusionIntegrator(fes, *ir, Q);
      return;
   }
   MFEM_ABORT("Error: VectorDiffusionIntegrator::AssembleMF only implemented"
              " with libCEED");
}

void VectorDiffusionIntegrator::AddMultMF(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      MFEM_ABORT("Error: VectorDiffusionIntegrator::AddMultMF only implemented"
                 " with libCEED");
   }
}

void VectorDiffusionIntegrator::AssembleDiagonalMF(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else
   {
      MFEM_ABORT("Error: VectorDiffusionIntegrator::AssembleDiagonalMF only"
                 " implemented with libCEED");
   }
}

} // namespace mfem
