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

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "../ceed/integrators/interp/interp.hpp"

namespace mfem
{

void CurlInterpolator::AssemblePA(const FiniteElementSpace &trial_fes,
                                  const FiniteElementSpace &test_fes)
{
   Mesh *mesh = trial_fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      ceedOp = new ceed::PADiscreteInterpolator(*this, trial_fes, test_fes);
      return;
   }

   // Assumes tensor-product elements, with a vector test space and H^1 trial space.
   // const FiniteElement *trial_fel = trial_fes.GetFE(0);
   // const FiniteElement *test_fel = test_fes.GetFE(0);
   MFEM_ABORT("Error: CurlInterpolator::AssemblePA only implemented with libCEED");
}

void CurlInterpolator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      MFEM_ABORT("Error: CurlInterpolator::AddMultPA only implemented with"
                 " libCEED");
   }
}

void CurlInterpolator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMultTranspose(x, y);
   }
   else
   {
      MFEM_ABORT("Error: CurlInterpolator::AddMultTransposePA only implemented"
                 "with libCEED");
   }
}

} // namespace mfem
