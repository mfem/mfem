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
#include "../ceed/integrators/mixedvecgrad/mixedvecgrad.hpp"

namespace mfem
{

void MixedVectorGradientIntegrator::AssembleMF(
   const FiniteElementSpace &trial_fes,
   const FiniteElementSpace &test_fes)
{
   Mesh *mesh = trial_fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      if (MQ)
      {
         ceedOp = new ceed::MFMixedVectorGradientIntegrator(*this, trial_fes,
                                                            test_fes, MQ);
      }
      else if (VQ)
      {
         ceedOp = new ceed::MFMixedVectorGradientIntegrator(*this, trial_fes,
                                                            test_fes, VQ);
      }
      else
      {
         ceedOp = new ceed::MFMixedVectorGradientIntegrator(*this, trial_fes,
                                                            test_fes, Q);
      }
      return;
   }

   // Assuming the same element type
   MFEM_ABORT("Error: MixedVectorGradientIntegrator::AssembleMF only"
              " implemented with libCEED");
}

void MixedVectorGradientIntegrator::AssembleMFBoundary(
   const FiniteElementSpace &trial_fes,
   const FiniteElementSpace &test_fes)
{
   Mesh *mesh = trial_fes.GetMesh();
   if (mesh->GetNBE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      if (MQ)
      {
         ceedOp = new ceed::MFMixedVectorGradientIntegrator(*this, trial_fes,
                                                            test_fes, MQ, true);
      }
      else if (VQ)
      {
         ceedOp = new ceed::MFMixedVectorGradientIntegrator(*this, trial_fes,
                                                            test_fes, VQ, true);
      }
      else
      {
         ceedOp = new ceed::MFMixedVectorGradientIntegrator(*this, trial_fes,
                                                            test_fes, Q, true);
      }
      return;
   }

   // Assuming the same element type
   MFEM_ABORT("Error: MixedVectorGradientIntegrator::AssembleMFBoundary only"
              " implemented with libCEED");
}

void MixedVectorGradientIntegrator::AddMultMF(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      MFEM_ABORT("Error: MixedVectorGradientIntegrator::AddMultMF only"
                 " implemented with libCEED");
   }
}

void MixedVectorWeakDivergenceIntegrator::AssembleMF(
   const FiniteElementSpace &trial_fes,
   const FiniteElementSpace &test_fes)
{
   Mesh *mesh = trial_fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      if (MQ)
      {
         ceedOp = new ceed::MFMixedVectorWeakDivergenceIntegrator(*this, trial_fes,
                                                                  test_fes, MQ);
      }
      else if (VQ)
      {
         ceedOp = new ceed::MFMixedVectorWeakDivergenceIntegrator(*this, trial_fes,
                                                                  test_fes, VQ);
      }
      else
      {
         ceedOp = new ceed::MFMixedVectorWeakDivergenceIntegrator(*this, trial_fes,
                                                                  test_fes, Q);
      }
      return;
   }

   // Assuming the same element type
   MFEM_ABORT("Error: MixedVectorWeakDivergenceIntegrator::AssembleMF only"
              " implemented with libCEED");
}

void MixedVectorWeakDivergenceIntegrator::AssembleMFBoundary(
   const FiniteElementSpace &trial_fes,
   const FiniteElementSpace &test_fes)
{
   Mesh *mesh = trial_fes.GetMesh();
   if (mesh->GetNBE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      if (MQ)
      {
         ceedOp = new ceed::MFMixedVectorWeakDivergenceIntegrator(*this, trial_fes,
                                                                  test_fes, MQ, true);
      }
      else if (VQ)
      {
         ceedOp = new ceed::MFMixedVectorWeakDivergenceIntegrator(*this, trial_fes,
                                                                  test_fes, VQ, true);
      }
      else
      {
         ceedOp = new ceed::MFMixedVectorWeakDivergenceIntegrator(*this, trial_fes,
                                                                  test_fes, Q, true);
      }
      return;
   }

   // Assuming the same element type
   MFEM_ABORT("Error: MixedVectorWeakDivergenceIntegrator::AssembleMFBoundary only"
              " implemented with libCEED");
}

void MixedVectorWeakDivergenceIntegrator::AddMultMF(const Vector &x,
                                                    Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      MFEM_ABORT("Error: MixedVectorWeakDivergenceIntegrator::AddMultMF only"
                 " implemented with libCEED");
   }
}

} // namespace mfem
