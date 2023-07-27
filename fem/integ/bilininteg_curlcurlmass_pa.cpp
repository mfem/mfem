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
#include "../ceed/integrators/curlcurlmass/curlcurlmass.hpp"

using namespace std;

namespace mfem
{

void CurlCurlMassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      if (MQd)
      {
         if (MQm) { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, MQd, MQm); }
         else if (VQm) { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, MQd, VQm); }
         else { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, MQd, Qm); }
      }
      else if (VQd)
      {
         if (MQm) { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, VQd, MQm); }
         else if (VQm) { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, VQd, VQm); }
         else { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, VQd, Qm); }
      }
      else
      {
         if (MQm) { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, Qd, MQm); }
         else if (VQm) { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, Qd, VQm); }
         else { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, Qd, Qm); }
      }
      return;
   }

   // Assumes tensor-product elements
   // const FiniteElement &el = *fes.GetFE(0);
   // ElementTransformation &T = *mesh->GetElementTransformation(0);
   // const IntegrationRule *ir = IntRule ? IntRule : &GetRule(*el, T);
   MFEM_ABORT("Error: CurlCurlMassIntegrator::AssemblePA only implemented with"
              " libCEED");
}

void CurlCurlMassIntegrator::AssemblePABoundary(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNBE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      if (MQd)
      {
         if (MQm) { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, MQd, MQm, true); }
         else if (VQm) { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, MQd, VQm, true); }
         else { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, MQd, Qm, true); }
      }
      else if (VQd)
      {
         if (MQm) { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, VQd, MQm, true); }
         else if (VQm) { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, VQd, VQm, true); }
         else { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, VQd, Qm, true); }
      }
      else
      {
         if (MQm) { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, Qd, MQm, true); }
         else if (VQm) { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, Qd, VQm, true); }
         else { ceedOp = new ceed::PACurlCurlMassIntegrator(*this, fes, Qd, Qm, true); }
      }
      return;
   }

   // Assumes tensor-product elements
   // const FiniteElement &el = *fes.GetBE(0);
   // ElementTransformation &T = *mesh->GetBdrElementTransformation(0);
   // const IntegrationRule *ir = IntRule ? IntRule : &GetRule(*el, T);
   MFEM_ABORT("Error: CurlCurlMassIntegrator::AssemblePABoundary only implemented with"
              " libCEED");
}

void CurlCurlMassIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      if (ceedOp) { ceedOp->GetDiagonal(diag); }
   }
   else
   {
      MFEM_ABORT("Error: CurlCurlMassIntegrator::AssembleDiagonalPA only"
                 " implemented with libCEED");
   }
}

void CurlCurlMassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      if (ceedOp) { ceedOp->AddMult(x, y); }
   }
   else
   {
      MFEM_ABORT("Error: CurlCurlMassIntegrator::AddMultPA only implemented with"
                 " libCEED");
   }
}

}
