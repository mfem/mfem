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
#include "../ceed/integrators/curlcurl/curlcurl.hpp"

using namespace std;

namespace mfem
{

void CurlCurlIntegrator::AssembleMF(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      if (MQ) { ceedOp = new ceed::MFCurlCurlIntegrator(*this, fes, MQ); }
      else if (DQ) { ceedOp = new ceed::MFCurlCurlIntegrator(*this, fes, DQ); }
      else { ceedOp = new ceed::MFCurlCurlIntegrator(*this, fes, Q); }
      return;
   }

   // Assumes tensor-product elements
   // const FiniteElement &el = *fes.GetFE(0);
   // ElementTransformation &T = *mesh->GetElementTransformation(0);
   // const IntegrationRule *ir = IntRule ? IntRule : &GetRule(*el, T);
   MFEM_ABORT("Error: CurlCurlIntegrator::AssembleMF only implemented with"
              " libCEED");
}

void CurlCurlIntegrator::AssembleMFBoundary(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNBE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      if (MQ) { ceedOp = new ceed::MFCurlCurlIntegrator(*this, fes, MQ, true); }
      else if (DQ) { ceedOp = new ceed::MFCurlCurlIntegrator(*this, fes, DQ, true); }
      else { ceedOp = new ceed::MFCurlCurlIntegrator(*this, fes, Q, true); }
      return;
   }

   // Assumes tensor-product elements
   // const FiniteElement &el = *fes.GetBE(0);
   // ElementTransformation &T = *mesh->GetBdrElementTransformation(0);
   // const IntegrationRule *ir = IntRule ? IntRule : &GetRule(*el, T);
   MFEM_ABORT("Error: CurlCurlIntegrator::AssembleMFBoundary only implemented with"
              " libCEED");
}

void CurlCurlIntegrator::AssembleDiagonalMF(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      if (ceedOp) { ceedOp->GetDiagonal(diag); }
   }
   else
   {
      MFEM_ABORT("Error: CurlCurlIntegrator::AssembleDiagonalMF only"
                 " implemented with libCEED");
   }
}

void CurlCurlIntegrator::AddMultMF(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      if (ceedOp) { ceedOp->AddMult(x, y); }
   }
   else
   {
      MFEM_ABORT("Error: CurlCurlIntegrator::AddMultMF only implemented with"
                 " libCEED");
   }
}

}
