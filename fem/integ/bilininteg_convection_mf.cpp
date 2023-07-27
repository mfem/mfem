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
#include "../ceed/integrators/convection/convection.hpp"

namespace mfem
{

void ConvectionIntegrator::AssembleMF(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      ceedOp = new ceed::MFConvectionIntegrator(*this, fes, Q, alpha);
      return;
   }

   // Assuming the same element type
   // const FiniteElement &el = *fes.GetFE(0);
   // ElementTransformation &T = *fes.GetElementTransformation(0);
   // const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   MFEM_ABORT("Error: ConvectionIntegrator::AssembleMF only implemented with"
              " libCEED");
}

void ConvectionIntegrator::AssembleMFBoundary(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNBE() == 0) { return; }
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      ceedOp = new ceed::MFConvectionIntegrator(*this, fes, Q, alpha, true);
      return;
   }

   // Assuming the same element type
   // const FiniteElement &el = *fes.GetBE(0);
   // ElementTransformation &T = *fes.GetBdrElementTransformation(0);
   // const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   MFEM_ABORT("Error: ConvectionIntegrator::AssembleMFBoundary only implemented with"
              " libCEED");
}

void ConvectionIntegrator::AssembleDiagonalMF(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      if (ceedOp) { ceedOp->GetDiagonal(diag); }
   }
   else
   {
      MFEM_ABORT("Error: ConvectionIntegrator::AssembleDiagonalMF only"
                 " implemented with libCEED");
   }
}

void ConvectionIntegrator::AddMultMF(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      if (ceedOp) { ceedOp->AddMult(x, y); }
   }
   else
   {
      MFEM_ABORT("Error: ConvectionIntegrator::AddMultMF only implemented with"
                 " libCEED");
   }
}

}
