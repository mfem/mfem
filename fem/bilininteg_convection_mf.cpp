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

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "ceed/convection.hpp"

using namespace std;

namespace mfem
{

void ConvectionIntegrator::AssembleMF(const FiniteElementSpace &fes)
{
   // Assuming the same element type
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation &Trans = *fes.GetElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, Trans);
   if (DeviceCanUseCeed())
   {
      MFEM_VERIFY(alpha==-1, "Only alpha=-1 currently supported with libCEED.");
      delete ceedOp;
      ceedOp = new ceed::MFConvectionIntegrator(fes, *ir, Q, alpha);
      return;
   }
   MFEM_ABORT("Error: ConvectionIntegrator::AssembleMF only implemented with"
              " libCEED");
}

void ConvectionIntegrator::AssembleDiagonalMF(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
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
      ceedOp->AddMult(x, y);
   }
   else
   {
      MFEM_ABORT("Error: ConvectionIntegrator::AddMultMF only implemented with"
                 " libCEED");
   }
}

}
