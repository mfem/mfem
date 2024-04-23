// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "../ceed/integrators/mass/mass.hpp"

namespace mfem
{

void MassIntegrator::AssembleMF(const FiniteElementSpace &fes)
{
   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation *T = mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, *T);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed)
      {
         ceedOp = new ceed::MixedMFMassIntegrator(*this, fes, Q);
      }
      else
      {
         ceedOp = new ceed::MFMassIntegrator(fes, *ir, Q);
      }
      return;
   }
   MFEM_ABORT("Error: MassIntegrator::AssembleMF only implemented with"
              " libCEED");
}

void MassIntegrator::AddMultMF(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      MFEM_ABORT("Error: MassIntegrator::AddMultMF only implemented with"
                 " libCEED");
   }
}

void MassIntegrator::AssembleDiagonalMF(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else
   {
      MFEM_ABORT("Error: MassIntegrator::AssembleDiagonalMF only implemented"
                 " with libCEED");
   }
}

} // namespace mfem
