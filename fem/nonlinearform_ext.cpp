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

// Implementations of classes FABilinearFormExtension, EABilinearFormExtension,
// PABilinearFormExtension and MFBilinearFormExtension.

#include "nonlinearform.hpp"
#include "../general/forall.hpp"

namespace mfem
{

NonlinearFormExtension::NonlinearFormExtension(const NonlinearForm *nlf)
   : Operator(nlf->FESpace()->GetTrueVSize()), nlf(nlf) { }

PANonlinearFormExtension::PANonlinearFormExtension(NonlinearForm *nlf):
   NonlinearFormExtension(nlf),
   fes(*nlf->FESpace()),
   dnfi(*nlf->GetDNFI()),
   elemR(fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC))
{
   MFEM_VERIFY(elemR, "Not yet implemented!");
   xe.SetSize(elemR->Height(), Device::GetMemoryType());
   ye.SetSize(elemR->Height(), Device::GetMemoryType());
   ye.UseDevice(true);
}

double PANonlinearFormExtension::GetGridFunctionEnergy(const Vector &x) const
{
   double energy = 0.0;

   elemR->Mult(x, xe);
   for (int i = 0; i < dnfi.Size(); i++)
   {
      energy += dnfi[i]->GetGridFunctionEnergyPA(xe);
   }
   return energy;
}

void PANonlinearFormExtension::Assemble()
{
   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AssemblePA(fes); }
}

void PANonlinearFormExtension::AssembleGradient(const Vector &x)
{
   elemR->Mult(x, xe);
   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AssembleGradPA(xe, fes); }
}

void PANonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   ye = 0.0;
   elemR->Mult(x, xe);
   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AddMultPA(xe, ye); }
   elemR->MultTranspose(ye, y);
}

Operator &PANonlinearFormExtension::GetGradient(const Vector &x) const
{
   if (Grad.Ptr() == nullptr)
   {
      Grad.Reset(new PANonlinearFormExtension::Gradient(x, *this));
   }
   else
   {
      dynamic_cast<PANonlinearFormExtension::Gradient *>(Grad.Ptr())->ReInit(x);
   }
   return *Grad.Ptr();
}

PANonlinearFormExtension::Gradient::Gradient(const Vector &g,
                                             const PANonlinearFormExtension &e):
   Operator(e.fes.GetVSize()), elemR(e.elemR), fes(e.fes), dnfi(e.dnfi)
{
   ge.UseDevice(true);
   ge.SetSize(elemR->Height(), Device::GetMemoryType());
   elemR->Mult(g, ge);

   xe.UseDevice(true);
   xe.SetSize(elemR->Height(), Device::GetMemoryType());

   ye.UseDevice(true);
   ye.SetSize(elemR->Height(), Device::GetMemoryType());

   ze.UseDevice(true);
   ze.SetSize(elemR->Height(), Device::GetMemoryType());
}

void PANonlinearFormExtension::Gradient::Mult(const Vector &x, Vector &y) const
{
   ze = x;
   ye = 0.0;
   elemR->Mult(ze, xe);
   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AddMultGradPA(ge, xe, ye); }
   elemR->MultTranspose(ye, y);
}

void PANonlinearFormExtension::Gradient::AssembleDiagonal(Vector &diag) const
{
   MFEM_ASSERT(diag.Size() == fes.GetVSize(),
               "Vector for holding diagonal has wrong size!");
   ye = 0.0;
   for (int i = 0; i < dnfi.Size(); ++i)
   {
      dnfi[i]->AssembleGradDiagonalPA(ge, ye);
   }
   elemR->MultTranspose(ye, diag);
}

} // namespace mfem
