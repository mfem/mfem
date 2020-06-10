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

// Implementations of classes FABilinearFormExtension, EABilinearFormExtension,
// PABilinearFormExtension and MFBilinearFormExtension.

#include "nonlinearform.hpp"
#include "../general/forall.hpp"

namespace mfem
{

NonlinearFormExtension::NonlinearFormExtension(const NonlinearForm *nlf)
   : Operator(nlf->FESpace()->GetTrueVSize()), nlf(nlf) { }

PANonlinearForm::PANonlinearForm(NonlinearForm *nlf):
   NonlinearFormExtension(nlf),
   fes(*nlf->FESpace()),
   dnfi(*nlf->GetDNFI()),
   R(fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC))
{
   MFEM_VERIFY(R, "Not yet implemented!");
   xe.SetSize(R->Height(), Device::GetMemoryType());
   ye.SetSize(R->Height(), Device::GetMemoryType());
   ye.UseDevice(true);
}

double PANonlinearForm::GetGridFunctionEnergy(const Vector &x) const
{
   double energy = 0.0;

   for (int i = 0; i < dnfi.Size(); i++)
   {
      energy += dnfi[i]->GetGridFunctionEnergyPA(x);
   }
   return energy;
}

void PANonlinearForm::Setup()
{
   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AssemblePA(fes); }
}

void PANonlinearForm::Mult(const Vector &x, Vector &y) const
{
   ye = 0.0;
   R->Mult(x, xe);
   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AddMultPA(xe, ye); }
   R->MultTranspose(ye, y);
}

Operator &PANonlinearForm::GetGradient(const Vector &x) const
{
   Grad.Reset(new PANonlinearForm::Gradient(x, *this));
   return *Grad.Ptr();
}

PANonlinearForm::Gradient::Gradient(const Vector &x, const PANonlinearForm &e):
   Operator(e.fes.GetVSize()), R(e.R), dnfi(e.dnfi)
{
   ge.UseDevice(true);
   ge.SetSize(R->Height(), Device::GetMemoryType());
   R->Mult(x, ge);

   xe.UseDevice(true);
   xe.SetSize(R->Height(), Device::GetMemoryType());

   ye.UseDevice(true);
   ye.SetSize(R->Height(), Device::GetMemoryType());

   ze.UseDevice(true);
   ze.SetSize(R->Height(), Device::GetMemoryType());

   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AssemblePA(e.fes); }
}

void PANonlinearForm::Gradient::Mult(const Vector &x, Vector &y) const
{
   ze = x;
   ye = 0.0;
   R->Mult(ze, xe);
   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AddMultGradPA(ge, xe, ye); }
   R->MultTranspose(ye, y);
}

} // namespace mfem
