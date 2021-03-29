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
#include "ceed/util.hpp"

namespace mfem
{

NonlinearFormExtension::NonlinearFormExtension(NonlinearForm *form)
   : Operator(form->FESpace()->GetTrueVSize()), n(form)
{
   // empty
}

PANonlinearFormExtension::PANonlinearFormExtension(NonlinearForm *form):
   NonlinearFormExtension(form),
   x_grad(nullptr),
   fes(*n->FESpace()),
   dnfi(*n->GetDNFI()),
   R(fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC))
{
   MFEM_VERIFY(R, "Not yet implemented!");
   xe.SetSize(R->Height(), Device::GetMemoryType());
   ye.SetSize(R->Height(), Device::GetMemoryType());
   ye.UseDevice(true);
}

void PANonlinearFormExtension::Assemble()
{
   Array<NonlinearFormIntegrator*> &integrators = *n->GetDNFI();
   const int Ni = integrators.Size();
   for (int i = 0; i < Ni; ++i)
   {
      integrators[i]->AssemblePA(*n->FESpace());
   }
}

void PANonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   Array<NonlinearFormIntegrator*> &integrators = *n->GetDNFI();
   const int iSz = integrators.Size();
   if (R && !DeviceCanUseCeed())
   {
      R->Mult(x, xe);
      ye = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultPA(xe, ye);
      }
      R->MultTranspose(ye, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultPA(x, y);
      }
   }
}

double PANonlinearFormExtension::GetGridFunctionEnergy(const Vector &x) const
{
   double energy = 0.0;

   R->Mult(x, xe);
   for (int i = 0; i < dnfi.Size(); i++)
   {
      energy += dnfi[i]->GetGridFunctionEnergyPA(xe);
   }
   return energy;
}

void PANonlinearFormExtension::AssembleGradientDiagonal(Vector &diag) const
{
   MFEM_VERIFY(x_grad, "GetGradient() has not been called");
   R->Mult(*x_grad, xe);

   ye = 0.0;
   for (int i = 0; i < dnfi.Size(); ++i)
   {
      dnfi[i]->AssembleGradientDiagonalPA(xe, ye);
   }
   R->MultTranspose(ye, diag);
}

Operator &PANonlinearFormExtension::GetGradient(const Vector &x) const
{
   // Store the last x that was used to compute the gradient.
   x_grad = &x;

   Grad.Reset(new PANonlinearFormExtension::Gradient(x, *this));
   return *Grad.Ptr();
}

PANonlinearFormExtension::Gradient::Gradient(const Vector &x,
                                             const PANonlinearFormExtension &e):
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

   // Do we still need to do this?
   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AssemblePA(e.fes); }
}

void PANonlinearFormExtension::Gradient::Mult(const Vector &x, Vector &y) const
{
   ze = x;
   ye = 0.0;
   R->Mult(ze, xe);
   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AddMultGradPA(ge, xe, ye); }
   R->MultTranspose(ye, y);
}

MFNonlinearFormExtension::MFNonlinearFormExtension(NonlinearForm *form):
   NonlinearFormExtension(form), fes(*form->FESpace())
{
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   elem_restrict_lex = fes.GetElementRestriction(ordering);
   if (elem_restrict_lex)
   {
      localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
   }
}

void MFNonlinearFormExtension::Assemble()
{
   Array<NonlinearFormIntegrator*> &integrators = *n->GetDNFI();
   const int Ni = integrators.Size();
   for (int i = 0; i < Ni; ++i)
   {
      integrators[i]->AssembleMF(*n->FESpace());
   }
}

void MFNonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   Array<NonlinearFormIntegrator*> &integrators = *n->GetDNFI();
   const int iSz = integrators.Size();
   if (elem_restrict_lex && !DeviceCanUseCeed())
   {
      elem_restrict_lex->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultMF(localX, localY);
      }
      elem_restrict_lex->MultTranspose(localY, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultMF(x, y);
      }
   }
}

} // namespace mfem
