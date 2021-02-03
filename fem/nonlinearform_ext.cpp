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
   : Operator(nlf->FESpace()->GetVSize()), nlf(nlf) { }

PANonlinearFormExtension::PANonlinearFormExtension(const NonlinearForm *nlf):
   NonlinearFormExtension(nlf),
   fes(*nlf->FESpace()),
   dnfi(*nlf->GetDNFI()),
   elemR(fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
   Grad(*this)
{
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
      energy += dnfi[i]->GetLocalStateEnergyPA(xe);
   }
   return energy;
}

void PANonlinearFormExtension::Assemble()
{
   MFEM_VERIFY(nlf->GetInteriorFaceIntegrators().Size() == 0 &&
               nlf->GetBdrFaceIntegrators().Size() == 0,
               "face integrators are not supported yet");

   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AssemblePA(fes); }
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
   Grad.AssembleGrad(x);
   return Grad;
}

void PANonlinearFormExtension::Update()
{
   height = width = fes.GetVSize();
   elemR = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   xe.SetSize(elemR->Height());
   ye.SetSize(elemR->Height());
   Grad.Update();
}

PANonlinearFormExtension::Gradient::Gradient(const PANonlinearFormExtension &e):
   Operator(e.Height()), ext(e)
{ }

void PANonlinearFormExtension::Gradient::AssembleGrad(const Vector &g)
{
   ext.elemR->Mult(g, ext.xe);
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AssembleGradPA(ext.xe, ext.fes);
   }
}

void PANonlinearFormExtension::Gradient::Mult(const Vector &x, Vector &y) const
{
   ext.ye = 0.0;
   ext.elemR->Mult(x, ext.xe);
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AddMultGradPA(ext.xe, ext.ye);
   }
   ext.elemR->MultTranspose(ext.ye, y);
}

void PANonlinearFormExtension::Gradient::AssembleDiagonal(Vector &diag) const
{
   MFEM_ASSERT(diag.Size() == Height(),
               "Vector for holding diagonal has wrong size!");
   ext.ye = 0.0;
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AssembleGradDiagonalPA(ext.ye);
   }
   ext.elemR->MultTranspose(ext.ye, diag);
}

void PANonlinearFormExtension::Gradient::Update()
{
   height = width = ext.Height();
}

} // namespace mfem
