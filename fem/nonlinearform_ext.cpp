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
#define MFEM_DBG_COLOR 165
#include "../general/dbg.hpp"

namespace mfem
{

NonlinearFormExtension::NonlinearFormExtension(const NonlinearForm *nlf)
   : Operator(nlf->FESpace()->GetTrueVSize()), nlf(nlf)
{
   // empty
}

PANonlinearFormExtension::PANonlinearFormExtension(NonlinearForm *nlf):
   NonlinearFormExtension(nlf), fes(*nlf->FESpace())
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

double PANonlinearFormExtension::GetGridFunctionEnergy(const Vector &x) const
{
   double energy = 0.0;
   const Array<NonlinearFormIntegrator*> &dnfi = *nlf->GetDNFI();

   if (dnfi.Size())
   {
      for (int k = 0; k < dnfi.Size(); k++)
      {
         energy += dnfi[k]->GetGridFunctionEnergyPA(fes, x);
      }
   }

   return energy;
}

void PANonlinearFormExtension::Assemble()
{
   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int Ni = integrators.Size();
   for (int i = 0; i < Ni; ++i)
   {
      integrators[i]->AssemblePA(*nlf->FESpace());
   }
}

void PANonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int iSz = integrators.Size();
   if (elem_restrict_lex)
   {
      elem_restrict_lex->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultPA(localX, localY);
      }
      elem_restrict_lex->MultTranspose(localY, y);
   }
   else
   {
      MFEM_ABORT("Not yet implemented!");
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultPA(x, y);
      }
   }
}

Operator&
PANonlinearFormExtension::GetGradient(const Vector &x) const
{
   GradOp.SetType(Operator::ANY_TYPE);
   const Array<int> &esstdofs = nlf->GetEssentialTrueDofs();
   Operator *oper =  new PAGradOperator(x, nlf, fes, esstdofs, elem_restrict_lex);
   GradOp.Reset(oper);
   GradOp.SetOperatorOwner(false);
   MFEM_VERIFY(GradOp.Ptr(), "GetGradientPA error!");
   return *GradOp.Ptr();
}


/// PAGradOperator
PAGradOperator::PAGradOperator(const Vector &g,
                               const NonlinearForm *nlf,
                               const FiniteElementSpace &fes,
                               const Array<int> &ess_tdof_list,
                               const Operator *elem_restrict_lex):
   Operator(fes.GetVSize()),
   nlf(nlf),
   fes(fes),
   ess_tdof_list(ess_tdof_list),
   elem_restrict_lex(elem_restrict_lex)
{
   if (elem_restrict_lex)
   {
      ge.UseDevice(true);
      ge.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      elem_restrict_lex->Mult(g, ge);
      xe.UseDevice(true);
      xe.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      ye.UseDevice(true);
      ye.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
   }

   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int Ni = integrators.Size();
   for (int i = 0; i < Ni; ++i)
   {
      integrators[i]->AssemblePA(*nlf->FESpace());
   }
}

void PAGradOperator::Mult(const Vector &x, Vector &y) const
{
   Vector z(x);
   z.UseDevice(true);
   y.SetSize(x.Size());
   const int csz = ess_tdof_list.Size();

   auto idx = ess_tdof_list.Read();
   auto d_z = z.ReadWrite();
   MFEM_FORALL(i, csz, d_z[idx[i]] = 0.0;);

   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int Ni = integrators.Size();
   if (elem_restrict_lex)
   {
      ye = 0.0;
      elem_restrict_lex->Mult(z, xe);
      for (int i = 0; i < Ni; ++i)
      {
         integrators[i]->AddMultGradPA(ge, xe, ye);
      }
      elem_restrict_lex->MultTranspose(ye, y);
   }
   else { MFEM_ABORT("Not yet implemented!"); }

   auto d_r = x.Read();
   auto d_c = y.ReadWrite();
   MFEM_FORALL(i, csz, d_c[idx[i]] = d_r[idx[i]];);
}

} // namespace mfem
