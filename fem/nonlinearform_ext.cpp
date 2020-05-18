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
#define MFEM_DBG_COLOR 141
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
      dbg("elem_restrict_lex->Height(): %d",elem_restrict_lex->Height());
      localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
   }
}

void PANonlinearFormExtension::AssemblePA()
{
   dbg("");
   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int Ni = integrators.Size();
   for (int i = 0; i < Ni; ++i)
   {
      integrators[i]->AssemblePA(*nlf->FESpace());
   }
}

Operator&
PANonlinearFormExtension::GetGradientPA(const Array<int> &ess_tdof_list,
                                        Vector &x)
{
   dbg("");
   GradOp.SetType(Operator::ANY_TYPE);
   Operator *oper =
      new PAGradOperator(nlf, fes, ess_tdof_list, elem_restrict_lex);
   GradOp.Reset(oper);
   GradOp.SetOperatorOwner(false);
   MFEM_VERIFY(GradOp.Ptr(), "AssembleGradPA error!");
   return *GradOp.Ptr();
}

void PANonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   dbg("");
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
      dbg("!elem_restrict_lex");
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultPA(x, y);
      }
   }
}

PAGradOperator::PAGradOperator(const NonlinearForm *nlf,
                               const FiniteElementSpace &fes,
                               const Array<int> &ess_tdof_list,
                               const Operator *elem_restrict_lex): Operator(),
   nlf(nlf),
   fes(fes),
   ess_tdof_list(ess_tdof_list),
   elem_restrict_lex(elem_restrict_lex)
{
   dbg("\033[7mPAGradOperator");
   if (elem_restrict_lex)
   {
      dbg("elem_restrict_lex->Height(): %d",elem_restrict_lex->Height());
      localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localX.UseDevice(true);
      localY.UseDevice(true);
   }
}

void PAGradOperator::Mult(const Vector &x, Vector &y) const
{
   // Should see where this is done usually
   y.SetSize(x.Size());
   dbg("Sizes: %d, %d", x.Size(), y.Size());
   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int Ni = integrators.Size();
   if (elem_restrict_lex)
   {
      elem_restrict_lex->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < Ni; ++i)
      {
         integrators[i]->AddMultGradPA(localX, localY);
      }
      elem_restrict_lex->MultTranspose(localY, y);
   }
   else
   {
      MFEM_ABORT("Not yet implemented!");
   }
   // Should if (Serial())...
}

} // namespace mfem
