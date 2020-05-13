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

NonlinearFormExtension::NonlinearFormExtension(NonlinearForm *form)
   : Operator(form->FESpace()->GetTrueVSize()), n(form)
{
   // empty
}

PANonlinearFormExtension::PANonlinearFormExtension(NonlinearForm *form):
   NonlinearFormExtension(form), fes(*form->FESpace())
{
   dbg("");
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
   Array<NonlinearFormIntegrator*> &integrators = *n->GetDNFI();
   const int Ni = integrators.Size();
   for (int i = 0; i < Ni; ++i)
   {
      integrators[i]->AssemblePA(*n->FESpace());
   }
}

void PANonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   dbg("");
   Array<NonlinearFormIntegrator*> &integrators = *n->GetDNFI();
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

}
