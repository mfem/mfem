// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementations of classes FABilinearFormExtension, EABilinearFormExtension,
// PABilinearFormExtension and MFBilinearFormExtension.

#include "nonlinearform.hpp"

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
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   elem_restrict_lex = fes.GetElementRestriction(ordering);
   if (elem_restrict_lex)
   {
      localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
   }
}

void PANonlinearFormExtension::AssemblePA()
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
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultPA(x, y);
      }
   }
}

}
