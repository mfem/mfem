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
#include "../general/forall.hpp"

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

Operator&
PANonlinearFormExtension::GetGradientPA(const Array<int> &ess_tdof_list,
                                        const Vector &GradX)
{
   dbg("Returning new Grad(X) Operator of size: %d", GradX.Size());
   GradOp.SetType(Operator::ANY_TYPE);
   Operator *oper =
      new PAGradOperator(GradX, nlf, fes, ess_tdof_list, elem_restrict_lex);
   GradOp.Reset(oper);
   GradOp.SetOperatorOwner(false);
   MFEM_VERIFY(GradOp.Ptr(), "GetGradientPA error!");
   return *GradOp.Ptr();
}


/// PAGradOperator
PAGradOperator::PAGradOperator(const Vector &x,
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
   dbg("[Setup]");
   if (elem_restrict_lex)
   {
      dbg("[Setup] elem_restrict_lex->Height(): %d",elem_restrict_lex->Height());
      Xe.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      Ye.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      Xe.UseDevice(true);
      Ye.UseDevice(true);
      //
      Ge.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      Ge.UseDevice(true);
      elem_restrict_lex->Mult(x, Ge);
   }
   //Grad = new SparseMatrix(fes.GetVSize());
}

void PAGradOperator::Mult(const Vector &r, Vector &c) const
{
   dbg("r:"); r.Print();
   const int csz = ess_tdof_list.Size();
   Vector z;
   z = r;
   // Should see where this is done usually
   c.SetSize(r.Size());
   dbg("Sizes: r:%d, c:%d", r.Size(), c.Size());

   auto idx = ess_tdof_list.Read();
   auto d_z = z.ReadWrite();
   MFEM_FORALL(i, csz, d_z[idx[i]] = 0.0;);
   dbg("BC z:"); z.Print();

   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int Ni = integrators.Size();
   if (elem_restrict_lex)
   {
      Ye = 0.0;
      elem_restrict_lex->Mult(z, Xe);
      dbg("Xe:"); Xe.Print();
      for (int i = 0; i < Ni; ++i)
      {
         integrators[i]->AddMultGradPA(Ge, Xe, Ye);
      }
      dbg("Ye:"); Ye.Print();
      elem_restrict_lex->MultTranspose(Ye, c);
      dbg("c:"); c.Print();
   }
   else
   {
      MFEM_ABORT("Not yet implemented!");
   }
   //#warning Should if (Serial())...
   auto d_r = r.Read();
   auto d_c = c.ReadWrite();
   MFEM_FORALL(i, csz, d_c[idx[i]] = d_r[idx[i]];);
}

} // namespace mfem
