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

#include "../general/forall.hpp"
#include "bilinearform.hpp"

namespace mfem
{

BilinearFormExtension::BilinearFormExtension(BilinearForm *form)
   : Operator(form->Size()), a(form)
{
   // empty
}

const Operator *BilinearFormExtension::GetProlongation() const
{
   return a->GetProlongation();
}

const Operator *BilinearFormExtension::GetRestriction() const
{
   return a->GetRestriction();
}


// Data and methods for partially-assembled bilinear forms
PABilinearFormExtension::PABilinearFormExtension(BilinearForm *form)
   : BilinearFormExtension(form),
     trialFes(a->FESpace()), testFes(a->FESpace())
{
   elem_restrict_lex = trialFes->GetElementRestriction(
                          ElementDofOrdering::LEXICOGRAPHIC);
   if (elem_restrict_lex)
   {
      localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
   }

   int_face_restrict_lex = trialFes->GetFaceRestriction(
                          ElementDofOrdering::LEXICOGRAPHIC, FaceType::Interior);
   if (int_face_restrict_lex)
   {
      faceIntX.SetSize(int_face_restrict_lex->Height(), Device::GetMemoryType());
      faceIntY.SetSize(int_face_restrict_lex->Height(), Device::GetMemoryType());
      faceIntY.UseDevice(true); // ensure 'localY = 0.0' is done on device
   }

   bound_face_restrict_lex = trialFes->GetFaceRestriction(
                          ElementDofOrdering::LEXICOGRAPHIC, FaceType::Boundary);
   if (bound_face_restrict_lex)
   {
      faceBoundX.SetSize(bound_face_restrict_lex->Height(), Device::GetMemoryType());
      faceBoundY.SetSize(bound_face_restrict_lex->Height(), Device::GetMemoryType());
      faceBoundY.UseDevice(true); // ensure 'localY = 0.0' is done on device
   }
}

void PABilinearFormExtension::Assemble()
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->AssemblePA(*a->FESpace());
   }

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int intFaceIntegratorCount = intFaceIntegrators.Size();
   for (int i = 0; i < intFaceIntegratorCount; ++i)
   {
      intFaceIntegrators[i]->AssemblePAInteriorFaces(*a->FESpace());
   }

   Array<BilinearFormIntegrator*> &boundFaceIntegrators = *a->GetBFBFI();
   const int boundFaceIntegratorCount = boundFaceIntegrators.Size();
   for (int i = 0; i < boundFaceIntegratorCount; ++i)
   {
      boundFaceIntegrators[i]->AssemblePABoundaryFaces(*a->FESpace());
   }
}

void PABilinearFormExtension::Update()
{
   FiniteElementSpace *fes = a->FESpace();
   height = width = fes->GetVSize();
   trialFes = fes;
   testFes = fes;
   elem_restrict_lex = trialFes->GetElementRestriction(
                          ElementDofOrdering::LEXICOGRAPHIC);
   if (elem_restrict_lex)
   {
      localX.SetSize(elem_restrict_lex->Height());
      localY.SetSize(elem_restrict_lex->Height());
   }
   int_face_restrict_lex = trialFes->GetFaceRestriction(
                          ElementDofOrdering::LEXICOGRAPHIC, FaceType::Interior);
   if (int_face_restrict_lex)
   {
      faceIntX.SetSize(int_face_restrict_lex->Height());
      faceIntY.SetSize(int_face_restrict_lex->Height());
   }
   bound_face_restrict_lex = trialFes->GetFaceRestriction(
                          ElementDofOrdering::LEXICOGRAPHIC, FaceType::Boundary);
   if (bound_face_restrict_lex)
   {
      faceBoundX.SetSize(bound_face_restrict_lex->Height());
      faceBoundY.SetSize(bound_face_restrict_lex->Height());
   }
}

void PABilinearFormExtension::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                               OperatorHandle &A)
{
   const Operator* trialP = trialFes->GetProlongationMatrix();
   const Operator* testP  = testFes->GetProlongationMatrix();
   Operator *rap = this;
   if (trialP) { rap = new RAPOperator(*testP, *this, *trialP); }
   const bool own_A = (rap!=this);
   A.Reset(new ConstrainedOperator(rap, ess_tdof_list, own_A));
}

void PABilinearFormExtension::FormLinearSystem(const Array<int> &ess_tdof_list,
                                               Vector &x, Vector &b,
                                               OperatorHandle &A,
                                               Vector &X, Vector &B,
                                               int copy_interior)
{
   Operator *oper;
   Operator::FormLinearSystem(ess_tdof_list, x, b, oper, X, B, copy_interior);
   A.Reset(oper); // A will own oper
}

void PABilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();

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

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      int_face_restrict_lex->Mult(x, faceIntX);
      faceIntY = 0.0;
      for (int i = 0; i < iFISz; ++i)
      {
         intFaceIntegrators[i]->AddMultPA(faceIntX, faceIntY);
      }
      int_face_restrict_lex->MultTranspose(faceIntY, y);
   }

   Array<BilinearFormIntegrator*> &boundFaceIntegrators = *a->GetBFBFI();
   const int bFISz = boundFaceIntegrators.Size();
   if (bound_face_restrict_lex && bFISz>0)
   {
      bound_face_restrict_lex->Mult(x, faceBoundX);
      faceBoundY = 0.0;
      for (int i = 0; i < bFISz; ++i)
      {
         boundFaceIntegrators[i]->AddMultPA(faceBoundX, faceBoundY);
      }
      bound_face_restrict_lex->MultTranspose(faceBoundY, y);
   }
}

void PABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int iSz = integrators.Size();
   if (elem_restrict_lex)
   {
      elem_restrict_lex->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultTransposePA(localX, localY);
      }
      elem_restrict_lex->MultTranspose(localY, y);
   }
   else
   {
      y.UseDevice(true);
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultTransposePA(x, y);
      }
   }

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      int_face_restrict_lex->Mult(x, faceIntX);
      faceIntY = 0.0;
      for (int i = 0; i < iFISz; ++i)
      {
         intFaceIntegrators[i]->AddMultTransposePA(faceIntX, faceIntY);
      }
      int_face_restrict_lex->MultTranspose(faceIntY, y);
   }

   Array<BilinearFormIntegrator*> &boundFaceIntegrators = *a->GetBFBFI();
   const int bFISz = boundFaceIntegrators.Size();
   if (bound_face_restrict_lex && bFISz>0)
   {
      bound_face_restrict_lex->Mult(x, faceBoundX);
      faceBoundY = 0.0;
      for (int i = 0; i < bFISz; ++i)
      {
         boundFaceIntegrators[i]->AddMultTransposePA(faceBoundX, faceBoundY);
      }
      bound_face_restrict_lex->MultTranspose(faceBoundY, y);
   }
}

} // namespace mfem
