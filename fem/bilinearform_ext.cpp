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

#include "fem.hpp"
#include "bilininteg.hpp"
#include "fespace_ext.hpp"
#include "bilinearform_ext.hpp"
#include "../linalg/kernels/vector.hpp"

namespace mfem
{

// Data and methods for fully-assembled bilinear forms
FABilinearFormExtension::FABilinearFormExtension(BilinearForm *form) :
   Operator(form->Size()), a(form) { }

// Data and methods for element-assembled bilinear forms
EABilinearFormExtension::EABilinearFormExtension(BilinearForm *form)
   : Operator(form->Size()), a(form) { }

// Data and methods for partially-assembled bilinear forms
PABilinearFormExtension::PABilinearFormExtension(BilinearForm *form) :
   Operator(form->Size()), a(form),
   trialFes(a->fes), testFes(a->fes),
   localX(a->fes->GetNE() * trialFes->GetFE(0)->GetDof() * trialFes->GetVDim()),
   localY(a->fes->GetNE() * testFes->GetFE(0)->GetDof() * testFes->GetVDim()),
   fes_ext(new FiniteElementSpaceExtension(*(a->fes))) { }

PABilinearFormExtension::~PABilinearFormExtension()
{
   for (int i = 0; i < integrators.Size(); ++i)
   {
      delete integrators[i];
   }
   delete fes_ext;
}

// Adds new Domain Integrator.
void PABilinearFormExtension::AddDomainIntegrator(BilinearFormIntegrator *i)
{
   integrators.Append(i);
}

void PABilinearFormExtension::Assemble()
{
   assert(integrators.Size()==1);
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->Assemble(*a->fes);
   }
}

void PABilinearFormExtension::FormSystemOperator(const Array<int>
                                                 &ess_tdof_list,
                                                 Operator *&A)
{
   const Operator* trialP = trialFes->GetProlongationMatrix();
   const Operator* testP  = testFes->GetProlongationMatrix();
   Operator *rap = this;
   if (trialP) { rap = new RAPOperator(*testP, *this, *trialP); }
   const bool own_A = (rap!=this);
   assert(rap);
   A = new ConstrainedOperator(rap, ess_tdof_list, own_A);
}

void PABilinearFormExtension::FormLinearSystem(const Array<int> &ess_tdof_list,
                                               Vector &x, Vector &b,
                                               Operator *&A, Vector &X, Vector &B,
                                               int copy_interior)
{
   FormSystemOperator(ess_tdof_list, A);

   const Operator* P = trialFes->GetProlongationMatrix();
   const Operator* R = trialFes->GetRestrictionMatrix();
   if (P)
   {
      // Variational restriction with P
      B.SetSize(P->Width());
      P->MultTranspose(b, B);
      X.SetSize(R->Height());
      R->Mult(x, X);
   }
   else
   {
      // rap, X and B point to the same data as this, x and b
      X.SetSize(x.Size()); X = x;
      B.SetSize(b.Size()); B = b;
   }

   if (!copy_interior && ess_tdof_list.Size()>0)
   {
      const int csz = ess_tdof_list.Size();
      Vector subvec(csz);
      subvec = 0.0;
      kernels::vector::GetSubvector(csz,
                                    subvec.GetData(),
                                    X.GetData(),
                                    ess_tdof_list.GetData());
      X = 0.0;
      kernels::vector::SetSubvector(csz,
                                    X.GetData(),
                                    subvec.GetData(),
                                    ess_tdof_list.GetData());
   }

   ConstrainedOperator *cA = static_cast<ConstrainedOperator*>(A);
   assert(cA);
   if (cA)
   {
      cA->EliminateRHS(X, B);
   }
   else
   {
      mfem_error("BilinearForm::InitRHS expects an ConstrainedOperator");
   }
}

void PABilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   fes_ext->L2E(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->MultAssembled(localX, localY);
   }
   fes_ext->E2L(localY, y);
}

void PABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   fes_ext->L2E(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->MultAssembledTranspose(localX, localY);
   }
   fes_ext->E2L(localY, y);
}

void PABilinearFormExtension::RecoverFEMSolution(const Vector &X,
                                                 const Vector &b,
                                                 Vector &x)
{
   const Operator *P = a->GetProlongation();
   if (P)
   {
      // Apply conforming prolongation
      x.SetSize(P->Height());
      P->Mult(X, x);
      return;
   }
   // Otherwise X and x point to the same data
   x = X;
}

// Data and methods for matrix-free bilinear forms
MFBilinearFormExtension::MFBilinearFormExtension(BilinearForm *form)
   : Operator(form->Size()), a(form) { }

}
