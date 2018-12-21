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
#include "bilinearform_ext.hpp"
#include "kBilinIntegDiffusion.hpp"
#include "kfespace.hpp"
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
   kfes(new kFiniteElementSpace(a->fes)) { }

PABilinearFormExtension::~PABilinearFormExtension() { delete kfes; }

// Adds new Domain Integrator.
void PABilinearFormExtension::AddDomainIntegrator(
   AbstractBilinearFormIntegrator *i)
{
   integrators.Append(static_cast<BilinearPAFormIntegrator*>(i));
}

// *****************************************************************************
// * WARNING DiffusionGetRule Q order
// *****************************************************************************
static const IntegrationRule &DiffusionGetRule(const FiniteElement &trial_fe,
                                               const FiniteElement &test_fe)
{
   int order;
   if (trial_fe.Space() == FunctionSpace::Pk)
   {
      order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
   }
   else
   {
      // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
      order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() - 1;
   }
   if (trial_fe.Space() == FunctionSpace::rQk)
   {
      return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   return IntRules.Get(trial_fe.GetGeomType(), order);
}

void PABilinearFormExtension::Assemble()
{
   assert(integrators.Size()==1);
   const FiniteElement &fe = *a->fes->GetFE(0);
   const IntegrationRule *ir = &DiffusionGetRule(fe,fe);
   assert(ir);
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->Setup(a->fes,ir);
      integrators[i]->Assemble();
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
      const int xsz = X.Size();
      assert(xsz>=csz);
      Vector subvec(xsz);
      subvec = 0.0;
      kVectorGetSubvector(csz,
                          subvec.GetData(),
                          X.GetData(),
                          ess_tdof_list.GetData());
      X = 0.0;
      kVectorSetSubvector(csz,
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
   kfes->GlobalToLocal(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   assert(iSz==1);
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->MultAdd(localX, localY);
   }
   kfes->LocalToGlobal(localY, y);
}

void PABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   kfes->GlobalToLocal(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   assert(iSz==1);
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->MultTransposeAdd(localX, localY);
   }
   kfes->LocalToGlobal(localY, y);
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
