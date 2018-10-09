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

// Implementation of class BilinearForm

#include "fem.hpp"
#include "bilininteg.hpp"
#include "kBilinIntegDiffusion.hpp"
#include "kfespace.hpp"

#include <cmath>

// *****************************************************************************
MFEM_NAMESPACE

// ***************************************************************************
// * PABilinearForm
// ***************************************************************************
PABilinearForm::PABilinearForm(FiniteElementSpace* fes) :
   AbstractBilinearForm(fes),
   mesh(fes->GetMesh()),
   trialFes(fes),
   testFes(fes),
   localX(mesh->GetNE() * trialFes->GetFE(0)->GetDof() * trialFes->GetVDim()),
   localY(mesh->GetNE() * testFes->GetFE(0)->GetDof() * testFes->GetVDim()),
   kfes(new kFiniteElementSpace(*fes)) { dbg("\033[7mPABilinearForm"); }

// ***************************************************************************
PABilinearForm::~PABilinearForm(){ /*delete kfes;*/}

// *****************************************************************************
void PABilinearForm::EnableStaticCondensation(){ assert(false);}

// ***************************************************************************
// Adds new Domain Integrator.
void PABilinearForm::AddDomainIntegrator(AbstractBilinearFormIntegrator *i) {
   dbg();
   dbg("\033[7mAddDomainIntegrator");
   integrators.Append(static_cast<BilinearPAFormIntegrator*>(i));
}

// Adds new Boundary Integrator.
void PABilinearForm::AddBoundaryIntegrator(AbstractBilinearFormIntegrator *i) {
   assert(false);
   //AddIntegrator(i, BoundaryIntegrator);
}

// Adds new interior Face Integrator.
void PABilinearForm::AddInteriorFaceIntegrator(AbstractBilinearFormIntegrator *i) {
   assert(false);
   //AddIntegrator(i, InteriorFaceIntegrator);
}

// Adds new boundary Face Integrator.
void PABilinearForm::AddBoundaryFaceIntegrator(AbstractBilinearFormIntegrator *i) {
   assert(false);
   //AddIntegrator(i, BoundaryFaceIntegrator);
}

// ***************************************************************************
void PABilinearForm::Assemble(int skip_zeros) {
   dbg("\033[7mAssemble");
   const int nbi = integrators.Size();
   dbg("nbi=%d",nbi);
   assert(integrators.Size()==1);
   const IntegrationRule *ir0 = integrators[0]->GetIntRule();
   const FiniteElement &fe = *fes->GetFE(0);
   const int order = ir0?ir0->GetOrder():2*fe.GetOrder() - 2;
   const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), order);
   assert(ir);
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i) {
      integrators[i]->Setup(fes,ir);
      integrators[i]->Assemble();
   }
   
}

// ***************************************************************************
void PABilinearForm::FormOperator(const Array<int> &ess_tdof_list,
                                  Operator &A) {
#warning move semantic?
   dbg();assert(false);
   const Operator* trialP = trialFes->GetProlongationMatrix();
   const Operator* testP  = testFes->GetProlongationMatrix();
   Operator *rap = this;
   if (trialP) { rap = new RAPOperator(*testP, *this, *trialP); }
   const bool own_A = rap!=this;
   assert(rap);
   Operator *CO = new ConstrainedOperator(rap, ess_tdof_list, own_A);
   A = *CO;
}

// ***************************************************************************
void PABilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list,
                                      Vector &x, Vector &b,
                                      Operator &A, Vector &X, Vector &B,
                                      int copy_interior) {
   dbg();
   //FormOperator(ess_tdof_list, A);

   const Operator* trialP = trialFes->GetProlongationMatrix();
   const Operator* testP  = testFes->GetProlongationMatrix();
   Operator *rap = this;
   if (trialP) { assert(false);rap = new RAPOperator(*testP, *this, *trialP); }
   const bool own_A = rap!=this;
   assert(rap);
   
   ConstrainedOperator *CO = new ConstrainedOperator(rap, ess_tdof_list, own_A);
   A = *CO;
   
   const Operator* P = trialFes->GetProlongationMatrix();
   const Operator* R = trialFes->GetRestrictionMatrix();
   if (P) {
      // Variational restriction with P
      B.SetSize(P->Width());
      P->MultTranspose(b, B);
      X.SetSize(R->Height());
      R->Mult(x, X);
   } else {
      // rap, X and B point to the same data as this, x and b
#warning look here too
      X.SetSize(x.Size()/*,x*/); X = x;
      B.SetSize(b.Size()/*,b*/); B = b;
      //assert(false);
   }
   //ConstrainedOperator *cA = static_cast<ConstrainedOperator*>(&A);
   assert(CO);
   if (CO) {
      dbg("ConstrainedOperator");
#warning and there
      //cA->EliminateRHS(X, B);
      dbg("z, w");
      Vector z(A.Height());
      Vector w(A.Height());
      dbg("w = 0.0");
      w = 0.0;
      dbg("CO->Mult(w, z)");
      CO->Mult(w, z);
      dbg("b -= z");
      b -= z;
   } else {
      mfem_error("BilinearForm::InitRHS expects an ConstrainedOperator");
   }
}

// ***************************************************************************
void PABilinearForm::Mult(const Vector &x, Vector &y) const {
   dbg();
   //trialFes
   kfes->GlobalToLocal(x, localX);
   localY = 0.0;
   //assert(diffusion);
//#warning diffusion Assemble
   //diffusion->MultAdd(localX, localY);
   
   const int iSz = integrators.Size();
   for (int i = 0; i < iSz; ++i) {
      integrators[i]->MultAdd(localX, localY);
   }
   //testFes
   kfes->LocalToGlobal(localY, y);
   //stk(true);
   //assert(false);
   dbg("done");
}

// ***************************************************************************
void PABilinearForm::MultTranspose(const Vector &x, Vector &y) const {
   dbg();
   assert(false);
   //testFes
   kfes->GlobalToLocal(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   for (int i = 0; i < iSz; ++i) {
      integrators[i]->MultTransposeAdd(localX, localY);
   }
   //trialFes
   kfes->LocalToGlobal(localY, y);
}

// ***************************************************************************
void PABilinearForm::RecoverFEMSolution(const Vector &X,
                                          const Vector &b,
                                        Vector &x) {
   dbg();
   const Operator *P = this->GetProlongation();
   if (P) {
      // Apply conforming prolongation
      x.SetSize(P->Height());
      P->Mult(X, x);
   }
   // Otherwise X and x point to the same data
}

// *****************************************************************************
MFEM_NAMESPACE_END
