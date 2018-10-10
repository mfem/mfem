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
#include "../linalg/kvector.hpp"

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
   kfes(new kFiniteElementSpace(*fes)) { }

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

// *****************************************************************************
static const IntegrationRule &DiffusionGetRule(const FiniteElement &trial_fe,
                                               const FiniteElement &test_fe){
   int order;
   if (trial_fe.Space() == FunctionSpace::Pk){
      order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
   }else{
      // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
      order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() - 1;
   }
   if (trial_fe.Space() == FunctionSpace::rQk){
      return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   return IntRules.Get(trial_fe.GetGeomType(), order);
}

// ***************************************************************************
void PABilinearForm::Assemble(int skip_zeros) {
   dbg("\033[7mAssemble");
   const int nbi = integrators.Size();
   dbg("nbi=%d",nbi);
   assert(integrators.Size()==1);
   const IntegrationRule *ir0 = integrators[0]->GetIntRule();
   const FiniteElement &fe = *fes->GetFE(0);
   const IntegrationRule *diffusionIR = &DiffusionGetRule(fe,fe);
   const int order = ir0?ir0->GetOrder():diffusionIR->GetOrder();
   const IntegrationRule *ir = ir0 ? ir0 : diffusionIR;
   assert(ir);
   //const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), order);
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i) {
      integrators[i]->Setup(fes,ir);
      //assert(false);
      integrators[i]->Assemble();
   }
   //assert(false);   
}

// ***************************************************************************
void PABilinearForm::FormOperator(const Array<int> &ess_tdof_list,
                                  Operator &A) {
//#warning move semantic?
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
                                      Operator **A, Vector &X, Vector &B,
                                      int copy_interior) {
   dbg();
   
   //FormOperator(ess_tdof_list, A);
   const Operator* trialP = trialFes->GetProlongationMatrix();
   const Operator* testP  = testFes->GetProlongationMatrix();
   Operator *rap = this;
   if (trialP) { rap = new RAPOperator(*testP, *this, *trialP); }
   const bool own_A = rap!=this;
   assert(rap);
   
   *A = new ConstrainedOperator(rap, ess_tdof_list, own_A);
   
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
      // Could MakeRef
      X.SetSize(x.Size()); X = x;
      B.SetSize(b.Size()); B = b;
   }
   
   if (!copy_interior and ess_tdof_list.Size()>0) {
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
      
   ConstrainedOperator *cA = static_cast<ConstrainedOperator*>(*A);
   assert(cA);
   if (cA) {
      dbg("ConstrainedOperator");
      cA->EliminateRHS(X, B);
   } else {
      mfem_error("BilinearForm::InitRHS expects an ConstrainedOperator");
   }
}

// ***************************************************************************
void PABilinearForm::Mult(const Vector &x, Vector &y) const {
   dbg();//stk(true);
   kfes->GlobalToLocal(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   assert(iSz==1);
   dbg("iSz=%d",iSz);
   for (int i = 0; i < iSz; ++i) {
      dbg("integrators #%d",i);
      integrators[i]->MultAdd(localX, localY);
      //dbg("localY");localY.Print();
   }
   kfes->LocalToGlobal(localY, y);
}

// ***************************************************************************
void PABilinearForm::MultTranspose(const Vector &x, Vector &y) const {
   dbg();
   assert(false);
   kfes->GlobalToLocal(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   assert(iSz==1);
   for (int i = 0; i < iSz; ++i) {
      integrators[i]->MultTransposeAdd(localX, localY);
   }
   kfes->LocalToGlobal(localY, y);
}

// ***************************************************************************
void PABilinearForm::RecoverFEMSolution(const Vector &X,
                                        const Vector &b,
                                        Vector &x) {
   dbg();
   dbg("X=");kVectorPrint(X.Size(),X);
   mm::Get().Rsync(X.GetData());
   mm::Get().Rsync(b.GetData());
   mm::Get().Rsync(x.GetData());
   const Operator *P = this->GetProlongation();
   if (P) {
      // Apply conforming prolongation
      x.SetSize(P->Height());
      P->Mult(X, x);
   }
   // Otherwise X and x point to the same data
   x = X;
   dbg("x:"); x.Print(); //assert(false);
}

// *****************************************************************************
MFEM_NAMESPACE_END
