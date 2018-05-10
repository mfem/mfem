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
#include "../raja.hpp"

namespace mfem {

// ***************************************************************************
// * RajaBilinearForm
// ***************************************************************************
RajaBilinearForm::RajaBilinearForm(RajaFiniteElementSpace* fes) :
  RajaOperator(fes->GetVSize(),fes->GetVSize()),
  mesh(fes->GetMesh()),
  trialFes(fes),
  testFes(fes),
  localX(mesh->GetNE() * trialFes->GetLocalDofs() * trialFes->GetVDim()),
  localY(mesh->GetNE() * testFes->GetLocalDofs() * testFes->GetVDim()){}

  // ***************************************************************************
  RajaBilinearForm::~RajaBilinearForm(){ }
  
// ***************************************************************************
// Adds new Domain Integrator.
void RajaBilinearForm::AddDomainIntegrator(RajaIntegrator* i) {
  push(SteelBlue); 
  AddIntegrator(i, DomainIntegrator);
  pop();
}

// Adds new Boundary Integrator.
void RajaBilinearForm::AddBoundaryIntegrator(RajaIntegrator* i) {
  push(SteelBlue);
  AddIntegrator(i, BoundaryIntegrator);
  pop();
}

// Adds new interior Face Integrator.
void RajaBilinearForm::AddInteriorFaceIntegrator(RajaIntegrator* i) {
  push(SteelBlue);
  AddIntegrator(i, InteriorFaceIntegrator);
  pop();
}

// Adds new boundary Face Integrator.
void RajaBilinearForm::AddBoundaryFaceIntegrator(RajaIntegrator* i) {
  push(SteelBlue);
  AddIntegrator(i, BoundaryFaceIntegrator);
  pop();
}

// Adds Integrator based on RajaIntegratorType
void RajaBilinearForm::AddIntegrator(RajaIntegrator* i,
                                     const RajaIntegratorType itype) {
  push(SteelBlue);
  assert(i);
  i->SetupIntegrator(*this, itype);
  integrators.push_back(i);
  pop();
}

// ***************************************************************************
void RajaBilinearForm::Assemble() {
  push(SteelBlue);
  const int integratorCount = (int) integrators.size();
  for (int i = 0; i < integratorCount; ++i) {
    integrators[i]->Assemble();
  }
  pop();
}

// ***************************************************************************
void RajaBilinearForm::FormLinearSystem(const Array<int>& constraintList,
                                        RajaVector& x, RajaVector& b,
                                        RajaOperator*& Aout,
                                        RajaVector& X, RajaVector& B,
                                        int copy_interior) {
  push(SteelBlue);
  FormOperator(constraintList, Aout);
  InitRHS(constraintList, x, b, Aout, X, B, copy_interior);
  pop();
}

// ***************************************************************************
void RajaBilinearForm::FormOperator(const Array<int>& constraintList,
                                    RajaOperator*& Aout) {
  push(SteelBlue);
  const RajaOperator* trialP = trialFes->GetProlongationOperator();
  const RajaOperator* testP  = testFes->GetProlongationOperator();
  RajaOperator *rap = this;
  if (trialP) { rap = new RajaRAPOperator(*testP, *this, *trialP); }
  Aout = new RajaConstrainedOperator(rap, constraintList, rap!=this);
  pop();
}

// ***************************************************************************
void RajaBilinearForm::InitRHS(const Array<int>& constraintList,
                               const RajaVector& x, const RajaVector& b,
                               RajaOperator* A,
                               RajaVector& X, RajaVector& B,
                               int copy_interior) {
  push(SteelBlue);
  const RajaOperator* P = trialFes->GetProlongationOperator();
  const RajaOperator* R = trialFes->GetRestrictionOperator();
  if (P) {
    // Variational restriction with P
    B.SetSize(P->Width());
    P->MultTranspose(b, B);
    X.SetSize(R->Height());
    R->Mult(x, X);
  } else {
    // rap, X and B point to the same data as this, x and b
    X.SetSize(x.Size(),x);
    B.SetSize(b.Size(),b);
  }
  RajaConstrainedOperator* cA = static_cast<RajaConstrainedOperator*>(A);
  if (cA) {
    cA->EliminateRHS(X, B);
  } else {
    mfem_error("RajaBilinearForm::InitRHS expects an RajaConstrainedOperator");
  }
  pop();
}

// ***************************************************************************
void RajaBilinearForm::Mult(const RajaVector& x, RajaVector& y) const {
  push(SteelBlue);
  trialFes->GlobalToLocal(x, localX);
  localY = 0;
  const int integratorCount = (int) integrators.size();
  for (int i = 0; i < integratorCount; ++i) {
    integrators[i]->MultAdd(localX, localY);
  }
  testFes->LocalToGlobal(localY, y);
  pop();
}

// ***************************************************************************
void RajaBilinearForm::MultTranspose(const RajaVector& x, RajaVector& y) const {
  push(SteelBlue);
  testFes->GlobalToLocal(x, localX);
  localY = 0;
  const int integratorCount = (int) integrators.size();
  for (int i = 0; i < integratorCount; ++i) {
    integrators[i]->MultTransposeAdd(localX, localY);
  }
  trialFes->LocalToGlobal(localY, y);
  pop();
}

// ***************************************************************************
void RajaBilinearForm::RecoverFEMSolution(const RajaVector& X,
                                          const RajaVector& b,
                                          RajaVector& x) {
  push(SteelBlue);
  const RajaOperator *P = this->GetProlongation();
  if (P)
  {
    // Apply conforming prolongation
    x.SetSize(P->Height());
    P->Mult(X, x);
  }
  // Otherwise X and x point to the same data
  pop();
}


// ***************************************************************************
// * RajaConstrainedOperator
// ***************************************************************************
RajaConstrainedOperator::RajaConstrainedOperator(RajaOperator* A_,
                                                 const Array<int>& constraintList_,
                                                 bool own_A_) :
  RajaOperator(A_->Height(), A_->Width()) {
  push(SteelBlue);
  Setup(A_, constraintList_, own_A_);
  pop();
}

void RajaConstrainedOperator::Setup(RajaOperator* A_,
                                    const Array<int>& constraintList_,
                                    bool own_A_) {
  push(SteelBlue);
  A = A_;
  own_A = own_A_;
  constraintIndices = constraintList_.Size();
  if (constraintIndices) {
    constraintList.allocate(constraintIndices);
  }
  z.SetSize(height);
  w.SetSize(height);
  pop();
}

void RajaConstrainedOperator::EliminateRHS(const RajaVector& x,
                                           RajaVector& b) const {
  push(SteelBlue);
  w = 0.0;
  A->Mult(w, z);
  b -= z;
  pop();
}

void RajaConstrainedOperator::Mult(const RajaVector& x, RajaVector& y) const {
  push(SteelBlue);
  if (constraintIndices == 0) {
    A->Mult(x, y);
    pop();
    return;
  }
  z = x;
  A->Mult(z, y); // roperator.hpp:76
  pop();
}

} // mfem
