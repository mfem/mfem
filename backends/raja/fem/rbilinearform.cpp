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

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{

// ***************************************************************************
// * RajaBilinearForm
// ***************************************************************************
RajaBilinearForm::RajaBilinearForm(FiniteElementSpace *fes) :
   Operator(fes->RajaVLayout),
   mesh(fes->GetMesh()),
   rtrialFESpace(fes),
   trialFESpace(fes->GetFESpace()),
   rtestFESpace(fes),
   testFESpace(fes->GetFESpace()),
   localX(fes->RajaEVLayout()),
   localY(fes->RajaEVLayout()) {
   dbg("\033[31m[RajaBilinearForm]");
}

// ***************************************************************************
RajaBilinearForm::~RajaBilinearForm() { }

// ***************************************************************************
// Adds new Domain Integrator.
void RajaBilinearForm::AddDomainIntegrator(RajaIntegrator* i)
{
   push(SteelBlue);
   AddIntegrator(i, DomainIntegrator);
   pop();
}

// Adds new Boundary Integrator.
void RajaBilinearForm::AddBoundaryIntegrator(RajaIntegrator* i)
{
   push(SteelBlue);
   AddIntegrator(i, BoundaryIntegrator);
   pop();
}

// Adds new interior Face Integrator.
void RajaBilinearForm::AddInteriorFaceIntegrator(RajaIntegrator* i)
{
   push(SteelBlue);
   AddIntegrator(i, InteriorFaceIntegrator);
   pop();
}

// Adds new boundary Face Integrator.
void RajaBilinearForm::AddBoundaryFaceIntegrator(RajaIntegrator* i)
{
   push(SteelBlue);
   AddIntegrator(i, BoundaryFaceIntegrator);
   pop();
}

// Adds Integrator based on RajaIntegratorType
void RajaBilinearForm::AddIntegrator(RajaIntegrator* i,
                                     const RajaIntegratorType itype)
{
   push(SteelBlue);
   assert(i);
   i->SetupIntegrator(*this, itype);
   integrators.push_back(i);
   pop();
}
// *****************************************************************************
const mfem::Operator* RajaBilinearForm::GetTrialProlongation() const
{
   return rtrialFESpace->GetProlongationOperator();
}

const mfem::Operator* RajaBilinearForm::GetTestProlongation() const
{
   return rtestFESpace->GetProlongationOperator();
}

const mfem::Operator* RajaBilinearForm::GetTrialRestriction() const
{
   return rtrialFESpace->GetRestrictionOperator();
}

const mfem::Operator* RajaBilinearForm::GetTestRestriction() const
{
   return rtestFESpace->GetRestrictionOperator();
}

// ***************************************************************************
void RajaBilinearForm::Assemble()
{
   push(SteelBlue);
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->Assemble();
   }
   pop();
}

// ***************************************************************************
void RajaBilinearForm::FormLinearSystem(const mfem::Array<int>& constraintList,
                                        mfem::Vector& x, mfem::Vector& b,
                                        mfem::Operator*& Aout,
                                        mfem::Vector& X, mfem::Vector& B,
                                        int copy_interior)
{
   push(SteelBlue);
   FormOperator(constraintList, Aout);
   InitRHS(constraintList, x, b, Aout, X, B, copy_interior);
   pop();
}

// ***************************************************************************
void RajaBilinearForm::FormOperator(const mfem::Array<int>& constraintList,
                                    mfem::Operator*& Aout)
{
   push(SteelBlue);
   const mfem::Operator* trialP = GetTrialProlongation();
   const mfem::Operator* testP  = GetTestProlongation();
   mfem::Operator *rap = this;
   if (trialP) { rap = new RAPOperator(*testP, *this, *trialP); }
   Aout = new RajaConstrainedOperator(rap, constraintList, rap!=this);
   pop();
}

// ***************************************************************************
void RajaBilinearForm::InitRHS(const mfem::Array<int>& constraintList,
                               const mfem::Vector& x, const mfem::Vector& b,
                               mfem::Operator* A,
                               mfem::Vector& X, mfem::Vector& B,
                               int copy_interior)
{
   push(SteelBlue);
   const mfem::Operator* P = GetTrialProlongation();
   const mfem::Operator* R = GetTrialRestriction();
   if (P)
   {
      // Variational restriction with P
      B.Resize(P->InLayout());
      P->MultTranspose(b, B);
      X.Resize(R->OutLayout());
      R->Mult(x, X);
   }
   else
   {
      // rap, X and B point to the same data as this, x and b
      X.MakeRef(x);
      B.MakeRef(b);
   }
   RajaConstrainedOperator* cA = static_cast<RajaConstrainedOperator*>(A);
   if (cA)
   {
      cA->EliminateRHS(X.Get_PVector()->As<Vector>(),
                       B.Get_PVector()->As<Vector>());
      //cA->EliminateRHS(X, B);
   }
   else
   {
      mfem_error("RajaBilinearForm::InitRHS expects an RajaConstrainedOperator");
   }
   pop();
}

// ***************************************************************************
void RajaBilinearForm::Mult_(const Vector& x, Vector& y) const
{
   push(SteelBlue);
   trialFes->GlobalToLocal(x, localX);
   localY = 0;
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->MultAdd(localX, localY);
   }
   testFes->LocalToGlobal(localY, y);
   pop();
}

// ***************************************************************************
void RajaBilinearForm::MultTranspose_(const Vector& x, Vector& y) const
{
   push(SteelBlue);
   testFes->GlobalToLocal(x, localX);
   localY = 0;
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->MultTransposeAdd(localX, localY);
   }
   trialFes->LocalToGlobal(localY, y);
   pop();
}

// ***************************************************************************
void RajaBilinearForm::RecoverFEMSolution(const RajaVector& X,
                                          const RajaVector& b,
                                          RajaVector& x)
{
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
RajaConstrainedOperator::RajaConstrainedOperator(mfem::Operator* A_,
                                                 const mfem::Array<int>& constraintList_,
                                                 bool own_A_) :
   Operator(A_->Height(), A_->Width())
{
   push(SteelBlue);
   Setup(A_, constraintList_, own_A_);
   pop();
}

void RajaConstrainedOperator::Setup(mfem::Operator* A_,
                                    const mfem::Array<int>& constraintList_,
                                    bool own_A_)
{
   push(SteelBlue);
   A = A_;
   own_A = own_A_;
   constraintIndices = constraintList_.Size();
   if (constraintIndices)
   {
      constraintList.allocate(constraintIndices);
   }
   z.SetSize(height);
   w.SetSize(height);
   pop();
}

void RajaConstrainedOperator::EliminateRHS(const Vector& x,
                                           Vector& b) const
{
   push(SteelBlue);
   w = 0.0;
   A->Mult(w, z);
   b -= z;
   pop();
}

void RajaConstrainedOperator::Mult_(const Vector& x, Vector& y) const
{
   push(SteelBlue);
   if (constraintIndices == 0)
   {
      A->Mult(x, y);
      pop();
      return;
   }
   z = x;
   A->Mult(z, y); // roperator.hpp:76
   pop();
}
   
} // raja
   
} // mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
