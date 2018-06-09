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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../raja.hpp"

namespace mfem
{

namespace raja
{

// ***************************************************************************
// * RajaParBilinearForm
// ***************************************************************************
RajaParBilinearForm::RajaParBilinearForm(RajaParFiniteElementSpace *pfes) :
   Operator(pfes->RajaVLayout()),
   mesh(pfes->GetFESpace()->GetMesh()),
   rtrialFESpace(pfes),
   trialFESpace(pfes->GetParFESpace()),
   rtestFESpace(pfes),
   testFESpace(pfes->GetParFESpace()),
   localX(pfes->RajaEVLayout()),
   localY(pfes->RajaEVLayout()) {
   dbg("\n\033[31m[RajaParBilinearForm]");
}

// ***************************************************************************
RajaParBilinearForm::~RajaParBilinearForm() { }

// ***************************************************************************
// Adds new Domain Integrator.
void RajaParBilinearForm::AddDomainIntegrator(RajaIntegrator* i)
{
   push(SteelBlue);
   AddIntegrator(i, DomainIntegrator);
   pop();
}

// Adds new Boundary Integrator.
void RajaParBilinearForm::AddBoundaryIntegrator(RajaIntegrator* i)
{
   push(SteelBlue);
   AddIntegrator(i, BoundaryIntegrator);
   pop();
}

// Adds new interior Face Integrator.
void RajaParBilinearForm::AddInteriorFaceIntegrator(RajaIntegrator* i)
{
   push(SteelBlue);
   AddIntegrator(i, InteriorFaceIntegrator);
   pop();
}

// Adds new boundary Face Integrator.
void RajaParBilinearForm::AddBoundaryFaceIntegrator(RajaIntegrator* i)
{
   push(SteelBlue);
   AddIntegrator(i, BoundaryFaceIntegrator);
   pop();
}

// Adds Integrator based on RajaIntegratorType
void RajaParBilinearForm::AddIntegrator(RajaIntegrator* i,
                                     const RajaIntegratorType itype)
{
   push(SteelBlue);
   assert(i);
   assert(false);//i->SetupIntegrator(*this, itype);
   integrators.push_back(i);
   pop();
}
// *****************************************************************************
const mfem::Operator* RajaParBilinearForm::GetTrialProlongation() const
{
   return rtrialFESpace->GetProlongationOperator();
}

const mfem::Operator* RajaParBilinearForm::GetTestProlongation() const
{
   return rtestFESpace->GetProlongationOperator();
}

const mfem::Operator* RajaParBilinearForm::GetTrialRestriction() const
{
   return rtrialFESpace->GetRestrictionOperator();
}

const mfem::Operator* RajaParBilinearForm::GetTestRestriction() const
{
   return rtestFESpace->GetRestrictionOperator();
}

// ***************************************************************************
void RajaParBilinearForm::Assemble()
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
void RajaParBilinearForm::FormLinearSystem(const mfem::Array<int>& constraintList,
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
void RajaParBilinearForm::FormOperator(const mfem::Array<int>& constraintList,
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
void RajaParBilinearForm::InitRHS(const mfem::Array<int>& constraintList,
                               const mfem::Vector& x, const mfem::Vector& b,
                               mfem::Operator* A,
                               mfem::Vector& X, mfem::Vector& B,
                               int copy_interior)
{
/*   push(SteelBlue);
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
      mfem_error("RajaParBilinearForm::InitRHS expects an RajaConstrainedOperator");
      }*/
   pop();
}

// ***************************************************************************
void RajaParBilinearForm::Mult_(const Vector& x, Vector& y) const
{
   push(SteelBlue);
   assert(false);//trialFes->GlobalToLocal(x, localX);
   localY.Fill<double>(0.0);
   //localY = 0;
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->MultAdd(localX, localY);
   }
   assert(false);//testFes->LocalToGlobal(localY, y);
   pop();
}

// ***************************************************************************
void RajaParBilinearForm::MultTranspose_(const Vector& x, Vector& y) const
{
   push(SteelBlue);
   assert(false);//testFes->GlobalToLocal(x, localX);
   localY.Fill<double>(0.0);
   //localY = 0;
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->MultTransposeAdd(localX, localY);
   }
   assert(false);//trialFes->LocalToGlobal(localY, y);
   pop();
}

// ***************************************************************************
/*void RajaParBilinearForm::RecoverFEMSolution(const RajaVector& X,
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
   }*/
   
   // **************************************************************************
   void RajaParBilinearForm::RecoverFEMSolution(const mfem::Vector &x,
                                                const mfem::Vector &b,
                                                mfem::Vector &y){
      assert(false);
   }

/*
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
*/
} // raja
   
} // mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
