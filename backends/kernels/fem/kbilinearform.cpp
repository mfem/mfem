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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

// *****************************************************************************
kBilinearForm::kBilinearForm(kFiniteElementSpace *ofespace_) :
   Operator(ofespace_->KernelsVLayout()),
   localX(ofespace_->KernelsEVLayout()),
   localY(ofespace_->KernelsEVLayout())
{
   push();
   Init(ofespace_->KernelsEngine(), ofespace_, ofespace_);
   pop();
}

// *****************************************************************************
kBilinearForm::kBilinearForm(kFiniteElementSpace
                             *otrialFESpace_,
                             kFiniteElementSpace *otestFESpace_) :
   Operator(otrialFESpace_->KernelsVLayout(), otestFESpace_->KernelsVLayout()),
   localX(otrialFESpace_->KernelsEVLayout()),
   localY(otestFESpace_->KernelsEVLayout())
{
   push();
   Init(otrialFESpace_->KernelsEngine(), otrialFESpace_, otestFESpace_);
   pop();
}

// *****************************************************************************
void kBilinearForm::Init(const Engine &e,
                         kFiniteElementSpace *otrialFESpace_,
                         kFiniteElementSpace *otestFESpace_)
{
   push();
   ng.Reset(&e);
   otrialFESpace = otrialFESpace_;
   trialFESpace  = otrialFESpace_->GetFESpace();

   otestFESpace = otestFESpace_;
   testFESpace  = otestFESpace_->GetFESpace();

   mesh = trialFESpace->GetMesh();
   pop();
}

// *****************************************************************************
kFiniteElementSpace& kBilinearForm::GetTrialKernelsFESpace() const
{
   push(); pop();
   return *otrialFESpace;
}

// *****************************************************************************
kFiniteElementSpace& kBilinearForm::GetTestKernelsFESpace() const
{
   push(); pop();
   return *otestFESpace;
}

// *****************************************************************************
mfem::FiniteElementSpace& kBilinearForm::GetTrialFESpace() const
{
   push(); pop();
   assert(trialFESpace);
   return *trialFESpace;
}

// *****************************************************************************
mfem::FiniteElementSpace& kBilinearForm::GetTestFESpace() const
{
   push(); pop();
   assert(testFESpace);
   return *testFESpace;
}

int64_t kBilinearForm::GetTrialNDofs() const
{
   push(); pop();
   return trialFESpace->GetNDofs();
}

int64_t kBilinearForm::GetTestNDofs() const
{
   return testFESpace->GetNDofs();
}

int64_t kBilinearForm::GetTrialVDim() const
{
   return trialFESpace->GetVDim();
}

int64_t kBilinearForm::GetTestVDim() const
{
   return testFESpace->GetVDim();
}

const FiniteElement& kBilinearForm::GetTrialFE(const int i) const
{
   return *(trialFESpace->GetFE(i));
}

const FiniteElement& kBilinearForm::GetTestFE(const int i) const
{
   return *(testFESpace->GetFE(i));
}

// Adds new Domain Integrator.
void kBilinearForm::AddDomainIntegrator(KernelsIntegrator *integrator)
{
   push();
   AddIntegrator(integrator, DomainIntegrator);
   pop();
}

// Adds new Boundary Integrator.
void kBilinearForm::AddBoundaryIntegrator(KernelsIntegrator *integrator)
{
   push();
   AddIntegrator(integrator, BoundaryIntegrator);
   pop();
}

// Adds new interior Face Integrator.
void kBilinearForm::AddInteriorFaceIntegrator(KernelsIntegrator
                                                    *integrator)
{
   push();
   AddIntegrator(integrator, InteriorFaceIntegrator);
   pop();
}

// Adds new boundary Face Integrator.
void kBilinearForm::AddBoundaryFaceIntegrator(KernelsIntegrator
                                                    *integrator)
{
   push();
   AddIntegrator(integrator, BoundaryFaceIntegrator);
   pop();
}

// Adds Integrator based on KernelsIntegratorType
void kBilinearForm::AddIntegrator(KernelsIntegrator *integrator,
                                        const KernelsIntegratorType itype)
{
   push();
   if (integrator == NULL)
   {
      std::stringstream error_ss;
      error_ss << "kBilinearForm::";
      switch (itype)
      {
         case DomainIntegrator      : error_ss << "AddDomainIntegrator";       break;
         case BoundaryIntegrator    : error_ss << "AddBoundaryIntegrator";     break;
         case InteriorFaceIntegrator: error_ss << "AddInteriorFaceIntegrator"; break;
         case BoundaryFaceIntegrator: error_ss << "AddBoundaryFaceIntegrator"; break;
      }
      error_ss << " (...):\n"
               << "  Integrator is NULL";
      const std::string error = error_ss.str();
      mfem_error(error.c_str());
   }
   integrator->SetupIntegrator(*this, itype);
   integrators.push_back(integrator);
   pop();
}

const mfem::Operator* kBilinearForm::GetTrialProlongation() const
{
   return otrialFESpace->GetProlongationOperator();
}

const mfem::Operator* kBilinearForm::GetTestProlongation() const
{
   return otestFESpace->GetProlongationOperator();
}

const mfem::Operator* kBilinearForm::GetTrialRestriction() const
{
   return otrialFESpace->GetRestrictionOperator();
}

const mfem::Operator* kBilinearForm::GetTestRestriction() const
{
   return otestFESpace->GetRestrictionOperator();
}

// *****************************************************************************
void kBilinearForm::Assemble()
{
   push();
   // [MISSING] Find geometric information that is needed by intergrators
   //             to share between integrators.
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->Assemble();
   }
   pop();
}

// *****************************************************************************
void kBilinearForm::FormLinearSystem(const mfem::Array<int>
                                           &constraintList,
                                           mfem::Vector &x, mfem::Vector &b,
                                           mfem::Operator *&Aout,
                                           mfem::Vector &X, mfem::Vector &B,
                                           int copy_interior)
{
   push();
   assert(false);
   FormOperator(constraintList, Aout);
   InitRHS(constraintList, x, b, Aout, X, B, copy_interior);
   pop();
}

// *****************************************************************************
void kBilinearForm::FormOperator(const mfem::Array<int> &constraintList,
                                       mfem::Operator *&Aout)
{
   push();
   //assert(constraintList.Size()==0);
   const mfem::Operator *trialP = GetTrialProlongation();
   const mfem::Operator *testP  = GetTestProlongation();
   mfem::Operator *rap = this;
   if (trialP)
   {
      rap = new RAPOperator(*testP, *this, *trialP);
   }
   dbg("KernelsConstrainedOperator");
   Aout = new KernelsConstrainedOperator(rap, constraintList,
                                         rap != this);
   pop();
}

// *****************************************************************************
void kBilinearForm::InitRHS(const mfem::Array<int> &constraintList,
                                  mfem::Vector &x, mfem::Vector &b,
                                  mfem::Operator *A,
                                  mfem::Vector &X, mfem::Vector &B,
                                  int copy_interior)
{
  push(); //assert(false);
   const mfem::Operator *P = GetTrialProlongation();
   const mfem::Operator *R = GetTrialRestriction();
   if (P) {
      // Variational restriction with P
      B.Resize(P->InLayout());
      P->MultTranspose(b, B);
      X.Resize(R->OutLayout());
      R->Mult(x, X);
   } else {
      // rap, X and B point to the same data as this, x and b
      assert(false);
      X.MakeRef(x);
      B.MakeRef(b);
   }

   if (!copy_interior && constraintList.Size() > 0)
   {
     //assert(false);
      const Array &constrList = constraintList.Get_PArray()->As<Array>();
      Vector subvec(constrList.KernelsLayout());
      vector_get_subvector(constraintList.Size(),
                           (double*)subvec.KernelsMem().ptr(),
                           (double*)X.Get_PVector()->As<Vector>().KernelsMem().ptr(),
                           (int*)constrList.KernelsMem().ptr());
      X.Fill(0.0);
      vector_set_subvector(constraintList.Size(),
                           (double*)X.Get_PVector()->As<Vector>().KernelsMem().ptr(),
                           (double*)subvec.KernelsMem().ptr(),
                           (int*)constrList.KernelsMem().ptr());
   }

   KernelsConstrainedOperator *cA = dynamic_cast<KernelsConstrainedOperator*>(A);
   if (cA)
   {
      cA->EliminateRHS(X.Get_PVector()->As<Vector>(),
                       B.Get_PVector()->As<Vector>());
   }
   else
   {
      mfem_error("kBilinearForm::InitRHS expects an KernelsConstrainedOperator");
   }
   pop();
}


// Matrix vector multiplication ************************************************
void kBilinearForm::Mult_(const Vector &x, Vector &y) const
{
   dbg("\033[7mkBilinearForm::Mult_");
   otrialFESpace->GlobalToLocal(x, localX);
   localY.Fill<double>(0.0);
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i) {
      integrators[i]->MultAdd(localX, localY);
   }
   otestFESpace->LocalToGlobal(localY, y);
   pop();
}

// Matrix transpose vector multiplication **************************************
void kBilinearForm::MultTranspose_(const Vector &x, Vector &y) const
{
   push();
   otestFESpace->GlobalToLocal(x, localX);
   localY.Fill<double>(0.0);
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i){
      integrators[i]->MultTransposeAdd(localX, localY);
   }
   otrialFESpace->LocalToGlobal(localY, y);
   pop();
}

// *****************************************************************************
void kBilinearForm::KernelsRecoverFEMSolution(const mfem::Vector &X,
                                                    const mfem::Vector &b,
                                                    mfem::Vector &x)
{
   push();
   const mfem::Operator *P = this->GetTrialProlongation();
   if (P) {
      // Apply conforming prolongation
      x.Resize(P->OutLayout());
      P->Mult(X, x);
   }
   // Otherwise X and x point to the same data
   pop();
}

// Frees memory bilinear form **************************************************
kBilinearForm::~kBilinearForm()
{
   // Make sure all integrators free their data
   IntegratorVector::iterator it = integrators.begin();
   while (it != integrators.end())
   {
      delete *it;
      ++it;
   }
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
