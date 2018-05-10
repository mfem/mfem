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

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include "backend.hpp"
#include "bilininteg.hpp"
#include "../../fem/bilinearform.hpp"

namespace mfem
{

namespace occa
{

OccaBilinearForm::OccaBilinearForm(FiniteElementSpace *ofespace_) :
   Operator(ofespace_->OccaVLayout()),
   localX(ofespace_->OccaEVLayout()),
   localY(ofespace_->OccaEVLayout())
{
   Init(ofespace_->OccaEngine(), ofespace_, ofespace_);
}

OccaBilinearForm::OccaBilinearForm(FiniteElementSpace *otrialFESpace_,
                                   FiniteElementSpace *otestFESpace_) :
   Operator(otrialFESpace_->OccaVLayout(),
            otestFESpace_->OccaVLayout()),
   localX(otrialFESpace_->OccaEVLayout()),
   localY(otestFESpace_->OccaEVLayout())
{
   Init(otrialFESpace_->OccaEngine(), otrialFESpace_, otestFESpace_);
}

void OccaBilinearForm::Init(const Engine &e,
                            FiniteElementSpace *otrialFESpace_,
                            FiniteElementSpace *otestFESpace_)
{
   engine.Reset(&e);

   otrialFESpace = otrialFESpace_;
   trialFESpace  = otrialFESpace_->GetFESpace();

   otestFESpace = otestFESpace_;
   testFESpace  = otestFESpace_->GetFESpace();

   mesh = trialFESpace->GetMesh();

   const int elements = GetNE();

   const int trialVDim = trialFESpace->GetVDim();

   const int trialLocalDofs = otrialFESpace->GetLocalDofs();
   const int testLocalDofs  = otestFESpace->GetLocalDofs();

   // First-touch policy when running with OpenMP
   if (GetDevice().mode() == "OpenMP")
   {
      const std::string &okl_path = OccaEngine().GetOklPath();
      const std::string &okl_defines = OccaEngine().GetOklDefines();
      ::occa::kernel initLocalKernel =
         GetDevice().buildKernel(okl_path + "/utils.okl",
                                 "InitLocalVector",
                                 okl_defines);

      const std::size_t sd = sizeof(double);
      const uint64_t trialEntries = sd * (elements * trialLocalDofs);
      const uint64_t testEntries  = sd * (elements * testLocalDofs);
      for (int v = 0; v < trialVDim; ++v)
      {
         const uint64_t trialOffset = v * trialEntries;
         const uint64_t testOffset  = v * testEntries;

         initLocalKernel(elements, trialLocalDofs,
                         localX.OccaMem().slice(trialOffset, trialEntries));
         initLocalKernel(elements, testLocalDofs,
                         localY.OccaMem().slice(testOffset, testEntries));
      }
   }
}

int OccaBilinearForm::BaseGeom() const
{
   return mesh->GetElementBaseGeometry();
}

int OccaBilinearForm::GetDim() const
{
   return mesh->Dimension();
}

int64_t OccaBilinearForm::GetNE() const
{
   return mesh->GetNE();
}

Mesh& OccaBilinearForm::GetMesh() const
{
   return *mesh;
}

FiniteElementSpace& OccaBilinearForm::GetTrialOccaFESpace() const
{
   return *otrialFESpace;
}

FiniteElementSpace& OccaBilinearForm::GetTestOccaFESpace() const
{
   return *otestFESpace;
}

mfem::FiniteElementSpace& OccaBilinearForm::GetTrialFESpace() const
{
   return *trialFESpace;
}

mfem::FiniteElementSpace& OccaBilinearForm::GetTestFESpace() const
{
   return *testFESpace;
}

int64_t OccaBilinearForm::GetTrialNDofs() const
{
   return trialFESpace->GetNDofs();
}

int64_t OccaBilinearForm::GetTestNDofs() const
{
   return testFESpace->GetNDofs();
}

int64_t OccaBilinearForm::GetTrialVDim() const
{
   return trialFESpace->GetVDim();
}

int64_t OccaBilinearForm::GetTestVDim() const
{
   return testFESpace->GetVDim();
}

const FiniteElement& OccaBilinearForm::GetTrialFE(const int i) const
{
   return *(trialFESpace->GetFE(i));
}

const FiniteElement& OccaBilinearForm::GetTestFE(const int i) const
{
   return *(testFESpace->GetFE(i));
}

// Adds new Domain Integrator.
void OccaBilinearForm::AddDomainIntegrator(OccaIntegrator *integrator,
                                           const ::occa::properties &props)
{
   AddIntegrator(integrator, props, DomainIntegrator);
}

// Adds new Boundary Integrator.
void OccaBilinearForm::AddBoundaryIntegrator(OccaIntegrator *integrator,
                                             const ::occa::properties &props)
{
   AddIntegrator(integrator, props, BoundaryIntegrator);
}

// Adds new interior Face Integrator.
void OccaBilinearForm::AddInteriorFaceIntegrator(OccaIntegrator *integrator,
                                                 const ::occa::properties &props)
{
   AddIntegrator(integrator, props, InteriorFaceIntegrator);
}

// Adds new boundary Face Integrator.
void OccaBilinearForm::AddBoundaryFaceIntegrator(OccaIntegrator *integrator,
                                                 const ::occa::properties &props)
{
   AddIntegrator(integrator, props, BoundaryFaceIntegrator);
}

// Adds Integrator based on OccaIntegratorType
void OccaBilinearForm::AddIntegrator(OccaIntegrator *integrator,
                                     const ::occa::properties &props,
                                     const OccaIntegratorType itype)
{
   if (integrator == NULL)
   {
      std::stringstream error_ss;
      error_ss << "OccaBilinearForm::";
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
   integrator->SetupIntegrator(*this, baseKernelProps + props, itype);
   integrators.push_back(integrator);
}

const mfem::Operator* OccaBilinearForm::GetTrialProlongation() const
{
   return otrialFESpace->GetProlongationOperator();
}

const mfem::Operator* OccaBilinearForm::GetTestProlongation() const
{
   return otestFESpace->GetProlongationOperator();
}

const mfem::Operator* OccaBilinearForm::GetTrialRestriction() const
{
   return otrialFESpace->GetRestrictionOperator();
}

const mfem::Operator* OccaBilinearForm::GetTestRestriction() const
{
   return otestFESpace->GetRestrictionOperator();
}

void OccaBilinearForm::Assemble()
{
   // [MISSING] Find geometric information that is needed by intergrators
   //             to share between integrators.
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->Assemble();
   }
}

void OccaBilinearForm::FormLinearSystem(const mfem::Array<int> &constraintList,
                                        mfem::Vector &x, mfem::Vector &b,
                                        mfem::Operator *&Aout,
                                        mfem::Vector &X, mfem::Vector &B,
                                        int copy_interior)
{
   FormOperator(constraintList, Aout);
   InitRHS(constraintList, x, b, Aout, X, B, copy_interior);
}

void OccaBilinearForm::FormOperator(const mfem::Array<int> &constraintList,
                                    mfem::Operator *&Aout)
{
   const mfem::Operator *trialP = GetTrialProlongation();
   const mfem::Operator *testP  = GetTestProlongation();
   mfem::Operator *rap = this;

   if (trialP)
   {
      rap = new RAPOperator(*testP, *this, *trialP);
   }

   Aout = new OccaConstrainedOperator(rap, constraintList,
                                      rap != this);
}

void OccaBilinearForm::InitRHS(const mfem::Array<int> &constraintList,
                               mfem::Vector &x, mfem::Vector &b,
                               mfem::Operator *A,
                               mfem::Vector &X, mfem::Vector &B,
                               int copy_interior)
{
   const std::string okl_defines = OccaEngine().GetOklDefines();

   // FIXME: move these kernels to the Backend?
   static ::occa::kernelBuilder get_subvector_builder =
      ::occa::linalg::customLinearMethod(
         "vector_get_subvector",

         "const int dof_i = v2[i];"
         "v0[i] = dof_i >= 0 ? v1[dof_i] : -v1[-dof_i - 1];",

         "defines: {"
         "  VTYPE0: 'double',"
         "  VTYPE1: 'double',"
         "  VTYPE2: 'int',"
         "  TILESIZE: 128,"
         "}" + okl_defines);

   static ::occa::kernelBuilder set_subvector_builder =
      ::occa::linalg::customLinearMethod(
         "vector_set_subvector",
         "const int dof_i = v2[i];"
         "if (dof_i >= 0) { v0[dof_i]      = v1[i]; }"
         "else            { v0[-dof_i - 1] = -v1[i]; }",

         "defines: {"
         "  VTYPE0: 'double',"
         "  VTYPE1: 'double',"
         "  VTYPE2: 'int',"
         "  TILESIZE: 128,"
         "}" + okl_defines);

   const mfem::Operator *P = GetTrialProlongation();
   const mfem::Operator *R = GetTrialRestriction();

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

   if (!copy_interior && constraintList.Size() > 0)
   {
      ::occa::kernel get_subvector_kernel =
            get_subvector_builder.build(GetDevice());
      ::occa::kernel set_subvector_kernel =
            set_subvector_builder.build(GetDevice());

      const Array &constrList = constraintList.Get_PArray()->As<Array>();
      Vector subvec(constrList.OccaLayout());

      get_subvector_kernel(constraintList.Size(),
                           subvec.OccaMem(),
                           X.Get_PVector()->As<Vector>().OccaMem(),
                           constrList.OccaMem());

      X.Fill(0.0);

      set_subvector_kernel(constraintList.Size(),
                           X.Get_PVector()->As<Vector>().OccaMem(),
                           subvec.OccaMem(),
                           constrList.OccaMem());
   }

   OccaConstrainedOperator *cA = dynamic_cast<OccaConstrainedOperator*>(A);
   if (cA)
   {
      cA->EliminateRHS(X.Get_PVector()->As<Vector>(),
                       B.Get_PVector()->As<Vector>());
   }
   else
   {
      mfem_error("OccaBilinearForm::InitRHS expects an OccaConstrainedOperator");
   }
}

// Matrix vector multiplication.
void OccaBilinearForm::Mult_(const Vector &x, Vector &y) const
{
   otrialFESpace->GlobalToLocal(x, localX);
   localY.Fill<double>(0.0);

   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->MultAdd(localX, localY);
   }

   otestFESpace->LocalToGlobal(localY, y);
}

// Matrix transpose vector multiplication.
void OccaBilinearForm::MultTranspose_(const Vector &x, Vector &y) const
{
   otestFESpace->GlobalToLocal(x, localX);
   localY.Fill<double>(0.0);

   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->MultTransposeAdd(localX, localY);
   }

   otrialFESpace->LocalToGlobal(localY, y);
}

void OccaBilinearForm::OccaRecoverFEMSolution(const mfem::Vector &X,
                                              const mfem::Vector &b,
                                              mfem::Vector &x)
{
   const mfem::Operator *P = this->GetTrialProlongation();
   if (P)
   {
      // Apply conforming prolongation
      x.Resize(P->OutLayout());
      P->Mult(X, x);
   }
   // Otherwise X and x point to the same data
}

// Frees memory bilinear form.
OccaBilinearForm::~OccaBilinearForm()
{
   // Make sure all integrators free their data
   IntegratorVector::iterator it = integrators.begin();
   while (it != integrators.end())
   {
      delete *it;
      ++it;
   }
}


void BilinearForm::InitOccaBilinearForm()
{
   // Init 'obform' using 'bform'
   MFEM_ASSERT(bform != NULL, "");
   MFEM_ASSERT(obform == NULL, "");

   FiniteElementSpace &ofes =
      bform->FESpace()->Get_PFESpace()->As<FiniteElementSpace>();
   obform = new OccaBilinearForm(&ofes);

   // Transfer domain integrators
   mfem::Array<mfem::BilinearFormIntegrator*> &dbfi = *bform->GetDBFI();
   for (int i = 0; i < dbfi.Size(); i++)
   {
      std::string integ_name(dbfi[i]->Name());
      Coefficient *scal_coeff = dbfi[i]->GetScalarCoefficient();
      ConstantCoefficient *const_coeff =
         dynamic_cast<ConstantCoefficient*>(scal_coeff);
      // TODO: other types of coefficients ...
      double val = const_coeff ? const_coeff->constant : 1.0;
      OccaCoefficient ocoeff(obform->OccaEngine(), val);

      OccaIntegrator *ointeg = NULL;

      if (integ_name == "(undefined)")
      {
         MFEM_ABORT("BilinearFormIntegrator does not define Name()");
      }
      else if (integ_name == "diffusion")
      {
         ointeg = new OccaDiffusionIntegrator(ocoeff);
      }
      else
      {
         MFEM_ABORT("BilinearFormIntegrator [Name() = " << integ_name
                    << "] is not supported");
      }

      const mfem::IntegrationRule *ir = dbfi[i]->GetIntRule();
      if (ir) { ointeg->SetIntegrationRule(*ir); }

      obform->AddDomainIntegrator(ointeg);
   }

   // TODO: other types of integrators ...
}

bool BilinearForm::Assemble()
{
   if (obform == NULL) { InitOccaBilinearForm(); }

   obform->Assemble();

   return true; // --> host assembly is not needed
}

void BilinearForm::FormSystemMatrix(const mfem::Array<int> &ess_tdof_list,
                                    mfem::OperatorHandle &A)
{
   if (A.Type() == mfem::Operator::ANY_TYPE)
   {
      mfem::Operator *Aout = NULL;
      obform->FormOperator(ess_tdof_list, Aout);
      A.Reset(Aout);
   }
   else
   {
      MFEM_ABORT("Operator::Type is not supported, type = " << A.Type());
   }
}

void BilinearForm::FormLinearSystem(const mfem::Array<int> &ess_tdof_list,
                                    mfem::Vector &x, mfem::Vector &b,
                                    mfem::OperatorHandle &A,
                                    mfem::Vector &X, mfem::Vector &B,
                                    int copy_interior)
{
   FormSystemMatrix(ess_tdof_list, A);
   obform->InitRHS(ess_tdof_list, x, b, A.Ptr(), X, B, copy_interior);
}

void BilinearForm::RecoverFEMSolution(const mfem::Vector &X,
                                      const mfem::Vector &b,
                                      mfem::Vector &x)
{
   obform->OccaRecoverFEMSolution(X, b, x);
}

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)
