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

#include "../config/config.hpp"

#ifdef MFEM_USE_OCCA

#include "obilinearform.hpp"
#include "obilininteg.hpp"
#include "../linalg/osparsemat.hpp"

#include "tfe.hpp"

namespace mfem {
  //---[ Bilinear Form ]----------------
  OccaBilinearForm::OccaBilinearForm(OccaFiniteElementSpace *ofespace_) :
    Operator(ofespace_->GetVSize(),
             ofespace_->GetVSize()) {
    Init(occa::getDevice(), ofespace_, ofespace_);
  }

  OccaBilinearForm::OccaBilinearForm(occa::device device_,
                                     OccaFiniteElementSpace *ofespace_) :
    Operator(ofespace_->GetVSize(),
             ofespace_->GetVSize()) {
    Init(device, ofespace_, ofespace_);
  }

  OccaBilinearForm::OccaBilinearForm(OccaFiniteElementSpace *otrialFespace_,
                                     OccaFiniteElementSpace *otestFespace_) :
    Operator(otrialFespace_->GetVSize(),
             otestFespace_->GetVSize()) {
    Init(occa::getDevice(), otrialFespace_, otestFespace_);
  }

  OccaBilinearForm::OccaBilinearForm(occa::device device_,
                                     OccaFiniteElementSpace *otrialFespace_,
                                     OccaFiniteElementSpace *otestFespace_) :
    Operator(otrialFespace_->GetVSize(),
             otestFespace_->GetVSize()) {
    Init(device, otrialFespace_, otestFespace_);
  }

  void OccaBilinearForm::Init(occa::device device_,
                              OccaFiniteElementSpace *otrialFespace_,
                              OccaFiniteElementSpace *otestFespace_) {
    device = device_;

    otrialFespace = otrialFespace_;
    trialFespace  = otrialFespace_->GetFESpace();

    otestFespace = otestFespace_;
    testFespace  = otestFespace_->GetFESpace();

    mesh = trialFespace->GetMesh();

    const int elements = GetNE();

    const int trialVDim = trialFespace->GetVDim();
    const int testVDim  = testFespace->GetVDim();

    const int trialLocalDofs = otrialFespace->GetLocalDofs();
    const int testLocalDofs  = otestFespace->GetLocalDofs();

    const int trialElementEntries = (trialLocalDofs * trialVDim);
    const int testElementEntries  = (testLocalDofs * testVDim);

    localX.SetSize(device, elements * trialElementEntries);
    localY.SetSize(device, elements * testElementEntries);

    // First-touch policy when running with OpenMP
    if (device.mode() == "OpenMP") {
      occa::kernel initLocalKernel = device.buildKernel("occa://mfem/fem/utils.okl",
                                                        "InitLocalVector");

      const uint64_t trialEntries = (elements * trialLocalDofs);
      const uint64_t testEntries  = (elements * testLocalDofs);
      for (int v = 0; v < trialVDim; ++v) {
        const uint64_t trialOffset = v * (elements * trialLocalDofs);
        const uint64_t testOffset  = v * (elements * testLocalDofs);

        initLocalKernel(elements, trialLocalDofs, localX.GetRange(trialOffset, trialEntries));
        initLocalKernel(elements, testLocalDofs , localY.GetRange(testOffset , testEntries));
      }
    }
  }

  occa::device OccaBilinearForm::GetDevice() {
    return device;
  }

  int OccaBilinearForm::BaseGeom() const {
    return mesh->GetElementBaseGeometry();
  }

  int OccaBilinearForm::GetDim() const {
    return mesh->Dimension();
  }

  int64_t OccaBilinearForm::GetNE() const {
    return mesh->GetNE();
  }

  Mesh& OccaBilinearForm::GetMesh() const {
    return *mesh;
  }

  FiniteElementSpace& OccaBilinearForm::GetTrialFESpace() const {
    return *trialFespace;
  }

  FiniteElementSpace& OccaBilinearForm::GetTestFESpace() const {
    return *testFespace;
  }

  OccaFiniteElementSpace& OccaBilinearForm::GetTrialOccaFESpace() const {
    return *otrialFespace;
  }

  OccaFiniteElementSpace& OccaBilinearForm::GetTestOccaFESpace() const {
    return *otestFespace;
  }

  int64_t OccaBilinearForm::GetTrialNDofs() const {
    return trialFespace->GetNDofs();
  }

  int64_t OccaBilinearForm::GetTestNDofs() const {
    return testFespace->GetNDofs();
  }

  int64_t OccaBilinearForm::GetTrialVDim() const {
    return trialFespace->GetVDim();
  }

  int64_t OccaBilinearForm::GetTestVDim() const {
    return testFespace->GetVDim();
  }

  const FiniteElement& OccaBilinearForm::GetTrialFE(const int i) const {
    return *(trialFespace->GetFE(i));
  }

  const FiniteElement& OccaBilinearForm::GetTestFE(const int i) const {
    return *(testFespace->GetFE(i));
  }

  // Adds new Domain Integrator.
  void OccaBilinearForm::AddDomainIntegrator(OccaIntegrator *integrator,
                                             const occa::properties &props) {
    AddIntegrator(integrator, props, DomainIntegrator);
  }

  // Adds new Boundary Integrator.
  void OccaBilinearForm::AddBoundaryIntegrator(OccaIntegrator *integrator,
                                               const occa::properties &props) {
    AddIntegrator(integrator, props, BoundaryIntegrator);
  }

  // Adds new interior Face Integrator.
  void OccaBilinearForm::AddInteriorFaceIntegrator(OccaIntegrator *integrator,
                                                   const occa::properties &props) {
    AddIntegrator(integrator, props, InteriorFaceIntegrator);
  }

  // Adds new boundary Face Integrator.
  void OccaBilinearForm::AddBoundaryFaceIntegrator(OccaIntegrator *integrator,
                                                   const occa::properties &props) {
    AddIntegrator(integrator, props, BoundaryFaceIntegrator);
  }

  // Adds Integrator based on OccaIntegratorType
  void OccaBilinearForm::AddIntegrator(OccaIntegrator *integrator,
                                       const occa::properties &props,
                                       const OccaIntegratorType itype) {
    if (integrator == NULL) {
      std::stringstream error_ss;
      error_ss << "OccaBilinearForm::";
      switch (itype) {
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

  const Operator* OccaBilinearForm::GetTrialProlongation() const {
    return otrialFespace->GetProlongationOperator();
  }

  const Operator* OccaBilinearForm::GetTestProlongation() const {
    return otestFespace->GetProlongationOperator();
  }

  const Operator* OccaBilinearForm::GetTrialRestriction() const {
    return otrialFespace->GetRestrictionOperator();
  }

  const Operator* OccaBilinearForm::GetTestRestriction() const {
    return otestFespace->GetRestrictionOperator();
  }

  //
  void OccaBilinearForm::Assemble() {
    // [MISSING] Find geometric information that is needed by intergrators
    //             to share between integrators.
    const int integratorCount = (int) integrators.size();
    for (int i = 0; i < integratorCount; ++i) {
      integrators[i]->Assemble();
    }
  }

  void OccaBilinearForm::FormLinearSystem(const Array<int> &constraintList,
                                          OccaVector &x, OccaVector &b,
                                          Operator *&Aout,
                                          OccaVector &X, OccaVector &B,
                                          int copy_interior) {
    FormOperator(constraintList, Aout);
    InitRHS(constraintList, x, b, Aout, X, B, copy_interior);
  }

  void OccaBilinearForm::FormOperator(const Array<int> &constraintList,
                                      Operator *&Aout) {
    const Operator *trialP = GetTrialProlongation();
    const Operator *testP  = GetTestProlongation();
    Operator *rap = this;

    if (trialP) {
      rap = new OccaRAPOperator(*testP, *this, *trialP);
    }

    Aout = new OccaConstrainedOperator(device,
                                       rap, constraintList,
                                       rap != this);
  }

  void OccaBilinearForm::InitRHS(const Array<int> &constraintList,
                                 OccaVector &x, OccaVector &b,
                                 Operator *A,
                                 OccaVector &X, OccaVector &B,
                                 int copy_interior) {
    const Operator *P = GetTrialProlongation();
    const Operator *R = GetTrialRestriction();

    if (P) {
      // Variational restriction with P
      B.SetSize(device, P->Width());
      P->MultTranspose(b, B);
      X.SetSize(device, R->Height());
      R->Mult(x, X);
    } else {
      // rap, X and B point to the same data as this, x and b
      X.NewDataAndSize(x.GetData(), x.Size());
      B.NewDataAndSize(b.GetData(), b.Size());
    }

    if (!copy_interior) {
      X.SetSubVectorComplement(constraintList, 0.0);
    }

    OccaConstrainedOperator *cA = static_cast<OccaConstrainedOperator*>(A);
    if (cA) {
      cA->EliminateRHS(X, B);
    } else {
      mfem_error("OccaBilinearForm::InitRHS expects an OccaConstrainedOperator");
    }
  }

  // Matrix vector multiplication.
  void OccaBilinearForm::Mult(const OccaVector &x, OccaVector &y) const {
    otrialFespace->GlobalToLocal(x, localX);
    localY = 0;

    const int integratorCount = (int) integrators.size();
    for (int i = 0; i < integratorCount; ++i) {
      integrators[i]->MultAdd(localX, localY);
    }

    otestFespace->LocalToGlobal(localY, y);
  }

  // Matrix transpose vector multiplication.
  void OccaBilinearForm::MultTranspose(const OccaVector &x, OccaVector &y) const {
    otestFespace->GlobalToLocal(x, localX);
    localY = 0;

    const int integratorCount = (int) integrators.size();
    for (int i = 0; i < integratorCount; ++i) {
      integrators[i]->MultTransposeAdd(localX, localY);
    }

    otrialFespace->LocalToGlobal(localY, y);
  }


  void OccaBilinearForm::RecoverFEMSolution(const OccaVector &X,
                                            const OccaVector &b,
                                            OccaVector &x) {
    TRecoverFEMSolution<OccaVector>(X, b, x);
  }

  // Frees memory bilinear form.
  OccaBilinearForm::~OccaBilinearForm() {
    // Make sure all integrators free their data
    IntegratorVector::iterator it = integrators.begin();
    while (it != integrators.end()) {
      delete *it;
      ++it;
    }
  }
  //====================================

  //---[ Constrained Operator ]---------
  occa::kernelBuilder OccaConstrainedOperator::mapDofBuilder =
                  makeCustomBuilder("vector_map_dofs",
                                    "const int idx = v2[i];"
                                    "v0[idx] = v1[idx];",
                                    "defines: { VTYPE2: 'int' }");

  occa::kernelBuilder OccaConstrainedOperator::clearDofBuilder =
                  makeCustomBuilder("vector_clear_dofs",
                                    "v0[v1[i]] = 0.0;",
                                    "defines: { VTYPE1: 'int' }");

  OccaConstrainedOperator::OccaConstrainedOperator(Operator *A_,
                                                   const Array<int> &constraintList_,
                                                   bool own_A_) :
    Operator(A_->Height(), A_->Width()) {
    Setup(occa::getDevice(), A_, constraintList_, own_A_);
  }

  OccaConstrainedOperator::OccaConstrainedOperator(occa::device device_,
                                                   Operator *A_,
                                                   const Array<int> &constraintList_,
                                                   bool own_A_) :
    Operator(A_->Height(), A_->Width()) {
    Setup(device_, A_, constraintList_, own_A_);
  }

  void OccaConstrainedOperator::Setup(occa::device device_,
                                      Operator *A_,
                                      const Array<int> &constraintList_,
                                      bool own_A_) {
    device = device_;

    A = A_;
    own_A = own_A_;

    constraintIndices = constraintList_.Size();
    if (constraintIndices) {
      constraintList.allocate(device,
                              constraintIndices,
                              constraintList_.GetData());
      constraintList.keepInDevice();
    }

    z.SetSize(device, height);
    w.SetSize(device, height);
  }

  void OccaConstrainedOperator::EliminateRHS(const OccaVector &x, OccaVector &b) const {
    occa::kernel mapDofs = mapDofBuilder.build(device);

    w = 0.0;

    if (constraintIndices) {
      mapDofs(constraintIndices, w, x, constraintList);
    }

    A->Mult(w, z);

    b -= z;

    if (constraintIndices) {
      mapDofs(constraintIndices, b, x, constraintList);
    }
  }

  void OccaConstrainedOperator::Mult(const OccaVector &x, OccaVector &y) const {
    if (constraintIndices == 0) {
      A->Mult(x, y);
      return;
    }

    occa::kernel mapDofs   = mapDofBuilder.build(device);
    occa::kernel clearDofs = clearDofBuilder.build(device);

    z = x;

    clearDofs(constraintIndices, z, constraintList);

    A->Mult(z, y);

    mapDofs(constraintIndices, y, x, constraintList);
  }

  OccaConstrainedOperator::~OccaConstrainedOperator() {
    if (own_A) {
      delete A;
    }
  }
  //====================================
}

#endif
