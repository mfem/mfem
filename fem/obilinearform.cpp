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

namespace mfem {
  OccaBilinearForm::KernelBuilderCollection OccaBilinearForm::kernelBuilderCollection;

  OccaBilinearForm::OccaBilinearForm(FiniteElementSpace *f) :
    fes(f),
    mesh(fes->GetMesh()),
    device(occa::currentDevice()) {

    SetupKernelBuilderCollection();
    SetupBaseKernelProps();
  }

  OccaBilinearForm::OccaBilinearForm(occa::device dev, FiniteElementSpace *f) :
    fes(f),
    mesh(fes->GetMesh()),
    device(dev) {

    SetupKernelBuilderCollection();
    SetupBaseKernelProps();
  }

  void OccaBilinearForm::SetupKernelBuilderCollection() {
    if (0 < kernelBuilderCollection.size()) {
      return;
    }
    kernelBuilderCollection[DiffusionIntegrator::StaticName()] =
      &OccaBilinearForm::BuildDiffusionIntegrator;
  }

  void OccaBilinearForm::SetupBaseKernelProps() {
    const std::string &mode = device.properties()["mode"];
    const bool inGPU = (mode == "CUDA") || (mode == "OpenCL");

    baseKernelProps["kernel/defines/NUM_ELEMENTS"]  = GetNE();
    baseKernelProps["kernel/defines/ELEMENT_BATCH"] = 1;
    baseKernelProps["kernel/defines/NUM_DOFS"]      = GetNDofs();
    baseKernelProps["kernel/defines/NUM_VDIM"]      = GetVSize();
    baseKernelProps["kernel/defines/IN_GPU"]        = inGPU ? 1 : 0;
  }

  int OccaBilinearForm::GetDim()
  { return mesh->Dimension(); }

  int64_t OccaBilinearForm::GetNE()
  { return mesh->GetNE(); }

  int64_t OccaBilinearForm::GetNDofs()
  { return fes->GetNDofs(); }

  int64_t OccaBilinearForm::GetVSize()
  { return fes->GetVDim(); }

  /// Adds new Domain Integrator.
  void OccaBilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi,
                                             const occa::properties &props) {
    AddIntegrator(*bfi, props, DomainIntegrator);
  }

  /// Adds new Boundary Integrator.
  void OccaBilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi,
                                               const occa::properties &props) {
    AddIntegrator(*bfi, props, BoundaryIntegrator);
  }

  /// Adds new interior Face Integrator.
  void OccaBilinearForm::AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi,
                                                   const occa::properties &props) {
    AddIntegrator(*bfi, props, InteriorFaceIntegrator);
  }

  /// Adds new boundary Face Integrator.
  void OccaBilinearForm::AddBoundaryFaceIntegrator(BilinearFormIntegrator *bfi,
                                                   const occa::properties &props) {
    AddIntegrator(*bfi, props, BoundaryFaceIntegrator);
  }

  /// Adds Integrator based on IntegratorType
  void OccaBilinearForm::AddIntegrator(BilinearFormIntegrator &bfi,
                                       const occa::properties &props,
                                       const IntegratorType itype) {
    KernelBuilder kernelBuilder = kernelBuilderCollection[bfi.Name()];
    if (kernelBuilder == NULL) {
      std::stringstream error_ss;
      error_ss << "OccaBilinearForm::";
      switch (itype) {
      case DomainIntegrator      : error_ss << "AddDomainIntegrator";       break;
      case BoundaryIntegrator    : error_ss << "AddBoundaryIntegrator";     break;
      case InteriorFaceIntegrator: error_ss << "AddInteriorFaceIntegrator"; break;
      case BoundaryFaceIntegrator: error_ss << "AddBoundaryFaceIntegrator"; break;
      }
      error_ss << " (...):\n"
               << "  No kernel builder for BilinearFormIntegrator '" << bfi.Name() << "'";
      const std::string error = error_ss.str();
      mfem_error(error.c_str());
    }
    integrators.push_back((this->*kernelBuilder)(bfi,
                                                 baseKernelProps + props,
                                                 itype));
  }

  /// Builds the DiffusionIntegrator
  occa::kernel OccaBilinearForm::BuildDiffusionIntegrator(BilinearFormIntegrator &bfi,
                                                          const occa::properties &props,
                                                          const IntegratorType itype) {
    DiffusionIntegrator &integ = (DiffusionIntegrator&) bfi;

    // Get kernel name
    std::string kernelName = integ.Name();
    // Append 1D, 2D, or 3D
    kernelName += '0' + (char) GetDim();
    kernelName += 'D';

    occa::properties kernelProps = props;
    // Add quadrature points
    const FiniteElement &fe = *fes->GetFE(0);
    const IntegrationRule &ir = integ.GetIntegrationRule(fe, fe);
    kernelProps["kernel/defines/NUM_QPTS"] = ir.GetNPoints();

    // Hard-coded to ConstantCoefficient for now
    const ConstantCoefficient* coeff =
      (const ConstantCoefficient*) integ.GetCoefficient();
    kernelProps["kernel/defines/COEFF_EVAL(el,q)"] = coeff->constant;

    return device.buildKernel("occaBilinearIntegrators.okl",
                              kernelName,
                              kernelProps);
  }

  /// Assembles the Jacobian information
  void OccaBilinearForm::Assemble() {
    // Allocate memory in device if needed
    const int64_t entries  = GetNE() * GetNDofs() * GetVSize();
    const int64_t gf_bytes = entries * sizeof(double);

    if ((0 < geometricFactors.size()) &&
        (geometricFactors.size() < gf_bytes)) {
      geometricFactors.free();
    }
    if (geometricFactors.size() == 0) {
      geometricFactors = device.malloc(gf_bytes);
    }

    // Global -> Local kernel
    // Look at tfespace.hpp:53
    // Assumptions:
    //  - Element types are fixed for mesh
    //  - Same FE space throughout the mesh (same dofs per element)
  }

  /// Matrix vector multiplication.
  void OccaBilinearForm::Mult(const OccaVector &x, OccaVector &y) const {
  }

  /// Matrix transpose vector multiplication.
  void OccaBilinearForm::MultTranspose(const OccaVector &x, OccaVector &y) const {
  }

  /** @brief Eliminate "essential boundary condition" values specified in @a x
      from the given right-hand side @a b.

      Performs the following steps:

      z = A((0,x_b));  b_i -= z_i;  b_b = x_b;

      where the "_b" subscripts denote the essential (boundary) indices/dofs of
      the vectors, and "_i" -- the rest of the entries. */
  void OccaBilinearForm::EliminateRHS(const OccaVector &x, OccaVector &b) const {
  }

  /// Destroys bilinear form.
  OccaBilinearForm::~OccaBilinearForm() {
    // Free memory
    geometricFactors.free();

    // Free all integrator kernels
    IntegratorVector::iterator it = integrators.begin();
    while (it != integrators.end()) {
      it->free();
      ++it;
    }

    // Free device
    device.free();
  }
}

#endif