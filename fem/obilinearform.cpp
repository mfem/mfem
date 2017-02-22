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

namespace mfem {
  //---[ Bilinear Form ]----------------
  OccaBilinearForm::IntegratorBuilderMap OccaBilinearForm::integratorBuilders;

  OccaBilinearForm::OccaBilinearForm(FiniteElementSpace *f) :
    Operator(f->GetVSize()),
    fes(f),
    mesh(fes->GetMesh()),
    device(occa::currentDevice()) {

    SetupIntegratorBuilderMap();
    SetupBaseKernelProps();
  }

  OccaBilinearForm::OccaBilinearForm(occa::device dev, FiniteElementSpace *f) :
    fes(f),
    mesh(fes->GetMesh()),
    device(dev) {

    SetupIntegratorBuilderMap();
    SetupBaseKernelProps();
  }

  void OccaBilinearForm::SetupIntegratorBuilderMap() {
    if (0 < integratorBuilders.size()) {
      return;
    }
    integratorBuilders[DiffusionIntegrator::StaticName()] =
      &DiffusionIntegratorBuilder;
  }

  void OccaBilinearForm::SetupBaseKernelProps() {
    const std::string &mode = device.properties()["mode"];

    baseKernelProps["defines/ELEMENT_BATCH"] = 1;
    baseKernelProps["defines/NUM_DOFS"]      = GetNDofs();
    baseKernelProps["defines/NUM_VDIM"]      = GetVSize();
  }

  occa::device OccaBilinearForm::getDevice() {
    return device;
  }

  int OccaBilinearForm::GetDim()
  { return mesh->Dimension(); }

  int64_t OccaBilinearForm::GetNE()
  { return mesh->GetNE(); }

  int64_t OccaBilinearForm::GetNDofs()
  { return fes->GetNDofs(); }

  int64_t OccaBilinearForm::GetVSize()
  { return fes->GetVDim(); }

  const FiniteElement& OccaBilinearForm::GetFE(const int i) {
    return *(fes->GetFE(i));
  }

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
    IntegratorBuilder builder = integratorBuilders[bfi.Name()];
    if (builder == NULL) {
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
    integrators.push_back(builder(*this,
                                  bfi,
                                  baseKernelProps + props,
                                  itype));
  }

  /// Get the finite element space prolongation matrix
  const Operator* OccaBilinearForm::GetProlongation() const {
    if (fes->GetConformingProlongation() != NULL) {
      mfem_error("OccaBilinearForm::GetProlongation() is not overloaded!");
    }
    return NULL;
  }

  /// Get the finite element space restriction matrix
  const Operator* OccaBilinearForm::GetRestriction() const {
    if (fes->GetConformingRestriction() != NULL) {
      mfem_error("OccaBilinearForm::GetRestriction() is not overloaded!");
    }
    return NULL;
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
    y = 0.0;

    // tevaluator.hpp:982
    // typedef FieldEvaluator<solFESpace,
    //                        solVecLayout_t,
    //                        IR,
    //                        complex_t,
    //                        real_t> solFieldEval;

    // solFieldEval solFEval(solFES, solEval, solVecLayout,
    //                       x.GetData(), y.GetData());

    // for (int el = 0; el < NE; el++) {
    //   typename S_spec<BE = 1>::DataType R; <-- InData/OutData
    //   -> Spec = solFieldEval::Spec<diffusion integrator kernel, BE>
    //   -> DataType = Spec::DataType
    //     tevaluator.hpp:1408
    //       tevaluator.hpp:1132
    //       -> Values    = 1
    //       -> Gradients = 2
    //     -> InData  = Values*kernel_t::in_values  + Gradients*kernel_t::in_gradients;
    //                = (1 * false) + (2 * true)
    //                = 2
    //     -> OutData = Values*kernel_t::out_values + Gradients*kernel_t::out_gradients;
    //                = (1 * false) + (2 * true)
    //                = 2
    //     -> Spec::Datatype = BData<InData,OutData,NE>
    //                       = BData<2,2,NE>
    //
    //   tevaluator.hpp:1089
    //   solFEval.Eval(el, R); {
    //     SetElement(el);
    //     Action<DataType::InData,true>::Eval(vec_layout, *this, R);
    //       tevaluator.hpp:654
    //       -> T.shapeEval.Calc<Add = false>
    //       tevaluator.hpp:702
    //       -> T.shapeEval.CalcGrad<Add = true>
    //   }
    //
    //   for (int i = 0; i < NUM_QPTS; i++) {
    //     for (int j = 0; j < NUM_VDIM; j++) {
    //       R.grad_qpts(i,0,j,0) *= assembled_data[el](i,0);
    //     }
    //   }
    //
    //   tevaluator.hpp:1097
    //   solFEval.Assemble<true>(R); {
    //     tevaluator.hpp:1331
    //     Action<DataType::OutData,true>::Assemble<Add>(vec_layout, *this, R);
    //       tevaluator.hpp:691
    //       -> T.shapeEval.CalcT<Add = false>
    //       tevaluator.hpp:724
    //       -> T.shapeEval.CalcGradT<Add = true>
    //       tfespace.hpp:239
    //       -> T.fespace.VectorAssemble<Op = Add>
    //   }
    // }
  }

  /// Matrix transpose vector multiplication.
  void OccaBilinearForm::MultTranspose(const OccaVector &x, OccaVector &y) const {
    mfem_error("OccaBilinearForm::MultTranspose() is not overloaded!");
  }


  void OccaBilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list,
                                          OccaVector &x, OccaVector &b,
                                          Operator* &Aout, OccaVector &X, OccaVector &B,
                                          int copy_interior) {

    TFormLinearSystem<OccaVector>(ess_tdof_list,
                                  x, b, Aout, X, B,
                                  copy_interior);
  }


  void OccaBilinearForm::RecoverFEMSolution(const OccaVector &X,
                                            const OccaVector &b,
                                            OccaVector &x) {
    TRecoverFEMSolution<OccaVector>(X, b, x);
  }

  void OccaBilinearForm::ImposeBoundaryConditions(const Array<int> &ess_tdof_list,
                                                  Operator *rap,
                                                  Operator* &Aout,
                                                  OccaVector &X, OccaVector &B) {

    OccaConstrainedOperator *A = new OccaConstrainedOperator(device,
                                                             rap, ess_tdof_list,
                                                             rap != this);
    A->EliminateRHS(X, B);
    Aout = A;
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
  }
  //====================================

  //---[ Constrained Operator ]---------
  occa::kernelBuilder OccaConstrainedOperator::map_dof_builder =
    makeCustomBuilder("vector_map_dofs",
                      "const int idx = v2[i];"
                      "v0[idx] = v1[idx];",
                      "defines: { VTYPE2: 'int' }");

  occa::kernelBuilder OccaConstrainedOperator::clear_dof_builder =
    makeCustomBuilder("vector_clear_dofs",
                      "v0[v1[i]] = 0.0;",
                      "defines: { VTYPE1: 'int' }");

  OccaConstrainedOperator::OccaConstrainedOperator(Operator *A_,
                                                   const Array<int> &constraint_list_,
                                                   bool own_A_) :
    Operator(A_->Height(), A_->Width()) {
    setup(occa::currentDevice(), A_, constraint_list_, own_A_);
  }

  OccaConstrainedOperator::OccaConstrainedOperator(occa::device device_,
                                                   Operator *A_,
                                                   const Array<int> &constraint_list_,
                                                   bool own_A_) :
    Operator(A_->Height(), A_->Width()) {
    setup(device_, A_, constraint_list_, own_A_);
  }

  void OccaConstrainedOperator::setup(occa::device device_,
                                      Operator *A_,
                                      const Array<int> &constraint_list_,
                                      bool own_A_) {
    device = device_;

    A = A_;
    own_A = own_A_;

    constraint_indices = constraint_list_.Size();
    if (constraint_indices) {
      constraint_list = device.malloc(constraint_indices * sizeof(int),
                                      constraint_list_.GetData());
    }

    z.SetSize(device, height);
    w.SetSize(device, height);
  }

  void OccaConstrainedOperator::EliminateRHS(const OccaVector &x, OccaVector &b) const {
    if (constraint_indices == 0) {
      return;
    }

    occa::kernel map_dofs = map_dof_builder.build(device);

    w = 0.0;

    map_dofs(constraint_indices, w.GetData(), x.GetData(), constraint_list);

    A->Mult(w, z);

    b -= z;

    map_dofs(constraint_indices, b.GetData(), x.GetData(), constraint_list);
  }

  void OccaConstrainedOperator::Mult(const OccaVector &x, OccaVector &y) const {
    if (constraint_indices == 0) {
      A->Mult(x, y);
      return;
    }

    occa::kernel map_dofs = map_dof_builder.build(device);
    occa::kernel clear_dofs = clear_dof_builder.build(device);

    z = x;

    clear_dofs(constraint_indices, z.GetData(), constraint_list);

    A->Mult(z, y);

    map_dofs(constraint_indices, y.GetData(), x.GetData(), constraint_list);
  }

  OccaConstrainedOperator::~OccaConstrainedOperator() {
    if (own_A) {
      delete A;
    }
    constraint_list.free();
  }
  //====================================
}

#endif