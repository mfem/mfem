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

#if defined(MFEM_USE_OCCA) && defined(MFEM_USE_ACROTENSOR)

#include "abilininteg.hpp"

namespace mfem {
  //---[ Diffusion Integrator ]---------
  AcroDiffusionIntegrator::AcroDiffusionIntegrator(Coefficient &q) :
    Q(q) {}

  AcroDiffusionIntegrator::~AcroDiffusionIntegrator() {}

  OccaIntegrator* AcroDiffusionIntegrator::CreateInstance() {
    return new AcroDiffusionIntegrator(Q);
  }

  std::string AcroDiffusionIntegrator::GetName() {
    return "AcroDiffusionIntegrator";
  }

  void AcroDiffusionIntegrator::Setup() {
    if (device.mode() != "CUDA") {
      mfem_error("AcroDiffusionIntegrator can only run in CUDA mode");
    }

    occa::properties kernelProps = props;
    kernelProps["OKL"] = false;

    DiffusionIntegrator integ;

    const FiniteElement &fe   = *(fespace->GetFE(0));
    const IntegrationRule &ir = integ.GetIntegrationRule(fe, fe);

    const H1_TensorBasisElement *el = dynamic_cast<const H1_TensorBasisElement*>(&fe);
    if (el) {
      maps = OccaDofQuadMaps::GetTensorMaps(device, *el, ir);
      setTensorProperties(fe, ir, kernelProps);
    } else {
      maps = OccaDofQuadMaps::GetSimplexMaps(device, fe, ir);
      setSimplexProperties(fe, ir, kernelProps);
    }

    const int dims = fe.GetDim();
    const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6

    elements = fespace->GetNE();
    numDofs  = fespace->GetNDofs();
    numQuad  = ir.GetNPoints();

    // Get coefficient from integrator
    // [MISSING] Hard-coded to ConstantCoefficient for now
    const ConstantCoefficient* coeff =
      dynamic_cast<const ConstantCoefficient*>(integ.GetCoefficient());

    if (coeff) {
      hasConstantCoefficient = true;
      kernelProps["defines/CONST_COEFF"] = coeff->constant;
    } else {
      mfem_error("AcroDiffusionIntegrator can only handle ConstantCoefficients");
    }

    assembledOperator.allocate(symmDims, numQuad, elements);

    jacobian = getJacobian(device, fespace, ir);

    // Setup assemble and mult kernels
    assembleKernel = GetKernel("Assemble", kernelProps);
    multKernel     = GetKernel("Mult"    , kernelProps);
  }

  void AcroDiffusionIntegrator::Assemble() {
    assembleKernel(elements, numQuad, numDofs,
                   NULL,
                   NULL,
                   maps.quadWeights.memory().getHandle(),
                   NULL,
                   NULL,
                   &TE);
  }

  void AcroDiffusionIntegrator::Mult(OccaVector &x) {
    assembleKernel(elements, numQuad, numDofs,
                   &TE);
  }
  //====================================
}

#endif