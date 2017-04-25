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
      TE.SetExecutorType("OneOutPerThread");
    } else {
      mfem_error("AcroDiffusionIntegrator non-CUDA mode unsupported.");
      TE.SetExecutorType("IndexCached");
    }

    DiffusionIntegrator integ;
    const FiniteElement &fe   = *(fespace->GetFE(0));
    const IntegrationRule &ir = integ.GetIntegrationRule(fe, fe);
    const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder());
    numDim = fe.GetDim();
    numElems = fespace->GetNE();

    const H1_TensorBasisElement *el = dynamic_cast<const H1_TensorBasisElement*>(&fe);
    if (el) {
      maps = OccaDofQuadMaps::GetTensorMaps(device, *el, ir);
      numDofs  = fe.GetOrder() + 1;
      numQuad  = ir1D.GetNPoints();

      //Compute the Btildes
      B = new acrobatic::Tensor(numQuad, numDof, nullptr, maps.dofToQuad.getMemoryHandle(), true);
      G = new acrobatic::Tensor(numQuad, numDof, nullptr, maps.dofToQuadD.getMemoryHandle(), true);
      Btil.SetSize(numDim);
      for (int d = 0; d < numDim; ++d) {
        Btil[d] = new acrobatic::Tensor(numDim, numDim, numQuad, numDof, numDof);
        Btil[d]->SwitchToGPU();
        acrobatic::Tensor *Bsub(numQuad, numDof, numDof, nullptr, Btil[d]->GetDeviceData(), true);
        for (mi = 0; mi < 2; ++mi) {
          for (ni = 0; ni < 2; ++ni) {
            int offset = (2*mi + ni) * numQuad*numDof*numDof;
            Bsub->Retarget(nullptr, Btil[d]->GetDeviceData() + offset);
            acrobatic::Tensor *BGM = (mi == d) ? G : B;
            acrobatic::Tensor *BGN = (ni == d) ? G : B;
            TE["Bsub_k1_i1_j1 = M_k1_i1 N_k1_j1"](Bsub, BGM, BGN);
          }
        }
      }

      //Compute the coefficient * integrations weights
      std::vector<int> wdims(numDim, numQuad);
      acrobatic::Tensor *W = new acrobatic::Tensor(maps.quadWeights.size(), nullptr, maps.quadWeights.getMemoryHandle(), true);
      const ConstantCoefficient* coeff =
        dynamic_cast<const ConstantCoefficient*>(integ.GetCoefficient());
      if (!coeff) {
        mfem_error("AcroDiffusionIntegrator can only handle ConstantCoefficients");
      }
      WC = new acrobatic::Tensor(maps.quadWeights.size());
      WC->SwitchToGPU();
      TE["WC_i=W_i"](WC, W);
      WC->Mult(coeff->const);
      WC->Reshape(wdims);

      //Compute the jacobians and D
      occa::array<double> jac, jacinv, jacdet;
      getJacobianData(device, fespace, ir, jac, jacinv, jacdet); 
      if (Dim == 1) {
        D = new acrobatic::Tensor(numElems, 1, 1, numQuad);
        J = new acrobatic::Tensor(numElems, numQuad, 1, 1, nullptr, jac.getMemoryHandle(), true);
        Jinv = new acrobatic::Tensor(numElems, numQuad, 1, 1, nullptr, jacinv.getMemoryHandle(), true);
        Jdet = new acrobatic::Tensor(numElems, numQuad, nullptr, jacinv.getMemoryHandle(), true);
        TE["D_e_m_n_k = WC_k Jdet_e_k_m_n Jinv_e_k_m_n Jinv_e_k_n_m"](D, WC, Jdet, Jinv, Jinv);
      } else if (Dim == 2) {
        D = new acrobatic::Tensor(numElems, 2, 2, numQuad, numQuad);
        J = new acrobatic::Tensor(numElems, numQuad, numQuad, 2, 2, nullptr, jac.getMemoryHandle(), true);
        Jinv = new acrobatic::Tensor(numElems, numQuad, numQuad, 2, 2, nullptr, jacinv.getMemoryHandle(), true);
        Jdet = new acrobatic::Tensor(numElems, numQuad, numQuad, nullptr, jacinv.getMemoryHandle(), true);
        TE["D_e_m_n_k1_k2 = WC_k1_k2 Jdet_e_k1_k2_m_n Jinv_e_k1_k2_m_n Jinv_e_k1_k2_n_m"](D, WC, Jdet, Jinv, Jinv);
      } else {
        D = new acrobatic::Tensor(numElems, 3, 3, numQuad, numQuad, numQuad);
        J = new acrobatic::Tensor(numElems, numQuad, numQuad, numQuad, 3, 3, nullptr, jac.getMemoryHandle(), true);
        Jinv = new acrobatic::Tensor(numElems, numQuad, numQuad, numQuad, 3, 3, nullptr, jacinv.getMemoryHandle(), true);
        Jdet = new acrobatic::Tensor(numElems, numQuad, numQuad, numQuad, nullptr, jacinv.getMemoryHandle(), true);
        TE["D_e_m_n_k1_k2_k3 = WC_k Jdet_e_k1_k2_k3_m_n Jinv_e_k1_k2_k3_m_n Jinv_e_k1_k2_k3_n_m"](D, WC, Jdet, Jinv, Jinv);
      }

    } else {
      mfem_error("AcroDiffusionIntegrator simplices currently unsupported.");
      numDofs  = fespace->GetNDofs();
      numQuad  = ir.GetNPoints();
      maps = OccaDofQuadMaps::GetSimplexMaps(device, fe, ir);
    }



    // Get coefficient from integrator
    // [MISSING] Hard-coded to ConstantCoefficient for now


    if (coeff) {
      hasConstantCoefficient = true;
      kernelProps["defines/CONST_COEFF"] = coeff->constant;
    } else {
      
    }

    assembledOperator.allocate(symmDims, numQuad, numElems);

    jacobian = getJacobian(device, fespace, ir);
  }

  void AcroDiffusionIntegrator::Assemble() {

  }

  void AcroDiffusionIntegrator::Mult(OccaVector &x) {

  }
  //====================================
}

#endif