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

AcroDiffusionIntegrator::AcroDiffusionIntegrator(Coefficient &q) :
  Q(q) {}

AcroDiffusionIntegrator::~AcroDiffusionIntegrator() {
  *B, *G, *WC, *D;
  Array<acrobatic::Tensor*> Btil;
  if (B) {delete B;}
  if (G) {delete G;}
  if (D) {delete D;}
  for (int i = 0; i < Btil.Size(); ++i) {
    if (Btil[i]) {delete Btil[i];}
  }

}

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
  numDim    = fe.GetDim();
  numElems  = fespace->GetNE();
  numDofs   = fespace->GetNDofs();
  numQuad   = ir.GetNPoints();
  numDofs1D = fe.GetOrder() + 1;
  numQuad1D = ir1D.GetNPoints();
  const H1_TensorBasisElement *el = dynamic_cast<const H1_TensorBasisElement*>(&fe);
  haveTensorBasis = (el != NULL);
  if (haveTensorBasis) {
    maps = OccaDofQuadMaps::GetTensorMaps(device, *el, ir);
    B = new acrobatic::Tensor(numQuad1D, numDof1D, nullptr, maps.quadToDof.getMemoryHandle(), true);
    G = new acrobatic::Tensor(numQuad1D, numDof1D, nullptr, maps.quadToDofD.getMemoryHandle(), true);
  } else {
    maps = OccaDofQuadMaps::GetSimplexMaps(device, fe, ir);
    B = new acrobatic::Tensor(numQuad, numDof, nullptr, maps.quadToDof.getMemoryHandle(), true);
    G = new acrobatic::Tensor(numQuad, numDof, numDim, nullptr, maps.quadToDofD.getMemoryHandle(), true);  
  }

  if (haveTensorBasis) {
    computeBtilde();
  }

  occa::array<double> jac, jacinv, jacdet;
  getJacobianData(device, fespace, ir, jac, jacinv, jacdet); 
  computeD(jac, jacinv, jacdet);

  //assembledOperator.allocate(symmDims, numQuad, numElems);
}


void AcroDiffusionIntegrator::ComputeBtilde() {
  Btil.SetSize(numDim);
  for (int d = 0; d < numDim; ++d) {
    Btil[d] = new acrobatic::Tensor(numDim, numDim, numQuad1D, numDof1D, numDof1D);
    Btil[d]->SwitchToGPU();
    acrobatic::Tensor *Bsub(numQuad1D, numDof1D, numDof1D, nullptr, Btil[d]->GetDeviceData(), true);
    for (mi = 0; mi < numDim; ++mi) {
      for (ni = 0; ni < numDim; ++ni) {
        int offset = (numDim*mi + ni) * numQuad1D*numDof1D*numDof1D;
        Bsub->Retarget(nullptr, Btil[d]->GetDeviceData() + offset);
        acrobatic::Tensor *BGM = (mi == d) ? G : B;
        acrobatic::Tensor *BGN = (ni == d) ? G : B;
        TE["Bsub_k1_i1_j1 = M_k1_i1 N_k1_j1"](*Bsub, *BGM, *BGN);
      }
    }
  }
}


void AcroDiffusionIntegrator::ComputeD(occa::array<double> &jac, 
                                       occa::array<double> &jacinv, 
                                       occa::array<double> &jacdet) {
  //Compute the coefficient * integrations weights
  std::vector<int> wdims(numDim, numQuad);
  acrobatic::Tensor *W = new acrobatic::Tensor(maps.quadWeights.size(), nullptr, maps.quadWeights.getMemoryHandle(), true);
  const ConstantCoefficient* const_coeff = dynamic_cast<const ConstantCoefficient*>(Q);
  if (!const_coeff) {
    mfem_error("AcroDiffusionIntegrator can only handle ConstantCoefficients");
  }
  acrobatic::Tensor *WC = new acrobatic::Tensor(maps.quadWeights.size());
  WC->SwitchToGPU();
  TE["WC_i=W_i"](WC, W);
  WC->Mult(const_coeff->const);
  WC->Reshape(wdims);

  //Get the jacobians and compute D with them
  acrobatic::Tensor *J, *Jinv, *Jdet;
  if (haveTensorBasis) {
    if (Dim == 1) {
      D = new acrobatic::Tensor(numElems, Dim, Dim, numQuad1D);
      J = new acrobatic::Tensor(numElems, numQuad1D, Dim, Dim, nullptr, jac.getMemoryHandle(), true);
      Jinv = new acrobatic::Tensor(numElems, numQuad1D, Dim, Dim, nullptr, jacinv.getMemoryHandle(), true);
      Jdet = new acrobatic::Tensor(numElems, numQuad1D, nullptr, jacinv.getMemoryHandle(), true);
      TE["D_e_m_n_k = WC_k Jdet_e_k_m_n Jinv_e_k_m_n Jinv_e_k_n_m"](D, WC, Jdet, Jinv, Jinv);
    } else if (Dim == 2) {
      D = new acrobatic::Tensor(numElems, Dim, Dim, numQuad1D, numQuad1D);
      J = new acrobatic::Tensor(numElems, numQuad1D, numQuad1D, Dim, Dim, nullptr, jac.getMemoryHandle(), true);
      Jinv = new acrobatic::Tensor(numElems, numQuad1D, numQuad1D, Dim, Dim, nullptr, jacinv.getMemoryHandle(), true);
      Jdet = new acrobatic::Tensor(numElems, numQuad1D, numQuad1D, nullptr, jacinv.getMemoryHandle(), true);
      TE["D_e_m_n_k1_k2 = WC_k1_k2 Jdet_e_k1_k2_m_n Jinv_e_k1_k2_m_n Jinv_e_k1_k2_n_m"](D, WC, Jdet, Jinv, Jinv);
    } else if (Dim == 3){
      D = new acrobatic::Tensor(numElems, Dim, Dim, numQuad1D, numQuad1D, numQuad1D);
      J = new acrobatic::Tensor(numElems, numQuad1D, numQuad1D, numQuad1D, Dim, Dim, nullptr, jac.getMemoryHandle(), true);
      Jinv = new acrobatic::Tensor(numElems, numQuad1D, numQuad1D, numQuad1D, Dim, Dim, nullptr, jacinv.getMemoryHandle(), true);
      Jdet = new acrobatic::Tensor(numElems, numQuad1D, numQuad1D, numQuad1D, nullptr, jacinv.getMemoryHandle(), true);
      TE["D_e_m_n_k1_k2_k3 = WC_k1_k2_k3 Jdet_e_k1_k2_k3_m_n Jinv_e_k1_k2_k3_m_n Jinv_e_k1_k2_k3_n_m"](D, WC, Jdet, Jinv, Jinv);
    } else {
      mfem_error("AcroDiffusionIntegrator tensor computations don't support dim > 3.");
    }
  } else {
    D = new acrobatic::Tensor(numElems, Dim, Dim, numQuad);
    J = new acrobatic::Tensor(numElems, numQuad, Dim, Dim, nullptr, jac.getMemoryHandle(), true);
    Jinv = new acrobatic::Tensor(numElems, numQuad, Dim, Dim, nullptr, jacinv.getMemoryHandle(), true);
    Jdet = new acrobatic::Tensor(numElems, numQuad, nullptr, jacinv.getMemoryHandle(), true);
    TE["D_e_m_n_k = WC_k Jdet_e_k_m_n Jinv_e_k_m_n Jinv_e_k_n_m"](D, WC, Jdet, Jinv, Jinv);
  }

  delete W;
  delete WC;
  delete J;
  delete Jinv;
  delete Jdet;
}  

void AcroDiffusionIntegrator::Assemble() {
  if (haveTensorBasis) {
    if (numDim == 1) {
      TE["S_e_i1_j1 = Btil_m_n_k1_i1_j1 D_e_m_n_k1"]
        (S, Btil[0], D);
    } else if (numDim == 2) {
      TE["S_e_i1_i2_j1_j2 = Btil1_m_n_k1_i1_j1 Btil2_m_n_k2_i2_j2 D_e_m_n_k1_k2"]
        (S, Btil[0], Btil[1], D);
    } else if (numDim == 3) {
      TE["S_e_i1_i2_i3_j1_j2_j3 = Btil1_m_n_k1_i1_j1 Btil2_m_n_k2_i2_j2 Btil3_m_n_k3_i3_j3 D_e_m_n_k1_k2"]
        (S, Btil[0], Btil[1], Btil[2], D);
    }
  } else {
    TE["S_e_i_j = G_k_i_m G_k_i_n D_e_m_n_k"]
        (S, G, G, D);
  }
}

void AcroDiffusionIntegrator::Mult(OccaVector &v) {
  if (haveTensorBasis) {
    if (numDim == 1) {
      acrobatic::Tensor V(numElems, numDofs1D, nullptr, x.getMemoryHandle(), true);
      acrobatic::Tensor U(numDim, numElems, numQuad1D);
      acrobatic::Tensor W(numDim, numElems, numQuad1D);
      acrobatic::Tensor X(numElems, numDofs1D);
      U.SwitchToGPU();
      W.SwitchToGPU()
      X.SwitchToGPU();
      TE["U_e_n_k1 = G_k1_i1 V_e_i1"](U, *G, V);
      TE["W_e_m_k1 = D_m_n_k1 U_e_n_k1"](W, *D, U);
      TE["X_e_i1 = G_k1_i1 W_e_m_k1"](X, *G, W);

    } else if (numDim == 2) {
      X = new acrobatic::Tensor(numElems, numDofs1D, numDofs1D, nullptr, x.getMemoryHandle(), true);
      TE["S_e_i1_i2_j1_j2 = Btil1_m_n_k1_i1_j1 Btil2_m_n_k2_i2_j2 D_e_m_n_k1_k2"]
        (S, Btil[0], Btil[1], D);
    } else if (numDim == 3) {
      X = new acrobatic::Tensor(numElems, numDofs1D, numDofs1D, numDofs1D, nullptr, x.getMemoryHandle(), true);
      TE["S_e_i1_i2_i3_j1_j2_j3 = Btil1_m_n_k1_i1_j1 Btil2_m_n_k2_i2_j2 Btil3_m_n_k3_i3_j3 D_e_m_n_k1_k2"]
        (S, Btil[0], Btil[1], Btil[2], D);
    }
  } else {
    X = new acrobatic::Tensor(numElems, numDofs, nullptr, x.getMemoryHandle(), true);
    TE["S_e_i_j = G_k_i_m G_k_i_n D_e_m_n_k"]
        (S, G, G, D);
  }

}

}

#endif