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

}

OccaIntegrator* AcroDiffusionIntegrator::CreateInstance() {
  return new AcroDiffusionIntegrator(Q);
}

std::string AcroDiffusionIntegrator::GetName() {
  return "AcroDiffusionIntegrator";
}

void AcroDiffusionIntegrator::Setup() {
  if (device.mode() == "CUDA") {
    onGPU = true;
    TE.SetExecutorType("OneOutPerThread");

    CUcontext cudaContext = (CUcontext) device.getHandle("type: 'context'");
    acro::setCudaContext(cudaContext);
  } else {
    onGPU = false;
    TE.SetExecutorType("CPUInterpreted");
  }

  DiffusionIntegrator integ;
  const FiniteElement &fe   = *(fespace->GetFE(0));
  const IntegrationRule &ir = integ.GetIntegrationRule(fe, fe);
  const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder());
  nDim    = fe.GetDim();
  nElem  = fespace->GetNE();
  nDof   = fe.GetDof();
  nQuad   = ir.GetNPoints();
  nDof1D = fe.GetOrder() + 1;
  nQuad1D = ir1D.GetNPoints();

  if (nDim > 3) {
    mfem_error("AcroDiffusionIntegrator tensor computations don't support dim > 3.");
  }

  //Get AcroTensors pointing to the B/G data
  //Note:  We are giving acrotensor the same pointer for the GPU and CPU
  //so one of them is obviously wrong.  This works as long as we don't use
  //touch the wrong one
  const H1_TensorBasisElement *el = dynamic_cast<const H1_TensorBasisElement*>(&fe);
  haveTensorBasis = (el != NULL);  
  if (haveTensorBasis) {
    maps = OccaDofQuadMaps::GetTensorMaps(device, *el, ir);
    double *b_ptr = *((double**) maps.quadToDof.memory().getHandle());
    double *g_ptr = *((double**) maps.quadToDofD.memory().getHandle());
    double *w_ptr = *((double**) maps.quadWeights.memory().getHandle());    
    B.Init(nQuad1D, nDof1D, b_ptr, b_ptr, onGPU);
    G.Init(nQuad1D, nDof1D, g_ptr, g_ptr, onGPU);
    std::vector<int> wdims(nDim, nQuad1D);
    WC.Init(wdims, w_ptr, w_ptr, onGPU);
  } else {
    maps = OccaDofQuadMaps::GetSimplexMaps(device, fe, ir);
    double *b_ptr = *((double**) maps.quadToDof.memory().getHandle());
    double *g_ptr = *((double**) maps.quadToDofD.memory().getHandle());
    double *w_ptr = *((double**) maps.quadWeights.memory().getHandle());    
    B.Init(nQuad, nDof, b_ptr, b_ptr, onGPU);
    G.Init(nQuad, nDof, nDim, g_ptr, g_ptr, onGPU);
    WC.Init(nQuad, w_ptr, w_ptr, onGPU);
  }
}


void AcroDiffusionIntegrator::ComputeBTilde() {
  Btil.SetSize(nDim);
  for (int d = 0; d < nDim; ++d) {
    Btil[d] = new acro::Tensor(nDim, nDim, nQuad1D, nDof1D, nDof1D);
    if (onGPU) {
      Btil[d]->SwitchToGPU();
    }
    acro::Tensor Bsub(nQuad1D, nDof1D, nDof1D, Btil[d]->GetCurrentData(), Btil[d]->GetCurrentData(), onGPU);
    for (int mi = 0; mi < nDim; ++mi) {
      for (int ni = 0; ni < nDim; ++ni) {
        int offset = (nDim*mi + ni) * nQuad1D*nDof1D*nDof1D;
        Bsub.Retarget(Btil[d]->GetCurrentData() + offset, Btil[d]->GetCurrentData() + offset);
        acro::Tensor &BGM = (mi == d) ? G : B;
        acro::Tensor &BGN = (ni == d) ? G : B;
        TE["Bsub_k1_i1_j1 = M_k1_i1 N_k1_j1"](Bsub, BGM, BGN);
      }
    }
  }
}


void AcroDiffusionIntegrator::ComputeD(OccaGeometry &geom) {
  //Get the jacobians and compute D with them
  double *jac_ptr = *((double**) geom.J.memory().getHandle());
  double *jacinv_ptr = *((double**) geom.invJ.memory().getHandle());
  double *jacdet_ptr = *((double**) geom.detJ.memory().getHandle());
  if (haveTensorBasis) {
    if (nDim == 1) {
      D.Init(nElem, nDim, nDim, nQuad1D);
      acro::Tensor J(nElem, nQuad1D, nDim, nDim, 
                     jac_ptr, jac_ptr, onGPU);
      acro::Tensor Jinv(nElem, nQuad1D, nDim, nDim, 
                        jacinv_ptr, jacinv_ptr, onGPU);
      acro::Tensor Jdet(nElem, nQuad1D, 
                        jacdet_ptr, jacdet_ptr, onGPU);
      TE["D_e_m_n_k = WC_k Jdet_e_k Jinv_e_k_m_n Jinv_e_k_n_m"]
        (D, WC, Jdet, Jinv, Jinv);
    } else if (nDim == 2) {
      D.Init(nElem, nDim, nDim, nQuad1D, nQuad1D);
      acro::Tensor J(nElem, nQuad1D, nQuad1D, nDim, nDim, 
                     jac_ptr, jac_ptr, onGPU);
      acro::Tensor Jinv(nElem, nQuad1D, nQuad1D, nDim, nDim, 
                        jacinv_ptr, jacinv_ptr, onGPU);
      acro::Tensor Jdet(nElem, nQuad1D, nQuad1D, 
                        jacdet_ptr, jacdet_ptr, onGPU);
      TE["D_e_m_n_k1_k2 = WC_k1_k2 Jdet_e_k1_k2 Jinv_e_k1_k2_m_n Jinv_e_k1_k2_n_m"]
        (D, WC, Jdet, Jinv, Jinv);
    } else if (nDim == 3){
      D.Init(nElem, nDim, nDim, nQuad1D, nQuad1D, nQuad1D);
      acro::Tensor J(nElem, nQuad1D, nQuad1D, nQuad1D, nDim, nDim, 
                     jac_ptr, jac_ptr, onGPU);
      acro::Tensor Jinv(nElem, nQuad1D, nQuad1D, nQuad1D, nDim, nDim, 
                        jacinv_ptr, jacinv_ptr, onGPU);
      acro::Tensor Jdet(nElem, nQuad1D, nQuad1D, nQuad1D, 
                        jacdet_ptr, jacdet_ptr, onGPU);
      TE["D_e_m_n_k1_k2_k3 = WC_k1_k2_k3 Jdet_e_k1_k2_k3 Jinv_e_k1_k2_k3_m_n Jinv_e_k1_k2_k3_n_m"]
        (D, WC, Jdet, Jinv, Jinv);
    } else {
      mfem_error("AcroDiffusionIntegrator tensor computations don't support dim > 3.");
    }
  } else {
    D.Init(nElem, nDim, nDim, nQuad);
    acro::Tensor J(nElem, nQuad, nDim, nDim, 
                   jac_ptr, jac_ptr, onGPU);
    acro::Tensor Jinv(nElem, nQuad, nDim, nDim, 
                      jacinv_ptr, jacinv_ptr, onGPU);
    acro::Tensor Jdet(nElem, nQuad, 
                      jacdet_ptr, jacdet_ptr, onGPU);
    TE["D_e_m_n_k = WC_k Jdet_e_k Jinv_e_k_m_n Jinv_e_k_n_m"]
      (D, WC, Jdet, Jinv, Jinv);
  }
}  

void AcroDiffusionIntegrator::Assemble() {
  const ConstantCoefficient* const_coeff = dynamic_cast<const ConstantCoefficient*>(&Q);
  if (!const_coeff) {
    mfem_error("AcroDiffusionIntegrator can only handle ConstantCoefficients");
  }  
  WC.Mult(const_coeff->constant);

  DiffusionIntegrator integ;
  const FiniteElement &fe   = *(fespace->GetFE(0));
  const IntegrationRule &ir = integ.GetIntegrationRule(fe, fe);
  OccaGeometry geom = OccaGeometry::Get(device, *mesh, ir);  
  ComputeD(geom);
}


void AcroDiffusionIntegrator::AssembleMatrix() {
  
  if (haveTensorBasis && Btil.Size() == 0) {
    ComputeBTilde();
  }

  if (!S.IsInitialized()) {
    if (haveTensorBasis) {
      if (nDim == 1) {
        S.Init(nElem, nDof1D, nDof1D);
        if (onGPU) {S.SwitchToGPU();}
      } else if (nDim == 2) {
        S.Init(nElem, nDof1D, nDof1D, nDof1D, nDof1D);
        if (onGPU) {S.SwitchToGPU();}
      } else if (nDim == 3) {
        S.Init(nElem, nDof1D, nDof1D, nDof1D, nDof1D, nDof1D, nDof1D);
        if (onGPU) {S.SwitchToGPU();}
      }
    } else {
      S.Init(nElem, nDof, nDof);
      if (onGPU) {S.SwitchToGPU();}
    }
  }


  if (haveTensorBasis) {
    if (nDim == 1) {
      TE["S_e_i1_j1 = Btil_m_n_k1_i1_j1 D_e_m_n_k1"]
        (S, *Btil[0], D);
    } else if (nDim == 2) {
      TE["S_e_i1_i2_j1_j2 = Btil1_m_n_k1_i1_j1 Btil2_m_n_k2_i2_j2 D_e_m_n_k1_k2"]
        (S, *Btil[0], *Btil[1], D);
    } else if (nDim == 3) {
      TE["S_e_i1_i2_i3_j1_j2_j3 = Btil1_m_n_k1_i1_j1 Btil2_m_n_k2_i2_j2 Btil3_m_n_k3_i3_j3 D_e_m_n_k1_k2_k3"]
        (S, *Btil[0], *Btil[1], *Btil[2], D);
    }
  } else {
    TE["S_e_i_j = G_k_i_m G_k_i_n D_e_m_n_k"]
      (S, G, G, D);
  }
}

void AcroDiffusionIntegrator::Mult(OccaVector &v) {
  
  if (!U.IsInitialized() && haveTensorBasis) {
    if (nDim == 1) {
      U.Init(nDim, nElem, nQuad1D);
      W.Init(nDim, nElem, nQuad1D);
      if (onGPU) {
        U.SwitchToGPU(); 
        W.SwitchToGPU();
      }
    } else if (nDim == 2) {
      U.Init(nDim, nElem, nQuad1D, nQuad1D);
      W.Init(nDim, nElem, nQuad1D, nQuad1D);
      T1.Init(nElem,nDof1D,nQuad1D);
      if (onGPU) {
        U.SwitchToGPU();
        W.SwitchToGPU();
        T1.SwitchToGPU();
      }
    } else if (nDim == 3) {
      U.Init(nDim, nElem, nQuad1D, nQuad1D, nQuad1D);
      W.Init(nDim, nElem, nQuad1D, nQuad1D, nQuad1D);
      T1.Init(nElem,nDof1D,nQuad1D,nQuad1D);
      T2.Init(nElem,nDof1D,nDof1D,nQuad1D);
      if (onGPU) {
        U.SwitchToGPU();
        W.SwitchToGPU();
        T1.SwitchToGPU();
        T2.SwitchToGPU();
      }
    }
  }

  double *v_ptr = *((double**) v.GetData().getHandle());
  if (haveTensorBasis) {
    if (nDim == 1) {
      acro::Tensor V(nElem, nDof1D, 
                     v_ptr, v_ptr, onGPU);
      acro::Tensor X(nElem, nDof1D,
                     v_ptr, v_ptr, onGPU);
      TE["U_n_e_k1 = G_k1_i1 V_e_i1"](U, G, V);
      TE["W_m_e_k1 = D_e_m_n_k1 U_n_e_k1"](W, D, U);
      TE["X_e_i1 = G_k1_i1 W_m_e_k1"](X, G, W);
    } else if (nDim == 2) {
      acro::Tensor V(nElem, nDof1D, nDof1D, 
                     v_ptr, v_ptr, onGPU);
      acro::Tensor X(nElem, nDof1D, nDof1D,
                     v_ptr, v_ptr, onGPU);
      acro::SliceTensor U1(U, 0), U2(U, 1);
      acro::SliceTensor W1(W, 0), W2(W, 1);

      //U1_e_k1_k2 = G_k1_i1 B_k2_i2 V_e_i1_i2
      TE["BV_e_i1_k2 = B_k2_i2 V_e_i1_i2"](T1, B, V);
      TE["U1_e_k1_k2 = G_k1_i1 BV_i1_k2"](U1, G, T1);

      //U2_e_k1_k2 = B_k1_i1 G_k2_i2 V_e_i1_i2
      TE["GV_e_i1_k2 = B_k2_i2 V_e_i1_i2"](T1, G, V);
      TE["U2_e_k1_k2 = B_k1_i1 GV_e_i1_k2"](U2, B, T1);

      TE["W_m_e_k1_k2 = D_e_m_n_k1_k2 U_n_e_k1_k2"](W, D, U);

      //X_e_i1_i2 = G_k1_i1 B_k2_i2 W1_e_k1_k2
      TE["BW1_e_i2_k1 = B_k2_i2 W1_e_k1_k2"](T1, B, W1);
      TE["X_e_i1_i2 = G_k1_i1 BW1_e_i2_k1"](X, G, T1);

      //X_e_i1_i2 += B_k1_i1 G_k2_i2 W2_e_k1_k2
      TE["GW1_e_i2_k1 = G_k2_i2 W1_e_k1_k2"](T1, G, W1);
      TE["X_e_i1_i2 += B_k1_i1 GW1_e_i2_k1"](X, B, T1);
    } else if (nDim == 3) {
      acro::Tensor V(nElem, nDof1D, nDof1D, nDof1D, 
                     v_ptr, v_ptr, onGPU);
      acro::Tensor X(nElem, nDof1D, nDof1D, nDof1D,
                     v_ptr, v_ptr, onGPU);
      acro::SliceTensor U1(U, 0), U2(U, 1), U3(U, 2);
      acro::SliceTensor W1(W, 0), W2(W, 1), W3(W, 2);

      //U1_e_k1_k2_k3 = G_k1_i1 B_k2_i2 B_k3_i3 V_e_i1_i2_i3
      TE["BV_e_i1_i2_k3 = B_k3_i3 V_e_i1_i2_i3"](T2, B, V);
      TE["BBV_e_i1_k2_k3 = B_k2_i2 BV_e_i1_i2_k3"](T1, B, T2);
      TE["U1_e_k1_k2_k3 = G_k1_i1 BBV_e_i1_k2_k3"](U1, G, T1);

      //U2_e_k1_k2_k3 = B_k1_i1 G_k2_i2 B_k3_i3 V_e_i1_i2_i3
      TE["GBV_e_i1_k2_k3 = G_k2_i2 BV_e_i1_i2_k3"](T1, G, T2);
      TE["U2_e_k1_k2_k3 = B_k1_i1 GBV_e_i1_k2_k3"](U2, B, T1);

      //U3_e_k1_k2_k3 = B_k1_i1 B_k2_i2 G_k3_i3 V_e_i1_i2_i3
      TE["GV_e_i1_i2_k3 = G_k3_i3 V_e_i1_i2_i3"](T2, G, V);
      TE["BGV_e_i1_k2_k3 = B_k2_i2 GV_e_i1_i2_k3"](T1, B, T2);
      TE["U3_e_k1_k2_k3 = B_k1_i1 BGV_e_i1_k2_k3"](U3, B, T1);
   
      TE["W_m_e_k1_k2_k3 = D_e_m_n_k1_k2_k3 U_n_e_k1_k2_k3"](W, D, U);

      //X_e_i1_i2_i3 =  G_k1_i1 B_k2_i2 B_k3_i3 W1_e_k1_k2_k3
      TE["BW1_e_i3_k1_k2 = B_k3_i3 W1_e_k1_k2_k3"](T1, B, W1);
      TE["BBW1_e_i2_i3_k1 = B_k2_i2 BW1_e_i3_k1_k2"](T2, B, T1);
      TE["X_e_i1_i2_i3 = G_k1_i1 BBW1_e_i2_i3_k1"](X, G, T2);

      //X_e_i1_i2_i3 =  B_k1_i1 G_k2_i2 B_k3_i3 W2_e_k1_k2_k3
      TE["BW2_e_i3_k1_k2 = B_k3_i3 W2_e_k1_k2_k3"](T1, B, W2);
      TE["GBW2_e_i2_i3_k1 = G_k2_i2 BW2_e_i3_k1_k2"](T2, G, T1);
      TE["X_e_i1_i2_i3 += B_k1_i1 GBW2_e_i2_i3_k1"](X, B, T2);

      //X_e_i1_i2_i3 =  B_k1_i1 B_k2_i2 G_k3_i3 W3_e_k1_k2_k3
      TE["GW3_e_i3_k1_k2 = G_k3_i3 W3_e_k1_k2_k3"](T1, G, W3);
      TE["BGW3_e_i2_i3_k1 = B_k2_i2 GW3_e_i3_k1_k2"](T2, B, T1);
      TE["X_e_i1_i2_i3 += B_k1_i1 BGW3_e_i2_i3_k1"](X, B, T2);
    }
  } else {
    mfem_error("AcroDiffusionIntegrator partial assembly on simplices not supported");
  }

}

}

#endif
