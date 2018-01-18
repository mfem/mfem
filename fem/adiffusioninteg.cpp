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

#include "adiffusioninteg.hpp"

namespace mfem {

AcroDiffusionIntegrator::AcroDiffusionIntegrator(const OccaCoefficient &q) :
  Q(q) {}

AcroDiffusionIntegrator::~AcroDiffusionIntegrator() {

}

OccaIntegrator* AcroDiffusionIntegrator::CreateInstance() {
  return new AcroDiffusionIntegrator(Q);
}

std::string AcroDiffusionIntegrator::GetName() {
  return "AcroDiffusionIntegrator";
}

void AcroDiffusionIntegrator::SetupIntegrationRule() {
  const FiniteElement &trialFE = *(trialFespace->GetFE(0));
  const FiniteElement &testFE  = *(testFespace->GetFE(0));
  ir = &(GetDiffusionIntegrationRule(trialFE, testFE));
}

void AcroDiffusionIntegrator::Setup() {
  AcroIntegrator::Setup();
  Q.Setup(*this, props);
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


void AcroDiffusionIntegrator::Assemble() {
  //Get the jacobians and compute D with them
  OccaGeometry geom = GetGeometry();
  double *jac_ptr    = (double*) geom.J.memory().ptr();
  double *jacinv_ptr = (double*) geom.invJ.memory().ptr();
  double *jacdet_ptr = (double*) geom.detJ.memory().ptr();

  // Keep quadQ so GC won't free it
  OccaVector quadQ;
  if (!Q.IsConstant()) {
    quadQ = Q.Eval();
  }
  double *q_ptr = (double*) quadQ.GetData().ptr();

  if (hasTensorBasis) {
    if (nDim == 1) {
      D.Init(nElem, nDim, nDim, nQuad1D);
      acro::Tensor Jdet(nElem, nQuad1D,
                        jacdet_ptr, jacdet_ptr, onGPU);
      if (!Q.IsConstant()) {
        acro::Tensor q(nElem, nQuad1D,
                       q_ptr, q_ptr, onGPU);
        TE["D_e_m_n_k = W_k Q_e_k Jdet_e_k"]
          (D, W, q, Jdet);
      } else {
        TE["D_e_m_n_k = W_k Jdet_e_k"]
          (D, W, Jdet);
      }
    } else if (nDim == 2) {
      D.Init(nElem, nDim, nDim, nQuad1D, nQuad1D);
      acro::Tensor J(nElem, nQuad1D, nQuad1D, nDim, nDim,
                     jac_ptr, jac_ptr, onGPU);
      acro::Tensor Jinv(nElem, nQuad1D, nQuad1D, nDim, nDim,
                        jacinv_ptr, jacinv_ptr, onGPU);
      acro::Tensor Jdet(nElem, nQuad1D, nQuad1D,
                        jacdet_ptr, jacdet_ptr, onGPU);
      if (!Q.IsConstant()) {
        acro::Tensor q(nElem, nQuad1D, nQuad1D,
                       q_ptr, q_ptr, onGPU);
        TE["D_e_m_n_k1_k2 = W_k1_k2 Q_e_k1_k2 Jdet_e_k1_k2 Jinv_e_k1_k2_m_n Jinv_e_k1_k2_n_m"]
          (D, W, q, Jdet, Jinv, Jinv);
      } else {
        TE["D_e_m_n_k1_k2 = W_k1_k2 Jdet_e_k1_k2 Jinv_e_k1_k2_m_n Jinv_e_k1_k2_n_m"]
          (D, W, Jdet, Jinv, Jinv);
      }
    } else if (nDim == 3){
      D.Init(nElem, nDim, nDim, nQuad1D, nQuad1D, nQuad1D);
      acro::Tensor J(nElem, nQuad1D, nQuad1D, nQuad1D, nDim, nDim,
                     jac_ptr, jac_ptr, onGPU);
      acro::Tensor Jinv(nElem, nQuad1D, nQuad1D, nQuad1D, nDim, nDim,
                        jacinv_ptr, jacinv_ptr, onGPU);
      acro::Tensor Jdet(nElem, nQuad1D, nQuad1D, nQuad1D,
                        jacdet_ptr, jacdet_ptr, onGPU);
      if (!Q.IsConstant()) {
        acro::Tensor q(nElem, nQuad1D, nQuad1D, nQuad1D,
                       q_ptr, q_ptr, onGPU);
        TE["D_e_m_n_k1_k2_k3 = W_k1_k2_k3 Q_e_k1_k2_k3 Jdet_e_k1_k2_k3 Jinv_e_k1_k2_k3_m_n Jinv_e_k1_k2_k3_n_m"]
          (D, W, q, Jdet, Jinv, Jinv);
      } else {
        TE["D_e_m_n_k1_k2_k3 = W_k1_k2_k3 Jdet_e_k1_k2_k3 Jinv_e_k1_k2_k3_m_n Jinv_e_k1_k2_k3_n_m"]
          (D, W, Jdet, Jinv, Jinv);
      }
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
    if (!Q.IsConstant()) {
      acro::Tensor q(nElem, nQuad,
                     q_ptr, q_ptr, onGPU);
      TE["D_e_m_n_k = W_k Q_e_k Jdet_e_k Jinv_e_k_m_n Jinv_e_k_n_m"]
        (D, W, q, Jdet, Jinv, Jinv);
    } else {
      TE["D_e_m_n_k = W_k Jdet_e_k Jinv_e_k_m_n Jinv_e_k_n_m"]
        (D, W, Jdet, Jinv, Jinv);
    }
  }

  if (Q.IsConstant()) {
    D.Mult(Q.GetConstantValue());
  }
}


void AcroDiffusionIntegrator::AssembleMatrix() {

  if (hasTensorBasis && Btil.Size() == 0) {
    ComputeBTilde();
  }

  if (!S.IsInitialized()) {
    if (hasTensorBasis) {
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


  if (hasTensorBasis) {
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

void AcroDiffusionIntegrator::MultAdd(OccaVector &x, OccaVector &y) {

  if (!U.IsInitialized() && hasTensorBasis) {
    if (nDim == 1) {
      U.Init(nDim, nElem, nQuad1D);
      Z.Init(nDim, nElem, nQuad1D);
      if (onGPU) {
        U.SwitchToGPU();
        Z.SwitchToGPU();
      }
    } else if (nDim == 2) {
      U.Init(nDim, nElem, nQuad1D, nQuad1D);
      Z.Init(nDim, nElem, nQuad1D, nQuad1D);
      T1.Init(nElem,nDof1D,nQuad1D);
      if (onGPU) {
        U.SwitchToGPU();
        Z.SwitchToGPU();
        T1.SwitchToGPU();
      }
    } else if (nDim == 3) {
      U.Init(nDim, nElem, nQuad1D, nQuad1D, nQuad1D);
      Z.Init(nDim, nElem, nQuad1D, nQuad1D, nQuad1D);
      T1.Init(nElem,nDof1D,nQuad1D,nQuad1D);
      T2.Init(nElem,nDof1D,nDof1D,nQuad1D);
      if (onGPU) {
        U.SwitchToGPU();
        Z.SwitchToGPU();
        T1.SwitchToGPU();
        T2.SwitchToGPU();
      }
    }
  }

  double *x_ptr = (double*) x.GetData().ptr();
  double *y_ptr = (double*) y.GetData().ptr();
  if (hasTensorBasis) {
    if (nDim == 1) {
      acro::Tensor X(nElem, nDof1D,
                     x_ptr, x_ptr, onGPU);
      acro::Tensor Y(nElem, nDof1D,
                     y_ptr, y_ptr, onGPU);
      TE["U_n_e_k1 = G_k1_i1 X_e_i1"](U, G, X);
      TE["Z_m_e_k1 = D_e_m_n_k1 U_n_e_k1"](Z, D, U);
      TE["Y_e_i1 += G_k1_i1 Z_m_e_k1"](Y, G, Z);
    } else if (nDim == 2) {
      acro::Tensor X(nElem, nDof1D, nDof1D,
                     x_ptr, x_ptr, onGPU);
      acro::Tensor Y(nElem, nDof1D, nDof1D,
                     y_ptr, y_ptr, onGPU);
      acro::SliceTensor U1(U, 0), U2(U, 1);
      acro::SliceTensor Z1(Z, 0), Z2(Z, 1);

      //U1_e_k1_k2 = G_k1_i1 B_k2_i2 X_e_i1_i2
      TE["BX_e_i1_k2 = B_k2_i2 X_e_i1_i2"](T1, B, X);
      TE["U1_e_k1_k2 = G_k1_i1 BX_e_i1_k2"](U1, G, T1);

      //U2_e_k1_k2 = B_k1_i1 G_k2_i2 X_e_i1_i2
      TE["GX_e_i1_k2 = G_k2_i2 X_e_i1_i2"](T1, G, X);
      TE["U2_e_k1_k2 = B_k1_i1 GX_e_i1_k2"](U2, B, T1);

      TE["Z_m_e_k1_k2 = D_e_m_n_k1_k2 U_n_e_k1_k2"](Z, D, U);

      //Y_e_i1_i2 += G_k1_i1 B_k2_i2 Z1_e_k1_k2
      TE["BZ1_e_i2_k1 = B_k2_i2 Z1_e_k1_k2"](T1, B, Z1);
      TE["Y_e_i1_i2 += G_k1_i1 BZ1_e_i2_k1"](Y, G, T1);

      //Y_e_i1_i2 += B_k1_i1 G_k2_i2 Z2_e_k1_k2
      TE["GZ2_e_i2_k1 = G_k2_i2 Z2_e_k1_k2"](T1, G, Z2);
      TE["Y_e_i1_i2 += B_k1_i1 GZ2_e_i2_k1"](Y, B, T1);
    } else if (nDim == 3) {
      acro::Tensor X(nElem, nDof1D, nDof1D, nDof1D,
                     x_ptr, x_ptr, onGPU);
      acro::Tensor Y(nElem, nDof1D, nDof1D, nDof1D,
                     y_ptr, y_ptr, onGPU);
      acro::SliceTensor U1(U, 0), U2(U, 1), U3(U, 2);
      acro::SliceTensor Z1(Z, 0), Z2(Z, 1), Z3(Z, 2);

      //U1_e_k1_k2_k3 = G_k1_i1 B_k2_i2 B_k3_i3 X_e_i1_i2_i3
      TE["BX_e_i1_i2_k3 = B_k3_i3 X_e_i1_i2_i3"](T2, B, X);
      TE["BBX_e_i1_k2_k3 = B_k2_i2 BX_e_i1_i2_k3"](T1, B, T2);
      TE["U1_e_k1_k2_k3 = G_k1_i1 BBX_e_i1_k2_k3"](U1, G, T1);

      //U2_e_k1_k2_k3 = B_k1_i1 G_k2_i2 B_k3_i3 X_e_i1_i2_i3
      TE["GBX_e_i1_k2_k3 = G_k2_i2 BX_e_i1_i2_k3"](T1, G, T2);
      TE["U2_e_k1_k2_k3 = B_k1_i1 GBX_e_i1_k2_k3"](U2, B, T1);

      //U3_e_k1_k2_k3 = B_k1_i1 B_k2_i2 G_k3_i3 X_e_i1_i2_i3
      TE["GX_e_i1_i2_k3 = G_k3_i3 X_e_i1_i2_i3"](T2, G, X);
      TE["BGX_e_i1_k2_k3 = B_k2_i2 GX_e_i1_i2_k3"](T1, B, T2);
      TE["U3_e_k1_k2_k3 = B_k1_i1 BGX_e_i1_k2_k3"](U3, B, T1);

      TE["Z_m_e_k1_k2_k3 = D_e_m_n_k1_k2_k3 U_n_e_k1_k2_k3"](Z, D, U);

      //Y_e_i1_i2_i3 =  G_k1_i1 B_k2_i2 B_k3_i3 Z1_e_k1_k2_k3
      TE["BZ1_e_i3_k1_k2 = B_k3_i3 Z1_e_k1_k2_k3"](T1, B, Z1);
      TE["BBZ1_e_i2_i3_k1 = B_k2_i2 BZ1_e_i3_k1_k2"](T2, B, T1);
      TE["Y_e_i1_i2_i3 += G_k1_i1 BBZ1_e_i2_i3_k1"](Y, G, T2);

      //Y_e_i1_i2_i3 =  B_k1_i1 G_k2_i2 B_k3_i3 Z2_e_k1_k2_k3
      TE["BZ2_e_i3_k1_k2 = B_k3_i3 Z2_e_k1_k2_k3"](T1, B, Z2);
      TE["GBZ2_e_i2_i3_k1 = G_k2_i2 BZ2_e_i3_k1_k2"](T2, G, T1);
      TE["Y_e_i1_i2_i3 += B_k1_i1 GBZ2_e_i2_i3_k1"](Y, B, T2);

      //Y_e_i1_i2_i3 =  B_k1_i1 B_k2_i2 G_k3_i3 Z3_e_k1_k2_k3
      TE["GZ3_e_i3_k1_k2 = G_k3_i3 Z3_e_k1_k2_k3"](T1, G, Z3);
      TE["BGZ3_e_i2_i3_k1 = B_k2_i2 GZ3_e_i3_k1_k2"](T2, B, T1);
      TE["Y_e_i1_i2_i3 += B_k1_i1 BGZ3_e_i2_i3_k1"](Y, B, T2);
    }
  } else {
    mfem_error("AcroDiffusionIntegrator partial assembly on simplices not supported");
  }

}

}

#endif
