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

#include "amassinteg.hpp"

namespace mfem {

AcroMassIntegrator::AcroMassIntegrator(OccaCoefficient &q) :
  Q(q) {}

AcroMassIntegrator::~AcroMassIntegrator() {

}

OccaIntegrator* AcroMassIntegrator::CreateInstance() {
  return new AcroMassIntegrator(Q);
}

std::string AcroMassIntegrator::GetName() {
  return "AcroMassIntegrator";
}

void AcroMassIntegrator::SetupIntegrationRule() {
  const FiniteElement &fe = *(fespace->GetFE(0));
  ir = &(GetMassIntegrationRule(fe, fe));
}

void AcroMassIntegrator::Setup() {
  AcroIntegrator::Setup();

  occa::properties props;
  SetupProperties(props);
  Q.Setup(ir, props);
}


void AcroMassIntegrator::Assemble() {
  //Get the jacobians and compute D with them
  OccaGeometry geom = GetGeometry();
  double *jacdet_ptr = (double*) geom.detJ.memory().ptr();

  // Keep quadQ so GC won't free it
  OccaVector quadQ;
  if (!Q.isConstant()) {
    quadQ = Q.Eval();
  }
  double *q_ptr = (double*) quadQ.GetData().ptr();

  if (hasTensorBasis) {
    if (nDim == 1) {
      D.Init(nElem, nDim, nDim, nQuad1D);
      acro::Tensor Jdet(nElem, nQuad1D,
                        jacdet_ptr, jacdet_ptr, onGPU);
      if (!Q.isConstant()) {
        acro::Tensor q(nElem, nQuad1D,
                       q_ptr, q_ptr, onGPU);
        TE["D_e_m_n_k = W_k q_e_k Jdet_e_k"]
          (D, W, q, Jdet);
      } else {
        TE["D_e_m_n_k = W_k Jdet_e_k"]
          (D, W, Jdet);
      }
    } else if (nDim == 2) {
      D.Init(nElem, nDim, nDim, nQuad1D, nQuad1D);
      acro::Tensor Jdet(nElem, nQuad1D, nQuad1D,
                        jacdet_ptr, jacdet_ptr, onGPU);
      if (!Q.isConstant()) {
        acro::Tensor q(nElem, nQuad1D, nQuad1D,
                       q_ptr, q_ptr, onGPU);
        TE["D_e_m_n_k1_k2 = W_k1_k2 q_e_k1_k2 Jdet_e_k1_k2"]
          (D, Jdet);
      } else {
        TE["D_e_m_n_k1_k2 = W_k1_k2 Jdet_e_k1_k2"]
          (D, Jdet);
      }
    } else if (nDim == 3){
      D.Init(nElem, nDim, nDim, nQuad1D, nQuad1D, nQuad1D);
      acro::Tensor Jdet(nElem, nQuad1D, nQuad1D, nQuad1D,
                        jacdet_ptr, jacdet_ptr, onGPU);
      if (!Q.isConstant()) {
        acro::Tensor q(nElem, nQuad1D, nQuad1D, nQuad1D,
                       q_ptr, q_ptr, onGPU);
        TE["D_e_m_n_k1_k2_k3 = W_k1_k2_k3 q_e_k1_k2_k3 Jdet_e_k1_k2_k3"]
          (D, W, q, Jdet);
      } else {
        TE["D_e_m_n_k1_k2_k3 = W_k1_k2_k3 Jdet_e_k1_k2_k3"]
          (D, W, Jdet);
      }
    } else {
      mfem_error("AcroMassIntegrator tensor computations don't support dim > 3.");
    }
  } else {
    D.Init(nElem, nDim, nDim, nQuad);
    acro::Tensor Jdet(nElem, nQuad,
                      jacdet_ptr, jacdet_ptr, onGPU);
    if (!Q.isConstant()) {
      acro::Tensor q(nElem, nQuad,
                     q_ptr, q_ptr, onGPU);
      TE["D_e_m_n_k = W_k q_e_k Jdet_e_k"]
        (D, W, q, Jdet);
    } else {
      TE["D_e_m_n_k = W_k Jdet_e_k"]
        (D, W, Jdet);
    }
  }

  if (Q.isConstant()) {
    D.Mult(Q.GetConstantValue());
  }
}


void AcroMassIntegrator::AssembleMatrix() {

  if (!M.IsInitialized()) {
    if (hasTensorBasis) {
      if (nDim == 1) {
        M.Init(nElem, nDof1D, nDof1D);
        if (onGPU) {M.SwitchToGPU();}
      } else if (nDim == 2) {
        M.Init(nElem, nDof1D, nDof1D, nDof1D, nDof1D);
        if (onGPU) {M.SwitchToGPU();}
      } else if (nDim == 3) {
        M.Init(nElem, nDof1D, nDof1D, nDof1D, nDof1D, nDof1D, nDof1D);
        if (onGPU) {M.SwitchToGPU();}
      }
    } else {
      M.Init(nElem, nDof, nDof);
      if (onGPU) {M.SwitchToGPU();}
    }
  }

  if (hasTensorBasis) {
    if (nDim == 1) {
      TE["M_e_i1_j1 = B_k1_i1 B_k1_j1 D_e_k1"](M, B, B, D);
    } else if (nDim == 2) {
      TE["M_e_i1_i2_j1_j2 = B_k1_i1 B_k1_j1 B_k2_i2 B_k2_j2 D_e_k1_k2"](M, B, B, B, B, D);
    } else if (nDim == 3) {
      TE["M_e_i1_i2_i3_j1_j2_j3 = B_k1_i1 B_k1_j1 B_k2_i2 B_k2_j2 B_k3_i3 B_k3_j3 D_e_k1_k2_k3"](M, B, B, B, B, B, B, D);
    }
  } else {
    TE["M_e_i_j = B_k_i B_k_j D_e_k"](M, B, B, D);
  }
}

void AcroMassIntegrator::Mult(OccaVector &v) {

  if (!T1.IsInitialized() && hasTensorBasis) {
    if (nDim == 1) {
      T1.Init(nElem, nQuad1D);
      if (onGPU) {
        T1.SwitchToGPU();
      }
    } else if (nDim == 2) {
      T1.Init(nElem, nQuad1D, nDof1D);
      T2.Init(nElem, nQuad1D, nQuad1D);
      if (onGPU) {
        T1.SwitchToGPU();
        T2.SwitchToGPU();
      }
    } else if (nDim == 3) {
      T1.Init(nElem, nQuad1D, nDof1D, nDof1D);
      T2.Init(nElem, nQuad1D, nQuad1D, nDof1D);
      T3.Init(nElem, nQuad1D, nQuad1D, nQuad1D);
      if (onGPU) {
        T1.SwitchToGPU();
        T2.SwitchToGPU();
        T3.SwitchToGPU();
      }
    }
  }

  double *v_ptr = (double*) v.GetData().ptr();
  if (hasTensorBasis) {
    if (nDim == 1) {
      acro::Tensor V(nElem, nDof1D, v_ptr, v_ptr, onGPU);
      acro::Tensor X(nElem, nDof1D, v_ptr, v_ptr, onGPU);
      TE["T1_e_k1 = D_e_k1 B_k1_j1 V_e_j1"](T1, D, B, V);
      TE["X_e_i1 = B_k1_i1 T1_e_k1"](X, B, T1);
    } else if (nDim == 2) {
      acro::Tensor V(nElem, nDof1D, nDof1D, v_ptr, v_ptr, onGPU);
      acro::Tensor X(nElem, nDof1D, nDof1D, v_ptr, v_ptr, onGPU);
      TE["T1_e_k2_j1 = B_k2_j2 V_j1_j2"](T1, B, V);
      TE["T2_e_k1_k2 = D_e_k1_k2 B_k1_j1 T1_e_k2_j1"](T2, D, B, T1);
      TE["T1_e_k1_i2 = B_k2_i2 T2_e_k1_k2"](T1, B, T2);
      TE["X_e_i1_i2 = B_k1_i1 T1_e_k1_i2"](X, B, T1);
    } else if (nDim == 3) {
      acro::Tensor V(nElem, nDof1D, nDof1D, nDof1D, v_ptr, v_ptr, onGPU);
      acro::Tensor X(nElem, nDof1D, nDof1D, nDof1D, v_ptr, v_ptr, onGPU);
      TE["T1_e_k3_j1_j2 = B_k3_j3 V_j1_j2_j3"](T1, B, V);
      TE["T2_e_k2_k3_j1 = B_k2_j2 T1_e_k3_j1_j2"](T2, B, T1);
      TE["T3_e_k1_k2_k3 = D_e_k1_k2_k3 B_k1_j1 T2_e_k2_k3_j1"](T3, D, B, T2);
      TE["T2_e_k1_k2_i3 = B_k3_i3 T3_e_k1_k2_k3"](T2, B, T3);
      TE["T1_e_k1_i2_i3 = B_k2_i2 T2_e_k1_k2_i3"](T1, B, T2);
      TE["X_e_i1_i2_i3 = B_k1_i1 T1_e_k1_i2_i3"](X, B, T1);
    }
  } else {
    mfem_error("AcroMassIntegrator partial assembly on simplices not supported");
  }

}

}

#endif
