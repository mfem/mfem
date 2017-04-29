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

#if defined(MFEM_USE_OCCA) && defined(MFEM_USE_ACROTENSOR)
#  ifndef MFEM_ACROTENSOR_BILININTEG
#  define MFEM_ACROTENSOR_BILININTEG

#include "AcroTensor.hpp"
#include "obilininteg.hpp"

namespace mfem {

class AcroDiffusionIntegrator : public OccaIntegrator {
private:
  acro::TensorEngine TE;

  Coefficient &Q;
  OccaDofQuadMaps maps;

  int nDim;
  int nElem;
  int nDof;
  int nQuad;
  int nDof1D;
  int nQuad1D;
  bool haveTensorBasis;
  bool onGPU;

  acro::Tensor B, G;         //Basis and dbasis evaluated on the quad points
  Array<acro::Tensor*> Btil; //Btilde used to compute stiffness matrix
  acro::Tensor D;            //Product of integration weight, physical consts, and element shape info
  acro::Tensor S;            //The assembled local stiffness matrices
  acro::Tensor U, W;         //Intermediate computations for tensor product partial assembly

  void ComputeBTilde();
  void ComputeD(occa::array<double> &jac, 
                occa::array<double> &jacinv, 
                occa::array<double> &jacdet);

public:
  AcroDiffusionIntegrator(Coefficient &q);
  virtual ~AcroDiffusionIntegrator();

  virtual OccaIntegrator* CreateInstance();

  virtual std::string GetName();

  virtual void Setup();

  virtual void Assemble();
  virtual void Mult(OccaVector &x);
};

}

#  endif
#endif
