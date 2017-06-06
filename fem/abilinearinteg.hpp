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
#  ifndef MFEM_ACROTENSOR_BILINEARINTEG
#  define MFEM_ACROTENSOR_BILINEARINTEG

#include "AcroTensor.hpp"
#include "obilininteg.hpp"

namespace mfem {

class AcroIntegrator : public OccaIntegrator {
protected:
  acro::TensorEngine TE;

  int nDim;
  int nElem;
  int nDof;
  int nQuad;
  int nDof1D;
  int nQuad1D;
  bool onGPU;

  acro::Tensor B, G;         //Basis and dbasis evaluated on the quad points
  acro::Tensor W;            //Integration weights

public:
  AcroIntegrator();
  virtual ~AcroIntegrator();
  virtual void Setup();

  virtual OccaIntegrator* CreateInstance() = 0;
  virtual std::string GetName() = 0;
  virtual void Assemble() = 0;
  virtual void AssembleMatrix() = 0;
  virtual void MultAdd(OccaVector &x, OccaVector &y) = 0;
};

}

#  endif
#endif