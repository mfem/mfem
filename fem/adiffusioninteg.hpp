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
#  ifndef MFEM_ACROTENSOR_DIFFUSIONINTEG
#  define MFEM_ACROTENSOR_DIFFUSIONINTEG

#include "abilinearinteg.hpp"

namespace mfem {

class AcroDiffusionIntegrator : public AcroIntegrator {
private:
  OccaCoefficient Q;
  Array<acro::Tensor*> Btil; //Btilde used to compute stiffness matrix
  acro::Tensor D;            //Product of integration weight, physical consts, and element shape info
  acro::Tensor S;            //The assembled local stiffness matrices
  acro::Tensor U, Z, T1, T2; //Intermediate computations for tensor product partial assembly

  void ComputeBTilde();

public:
  AcroDiffusionIntegrator(const OccaCoefficient &q);
  virtual ~AcroDiffusionIntegrator();

  virtual OccaIntegrator* CreateInstance();

  virtual std::string GetName();

  virtual void SetupIntegrationRule();

  virtual void Setup();

  virtual void Assemble();
  virtual void AssembleMatrix();
  virtual void MultAdd(OccaVector &x, OccaVector &y);
};

}

#  endif
#endif
