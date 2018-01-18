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
#  ifndef MFEM_ACROTENSOR_MASSINTEG
#  define MFEM_ACROTENSOR_MASSINTEG

#include "abilinearinteg.hpp"

namespace mfem {

class AcroMassIntegrator : public AcroIntegrator {
private:
  OccaCoefficient Q;
  acro::Tensor D;            //Product of integration weight, physical consts, and element shape info
  acro::Tensor M;            //The assembled local mass matrices
  acro::Tensor T1, T2, T3;   //Intermediate computations for tensor product partial assembly

public:
  AcroMassIntegrator(const OccaCoefficient &q);
  virtual ~AcroMassIntegrator();

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
