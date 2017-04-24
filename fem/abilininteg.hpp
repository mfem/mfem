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
  //---[ Diffusion Integrator ]---------
  class AcroDiffusionIntegrator : public OccaIntegrator {
  private:
    acrobatic::TensorEngine TE;

    Coefficient &Q;
    OccaDofQuadMaps maps;

    int elements;
    int numDofs;
    int numQuad;

    occa::kernel assembleKernel, multKernel;

    occa::array<double> coefficients;
    occa::array<double> jacobian, assembledOperator;

    bool hasConstantCoefficient;

  public:
    AcroDiffusionIntegrator(Coefficient &q);
    virtual ~AcroDiffusionIntegrator();

    virtual OccaIntegrator* CreateInstance();

    virtual std::string GetName();

    virtual void Setup();

    virtual void Assemble();
    virtual void Mult(OccaVector &x);
  };
  //====================================
}

#  endif
#endif
