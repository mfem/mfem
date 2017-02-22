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

#ifdef MFEM_USE_OCCA

#include "obilininteg.hpp"

namespace mfem {
  occa::kernel DiffusionIntegratorBuilder(OccaBilinearForm &bf,
                                          BilinearFormIntegrator &bfi,
                                          const occa::properties &props,
                                          const IntegratorType itype) {

    DiffusionIntegrator &integ = (DiffusionIntegrator&) bfi;

    // Get kernel name
    std::string kernelName = integ.Name();
    // Append 1D, 2D, or 3D
    kernelName += '0' + (char) bf.GetDim();
    kernelName += 'D';

    occa::properties kernelProps = props;
    // Add quadrature points
    const FiniteElement &fe = bf.GetFE(0);
    const IntegrationRule &ir = integ.GetIntegrationRule(fe, fe);
    kernelProps["kernel/defines/NUM_QPTS"] = ir.GetNPoints();

    // Hard-coded to ConstantCoefficient for now
    const ConstantCoefficient* coeff =
      (const ConstantCoefficient*) integ.GetCoefficient();

    kernelProps["kernel/defines/COEFF_EVAL(el,q)"] = coeff->constant;

    return bf.getDevice().buildKernel("occaBilinearIntegrators.okl",
                                      kernelName,
                                      kernelProps);
  }
}

#endif