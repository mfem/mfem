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

#include "ointerpolation.hpp"

namespace mfem {
  void CreateRAPOperators(occa::device device,
                          const Operator *P, const Operator *R,
                          Operator *&OccaP, Operator *&OccaR,
                          const occa::properties &props) {
    if (!P) {
      OccaP = OccaR = NULL;
      return;
    }

    occa::memory childIndices, childOffsets, childWeights;
    occa::memory parentIndices, parentOffsets, parentWeights;

    OccaR = new OccaRestrictionOperator(device,
                                        R->Height(), R->Width(),
                                        childIndices,
                                        props);

    OccaP = new OccaProlongationOperator(device,
                                         P->Height(), P->Width(),
                                         childIndices, childOffsets, childWeights,
                                         parentIndices, parentOffsets, parentWeights,
                                         props);
  }

  OccaRestrictionOperator::OccaRestrictionOperator(occa::device device,
                                                   const int height, const int width,
                                                   occa::memory childIndices_,
                                                   const occa::properties &props) :
    Operator(height, width),
    childIndices(childIndices_) {

    zeroOutChildrenKernel = device.buildKernel("occa://mfem/fem/interpolation.okl",
                                               "zeroOutChildren",
                                               props);
  }

  void OccaRestrictionOperator::Mult(const OccaVector &x, OccaVector &y) const {
    zeroOutChildrenKernel(height, x, y);
  }

  OccaProlongationOperator::OccaProlongationOperator(occa::device device,
                                                     const int height, const int width,
                                                     occa::memory childOffsets_,
                                                     occa::memory childIndices_,
                                                     occa::memory childWeights_,
                                                     occa::memory parentOffsets_,
                                                     occa::memory parentIndices_,
                                                     occa::memory parentWeights_,
                                                     const occa::properties &props) :
    Operator(height, width),
    childOffsets(childOffsets_),
    childIndices(childIndices_),
    childWeights(childWeights_),
    parentOffsets(parentOffsets_),
    parentIndices(parentIndices_),
    parentWeights(parentWeights_) {

    updateKernel = device.buildKernel("occa://mfem/fem/interpolation.okl",
                                      "update",
                                      props);
  }

  void OccaProlongationOperator::Mult(const OccaVector &x, OccaVector &y) const {
    updateKernel(height,
                 parentOffsets, parentIndices, parentWeights,
                 x, y);
  }

  void OccaProlongationOperator::MultTranspose(const OccaVector &x, OccaVector &y) const {
    updateKernel(width,
                 childOffsets, childIndices, childWeights,
                 x, y);
  }
}

#endif
