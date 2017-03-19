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

#include "osparsemat.hpp"

namespace mfem {
  OccaSparseMatrix::OccaSparseMatrix(const SparseMatrix &m,
                                     const occa::properties &props) :
    Operator(m.Height(), m.Width()) {

    Setup(occa::currentDevice(), m, props);
  }

  OccaSparseMatrix::OccaSparseMatrix(occa::device device, const SparseMatrix &m,
                                     const occa::properties &props) :
    Operator(m.Height(), m.Width()) {

    Setup(device, m, props);
  }

  void OccaSparseMatrix::Setup(occa::device device, const SparseMatrix &m,
                               const occa::properties &props) {

    const int nnz = m.GetI()[height];
    offsets = device.malloc((height + 1) * sizeof(int), m.GetI());
    indices = device.malloc(nnz * sizeof(int)   , m.GetJ());
    weights = device.malloc(nnz * sizeof(double), m.GetData());

    occa::properties defaultProps("defines: {"
                                  "  TILESIZE: 256,"
                                  "}");

    multKernel = device.buildKernel("occa://mfem/linalg/sparse.okl",
                                    "Mult",
                                    defaultProps + props);

    multTransposeKernel = device.buildKernel("occa://mfem/linalg/sparse.okl",
                                             "MultTranspose",
                                             defaultProps + props);
  }

  void OccaSparseMatrix::Mult(const OccaVector &x, OccaVector &y) const {
    multKernel((int) height,
               offsets, indices, weights,
               x, y);
  }

  void OccaSparseMatrix::MultTranspose(const OccaVector &x, OccaVector &y) const {
    y = 0;
    multTransposeKernel((int) height,
                        offsets, indices, weights,
                        x, y);
  }
}

#endif
