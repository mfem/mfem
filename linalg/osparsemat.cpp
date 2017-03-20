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

  OccaSparseMatrix::OccaSparseMatrix(const SparseMatrix &m,
                                     occa::memory reorderIndices_,
                                     occa::memory mappedIndices_,
                                     const occa::properties &props) :
    Operator(m.Height(), m.Width()) {

    Setup(occa::currentDevice(), m, reorderIndices, mappedIndices_, props);
  }

  OccaSparseMatrix::OccaSparseMatrix(occa::device device, const SparseMatrix &m,
                                     const occa::properties &props) :
    Operator(m.Height(), m.Width()) {

    Setup(device, m, props);
  }

  OccaSparseMatrix::OccaSparseMatrix(occa::device device, const SparseMatrix &m,
                                     occa::memory reorderIndices_,
                                     occa::memory mappedIndices_,
                                     const occa::properties &props) :
    Operator(m.Height(), m.Width()) {

    Setup(device, m, reorderIndices, mappedIndices_, props);
  }

  OccaSparseMatrix::OccaSparseMatrix(const int height_, const int width_,
                                     occa::memory offsets_,
                                     occa::memory indices_,
                                     occa::memory weights_,
                                     const occa::properties &props) :
    Operator(height_, width_),
    offsets(offsets_),
    indices(indices_),
    weights(weights_) {

    SetupKernel(offsets.getDevice(), props);
  }

  OccaSparseMatrix::OccaSparseMatrix(const int height_, const int width_,
                                     occa::memory offsets_,
                                     occa::memory indices_,
                                     occa::memory weights_,
                                     occa::memory reorderIndices_,
                                     occa::memory mappedIndices_,
                                     const occa::properties &props) :
    Operator(height_, width_),
    offsets(offsets_),
    indices(indices_),
    weights(weights_),
    reorderIndices(reorderIndices_),
    mappedIndices(mappedIndices_) {

    SetupKernel(offsets.getDevice(), props);
  }

  void OccaSparseMatrix::Setup(occa::device device, const SparseMatrix &m,
                               const occa::properties &props) {
    Setup(device, m, occa::memory(), occa::memory(), props);
  }

  void OccaSparseMatrix::Setup(occa::device device, const SparseMatrix &m,
                               occa::memory reorderIndices_,
                               occa::memory mappedIndices_,
                               const occa::properties &props) {

    const int nnz = m.GetI()[height];
    offsets = device.malloc((height + 1) * sizeof(int), m.GetI());
    indices = device.malloc(nnz * sizeof(int)   , m.GetJ());
    weights = device.malloc(nnz * sizeof(double), m.GetData());

    reorderIndices = reorderIndices_;
    mappedIndices  = mappedIndices_;

    SetupKernel(device, props);
  }

  void OccaSparseMatrix::SetupKernel(occa::device device,
                                     const occa::properties &props) {

    const bool hasOutIndices = mappedIndices.isInitialized();

    const occa::properties defaultProps("defines: {"
                                        "  TILESIZE: 256,"
                                        "}");

    mapKernel = device.buildKernel("occa://mfem/linalg/mappings.okl",
                                   "MapSubVector",
                                   defaultProps + props);

    multKernel = device.buildKernel("occa://mfem/linalg/sparse.okl",
                                    hasOutIndices ? "MappedMult" : "Mult",
                                    defaultProps + props);
  }

  void OccaSparseMatrix::Mult(const OccaVector &x, OccaVector &y) const {
    if (mappedIndices.isInitialized()) {
      mapKernel((int) (reorderIndices.entries<int>() / 2),
                reorderIndices,
                x, y);
      multKernel((int) (mappedIndices.entries<int>()),
                 offsets, indices, weights,
                 mappedIndices,
                 x, y);
    } else {
      multKernel((int) height,
                 offsets, indices, weights,
                 x, y);
    }
  }

  OccaSparseMatrix* CreateMappedSparseMatrix(occa::device device,
                                             const SparseMatrix &m,
                                             const occa::properties &props) {
    const int mHeight = m.Height();
    const int mWidth  = m.Width();

    // Count indices that are only reordered (true dofs)
    const int *I = m.GetI();
    const int *J = m.GetJ();
    const double *D = m.GetData();

    int trueCount = 0;
    for (int i = 0; i < mHeight; ++i) {
      trueCount += ((I[i + 1] - I[i]) == 1);
    }
    const int dupCount = (mHeight - trueCount);

    // Create the reordering map for entries that aren't modified (true dofs)
    int *reorderIndices = new int[2 * trueCount];
    int *mappedIndices  = new int[dupCount];
    int trueIdx = 0, dupIdx = 0;
    for (int i = 0; i < mHeight; ++i) {
      const int i1 = I[i];
      if ((I[i + 1] - i1) == 1) {
        reorderIndices[trueIdx++] = J[i1];
        reorderIndices[trueIdx++] = i;
      } else {
        mappedIndices[dupIdx++] = i;
      }
    }

    // Extract sparse matrix without reordered identity
    const int dupNnz = I[mHeight] - trueCount;
    int *offsets    = new int[dupCount + 1];
    int *indices    = new int[dupNnz];
    double *weights = new double[dupNnz];

    int nnz = 0;
    offsets[0] = 0;
    for (int i = 0; i < dupCount; ++i) {
      const int idx = mappedIndices[i];
      const int offStart = I[idx];
      const int offEnd   = I[idx + 1];
      offsets[i + 1] = offsets[i] + (offEnd - offStart);
      for (int j = offStart; j < offEnd; ++j) {
        indices[nnz] = J[j];
        weights[nnz] = D[j];
        ++nnz;
      }
    }

    occa::memory mOffsets = device.malloc((dupCount + 1) * sizeof(int), offsets);
    occa::memory mIndices = device.malloc(dupNnz * sizeof(int)   , indices);
    occa::memory mWeights = device.malloc(dupNnz * sizeof(double), weights);

    occa::memory mReorderIndices = device.malloc(2 * trueCount * sizeof(int),
                                                 reorderIndices);
    occa::memory mMappedIndices = device.malloc(dupCount * sizeof(int),
                                                mappedIndices);

    delete [] offsets;
    delete [] indices;
    delete [] weights;
    delete [] reorderIndices;
    delete [] mappedIndices;

    return new OccaSparseMatrix(mHeight, mWidth,
                                mOffsets, mIndices, mWeights,
                                mReorderIndices, mMappedIndices,
                                props);
  }
}

#endif
