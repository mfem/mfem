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
#  ifndef MFEM_OCCA_SPARSEMAT
#  define MFEM_OCCA_SPARSEMAT

#include "linalg.hpp"

#include "occa.hpp"

namespace mfem {
  // [MISSING] Proper destructors
  class OccaSparseMatrix : public Operator {
  public:
    occa::memory offsets, indices, weights;
    occa::memory reorderIndices, mappedIndices;
    occa::kernel mapKernel, multKernel;

    OccaSparseMatrix();

    OccaSparseMatrix(const SparseMatrix &m,
                     const occa::properties &props = occa::properties());

    OccaSparseMatrix(const SparseMatrix &m,
                     occa::memory reorderIndices_,
                     occa::memory mappedIndices_,
                     const occa::properties &props = occa::properties());

    OccaSparseMatrix(occa::device device, const SparseMatrix &m,
                     const occa::properties &props = occa::properties());

    OccaSparseMatrix(occa::device device, const SparseMatrix &m,
                     occa::memory reorderIndices_,
                     occa::memory mappedIndices_,
                     const occa::properties &props = occa::properties());

    OccaSparseMatrix(const int height_, const int width_,
                     occa::memory offsets_,
                     occa::memory indices_,
                     occa::memory weights_,
                     const occa::properties &props = occa::properties());

    OccaSparseMatrix(const int height_, const int width_,
                     occa::memory offsets_,
                     occa::memory indices_,
                     occa::memory weights_,
                     occa::memory reorderIndices_,
                     occa::memory mappedIndices_,
                     const occa::properties &props = occa::properties());

    void Setup(occa::device device, const SparseMatrix &m,
               const occa::properties &props);

    void Setup(occa::device device, const SparseMatrix &m,
               occa::memory reorderIndices_,
               occa::memory mappedIndices_,
               const occa::properties &props);

    void SetupKernel(occa::device device,
                     const occa::properties &props);

    virtual void Mult(const OccaVector &x, OccaVector &y) const;

    void Free();
  };

  OccaSparseMatrix* CreateMappedSparseMatrix(occa::device device,
                                             const SparseMatrix &m,
                                             const occa::properties &props = occa::properties());
}

#  endif
#endif
