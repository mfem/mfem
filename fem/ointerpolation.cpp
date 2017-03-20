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
  void CreateRPOperators(occa::device device,
                         const Operator *R, const Operator *P,
                         Operator *&OccaR, Operator *&OccaP) {
    if (!P) {
      OccaR = OccaP = NULL;
      return;
    }

    const SparseMatrix *pmat = dynamic_cast<const SparseMatrix*>(P);
    const SparseMatrix *rmat = dynamic_cast<const SparseMatrix*>(R);
    if (!pmat) {
      mfem_error("OccaBilinearForm can only take a NULL or SparseMatrix"
                 " prolongation operator");
    }
    if (!rmat) {
      mfem_error("OccaBilinearForm can only take a NULL or SparseMatrix"
                 " restriction operator");
    }
    const SparseMatrix *pmatT = Transpose(*pmat);

    OccaSparseMatrix *occaP  = CreateMappedSparseMatrix(device, *pmat);
    OccaSparseMatrix *occaPT = CreateMappedSparseMatrix(device, *pmatT);

    OccaP = new OccaProlongationOperator(*occaP, *occaPT);
    OccaR = new OccaRestrictionOperator(device, occaP->reorderIndices, rmat->Width());

    delete pmatT;
  }

  OccaRestrictionOperator::OccaRestrictionOperator(occa::device device,
                                                   occa::memory indices,
                                                   const int width_) :
    Operator(indices.entries<int>(), width_) {

    entries     = indices.entries<int>();
    trueIndices = indices;

    mapKernel = device.buildKernel("occa://mfem/linalg/mappings.okl",
                                   "ExtractSubVector",
                                   "defines: { TILESIZE: 256 }");
  }

  void OccaRestrictionOperator::Mult(const OccaVector &x, OccaVector &y) const {
    mapKernel(entries, trueIndices, x, y);
  }

  OccaProlongationOperator::OccaProlongationOperator(OccaSparseMatrix &multOp_,
                                                     OccaSparseMatrix &multTransposeOp_) :
    Operator(multOp_.Height(), multOp_.Width()),
    multOp(multOp_),
    multTransposeOp(multTransposeOp_) {}

  void OccaProlongationOperator::Mult(const OccaVector &x, OccaVector &y) const {
    multOp.Mult(x, y);
  }

  void OccaProlongationOperator::MultTranspose(const OccaVector &x, OccaVector &y) const {
    multTransposeOp.Mult(x, y);
  }
}

#endif
