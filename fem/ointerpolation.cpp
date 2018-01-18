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
#include "../general/outils.hpp"

namespace mfem {
  void CreateRPOperators(occa::device device,
                         const int size,
                         const SparseMatrix *R, const Operator *P,
                         Operator *&OccaR, Operator *&OccaP) {
    if (!P) {
      OccaR = new OccaIdentityOperator(size);
      OccaP = new OccaIdentityOperator(size);
      return;
    }

    const SparseMatrix *pmat = dynamic_cast<const SparseMatrix*>(P);

    if (R) {
      OccaSparseMatrix *occaR = CreateMappedSparseMatrix(device, *R);
      occa::array<int> reorderIndices = occaR->reorderIndices;
      delete occaR;

      OccaR = new OccaRestrictionOperator(device,
                                          R->Height(), R->Width(),
                                          reorderIndices);
    }

    if (pmat) {
      const SparseMatrix *pmatT = Transpose(*pmat);

      OccaSparseMatrix *occaP  = CreateMappedSparseMatrix(device, *pmat);
      OccaSparseMatrix *occaPT = CreateMappedSparseMatrix(device, *pmatT);

      OccaP = new OccaProlongationOperator(*occaP, *occaPT);
    } else {
      OccaP = new OccaProlongationOperator(P);
    }
  }

  OccaRestrictionOperator::OccaRestrictionOperator(occa::device device,
                                                   const int height_, const int width_,
                                                   occa::array<int> indices) :
    Operator(height_, width_) {

    entries     = indices.size() / 2;
    trueIndices = indices;

    multOp = device.buildKernel("occa://mfem/linalg/mappings.okl",
                                "ExtractSubVector",
                                "defines: { TILESIZE: 256 }");

    multTransposeOp = device.buildKernel("occa://mfem/linalg/mappings.okl",
                                         "SetSubVector",
                                         "defines: { TILESIZE: 256 }");
  }

  void OccaRestrictionOperator::Mult(const OccaVector &x, OccaVector &y) const {
    multOp(entries, trueIndices, x, y);
  }

  void OccaRestrictionOperator::MultTranspose(const OccaVector &x, OccaVector &y) const {
    y = 0;
    multTransposeOp(entries, trueIndices, x, y);
  }

  OccaProlongationOperator::OccaProlongationOperator(OccaSparseMatrix &multOp_,
                                                     OccaSparseMatrix &multTransposeOp_) :
    Operator(multOp_.Height(), multOp_.Width()),
    pmat(NULL),
    multOp(multOp_),
    multTransposeOp(multTransposeOp_) {}

  OccaProlongationOperator::OccaProlongationOperator(const Operator *pmat_) :
    Operator(pmat_->Height(), pmat_->Width()),
    pmat(pmat_) {}

  void OccaProlongationOperator::Mult(const OccaVector &x, OccaVector &y) const {
    if (pmat) {
      OccaMult(*pmat, x, y);
    } else {
      multOp.Mult(x, y);
    }
  }

  void OccaProlongationOperator::MultTranspose(const OccaVector &x, OccaVector &y) const {
    if (pmat) {
      OccaMultTranspose(*pmat, x, y);
    } else {
      multTransposeOp.Mult(x, y);
    }
  }
}

#endif
