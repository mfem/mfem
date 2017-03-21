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
#  ifndef MFEM_OCCA_INTERPOLATION
#  define MFEM_OCCA_INTERPOLATION

#include "../linalg/osparsemat.hpp"

#include "occa.hpp"

namespace mfem {
  // [MISSING] Proper destructors
  void CreateRPOperators(occa::device device,
                         const Operator *R, const Operator *P,
                         Operator *&OccaR, Operator *&OccaP);

  class OccaRestrictionOperator : public Operator {
  protected:
    int entries;
    occa::memory trueIndices;
    occa::kernel mapKernel;

  public:
    OccaRestrictionOperator(occa::device device,
                            occa::memory indices,
                            const int width_);

    virtual void Mult(const OccaVector &x, OccaVector &y) const;
  };

  class OccaProlongationOperator : public Operator {
  protected:
    const Operator *pmat;
    mutable Vector hostX, hostY;
    OccaSparseMatrix multOp, multTransposeOp;

  public:
    OccaProlongationOperator(OccaSparseMatrix &multOp_,
                             OccaSparseMatrix &multTransposeOp_);

    OccaProlongationOperator(const Operator *pmat_);

    virtual void Mult(const OccaVector &x, OccaVector &y) const;
    virtual void MultTranspose(const OccaVector &x, OccaVector &y) const;
  };
}

#  endif
#endif
