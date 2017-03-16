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
#  ifndef MFEM_OCCAINTERPOLATIONFORM
#  define MFEM_OCCAINTERPOLATIONFORM

#include "../linalg/operator.hpp"
#include "bilinearform.hpp"

#include "occa.hpp"

namespace mfem {
  void CreateRAPOperators(occa::device device,
                          const Operator *R, const Operator *P,
                          Operator *&OccaR, Operator *&OccaP,
                          const occa::properties &props = occa::properties());

  class OccaRestrictionOperator : public Operator {
  public:
    occa::memory childIndices;
    occa::kernel zeroOutChildrenKernel;

    OccaRestrictionOperator(occa::device device,
                            const int height, const int width,
                            occa::memory childIndices_,
                            const occa::properties &props = occa::properties());

    virtual void Mult(const OccaVector &x, OccaVector &y) const;
  };

  class OccaProlongationOperator : public Operator {
  public:
    occa::memory childOffsets, childIndices, childWeights;
    occa::memory parentOffsets, parentIndices, parentWeights;
    occa::kernel updateKernel;

    OccaProlongationOperator(occa::device device,
                             const int height, const int width,
                             occa::memory childOffsets_,
                             occa::memory childIndices_,
                             occa::memory childWeights_,
                             occa::memory parentOffsets_,
                             occa::memory parentIndices_,
                             occa::memory parentWeights_,
                             const occa::properties &props = occa::properties());

    virtual void Mult(const OccaVector &x, OccaVector &y) const;

    virtual void MultTranspose(const OccaVector &x, OccaVector &y) const;
  };

  /// The operator x -> R*A*P*x.
  class OccaRAPOperator : public Operator
  {
  private:
    const Operator &Rt, &A, &P;
    mutable OccaVector Px, APx;

  public:
    /// Construct the RAP operator given R^T, A and P.
    inline OccaRAPOperator(occa::device device,
                           const Operator &Rt_, const Operator &A_, const Operator &P_)
      : Operator(Rt_.Width(), P_.Width()),
        Rt(Rt_), A(A_), P(P_),
        Px(device, P.Height()), APx(device, A.Height()) { }

    /// Operator application.
    inline virtual void Mult(const OccaVector & x, OccaVector & y) const {
      P.Mult(x, Px);
      A.Mult(Px, APx);
      Rt.MultTranspose(APx, y);
    }
  };
}

#  endif
#endif
