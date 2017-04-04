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

#include "../general/outils.hpp"
#include "osolvers.hpp"

namespace mfem {
  OccaSolverWrapper::OccaSolverWrapper(Solver &s) :
    sol(s),
    hostX(&GetOccaHostVector(0, s.Width())),
    hostY(&GetOccaHostVector(1, s.Height())) {}

  void OccaSolverWrapper::Mult(const OccaVector &x, OccaVector &y) const {
    x.CopyTo(*hostX);
    sol.Mult(*hostX, *hostY);
    y.CopyFrom(*hostY);
  }

  void OccaSolverWrapper::MultTranspose(const OccaVector &x, OccaVector &y) const {
    x.CopyTo(*hostY);
    sol.MultTranspose(*hostY, *hostX);
    y.CopyFrom(*hostX);
  }
}

#endif
