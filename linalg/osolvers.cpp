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
    sol(s) {}

  void OccaSolverWrapper::Mult(const OccaVector &x, OccaVector &y) const {
    OccaMult(sol, x, y);
  }

  void OccaSolverWrapper::MultTranspose(const OccaVector &x, OccaVector &y) const {
    OccaMultTranspose(sol, x, y);
  }
}

#endif
