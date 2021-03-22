// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CEEDSOLVERS_QCOARSEN_H
#define MFEM_CEEDSOLVERS_QCOARSEN_H

#include "../../config/config.hpp"

#ifdef MFEM_USE_CEED

#include <ceed.h>

namespace mfem
{

namespace ceed
{

int CeedOperatorQCoarsen(CeedOperator oper, int qorder_reduction,
                         CeedOperator* out, CeedVector* coarse_assembledqf,
                         CeedQFunctionContext* context_ptr,
                         CeedQuadMode fine_qmode, CeedQuadMode coarse_qmode);

/** @brief Return some information about the D operator numerically.

    This can be used to guide a very rough adaptive p-coarsening
    procedure. At the moment this is very ad-hoc. */
int CeedOperatorGetHeuristics(CeedOperator oper, CeedScalar* minq,
                              CeedScalar* maxq, CeedScalar* absmin);

}

}

#endif

#endif
