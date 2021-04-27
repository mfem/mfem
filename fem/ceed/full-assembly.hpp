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

#ifndef MFEM_CEED_ASSEMBLE_HPP
#define MFEM_CEED_ASSEMBLE_HPP

#include "../../config/config.hpp"

#ifdef MFEM_USE_CEED
#include <ceed.h>
#include "../../linalg/sparsemat.hpp"

namespace mfem
{

namespace ceed
{

/** @brief Assembles a CeedOperator as an mfem::SparseMatrix

    In parallel, this assembles independently on each processor, that is, it
    assembles at the L-vector level. The assembly procedure is always performed
    on the host, but this works also for operators stored on device by copying
    memory. */
int CeedOperatorFullAssemble(CeedOperator op, SparseMatrix **mat);

} // namespace ceed

} // namespace mfem

#endif

#endif
