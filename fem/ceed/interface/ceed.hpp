// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LIBCEED_CEED
#define MFEM_LIBCEED_CEED

#include "../../../config/config.hpp"

#ifdef MFEM_USE_CEED

#include <ceed.h>

#if !CEED_VERSION_GE(0, 12, 0)
#error MFEM requires a libCEED version >= 0.12.0
#endif

#if !CEED_VERSION_GE(0, 13, 0)
#define CeedOperatorCreateComposite(ceed, op)                                  \
    CeedCompositeOperatorCreate((ceed), (op))
#define CeedOperatorCompositeAddSub(op, sub)                                   \
    CeedCompositeOperatorAddSub((op), (sub))
#define CeedOperatorCompositeGetNumSub(op, num)                                \
    CeedCompositeOperatorGetNumSub((op), (num))
#define CeedOperatorCompositeGetSubList(op, list)                              \
    CeedCompositeOperatorGetSubList((op), (list))
#endif

namespace mfem
{

namespace internal
{

extern Ceed ceed;

} // namespace internal

} // namespace mfem

#endif // MFEM_USE_CEED

#endif // MFEM_LIBCEED_CEED
