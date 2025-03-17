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

#define MFEM_TMOP_ARG_COUNT(...) MFEM_TMOP_ARG_COUNT_N(__VA_ARGS__, 5, 4, 3, 2, 1, 0)
#define MFEM_TMOP_ARG_COUNT_N(_1, _2, _3, _4, _5, N, ...) N

#define MFEM_TMOP_FOREACH_THREAD_N(MFEM_TMOP_FOREACH_THREAD_, N) MFEM_TMOP_FOREACH_THREAD_ ## N
#define MFEM_TMOP_FOREACH_THREAD_(N)  MFEM_TMOP_FOREACH_THREAD_N(MFEM_TMOP_FOREACH_THREAD_, N)
#define MFEM_TMOP_FOREACH_THREAD(...) MFEM_TMOP_FOREACH_THREAD_(ARG_COUNT(__VA_ARGS__))(__VA_ARGS__)

#define MFEM_TMOP_FOREACH_ERROR \
   static_assert(false, "MFEM_TMOP_FOREACH_THREAD requires exactly 2 or 3 arguments");

#define MFEM_TMOP_FOREACH_THREAD_0(...) MFEM_TMOP_FOREACH_ERROR
#define MFEM_TMOP_FOREACH_THREAD_1(...) MFEM_TMOP_FOREACH_ERROR
#define MFEM_TMOP_FOREACH_THREAD_4(...) MFEM_TMOP_FOREACH_ERROR
#define MFEM_TMOP_FOREACH_THREAD_5(...) MFEM_TMOP_FOREACH_ERROR

#define MFEM_TMOP_FOREACH_THREAD_2(i, k) int i = hipThreadIdx_ ##k; if (i<hipBlockDim_ ##k)

#define MFEM_TMOP_FOREACH_THREAD_3(i, k, N) MFEM_FOREACH_THREAD(i, k, N)