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

/** Shared (DIM, D1D, NQ) specialization keys for simplex MMA mass, diffusion,
    and DomainLF. Expand with a macro M(DIM,D1D,NQ). Operator-specific extras
    remain in each RegisterSimplexMmaKernels(). */

#ifndef MFEM_SIMPLEX_MMA_FOR_EACH_COMMON_KEY
#define MFEM_SIMPLEX_MMA_FOR_EACH_COMMON_KEY(M) \
   /* 2D */ \
   M(2,2,3)  \
   M(2,2,12) \
   M(2,3,6)  \
   M(2,3,15) /* GLL BP1/BP3 tri p=2 */ \
   M(2,3,16) \
   M(2,4,12) \
   M(2,4,19) /* GLL BP1/BP3 tri p=3 */ \
   M(2,4,25) \
   M(2,5,16) \
   M(2,5,28) /* GLL BP1/BP3 tri p=4 */ \
   M(2,5,33) \
   M(2,6,25) \
   M(2,6,37) /* GLL BP1/BP3 tri p=5 */ \
   M(2,6,42) \
   M(2,7,33) \
   M(2,7,49) /* GLL BP1/BP3 tri p=6 */ \
   M(2,7,55) \
   M(2,8,42) /* BP5tri p=7 / BP1 RHS */ \
   M(2,8,60) /* GLL BP1/BP3 tri p=7 */ \
   /* 3D */ \
   M(3,2,4)   \
   M(3,2,24)  \
   M(3,3,14)  \
   M(3,3,35)  \
   M(3,3,46)  \
   M(3,4,24)  \
   M(3,4,81)  \
   M(3,5,46)  \
   M(3,5,96)  \
   M(3,5,123) \
   M(3,6,81)  \
   M(3,6,175) \
   M(3,7,123) \
   M(3,7,209) \
   M(3,7,248) \
   M(3,8,175) \
   M(3,8,284)
#endif
