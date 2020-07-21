// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SIMD_X86_HPP
#define MFEM_SIMD_X86_HPP

#include "../../config/tconfig.hpp"

#ifndef MFEM_USE_CUDA

#if defined(__x86_64__)
#include <x86intrin.h>
#else // assuming MSVC with _M_X64 or _M_IX86
#include <intrin.h>
#endif

#include "m128.hpp"

#include "m256.hpp"

#include "m512.hpp"

#endif // MFEM_USE_CUDA

#endif // MFEM_SIMD_X86_HPP
