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

#if defined(MFEM_USE_CUDA) && defined(__x86_64__) && \
    defined(__GNUC__) && (__GNUC__ == 7) && (__GNUC_MINOR__ == 5) && \
    !defined(__clang__)
#define MFEM_SIMD_X86_SKIP
#endif

#if defined(MFEM_SIMD_X86_SKIP) && defined(MFEM_DEBUG)
#pragma message("warning: Compiler is known to fail with intrinsics: skipping!")
#endif

#if defined(__x86_64__) && !defined(MFEM_SIMD_X86_SKIP)
#warning x86intrin
#include <x86intrin.h>
// Assuming MSVC with _M_X64 or _M_IX86
#elif defined(_MSC_VER)
#include <intrin.h>
#else
#pragma message("warning: Unknown intrinsic header")
#endif

#if !defined(MFEM_SIMD_X86_SKIP)

#include "m128.hpp"

#include "m256.hpp"

#include "m512.hpp"

#endif // !MFEM_SIMD_SKIP

#endif // MFEM_SIMD_X86_HPP
