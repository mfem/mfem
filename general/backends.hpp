// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BACKENDS_HPP
#define MFEM_BACKENDS_HPP

#include "../config/config.hpp"

#include "backends/openmp.hpp"

#include "backends/raja.hpp"

#ifdef MFEM_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <library_types.h>
#endif
#include "backends/cuda.hpp"

#ifdef MFEM_USE_HIP
#include <hip/hip_runtime.h>
#endif
#include "backends/hip.hpp"

#ifdef MFEM_USE_OCCA
#include "backends/occa.hpp"
#endif

#ifdef MFEM_USE_SYCL
#endif
#include "backends/sycl.hpp"

#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP) || defined(MFEM_USE_SYCL))
#define MFEM_DEVICE
#define MFEM_LAMBDA
// #define MFEM_HOST_DEVICE // defined in config/config.hpp
// MFEM_DEVICE_SYNC is made available for debugging purposes
#define MFEM_DEVICE_SYNC
// MFEM_STREAM_SYNC is used for UVM and MPI GPU-Aware kernels
#define MFEM_STREAM_SYNC
#endif

#if !((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) ||                    \
      (defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)) ||            \
      (defined(MFEM_USE_SYCL) && defined(__SYCL_DEVICE_ONLY__)))

#define MFEM_SHARED

template <typename T, size_t = 0> T mfem_shared() {
  T t;
  return t;
}

#define MFEM_STATIC_SHARED_VAR(var, ...) __VA_ARGS__ var

#define MFEM_DYNAMIC_SHARED_VAR(var, sm, ...)                                  \
  __VA_ARGS__ var;                                                             \
  sm += sizeof(__VA_ARGS__) / sizeof(*sm);

#define MFEM_SYNC_THREAD
#define MFEM_BLOCK_ID(k) 0
#define MFEM_THREAD_ID(k) 0
#define MFEM_THREAD_SIZE(k) 1
#define MFEM_FOREACH_THREAD(i, k, N) for (int i = 0; i < N; i++)
#endif

// 'double' and 'float' atomicAdd implementation for previous versions of CUDA
#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
MFEM_DEVICE inline mfem::real_t atomicAdd(mfem::real_t *add, mfem::real_t val) {
  unsigned long long int *ptr = (unsigned long long int *)add;
  unsigned long long int old = *ptr, reg;
  do {
    reg = old;
    old = atomicCAS(ptr, reg,
#ifdef MFEM_USE_SINGLE
                    __float_as_int(val + __int_as_float(reg)));
#else
                    __double_as_longlong(val + __longlong_as_double(reg)));
#endif
  } while (reg != old);
#ifdef MFEM_USE_SINGLE
  return __int_as_float(old);
#else
  return __longlong_as_double(old);
#endif
}
#endif

template <typename T> MFEM_HOST_DEVICE T AtomicAdd(T &add, const T val) {
#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) ||                     \
     (defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)))
  return atomicAdd(&add, val);
#else
  T old = add;
#ifdef MFEM_USE_OPENMP
#pragma omp atomic
#endif
  add += val;
  return old;
#endif
}

#endif // MFEM_BACKENDS_HPP
