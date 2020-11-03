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

#ifndef MFEM_BACKENDS_HPP
#define MFEM_BACKENDS_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif
#include "cuda.hpp"

#ifdef MFEM_USE_HIP
#include <hip/hip_runtime.h>
#endif
#include "hip.hpp"

#ifdef MFEM_USE_OCCA
#include "occa.hpp"
#endif

#ifdef MFEM_USE_RAJA
#include "RAJA/RAJA.hpp"
#if defined(RAJA_ENABLE_CUDA) && !defined(MFEM_USE_CUDA)
#error When RAJA is built with CUDA, MFEM_USE_CUDA=YES is required
#endif
#endif

#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
#define MFEM_DEVICE
#define MFEM_LAMBDA
#define MFEM_HOST_DEVICE
// MFEM_DEVICE_SYNC is made available for debugging purposes
#define MFEM_DEVICE_SYNC
// MFEM_STREAM_SYNC is used for UVM and MPI GPU-Aware kernels
#define MFEM_STREAM_SYNC
#endif

#if !((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
      (defined(MFEM_USE_HIP)  && defined(__HIP_DEVICE_COMPILE__)))
#define MFEM_SHARED
#define MFEM_SYNC_THREAD
#define MFEM_THREAD_ID(k) 0
#define MFEM_THREAD_SIZE(k) 1
#define MFEM_FOREACH_THREAD(i,k,N) for(int i=0; i<N; i++)
#endif

template <typename T>
MFEM_HOST_DEVICE T AtomicAdd(T &add, const T val)
{
#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
     (defined(MFEM_USE_HIP)  && defined(__HIP_DEVICE_COMPILE__)))
   return atomicAdd(&add,val);
#else
   T old = add;
   add += val;
   return old;
#endif
}

#endif // MFEM_BACKENDS_HPP
