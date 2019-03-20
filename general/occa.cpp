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

#include "okina.hpp"

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
OccaDevice OccaWrapDevice(CUdevice dev, CUcontext ctx)
{
#if defined(__OCCA__) && defined(__NVCC__)
   return occa::cuda::wrapDevice(dev, ctx);
#else
   return 0;
#endif
}

// *****************************************************************************
OccaMemory OccaDeviceMalloc(OccaDevice device, const size_t bytes)
{
#ifdef __OCCA__
   return device.malloc(bytes);
#else
   return (void*)NULL;
#endif
}

// *****************************************************************************
OccaMemory OccaWrapMemory(const OccaDevice device,
                          void *d_adrs,
                          const size_t bytes)
{
   // OCCA + NVCC will use CUDA
#if defined(__OCCA__) && defined(__NVCC__)
   return occa::cuda::wrapMemory(device, d_adrs, bytes);
#else // Just OCCA uses CPU
#if defined(__OCCA__)
   return occa::cpu::wrapMemory(device, d_adrs, bytes);
#else
   return (void*)NULL;
#endif
#endif
}

// *****************************************************************************
void *OccaMemoryPtr(OccaMemory o_adrs)
{
#ifdef __OCCA__
   return o_adrs.ptr();
#else
   return (void*)NULL;
#endif
}

// *****************************************************************************
void OccaCopyFrom(OccaMemory o_adrs, const void *h_adrs)
{
#ifdef __OCCA__
   o_adrs.copyFrom(h_adrs);
#endif
}

// *****************************************************************************
void OccaCopyTo(OccaMemory o_adrs, void *h_adrs)
{
#ifdef __OCCA__
   o_adrs.copyTo(h_adrs);
#endif
}

// *****************************************************************************
} // namespace mfem
