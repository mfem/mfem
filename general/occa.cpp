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
OccaDevice okWrapDevice(CUdevice dev, CUcontext ctx)
{
#if defined(__NVCC__) and defined(__OCCA__)
   return occa::cuda::wrapDevice(dev, ctx);
#else
   return 0;
#endif
}

// *****************************************************************************
OccaMemory okDeviceMalloc(OccaDevice device, const size_t bytes)
{
#ifdef __OCCA__
   return device.malloc(bytes);
#else
   return (void*)NULL;
#endif
}

// *****************************************************************************
OccaMemory okWrapMemory(const OccaDevice device,
                        void *d_adrs,
                        const size_t bytes)
{
#if defined(__NVCC__) and defined(__OCCA__)
   return occa::cuda::wrapMemory(device, d_adrs, bytes);
#else
   return (void*)NULL;
#endif
}

// *****************************************************************************
void *okMemoryPtr(OccaMemory o_adrs)
{
#ifdef __OCCA__
   return o_adrs.ptr();
#else
   return (void*)NULL;
#endif
}

// *****************************************************************************
void okCopyFrom(OccaMemory o_adrs, const void *h_adrs)
{
#ifdef __OCCA__
   o_adrs.copyFrom(h_adrs);
#endif
}

// *****************************************************************************
void okCopyTo(OccaMemory o_adrs, void *h_adrs)
{
#ifdef __OCCA__
   o_adrs.copyTo(h_adrs);
#endif
}

// *****************************************************************************
} // namespace mfem
