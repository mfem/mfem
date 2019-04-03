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

namespace mfem
{

OccaDevice OccaWrapDevice(CUdevice dev, CUcontext ctx)
{
#if defined(MFEM_USE_OCCA) && defined(MFEM_USE_CUDA)
   return occa::cuda::wrapDevice(dev, ctx);
#else
   return 0;
#endif
}

OccaMemory OccaDeviceMalloc(OccaDevice device, const size_t bytes)
{
#ifdef MFEM_USE_OCCA
   return device.malloc(bytes);
#else
   return (void*)NULL;
#endif
}

OccaMemory OccaWrapMemory(const OccaDevice device, const void *d_adrs,
                          const size_t bytes)
{
#if defined(MFEM_USE_OCCA) && defined(MFEM_USE_CUDA)
   void *adrs = const_cast<void*>(d_adrs);
   // OCCA & UsingCuda => occa::cuda
   if (Device::UsingCuda())
   {
      return occa::cuda::wrapMemory(device, adrs, bytes);
   }
   // otherwise, fallback to occa::cpu address space
   return occa::cpu::wrapMemory(device, adrs, bytes);
#else // MFEM_USE_OCCA && MFEM_USE_CUDA
#if defined(MFEM_USE_OCCA)
   return occa::cpu::wrapMemory(device, const_cast<void*>(d_adrs), bytes);
#else
   return (void*)NULL;
#endif
#endif
}

void *OccaMemoryPtr(OccaMemory o_adrs)
{
#ifdef MFEM_USE_OCCA
   return o_adrs.ptr();
#else
   return (void*)NULL;
#endif
}

void OccaCopyFrom(OccaMemory o_adrs, const void *h_adrs)
{
#ifdef MFEM_USE_OCCA
   o_adrs.copyFrom(h_adrs);
#endif
}

void OccaCopyTo(OccaMemory o_adrs, void *h_adrs)
{
#ifdef MFEM_USE_OCCA
   o_adrs.copyTo(h_adrs);
#endif
}

} // namespace mfem
