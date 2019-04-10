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

extern OccaDevice occaDevice;

static OccaMemory OccaWrapMemory(const OccaDevice dev, const void *d_adrs,
                                 const size_t bytes)
{
#if defined(MFEM_USE_OCCA) && defined(MFEM_USE_CUDA)
   void *adrs = const_cast<void*>(d_adrs);
   // OCCA & UsingCuda => occa::cuda
   if (Device::UsingCuda())
   {
      return occa::cuda::wrapMemory(dev, adrs, bytes);
   }
   // otherwise, fallback to occa::cpu address space
   return occa::cpu::wrapMemory(dev, adrs, bytes);
#else // MFEM_USE_OCCA && MFEM_USE_CUDA
#ifdef MFEM_USE_OCCA
   return occa::cpu::wrapMemory(dev, const_cast<void*>(d_adrs), bytes);
#else
   return (void*)NULL;
#endif
#endif
}

OccaMemory OccaPtr(const void *ptr)
{
   OccaDevice dev = occaDevice;
   if (!Device::UsingMM()) { return OccaWrapMemory(dev, ptr, 0); }
   const bool known = mm.IsKnown(ptr);
   if (!known) { mfem_error("OccaPtr: Unknown address!"); }
   const bool host = mm.IsOnHost(ptr);
   const size_t bytes = mm.Bytes(ptr);
   const bool gpu = Device::UsingDevice();
   if (host && !gpu) { return OccaWrapMemory(dev, ptr, bytes); }
   if (!gpu) { mfem_error("OccaPtr: !gpu"); }
   void *d_ptr = mm.GetDevicePtr(ptr);
   return OccaWrapMemory(dev, d_ptr, bytes);
}

OccaDevice OccaWrapDevice(CUdevice dev, CUcontext ctx)
{
#if defined(MFEM_USE_OCCA) && defined(MFEM_USE_CUDA)
   return occa::cuda::wrapDevice(dev, ctx);
#else
   return 0;
#endif
}

} // namespace mfem
