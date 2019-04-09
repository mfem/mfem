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

// This variable is defined in device.cpp:
namespace internal { extern OccaDevice occaDevice; }

static OccaMemory OccaWrapMemory(const OccaDevice dev, const void *d_adrs,
                                 const size_t bytes)
{
   // This function is called when an OCCA kernel is going to be used.
#if defined(MFEM_USE_OCCA) && defined(MFEM_USE_CUDA)
   void *adrs = const_cast<void*>(d_adrs);
   // OCCA & UsingCuda => occa::cuda
   if (Device::Allows(Backend::OCCA_CUDA))
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
   // This function is called when 'ptr' needs to be passed to and OCCA kernel.
   OccaDevice dev = internal::occaDevice;
   if (!mm::UsingMM()) { return OccaWrapMemory(dev, ptr, 0); }
   const bool known = mm::known(ptr);
   if (!known) { mfem_error("OccaPtr: Unknown address!"); }
   mm::memory &base = mm::mem(ptr);
   const bool ptr_on_host = base.host;
   const size_t bytes = base.bytes;
   const bool run_on_host = Device::IsDisabled();
   // If the priority of a host OCCA backend is higher than all device OCCA
   // backends, then we will need to run-on-host even if Device::IsEnabled().
   if (ptr_on_host && run_on_host) { return OccaWrapMemory(dev, ptr, bytes); }
   if (run_on_host) { mfem_error("OccaPtr: !ptr_on_host && run_on_host"); }
   if (!base.d_ptr)
   {
      CuMemAlloc(&base.d_ptr, bytes);
      CuMemcpyHtoD(base.d_ptr, ptr, bytes);
      base.host = false;
   }
   return OccaWrapMemory(dev, base.d_ptr, bytes);
}

OccaDevice OccaDev() { return internal::occaDevice; }

OccaDevice OccaWrapDevice(CUdevice dev, CUcontext ctx)
{
#if defined(MFEM_USE_OCCA) && defined(MFEM_USE_CUDA)
   return occa::cuda::wrapDevice(dev, ctx);
#else
   return 0;
#endif
}

} // namespace mfem
