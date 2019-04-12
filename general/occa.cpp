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

#include "forall.hpp"

namespace mfem
{

// This variable is defined in device.cpp:
namespace internal { extern OccaDevice occaDevice; }

static OccaMemory OccaWrapMemory(const OccaDevice dev, const void *d_adrs,
                                 const size_t bytes)
{
   // This function is called when an OCCA kernel is going to be used.
#ifdef MFEM_USE_OCCA
   void *adrs = const_cast<void*>(d_adrs);
#if defined(MFEM_USE_CUDA) && OCCA_CUDA_ENABLED
   // If OCCA_CUDA is allowed, it will be used since it has the highest priority
   if (Device::Allows(Backend::OCCA_CUDA))
   {
      return occa::cuda::wrapMemory(dev, adrs, bytes);
   }
#endif // MFEM_USE_CUDA && OCCA_CUDA_ENABLED
   // otherwise, fallback to occa::cpu address space
   return occa::cpu::wrapMemory(dev, adrs, bytes);
#else // MFEM_USE_OCCA
   return (void*)NULL;
#endif
}

OccaMemory OccaPtr(const void *ptr)
{
   // This function is called when 'ptr' needs to be passed to an OCCA kernel.
   OccaDevice dev = internal::occaDevice;
   if (!mm.UsingMM()) { return OccaWrapMemory(dev, ptr, 0); }
   const bool known = mm.IsKnown(ptr);
   if (!known) { mfem_error("OccaPtr: Unknown address!"); }
   const bool ptr_on_host = mm.IsOnHost(ptr);
   const size_t bytes = mm.Bytes(ptr);
   const bool run_on_host = !Device::Allows(Backend::DEVICE_MASK);
   // If the priority of a host OCCA backend is higher than all device OCCA
   // backends, then we will need to run-on-host even if the Device allows a
   // device backend.
   if (ptr_on_host && run_on_host) { return OccaWrapMemory(dev, ptr, bytes); }
   if (run_on_host) { mfem_error("OccaPtr: !ptr_on_host && run_on_host"); }
   void *d_ptr = mm.GetDevicePtr(ptr);
   return OccaWrapMemory(dev, d_ptr, bytes);
}

OccaDevice OccaDev() { return internal::occaDevice; }

} // namespace mfem
