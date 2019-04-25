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

#include "occa.hpp"
#include "device.hpp"
#include "mem_manager.hpp"

namespace mfem
{

namespace internal { occa::device occaDevice; }

static occa::memory OccaWrapMemory(const occa::device dev, const void *d_adrs,
                                   const size_t bytes)
{
   // This function is called when an OCCA kernel is going to be used.
#ifdef MFEM_USE_OCCA
   void *adrs = const_cast<void*>(d_adrs);
#if defined(MFEM_USE_CUDA) && OCCA_CUDA_ENABLED
   // If OCCA_CUDA is allowed, it will be used since it has the highest priority
   const bool cuda = Device::Allows(Backend::OCCA_CUDA);
   if (cuda) { return occa::cuda::wrapMemory(dev, adrs, bytes); }
#endif // MFEM_USE_CUDA && OCCA_CUDA_ENABLED
   // otherwise, fallback to occa::cpu address space
   return occa::cpu::wrapMemory(dev, adrs, bytes);
#else // MFEM_USE_OCCA
   return occa::memory();
#endif
}

occa::memory OccaPtr(const void *ptr)
{
   // This function is called when 'ptr' needs to be passed to an OCCA kernel.
   occa::device dev = internal::occaDevice;
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

occa::kernel OccaBuildKernel(const char *file,
                             const char *name,
                             const occa::properties props){
#ifdef MFEM_USE_OCCA
   return internal::occaDevice.buildKernel(file, name, props);
#else // MFEM_USE_OCCA
   return occa::kernel();
#endif
}

void OccaDeviceSetup(const int dev)
{
#ifdef MFEM_USE_OCCA
   const int cpu  = Device::Allows(Backend::OCCA_CPU);
   const int omp  = Device::Allows(Backend::OCCA_OMP);
   const int cuda = Device::Allows(Backend::OCCA_CUDA);
   if (cpu + omp + cuda > 1)
   {
      MFEM_ABORT("Only one OCCA backend can be configured at a time!");
   }
   if (cuda)
   {
#if OCCA_CUDA_ENABLED
      std::string mode("mode: 'CUDA', device_id : ");
      internal::occaDevice.setup(mode.append(1,'0'+dev));
#else
      MFEM_ABORT("the OCCA CUDA backend requires OCCA built with CUDA!");
#endif
   }
   else if (omp)
   {
#if OCCA_OPENMP_ENABLED
      internal::occaDevice.setup("mode: 'OpenMP'");
#else
      MFEM_ABORT("the OCCA OpenMP backend requires OCCA built with OpenMP!");
#endif
   }
   else
   {
      internal::occaDevice.setup("mode: 'Serial'");
   }

   std::string mfemDir;
   if (occa::io::exists(MFEM_INSTALL_DIR "/include/mfem/"))
   {
      mfemDir = MFEM_INSTALL_DIR "/include/mfem/";
   }
   else if (occa::io::exists(MFEM_SOURCE_DIR))
   {
      mfemDir = MFEM_SOURCE_DIR;
   }
   else
   {
      MFEM_ABORT("Cannot find OCCA kernels in MFEM_INSTALL_DIR or MFEM_SOURCE_DIR");
   }

   occa::io::addLibraryPath("mfem", mfemDir);
   occa::loadKernels("mfem");
#else
   MFEM_ABORT("the OCCA backends require MFEM built with MFEM_USE_OCCA=YES");
#endif
}

} // namespace mfem
