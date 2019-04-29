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
#include "cuda.hpp"
#include "occa.hpp"

#include <string>
#include <map>

namespace mfem
{

// Place the following variables in the mfem::internal namespace, so that they
// will not be included in the doxygen documentation.
namespace internal
{

// Backends listed by priority, high to low:
static const Backend::Id backend_list[Backend::NUM_BACKENDS] =
{
   Backend::OCCA_CUDA, Backend::RAJA_CUDA, Backend::CUDA, Backend::CUDA_UVM,
   Backend::OCCA_OMP, Backend::RAJA_OMP, Backend::OMP,
   Backend::OCCA_CPU, Backend::RAJA_CPU, Backend::DEBUG,
   Backend::CPU
};

// Backend names listed by priority, high to low:
static const char *backend_name[Backend::NUM_BACKENDS] =
{
   "occa-cuda", "raja-cuda", "cuda", "cuda-uvm",
   "occa-omp", "raja-omp", "omp",
   "occa-cpu", "raja-cpu", "debug", "cpu"
};

} // namespace mfem::internal

void Device::Configure(const std::string &device, const int dev)
{
   std::map<std::string, Backend::Id> bmap;
   for (int i = 0; i < Backend::NUM_BACKENDS; i++)
   {
      bmap[internal::backend_name[i]] = internal::backend_list[i];
   }
   std::string::size_type beg = 0, end;
   while (1)
   {
      end = device.find(',', beg);
      end = (end != std::string::npos) ? end : device.size();
      const std::string bname = device.substr(beg, end - beg);
      std::map<std::string, Backend::Id>::iterator it = bmap.find(bname);
      MFEM_VERIFY(it != bmap.end(), "invalid backend name: '" << bname << '\'');
      Get().MarkBackend(it->second);
      if (end == device.size()) { break; }
      beg = end + 1;
   }

   // OCCA_CUDA needs CUDA or RAJA_CUDA:
   Get().allowed_backends = Get().backends;
   if (Allows(Backend::OCCA_CUDA) && !Allows(Backend::RAJA_CUDA))
   {
      Get().MarkBackend(Backend::CUDA);
   }

   // Activate all backends for Setup().
   Get().allowed_backends = Get().backends;
   Get().Setup(dev);

   // Enable only the default host CPU backend.
   Get().allowed_backends = Backend::CPU;
}

void Device::Print(std::ostream &out)
{
   out << "Device configuration: ";
   bool add_comma = false;
   for (int i = 0; i < Backend::NUM_BACKENDS; i++)
   {
      if (Get().backends & internal::backend_list[i])
      {
         if (add_comma) { out << ','; }
         add_comma = true;
         out << internal::backend_name[i];
      }
   }
   out << '\n';
}

void Device::Setup(const int device)
{
   MFEM_VERIFY(ngpu == -1, "the mfem::Device is already configured!");

   ngpu = 0;
   dev = device;

#ifndef MFEM_USE_CUDA
   MFEM_VERIFY(!Allows(Backend::CUDA_MASK),
               "the CUDA backends require MFEM built with MFEM_USE_CUDA=YES");
#endif
#ifndef MFEM_USE_RAJA
   MFEM_VERIFY(!Allows(Backend::RAJA_MASK),
               "the RAJA backends require MFEM built with MFEM_USE_RAJA=YES");
#endif
#ifndef MFEM_USE_OPENMP
   MFEM_VERIFY(!Allows(Backend::OMP|Backend::RAJA_OMP),
               "the OpenMP and RAJA OpenMP backends require MFEM built with"
               " MFEM_USE_OPENMP=YES");
#endif
   // The check for MFEM_USE_OCCA is in the function OccaDeviceSetup().

   // Device backends setup
   if (Allows(Backend::CUDA_MASK)) { CudaDeviceSetup(dev, ngpu); }
   if (Allows(Backend::OCCA_MASK)) { OccaDeviceSetup(dev); }
   if (Allows(Backend::DEBUG)) { ngpu = 1; }

   // Memory backends setup
   if (Allows(Backend::CUDA_MASK))
   {
      if (Allows(Backend::CUDA_UVM)) { mm.SetMemFeature(MemorySpace::UNIFIED);  }
      else                           { mm.SetMemFeature(MemorySpace::CUDA); }
   }
   if (Allows(Backend::DEBUG)) { mm.SetMemFeature(MemorySpace::PROTECTED); }
}

} // mfem
