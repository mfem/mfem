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

#ifdef MFEM_USE_OCCA
#include "device.hpp"

#if defined(MFEM_USE_CUDA) && OCCA_CUDA_ENABLED
#include <occa/modes/cuda/utils.hpp>
#endif

namespace mfem
{

// This variable is defined in device.cpp:
namespace internal { extern occa::device occaDevice; }

occa::device &OccaDev() { return internal::occaDevice; }

occa::memory OccaMemoryWrap(void *ptr, std::size_t bytes)
{
#if defined(MFEM_USE_CUDA) && OCCA_CUDA_ENABLED
   // If OCCA_CUDA is allowed, it will be used since it has the highest priority
   if (Device::Allows(Backend::OCCA_CUDA))
   {
      return occa::cuda::wrapMemory(internal::occaDevice, ptr, bytes);
   }
#endif // MFEM_USE_CUDA && OCCA_CUDA_ENABLED
   // otherwise, fallback to occa::cpu address space
   return occa::cpu::wrapMemory(internal::occaDevice, ptr, bytes);
}

} // namespace mfem

#endif // MFEM_USE_OCCA
