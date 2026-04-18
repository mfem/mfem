// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

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
