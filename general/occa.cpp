// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
   return internal::occaDevice.wrapMemory(ptr, bytes);
}

} // namespace mfem

#endif // MFEM_USE_OCCA
