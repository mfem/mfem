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

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "memory_resource.hpp"
#include "../../general/error.hpp"

#ifdef MFEM_USE_CUDAUM
#include "cuda_runtime.h"
#include "cuda.h"
#endif

namespace mfem
{

namespace omp
{

#ifdef MFEM_USE_CUDAUM
void *UnifiedMemoryResource::DoAllocate(std::size_t bytes,
					std::size_t alignment)
{
   void *p = NULL;
   if (bytes > 0)
   {
      cudaError_t err = cudaMallocManaged(&p, bytes);
      MFEM_VERIFY(err == CUDA_SUCCESS, "");
   }
   return p;
}

void UnifiedMemoryResource::DoDeallocate(void *p, std::size_t bytes,
					 std::size_t alignment)
{
   if (p != NULL)
   {
      cudaFree(p);
   }
}
#endif


} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)
