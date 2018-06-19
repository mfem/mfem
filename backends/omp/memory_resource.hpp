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

#ifndef MFEM_BACKENDS_OMP_MEMORY_RESOURCE_HPP
#define MFEM_BACKENDS_OMP_MEMORY_RESOURCE_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "../../backends/base/memory_resource.hpp"

namespace mfem
{

namespace omp
{

/// Polymorphic memory resource. Similar to C++17's std::pmr::memory_resource.

#ifdef MFEM_USE_CUDAUM
/** @brief Memory resource using unified memory. */
class UnifiedMemoryResource : public MemoryResource
{
protected:
   virtual void *DoAllocate(std::size_t bytes, std::size_t alignment);
   virtual void DoDeallocate(void *p, std::size_t bytes, std::size_t alignment);
};
#endif

} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#endif // MFEM_BACKENDS_OMP_MEMORY_RESOURCE_HPP
