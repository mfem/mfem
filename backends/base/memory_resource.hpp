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

#ifndef MFEM_BACKENDS_BASE_MEMORY_RESOURCE_HPP
#define MFEM_BACKENDS_BASE_MEMORY_RESOURCE_HPP

#include "../../config/config.hpp"
#ifdef MFEM_USE_BACKENDS

#include <cstddef>

namespace mfem
{

/// Polymorphic memory resource. Similar to C++17's std::pmr::memory_resource.
class MemoryResource
{
protected:
   virtual void *DoAllocate(std::size_t bytes, std::size_t alignment) = 0;
   virtual void DoDeallocate(void* p, std::size_t bytes,
                             std::size_t alignment) = 0;

public:
   // Implicitly defined default & copy constructors

   /// Virtual destructor.
   virtual ~MemoryResource() { }

   /// If alignment == 0, use default alignment.
   void *Allocate(std::size_t bytes, std::size_t alignment = 0)
   { return DoAllocate(bytes, alignment); }

   /// If alignment == 0, use default alignment.
   void Deallocate(void *p, std::size_t bytes, std::size_t alignment = 0)
   { DoDeallocate(p, bytes, alignment); }
};


/** @brief Dynamic host memory resource using operator new[](std::size_t) for
    allocation and operator delete[](void*) for deallocation. */
class NewDeleteMemoryResource : public MemoryResource
{
protected:
   virtual void *DoAllocate(std::size_t bytes, std::size_t alignment);
   virtual void DoDeallocate(void *p, std::size_t bytes, std::size_t alignment);
};


/** @brief Dynamic host memory resource using posix_memalign() for aligned
    allocation and free() for deallocation. */
class AlignedMemoryResource : public MemoryResource
{
protected:
   virtual void *DoAllocate(std::size_t bytes, std::size_t alignment);
   virtual void DoDeallocate(void *p, std::size_t bytes, std::size_t alignment);
};

} // namespace mfem

#endif // MFEM_USE_BACKENDS

#endif // MFEM_BACKENDS_BASE_MEMORY_RESOURCE_HPP
