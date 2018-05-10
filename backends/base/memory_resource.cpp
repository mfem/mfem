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
#ifdef MFEM_USE_BACKENDS

#include "memory_resource.hpp"
#include "../../general/error.hpp"

#include <cstdlib>
#include <cstring>
#include <cerrno>

namespace mfem
{

void *NewDeleteMemoryResource::DoAllocate(std::size_t bytes,
                                          std::size_t alignment)
{
   void *p = ::operator new[](bytes);
   MFEM_VERIFY(!alignment || (std::size_t)(p) % alignment == 0,
               "invalid alignment");
   return p;
}

void NewDeleteMemoryResource::DoDeallocate(void *p, std::size_t bytes,
                                           std::size_t alignment)
{
   ::operator delete[](p);
}


void *AlignedMemoryResource::DoAllocate(std::size_t bytes,
                                        std::size_t alignment)
{
   void *p;
   if (!alignment) { alignment = sizeof(long double); }
   MFEM_VERIFY(posix_memalign(&p, alignment, bytes) == 0,
               "error in posix_memalign(): " << strerror(errno));
   return p;
}

void AlignedMemoryResource::DoDeallocate(void *p, std::size_t bytes,
                                         std::size_t alignment)
{
   free(p);
}

} // namespace mfem

#endif // MFEM_USE_BACKENDS
