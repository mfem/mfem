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

#ifndef MFEM_MM_UMPIRE
#define MFEM_MM_UMPIRE

#if defined(MFEM_USE_UMPIRE)

#include "umpire/Umpire.hpp"
//#include "umpire/Allocator.hpp"

namespace mfem
{

// Umpire memory manager implementation
class UmpireMemoryManager
{
public:
   // Allocate a host pointer and add it to the registry
   template<typename T>
   inline T* allocate(const std::size_t n)
   {
      const std::size_t bytes = n * sizeof(T);
      void *mem = m_host.allocate(bytes);
      //insertAddress(mem, bytes);
      T* objs = new (mem) T[n];
      return objs;
   }

   // Deallocate a pointer and remove it and all device allocations from the registry
   template<typename T>
   inline void deallocate(T *ptr)
   {
      // TODO: Missing array placement delete
      m_host.deallocate(ptr);
   }

   // Register an address
   void insertAddress(void *ptr, const std::size_t bytes);

   // Remove an address
   void removeAddress(void *ptr);

   // For a given host address, return the current mode's address
   // NOTE This may be offset from the original pointer in the registry
   void* getPtr(void *a);

   // Get the matching OCCA pointer
   // TODO Remove this method -- wrap the d_ptr instead
   OccaMemory getOccaPtr(const void *a);

   // Given a host pointer, push bytes beginning at address ptr to the device allocation
   // NOTE This may be offset from the original pointer in the registry
   void pushData(const void *ptr, const std::size_t bytes = 0);

   // Given a device pointer, pull size bytes beginning at address ptr to the host allocation
   // NOTE This may be offset from the original pointer in the registry
   void pullData(const void *ptr, const std::size_t bytes = 0);

   // Copies bytes from src to dst, which are both device addresses
   // NOTE These may be offset from the original pointers in the registry
   void copyData(void *dst, const void *src, std::size_t bytes,
                 const bool async = false);

   // Constructor
   UmpireMemoryManager();

public:
   struct umpire_memory{
      bool host;
      //std::size_t bytes;
      char *d_ptr;
      umpire_memory(char *p/*, std::size_t b*/):host(true), /*bytes(b),*/ d_ptr(p){}
   };
   typedef std::unordered_map< char*, umpire_memory > MapType;

private:
   MapType m_map;

   umpire::ResourceManager& m_rm;
   umpire::Allocator m_host;
   umpire::Allocator m_device;
};

} // namespace mfem

#endif // defined(MFEM_USE_UMPIRE)

#endif // MFEM_MM_UMPIRE
