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

#ifndef MFEM_MM
#define MFEM_MM

#include <new>
#include <list>
#include <cstddef>
#include <unordered_map>

#include "../config/config.hpp"
#include "occa.hpp"
#include "config.hpp"
#include "umpire.hpp"

namespace mfem
{

// *****************************************************************************
// * The default memory manager implementation
// *****************************************************************************
class DefaultMemoryManager
{
public:
   struct alias;
   struct memory
   {
      bool host;
      const std::size_t bytes;
      void *const h_ptr;
      void *d_ptr;
      OccaMemory o_ptr;
      std::list<const alias *> aliases;
      memory(void* const h, const std::size_t b):
         host(true), bytes(b), h_ptr(h), d_ptr(NULL), aliases() {}
   };
   struct alias
   {
      memory *const mem;
      const std::size_t offset;
   };
   typedef std::unordered_map<const void*, memory> memory_map;
   typedef std::unordered_map<const void*, const alias*> alias_map;
   struct ledger
   {
      memory_map memories;
      alias_map aliases;
   };

   // Allocate a host pointer and add it to the registry
   template<typename T>
   inline T* allocate(const std::size_t n)
   {
      return ::new T[n];
   }

   // Deallocate a pointer and remove it and all device allocations from the registry
   template<typename T>
   inline void deallocate(T *ptr)
   {
      ::delete[] ptr;
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

   // Tells if the host address is known by the memory manager
   bool Known(const void *a);

   // Tells if the host address is a known alias by the memory manager
   bool Alias(const void *ptr);

   // Default constructor
   DefaultMemoryManager() = default;

private:
   ledger maps;
};

// *****************************************************************************
// * Define the type of memory manager
// * NOTE This is needed because malloc and free are templated on the
// * type throughout okina, and virtual templated methods are forbidden.
// *****************************************************************************
#if defined(MFEM_USE_UMPIRE)
using MemoryManager = UmpireMemoryManager;
#else
using MemoryManager = DefaultMemoryManager;
#endif

// *****************************************************************************
// * This namespace defines the interface functions that the rest of MFEM uses
// *****************************************************************************
namespace mm
{

// Get the memory manger instance
MemoryManager& getInstance();

// Allocate memory
template<class T>
T* malloc(const std::size_t n, const std::size_t size = sizeof(T))
{
   T* ptr = getInstance().allocate<T>(n);
   if (config::usingMM())
   {
      getInstance().insertAddress(ptr, n*size);
   }
   return ptr;
}

// Deallocate memory
template<class T>
void free(void *ptr)
{
   if (ptr != nullptr)
   {
      if (config::usingMM())
      {
         getInstance().removeAddress(ptr);
      }
      getInstance().deallocate(static_cast<T*>(ptr));
   }
}

// Get the device pointer
void* ptr(void *a);

// Get the device pointer (const version)
const void* ptr(const void *a);

// Tells if the pointer is known by the memory manager
bool Known(const void *a);

// Get the device memory wrapped in an occa::Device
OccaMemory occaPtr(const void *a);

// Push data from the host to device alloc
void push(const void *ptr, const std::size_t bytes = 0);

// Pull data from the device to host alloc
void pull(const void *ptr, const std::size_t bytes = 0);

// Copy data between device addresses
void memcpy(void *dst, const void *src,
            const std::size_t bytes, const bool async = false);

// Tells if the host address is known by the memory manager
bool Known(const void *ptr);

// Tells if the host address is a known alias by the memory manager
bool Alias(const void *ptr);

} // namespace mm

} // namespace mfem

#endif
