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

#include <cstddef> // for size_t

#include <list>
#include <unordered_map>

#include "occa.hpp" // for OccaMemory

#if defined(MFEM_USE_UMPIRE)
#include "umpire/Umpire.hpp"
#endif

namespace mfem
{

class DefaultMemoryManager
{
public:
   struct alias;

   // TODO: Change this to ptr
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

   // **************************************************************************

   // Allocate a host pointer and add it to the registry
   template<typename T>
   inline T* allocate(const std::size_t n)
   {
      return static_cast<T*>(Insert(::new T[n], n*sizeof(T)));
   }

   // Deallocate a pointer and remove it and all device allocations from the registry
   template<typename T>
   inline void deallocate(T *ptr)
   {
      if (ptr != nullptr) {
         Erase(ptr);
         ::delete[] static_cast<T*>(ptr);
      }
   }

   // For a given host or device pointer, return the device or host corresponding pointer
   // NOTE This may be offset from the original pointer in the registry
   const void* getMatchingPointer(const void *a);
   void* getMatchingPointer(void *a);

   // Get the matching OCCA pointer
   // TODO Remove this method -- wrap the d_ptr instead
   OccaMemory getOccaPointer(const void *a);

   // Given a host pointer, push bytes beginning at address ptr to the device allocation
   // NOTE This may be offset from the original pointer in the registry
   void pushData(const void *ptr, const std::size_t bytes = 0);

   // Given a device pointer, pull size bytes beginning at address ptr to the host allocation
   // NOTE This may be offset from the original pointer in the registry
   void pullData(const void *ptr, const std::size_t bytes = 0);

   // Copies bytes from src to dst, using the registry to determine where src and dst are located.
   // NOTE These may be offset from the original pointers in the registry
   void copyData(void *dst, const void *src, std::size_t bytes, const bool async = false);

   // Default constructor
   DefaultMemoryManager() = default;

private:
   ledger maps;

   void *Insert(void *ptr, const std::size_t bytes);
   void *Erase(void *ptr);
};

#if defined(MFEM_USE_UMPIRE)
class UmpireMemoryManager
{
public:
   // Allocate a host pointer and add it to the registry
   template<typename T>
   inline T* allocate(const std::size_t n)
   {
      void *mem = m_host.allocate(n * sizeof(T));
      Insert(new (mem) T[n]);
   }

   // Deallocate a pointer and remove it and all device allocations from the registry
   template<typename T>
   inline void deallocate(T *ptr)
   {
      // TODO: Missing array placement delete
      m_host.deallocate(ptr);
   }

   // For a given host or device pointer, return the device or host corresponding pointer
   // NOTE This may be offset from the original pointer in the registry
   const void* getMatchingPointer(const void *a);
   void* getMatchingPointer(void *a);

   // Get the matching OCCA pointer
   // TODO Remove this method -- wrap the d_ptr instead
   OccaMemory getOccaPointer(const void *a);

   // Given a host pointer, push bytes beginning at address ptr to the device allocation
   // NOTE This may be offset from the original pointer in the registry
   void pushData(const void *ptr, const std::size_t bytes = 0);

   // Given a device pointer, pull size bytes beginning at address ptr to the host allocation
   // NOTE This may be offset from the original pointer in the registry
   void pullData(const void *ptr, const std::size_t bytes = 0);

   // Copies bytes from src to dst, using the registry to determine where src and dst are located.
   // NOTE These may be offset from the original pointers in the registry
   void copyData(void *dst, const void *src, std::size_t bytes, const bool async = false);

   // Constructor
   UmpireMemoryManager();

private:

   void *Insert(void *ptr, const std::size_t bytes);
   void *Erase(void *ptr);

   std::map< void*, std::vector<void*> >

   umpire::ResourceManager& m_rm;
   umpire::Allocator m_host;
   umpire::Allocator m_device;
};
#endif

#if defined(MFEM_USE_UMPIRE)
using MemoryManager = UmpireMemoryManager;
#else
using MemoryManager = DefaultMemoryManager;
#endif

namespace mm
{
MemoryManager& getInstance();

template<class T>
T* malloc(const std::size_t n, const std::size_t size = sizeof(T)) { return getInstance().allocate<T>(n * size); }

template<class T>
void free(void *ptr) { getInstance().deallocate(static_cast<T*>(ptr)); }

void* ptr(void *a);

const void* ptr(const void *a);

OccaMemory occaPtr(const void *a);

void push(const void *ptr, const std::size_t bytes = 0);

void pull(const void *ptr, const std::size_t bytes = 0);

void memcpy(void *dst, const void *src,
            const std::size_t bytes, const bool async = false);

} // namespace mm

} // namespace mfem

#endif
