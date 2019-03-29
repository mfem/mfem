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

#include <list>
#include <cstddef> // for size_t
#include <unordered_map>

using std::size_t;

#include "occa.hpp" // for OccaMemory

namespace mfem
{

// Implementation of MFEM's lightweight host/device memory manager (mm) designed
// to work seamlessly with the okina device kernel interface.

/// The memory manager singleton
class mm
{
public:
   struct alias;

   // TODO: Change this to ptr
   struct memory
   {
      const size_t bytes;
      void *const h_ptr;
      void *d_ptr;
      OccaMemory o_ptr;
      std::list<const alias *> aliases;
      bool host;
      bool padding[7];
      memory(void* const h, const size_t b):
         bytes(b), h_ptr(h), d_ptr(nullptr), aliases(), host(true) {}
   };

   struct alias
   {
      memory *const mem;
      const long offset;
   };

   typedef std::unordered_map<const void*, memory> memory_map;
   typedef std::unordered_map<const void*, const alias*> alias_map;

   struct ledger
   {
      memory_map memories;
      alias_map aliases;
   };

   /// Main malloc template function. Allocates n*size bytes and returns a
   /// pointer to the allocated memory.
   template<class T>
   static inline T* malloc(const size_t n, const size_t size = sizeof(T))
   { return static_cast<T*>(MM().Insert(::new T[n], n*size)); }

   /// Frees the memory space pointed to by ptr, which must have been returned
   /// by a previous call to mm::malloc.
   template<class T>
   static inline void free(void *ptr)
   {
      if (!ptr) { return; }
      ::delete[] static_cast<T*>(ptr);
      mm::MM().Erase(ptr);
   }


   /// Translates ptr to host or device address, depending on config::Cuda() and
   /// the ptr state.
   static inline void *ptr(void *a) { return MM().Ptr(a); }
   static inline const void* ptr(const void *a) { return MM().Ptr(a); }
   static inline OccaMemory occaPtr(const void *a) { return MM().Memory(a); }

   static inline void push(const void *ptr, const size_t bytes = 0)
   {
      return MM().Push(ptr, bytes);
   }

   static inline void pull(const void *ptr, const size_t bytes = 0)
   {
      return MM().Pull(ptr, bytes);
   }

   /// Data will be pushed/pulled before the copy happens on the H or the D
   static void* memcpy(void *dst, const void *src,
                       size_t bytes, const bool async = false);

   static inline bool known(const void *a)
   {
      return MM().Known(a);
   }

   /// Registers external host pointer in the mm, which will manage
   /// the device pointer.
   template<class T>
   static inline void RegisterHostPtr(T *ptr_host, const size_t size)
   {
      MM().Insert(ptr_host, size*sizeof(T));
#ifdef MFEM_DEBUG
     RegisterCheck(ptr_host);
#endif
   }

   /// Registers external host and device pointers in the mm.
   template<class T>
   static void RegisterHostAndDevicePtr(T *ptr_host, T *ptr_device,
                                        const size_t size, bool host)
   {
      RegisterHostPtr(ptr_host, size);
      mm::memory &base = MM().maps.memories.at(ptr_host);
      base.d_ptr = ptr_device;
      base.host = host;
   }

   /// Unregisters the host pointer from the mm. To be used with
   /// memory not allocated by the mm
   template<class T>
   static inline void UnregisterHostPtr(T *ptr)
   {
      if (!ptr) { return; }
      mm::MM().Erase(ptr);
   }

   /// Check if pointer has been registered in the mm
   static void RegisterCheck(void *ptr);

private:
   ledger maps;
   mm() {}
   mm(mm const&) = delete;
   void operator=(mm const&) = delete;
   static inline mm& MM() { static mm *singleton = new mm(); return *singleton; }

   /// Adds an address
   void *Insert(void *ptr, const size_t bytes);

   /// Remove the address from the map, as well as all the address' aliases
   void *Erase(void *ptr);

   /// Turn an address to the right host or device one
   void *Ptr(void *ptr);
   const void *Ptr(const void *ptr);

   /// Tests if ptr is a known address
   bool Known(const void *ptr);

   /// Tests if ptr is an alias address
   bool Alias(const void *ptr);

   OccaMemory Memory(const void *ptr);

   void Push(const void *ptr, const size_t bytes = 0);
   void Pull(const void *ptr, const size_t bytes = 0);
};

} // namespace mfem

#endif
