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

#ifndef MFEM_MEM_MANAGER
#define MFEM_MEM_MANAGER

#include <list>
#include <cstddef>
#include <unordered_map>
#include <type_traits>

using std::size_t;

namespace mfem
{

// Implementation of MFEM's lightweight host/device memory manager (mm) designed
// to work seamlessly with the okina device kernel interface.

/// The memory manager singleton
class mm
{
public:
   struct alias;

   struct memory
   {
      const size_t bytes;
      void *const h_ptr;
      void *d_ptr;
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
   static inline T *New(const size_t n)
   { return static_cast<T*>(MM().Insert(new T[n], n*sizeof(T))); }

   /// Frees the memory space pointed to by ptr, which must have been returned
   /// by a previous call to mm::New.
   template<class T>
   static inline void Delete(T *ptr)
   {
      static_assert(!std::is_void<T>::value, "Cannot Delete a void pointer. "
                    "Explicitly provide the correct type as a template parameter.");
      if (!ptr) { return; }
      delete [] ptr;
      mm::MM().Erase(ptr);
   }

   /// Translates ptr to host or device address, depending on
   /// Device::UsingDevice() and the ptr state.
   template <class T>
   static inline T *ptr(T *a) { return static_cast<T*>(MM().Ptr(a)); }
   template <class T>
   static inline const T* ptr(const T *a) { return static_cast<const T*>(MM().Ptr(a)); }

   static inline memory &mem(const void *a) { return MM().maps.memories.at(a); }

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

   /// Tests if the pointer has been registered
   static inline bool known(const void *ptr)
   {
      return MM().Known(ptr);
   }

   /// Prints all pointers known by the memory manager
   static inline void PrintPtrs(void)
   {
      for (const auto& n : MM().maps.memories)
      {
         printf("key %p, host %p, device %p \n", n.first, n.second.h_ptr,
                n.second.d_ptr);
      }
   }

   /// Copies all memory to the current memory space
   static inline void GetAll(void)
   {
      for (const auto& n : MM().maps.memories)
      {
         const void *ptr = n.first;
         mm::ptr(ptr);
      }
   }

   /// Registers external host pointer in the mm. The mm will manage the
   /// corresponding device pointer (but not the provided host pointer).
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

   /// Unregisters the host pointer from the mm. To be used with memory not
   /// allocated by the mm.
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

   /// Push the data to the device
   void Push(const void *ptr, const size_t bytes = 0);

   /// Pull the data from the device
   void Pull(const void *ptr, const size_t bytes = 0);
};

} // namespace mfem

#endif // MFEM_MEM_MANAGER
