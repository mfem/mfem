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
using std::size_t;

#include <list>
#include <unordered_map>

#include "occa.hpp" // for OccaMemory


namespace mfem
{

// *****************************************************************************
/*#define ALIGNMENT 32
struct alignas(ALIGNMENT) MMNew {
   static_assert(ALIGNMENT > 0, "ALIGNMENT must be positive");
   static_assert((ALIGNMENT & (ALIGNMENT - 1)) == 0,
                 "ALIGNMENT must be a power of 2");
   static_assert((ALIGNMENT % sizeof(void*)) == 0,
                 "ALIGNMENT must be a multiple of sizeof(void *)");
   static void *operator new(size_t count) { return Allocate(count); }
   static void *operator new[](size_t count) { return Allocate(count); }
   static void operator delete(void* ptr) { free(ptr); }
   static void operator delete[](void* ptr) { free(ptr); }
 private:
   static void* Allocate(size_t count);
   };*/


// *****************************************************************************
// * Memory Manager Singleton
// *****************************************************************************
class mm
{

public:
   // **************************************************************************
   struct alias;
   
   // TODO: Change this to ptr
   struct memory
   {
      bool host;
      const size_t bytes;

      void *const h_ptr;
      void *d_ptr;
      OccaMemory o_ptr;

      std::list<const alias *> aliases;

      memory(void* const h, const size_t b):
         host(true), bytes(b), h_ptr(h), d_ptr(NULL), aliases() {}
   };

   struct alias
   {
      memory *const mem;
      const size_t offset;
   };

   // **************************************************************************
   typedef std::unordered_map<const void*, memory> memory_map;
   typedef std::unordered_map<const void*, const alias*> alias_map;

   struct ledger
   {
      memory_map memories;
      alias_map aliases;
   };

   // **************************************************************************
   // * Main malloc template function
   // * Allocates n*size bytes and returns a pointer to the allocated memory
   // **************************************************************************
   template<class T>
   static inline T* malloc(const size_t n, 
                           char const* const filename = __FILE__,
                           int const lineno = __LINE__,
                           char const* const function = __FUNCTION__,
                           size_t const size = sizeof(T))
   { return (T*) MM().Insert(::new T[n], n*size, filename, lineno, function); }
//#define mm_malloc(T,n) mm::malloc<T>(n,__FILE__,__LINE__,__FUNCTION__)
#define mm_malloc(T,n) mm::malloc<T>(n,_F_L_F_)
   
   // **************************************************************************
   // * Frees the memory space pointed to by ptr, which must have been
   // * returned by a previous call to mm::malloc
   // **************************************************************************
   template<class T>
   static inline void free(void *ptr,
                           char const* const filename = __FILE__,
                           int const lineno = __LINE__,
                           char const* const function = __FUNCTION__)
   {
      //if (!ptr) { return; }
      mm::MM().Erase(ptr, filename, lineno, function);
      ::delete[] static_cast<T*>(ptr);
   }
#define mm_free(T,p) mm::free<T>(p,_F_L_F_)

   
   // **************************************************************************
   // * Translates ptr to host or device address,
   // * depending on config::Cuda() and the ptr' state
   // **************************************************************************
   static inline void* ptr(void *a) { return MM().Ptr(a); }
   static inline const void* ptr(const void *a) { return MM().Ptr(a); }
   static inline OccaMemory occaPtr(const void *a) { return MM().Memory(a); }

   // **************************************************************************
   static inline void push(const void *ptr, const size_t bytes = 0)
   {
      return MM().Push(ptr, bytes);
   }

   // **************************************************************************
   static inline void pull(const void *ptr, const size_t bytes = 0)
   {
      return MM().Pull(ptr, bytes);
   }

   // **************************************************************************
   static void* memcpy(void *dst, const void *src,
                       size_t bytes, const bool async = false);

private:
   ledger maps;
   mm() {}
   mm(mm const&) = delete;
   void operator=(mm const&) = delete;
   static inline mm& MM() { static mm *singleton = ::new mm(); return *singleton; }

   // **************************************************************************
   void *Insert(void *ptr, const size_t bytes,
                char const* const, int const, char const* const);
   void *Erase(void *ptr, char const* const, int const, char const* const);
   void* Ptr(void *ptr);
   const void* Ptr(const void *ptr);
   OccaMemory Memory(const void *ptr);
   
   // **************************************************************************
   void Push(const void *ptr, const size_t bytes = 0);
   void Pull(const void *ptr, const size_t bytes = 0);
};

} // namespace mfem

#endif
