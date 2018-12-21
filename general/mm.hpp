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

#ifndef MFEM_MM_HPP
#define MFEM_MM_HPP

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
class alias_t;

// *****************************************************************************
class memory_t
{
public:
   bool host;
   const size_t bytes;
   void* const h_adrs;
   void* d_adrs;
   OccaMemory o_adrs;
   std::list<const alias_t*> aliases;
   memory_t(void* const h, const size_t b):
      host(true), bytes(b), h_adrs(h), d_adrs(NULL), aliases() {}
};

// *****************************************************************************
class alias_t
{
public:
   memory_t* const mem;
   const size_t offset;
   alias_t(memory_t* const m, const size_t o):mem(m), offset(o) {}
};

// *****************************************************************************
typedef std::unordered_map<const void*, memory_t*> memory_map_t;
typedef std::unordered_map<const void*, const alias_t*> alias_map_t;
typedef struct { memory_map_t *memories; alias_map_t *aliases; } mm_t;
typedef memory_map_t::iterator mm_iterator_t;

// *****************************************************************************
// * Memory Manager Singleton
// *****************************************************************************
class mm
{
private:
   memory_map_t *memories;
   alias_map_t  *aliases;
   mm_t *maps;
private:
   mm(): memories(new memory_map_t), aliases(new alias_map_t()),
      maps(new mm_t( {memories, aliases})) {}
   mm(mm const&);
   void operator=(mm const&);
   static inline mm& MM() { static mm singleton; return singleton; }
private:
   // **************************************************************************
   void *Insert(void *adrs, const size_t bytes);
   void *Erase(void *adrs);
   void* Adrs(void *adrs);
   const void* Adrs(const void *adrs);
   OccaMemory Memory(const void *adrs);
private:
   // **************************************************************************
   void Push(const void *adrs, const size_t bytes =0);
   void Pull(const void *adrs, const size_t bytes =0);
public:
   // **************************************************************************
   // * Main malloc template function
   // * Allocates n*size bytes and returns a pointer to the allocated memory
   // **************************************************************************
   template<class T>
   static inline T* malloc(const size_t n, const size_t size = sizeof(T))
   { return (T*) MM().Insert(::new T[n], n*size); }

   // **************************************************************************
   // * Frees the memory space pointed to by ptr, which must have been
   // * returned by a previous call to mm::malloc
   // **************************************************************************
   template<class T>
   static inline void free(void *ptr)
   {
      if (!ptr) { return; }
      void *adrs = mm::MM().Erase(ptr);
      ::delete[] static_cast<T*>(adrs);
      adrs = nullptr;
   }

   // **************************************************************************
   // * Translates adrs to host or device address,
   // * depending on config::Cuda() and the adrs' state
   // **************************************************************************
   static inline void* adrs(void *a) { return MM().Adrs(a); }
   static inline const void* adrs(const void *a) { return MM().Adrs(a); }
   static inline OccaMemory memory(const void *a) { return MM().Memory(a); }

   // **************************************************************************
   static inline void push(const void *adrs, const size_t bytes =0)
   {
      return MM().Push(adrs, bytes);
   }

   // **************************************************************************
   static inline void pull(const void *adrs, const size_t bytes =0)
   {
      return MM().Pull(adrs, bytes);
   }

   // **************************************************************************
   static void* memcpy(void *dst, const void *src,
                       size_t bytes, const bool async = false);
};

// *****************************************************************************
} // mfem

#endif // MFEM_MM_HPP
