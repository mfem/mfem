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
// * mm2dev_t: memory manager host_@ to device_@
// *****************************************************************************
typedef struct
{
   bool host;
   size_t bytes;
   void *h_adrs;
   void *d_adrs;
   OccaMemory o_adrs;
   std::list<const void*> aliases;
} memory_t;

// *****************************************************************************
typedef struct
{
   const void *base;
   const void *adrs;
   size_t offset;
} alias_t;

// *****************************************************************************
// * Mapping from one host_@ to its memory_t
// *****************************************************************************
typedef std::unordered_map<const void*, memory_t> memory_map_t;
typedef std::unordered_map<const void*, alias_t> alias_map_t;

// *****************************************************************************
// * Memory Manager Singleton
// *****************************************************************************
class mm
{
private:
   memory_map_t *memories = NULL;
   alias_map_t  *aliases = NULL;
private:
   mm() {}
   mm(mm const&);
   void operator=(mm const&);
private:
   static inline mm& Get()
   {
      static mm singleton;
      return singleton;
   }
private:
   // **************************************************************************
   void Setup();
   inline bool Known(const void *adrs);
   inline bool Alias(const void *adrs);
   const void *Range(const void *adrs);
   const void* InsertAlias(const void *base, const void *adrs);
   void *Insert(void *adrs, const size_t bytes);
   void *Erase(void *adrs);
   void* AdrsKnown(void *adrs);
   void* AdrsAlias(void *adrs);
   void PushKnown(const void *adrs, const size_t bytes =0);
   void PushAlias(const void *adrs, const size_t bytes =0);
   void PullKnown(const void *adrs, const size_t bytes =0);
   void PullAlias(const void *adrs, const size_t bytes =0);
private:
   void* Adrs(void *adrs);
   const void* Adrs(const void *adrs);
   OccaMemory Memory(const void *adrs);
   void Push(const void *adrs, const size_t bytes =0);
   void Pull(const void *adrs, const size_t bytes =0);
   void Dump();
public:
   // **************************************************************************
   // * Main malloc template function
   // * Allocates n*size bytes and returns a pointer to the allocated memory
   // **************************************************************************
   template<class T>
   static inline T* malloc(const size_t n, const size_t size = sizeof(T))
   { return (T*) Get().Insert(::new T[n], n*size); }
   
   // **************************************************************************
   // * Frees the memory space pointed to by ptr, which must have been
   // * returned by a previous call to mm::malloc
   // **************************************************************************
   template<class T>
   static inline void free(void *ptr)
   {
      if (!ptr) { return; }
      void *back = mm::Get().Erase(ptr);
      ::delete[] static_cast<T*>(back);
      back = nullptr;
   }

   // **************************************************************************
   // * Translates adrs to host or device address,
   // * depending on config::Cuda() and the adrs' state
   // **************************************************************************
   static inline void* adrs(void *a) { return Get().Adrs(a); }
   static inline const void* adrs(const void *a) { return Get().Adrs(a); }

   static inline void push(const void *adrs, const size_t bytes =0){
      return Get().Push(adrs, bytes);
   }
   
   static inline void pull(const void *adrs, const size_t bytes =0){
      return Get().Pull(adrs, bytes);
   }

};

// *****************************************************************************
} // mfem

#endif // MFEM_MM_HPP
