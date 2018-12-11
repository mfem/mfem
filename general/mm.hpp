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
   bool host = true;
   size_t bytes = 0;
   void *h_adrs = NULL;
   void *d_adrs = NULL;
   OccaMemory o_adrs;
} memory_t;

// *****************************************************************************
typedef struct
{
   void *base = NULL;
   void *adrs = NULL;
   size_t offset = 0;
} alias_t;

// *****************************************************************************
// * Mapping from one host_@ to its mm2dev_t
// *****************************************************************************
typedef std::unordered_map<const void*, memory_t> memory_map_t;
typedef std::unordered_map<const void*, alias_t> alias_map_t;

// *****************************************************************************
// * Memory Manager Singleton
// *****************************************************************************
class mm
{
protected:
   memory_map_t *memories = NULL;
   alias_map_t  *aliases = NULL;
private:
   mm() {}
   mm(mm const&);
   void operator=(mm const&);
public:
   static mm& Get()
   {
      static mm singleton;
      return singleton;
   }
private:
   void Setup();
   const void* InsertAlias(const void*, const void*);
   void *Insert(const void*, const size_t, const size_t, const void* = NULL);
   void *Erase(void*);
   const void *Range(const void*);
   bool Alias(const void*);
   bool Known(const void*);
   void Sync_p(const void*);
public:
   // **************************************************************************
   template<class T>
   static inline T* malloc(const size_t size, const size_t size_of_T = sizeof(T))
   { return (T*) mm::Get().Insert(::new T[size], size, size_of_T); }
   
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
   void* Adrs(const void*);
   OccaMemory Memory(const void*);

   // **************************************************************************
   static void Sync(const void *adrs) { mm::Get().Sync_p(adrs); }
   
   // **************************************************************************
   void Push(const void*, const size_t =0);

   // **************************************************************************
   void Pull(const void*, const size_t =0);

   // **************************************************************************
   static void* H2D(void*, const void*, size_t, const bool =false);

   // ***************************************************************************
   static void* D2H(void*, const void*, size_t, const bool =false);

   // **************************************************************************
   static void* D2D(void*, const void*, size_t, const bool =false);

   // **************************************************************************
   static void* memcpy(void*, const void*, size_t);
};

// *****************************************************************************
} // mfem

#endif // MFEM_MM_HPP
