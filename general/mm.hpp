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
typedef struct mm2dev
{
   bool host = true;
   size_t bytes = 0;
   void *h_adrs = NULL;
   void *d_adrs = NULL;
   memory o_adrs;
   bool ranged = false;
   void *b_adrs = NULL;
   size_t n_rangers = 0;
   void **rangers = NULL;
} mm2dev_t;

// *****************************************************************************
// * Mapping from one host_@ to its mm2dev_t
// *****************************************************************************
typedef std::unordered_map<const void*, mm2dev_t> mm_t;

// *****************************************************************************
// * Memory Manager Singleton
// *****************************************************************************
class mm
{
protected:
   mm_t *mng = NULL;
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
   void *Insert(const void*, const size_t, const size_t,
                const char* file, const int line, const void* = NULL);
   void *Erase(void*);
   void *Range(const void*);
   bool Known(const void*, const bool = false);
   void Sync_p(const void*);
public:
   // **************************************************************************
   template<class T>
   static inline T* malloc(const size_t size,
                           const size_t size_of_T = sizeof(T),
                           const char* file = __FILE__,
                           const int line = __LINE__)
   { return (T*) mm::Get().Insert(::new T[size], size, size_of_T, file, line); }

   // **************************************************************************
#define mm_malloc(T,bytes) mm::malloc<T>(bytes,sizeof(T),__FILE__,__LINE__)

   
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
   memory Memory(const void*);

   // **************************************************************************
   static void Sync(const void *adrs) { mm::Get().Sync_p(adrs); }
   
   // **************************************************************************
   void Push(const void*);

   // **************************************************************************
   void Pull(const void*);

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
