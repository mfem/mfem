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

namespace mfem
{

// *****************************************************************************
typedef struct mm2dev
{
   bool host = true;
   size_t bytes = 0;
   const void *h_adrs = NULL;
   const void *d_adrs = NULL;
} mm2dev_t;

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
   // **************************************************************************
private:
   void Setup();
   void *add(const void*, const size_t, const size_t);
   void *del(const void*);
   void Cuda();

   // **************************************************************************
public:
   template<class T>
   static inline T* malloc(size_t size, const size_t size_of_T = sizeof(T))
   {
      if (!mm::Get().mng) { mm::Get().Setup(); }
      return (T*) mm::Get().add(::new T[size], size, size_of_T);
   }

   // **************************************************************************
   template<class T>
   static inline void free(void *ptr)
   {
      if (!ptr) { return; }
      void *back = mm::Get().del(ptr);
      ::delete[] static_cast<T*>(back);
      back = nullptr;
   }

   // **************************************************************************
   void* Adrs(const void*);

   // **************************************************************************
   bool Known(const void*);

   // **************************************************************************
   void Rsync(const void*);

   // **************************************************************************
   void Push(const void*);

   // **************************************************************************
   static void* H2H(void*, const void*, size_t, const bool =false);

   // **************************************************************************
   static void* H2D(void*, const void*, size_t, const bool =false);

   // ***************************************************************************
   static void* D2H(void*, const void*, size_t, const bool =false);

   // **************************************************************************
   static void* D2D(void*, const void*, size_t, const bool =false);

   // **************************************************************************
   static void* memcpy(void*, const void*, size_t);
};

} // namespace mfem

#endif // MFEM_MM_HPP
