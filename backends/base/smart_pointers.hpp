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

#ifndef MFEM_BACKENDS_BASE_SMART_POINTERS_HPP
#define MFEM_BACKENDS_BASE_SMART_POINTERS_HPP

#include "../../config/config.hpp"
#ifdef MFEM_USE_BACKENDS

#include "utils.hpp"
#include "../../general/error.hpp"
#include <cstddef>

// #define MFEM_TRACE_SHARED_PTR
#ifdef MFEM_TRACE_SHARED_PTR
#include "../../general/globals.hpp"
#endif

namespace mfem
{

/// Base class for classes with simple reference counting.
/** Reference counting is performed by the class SharedPtr. */
class RefCounted
{
private:
   mutable unsigned ref_count;

   /// Only class SharedPtr can access ref_count.
   template <typename T> friend class SharedPtr;

public:
   RefCounted() : ref_count(0) { }

   /** @brief Prevent SharedPtr objects from deleting this object by
       incrementing the reference counter by one. */
   void DontDelete() const { ++ref_count; }
};


/** @brief Smart pointer class that manages objects of type T derived from class
    RefCounted. */
/** This class is generally meant to work with dynamically allocated object,
    specifically objects allocated with operator new(). It will invoke operator
    delete() to destroy the managed object when its reference counter reaches
    zero. This behavior can be overriden by calling RefCounted::DontDelete() to
    ensure that an object will not be deleted by a SharedPtr that holds a
    pointer to it.
    @note This class is NOT thread-safe and does not support circular ownership.
 */
template <typename T>
class SharedPtr
{
public:
   typedef T stored_type;

private:
   T *ptr;

   void Init(T *new_ptr)
   {
      ptr = new_ptr;
      if (ptr) { ++ptr->RefCounted::ref_count; }
#ifdef MFEM_TRACE_SHARED_PTR
#elif 0
      mfem::out << "  [" << _MFEM_FUNC_NAME << "]: ptr = " << ptr;
      if (ptr)
      {
         mfem::out << ", new ref_count = " << ptr->RefCounted::ref_count;
      }
      mfem::out << '\n';
#endif
   }
   void Destroy()
   {
      MFEM_ASSERT(!ptr || ptr->RefCounted::ref_count >= 1, "invalid use");
      if (ptr && --ptr->RefCounted::ref_count == 0) { delete ptr; }
#ifdef MFEM_TRACE_SHARED_PTR
#elif 0
      mfem::out << "  [" << _MFEM_FUNC_NAME << "]: ptr = " << ptr;
      if (ptr)
      {
         mfem::out << ", new ref_count = " << ptr->RefCounted::ref_count;
      }
      mfem::out << '\n';
#endif
   }

public:
   SharedPtr() : ptr(NULL)
   {
#ifdef MFEM_TRACE_SHARED_PTR
      mfem::out << '[' << _MFEM_FUNC_NAME << "]: ptr = " << ptr << '\n';
#endif
   }

   SharedPtr(const SharedPtr &other)
   {
#ifdef MFEM_TRACE_SHARED_PTR
      mfem::out << '[' << _MFEM_FUNC_NAME << "]\n";
#endif
      Init(other.ptr);
   }

   template <typename U>
   SharedPtr(const SharedPtr<U> &other)
   {
#ifdef MFEM_TRACE_SHARED_PTR
      mfem::out << '[' << _MFEM_FUNC_NAME << "]\n";
#endif
      Init(other.Get());
   }

   explicit SharedPtr(T *p)
   {
#ifdef MFEM_TRACE_SHARED_PTR
      mfem::out << '[' << _MFEM_FUNC_NAME << "]\n";
#endif
      Init(p);
   }

   ~SharedPtr()
   {
#ifdef MFEM_TRACE_SHARED_PTR
      mfem::out << '[' << _MFEM_FUNC_NAME << "]\n";
#endif
      Destroy();
   }

   SharedPtr &operator=(const SharedPtr &other)
   {
#ifdef MFEM_TRACE_SHARED_PTR
      mfem::out << '[' << _MFEM_FUNC_NAME << "]\n";
#endif
      Reset(other.ptr); return *this;
   }

   template <typename U>
   SharedPtr &operator=(const SharedPtr<U> &other)
   {
#ifdef MFEM_TRACE_SHARED_PTR
      mfem::out << '[' << _MFEM_FUNC_NAME << "]\n";
#endif
      Reset(other.Get()); return *this;
   }

   T &operator*() const { return *ptr; }
   T *operator->() const { return ptr; }

   operator bool() const { return ptr; }
   bool operator!() const { return !ptr; }

   template <typename U>
   bool operator==(const SharedPtr<U> &other) const
   { return ptr == other.Ptr(); }
   template <typename U>
   bool operator!=(const SharedPtr<U> &other) const
   { return ptr != other.Ptr(); }

   template <typename U>
   bool operator==(const U &p) const { return ptr == (void*) p; }
   template <typename U>
   bool operator!=(const U &p) const { return ptr != (void*) p; }

   T *Get() const { return ptr; }

   /// TODO
   template <typename derived_t>
   derived_t *As() const { return util::As<derived_t>(ptr); }

   unsigned UseCount() const { return ptr ? ptr->RefCounted::ref_count : 0; }

   void Reset()
   {
#ifdef MFEM_TRACE_SHARED_PTR
      mfem::out << '[' << _MFEM_FUNC_NAME << "]\n";
#endif
      Destroy();
      ptr = NULL;
   }

   /// The type U* needs to be implicitly convertible to T*
   template <typename U>
   void Reset(U *new_ptr)
   {
#ifdef MFEM_TRACE_SHARED_PTR
      mfem::out << '[' << _MFEM_FUNC_NAME << "]\n";
#endif
      if (ptr != new_ptr) { Destroy(); Init(new_ptr); }
   }

   void Swap(SharedPtr &other)
   {
#ifdef MFEM_TRACE_SHARED_PTR
      mfem::out << '[' << _MFEM_FUNC_NAME << "]\n";
#endif
      std::swap(ptr, other.ptr);
   }
};


template <class T>
inline void Swap(SharedPtr<T> &a, SharedPtr<T> &b) { a.Swap(b); }


class PLayout;
typedef SharedPtr<PLayout> DLayout;

class PArray;
typedef SharedPtr<PArray> DArray;

class PVector;
typedef SharedPtr<PVector> DVector;

class PParFiniteElementSpace;
typedef SharedPtr<PParFiniteElementSpace> DParFiniteElementSpace;
   
class PFiniteElementSpace;
typedef SharedPtr<PFiniteElementSpace> DFiniteElementSpace;

class PBilinearForm;
typedef SharedPtr<PBilinearForm> DBilinearForm;

} // namespace mfem

#endif // MFEM_USE_BACKENDS

#endif // MFEM_BACKENDS_BASE_SMART_POINTERS_HPP
