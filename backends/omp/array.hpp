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

#ifndef MFEM_BACKENDS_OMP_ARRAY_HPP
#define MFEM_BACKENDS_OMP_ARRAY_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "layout.hpp"
#include "../base/array.hpp"

namespace mfem
{

namespace omp
{

class Array : public virtual mfem::PArray
{
protected:
   //
   // Inherited fields
   //
   // DLayout layout;

   bool own_data;
   std::size_t bytes;
   char *data;

   //
   // Virtual interface
   //

   virtual PArray *DoClone(bool copy_data, void **buffer,
                           std::size_t item_size) const;

   virtual int DoResize(PLayout &new_layout, void **buffer,
                        std::size_t item_size);

   virtual void *DoPullData(void *buffer, std::size_t item_size);

   virtual void DoFill(const void *value_ptr, std::size_t item_size);

   virtual void DoPushData(const void *src_buffer, std::size_t item_size);

   virtual void DoAssign(const PArray &src, std::size_t item_size);

   //
   // Auxiliary methods
   //

   inline char *GetBuffer() const;

   inline int ResizeData(const Layout *lt, std::size_t item_size);

   inline bool IsUnifiedMemory() const { return OmpLayout().OmpEngine().UnifiedMemory(); }

   template <typename T>
   void OmpFill(const T *pval)
   {
      T *ptr = (T*) data;
      T val = *pval;
      const bool use_target = ComputeOnDevice();
      const bool use_parallel = (use_target || layout->Size() > 1000);
      const std::size_t size = layout->Size();
#pragma omp target teams distribute parallel for        \
   if (target: use_target)                              \
   if (parallel: use_parallel) map (to: ptr, val)
      for (int i = 0; i < size; i++) ptr[i] = val;
   }

public:
   Array(Layout &lt, std::size_t item_size)
      : PArray(lt),
        own_data(true),
        bytes(lt.Size() * item_size),
        data(static_cast<char *>(lt.Alloc(bytes)))
   {
#pragma omp target enter data map(alloc:data[:bytes]) if (!IsUnifiedMemory() && ComputeOnDevice())
   }

   Array(const Array &array)
      : PArray(array.GetLayout()),
        own_data(false),
        bytes(array.bytes),
        data(array.data) { }

   inline bool ComputeOnDevice() const { return (OmpLayout().OmpEngine().ExecTarget() == Device); }

   virtual ~Array()
   {
#pragma omp target exit data map(delete:data[:bytes]) if (!IsUnifiedMemory() && ComputeOnDevice())
      if (own_data) layout->As<Layout>().Dealloc(data);
   }

   inline void MakeRef(Array &master);

   template <class T>
   T* GetData() { return (T *) data; }

   template <class T>
   const T* GetData() const { return (T *) data; }

   Layout &OmpLayout() const
   { return *static_cast<Layout *>(layout.Get()); }
};


//
// Inline methods
//

inline char *Array::GetBuffer() const
{
   return data;
}

inline int Array::ResizeData(const Layout *lt, std::size_t item_size)
{
   const std::size_t new_bytes = lt->Size() * item_size;
   if (bytes < new_bytes)
   {
#pragma omp target exit data map(delete:data)
      OmpLayout().Dealloc(data);
      data = static_cast<char *>(OmpLayout().Alloc(new_bytes));
      MFEM_VERIFY(data != NULL, "");
      // If memory allocation fails - an exception is thrown.
#pragma omp target enter data map(alloc:data[:new_bytes])
   }
   return 0;
}

inline void Array::MakeRef(Array &master)
{
   layout = master.layout;
   data = master.data;
}

} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#endif // MFEM_BACKENDS_OMP_ARRAY_HPP
