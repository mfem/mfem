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

#ifndef MFEM_BACKENDS_OCCA_ARRAY_HPP
#define MFEM_BACKENDS_OCCA_ARRAY_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include <occa.hpp>
#include "layout.hpp"
#include "../base/array.hpp"

namespace mfem
{

namespace occa
{

class Array : public virtual PArray
{
protected:
   //
   // Inherited fields
   //
   // DLayout layout;

   // Always true: Size()*item_size == slice.size() <= data.size()
   mutable ::occa::memory data, slice;

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

   inline void *GetBuffer() const;

public:
   Array(const Engine &e)
      : PArray(*(new Layout(e, 0))),
        data(e.Alloc(0)),
        slice(data)
   { }

   Array(Layout &lt, std::size_t item_size)
      : PArray(lt),
        data(lt.OccaEngine().Alloc(lt.Size()*item_size)),
        slice(data)
   { }

   virtual ~Array() { }

   inline void MakeRef(Array &master);

   Layout &OccaLayout() const { return layout->As<Layout>(); }

   const Engine &OccaEngine() const { return OccaLayout().OccaEngine(); }

   ::occa::memory &OccaMem() { return slice; }
   const ::occa::memory &OccaMem() const { return slice; }

   inline int OccaResize(Layout *lt, std::size_t item_size);

   inline int OccaResize(std::size_t new_size, std::size_t item_size);

   template <typename T>
   inline void OccaFill(const T val);

   inline void OccaAssign(const Array &src);

   inline void OccaPush(const void *src);
};


//
// Inline methods
//

inline void *Array::GetBuffer() const
{
   if (!slice.getDevice().hasSeparateMemorySpace())
   {
      return slice.ptr();
   }
   return NULL;
}

inline int Array::OccaResize(Layout *lt, std::size_t item_size)
{
   layout.Reset(lt); // Reset() checks if the pointer is the same
   const std::size_t new_bytes = lt->Size()*item_size;
   if (data.size() < new_bytes ||
       data.getDevice() != lt->OccaEngine().GetDevice())
   {
      data = lt->OccaEngine().Alloc(new_bytes);
      slice = data;
      // If memory allocation fails - an exception is thrown.
   }
   else if (slice.size() != new_bytes)
   {
      slice = data.slice(0, new_bytes);
   }
   return 0;
}

inline void Array::MakeRef(Array &master)
{
   layout = master.layout;
   data = master.data;
   slice = master.slice;
}

inline int Array::OccaResize(std::size_t new_size, std::size_t item_size)
{
   Layout &ol = OccaLayout();
   ol.OccaResize(new_size);
   return OccaResize(&ol, item_size);
}

template <typename T>
inline void Array::OccaFill(const T val)
{
   ::occa::linalg::operator_eq<T>(slice, val);
}

inline void Array::OccaAssign(const Array &src)
{
   if (slice != src.slice && slice.size() != 0)
   {
      MFEM_ASSERT(slice.size() == src.slice.size(), "");
      slice.copyFrom(src.slice);
   }
}

inline void Array::OccaPush(const void *src)
{
   if (slice.size() != 0)
   {
      slice.copyFrom(src);
   }
}


} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#endif // MFEM_BACKENDS_OCCA_ARRAY_HPP
