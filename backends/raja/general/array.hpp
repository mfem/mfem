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

#ifndef MFEM_BACKENDS_RAJA_ARRAY_HPP
#define MFEM_BACKENDS_RAJA_ARRAY_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{

class Array : public virtual PArray
{
protected:
   //
   // Inherited fields
   //
   // DLayout layout;

   // Always true: Size()*item_size == slice.size() <= data.size()
   mutable raja::memory data, slice;

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

   int ResizeData(const Layout *lt, std::size_t item_size);

   template <typename T>
   inline void RajaFill(const T *val_ptr)
   { raja::linalg::operator_eq<T>(slice, *val_ptr); }

public:
   Array(Layout &lt, std::size_t item_size)
      : PArray(lt),
        data(lt.Alloc(lt.Size()*item_size)),
        slice(data)
   { }

   virtual ~Array() { }

   inline void MakeRef(Array &master);

   Layout &RajaLayout() const
   { return *static_cast<Layout *>(layout.Get()); }

   raja::memory &RajaMem() { return slice; }
   const raja::memory &RajaMem() const { return slice; }
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

inline void Array::MakeRef(Array &master)
{
   layout = master.layout;
   data = master.data;
   slice = master.slice;
}

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_ARRAY_HPP
