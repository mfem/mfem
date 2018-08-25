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

#ifndef MFEM_BACKENDS_KERNELS_ARRAY_HPP
#define MFEM_BACKENDS_KERNELS_ARRAY_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

class Array : public virtual mfem::PArray
{
protected:
   
   mutable kernels::memory data, slice;
   
   // Virtual interface ********************************************************
   virtual PArray *DoClone(bool copy_data, void **buffer,
                           std::size_t item_size) const;

   virtual int DoResize(PLayout &new_layout, void **buffer,
                        std::size_t item_size);

   virtual void *DoPullData(void *buffer, std::size_t item_size);

   virtual void DoFill(const void *value_ptr, std::size_t item_size);

   virtual void DoPushData(const void *src_buffer, std::size_t item_size);

   virtual void DoAssign(const PArray &src, std::size_t item_size);

   virtual void DoMakeRefOffset(const PArray &src,
                                const std::size_t offset,
                                const std::size_t size,
                                const std::size_t item_size);
   
   // Auxiliary methods ********************************************************
   
   inline void *GetBuffer() const;

   int ResizeData(const Layout *lt, std::size_t item_size);

   template <typename T>
   inline void KernelsFill(const T *val_ptr)
   {
      push();
      kernels::linalg::operator_eq<T>(slice, *val_ptr);
      pop();
   }

public:
   Array(Layout &lt, std::size_t item_size)
      : PArray(lt),
        data(lt.Alloc(lt.Size()*item_size)),
        slice(data)
   { }

   virtual ~Array() { }

   inline void MakeRef(Array &master);
   
   Layout &KernelsLayout() const
   { return *static_cast<Layout*>(layout.Get()); }

   kernels::memory &KernelsMem() { return slice; }

   const kernels::memory &KernelsMem() const { return slice; }
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

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_ARRAY_HPP
