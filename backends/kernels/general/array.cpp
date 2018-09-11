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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

// *****************************************************************************
PArray *Array::DoClone(bool copy_data, void **buffer,
                       std::size_t item_size) const
{
   push();
   Array *new_array = new Array(KernelsLayout(), item_size);
   if (copy_data)
   {
      new_array->slice.copyFrom(slice);
   }
   if (buffer)
   {
      *buffer = new_array->GetBuffer();
   }
   pop();
   return new_array;
}

// *****************************************************************************
int Array::DoResize(PLayout &new_layout, void **buffer,
                    std::size_t item_size)
{
   push();
   MFEM_ASSERT(dynamic_cast<Layout *>(&new_layout) != NULL,
               "new_layout is not an KERNELS Layout");
   Layout *lt = static_cast<Layout *>(&new_layout);
   layout.Reset(lt); // Reset() checks if the pointer is the same
   int err = ResizeData(lt, item_size);
   if (!err && buffer)
   {
      *buffer = GetBuffer();
   }
   pop();
   return err;
}

// *****************************************************************************
int Array::ResizeData(const Layout *lt, std::size_t item_size)
{
   push();
   const std::size_t new_bytes = lt->Size()*item_size;
   dbg("data.size()=%d, slice.size()=%d & new_bytes=%d", data.size(), slice.size(), new_bytes);
   if (data.size() < new_bytes )
   {
      dbg("Alloc");
      data = lt->Alloc(new_bytes);
      slice = data;
      // If memory allocation fails - an exception is thrown.
   }
   else if (slice.size() != new_bytes)
   {
      dbg("Slice");
      slice = data.slice(0, new_bytes);
   }
   pop();
   return 0;
}

// *****************************************************************************
void *Array::DoPullData(void *buffer, std::size_t item_size)
{
   push();
   if (!slice.getDevice().hasSeparateMemorySpace())
   {
      pop();
      return slice.ptr();
   }
   if (buffer)
   {
      slice.copyTo(buffer);
   }
   pop();
   return buffer;
}

// *****************************************************************************
void Array::DoFill(const void *value_ptr, std::size_t item_size)
{
   push();
   switch (item_size)
   {
      case sizeof(int8_t):
         KernelsFill((const int8_t *)value_ptr);
         break;
      case sizeof(int16_t):
         KernelsFill((const int16_t *)value_ptr);
         break;
      case sizeof(int32_t):
         KernelsFill((const int32_t *)value_ptr);
         break;
      case sizeof(double):
         KernelsFill((const double *)value_ptr);
         break;
      default:
         MFEM_ABORT("item_size = " << item_size << " is not supported");
   }
   pop();
}

// *****************************************************************************
void Array::DoPushData(const void *src_buffer, std::size_t item_size)
{
   push();
   if (slice.getDevice().hasSeparateMemorySpace() || slice.ptr() != src_buffer)
   {
      slice.copyFrom(src_buffer);
   }
   pop();
}

// *****************************************************************************
void Array::DoAssign(const PArray &src, std::size_t item_size)
{
   push();
   const kernels::Array *source = dynamic_cast<const kernels::Array*>(&src);
   MFEM_ASSERT(source != NULL, "invalid source Array type");
   MFEM_ASSERT(Size() == source->Size(), "");
   slice.copyFrom(source->slice);
   pop();
}

// *****************************************************************************
void Array::DoMakeRefOffset(const PArray &src,
                            const std::size_t offset,
                            const std::size_t size,
                            const std::size_t item_size){
   push();
   layout->Resize(size);
   const kernels::Array &ksrc = src.As<const kernels::Array>();
   const std::size_t bytes_size = size * item_size;
   const std::size_t bytes_offset = offset * item_size;
   memory m = memory(bytes_size,ksrc.data[bytes_offset]); 
   data = slice = m;
   pop();
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
