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

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "array.hpp"
#include <cstring>

namespace mfem
{

namespace pa
{

////////////
// CpuArray

Array *Array::DoClone(bool copy_data, void **buffer,
                      std::size_t item_size) const
{
   Layout& lay = dynamic_cast<Layout&>(*layout);
   Array *new_array = new Array(lay, item_size);
   if (copy_data)
   {
      memcpy(new_array->data, data, size);
   }
   if (buffer)
   {
      *buffer = new_array->data;
   }
   return new_array;
}

int Array::DoResize(PLayout &new_layout, void **buffer,
                    std::size_t item_size)
{
   Layout *lt = static_cast<Layout *>(&new_layout);
   layout.Reset(lt); // Reset() checks if the pointer is the same
   int err = ResizeData(lt, item_size);
   if (!err && buffer)
   {
      *buffer = this->data;
   }
   return err;
}

void* Array::DoPullData(void *buffer, std::size_t item_size)
{
   //Always on host
   return data;
}

void Array::DoFill(const void *value_ptr, std::size_t item_size)
{
   char* this_data = GetTypedData<char>();
   for (std::size_t i = 0; i < size; i += item_size)
   {
      memcpy(this_data + i, value_ptr, item_size);
   }
}

void Array::DoPushData(const void *src_buffer, std::size_t item_size)
{
   if (src_buffer) memcpy(data, src_buffer, size);
}

void Array::DoAssign(const PArray &src, std::size_t item_size)
{
   // called only when Size() != 0
   const Array* src_array = dynamic_cast<const Array*>(&src);
   memcpy(data, src_array->data, size);
}

} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)
