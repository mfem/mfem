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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include <cstring>
#include "array.hpp"

namespace mfem
{

namespace omp
{

PArray *Array::DoClone(bool copy_data, void **buffer,
                       std::size_t item_size) const
{
   Array *new_array = new Array(OmpLayout(), item_size);
   if (copy_data)
   {
      const std::size_t total_size = item_size * OmpLayout().Size();
      if (!ComputeOnDevice())
	 std::memcpy(new_array->GetBuffer(), data, total_size);
      else
      {
	 char *new_data = new_array->GetBuffer();
#pragma omp target teams distribute parallel for is_device_ptr(new_data)
	 for (int i = 0; i < total_size; i++) new_data[i] = data[i];
      }
   }
   if (buffer)
   {
      *buffer = new_array->GetBuffer();
   }
   return new_array;
}

int Array::DoResize(PLayout &new_layout, void **buffer,
                    std::size_t item_size)
{
   MFEM_ASSERT(dynamic_cast<Layout *>(&new_layout) != NULL,
               "new_layout is not an OMP Layout");
   Layout *lt = static_cast<Layout *>(&new_layout);
   layout.Reset(lt); // Reset() checks if the pointer is the same
   int err = ResizeData(lt, item_size);
   if (!err && buffer)
   {
      *buffer = GetBuffer();
   }
   return err;
}

void *Array::DoPullData(void *buffer, std::size_t item_size)
{
   // called only when Size() != 0

   if (!IsUnifiedMemory() && ComputeOnDevice())
   {
#pragma omp target update from(data)
      std::memcpy(buffer, data, item_size * Size());
   }
   else
   {
      buffer = data;
   }

   return buffer;
}

void Array::DoFill(const void *value_ptr, std::size_t item_size)
{
   // called only when Size() != 0

   switch (item_size)
   {
      case sizeof(int):
         OmpFill((const int *)value_ptr);
         break;
      case sizeof(double):
         OmpFill((const double *)value_ptr);
         break;
      default:
         MFEM_ABORT("item_size = " << item_size << " is not supported");
   }
}

void Array::DoPushData(const void *src_buffer, std::size_t item_size)
{
   // called only when Size() != 0

   std::memcpy(data, (char *) src_buffer, Size() * item_size);

   if ((!IsUnifiedMemory() && ComputeOnDevice()) && (data != src_buffer))
   {
#pragma omp target update to(data)
   }
}

void Array::DoAssign(const PArray &src, std::size_t item_size)
{
   // called only when Size() != 0

   // Note: static_cast can not be used here since PArray is a virtual base
   //       class.
   const Array *source = dynamic_cast<const Array *>(&src);
   MFEM_ASSERT(source != NULL, "invalid source Array type");
   MFEM_ASSERT(Size() == source->Size(), "");
   // All arrays from this engine are of the same type, so we can simply check *this and assume the same is used in src.
   DoPushData(source->GetBuffer(), item_size);
}

} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)
