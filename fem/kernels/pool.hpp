// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FEM_KERNELS_POOL_HPP
#define MFEM_FEM_KERNELS_POOL_HPP

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

namespace internal
{

namespace pool
{

/// @brief Resize an internal device memory suitable for a block of threads.
/// When @a size is 0, the internal data is deleted and the nullptr is returned.
template<int GRID, typename T = double>
static T *SetSize(const int size)
{
   static Memory<T> data;
   // when the kernel does not need this memory space, no pointer is needed
   if (GRID == 0) { return nullptr; }
   // when size is null, delete the internal data
   if (size == 0) { data.Delete(); return nullptr; }
   if (size*GRID > data.Capacity())
   {
      data.Delete();
      data.New(size*GRID, Device::GetDeviceMemoryType());
      data.UseDevice(true);
   }
   return data.Write(Device::GetDeviceMemoryClass(), data.Capacity());
}

/// Helper function to return and increment a given pointer with a given size
/**
* @brief DeviceMemAlloc
* @param[in,out] mem  The current base memory address.
* @param[in]     size The size to increment the current base memory address.
* @return The current base memory address.
*/
template<typename T> MFEM_HOST_DEVICE static
inline T *Alloc(T* &mem, const size_t size) noexcept
{
   T* base = mem;
   mem += size;
   return base;
}

} // namespace kernels::internal::pool

} // namespace kernels::internal

} // namespace kernels

} // namespace mfem

#endif // MFEM_FEM_KERNELS_POOL_HPP
