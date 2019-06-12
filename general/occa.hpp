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

#ifndef MFEM_OCCA_HPP
#define MFEM_OCCA_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_OCCA
#include "mem_manager.hpp"
#include "device.hpp"
#include <occa.hpp>

namespace mfem
{

/// Return the default occa::device used by MFEM.
occa::device &OccaDev();

/// Wrap a pointer as occa::memory with the default occa::device used by MFEM.
/** It is assumed that @a ptr is suitable for use with the current mfem::Device
    configuration. */
occa::memory OccaMemoryWrap(void *ptr, std::size_t bytes);

/** @brief Wrap a Memory object as occa::memory for read only access with the
    mfem::Device MemoryClass. The returned occa::memory is associated with the
    default occa::device used by MFEM. */
template <typename T>
const occa::memory OccaMemoryRead(const Memory<T> &mem, size_t size)
{
   mem.UseDevice(true);
   const void *ptr = mem.Read(Device::GetMemoryClass(), size);
   return OccaMemoryWrap(const_cast<void *>(ptr), size*sizeof(T));
}

/** @brief Wrap a Memory object as occa::memory for write only access with the
    mfem::Device MemoryClass. The returned occa::memory is associated with the
    default occa::device used by MFEM. */
template <typename T>
occa::memory OccaMemoryWrite(Memory<T> &mem, size_t size)
{
   mem.UseDevice(true);
   return OccaMemoryWrap(mem.Write(Device::GetMemoryClass(), size),
                         size*sizeof(T));
}

/** @brief Wrap a Memory object as occa::memory for read-write access with the
    mfem::Device MemoryClass. The returned occa::memory is associated with the
    default occa::device used by MFEM. */
template <typename T>
occa::memory OccaMemoryReadWrite(Memory<T> &mem, size_t size)
{
   mem.UseDevice(true);
   return OccaMemoryWrap(mem.ReadWrite(Device::GetMemoryClass(), size),
                         size*sizeof(T));
}


/** @brief Function that determines if an OCCA kernel should be used, based on
    the current mfem::Device configuration. */
inline bool DeviceCanUseOcca()
{
   return Device::Allows(Backend::OCCA_CUDA) ||
          (Device::Allows(Backend::OCCA_OMP) &&
           !Device::Allows(Backend::DEVICE_MASK)) ||
          (Device::Allows(Backend::OCCA_CPU) &&
           !Device::Allows(Backend::DEVICE_MASK|Backend::OMP_MASK));
}

typedef std::pair<int,int> occa_id_t;
typedef std::map<occa_id_t, occa::kernel> occa_kernel_t;

} // namespace mfem

#endif // MFEM_USE_OCCA

#endif // MFEM_OCCA_HPP
