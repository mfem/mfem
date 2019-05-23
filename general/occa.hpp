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
#include <occa.hpp>
#include "device.hpp"

#if defined(MFEM_USE_CUDA) && OCCA_CUDA_ENABLED
#include <occa/modes/cuda/utils.hpp>
#endif

typedef occa::device OccaDevice;
typedef occa::memory OccaMemory;

#else // MFEM_USE_OCCA

typedef void* OccaDevice;
typedef void* OccaMemory;

#endif // MFEM_USE_OCCA

namespace mfem
{

#ifdef MFEM_USE_OCCA

typedef std::pair<int,int> occa_id_t;
typedef std::map<occa_id_t, occa::kernel> occa_kernel_t;

// This function is currently used to determine if an OCCA kernel should be
// used.
inline bool DeviceUseOcca()
{
   return Device::Allows(Backend::OCCA_CUDA) ||
          (Device::Allows(Backend::OCCA_OMP) &&
           !Device::Allows(Backend::DEVICE_MASK)) ||
          (Device::Allows(Backend::OCCA_CPU) &&
           !Device::Allows(Backend::DEVICE_MASK|Backend::OMP_MASK));
}

#endif // MFEM_USE_OCCA

// Function called when the pointer 'a' needs to be passed to an OCCA kernel.
OccaMemory OccaPtr(const void *a);
OccaDevice OccaDev();

} // namespace mfem

#endif // MFEM_OCCA_HPP
