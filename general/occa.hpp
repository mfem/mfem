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
#include "cuda.hpp" // for CUdevice, CUcontext

#ifdef MFEM_USE_OCCA
#include <occa.hpp>

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

// Function called when the pointer 'a' needs to be passed to an OCCA kernel.
OccaMemory OccaPtr(const void *a);
OccaDevice OccaDev();

} // namespace mfem

#endif // MFEM_OCCA_HPP
