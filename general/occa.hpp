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

#ifdef MFEM_USE_OCCA
#include <occa.hpp>

#ifdef MFEM_USE_CUDA
#include <occa/mode/cuda/utils.hpp>
#endif

typedef occa::device OccaDevice;
typedef occa::memory OccaMemory;
#define MFEM_NEW_OCCA_KERNEL(ker, filepath,prop)                         \
   static occa::kernel ker = NULL;                                       \
   if (ker==NULL) {                                                      \
      ker = occaDevice.buildKernel("occa://mfem/" filepath, #ker, prop); \
   }
#else // MFEM_USE_OCCA

typedef void* OccaDevice;
typedef void* OccaMemory;

#endif // MFEM_USE_OCCA

namespace mfem
{

extern OccaDevice occaDevice;
OccaMemory OccaPtr(const void *a);
OccaDevice OccaWrapDevice(CUdevice dev, CUcontext ctx);

} // mfem

#endif // MFEM_OCCA_HPP
