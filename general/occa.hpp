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

// Macros
#define MFEM_GET_OCCA_MEMORY(v) occa::memory o_##v = mm::occaPtr(v)
#define MFEM_GET_OCCA_CONST_MEMORY(v) MFEM_GET_OCCA_MEMORY(v)
#define MFEM_NEW_OCCA_PROPERTY(props) occa::properties props;
#define MFEM_SET_OCCA_PROPERTY(props,name) props["defines/" #name] = name;
#define MFEM_NEW_OCCA_KERNEL(ker,library,filename,props)                \
   static occa::kernel ker = NULL;                                      \
   if (ker==NULL) {                                                     \
      OccaDevice device = Device::GetOccaDevice();                      \
      const std::string fdk = "occa://" #library "/" #filename;         \
      const std::string odk = occa::io::occaFileOpener().expand(fdk);   \
      ker = device.buildKernel(odk, #ker, props);                       \
   }

#else // MFEM_USE_OCCA

typedef void* OccaDevice;
typedef void* OccaMemory;

#endif // MFEM_USE_OCCA

namespace mfem
{

OccaDevice OccaWrapDevice(CUdevice device, CUcontext context);
OccaMemory OccaDeviceMalloc(OccaDevice device, const size_t bytes);
OccaMemory OccaWrapMemory(const OccaDevice device, void *d_adrs,
                          const size_t bytes);
void *OccaMemoryPtr(OccaMemory o_mem);
void OccaCopyFrom(OccaMemory o_mem, const void *h_adrs);
void OccaCopyTo(OccaMemory o_mem, void *h_adrs);

} // mfem

#endif // MFEM_OCCA_HPP
