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

#include "okina.hpp"

#include <string.h>
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

namespace mfem
{

#ifdef MFEM_USE_CUDA
void Device::GpuDeviceSetup(const int device)
{
   cudaGetDeviceCount(&ngpu);
   MFEM_ASSERT(ngpu>0, "No CUDA device found!");
   cuInit(0);
   dev = device;
   cuDeviceGet(&cuDevice,dev);
   cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice);
   cuStream = new CUstream;
   MFEM_ASSERT(cuStream, "CUDA stream could not be created!");
   cuStreamCreate(cuStream, CU_STREAM_DEFAULT);
}
#endif

void Device::CudaDeviceSetup(const int device)
{
#ifdef MFEM_USE_CUDA
   GpuDeviceSetup(device);
#else
   MFEM_ABORT("CUDA requested but MFEM was not build with MFEM_USE_CUDA=YES");
#endif
}

void Device::RajaDeviceSetup(const int device)
{
#if defined(MFEM_USE_CUDA) && defined(MFEM_USE_RAJA)
   GpuDeviceSetup(device);
#elif !defined(MFEM_USE_RAJA)
   MFEM_ABORT("RAJA requested but MFEM was not build with MFEM_USE_RAJA=YES");
#endif
}

void Device::OccaDeviceSetup(CUdevice cu_dev, CUcontext cu_ctx)
{
#ifdef MFEM_USE_OCCA
   if (cuda)
   {
      occaDevice = OccaWrapDevice(cu_dev, cu_ctx);
   }
   else if (omp)
   {
      occaDevice.setup("mode: 'OpenMP'");
   }
   else
   {
      occaDevice.setup("mode: 'Serial'");
   }
   const std::string mfem_dir = occa::io::dirname("../_config.hpp");
   occa::io::addLibraryPath("fem", mfem_dir + "fem");
   occa::loadKernels();
   occa::loadKernels("fem");
#else
   MFEM_ABORT("OCCA requested but MFEM was not built with MFEM_USE_OCCA=YES");
#endif
}

void Device::MFEMDeviceSetup(const int dev)
{
   MFEM_ASSERT(ngpu==-1, "Only one MFEMDeviceSetup allowed");
   ngpu = 0;

   // We initialize CUDA first so OccaDeviceSetup() can reuse the same
   // initialized cuDevice and cuContext objects
   if (cuda) { CudaDeviceSetup(dev); }
   if (raja) { RajaDeviceSetup(dev); }
   if (occa) { OccaDeviceSetup(cuDevice, cuContext); }

   if (cuda && ngpu==0)
   {
      MFEM_ABORT("CUDA requested but MFEM was not built with MFEM_USE_CUDA=YES");
   }
}

Device::~Device()
{
   if (raja || cuda) { delete cuStream; }
}

} // mfem
