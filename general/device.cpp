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
#include "cuda.hpp"
#include "occa.hpp"

namespace mfem
{

CUstream *cuStream;
static CUdevice cuDevice;
static CUcontext cuContext;
OccaDevice occaDevice;

#ifdef MFEM_USE_CUDA
static void DeviceSetup(const int dev, int &ngpu)
{
   cudaGetDeviceCount(&ngpu);
   MFEM_ASSERT(ngpu>0, "No CUDA device found!");
   cuInit(0);
   cuDeviceGet(&cuDevice,dev);
   cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice);
   cuStream = new CUstream;
   MFEM_ASSERT(cuStream, "CUDA stream could not be created!");
   cuStreamCreate(cuStream, CU_STREAM_DEFAULT);
}
#endif

static void CudaDeviceSetup(const int dev, int &ngpu)
{
#ifdef MFEM_USE_CUDA
   DeviceSetup(dev, ngpu);
#else
   MFEM_ABORT("CUDA requested but MFEM was not built with MFEM_USE_CUDA=YES");
#endif
}

static void RajaDeviceSetup(const int dev, int &ngpu)
{
#if defined(MFEM_USE_CUDA) && defined(MFEM_USE_RAJA)
   DeviceSetup(dev, ngpu);
#elif !defined(MFEM_USE_RAJA)
   MFEM_ABORT("RAJA requested but MFEM was not built with MFEM_USE_RAJA=YES");
#endif
}

static void OccaDeviceSetup(CUdevice cu_dev, CUcontext cu_ctx)
{
#ifdef MFEM_USE_OCCA
   const bool omp  = Device::UsingOmp();
   const bool cuda = Device::UsingCuda();
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

   std::string mfemDir;
   if (occa::io::exists(MFEM_INSTALL_DIR "/include/mfem/")) {
     mfemDir = MFEM_INSTALL_DIR "/include/mfem/";
   } else if (occa::io::exists(MFEM_SOURCE_DIR)) {
     mfemDir = MFEM_SOURCE_DIR;
   } else {
     MFEM_ABORT("Cannot find OCCA kernels in MFEM_INSTALL_DIR or MFEM_SOURCE_DIR");
   }

   occa::io::addLibraryPath("mfem", mfemDir);
   occa::loadKernels("mfem");
#else
   MFEM_ABORT("OCCA requested but MFEM was not built with MFEM_USE_OCCA=YES");
#endif
}

void Device::Setup(const int device)
{
   dev = device;

   MFEM_ASSERT(ngpu==-1, "Only one MFEMDeviceSetup allowed");
   ngpu = 0;

   // We initialize CUDA first so OccaDeviceSetup() can reuse the same
   // initialized cuDevice and cuContext objects
   if (cuda) { CudaDeviceSetup(dev, ngpu); }
   if (raja) { RajaDeviceSetup(dev, ngpu); }
   if (occa) { OccaDeviceSetup(cuDevice, cuContext); }

   if (cuda && ngpu==0)
   {
      MFEM_ABORT("CUDA requested but MFEM was not built with MFEM_USE_CUDA=YES");
   }
}

Device::~Device()
{
   if (cuda) { delete cuStream; }
}

} // mfem
