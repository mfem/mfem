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

#include "../general/okina.hpp"

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
// * CUDA device setup, called when CUDA or RAJA mode with MFEM_USE_CUDA
// *****************************************************************************
#ifdef MFEM_USE_CUDA
void config::GpuDeviceSetup(const int device)
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

// *****************************************************************************
// * cudaDeviceSetup will set: gpu_count, dev, cuDevice, cuContext & cuStream
// *****************************************************************************
void config::CudaDeviceSetup(const int device)
{
#ifdef MFEM_USE_CUDA
   GpuDeviceSetup(device);
#else
   MFEM_ABORT("CUDA requested but no GPU support has been built!");
#endif
}

// *****************************************************************************
// * RajaDeviceSetup will set: gpu_count, dev, cuDevice, cuContext & cuStream
// *****************************************************************************
void config::RajaDeviceSetup(const int device)
{
#if defined(MFEM_USE_CUDA) && defined(MFEM_USE_RAJA)
   GpuDeviceSetup(device);
#elif !defined(MFEM_USE_RAJA)
   MFEM_ABORT("RAJA requested but no RAJA support has been built!");
#endif
}

// *****************************************************************************
// * OCCA Settings: device, paths & kernels
// *****************************************************************************
void config::OccaDeviceSetup(CUdevice cu_dev, CUcontext cu_ctx)
{
#ifdef MFEM_USE_OCCA
   if (cuda)
   {
      occaDevice = occaWrapDevice(cu_dev, cu_ctx);
   }
   else
   {
      occaDevice.setup("mode: 'Serial'");
   }
   const std::string mfem_dir = occa::io::dirname(__FILE__) + "../";
   occa::io::addLibraryPath("fem", mfem_dir + "fem");
   occa::loadKernels();
   occa::loadKernels("fem");
#else
   MFEM_ABORT("OCCA requested but no support has been built!");
#endif
}

// *****************************************************************************
// * We initialize CUDA first so OccaDeviceSetup() can reuse
// * the same initialized cuDevice and cuContext objects
// *****************************************************************************
void config::MfemDeviceSetup(const int dev)
{
   MFEM_ASSERT(ngpu==-1, "Only one MfemDeviceSetup allowed");
   ngpu = 0;
   if (cuda) { CudaDeviceSetup(dev); }
   if (raja) { RajaDeviceSetup(dev); }
   if (occa) { OccaDeviceSetup(cuDevice, cuContext); }
   if (cuda && ngpu==0)
   {
      MFEM_ABORT("CUDA requested but no GPU has been initialized!");
   }
}

// *****************************************************************************
// * Destructor
// *****************************************************************************
config::~config()
{
   if (raja || cuda) { delete cuStream; }
}

} // mfem
