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
// * cudaDeviceSetup will set: gpu_count, dev, cuDevice, cuContext & cuStream
// *****************************************************************************
void config::CudaDeviceSetup(const int device)
{
#ifdef __NVCC__
   dbg("CUDA");
   gpu_count = 0;
   cudaGetDeviceCount(&gpu_count);
   MFEM_ASSERT(gpu_count>0, "No CUDA device found!");
   cuInit(0);
   dev = device;
   cuDeviceGet(&cuDevice,dev);
   cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice);
   cuStream = new CUstream;
   MFEM_ASSERT(cuStream, "CUDA stream could not be created!");
   cuStreamCreate(cuStream, CU_STREAM_DEFAULT);
#endif
}

// *****************************************************************************
// * OCCA Settings: device, paths & kernels
// *****************************************************************************
void config::OccaDeviceSetup(CUdevice cu_dev, CUcontext cu_ctx)
{
#ifdef __OCCA__
   if (cuda)
   {
      dbg("OCCA @ GPU, gpu_count=%d", gpu_count);
      occaDevice = occaWrapDevice(cu_dev, cu_ctx);
   }
   else
   {
      dbg("OCCA @ CPU, gpu_count=%d", gpu_count);
      occaDevice.setup("mode: 'Serial'");
   }
   const std::string mfem_dir = occa::io::dirname(__FILE__) + "../";
   occa::io::addLibraryPath("fem",     mfem_dir + "fem/kernels");
   occa::io::addLibraryPath("general", mfem_dir + "general/kernels");
   occa::io::addLibraryPath("linalg",  mfem_dir + "linalg/kernels");
   occa::io::addLibraryPath("mesh",    mfem_dir + "mesh/kernels");
   occa::loadKernels();
   occa::loadKernels("fem");
   occa::loadKernels("general");
   occa::loadKernels("linalg");
   occa::loadKernels("mesh");
#endif
}

// *****************************************************************************
// * We initialize CUDA first so OccaDeviceSetup() can reuse
// * the same initialized cuDevice and cuContext objects
// *****************************************************************************
void config::MfemDeviceSetup(const int device)
{
   stk();
   CudaDeviceSetup(device);
   OccaDeviceSetup(cuDevice, cuContext);
}

// *****************************************************************************
} // mfem
