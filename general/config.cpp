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
void config::SetCuda(const bool mode) { 
   if (mode){
      dbg("\033[7;1;32m[CUDA]\033[m");
   }else{
      dbg("\033[7;1;31m[CUDA]\033[m");
   }
   cuda = mode;
}

// *****************************************************************************
// * cudaDeviceSetup will set: gpu_count, dev, cuDevice, cuContext & cuStream
// *****************************************************************************
void config::cudaDeviceSetup(const int device)
{
#ifdef __NVCC__
   gpu_count=0;
   cudaGetDeviceCount(&gpu_count);
   assert(gpu_count>0);
   cuInit(0);
   dev = device;
   cuDeviceGet(&cuDevice,dev);
   cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice);
   cuStream = new CUstream;
   cuStreamCreate(cuStream, CU_STREAM_DEFAULT);
   dbg("\033[32;1mCUDA initialized!\033[m");
#endif
}

// *****************************************************************************
// * Setting device, paths & kernels
// *****************************************************************************
void config::occaDeviceSetup()
{
#ifdef __OCCA__
   assert(false);
   dbg("[occa] cuda is %s",cuda?"true":"false");
   dbg("[occa] occa is %s",occa?"true":"false");
   if (cuda)
   {
      dbg("[occa] Wrapping CUDA device");
      occaDevice = okWrapDevice(cuDevice, cuContext);
   }
   else
   {
      dbg("[occa] Using OCCA Serial device");
      occaDevice.setup("mode: 'Serial'");
   }
   const std::string pwd = occa::io::dirname(__FILE__) + "../";
   occa::io::addLibraryPath("fem",     pwd + "fem/kernels");
   occa::io::addLibraryPath("general", pwd + "general/kernels");
   occa::io::addLibraryPath("linalg",  pwd + "linalg/kernels");
   occa::io::addLibraryPath("mesh",    pwd + "mesh/kernels");
   occa::loadKernels();
   occa::loadKernels("fem");
   occa::loadKernels("general");
   occa::loadKernels("linalg");
   occa::loadKernels("mesh");
#endif
}

// *****************************************************************************
void config::devSetup(const int device)
{
   push();
   cudaDeviceSetup(device);
   occaDeviceSetup();
}

// *****************************************************************************
} // mfem
