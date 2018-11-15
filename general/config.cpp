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

namespace mfem
{

// *****************************************************************************
// * cuDeviceSetup will set: gpu_count, dev, cuDevice, cuContext & cuStream
// *****************************************************************************
void config::cuDeviceSetup(const int device)
{
#ifdef __NVCC__
   gpu_count=0;
   cudaGetDeviceCount(&gpu_count);
   assert(gpu_count>0);
   cuInit(0);
   dev = device; // findCudaDevice(argc, (const char **)argv);
   cuDeviceGet(&cuDevice,dev);
   cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice);
   cuStream = new CUstream;
   cuStreamCreate(cuStream, CU_STREAM_DEFAULT);
#endif
}

} // namespace mfem
