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

#include "cuda.hpp"
#include "globals.hpp"

namespace mfem
{

#ifdef MFEM_USE_CUDA
void mfem_cuda_error(cudaError_t err, const char *expr, const char *func,
                     const char *file, int line)
{
   mfem::err << "CUDA error: (" << expr << ") failed with error:\n --> "
             << cudaGetErrorString(err)
             << "\n ... in function: " << func
             << "\n ... in file: " << file << ':' << line << '\n';
   mfem_error();
}
#endif // MFEM_USE_CUDA

void CudaDeviceSetup(const int dev, int &ngpu)
{
#ifdef MFEM_USE_CUDA
   MFEM_CUDA_CHECK(cudaGetDeviceCount(&ngpu));
   MFEM_VERIFY(ngpu > 0, "No CUDA device found!");
   MFEM_CUDA_CHECK(cudaSetDevice(dev));
#endif // MFEM_USE_CUDA
}

} // namespace mfem
