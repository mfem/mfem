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

#ifndef MFEM_KERNELS_HPP
#define MFEM_KERNELS_HPP

// *****************************************************************************
#ifdef __NVCC__
template <typename BODY> __global__
void gpu_kernel(const size_t N, BODY d_body) {
  const size_t k = blockDim.x*blockIdx.x + threadIdx.x;
  if (k >= N) return;
  d_body(k);
}
#endif // __NVCC__

// *****************************************************************************
template <typename DBODY, typename HBODY>
void wrap(const size_t N, DBODY &&d_body, HBODY &&h_body){
   stk(true);
#ifdef __NVCC__
   const bool gpu = cfg::Get().Cuda();
   if (gpu){
      //dbg("\nGPU");
      const size_t blockSize = 256;
      const size_t gridSize = (N+blockSize-1)/blockSize;
      gpu_kernel<<<gridSize, blockSize>>>(N,d_body);
      cudaDeviceSynchronize();
      return;
   }
#endif // __NVCC__
   //dbg("\nCPU");
   for(size_t k=0; k<N; k+=1){ h_body(k); }
}

// *****************************************************************************
// * FORALL split
// *****************************************************************************
#define forall_this(i,end,body) wrap(end,                                    \
                                [=,*this] __device__ (size_t i){body},  \
                                [=] (size_t i){body})

#define forall(i,end,body) wrap(end,                                    \
                                [=] __device__ (size_t i){body},        \
                                [=] (size_t i){body})

#endif // MFEM_KERNELS_HPP
