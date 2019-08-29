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
#ifndef MFEM_RAJA_HPP
#define MFEM_RAJA_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_RAJA
#include "RAJA/RAJA.hpp"
#endif

#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)

namespace RAJA
{

namespace cuda
{

__device__ __forceinline__ int globalIdx2D()
{
   return blockIdx.x * blockDim.z + threadIdx.z;
}

__device__ __forceinline__ int globalIdx3D() { return blockIdx.x; }

template <bool batch, typename Iterator, typename DBODY, typename IndexType>
__global__ void RajaKernelND(DBODY body, Iterator I, IndexType N)
{
   const auto k = static_cast<IndexType>(batch ? globalIdx2D():globalIdx3D());
   if (k >= N) { return; }
   body(I[k]);
}

template <bool batch, typename Iterable, typename DBODY >
RAJA_INLINE void forallND(dim3 grid, dim3 block, Iterable&& iter, DBODY&& body)
{
   const size_t sm = 0;
   const cudaStream_t stream = 0;
   using Body = camp::decay<DBODY>;
   const auto End = std::end(iter);
   const auto Begin = std::begin(iter);
   const auto Distance = std::distance(Begin, End);
   using Iterator = camp::decay<decltype(Begin)>;
   using IndexType = camp::decay<decltype(Distance)>;
   const IndexType N = std::distance(Begin, End);
   const auto ker = RajaKernelND<batch, Iterator, Body, IndexType>;
   void *args[] = {(void*)&std::forward<DBODY>(body), (void*)&Begin, (void*)&N};
   cudaErrchk(cudaLaunchKernel((const void*)ker, grid, block, args, sm, stream));
   RAJA::cuda::launch(stream);
}

}  // namespace cuda

}  // namespace RAJA

#endif  // defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)

#endif  // MFEM_RAJA_HPP
