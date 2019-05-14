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

#if defined(RAJA_ENABLE_CUDA)

namespace RAJA
{

namespace cuda
{

__device__ __forceinline__ unsigned int getGlobalIdx_2D()
{
  return blockIdx.x * blockDim.z + threadIdx.z;
}

__device__ __forceinline__ unsigned int getGlobalIdx_3D()
{
  return blockIdx.x;
}

template <bool batched,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType> __global__
void forall_cuda_kernel_ND(LOOP_BODY loop_body,
                           const Iterator idx,
                           IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = static_cast<IndexType>(batched?getGlobalIdx_2D():getGlobalIdx_3D());
  if (ii >= length) { return; }
  body(idx[ii]);
}

template <bool batched, typename Iterable, typename LoopBody >
RAJA_INLINE void forallND(const dim3 gridSize,
                          const dim3 blockSize,
                          Iterable&& iter,
                          LoopBody&& loop_body)
{
  using LOOP_BODY = camp::decay<LoopBody>;
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  const auto func = forall_cuda_kernel_ND<batched, Iterator, LOOP_BODY, IndexType>;
  const Iterator begin = std::begin(iter);
  const Iterator end = std::end(iter);
  const IndexType length = std::distance(begin, end);
  if (length == 0) return;
  const size_t shmem = 0;
  const cudaStream_t stream = 0;
  void *args[] = {(void*)&std::forward<LoopBody>(loop_body), (void*)&begin, (void*)&length};
  cudaErrchk(cudaLaunchKernel((const void*)func, gridSize, blockSize, args, shmem, stream));
  RAJA::cuda::launch(stream);
}

}  // namespace cuda

}  // namespace RAJA

#endif  // defined(RAJA_ENABLE_CUDA)

#endif  // MFEM_RAJA_HPP
