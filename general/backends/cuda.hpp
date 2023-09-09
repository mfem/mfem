// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CUDA_HPP
#define MFEM_CUDA_HPP

#include "../../config/config.hpp"
#include "../error.hpp"
#include "../mem_manager.hpp"
#include <cassert>

// CUDA block size used by MFEM.
#define MFEM_CUDA_BLOCKS 256

#ifdef MFEM_USE_CUDA
namespace mfem
{
// Function used by the macro MFEM_GPU_CHECK.
void mfem_cuda_error(cudaError_t err, const char *expr, const char *func,
                     const char *file, int line);
} // namespace mfem
#endif // MFEM_USE_CUDA

#ifdef MFEM_USE_CUDA
#define MFEM_USE_CUDA_OR_HIP
#define MFEM_DEVICE __device__
#define MFEM_LAMBDA __host__
#define MFEM_HOST_DEVICE __host__ __device__
#define MFEM_DEVICE_SYNC MFEM_GPU_CHECK(cudaDeviceSynchronize())
#define MFEM_STREAM_SYNC MFEM_GPU_CHECK(cudaStreamSynchronize(0))
// Define a CUDA error check macro, MFEM_GPU_CHECK(x), where x returns/is of
// type 'cudaError_t'. This macro evaluates 'x' and raises an error if the
// result is not cudaSuccess.
#define MFEM_GPU_CHECK(x) \
   do \
   { \
      cudaError_t err = (x); \
      if (err != cudaSuccess) \
      { \
         mfem_cuda_error(err, #x, _MFEM_FUNC_NAME, __FILE__, __LINE__); \
      } \
   } \
   while (0)
#endif // MFEM_USE_CUDA

// Define the MFEM inner threading macros
#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
#define MFEM_SHARED __shared__

/*template <typename T, std::size_t UID>
inline __device__ T mfem_shared() { static __shared__ T smem_T; return smem_T; }*/

template <typename T, std::size_t UID>
__device__ static T mfem_shared()
{
   __shared__ uint8_t shMem alignas(alignof(T))[sizeof(T)];
   return *(reinterpret_cast<T*>(shMem));
}

template<typename T, std::size_t UID>
MFEM_DEVICE inline T& StaticSharedMemoryVariable()
{
   MFEM_SHARED uint8_t smem alignas(alignof(T))[sizeof(T)];
   return *(reinterpret_cast<T*>(smem));
}
#define MFEM_STATIC_SHARED_VAR(var, ...) \
__VA_ARGS__& var = StaticSharedMemoryVariable<__VA_ARGS__, __COUNTER__>()

template<typename T, typename U>
MFEM_DEVICE inline T& DynamicSharedMemoryVariable(U* &smem)
{
   T* base = reinterpret_cast<T*>(smem);
   return (smem += sizeof(T)/sizeof(U), *base);
}
#define MFEM_DYNAMIC_SHARED_VAR(var, smem, ...) \
__VA_ARGS__& var = DynamicSharedMemoryVariable<__VA_ARGS__>(smem)

#define MFEM_SYNC_THREAD __syncthreads()
#define MFEM_BLOCK_ID(k) blockIdx.k
#define MFEM_THREAD_ID(k) threadIdx.k
#define MFEM_THREAD_SIZE(k) blockDim.k
#define MFEM_FOREACH_THREAD(i,k,N) for(int i=threadIdx.k; i<N; i+=blockDim.k)
#endif

// 'double' atomicAdd implementation for previous versions of CUDA
#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
MFEM_DEVICE inline double atomicAdd(double *add, double val)
{
   unsigned long long int *ptr = (unsigned long long int *) add;
   unsigned long long int old = *ptr, reg;
   do
   {
      reg = old;
      old = atomicCAS(ptr, reg,
                      __double_as_longlong(val + __longlong_as_double(reg)));
   }
   while (reg != old);
   return __longlong_as_double(old);
}
#endif

namespace mfem
{

#ifdef MFEM_USE_CUDA
template <typename BODY> __global__ static
void CuKernel1D(const int N, BODY body)
{
   const int k = blockDim.x*blockIdx.x + threadIdx.x;
   if (k >= N) { return; }
   body(k);
}

template <typename BODY> __global__ static
void CuKernel2D(const int N, BODY body)
{
   const int k = blockIdx.x*blockDim.z + threadIdx.z;
   if (k >= N) { return; }
   body(k);
}

template <typename Tsmem = double, typename BODY> __global__ static
void CuKernel2DSmem(const int N, BODY body)
{
   const int k = blockIdx.x*blockDim.z + threadIdx.z;
   if (k >= N) { return; }
   extern __shared__ Tsmem smem[];
   body(k, smem);
}

template <typename Tsmem = double, typename BODY> __global__ static
void CuKernel2DGmem(const int N, BODY body, Tsmem *smem, const int smem_size)
{
   const int k = blockIdx.x*blockDim.z + threadIdx.z;
   if (k >= N) { return; }
   body(k, smem + smem_size*blockIdx.x);
}

template <typename BODY> __global__ static
void CuKernel3D(const int N, BODY body)
{
   for (int k = blockIdx.x; k < N; k += gridDim.x) { body(k); }
}

template <typename Tsmem = double, typename BODY> __global__ static
void CuKernel3DSmem(const int N, BODY body)
{
   extern __shared__ Tsmem sm[];
   for (int k = blockIdx.x; k < N; k += gridDim.x) { body(k, sm); }
}

template <typename Tsmem = double, typename BODY> __global__ static
void CuKernel3DGmem(const int N, BODY body, Tsmem *smem, const int smem_size)
{
   for (int k = blockIdx.x; k < N; k += gridDim.x)
   {
      body(k, smem + smem_size*blockIdx.x);
   }
}

template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
void CuWrap1D(const int N, DBODY &&d_body)
{
   if (N==0) { return; }
   const int GRID = (N+BLCK-1)/BLCK;
   CuKernel1D<<<GRID,BLCK>>>(N, d_body);
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename DBODY>
void CuWrap2D(const int N, DBODY &&d_body,
              const int X, const int Y, const int BZ)
{
   if (N==0) { return; }
   MFEM_VERIFY(BZ>0, "");
   const int GRID = (N+BZ-1)/BZ;
   const dim3 BLCK(X,Y,BZ);
   CuKernel2D<<<GRID,BLCK>>>(N,d_body);
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename Tsmem = double, typename DBODY>
void CuWrapSmem2D(const int N, DBODY &&d_body, const int smem_size,
                  const int X, const int Y, const int BZ, const int G)
{
   if (N==0) { return; }
   MFEM_VERIFY(BZ > 0, "");
   MFEM_VERIFY(G == 0, "Grid not implemented!");
   MFEM_VERIFY(smem_size > 0, "No Shared memory!");

   const dim3 BLCK(X,Y,BZ);

   if (smem_size*sizeof(Tsmem) < 64*1024) // V100, w/o extra config
   {
      const int GRID = (N+BZ-1)/BZ;
      CuKernel2DSmem<Tsmem><<<GRID, BLCK, sizeof(Tsmem)*smem_size>>>(N, d_body);
   }
   else
   {
      constexpr int SM = 80;
      const int GRID = SM;
      std::cout << "\033[33mFolding back to GLOBAL memory!" << std::endl;
      static Memory<Tsmem> smem(smem_size*sizeof(Tsmem)*GRID);
      smem.UseDevice(true);
      CuKernel2DGmem<Tsmem><<<GRID,BLCK>>>(N, d_body,
                                           smem.Write(MemoryClass::DEVICE, smem_size),
                                           smem_size);
   }
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename DBODY>
void CuWrap3D(const int N, DBODY &&d_body,
              const int X, const int Y, const int Z, const int G)
{
   if (N==0) { return; }
   const int GRID = G == 0 ? N : G;

   int TX = X, TY = Y, TZ = Z;
   if (X*Y*Z >= 1000) { TX = TY = TZ = 8; }
   const dim3 BLCK(TX,TY,TZ); // (X,Y,Z);
   CuKernel3D<<<GRID,BLCK>>>(N,d_body);
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename Tsmem = double, typename DBODY>
void CuWrapSmem3D(const int N, DBODY &&d_body, const int smem_size,
                  const int X, const int Y, const int Z, const int G)
{
   if (N==0) { return; }
   MFEM_VERIFY(smem_size > 0, "No Shared memory!");

   int TX = X, TY = Y, TZ = Z;
   if (X*Y*Z >= 1000) { TX = TY = TZ = 8; }
   const dim3 BLCK(TX,TY,TZ);

   if (smem_size*sizeof(Tsmem) < 64*1024) // V100, w/o extra config
   {
      //std::cout << "\033[33mStandard kernel!\033[m" << std::endl;
      const int NB = X*Y*Z < 16 ? 4 : 1;
      const int GRID_X = (N + NB - 1) / NB;
      const int GRID = G == 0 ? GRID_X : G;
      CuKernel3DSmem<Tsmem><<<GRID, BLCK, sizeof(Tsmem)*smem_size>>>(N, d_body);
   }
   else
   {
      constexpr int SM = 80;
      const int GRID = G == 0 ? SM : G;
      //std::cout << "\033[33mFolding back to GLOBAL memory" << std::endl;
      Memory<Tsmem> smem(smem_size*GRID);
      smem.UseDevice(true);
      CuKernel3DGmem<Tsmem><<<GRID,BLCK>>>(N, d_body,
                                           smem.Write(MemoryClass::DEVICE, smem_size),
                                           smem_size);
      smem.Delete();
   }
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <int Dim> struct CuWrap;
template <int Dim, typename Tsmem> struct CuWrapSmem;

template <>
struct CuWrap<1>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      CuWrap1D<BLCK>(N, d_body);
   }
};

template <>
struct CuWrap<2>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      CuWrap2D(N, d_body, X, Y, Z);
   }
};

template <typename Tsmem>
struct CuWrapSmem<2,Tsmem>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body, const int smem_size,
                   const int X, const int Y, const int Z, const int G)
   {
      CuWrapSmem2D<Tsmem>(N, d_body, smem_size, X, Y, Z, G);
   }
};

template <>
struct CuWrap<3>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body,
                   const int X, const int Y, const int Z, const int G)
   {
      CuWrap3D(N, d_body, X, Y, Z, G);
   }
};

template <typename Tsmem>
struct CuWrapSmem<3,Tsmem>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body, const int smem_size,
                   const int X, const int Y, const int Z, const int G)
   {
      CuWrapSmem3D<Tsmem>(N, d_body, smem_size, X, Y, Z, G);
   }
};

#endif // MFEM_USE_CUDA

/// Allocates device memory and returns destination ptr.
void* CuMemAlloc(void **d_ptr, size_t bytes);

/// Allocates managed device memory
void* CuMallocManaged(void **d_ptr, size_t bytes);

/// Allocates page-locked (pinned) host memory
void* CuMemAllocHostPinned(void **ptr, size_t bytes);

/// Frees device memory and returns destination ptr.
void* CuMemFree(void *d_ptr);

/// Frees page-locked (pinned) host memory and returns destination ptr.
void* CuMemFreeHostPinned(void *ptr);

/// Copies memory from Host to Device and returns destination ptr.
void* CuMemcpyHtoD(void *d_dst, const void *h_src, size_t bytes);

/// Copies memory from Host to Device and returns destination ptr.
void* CuMemcpyHtoDAsync(void *d_dst, const void *h_src, size_t bytes);

/// Copies memory from Device to Device
void* CuMemcpyDtoD(void *d_dst, const void *d_src, size_t bytes);

/// Copies memory from Device to Device
void* CuMemcpyDtoDAsync(void *d_dst, const void *d_src, size_t bytes);

/// Copies memory from Device to Host
void* CuMemcpyDtoH(void *h_dst, const void *d_src, size_t bytes);

/// Copies memory from Device to Host
void* CuMemcpyDtoHAsync(void *h_dst, const void *d_src, size_t bytes);

/// Check the error code returned by cudaGetLastError(), aborting on error.
void CuCheckLastError();

/// Get the number of CUDA devices
int CuGetDeviceCount();

} // namespace mfem

#endif // MFEM_CUDA_HPP
