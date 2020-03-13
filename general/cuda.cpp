// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "cuda.hpp"
#include "globals.hpp"

namespace mfem
{

// Internal debug option, useful for tracking CUDA allocations, deallocations
// and transfers.
// #define MFEM_TRACK_CUDA_MEM

#ifdef MFEM_USE_CUDA
void mfem_cuda_error(cudaError_t err, const char *expr, const char *func,
                     const char *file, int line)
{
   mfem::err << "\n\nCUDA error: (" << expr << ") failed with error:\n --> "
             << cudaGetErrorString(err)
             << "\n ... in function: " << func
             << "\n ... in file: " << file << ':' << line << '\n';
   mfem_error();
}
#endif

void* CuMemAlloc(void** dptr, size_t bytes)
{
#ifdef MFEM_USE_CUDA
#ifdef MFEM_TRACK_CUDA_MEM
   mfem::out << "CuMemAlloc(): allocating " << bytes << " bytes ... "
             << std::flush;
#endif
   MFEM_GPU_CHECK(cudaMalloc(dptr, bytes));
#ifdef MFEM_TRACK_CUDA_MEM
   mfem::out << "done: " << *dptr << std::endl;
#endif
#endif
   return *dptr;
}

void* CuMallocManaged(void** dptr, size_t bytes)
{
#ifdef MFEM_USE_CUDA
#ifdef MFEM_TRACK_CUDA_MEM
   mfem::out << "CuMallocManaged(): allocating " << bytes << " bytes ... "
             << std::flush;
#endif
   MFEM_GPU_CHECK(cudaMallocManaged(dptr, bytes));
#ifdef MFEM_TRACK_CUDA_MEM
   mfem::out << "done: " << *dptr << std::endl;
#endif
#endif
   return *dptr;
}

void* CuMemFree(void *dptr)
{
#ifdef MFEM_USE_CUDA
#ifdef MFEM_TRACK_CUDA_MEM
   mfem::out << "CuMemFree(): deallocating memory @ " << dptr << " ... "
             << std::flush;
#endif
   MFEM_GPU_CHECK(cudaFree(dptr));
#ifdef MFEM_TRACK_CUDA_MEM
   mfem::out << "done." << std::endl;
#endif
#endif
   return dptr;
}

void* CuMemcpyHtoD(void* dst, const void* src, size_t bytes)
{
#ifdef MFEM_USE_CUDA
#ifdef MFEM_TRACK_CUDA_MEM
   mfem::out << "CuMemcpyHtoD(): copying " << bytes << " bytes from "
             << src << " to " << dst << " ... " << std::flush;
#endif
   MFEM_GPU_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
#ifdef MFEM_TRACK_CUDA_MEM
   mfem::out << "done." << std::endl;
#endif
#endif
   return dst;
}

void* CuMemcpyHtoDAsync(void* dst, const void* src, size_t bytes)
{
#ifdef MFEM_USE_CUDA
   MFEM_GPU_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice));
#endif
   return dst;
}

void* CuMemcpyDtoD(void *dst, const void *src, size_t bytes)
{
#ifdef MFEM_USE_CUDA
#ifdef MFEM_TRACK_CUDA_MEM
   mfem::out << "CuMemcpyDtoD(): copying " << bytes << " bytes from "
             << src << " to " << dst << " ... " << std::flush;
#endif
   MFEM_GPU_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
#ifdef MFEM_TRACK_CUDA_MEM
   mfem::out << "done." << std::endl;
#endif
#endif
   return dst;
}

void* CuMemcpyDtoDAsync(void* dst, const void *src, size_t bytes)
{
#ifdef MFEM_USE_CUDA
   MFEM_GPU_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice));
#endif
   return dst;
}

void* CuMemcpyDtoH(void *dst, const void *src, size_t bytes)
{
#ifdef MFEM_USE_CUDA
#ifdef MFEM_TRACK_CUDA_MEM
   mfem::out << "CuMemcpyDtoH(): copying " << bytes << " bytes from "
             << src << " to " << dst << " ... " << std::flush;
#endif
   MFEM_GPU_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
#ifdef MFEM_TRACK_CUDA_MEM
   mfem::out << "done." << std::endl;
#endif
#endif
   return dst;
}

void* CuMemcpyDtoHAsync(void *dst, const void *src, size_t bytes)
{
#ifdef MFEM_USE_CUDA
   MFEM_GPU_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost));
#endif
   return dst;
}

void CuCheckLastError()
{
#ifdef MFEM_USE_CUDA
   MFEM_GPU_CHECK(cudaGetLastError());
#endif
}

int CuGetDeviceCount()
{
   int num_gpus = -1;
#ifdef MFEM_USE_CUDA
   MFEM_GPU_CHECK(cudaGetDeviceCount(&num_gpus));
#endif
   return num_gpus;
}

} // namespace mfem
