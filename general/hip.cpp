// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "backends.hpp"
#include "globals.hpp"

namespace mfem
{

// Internal debug option, useful for tracking HIP allocations, deallocations
// and transfers.
// #define MFEM_TRACK_HIP_MEM

#ifdef MFEM_USE_HIP
void mfem_hip_error(hipError_t err, const char *expr, const char *func,
                    const char *file, int line)
{
   mfem::err << "\n\nHIP error: (" << expr << ") failed with error:\n --> "
             << hipGetErrorString(err) << " [code: " << (int)err << ']'
             << "\n ... in function: " << func
             << "\n ... in file: " << file << ':' << line << '\n';
   mfem_error();
}
#endif

void* HipMemAlloc(void** dptr, size_t bytes)
{
#ifdef MFEM_USE_HIP
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "HipMemAlloc(): allocating " << bytes << " bytes ... "
             << std::flush;
#endif
   MFEM_GPU_CHECK(hipMalloc(dptr, bytes));
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "done: " << *dptr << std::endl;
#endif
#endif
   return *dptr;
}

void* HipMallocManaged(void** dptr, size_t bytes)
{
#ifdef MFEM_USE_HIP
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "HipMallocManaged(): allocating " << bytes << " bytes ... "
             << std::flush;
#endif
   MFEM_GPU_CHECK(hipMallocManaged(dptr, bytes));
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "done: " << *dptr << std::endl;
#endif
#endif
   return *dptr;
}

void* HipMemAllocHostPinned(void** ptr, size_t bytes)
{
#ifdef MFEM_USE_HIP
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "HipMemAllocHostPinned(): allocating " << bytes << " bytes ... "
             << std::flush;
#endif
   MFEM_GPU_CHECK(hipHostMalloc(ptr, bytes, hipHostMallocDefault));
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "done: " << *ptr << std::endl;
#endif
#endif
   return *ptr;
}

void* HipMemFree(void *dptr)
{
#ifdef MFEM_USE_HIP
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "HipMemFree(): deallocating memory @ " << dptr << " ... "
             << std::flush;
#endif
   MFEM_GPU_CHECK(hipFree(dptr));
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "done." << std::endl;
#endif
#endif
   return dptr;
}

void* HipMemFreeHostPinned(void *ptr)
{
#ifdef MFEM_USE_HIP
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "HipMemFreeHostPinned(): deallocating memory @ " << ptr << " ... "
             << std::flush;
#endif
   MFEM_GPU_CHECK(hipHostFree(ptr));
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "done." << std::endl;
#endif
#endif
   return ptr;
}

void* HipMemcpyHtoD(void* dst, const void* src, size_t bytes)
{
#ifdef MFEM_USE_HIP
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "HipMemcpyHtoD(): copying " << bytes << " bytes from "
             << src << " to " << dst << " ... " << std::flush;
#endif
   MFEM_GPU_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "done." << std::endl;
#endif
#endif
   return dst;
}

void* HipMemcpyHtoDAsync(void* dst, const void* src, size_t bytes)
{
#ifdef MFEM_USE_HIP
   MFEM_GPU_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice));
#endif
   return dst;
}

void* HipMemcpyDtoD(void *dst, const void *src, size_t bytes)
{
#ifdef MFEM_USE_HIP
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "HipMemcpyDtoD(): copying " << bytes << " bytes from "
             << src << " to " << dst << " ... " << std::flush;
#endif
   MFEM_GPU_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "done." << std::endl;
#endif
#endif
   return dst;
}

void* HipMemcpyDtoDAsync(void* dst, const void *src, size_t bytes)
{
#ifdef MFEM_USE_HIP
   MFEM_GPU_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice));
#endif
   return dst;
}

void* HipMemcpyDtoH(void *dst, const void *src, size_t bytes)
{
#ifdef MFEM_USE_HIP
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "HipMemcpyDtoH(): copying " << bytes << " bytes from "
             << src << " to " << dst << " ... " << std::flush;
#endif
   MFEM_GPU_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
#ifdef MFEM_TRACK_HIP_MEM
   mfem::out << "done." << std::endl;
#endif
#endif
   return dst;
}

void* HipMemcpyDtoHAsync(void *dst, const void *src, size_t bytes)
{
#ifdef MFEM_USE_HIP
   MFEM_GPU_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost));
#endif
   return dst;
}

void HipCheckLastError()
{
#ifdef MFEM_USE_HIP
   MFEM_GPU_CHECK(hipGetLastError());
#endif
}

int HipGetDeviceCount()
{
   int num_gpus = -1;
#ifdef MFEM_USE_HIP
   MFEM_GPU_CHECK(hipGetDeviceCount(&num_gpus));
#endif
   return num_gpus;
}

} // namespace mfem
