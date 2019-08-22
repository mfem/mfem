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

#include "hip.hpp"
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
             << hipGetErrorString(err)
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
#ifdef MFEM_TRACK_HPI_MEM
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

} // namespace mfem
