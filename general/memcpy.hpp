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

#ifndef MFEM_MEMCPY_HPP
#define MFEM_MEMCPY_HPP

#include <cstring>
#include "dbg.hpp"

// *****************************************************************************
// * memcpys
// ***************************************************************************
struct memcpy{
  static void* H2H(void *dest, const void *src, std::size_t bytes, const bool async=false) {
    dbg(">\033[m");
    if (bytes==0) return dest;
    assert(src); assert(dest);
    std::memcpy(dest,src,bytes);
    return dest;
  }

  // *************************************************************************
  static void* H2D(void *dest, const void *src, std::size_t bytes, const bool async=false) {
    dbg(">\033[m");
    if (bytes==0) return dest;
    assert(src); assert(dest);
    if (!cfg::Get().Cuda()) return std::memcpy(dest,src,bytes);
#ifdef __NVCC__
    if (!cfg::Get().Uvm())
      checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)dest,src,bytes));
    else checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes));
#endif
    return dest;
  }

  // ***************************************************************************
   static void* D2H(void *dest, const void *src, std::size_t bytes, const bool async=false) {
    dbg("<\033[m");
    if (bytes==0) return dest;
    assert(src); assert(dest);
    if (!cfg::Get().Cuda()) return std::memcpy(dest,src,bytes);
#ifdef __NVCC__
    if (!cfg::Get().Uvm())
      checkCudaErrors(cuMemcpyDtoH(dest,(CUdeviceptr)src,bytes));
    else checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes));
#endif
    return dest;
  }
  
  // ***************************************************************************
  static void* D2D(void *dest, const void *src, std::size_t bytes, const bool async=false) {
    dbg("<\033[m");
    if (bytes==0) return dest;
    assert(src); assert(dest);
    if (!cfg::Get().Cuda()) return std::memcpy(dest,src,bytes);
#ifdef __NVCC__
    if (!cfg::Get().Uvm()){
      if (!async)
        checkCudaErrors(cuMemcpyDtoD((CUdeviceptr)dest,(CUdeviceptr)src,bytes));
      else{
        const CUstream s = *cfg::Get().Stream();
        checkCudaErrors(cuMemcpyDtoDAsync((CUdeviceptr)dest,(CUdeviceptr)src,bytes,s));
      }
    } else checkCudaErrors(cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes));
#endif
    return dest;
  }
};

#endif // MFEM_MEMCPY_HPP
