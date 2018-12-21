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

#include "okina.hpp"

// *****************************************************************************
namespace mfem
{

// *****************************************************************************
#ifdef __NVCC__
#define CU_STUB(dst,...) __VA_ARGS__; return dst;

#else
#define CU_STUB(...) {                                                  \
      printf("No CUDA available!\n");                                   \
      fflush(0);                                                        \
      exit(-1);                                                         \
      return (void*)NULL;                                               \
   }
#endif

// *****************************************************************************
// * Allocates device memory
// *****************************************************************************
void* cuMemAlloc(void** dptr, size_t bytes)
{
   CU_STUB(*dptr,::cuMemAlloc((CUdeviceptr*)dptr, bytes););
}

// *****************************************************************************
// * Frees device memory
// *****************************************************************************
void* cuMemFree(void *dptr)
{
   CU_STUB(dptr,::cuMemFree((CUdeviceptr)dptr));
}

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
void* cuMemcpyHtoD(void* dst, const void* src, size_t bytes)
{
   CU_STUB(dst,::cuMemcpyHtoD((CUdeviceptr)dst, src, bytes));
}

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
void* cuMemcpyHtoDAsync(void* dst, const void* src, size_t bytes, void *s)
{
   CU_STUB(dst,::cuMemcpyHtoDAsync((CUdeviceptr)dst, src, bytes, (CUstream)s));
}

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
void* cuMemcpyDtoD(void* dst, void* src, size_t bytes)
{
   CU_STUB(dst,::cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, bytes));
}

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
void* cuMemcpyDtoDAsync(void* dst, void* src, size_t bytes, void *s)
{
   CU_STUB(dst,::cuMemcpyDtoDAsync((CUdeviceptr)dst,
                                   (CUdeviceptr)src,
                                   bytes,
                                   (CUstream)s));
}

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
void* cuMemcpyDtoH(void* dst, const void* src, size_t bytes)
{
   CU_STUB(dst,::cuMemcpyDtoH(dst, (CUdeviceptr)src, bytes));
}

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
void* cuMemcpyDtoHAsync(void* dst, void* src, size_t bytes, void *s)
{
   CU_STUB(dst,::cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, bytes, (CUstream)s));
}

// *****************************************************************************
} // namespace mfem
