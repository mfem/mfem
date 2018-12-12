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
#define CU_STUB(...) return __VA_ARGS__

#else
#define CU_STUB(...) (assert(false),return 0;);
#endif

// *****************************************************************************
// * Allocates device memory
// *****************************************************************************
int okMemAlloc(void** dptr, size_t bytes)
{
   //dbg("\033[31;7mokMemAlloc");
   CU_STUB(cuMemAlloc((CUdeviceptr*)dptr, bytes));
}

// *****************************************************************************
// * Frees device memory
// *****************************************************************************
int okMemFree(void *dptr) {
   //dbg("\033[31;7mokMemFree");
   CU_STUB(cuMemFree((CUdeviceptr)dptr));
}

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
int okMemcpyHtoD(void* dst, const void* src, size_t bytes)
{
   //dbg("\033[31;7mHtoD");
   CU_STUB(cuMemcpyHtoD((CUdeviceptr)dst, src, bytes));
}

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
int okMemcpyHtoDAsync(void* dst, const void* src, size_t bytes, void *s)
{
   //dbg("\033[31;7mHtoDAsync");
   CU_STUB(cuMemcpyHtoDAsync((CUdeviceptr)dst, src, bytes, (CUstream)s));
}

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
int okMemcpyDtoD(void* dst, void* src, size_t bytes)
{
   //dbg("\033[31;7mDtoD");
   CU_STUB(cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, bytes));
}

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
int okMemcpyDtoDAsync(void* dst, void* src, size_t bytes, void *s)
{
   //dbg("\033[31;7mDtoDAsync");
   CU_STUB(cuMemcpyDtoDAsync((CUdeviceptr)dst,
                             (CUdeviceptr)src, bytes, (CUstream)s));
}

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
int okMemcpyDtoH(void* dst, const void* src, size_t bytes)
{
   //dbg("\033[31;7mDtoH");
   CU_STUB(cuMemcpyDtoH(dst, (CUdeviceptr)src, bytes));
}

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
int okMemcpyDtoHAsync(void* dst, void* src, size_t bytes, void *s)
{
   //dbg("\033[31;7mDtoHAsync");
   CU_STUB(cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, bytes, (CUstream)s));
}

// *****************************************************************************
} // namespace mfem
