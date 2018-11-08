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

#ifndef MFEM_CU_STUB_HPP
#define MFEM_CU_STUB_HPP

namespace mfem
{

// *****************************************************************************
// * Allocates device memory
// *****************************************************************************
int okMemAlloc(void**, size_t);

// *****************************************************************************
// * Frees device memory
// *****************************************************************************
int okMemFree(void*);
   
// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
int okMemcpyHtoD(void*, const void*, size_t);

// *****************************************************************************
// * Copies memory from Host to Device
// *****************************************************************************
int okMemcpyHtoDAsync(void*, const void*, size_t, void*);

// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
int okMemcpyDtoD(void*, void*, size_t);
   
// *****************************************************************************
// * Copies memory from Device to Device
// *****************************************************************************
int okMemcpyDtoDAsync(void*, void*, size_t, void*);

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
int okMemcpyDtoH(void*, void*, size_t);

// *****************************************************************************
// * Copies memory from Device to Host
// *****************************************************************************
int okMemcpyDtoHAsync(void*, void*, size_t, void*);

} // namespace mfem

#endif // MFEM_CU_STUB_HPP
