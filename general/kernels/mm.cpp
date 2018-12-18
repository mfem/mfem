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

#include "../okina.hpp"

// *****************************************************************************
namespace mfem
{

// ******************************************************************************
void* kH2D(void *dest, const void *src, size_t bytes, const bool async)
{
   GET_ADRS(src);
   GET_ADRS(dest);
   if (not async) { cuMemcpyHtoD(d_dest, d_src, bytes); }
   else { cuMemcpyHtoDAsync(d_dest, d_src, bytes, config::Stream()); }
   return dest;
}

// *****************************************************************************
void* kD2H(void *dest, const void *src, size_t bytes, const bool async)
{
   GET_ADRS(src);
   GET_ADRS(dest);
   if (not async) { cuMemcpyDtoH(d_dest, d_src, bytes); }
   else { cuMemcpyDtoHAsync(d_dest, d_src, bytes, config::Stream()); }
   return dest;
}

// *****************************************************************************
void* kD2D(void *dest, const void *src, size_t bytes, const bool async)
{
   GET_ADRS(src);
   GET_ADRS(dest);
   if (not async) { cuMemcpyDtoD(d_dest, d_src, bytes); }
   else { cuMemcpyDtoDAsync(d_dest, d_src, bytes, config::Stream()); }
   return dest;
}

// *****************************************************************************
} // namespace mfem
