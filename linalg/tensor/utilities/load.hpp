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

#ifndef MFEM_TENSOR_UTIL_LOAD
#define MFEM_TENSOR_UTIL_LOAD

#include "../../../general/forall.hpp"

namespace mfem
{

/// Load row major values using 2d threads into a matrix of size nx x ny.
template <typename Matrix> MFEM_HOST_DEVICE
void load_with_2dthreads(const double *values, int nx, int ny, Matrix &mat)
{
   MFEM_FOREACH_THREAD(y,y,ny)
   {
      MFEM_FOREACH_THREAD(x,x,nx)
      {
         mat(x,y) = values[x+nx*y];
      }
   }
}

} // mfem namespace

#endif // MFEM_TENSOR_UTIL_LOAD
