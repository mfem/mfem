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

#ifndef MFEM_TENSOR_READ_VEC
#define MFEM_TENSOR_READ_VEC

#include "tensor.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"
#include <utility>

namespace mfem
{

// Non-tensor read with VDIM components
template<int D, int VDIM> MFEM_HOST_DEVICE inline
const Tensor<dTensor<VDIM>,D>&& ReadVector(const DeviceTensor<3> &e_vec, const int e)
{
   Tensor<dTensor<VDIM>,D> u;
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(d,x,D)
      {
         u(d)(c) = e_vec(d, c, e);
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u);
}

// 3D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
const Tensor<dTensor<VDIM>,D1d,D1d,D1d>&& ReadVector(const DeviceTensor<5> &e_vec,
                                                     const int e)
{
   Tensor<dTensor<VDIM>,D1d,D1d,D1d> u;
   for (int c = 0; c < VDIM; c++)
   {
      for (int dz = 0; dz < D1d; dz++)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            MFEM_FOREACH_THREAD(dx,x,D1d)
            {
               u(dx,dy,dz)(c) = e_vec(dx,dy,dz,c,e);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u);
}

// 2D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
const Tensor<dTensor<VDIM>,D1d,D1d>&& ReadVector(const DeviceTensor<4> &e_vec,
                                                 const int e)
{
   Tensor<dTensor<VDIM>,D1d,D1d> u;
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            u(dx,dy)(c) = e_vec(dx,dy,c,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u);
}

} // namespace mfem

#endif // MFEM_TENSOR_READ_VEC
