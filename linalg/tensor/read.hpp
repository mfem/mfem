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

#ifndef MFEM_TENSOR_READ
#define MFEM_TENSOR_READ

#include "tensor.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"
#include <utility>

namespace mfem
{

// Functions to read values (dofs or values at quadrature point)
// Non-tensor read and 1D
template<int D> MFEM_HOST_DEVICE inline
const dTensor<D>&& Read(const DeviceTensor<2> &e_vec, const int e)
{
   dTensor<D> u;
   MFEM_FOREACH_THREAD(d,x,D)
   {
      u(d) = e_vec(d, e);
   }
   MFEM_SYNC_THREAD;
   return std::move(u);
}

// 3D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
const dTensor<D1d,D1d,D1d>&& Read(const DeviceTensor<4> &e_vec, const int e)
{
   dTensor<D1d,D1d,D1d> u;
   for (int dz = 0; dz < D1d; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            u(dx,dy,dz) = e_vec(dx,dy,dz,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u);
}

// 2D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
const dTensor<D1d,D1d>&& Read(const DeviceTensor<3> &e_vec, const int e)
{
   dTensor<D1d,D1d> u;
   MFEM_FOREACH_THREAD(dy,y,D1d)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         u(dx,dy) = e_vec(dx,dy,e);
      }
   }
   MFEM_SYNC_THREAD;
   return std::move(u);
}

} // namespace mfem

#endif // MFEM_TENSOR_READ
