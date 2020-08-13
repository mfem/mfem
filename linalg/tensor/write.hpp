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

#ifndef MFEM_TENSOR_WRITE
#define MFEM_TENSOR_WRITE

#include "tensor.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"

namespace mfem
{

// Functions to write values (dofs or values at quadrature point)
// Non-tensor write
template<int P> MFEM_HOST_DEVICE inline
void Write(const dTensor<P> &u, const int e, DeviceTensor<2> &e_vec)
{
   MFEM_FOREACH_THREAD(p,x,P)
   {
      e_vec(p, e) = u(p);
   }
   MFEM_SYNC_THREAD;
}

// 3D tensor write
template <int D1d> MFEM_HOST_DEVICE inline
void Write(const dTensor<D1d,D1d,D1d> &u, const int e,
           DeviceTensor<4> &e_vec)
{
   for (int dz = 0; dz < D1d; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            e_vec(dx,dy,dz,e) = u(dx,dy,dz);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D tensor write
template <int D1d> MFEM_HOST_DEVICE inline
void Write(const dTensor<D1d,D1d> &u, const int e,
           DeviceTensor<3> &e_vec)
{
   MFEM_FOREACH_THREAD(dy,y,D1d)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         e_vec(dx,dy,e) = u(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
}


// Functions to write values (dofs or values at quadrature point)
// Non-tensor and 1D write
template<int P> MFEM_HOST_DEVICE inline
void WriteAdd(const dTensor<P> &u, const int e, DeviceTensor<2> &e_vec)
{
   MFEM_FOREACH_THREAD(p,x,P)
   {
      e_vec(p, e) += u(p);
   }
   MFEM_SYNC_THREAD;
}

// 3D tensor write
template <int D1d> MFEM_HOST_DEVICE inline
void WriteAdd(const dTensor<D1d,D1d,D1d> &u, const int e,
              DeviceTensor<4> &e_vec)
{
   for (int dz = 0; dz < D1d; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            e_vec(dx,dy,dz,e) += u(dx,dy,dz);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D tensor write
template <int D1d> MFEM_HOST_DEVICE inline
void WriteAdd(const dTensor<D1d,D1d> &u, const int e,
              DeviceTensor<3> &e_vec)
{
   MFEM_FOREACH_THREAD(dy,y,D1d)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         e_vec(dx,dy,e) += u(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
}

} // namespace mfem

#endif // MFEM_TENSOR_WRITE
