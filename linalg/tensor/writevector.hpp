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

#ifndef MFEM_TENSOR_WRITE_VEC
#define MFEM_TENSOR_WRITE_VEC

#include "tensor.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"

namespace mfem
{

// Non-tensor and 1D write with VDIM components
template<int P, int VDIM> MFEM_HOST_DEVICE inline
void WriteVector(const StaticTensor<dTensor<VDIM>,P> &u, const int e,
                 DeviceTensor<3> &e_vec)
{
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(p,x,P)
      {
         e_vec(p, c, e) = u(p)(c);
      }
   }
   MFEM_SYNC_THREAD;
}

// 3D tensor write with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void WriteVector(const StaticTensor<dTensor<VDIM>,D1d,D1d,D1d> &u, const int e,
                 DeviceTensor<5> &e_vec)
{
   for (int c = 0; c < VDIM; c++)
   {
      for (int dz = 0; dz < D1d; dz++)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            MFEM_FOREACH_THREAD(dx,x,D1d)
            {
               e_vec(dx,dy,dz,c,e) = u(dx,dy,dz)(c);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D tensor write with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void WriteVector(const StaticTensor<dTensor<VDIM>,D1d,D1d> &u, const int e,
                 DeviceTensor<4> &e_vec)
{
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            e_vec(dx,dy,c,e) = u(dx,dy)(c);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// Non-tensor and 1D write with VDIM components
template<int P, int VDIM> MFEM_HOST_DEVICE inline
void WriteAddVector(const StaticTensor<dTensor<VDIM>,P> &u, const int e,
                    DeviceTensor<3> &e_vec)
{
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(p,x,P)
      {
         e_vec(p, c, e) += u(p)(c);
      }
   }
   MFEM_SYNC_THREAD;
}

// 3D tensor write with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void WriteAddVector(const StaticTensor<dTensor<VDIM>,D1d,D1d,D1d> &u, const int e,
                    DeviceTensor<5> &e_vec)
{
   for (int c = 0; c < VDIM; c++)
   {
      for (int dz = 0; dz < D1d; dz++)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            MFEM_FOREACH_THREAD(dx,x,D1d)
            {
               e_vec(dx,dy,dz,c,e) += u(dx,dy,dz)(c);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D tensor write with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void WriteAddVector(const StaticTensor<dTensor<VDIM>,D1d,D1d> &u, const int e,
                    DeviceTensor<4> &e_vec)
{
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            e_vec(dx,dy,c,e) += u(dx,dy)(c);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

} // namespace mfem

#endif // MFEM_TENSOR_WRITE_VEC
