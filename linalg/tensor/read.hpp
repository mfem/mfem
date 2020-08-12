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
#include "../general/backends.hpp"
#include "../dtensor.hpp"

namespace mfem
{

// Functions to read values (dofs or values at quadrature point)
// Non-tensor read
template<int P> MFEM_HOST_DEVICE inline
dTensor<P>&& Read(const DeviceTensor<2> &e_vec, const int e)
{
   dTensor<P> u;
   MFEM_FOREACH_THREAD(p,x,P)
   {
      u(p) = e_vec(p, e);
   }
   MFEM_SYNC_THREAD;
   return u;
}

// Non-tensor read with VDIM components
template<int P, int VDIM> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDIM>,P>&& Read(const DeviceTensor<3> &e_vec, const int e)
{
   Tensor<dTensor<VDIM>,P> u;
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(p,x,P)
      {
         u(p)(c) = e_vec(p, c, e);
      }
   }
   MFEM_SYNC_THREAD;
   return u;
}

// 3D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d,D1d>&& Read(const DeviceTensor<4> &e_vec, const int e)
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
   return u;
}

// 3D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDIM>,D1d,D1d,D1d>&& Read(const DeviceTensor<5> &e_vec, const int e)
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
   return u;
}

// 3D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDIM,VDIM>,D1d,D1d,D1d>&& Read(const DeviceTensor<6> &e_vec, const int e)
{
   Tensor<dTensor<VDIM,VDIM>,D1d,D1d,D1d> u;
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         for (int dz = 0; dz < D1d; dz++)
         {
            MFEM_FOREACH_THREAD(dy,y,D1d)
            {
               MFEM_FOREACH_THREAD(dx,x,D1d)
               {
                  u(dx,dy,dz)(h,w) = e_vec(dx,dy,dz,h,w,e);
               }
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return u;
}

// 2D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
dTensor<D1d,D1d>&& Read(const DeviceTensor<3> &e_vec, const int e)
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
   return u;
}

// 2D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDIM>,D1d,D1d>&& Read(const DeviceTensor<4> &e_vec, const int e)
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
   return u;
}

// 2D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDIM,VDIM>,D1d,D1d>&& Read(const DeviceTensor<4> &e_vec,
                                          const int e)
{
   Tensor<dTensor<VDIM,VDIM>,D1d,D1d> u;
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            MFEM_FOREACH_THREAD(dx,x,D1d)
            {
               u(dx,dy)(h,w) = e_vec(dx,dy,h,w,e);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return u;
}

// 1D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
dTensor<D1d>&& Read(const DeviceTensor<2> &e_vec, const int e)
{
   dTensor<D1d> u;
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      u(dx) = e_vec(dx,e);
   }
   MFEM_SYNC_THREAD;
   return u;
}

// 1D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDIM>,D1d>&& Read(const DeviceTensor<3> &e_vec, const int e)
{
   Tensor<dTensor<VDIM>,D1d> u;
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         u(dx)(c) = e_vec(dx,c,e);
      }
   }
   MFEM_SYNC_THREAD;
   return u;
}

// 1D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
Tensor<dTensor<VDIM>,D1d>&& Read(const DeviceTensor<3> &e_vec, const int e)
{
   Tensor<dTensor<VDIM>,D1d> u;
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            u(dx)(h,w) = e_vec(dx,h,w,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
   return u;
}


} // namespace mfem

#endif // MFEM_TENSOR_READ
