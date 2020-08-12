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
#include "../general/backends.hpp"
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

// Non-tensor read with VDIM components
template<int P, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM>,P> &u, const int e,
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

// Non-tensor read with VDIMxVDIM components
template<int P, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM,VDIM>,P> &u, const int e,
           DeviceTensor<4> &e_vec)
{
   for (int w = 0; w < VDIM; w++)
   {   
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(p,x,P)
         {
            e_vec(p, h, w, e) = u(p)(h,w);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 3D tensor read
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

// 3D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM>,D1d,D1d,D1d> &u, const int e,
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

// 3D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM,VDIM>,D1d,D1d,D1d> &u, const int e,
           DeviceTensor<6> &e_vec)
{
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
                  e_vec(dx,dy,dz,h,w,e) = u(dx,dy,dz)(h,w);
               }
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D tensor read
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

// 2D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM>,D1d,D1d> &u, const int e,
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

// 2D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM,VDIM>,D1d,D1d> &u, const int e,
           DeviceTensor<4> &e_vec)
{
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            MFEM_FOREACH_THREAD(dx,x,D1d)
            {
               e_vec(dx,dy,h,w,e) = u(dx,dy)(h,w);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 1D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
void Write(const dTensor<D1d> &u, const int e,
           DeviceTensor<2> &e_vec)
{
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      e_vec(dx,e) = u(dx);
   }
   MFEM_SYNC_THREAD;
}

// 1D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM>,D1d> &u, const int e,
           DeviceTensor<3> &e_vec)
{
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         e_vec(dx,c,e) = u(dx)(c);
      }
   }
   MFEM_SYNC_THREAD;
}

// 1D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM>,D1d> &u, const int e,
           DeviceTensor<3> &e_vec)
{
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            e_vec(dx,h,w,e) = u(dx)(h,w);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// Functions to write values (dofs or values at quadrature point)
// Non-tensor write
template<int P> MFEM_HOST_DEVICE inline
void WriteAdd(const dTensor<P> &u, const int e, DeviceTensor<2> &e_vec)
{
   MFEM_FOREACH_THREAD(p,x,P)
   {
      e_vec(p, e) += u(p);
   }
   MFEM_SYNC_THREAD;
}

// Non-tensor read with VDIM components
template<int P, int VDIM> MFEM_HOST_DEVICE inline
void WriteAdd(const Tensor<dTensor<VDIM>,P> &u, const int e,
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

// Non-tensor read with VDIMxVDIM components
template<int P, int VDIM> MFEM_HOST_DEVICE inline
void WriteAdd(const Tensor<dTensor<VDIM,VDIM>,P> &u, const int e,
              DeviceTensor<4> &e_vec)
{
   for (int w = 0; w < VDIM; w++)
   {   
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(p,x,P)
         {
            e_vec(p, h, w, e) += u(p)(h,w);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 3D tensor read
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

// 3D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void WriteAdd(const Tensor<dTensor<VDIM>,D1d,D1d,D1d> &u, const int e,
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

// 3D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void WriteAdd(const Tensor<dTensor<VDIM,VDIM>,D1d,D1d,D1d> &u, const int e,
              DeviceTensor<6> &e_vec)
{
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
                  e_vec(dx,dy,dz,h,w,e) += u(dx,dy,dz)(h,w);
               }
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D tensor read
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

// 2D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void WriteAdd(const Tensor<dTensor<VDIM>,D1d,D1d> &u, const int e,
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

// 2D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void WriteAdd(const Tensor<dTensor<VDIM,VDIM>,D1d,D1d> &u, const int e,
              DeviceTensor<4> &e_vec)
{
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            MFEM_FOREACH_THREAD(dx,x,D1d)
            {
               e_vec(dx,dy,h,w,e) += u(dx,dy)(h,w);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 1D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
void WriteAdd(const dTensor<D1d> &u, const int e,
              DeviceTensor<2> &e_vec)
{
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      e_vec(dx,e) += u(dx);
   }
   MFEM_SYNC_THREAD;
}

// 1D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void WriteAdd(const Tensor<dTensor<VDIM>,D1d> &u, const int e,
              DeviceTensor<3> &e_vec)
{
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         e_vec(dx,c,e) += u(dx)(c);
      }
   }
   MFEM_SYNC_THREAD;
}

// 1D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void WriteAdd(const Tensor<dTensor<VDIM>,D1d> &u, const int e,
              DeviceTensor<3> &e_vec)
{
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            e_vec(dx,h,w,e) += u(dx)(h,w);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

} // namespace mfem

#endif // MFEM_TENSOR_WRITE
