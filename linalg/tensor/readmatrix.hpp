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

#ifndef MFEM_TENSOR_READ_MATRIX
#define MFEM_TENSOR_READ_MATRIX

#include "tensor.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"
#include <utility>

namespace mfem
{

// Functions to read a matrix, for example the dofs to quad matrix
template<int Q, int P> MFEM_HOST_DEVICE inline
const dTensor<Q,P> ReadMatrix(const DeviceBasis<Q,P> &d_B)
{
   dTensor<Q,P> s_B;
   for (int p = 0; p < P; p++)
   {
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         s_B(q,p) = d_B(q,p);
      }
   }
   MFEM_SYNC_THREAD;
   return s_B;
}

// Functions to read dofs to derivatives matrix
template<int P, int Q, int Dim> MFEM_HOST_DEVICE inline
const StaticTensor<dTensor<Dim>,Q,P> ReadMatrix(const DeviceTensor<3> &d_G)
{
   StaticTensor<dTensor<Dim>,Q,P> s_G;
   for (int p = 0; p < P; p++)
   {
      for (int s = 0; s < Dim; s++)
      {      
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            s_G(q,p)(s) = d_G(q,s,p);
         }
      }
   }
   MFEM_SYNC_THREAD;
   return s_G;
}

// Non-tensor and 1D read with VDIMxVDIM components
template<int D, int VDIM> MFEM_HOST_DEVICE inline
const StaticTensor<dTensor<VDIM,VDIM>,D> ReadMatrix(const DeviceTensor<4> &e_vec,
                                                    const int e)
{
   StaticTensor<dTensor<VDIM,VDIM>,D> u;
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(d,x,D)
         {
            u(d)(h,w) = e_vec(d, h, w, e);
         }
      }
   }
   MFEM_SYNC_THREAD;
   return u;
}

// Non-tensor and 1D read with VDIMxVDIM components
template<int D, int VDIM> MFEM_HOST_DEVICE inline
const StaticTensor<dTensor<VDIM,VDIM>,D> ReadSymmMatrix(const DeviceTensor<3,const double> &e_vec,
                                                  const int e)
{
   StaticTensor<dTensor<VDIM,VDIM>,D> u;
   for (int i = 0; i < VDIM*(VDIM+1)/2; i++)
   {
      const int h = i/VDIM;
      const int w = i%VDIM;
      MFEM_FOREACH_THREAD(d,x,D)
      {
         const double val = e_vec(d, i, e);
         if (h!=w)
         {
            u(d)(h,w) = val;
            u(d)(w,h) = val;
         }
         else
         {
            u(d)(h,w) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   return u;
}

// 3D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
const StaticTensor<dTensor<VDIM,VDIM>,D1d,D1d,D1d> ReadMatrix(const DeviceTensor<6> &e_vec,
                                                        const int e)
{
   StaticTensor<dTensor<VDIM,VDIM>,D1d,D1d,D1d> u;
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

// 3D tensor read with VDIMxVDIM components
template<int D1d, int VDIM> MFEM_HOST_DEVICE inline
const StaticTensor<dTensor<VDIM,VDIM>,D1d,D1d,D1d> ReadSymmMatrix(const DeviceTensor<5,const double> &e_vec,
                                                            const int e)
{
   StaticTensor<dTensor<VDIM,VDIM>,D1d,D1d,D1d> u;
   for (int i = 0; i < VDIM*(VDIM+1)/2; i++)
   {
      const int h = i/VDIM;
      const int w = i%VDIM;
      MFEM_FOREACH_THREAD(dz,z,D1d)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            MFEM_FOREACH_THREAD(dx,x,D1d)
            {
               const double val = e_vec(dx, dy, dz, i, e);
               if (h!=w)
               {
                  u(dx,dy,dz)(h,w) = val;
                  u(dx,dy,dz)(w,h) = val;
               }
               else
               {
                  u(dx,dy,dz)(h,w) = val;
               }
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return u;
}

// 2D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
const StaticTensor<dTensor<VDIM,VDIM>,D1d,D1d> ReadMatrix(const DeviceTensor<5> &e_vec,
                                                    const int e)
{
   StaticTensor<dTensor<VDIM,VDIM>,D1d,D1d> u;
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

// 2D read with VDIMxVDIM components
template<int D1d, int VDIM> MFEM_HOST_DEVICE inline
const StaticTensor<dTensor<VDIM,VDIM>,D1d,D1d> ReadSymmMatrix(const DeviceTensor<4,const double> &e_vec,
                                                        const int e)
{
   StaticTensor<dTensor<VDIM,VDIM>,D1d,D1d> u;
   for (int i = 0; i < VDIM*(VDIM+1)/2; i++)
   {
      const int h = i/VDIM;
      const int w = i%VDIM;
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            const double val = e_vec(dx, dy, i, e);
            if (h!=w)
            {
               u(dx,dy)(h,w) = val;
               u(dx,dy)(w,h) = val;
            }
            else
            {
               u(dx,dy)(h,w) = val;
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   return u;
}

} // namespace mfem

#endif // MFEM_TENSOR_READ_MATRIX
