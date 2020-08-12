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
#include "../general/backends.hpp"
#include "../dtensor.hpp"

namespace mfem
{

// Functions to read dofs to quad matrix
template<int P, int Q> MFEM_HOST_DEVICE inline
const dTensor<Q,P>&& ReadMatrix(const DeviceTensor<2> &d_B)
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
const Tensor<dTensor<Dim>,Q,P>&& ReadMatrix(const DeviceTensor<3> &d_G)
{
   Tensor<dTensor<Dim>,Q,P> s_G;
   for (int p = 0; p < P; p++)
   {
      for (int s = 0; s < Dim; s++)
      {      
         MFEM_FOREACH_THREAD(q,x,Q)
         {
            s_G(q,p)(s) = d_B(q,s,p);
         }
      }    
   }
   MFEM_SYNC_THREAD;
   return s_G;
}

// Non-tensor read with VDIMxVDIM components
template<int P, int VDIM> MFEM_HOST_DEVICE inline
const Tensor<dTensor<VDIM,VDIM>,P>&& ReadMatrix(const DeviceTensor<4> &e_vec,
                                                const int e)
{
   Tensor<dTensor<VDIM,VDIM>,P> u;
   for (int w = 0; w < VDIM; w++)
   {   
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(p,x,P)
         {
            u(p)(h,w) = e_vec(p, h, w, e);
         }
      }
   }
   MFEM_SYNC_THREAD;
   return u;
}

} // namespace mfem

#endif // MFEM_TENSOR_READ_MATRIX
