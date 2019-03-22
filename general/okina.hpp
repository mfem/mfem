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

#ifndef MFEM_OKINA_HPP
#define MFEM_OKINA_HPP

// *****************************************************************************
#include "../config/config.hpp"
#include "../general/error.hpp"

// *****************************************************************************
#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>

// *****************************************************************************
#include "./cuda.hpp"
#include "./occa.hpp"
#include "./raja.hpp"
#include "./openmp.hpp"

// *****************************************************************************
#include "./mm.hpp"
#include "./config.hpp"

namespace mfem
{

// *****************************************************************************
// * Kernel body wrapper
// *****************************************************************************
template <int BLOCKS, typename DBODY, typename HBODY>
void OkinaWrap(const int N, DBODY &&d_body, HBODY &&h_body)
{
   const bool omp  = mfem::config::UsingOmp();
   const bool gpu  = mfem::config::UsingDevice();
   const bool raja = mfem::config::UsingRaja();
   if (gpu && raja) { return mfem::RajaCudaWrap<BLOCKS>(N, d_body); }
   if (gpu)         { return CuWrap<BLOCKS>(N, d_body); }
   if (omp && raja) { return RajaOmpWrap(N, h_body); }
   if (raja)        { return RajaSeqWrap(N, h_body); }
   if (omp)         { return OmpWrap(N, h_body);  }
   for (int k=0; k<N; k+=1) { h_body(k); }
}

// *****************************************************************************
// * MFEM_FORALL wrapper
// *****************************************************************************
#define MFEM_BLOCKS 256
#define MFEM_FORALL(i,N,...) MFEM_FORALL_K(i,N,MFEM_BLOCKS,__VA_ARGS__)
#define MFEM_FORALL_K(i,N,BLOCKS,...)                                   \
   OkinaWrap<BLOCKS>(N,                                                 \
                     [=] __device__ (int i) mutable {__VA_ARGS__},      \
                     [&]            (int i) {__VA_ARGS__})

// *****************************************************************************
#ifndef MFEM_USE_CUDA
#define MFEM_HOST_DEVICE
#else
#define MFEM_HOST_DEVICE __host__ __device__
#endif

// *****************************************************************************
#define FILE_LINE __FILE__ && __LINE__
#define MFEM_GPU_CANNOT_PASS {assert(FILE_LINE && !config::UsingDevice());}

} // namespace mfem

#endif // MFEM_OKINA_HPP
