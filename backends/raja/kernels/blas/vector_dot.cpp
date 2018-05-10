// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "../raja.hpp"

// *****************************************************************************
#ifdef __NVCC__
__inline__ __device__ double4 operator*(double4 a, double4 b) {
  return make_double4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}
#include <cub/cub.cuh>

// *****************************************************************************
static double cub_vector_dot(const int N,
                             const double* __restrict vec1,
                             const double* __restrict vec2) {
  static double *h_dot = NULL;
  if (!h_dot) h_dot = (double*)mfem::rmalloc<double>::operator new(1,true);
  static double *d_dot = NULL;
  if (!d_dot) d_dot=(double*)mfem::rmalloc<double>::operator new(1);
  static void *d_storage = NULL;
  static size_t storage_bytes = 0;
  if (!d_storage){
    cub::DeviceReduce::Dot(d_storage, storage_bytes, vec1, vec2, d_dot, N);
    d_storage = mfem::rmalloc<char>::operator new(storage_bytes);
  }
  cub::DeviceReduce::Dot(d_storage, storage_bytes, vec1, vec2, d_dot, N);
  mfem::rmemcpy::rDtoH(h_dot,d_dot,sizeof(double));
  return *h_dot;
}
#endif // __NVCC__

// *****************************************************************************
double vector_dot(const int N,
                  const double* __restrict vec1,
                  const double* __restrict vec2) {
  push(dot,Cyan);
#ifdef __NVCC__
  if (mfem::rconfig::Get().Cuda()){
    const double result = cub_vector_dot(N,vec1,vec2);
    pop();
    return result;
  }
#endif
  ReduceDecl(Sum,dot,0.0);
  ReduceForall(i,N,dot += vec1[i]*vec2[i];);
  pop();
  return dot;
}
