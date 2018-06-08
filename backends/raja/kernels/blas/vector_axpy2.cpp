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

#ifndef __LAMBDA__
extern "C" kernel
void vector_axpby0(const int N,
                   const double alpha,
                   const double beta,
                   double* __restrict v0,
                   const double* __restrict v1)
{
   const int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < N) { v0[i] = alpha * v0[i] + beta * v1[i]; }
}
#endif
//*v0 = alpha * (*this) + beta * v1
void vector_axpby(const int N,
                  const double alpha,
                  const double beta,
                  double* __restrict v0,
                  const double* __restrict v1)
{
   push();
#ifndef __LAMBDA__
   cuKer(vector_axpy,N,alpha,beta,v0,v1);
#else
   forall(i,N,v0[i] = alpha * v0[i] + beta * v1[i];);
#endif
   pop();
}
