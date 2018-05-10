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
void vector_set_subvector_const0(const int N,
                                 const double value,
                                 double* __restrict data,
                                 const int* __restrict tdofs) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  const int dof_i = tdofs[i];
  data[dof_i] = value;
  if (dof_i >= 0) {
    data[dof_i] = value;
  } else {
    data[-dof_i-1] = -value;
  }
}
#endif

void vector_set_subvector_const(const int N,
                                const double value,
                                double* __restrict data,
                                const int* __restrict tdofs) {
  push(set,Cyan);
#ifndef __LAMBDA__
  cuKer(vector_set_subvector_const,N,value,data,tdofs);
#else
  forall(i,N,{
      const int dof_i = tdofs[i];
      data[dof_i] = value;
      if (dof_i >= 0) {
        data[dof_i] = value;
      } else {
        data[-dof_i-1] = -value;
      }
    });
#endif
  pop();
}
