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

/// A structure used to pass additional data to f_build_diff and f_apply_diff
struct BuildContext { CeedInt dim, space_dim; CeedScalar coeff; };

/// libCEED Q-function for building quadrature data for a mass operator
extern "C" __global__ void f_build_mass_const(void *ctx, CeedInt Q,
                        Fields_Cuda fields) {
  // in[0] is Jacobians with shape [dim, nc=dim, Q]
  // in[1] is quadrature weights, size (Q)
  struct BuildContext *bc = (struct BuildContext*)ctx;
  const CeedScalar coeff = bc->coeff;
  const CeedScalar *J = (const CeedScalar *)fields.inputs[0];
  const CeedScalar *qw = (const CeedScalar *)fields.inputs[1];
  CeedScalar *qd = fields.outputs[0];
  switch (bc->dim + 10*bc->space_dim) {
  case 11:
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < Q;
         i += blockDim.x * gridDim.x) {
      qd[i] = coeff * J[i] * qw[i];
    }
    break;
  case 22:
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < Q;
         i += blockDim.x * gridDim.x) {
      // 0 2
      // 1 3
      qd[i] = coeff * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
    }
    break;
  case 33:
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < Q;
         i += blockDim.x * gridDim.x) {
      // 0 3 6
      // 1 4 7
      // 2 5 8
      qd[i] = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
               J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
               J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * coeff * qw[i];
    }
    break;
  }
}

extern "C" __global__ void f_build_mass_grid(void *ctx, CeedInt Q,
                        Fields_Cuda fields) {
  // in[0] is Jacobians with shape [dim, nc=dim, Q]
  // in[1] is quadrature weights, size (Q)
  struct BuildContext *bc = (struct BuildContext*)ctx;
  const CeedScalar *c = (const CeedScalar *)fields.inputs[0];
  const CeedScalar *J = (const CeedScalar *)fields.inputs[1];
  const CeedScalar *qw = (const CeedScalar *)fields.inputs[2];
  CeedScalar *qd = fields.outputs[0];
  switch (bc->dim + 10*bc->space_dim) {
  case 11:
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < Q;
         i += blockDim.x * gridDim.x) {
      qd[i] = c[i] * J[i] * qw[i];
    }
    break;
  case 22:
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < Q;
         i += blockDim.x * gridDim.x) {
      // 0 2
      // 1 3
      qd[i] = c[i] * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
    }
    break;
  case 33:
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < Q;
         i += blockDim.x * gridDim.x) {
      // 0 3 6
      // 1 4 7
      // 2 5 8
      qd[i] = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
               J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
               J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * c[i] * qw[i];
    }
    break;
  }
}

/// libCEED Q-function for applying a mass operator
extern "C" __global__ void f_apply_mass(void *ctx, CeedInt Q,
                        Fields_Cuda fields) {
  const CeedScalar *u = (const CeedScalar *)fields.inputs[0];
  const CeedScalar *w = (const CeedScalar *)fields.inputs[1];
  CeedScalar *v = fields.outputs[0];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < Q;
       i += blockDim.x * gridDim.x) {
    v[i] = w[i] * u[i];
  }
}
