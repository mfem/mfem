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


/// A structure used to pass additional data to f_build_diff and f_apply_diff
struct BuildContext { CeedInt dim, space_dim, vdim; CeedScalar coeff; };

/// libCEED Q-function for building quadrature data for a mass operator with a
/// constant coefficient
CEED_QFUNCTION(f_build_mass_const)(void *ctx, CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out)
{
   // in[0] is Jacobians with shape [dim, nc=dim, Q]
   // in[1] is quadrature weights, size (Q)
   BuildContext *bc = (BuildContext *)ctx;
   const CeedScalar coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *rho = out[0];
   switch (bc->dim + 10*bc->space_dim)
   {
      case 11:
         for (CeedInt i=0; i<Q; i++)
         {
            rho[i] = coeff * J[i] * qw[i];
         }
         break;
      case 22:
         for (CeedInt i=0; i<Q; i++)
         {
            // 0 2
            // 1 3
            rho[i] = coeff * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
         }
         break;
      case 33:
         for (CeedInt i=0; i<Q; i++)
         {
            // 0 3 6
            // 1 4 7
            // 2 5 8
            rho[i] = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                      J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                      J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * coeff * qw[i];
         }
         break;
   }
   return 0;
}

/// libCEED Q-function for building quadrature data for a mass operator with a
/// coefficient evaluated at quadrature points.
CEED_QFUNCTION(f_build_mass_quad)(void *ctx, CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out)
{
   // in[0] is Jacobians with shape [dim, nc=dim, Q]
   // in[1] is quadrature weights, size (Q)
   BuildContext *bc = (BuildContext *)ctx;
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *rho = out[0];
   switch (bc->dim + 10*bc->space_dim)
   {
      case 11:
         for (CeedInt i=0; i<Q; i++)
         {
            rho[i] = c[i] * J[i] * qw[i];
         }
         break;
      case 22:
         for (CeedInt i=0; i<Q; i++)
         {
            // 0 2
            // 1 3
            rho[i] = c[i] * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
         }
         break;
      case 33:
         for (CeedInt i=0; i<Q; i++)
         {
            // 0 3 6
            // 1 4 7
            // 2 5 8
            rho[i] = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                      J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                      J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * c[i] * qw[i];
         }
         break;
   }
   return 0;
}

/// libCEED Q-function for applying a mass operator
CEED_QFUNCTION(f_apply_mass)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out)
{
   BuildContext *bc = (BuildContext *)ctx;
   const CeedScalar *u = in[0], *w = in[1];
   CeedScalar *v = out[0];
   switch (bc->vdim)
   {
      case 1:
         for (CeedInt i=0; i<Q; i++)
         {
            v[i] = w[i] * u[i];
         }
         break;
      case 2:
         for (CeedInt i=0; i<Q; i++)
         {
            const CeedScalar W = w[i];
            for (CeedInt c = 0; c < 2; c++)
            {
               v[i+c*Q] = W * u[i+c*Q];
            }
         }
         break;
      case 3:
         for (CeedInt i=0; i<Q; i++)
         {
            const CeedScalar W = w[i];
            for (CeedInt c = 0; c < 3; c++)
            {
               v[i+c*Q] = W * u[i+c*Q];
            }
         }
         break;
   }
   return 0;
}

/// libCEED Q-function for applying a diff operator
CEED_QFUNCTION(f_apply_mass_mf_const)(void *ctx, CeedInt Q,
                                      const CeedScalar *const *in, CeedScalar *const *out)
{
   BuildContext *bc = (BuildContext*)ctx;
   const CeedScalar coeff = bc->coeff;
   const CeedScalar *u = in[0], *J = in[1], *qw = in[2];
   CeedScalar *v = out[0];
   switch (10 * bc->dim + bc->vdim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar rho = coeff * qw[i] / J[i];
            v[i] = rho * u[i];
         }
         break;
      case 21:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar rho = coeff * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
            v[i] = rho * u[i];
         }
         break;
      case 22:
         for (CeedInt i=0; i<Q; i++)
         {
            // 0 2
            // 1 3
            const CeedScalar rho = coeff * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
            for (CeedInt c = 0; c < 2; c++)
            {
               v[i+c*Q] = rho * u[i+c*Q];
            }
         }
         break;
      case 31:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar rho = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                                    J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                                    J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * coeff * qw[i];
            v[i] = rho * u[i];
         }
         break;
      case 33:
         for (CeedInt i=0; i<Q; i++)
         {
            // 0 3 6
            // 1 4 7
            // 2 5 8
            const CeedScalar rho = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                                    J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                                    J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * coeff * qw[i];
            for (CeedInt c = 0; c < 3; c++)
            {
               v[i+c*Q] = rho * u[i+c*Q];
            }
         }
         break;
   }
   return 0;
}

CEED_QFUNCTION(f_apply_mass_mf_quad)(void *ctx, CeedInt Q,
                                     const CeedScalar *const *in, CeedScalar *const *out)
{
   BuildContext *bc = (BuildContext*)ctx;
   const CeedScalar *c = in[0], *u = in[1], *J = in[2], *qw = in[3];
   CeedScalar *v = out[0];
   switch (10 * bc->dim + bc->vdim)
   {
      case 11:
         for (CeedInt i=0; i<Q; i++)
         {
            const CeedScalar rho = c[i] * J[i] * qw[i];
            v[i] = rho * u[i];
         }
         break;
      case 21:
         for (CeedInt i=0; i<Q; i++)
         {
            // 0 2
            // 1 3
            const CeedScalar rho = c[i] * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
            v[i] = rho * u[i];
         }
         break;
      case 22:
         for (CeedInt i=0; i<Q; i++)
         {
            // 0 2
            // 1 3
            const CeedScalar rho = c[i] * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
            for (CeedInt c = 0; c < 2; c++)
            {
               v[i+c*Q] = rho * u[i+c*Q];
            }
         }
         break;
      case 31:
         for (CeedInt i=0; i<Q; i++)
         {
            // 0 3 6
            // 1 4 7
            // 2 5 8
            const CeedScalar rho = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                                    J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                                    J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * c[i] * qw[i];
            v[i] = rho * u[i];
         }
         break;
      case 33:
         for (CeedInt i=0; i<Q; i++)
         {
            // 0 3 6
            // 1 4 7
            // 2 5 8
            const CeedScalar rho = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                                    J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                                    J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * c[i] * qw[i];
            for (CeedInt c = 0; c < 3; c++)
            {
               v[i+c*Q] = rho * u[i+c*Q];
            }
         }
         break;
   }
   return 0;
}
