// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#include <ceed/types.h>

/// A structure used to pass additional data to f_build_conv and f_apply_conv
struct NLConvectionContext { CeedInt dim, space_dim, vdim; CeedScalar coeff; };

/// libCEED Q-function for building quadrature data for a convection operator
/// with a constant coefficient
CEED_QFUNCTION(f_build_conv_const)(void *ctx, CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out)
{
   NLConvectionContext *bc = (NLConvectionContext*)ctx;
   // in[0] is Jacobians with shape [dim, nc=dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute and store qw * adj(J).
   const CeedScalar coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (bc->dim + 10 * bc->space_dim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = coeff * qw[i] * J[i];
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 2   adj(J):  J22 -J12
            //    1 3       1 3           -J21  J11
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J12 = J[i + Q * 2];
            const CeedScalar J22 = J[i + Q * 3];
            const CeedScalar w = qw[i] * coeff;
            qd[i + Q * 0] =  w * J22;
            qd[i + Q * 1] = -w * J21;
            qd[i + Q * 2] = -w * J12;
            qd[i + Q * 3] =  w * J11;
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 3 6   qd: 0 3 6
            //    1 4 7       1 4 7
            //    2 5 8       2 5 8
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J31 = J[i + Q * 2];
            const CeedScalar J12 = J[i + Q * 3];
            const CeedScalar J22 = J[i + Q * 4];
            const CeedScalar J32 = J[i + Q * 5];
            const CeedScalar J13 = J[i + Q * 6];
            const CeedScalar J23 = J[i + Q * 7];
            const CeedScalar J33 = J[i + Q * 8];
            const CeedScalar A11 = J22 * J33 - J23 * J32;
            const CeedScalar A12 = J13 * J32 - J12 * J33;
            const CeedScalar A13 = J12 * J23 - J13 * J22;
            const CeedScalar A21 = J23 * J31 - J21 * J33;
            const CeedScalar A22 = J11 * J33 - J13 * J31;
            const CeedScalar A23 = J13 * J21 - J11 * J23;
            const CeedScalar A31 = J21 * J32 - J22 * J31;
            const CeedScalar A32 = J12 * J31 - J11 * J32;
            const CeedScalar A33 = J11 * J22 - J12 * J21;
            const CeedScalar w = qw[i] * coeff;
            qd[i + Q * 0] = w * A11;
            qd[i + Q * 1] = w * A21;
            qd[i + Q * 2] = w * A31;
            qd[i + Q * 3] = w * A12;
            qd[i + Q * 4] = w * A22;
            qd[i + Q * 5] = w * A32;
            qd[i + Q * 6] = w * A13;
            qd[i + Q * 7] = w * A23;
            qd[i + Q * 8] = w * A33;
         }
         break;
   }
   return CEED_ERROR_SUCCESS;
}

/// libCEED Q-function for building quadrature data for a convection operator
/// coefficient evaluated at quadrature points.
CEED_QFUNCTION(f_build_conv_quad)(void *ctx, CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out)
{
   NLConvectionContext *bc = (NLConvectionContext *)ctx;
   // in[1] is Jacobians with shape [dim, nc=dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute and store qw * adj(J).
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (bc->dim + 10 * bc->space_dim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff = c[i];
            qd[i] = coeff * qw[i] * J[i];
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 2   adj(J):  J22 -J12
            //    1 3       1 3           -J21  J11
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J12 = J[i + Q * 2];
            const CeedScalar J22 = J[i + Q * 3];
            const CeedScalar coeff = c[i];
            const CeedScalar w = qw[i] * coeff;
            qd[i + Q * 0] =  w * J22;
            qd[i + Q * 1] = -w * J21;
            qd[i + Q * 2] = -w * J12;
            qd[i + Q * 3] =  w * J11;
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 3 6   qd: 0 3 6
            //    1 4 7       1 4 7
            //    2 5 8       2 5 8
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J31 = J[i + Q * 2];
            const CeedScalar J12 = J[i + Q * 3];
            const CeedScalar J22 = J[i + Q * 4];
            const CeedScalar J32 = J[i + Q * 5];
            const CeedScalar J13 = J[i + Q * 6];
            const CeedScalar J23 = J[i + Q * 7];
            const CeedScalar J33 = J[i + Q * 8];
            const CeedScalar A11 = J22 * J33 - J23 * J32;
            const CeedScalar A12 = J13 * J32 - J12 * J33;
            const CeedScalar A13 = J12 * J23 - J13 * J22;
            const CeedScalar A21 = J23 * J31 - J21 * J33;
            const CeedScalar A22 = J11 * J33 - J13 * J31;
            const CeedScalar A23 = J13 * J21 - J11 * J23;
            const CeedScalar A31 = J21 * J32 - J22 * J31;
            const CeedScalar A32 = J12 * J31 - J11 * J32;
            const CeedScalar A33 = J11 * J22 - J12 * J21;
            const CeedScalar coeff = c[i];
            const CeedScalar w = qw[i] * coeff;
            qd[i + Q * 0] = w * A11;
            qd[i + Q * 1] = w * A21;
            qd[i + Q * 2] = w * A31;
            qd[i + Q * 3] = w * A12;
            qd[i + Q * 4] = w * A22;
            qd[i + Q * 5] = w * A32;
            qd[i + Q * 6] = w * A13;
            qd[i + Q * 7] = w * A23;
            qd[i + Q * 8] = w * A33;
         }
         break;
   }
   return CEED_ERROR_SUCCESS;
}

/// libCEED Q-function for applying a conv operator
CEED_QFUNCTION(f_apply_conv)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out)
{
   NLConvectionContext *bc = (NLConvectionContext *)ctx;
   // in[0], out[0] have shape [dim, nc=1, Q]
   const CeedScalar *u = in[0], *ug = in[1], *qd = in[2];
   CeedScalar *vg = out[0];
   switch (10*bc->dim + bc->vdim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            vg[i] = u[i] * ug[i] * qd[i];
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd00 = qd[i + Q * 0];
            const CeedScalar qd10 = qd[i + Q * 1];
            const CeedScalar qd01 = qd[i + Q * 2];
            const CeedScalar qd11 = qd[i + Q * 3];
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug01 = ug[i + Q * 2];
            const CeedScalar ug11 = ug[i + Q * 3];
            const CeedScalar Dxu0 = ug00 * qd00 + ug01 * qd10;
            const CeedScalar Dyu0 = ug00 * qd01 + ug01 * qd11;
            const CeedScalar Dxu1 = ug10 * qd00 + ug11 * qd10;
            const CeedScalar Dyu1 = ug10 * qd01 + ug11 * qd11;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd00 = qd[i + Q * 0];
            const CeedScalar qd10 = qd[i + Q * 1];
            const CeedScalar qd20 = qd[i + Q * 2];
            const CeedScalar qd01 = qd[i + Q * 3];
            const CeedScalar qd11 = qd[i + Q * 4];
            const CeedScalar qd21 = qd[i + Q * 5];
            const CeedScalar qd02 = qd[i + Q * 6];
            const CeedScalar qd12 = qd[i + Q * 7];
            const CeedScalar qd22 = qd[i + Q * 8];
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar u2   = u[i + Q * 2];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug20 = ug[i + Q * 2];
            const CeedScalar ug01 = ug[i + Q * 3];
            const CeedScalar ug11 = ug[i + Q * 4];
            const CeedScalar ug21 = ug[i + Q * 5];
            const CeedScalar ug02 = ug[i + Q * 6];
            const CeedScalar ug12 = ug[i + Q * 7];
            const CeedScalar ug22 = ug[i + Q * 8];
            const CeedScalar Dxu0 = ug00 * qd00 + ug01 * qd10 + ug02 * qd20;
            const CeedScalar Dyu0 = ug00 * qd01 + ug01 * qd11 + ug02 * qd21;
            const CeedScalar Dzu0 = ug00 * qd02 + ug01 * qd12 + ug02 * qd22;
            const CeedScalar Dxu1 = ug10 * qd00 + ug11 * qd10 + ug12 * qd20;
            const CeedScalar Dyu1 = ug10 * qd01 + ug11 * qd11 + ug12 * qd21;
            const CeedScalar Dzu1 = ug10 * qd02 + ug11 * qd12 + ug12 * qd22;
            const CeedScalar Dxu2 = ug20 * qd00 + ug21 * qd10 + ug22 * qd20;
            const CeedScalar Dyu2 = ug20 * qd01 + ug21 * qd11 + ug22 * qd21;
            const CeedScalar Dzu2 = ug20 * qd02 + ug21 * qd12 + ug22 * qd22;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
   }
   return CEED_ERROR_SUCCESS;
}

/// libCEED Q-function for applying a conv operator
CEED_QFUNCTION(f_apply_conv_mf_const)(void *ctx, CeedInt Q,
                                      const CeedScalar *const *in,
                                      CeedScalar *const *out)
{
   NLConvectionContext *bc = (NLConvectionContext*)ctx;
   // in[0], out[0] have shape [dim, nc=1, Q]
   // in[1] is Jacobians with shape [dim, nc=dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw * adj(J).
   const CeedScalar coeff = bc->coeff;
   const CeedScalar *u = in[0], *ug = in[1], *J = in[2], *qw = in[3];
   CeedScalar *vg = out[0];
   switch (10 * bc->dim + bc->vdim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = coeff * qw[i] * J[i];
            vg[i] = u[i] * ug[i] * qd;
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 2   adj(J):  J22 -J12
            //    1 3       1 3           -J21  J11
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J12 = J[i + Q * 2];
            const CeedScalar J22 = J[i + Q * 3];
            const CeedScalar w = qw[i] * coeff;
            const CeedScalar qd00 =  w * J22;
            const CeedScalar qd10 = -w * J21;
            const CeedScalar qd01 = -w * J12;
            const CeedScalar qd11 =  w * J11;
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug01 = ug[i + Q * 2];
            const CeedScalar ug11 = ug[i + Q * 3];
            const CeedScalar Dxu0 = ug00 * qd00 + ug01 * qd10;
            const CeedScalar Dyu0 = ug00 * qd01 + ug01 * qd11;
            const CeedScalar Dxu1 = ug10 * qd00 + ug11 * qd10;
            const CeedScalar Dyu1 = ug10 * qd01 + ug11 * qd11;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 3 6   qd: 0 3 6
            //    1 4 7       1 4 7
            //    2 5 8       2 5 8
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J31 = J[i + Q * 2];
            const CeedScalar J12 = J[i + Q * 3];
            const CeedScalar J22 = J[i + Q * 4];
            const CeedScalar J32 = J[i + Q * 5];
            const CeedScalar J13 = J[i + Q * 6];
            const CeedScalar J23 = J[i + Q * 7];
            const CeedScalar J33 = J[i + Q * 8];
            const CeedScalar A11 = J22 * J33 - J23 * J32;
            const CeedScalar A12 = J13 * J32 - J12 * J33;
            const CeedScalar A13 = J12 * J23 - J13 * J22;
            const CeedScalar A21 = J23 * J31 - J21 * J33;
            const CeedScalar A22 = J11 * J33 - J13 * J31;
            const CeedScalar A23 = J13 * J21 - J11 * J23;
            const CeedScalar A31 = J21 * J32 - J22 * J31;
            const CeedScalar A32 = J12 * J31 - J11 * J32;
            const CeedScalar A33 = J11 * J22 - J12 * J21;
            const CeedScalar w = qw[i] * coeff;
            const CeedScalar qd00 = w * A11;
            const CeedScalar qd10 = w * A21;
            const CeedScalar qd20 = w * A31;
            const CeedScalar qd01 = w * A12;
            const CeedScalar qd11 = w * A22;
            const CeedScalar qd21 = w * A32;
            const CeedScalar qd02 = w * A13;
            const CeedScalar qd12 = w * A23;
            const CeedScalar qd22 = w * A33;
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar u2   = u[i + Q * 2];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug20 = ug[i + Q * 2];
            const CeedScalar ug01 = ug[i + Q * 3];
            const CeedScalar ug11 = ug[i + Q * 4];
            const CeedScalar ug21 = ug[i + Q * 5];
            const CeedScalar ug02 = ug[i + Q * 6];
            const CeedScalar ug12 = ug[i + Q * 7];
            const CeedScalar ug22 = ug[i + Q * 8];
            const CeedScalar Dxu0 = ug00 * qd00 + ug01 * qd10 + ug02 * qd20;
            const CeedScalar Dyu0 = ug00 * qd01 + ug01 * qd11 + ug02 * qd21;
            const CeedScalar Dzu0 = ug00 * qd02 + ug01 * qd12 + ug02 * qd22;
            const CeedScalar Dxu1 = ug10 * qd00 + ug11 * qd10 + ug12 * qd20;
            const CeedScalar Dyu1 = ug10 * qd01 + ug11 * qd11 + ug12 * qd21;
            const CeedScalar Dzu1 = ug10 * qd02 + ug11 * qd12 + ug12 * qd22;
            const CeedScalar Dxu2 = ug20 * qd00 + ug21 * qd10 + ug22 * qd20;
            const CeedScalar Dyu2 = ug20 * qd01 + ug21 * qd11 + ug22 * qd21;
            const CeedScalar Dzu2 = ug20 * qd02 + ug21 * qd12 + ug22 * qd22;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
   }
   return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(f_apply_conv_mf_quad)(void *ctx, CeedInt Q,
                                     const CeedScalar *const *in,
                                     CeedScalar *const *out)
{
   NLConvectionContext *bc = (NLConvectionContext*)ctx;
   // in[0], out[0] have shape [dim, nc=1, Q]
   // in[1] is Jacobians with shape [dim, nc=dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw * adj(J).
   const CeedScalar *c = in[0], *u = in[1], *ug = in[2], *J = in[3], *qw = in[4];
   CeedScalar *vg = out[0];
   switch (10 * bc->dim + bc->vdim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = c[i] * qw[i] * J[i];
            vg[i] = u[i] * ug[i] * qd;
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 2   adj(J):  J22 -J12
            //    1 3       1 3           -J21  J11
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J12 = J[i + Q * 2];
            const CeedScalar J22 = J[i + Q * 3];
            const CeedScalar w = qw[i] * c[i];
            const CeedScalar qd00 =  w * J22;
            const CeedScalar qd10 = -w * J21;
            const CeedScalar qd01 = -w * J12;
            const CeedScalar qd11 =  w * J11;
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug01 = ug[i + Q * 2];
            const CeedScalar ug11 = ug[i + Q * 3];
            const CeedScalar Dxu0 = ug00 * qd00 + ug01 * qd10;
            const CeedScalar Dyu0 = ug00 * qd01 + ug01 * qd11;
            const CeedScalar Dxu1 = ug10 * qd00 + ug11 * qd10;
            const CeedScalar Dyu1 = ug10 * qd01 + ug11 * qd11;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 3 6   qd: 0 3 6
            //    1 4 7       1 4 7
            //    2 5 8       2 5 8
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J31 = J[i + Q * 2];
            const CeedScalar J12 = J[i + Q * 3];
            const CeedScalar J22 = J[i + Q * 4];
            const CeedScalar J32 = J[i + Q * 5];
            const CeedScalar J13 = J[i + Q * 6];
            const CeedScalar J23 = J[i + Q * 7];
            const CeedScalar J33 = J[i + Q * 8];
            const CeedScalar A11 = J22 * J33 - J23 * J32;
            const CeedScalar A12 = J13 * J32 - J12 * J33;
            const CeedScalar A13 = J12 * J23 - J13 * J22;
            const CeedScalar A21 = J23 * J31 - J21 * J33;
            const CeedScalar A22 = J11 * J33 - J13 * J31;
            const CeedScalar A23 = J13 * J21 - J11 * J23;
            const CeedScalar A31 = J21 * J32 - J22 * J31;
            const CeedScalar A32 = J12 * J31 - J11 * J32;
            const CeedScalar A33 = J11 * J22 - J12 * J21;
            const CeedScalar w = qw[i] * c[i];
            const CeedScalar qd00 = w * A11;
            const CeedScalar qd10 = w * A21;
            const CeedScalar qd20 = w * A31;
            const CeedScalar qd01 = w * A12;
            const CeedScalar qd11 = w * A22;
            const CeedScalar qd21 = w * A32;
            const CeedScalar qd02 = w * A13;
            const CeedScalar qd12 = w * A23;
            const CeedScalar qd22 = w * A33;
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar u2   = u[i + Q * 2];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug20 = ug[i + Q * 2];
            const CeedScalar ug01 = ug[i + Q * 3];
            const CeedScalar ug11 = ug[i + Q * 4];
            const CeedScalar ug21 = ug[i + Q * 5];
            const CeedScalar ug02 = ug[i + Q * 6];
            const CeedScalar ug12 = ug[i + Q * 7];
            const CeedScalar ug22 = ug[i + Q * 8];
            const CeedScalar Dxu0 = ug00 * qd00 + ug01 * qd10 + ug02 * qd20;
            const CeedScalar Dyu0 = ug00 * qd01 + ug01 * qd11 + ug02 * qd21;
            const CeedScalar Dzu0 = ug00 * qd02 + ug01 * qd12 + ug02 * qd22;
            const CeedScalar Dxu1 = ug10 * qd00 + ug11 * qd10 + ug12 * qd20;
            const CeedScalar Dyu1 = ug10 * qd01 + ug11 * qd11 + ug12 * qd21;
            const CeedScalar Dzu1 = ug10 * qd02 + ug11 * qd12 + ug12 * qd22;
            const CeedScalar Dxu2 = ug20 * qd00 + ug21 * qd10 + ug22 * qd20;
            const CeedScalar Dyu2 = ug20 * qd01 + ug21 * qd11 + ug22 * qd21;
            const CeedScalar Dzu2 = ug20 * qd02 + ug21 * qd12 + ug22 * qd22;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
   }
   return CEED_ERROR_SUCCESS;
}
