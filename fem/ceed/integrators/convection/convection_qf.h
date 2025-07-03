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
struct ConvectionContext {
   CeedInt dim, space_dim, vdim;
   CeedScalar coeff[3];
   CeedScalar alpha;
};

/// libCEED Q-function for building quadrature data for a convection operator
/// with a constant coefficient
CEED_QFUNCTION(f_build_conv_const)(void *ctx, CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out)
{
   ConvectionContext *bc = (ConvectionContext*)ctx;
   // in[0] is Jacobians with shape [dim, nc=dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute and store qw * adj(J).
   const CeedScalar coeff0 = bc->coeff[0];
   const CeedScalar coeff1 = bc->coeff[1];
   const CeedScalar coeff2 = bc->coeff[2];
   const CeedScalar alpha  = bc->alpha;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (bc->dim + 10 * bc->space_dim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = alpha * coeff0 * qw[i] * J[i];
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 1   adj(J):  J22 -J12
            //    1 3       1 2           -J21  J11
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J12 = J[i + Q * 2];
            const CeedScalar J22 = J[i + Q * 3];
            const CeedScalar w = alpha * qw[i];
            const CeedScalar wx = w * coeff0;
            const CeedScalar wy = w * coeff1;
            qd[i + Q * 0] =  wx * J22 - wy * J12;
            qd[i + Q * 1] = -wx * J21 + wy * J11;
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 3 6   qd: 0 1 2
            //    1 4 7       1 3 4
            //    2 5 8       2 4 5
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
            const CeedScalar w = alpha * qw[i];
            const CeedScalar wx = w * coeff0;
            const CeedScalar wy = w * coeff1;
            const CeedScalar wz = w * coeff2;
            qd[i + Q * 0] = wx * A11 + wy * A12 + wz * A13;
            qd[i + Q * 1] = wx * A21 + wy * A22 + wz * A23;
            qd[i + Q * 2] = wx * A31 + wy * A32 + wz * A33;
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
   ConvectionContext *bc = (ConvectionContext *)ctx;
   // in[1] is Jacobians with shape [dim, nc=dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute and store qw * adj(J).
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   const CeedScalar alpha  = bc->alpha;
   CeedScalar *qd = out[0];
   switch (bc->dim + 10 * bc->space_dim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff = c[i];
            qd[i] = alpha * coeff * qw[i] * J[i];
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 1   adj(J):  J22 -J12
            //    1 3       1 2           -J21  J11
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J12 = J[i + Q * 2];
            const CeedScalar J22 = J[i + Q * 3];
            const CeedScalar w = alpha * qw[i];
            const CeedScalar wx = w * c[i + Q * 0];
            const CeedScalar wy = w * c[i + Q * 1];
            qd[i + Q * 0] =  wx * J22 - wy * J12;
            qd[i + Q * 1] = -wx * J21 + wy * J11;
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 3 6   qd: 0 1 2
            //    1 4 7       1 3 4
            //    2 5 8       2 4 5
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
            const CeedScalar w = alpha * qw[i];
            const CeedScalar wx = w * c[i + Q * 0];
            const CeedScalar wy = w * c[i + Q * 1];
            const CeedScalar wz = w * c[i + Q * 2];
            qd[i + Q * 0] = wx * A11 + wy * A12 + wz * A13;
            qd[i + Q * 1] = wx * A21 + wy * A22 + wz * A23;
            qd[i + Q * 2] = wx * A31 + wy * A32 + wz * A33;
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
   ConvectionContext *bc = (ConvectionContext *)ctx;
   // in[0], out[0] have shape [dim, nc=1, Q]
   const CeedScalar *ug = in[0], *qd = in[1];
   CeedScalar *vg = out[0];
   switch (10*bc->dim + bc->vdim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            vg[i] = ug[i] * qd[i];
         }
         break;
      case 21:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i] = qd[i + Q * 0] * ug0 + qd[i + Q * 1] * ug1;
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd0 = qd[i + Q * 0];
            const CeedScalar qd1 = qd[i + Q * 1];
            for (CeedInt c = 0; c < 2; c++)
            {
               const CeedScalar ug0 = ug[i + Q * (c+2*0)];
               const CeedScalar ug1 = ug[i + Q * (c+2*1)];
               vg[i + Q * c] = qd0 * ug0 + qd1 * ug1;
            }
         }
         break;
      case 31:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i] = qd[i + Q * 0] * ug0 + qd[i + Q * 1] * ug1 + qd[i + Q * 2] * ug2;
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd0 = qd[i + Q * 0];
            const CeedScalar qd1 = qd[i + Q * 1];
            const CeedScalar qd2 = qd[i + Q * 2];
            for (CeedInt c = 0; c < 3; c++)
            {
               const CeedScalar ug0 = ug[i + Q * (c+3*0)];
               const CeedScalar ug1 = ug[i + Q * (c+3*1)];
               const CeedScalar ug2 = ug[i + Q * (c+3*2)];
               vg[i + Q * c] = qd0 * ug0 + qd1 * ug1 + qd2 * ug2;
            }
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
   ConvectionContext *bc = (ConvectionContext*)ctx;
   // in[0], out[0] have shape [dim, nc=1, Q]
   // in[1] is Jacobians with shape [dim, nc=dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw * adj(J).
   const CeedScalar coeff0 = bc->coeff[0];
   const CeedScalar coeff1 = bc->coeff[1];
   const CeedScalar coeff2 = bc->coeff[2];
   const CeedScalar alpha  = bc->alpha;
   const CeedScalar *ug = in[0], *J = in[1], *qw = in[2];
   CeedScalar *vg = out[0];
   switch (10 * bc->dim + bc->vdim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = alpha * coeff0 * qw[i] * J[i];
            vg[i] = ug[i] * qd;
         }
         break;
      case 21:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 1   adj(J):  J22 -J12
            //    1 3       1 2           -J21  J11
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J12 = J[i + Q * 2];
            const CeedScalar J22 = J[i + Q * 3];
            const CeedScalar w = alpha * qw[i];
            const CeedScalar wx = w * coeff0;
            const CeedScalar wy = w * coeff1;
            const CeedScalar qd0 =  wx * J22 - wy * J12;
            const CeedScalar qd1 = -wx * J21 + wy * J11;
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i] = qd0 * ug0 + qd1 * ug1;
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 1   adj(J):  J22 -J12
            //    1 3       1 2           -J21  J11
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J12 = J[i + Q * 2];
            const CeedScalar J22 = J[i + Q * 3];
            const CeedScalar w = alpha * qw[i];
            const CeedScalar wx = w * coeff0;
            const CeedScalar wy = w * coeff1;
            const CeedScalar qd0 =  wx * J22 - wy * J12;
            const CeedScalar qd1 = -wx * J21 + wy * J11;
            for (CeedInt c = 0; c < 2; c++)
            {
               const CeedScalar ug0 = ug[i + Q * (c+2*0)];
               const CeedScalar ug1 = ug[i + Q * (c+2*1)];
               vg[i + Q * c] = qd0 * ug0 + qd1 * ug1;
            }
         }
         break;
      case 31:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 3 6   qd: 0 1 2
            //    1 4 7       1 3 4
            //    2 5 8       2 4 5
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
            const CeedScalar w = alpha * qw[i];
            const CeedScalar wx = w * coeff0;
            const CeedScalar wy = w * coeff1;
            const CeedScalar wz = w * coeff2;
            const CeedScalar qd0 = wx * A11 + wy * A12 + wz * A13;
            const CeedScalar qd1 = wx * A21 + wy * A22 + wz * A23;
            const CeedScalar qd2 = wx * A31 + wy * A32 + wz * A33;
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i] = qd0 * ug0 + qd1 * ug1 + qd2 * ug2;
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 3 6   qd: 0 1 2
            //    1 4 7       1 3 4
            //    2 5 8       2 4 5
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
            const CeedScalar w = alpha * qw[i];
            const CeedScalar wx = w * coeff0;
            const CeedScalar wy = w * coeff1;
            const CeedScalar wz = w * coeff2;
            const CeedScalar qd0 = wx * A11 + wy * A12 + wz * A13;
            const CeedScalar qd1 = wx * A21 + wy * A22 + wz * A23;
            const CeedScalar qd2 = wx * A31 + wy * A32 + wz * A33;
            for (CeedInt c = 0; c < 3; c++)
            {
               const CeedScalar ug0 = ug[i + Q * (c+3*0)];
               const CeedScalar ug1 = ug[i + Q * (c+3*1)];
               const CeedScalar ug2 = ug[i + Q * (c+3*2)];
               vg[i + Q * c] = qd0 * ug0 + qd1 * ug1 + qd2 * ug2;
            }
         }
         break;
   }
   return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(f_apply_conv_mf_quad)(void *ctx, CeedInt Q,
                                     const CeedScalar *const *in,
                                     CeedScalar *const *out)
{
   ConvectionContext *bc = (ConvectionContext*)ctx;
   // in[0], out[0] have shape [dim, nc=1, Q]
   // in[1] is Jacobians with shape [dim, nc=dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw * adj(J).
   const CeedScalar *c = in[0], *ug = in[1], *J = in[2], *qw = in[3];
   const CeedScalar alpha  = bc->alpha;
   CeedScalar *vg = out[0];
   switch (10 * bc->dim + bc->vdim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = alpha * c[i] * qw[i] * J[i];
            vg[i] = ug[i] * qd;
         }
         break;
      case 21:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 1   adj(J):  J22 -J12
            //    1 3       1 2           -J21  J11
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J12 = J[i + Q * 2];
            const CeedScalar J22 = J[i + Q * 3];
            const CeedScalar w = alpha * qw[i];
            const CeedScalar wx = w * c[i + Q * 0];
            const CeedScalar wy = w * c[i + Q * 1];
            const CeedScalar qd0 =  wx * J22 - wy * J12;
            const CeedScalar qd1 = -wx * J21 + wy * J11;
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i] = qd0 * ug0 + qd1 * ug1;
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 1   adj(J):  J22 -J12
            //    1 3       1 2           -J21  J11
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J12 = J[i + Q * 2];
            const CeedScalar J22 = J[i + Q * 3];
            const CeedScalar w = alpha * qw[i];
            const CeedScalar wx = w * c[i + Q * 0];
            const CeedScalar wy = w * c[i + Q * 1];
            const CeedScalar qd0 =  wx * J22 - wy * J12;
            const CeedScalar qd1 = -wx * J21 + wy * J11;
            for (CeedInt d = 0; d < 2; d++)
            {
               const CeedScalar ug0 = ug[i + Q * (d+2*0)];
               const CeedScalar ug1 = ug[i + Q * (d+2*1)];
               vg[i + Q * d] = qd0 * ug0 + qd1 * ug1;
            }
         }
         break;
      case 31:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 3 6   qd: 0 1 2
            //    1 4 7       1 3 4
            //    2 5 8       2 4 5
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
            const CeedScalar w = alpha * qw[i];
            const CeedScalar wx = w * c[i + Q * 0];
            const CeedScalar wy = w * c[i + Q * 1];
            const CeedScalar wz = w * c[i + Q * 2];
            const CeedScalar qd0 = wx * A11 + wy * A12 + wz * A13;
            const CeedScalar qd1 = wx * A21 + wy * A22 + wz * A23;
            const CeedScalar qd2 = wx * A31 + wy * A32 + wz * A33;
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i] = qd0 * ug0 + qd1 * ug1 + qd2 * ug2;
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 3 6   qd: 0 1 2
            //    1 4 7       1 3 4
            //    2 5 8       2 4 5
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
            const CeedScalar w = alpha * qw[i];
            const CeedScalar wx = w * c[i + Q * 0];
            const CeedScalar wy = w * c[i + Q * 1];
            const CeedScalar wz = w * c[i + Q * 2];
            const CeedScalar qd0 = wx * A11 + wy * A12 + wz * A13;
            const CeedScalar qd1 = wx * A21 + wy * A22 + wz * A23;
            const CeedScalar qd2 = wx * A31 + wy * A32 + wz * A33;
            for (CeedInt d = 0; d < 3; d++)
            {
               const CeedScalar ug0 = ug[i + Q * (d+3*0)];
               const CeedScalar ug1 = ug[i + Q * (d+3*1)];
               const CeedScalar ug2 = ug[i + Q * (d+3*2)];
               vg[i + Q * d] = qd0 * ug0 + qd1 * ug1 + qd2 * ug2;
            }
         }
         break;
   }
   return CEED_ERROR_SUCCESS;
}
