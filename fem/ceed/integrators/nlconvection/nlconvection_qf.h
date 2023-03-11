// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LIBCEED_NLCONV_QF_H
#define MFEM_LIBCEED_NLCONV_QF_H

#include "../qf_utils.h"

/// A structure used to pass additional data to f_build_conv and f_apply_conv
struct NLConvectionContext
{
   CeedInt dim, space_dim;
   CeedScalar coeff;
};

/// libCEED Q-function for building quadrature data for a convection operator
/// with a constant coefficient
CEED_QFUNCTION(f_build_conv_const)(void *ctx, CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out)
{
   NLConvectionContext *bc = (NLConvectionContext *)ctx;
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute and store qw * c * adj(J)^T.
   const CeedScalar coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = coeff * qw[i] * J[i];
         }
         break;
      case 21:
         for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt21(J + i, Q, qw[i] * coeff, Q, qd + i);
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt22(J + i, Q, qw[i] * coeff, Q, qd + i);
         }
         break;
      case 32:
         for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt32(J + i, Q, qw[i] * coeff, Q, qd + i);
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt33(J + i, Q, qw[i] * coeff, Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED Q-function for building quadrature data for a convection operator
/// coefficient evaluated at quadrature points.
CEED_QFUNCTION(f_build_conv_quad)(void *ctx, CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out)
{
   NLConvectionContext *bc = (NLConvectionContext *)ctx;
   // in[0] is coefficients, size (Q)
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute and store qw * c * adj(J)^T.
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff = c[i];
            qd[i] = coeff * qw[i] * J[i];
         }
         break;
      case 21:
         for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt21(J + i, Q, qw[i] * c[i], Q, qd + i);
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt22(J + i, Q, qw[i] * c[i], Q, qd + i);
         }
         break;
      case 32:
         for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt32(J + i, Q, qw[i] * c[i], Q, qd + i);
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt33(J + i, Q, qw[i] * c[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED Q-function for applying a conv operator
CEED_QFUNCTION(f_apply_conv)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out)
{
   NLConvectionContext *bc = (NLConvectionContext *)ctx;
   // in[0] has shape [ncomp=space_dim, Q]
   // in[1] has shape [dim, ncomp=space_dim, Q]
   // out[0] has shape [ncomp=space_dim, Q]
   const CeedScalar *u = in[0], *ug = in[1], *qd = in[2];
   CeedScalar *vg = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            vg[i] = u[i] * ug[i] * qd[i];
         }
         break;
      case 21:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd00 = qd[i + Q * 0];
            const CeedScalar qd10 = qd[i + Q * 1];
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar Dxu0 = ug00 * qd00;
            const CeedScalar Dyu0 = ug00 * qd10;
            const CeedScalar Dxu1 = ug10 * qd00;
            const CeedScalar Dyu1 = ug10 * qd10;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
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
            const CeedScalar Dxu0 = ug00 * qd00 + ug01 * qd01;
            const CeedScalar Dyu0 = ug00 * qd10 + ug01 * qd11;
            const CeedScalar Dxu1 = ug10 * qd00 + ug11 * qd01;
            const CeedScalar Dyu1 = ug10 * qd10 + ug11 * qd11;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 32:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd00 = qd[i + Q * 0];
            const CeedScalar qd10 = qd[i + Q * 1];
            const CeedScalar qd20 = qd[i + Q * 2];
            const CeedScalar qd01 = qd[i + Q * 3];
            const CeedScalar qd11 = qd[i + Q * 4];
            const CeedScalar qd21 = qd[i + Q * 5];
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar u2   = u[i + Q * 2];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug20 = ug[i + Q * 2];
            const CeedScalar ug01 = ug[i + Q * 3];
            const CeedScalar ug11 = ug[i + Q * 4];
            const CeedScalar ug21 = ug[i + Q * 5];
            const CeedScalar Dxu0 = ug00 * qd00 + ug01 * qd01;
            const CeedScalar Dyu0 = ug00 * qd10 + ug01 * qd11;
            const CeedScalar Dzu0 = ug00 * qd20 + ug01 * qd21;
            const CeedScalar Dxu1 = ug10 * qd00 + ug11 * qd01;
            const CeedScalar Dyu1 = ug10 * qd10 + ug11 * qd11;
            const CeedScalar Dzu1 = ug10 * qd20 + ug11 * qd21;
            const CeedScalar Dxu2 = ug20 * qd00 + ug21 * qd01;
            const CeedScalar Dyu2 = ug20 * qd10 + ug21 * qd11;
            const CeedScalar Dzu2 = ug20 * qd20 + ug21 * qd21;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
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
            const CeedScalar Dxu0 = ug00 * qd00 + ug01 * qd01 + ug02 * qd02;
            const CeedScalar Dyu0 = ug00 * qd10 + ug01 * qd11 + ug02 * qd12;
            const CeedScalar Dzu0 = ug00 * qd20 + ug01 * qd21 + ug02 * qd22;
            const CeedScalar Dxu1 = ug10 * qd00 + ug11 * qd01 + ug12 * qd02;
            const CeedScalar Dyu1 = ug10 * qd10 + ug11 * qd11 + ug12 * qd12;
            const CeedScalar Dzu1 = ug10 * qd20 + ug11 * qd21 + ug12 * qd22;
            const CeedScalar Dxu2 = ug20 * qd00 + ug21 * qd01 + ug22 * qd02;
            const CeedScalar Dyu2 = ug20 * qd10 + ug21 * qd11 + ug22 * qd12;
            const CeedScalar Dzu2 = ug20 * qd20 + ug21 * qd21 + ug22 * qd22;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
   }
   return 0;
}

/// libCEED Q-function for applying a conv operator
CEED_QFUNCTION(f_apply_conv_mf_const)(void *ctx, CeedInt Q,
                                      const CeedScalar *const *in,
                                      CeedScalar *const *out)
{
   NLConvectionContext *bc = (NLConvectionContext *)ctx;
   // in[0] has shape [ncomp=space_dim, Q]
   // in[1] has shape [dim, ncomp=space_dim, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   // out[0] has shape [ncomp=space_dim, Q]
   //
   // At every quadrature point, compute qw * c * adj(J)^T.
   const CeedScalar coeff = bc->coeff;
   const CeedScalar *u = in[0], *ug = in[1], *J = in[2], *qw = in[3];
   CeedScalar *vg = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = coeff * qw[i] * J[i];
            vg[i] = u[i] * ug[i] * qd;
         }
         break;
      case 21:
         for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[2];
            MultAdjJt21(J + i, Q, qw[i] * coeff, 1, qd);
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar Dxu0 = ug00 * qd[0];
            const CeedScalar Dyu0 = ug00 * qd[1];
            const CeedScalar Dxu1 = ug10 * qd[0];
            const CeedScalar Dyu1 = ug10 * qd[1];
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[4];
            MultAdjJt22(J + i, Q, qw[i] * coeff, 1, qd);
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug01 = ug[i + Q * 2];
            const CeedScalar ug11 = ug[i + Q * 3];
            const CeedScalar Dxu0 = ug00 * qd[0] + ug01 * qd[2];
            const CeedScalar Dyu0 = ug00 * qd[1] + ug01 * qd[3];
            const CeedScalar Dxu1 = ug10 * qd[0] + ug11 * qd[2];
            const CeedScalar Dyu1 = ug10 * qd[1] + ug11 * qd[3];
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 32:
         for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultAdjJt32(J + i, Q, qw[i] * coeff, 1, qd);
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar u2   = u[i + Q * 2];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug20 = ug[i + Q * 2];
            const CeedScalar ug01 = ug[i + Q * 3];
            const CeedScalar ug11 = ug[i + Q * 4];
            const CeedScalar ug21 = ug[i + Q * 5];
            const CeedScalar Dxu0 = ug00 * qd[0] + ug01 * qd[3];
            const CeedScalar Dyu0 = ug00 * qd[1] + ug01 * qd[4];
            const CeedScalar Dzu0 = ug00 * qd[2] + ug01 * qd[5];
            const CeedScalar Dxu1 = ug10 * qd[0] + ug11 * qd[3];
            const CeedScalar Dyu1 = ug10 * qd[1] + ug11 * qd[4];
            const CeedScalar Dzu1 = ug10 * qd[2] + ug11 * qd[5];
            const CeedScalar Dxu2 = ug20 * qd[0] + ug21 * qd[3];
            const CeedScalar Dyu2 = ug20 * qd[1] + ug21 * qd[4];
            const CeedScalar Dzu2 = ug20 * qd[2] + ug21 * qd[5];
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[9];
            MultAdjJt33(J + i, Q, qw[i] * coeff, 1, qd);
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
            const CeedScalar Dxu0 = ug00 * qd[0] + ug01 * qd[3] + ug02 * qd[6];
            const CeedScalar Dyu0 = ug00 * qd[1] + ug01 * qd[4] + ug02 * qd[7];
            const CeedScalar Dzu0 = ug00 * qd[2] + ug01 * qd[5] + ug02 * qd[8];
            const CeedScalar Dxu1 = ug10 * qd[0] + ug11 * qd[3] + ug12 * qd[6];
            const CeedScalar Dyu1 = ug10 * qd[1] + ug11 * qd[4] + ug12 * qd[7];
            const CeedScalar Dzu1 = ug10 * qd[2] + ug11 * qd[5] + ug12 * qd[8];
            const CeedScalar Dxu2 = ug20 * qd[0] + ug21 * qd[3] + ug22 * qd[6];
            const CeedScalar Dyu2 = ug20 * qd[1] + ug21 * qd[4] + ug22 * qd[7];
            const CeedScalar Dzu2 = ug20 * qd[2] + ug21 * qd[5] + ug22 * qd[8];
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
   }
   return 0;
}

CEED_QFUNCTION(f_apply_conv_mf_quad)(void *ctx, CeedInt Q,
                                     const CeedScalar *const *in,
                                     CeedScalar *const *out)
{
   NLConvectionContext *bc = (NLConvectionContext *)ctx;
   // in[0] is coefficients, size (Q)
   // in[1] has shape [ncomp=space_dim, Q]
   // in[2] has shape [dim, ncomp=space_dim, Q]
   // in[3] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[4] is quadrature weights, size (Q)
   // out[0] has shape [ncomp=space_dim, Q]
   //
   // At every quadrature point, compute qw * c * adj(J)^T.
   const CeedScalar *c = in[0], *u = in[1], *ug = in[2], *J = in[3], *qw = in[4];
   CeedScalar *vg = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = c[i] * qw[i] * J[i];
            vg[i] = u[i] * ug[i] * qd;
         }
         break;
      case 21:
         for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[2];
            MultAdjJt21(J + i, Q, qw[i] * c[i], 1, qd);
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar Dxu0 = ug00 * qd[0];
            const CeedScalar Dyu0 = ug00 * qd[1];
            const CeedScalar Dxu1 = ug10 * qd[0];
            const CeedScalar Dyu1 = ug10 * qd[1];
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[4];
            MultAdjJt22(J + i, Q, qw[i] * c[i], 1, qd);
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug01 = ug[i + Q * 2];
            const CeedScalar ug11 = ug[i + Q * 3];
            const CeedScalar Dxu0 = ug00 * qd[0] + ug01 * qd[2];
            const CeedScalar Dyu0 = ug00 * qd[1] + ug01 * qd[3];
            const CeedScalar Dxu1 = ug10 * qd[0] + ug11 * qd[2];
            const CeedScalar Dyu1 = ug10 * qd[1] + ug11 * qd[3];
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 32:
         for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultAdjJt32(J + i, Q, qw[i] * c[i], 1, qd);
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar u2   = u[i + Q * 2];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug20 = ug[i + Q * 2];
            const CeedScalar ug01 = ug[i + Q * 3];
            const CeedScalar ug11 = ug[i + Q * 4];
            const CeedScalar ug21 = ug[i + Q * 5];
            const CeedScalar Dxu0 = ug00 * qd[0] + ug01 * qd[3];
            const CeedScalar Dyu0 = ug00 * qd[1] + ug01 * qd[4];
            const CeedScalar Dzu0 = ug00 * qd[2] + ug01 * qd[5];
            const CeedScalar Dxu1 = ug10 * qd[0] + ug11 * qd[3];
            const CeedScalar Dyu1 = ug10 * qd[1] + ug11 * qd[4];
            const CeedScalar Dzu1 = ug10 * qd[2] + ug11 * qd[5];
            const CeedScalar Dxu2 = ug20 * qd[0] + ug21 * qd[3];
            const CeedScalar Dyu2 = ug20 * qd[1] + ug21 * qd[4];
            const CeedScalar Dzu2 = ug20 * qd[2] + ug21 * qd[5];
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[9];
            MultAdjJt33(J + i, Q, qw[i] * c[i], 1, qd);
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
            const CeedScalar Dxu0 = ug00 * qd[0] + ug01 * qd[3] + ug02 * qd[6];
            const CeedScalar Dyu0 = ug00 * qd[1] + ug01 * qd[4] + ug02 * qd[7];
            const CeedScalar Dzu0 = ug00 * qd[2] + ug01 * qd[5] + ug02 * qd[8];
            const CeedScalar Dxu1 = ug10 * qd[0] + ug11 * qd[3] + ug12 * qd[6];
            const CeedScalar Dyu1 = ug10 * qd[1] + ug11 * qd[4] + ug12 * qd[7];
            const CeedScalar Dzu1 = ug10 * qd[2] + ug11 * qd[5] + ug12 * qd[8];
            const CeedScalar Dxu2 = ug20 * qd[0] + ug21 * qd[3] + ug22 * qd[6];
            const CeedScalar Dyu2 = ug20 * qd[1] + ug21 * qd[4] + ug22 * qd[7];
            const CeedScalar Dzu2 = ug20 * qd[2] + ug21 * qd[5] + ug22 * qd[8];
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
   }
   return 0;
}

#endif // MFEM_LIBCEED_NLCONV_QF_H
