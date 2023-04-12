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

#include "../util/util_qf.h"

/// A structure used to pass additional data to f_build_conv and f_apply_conv
struct NLConvectionContext
{
   CeedInt dim, space_dim;
   CeedScalar coeff;
};

/// libCEED QFunction for building quadrature data for a convection operator
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
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * coeff * J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt21(J + i, Q, qw[i] * coeff, Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt22(J + i, Q, qw[i] * coeff, Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt32(J + i, Q, qw[i] * coeff, Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt33(J + i, Q, qw[i] * coeff, Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a convection operator
/// with a coefficient evaluated at quadrature points.
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
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * c[i] * J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt21(J + i, Q, qw[i] * c[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt22(J + i, Q, qw[i] * c[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt32(J + i, Q, qw[i] * c[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJt33(J + i, Q, qw[i] * c[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a conv operator
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
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            vg[i] = qd[i] * u[i] * ug[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd00 = qd[i + Q * 0];
            const CeedScalar qd10 = qd[i + Q * 1];
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar Dxu0 = qd00 * ug00;
            const CeedScalar Dyu0 = qd10 * ug00;
            const CeedScalar Dxu1 = qd00 * ug10;
            const CeedScalar Dyu1 = qd10 * ug10;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
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
            const CeedScalar Dxu0 = qd00 * ug00 + qd01 * ug01;
            const CeedScalar Dyu0 = qd10 * ug00 + qd11 * ug01;
            const CeedScalar Dxu1 = qd00 * ug10 + qd01 * ug11;
            const CeedScalar Dyu1 = qd10 * ug10 + qd11 * ug11;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
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
            const CeedScalar Dxu0 = qd00 * ug00 + qd01 * ug01;
            const CeedScalar Dyu0 = qd10 * ug00 + qd11 * ug01;
            const CeedScalar Dzu0 = qd20 * ug00 + qd21 * ug01;
            const CeedScalar Dxu1 = qd00 * ug10 + qd01 * ug11;
            const CeedScalar Dyu1 = qd10 * ug10 + qd11 * ug11;
            const CeedScalar Dzu1 = qd20 * ug10 + qd21 * ug11;
            const CeedScalar Dxu2 = qd00 * ug20 + qd01 * ug21;
            const CeedScalar Dyu2 = qd10 * ug20 + qd11 * ug21;
            const CeedScalar Dzu2 = qd20 * ug20 + qd21 * ug21;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
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
            const CeedScalar Dxu0 = qd00 * ug00 + qd01 * ug01 + qd02 * ug02;
            const CeedScalar Dyu0 = qd10 * ug00 + qd11 * ug01 + qd12 * ug02;
            const CeedScalar Dzu0 = qd20 * ug00 + qd21 * ug01 + qd22 * ug02;
            const CeedScalar Dxu1 = qd00 * ug10 + qd01 * ug11 + qd02 * ug12;
            const CeedScalar Dyu1 = qd10 * ug10 + qd11 * ug11 + qd12 * ug12;
            const CeedScalar Dzu1 = qd20 * ug10 + qd21 * ug11 + qd22 * ug12;
            const CeedScalar Dxu2 = qd00 * ug20 + qd01 * ug21 + qd02 * ug22;
            const CeedScalar Dyu2 = qd10 * ug20 + qd11 * ug21 + qd12 * ug22;
            const CeedScalar Dzu2 = qd20 * ug20 + qd21 * ug21 + qd22 * ug22;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a conv operator
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
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * coeff * J[i];
            vg[i] = u[i] * qd * ug[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[2];
            MultAdjJt21(J + i, Q, qw[i] * coeff, 1, qd);
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar Dxu0 = qd[0] * ug00;
            const CeedScalar Dyu0 = qd[1] * ug00;
            const CeedScalar Dxu1 = qd[0] * ug10;
            const CeedScalar Dyu1 = qd[1] * ug10;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[4];
            MultAdjJt22(J + i, Q, qw[i] * coeff, 1, qd);
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug01 = ug[i + Q * 2];
            const CeedScalar ug11 = ug[i + Q * 3];
            const CeedScalar Dxu0 = qd[0] * ug00 + qd[2] * ug01;
            const CeedScalar Dyu0 = qd[1] * ug00 + qd[3] * ug01;
            const CeedScalar Dxu1 = qd[0] * ug10 + qd[2] * ug11;
            const CeedScalar Dyu1 = qd[1] * ug10 + qd[3] * ug11;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
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
            const CeedScalar Dxu0 = qd[0] * ug00 + qd[3] * ug01;
            const CeedScalar Dyu0 = qd[1] * ug00 + qd[4] * ug01;
            const CeedScalar Dzu0 = qd[2] * ug00 + qd[5] * ug01;
            const CeedScalar Dxu1 = qd[0] * ug10 + qd[3] * ug11;
            const CeedScalar Dyu1 = qd[1] * ug10 + qd[4] * ug11;
            const CeedScalar Dzu1 = qd[2] * ug10 + qd[5] * ug11;
            const CeedScalar Dxu2 = qd[0] * ug20 + qd[3] * ug21;
            const CeedScalar Dyu2 = qd[1] * ug20 + qd[4] * ug21;
            const CeedScalar Dzu2 = qd[2] * ug20 + qd[5] * ug21;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
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
            const CeedScalar Dxu0 = qd[0] * ug00 + qd[3] * ug01 + qd[6] * ug02;
            const CeedScalar Dyu0 = qd[1] * ug00 + qd[4] * ug01 + qd[7] * ug02;
            const CeedScalar Dzu0 = qd[2] * ug00 + qd[5] * ug01 + qd[8] * ug02;
            const CeedScalar Dxu1 = qd[0] * ug10 + qd[3] * ug11 + qd[6] * ug12;
            const CeedScalar Dyu1 = qd[1] * ug10 + qd[4] * ug11 + qd[7] * ug12;
            const CeedScalar Dzu1 = qd[2] * ug10 + qd[5] * ug11 + qd[8] * ug12;
            const CeedScalar Dxu2 = qd[0] * ug20 + qd[3] * ug21 + qd[6] * ug22;
            const CeedScalar Dyu2 = qd[1] * ug20 + qd[4] * ug21 + qd[7] * ug22;
            const CeedScalar Dzu2 = qd[2] * ug20 + qd[5] * ug21 + qd[8] * ug22;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a conv operator
CEED_QFUNCTION(f_apply_conv_mf_quad)(void *ctx, CeedInt Q,
                                     const CeedScalar *const *in,
                                     CeedScalar *const *out)
{
   NLConvectionContext *bc = (NLConvectionContext *)ctx;
   // in[0] has shape [ncomp=space_dim, Q]
   // in[1] has shape [dim, ncomp=space_dim, Q]
   // in[2] is coefficients, size (Q)
   // in[3] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[4] is quadrature weights, size (Q)
   // out[0] has shape [ncomp=space_dim, Q]
   //
   // At every quadrature point, compute qw * c * adj(J)^T.
   const CeedScalar *u = in[0], *ug = in[1], *c = in[2], *J = in[3], *qw = in[4];
   CeedScalar *vg = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * c[i] * J[i];
            vg[i] = u[i] * qd * ug[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[2];
            MultAdjJt21(J + i, Q, qw[i] * c[i], 1, qd);
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar Dxu0 = qd[0] * ug00;
            const CeedScalar Dyu0 = qd[1] * ug00;
            const CeedScalar Dxu1 = qd[0] * ug10;
            const CeedScalar Dyu1 = qd[1] * ug10;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[4];
            MultAdjJt22(J + i, Q, qw[i] * c[i], 1, qd);
            const CeedScalar u0   = u[i + Q * 0];
            const CeedScalar u1   = u[i + Q * 1];
            const CeedScalar ug00 = ug[i + Q * 0];
            const CeedScalar ug10 = ug[i + Q * 1];
            const CeedScalar ug01 = ug[i + Q * 2];
            const CeedScalar ug11 = ug[i + Q * 3];
            const CeedScalar Dxu0 = qd[0] * ug00 + qd[2] * ug01;
            const CeedScalar Dyu0 = qd[1] * ug00 + qd[3] * ug01;
            const CeedScalar Dxu1 = qd[0] * ug10 + qd[2] * ug11;
            const CeedScalar Dyu1 = qd[1] * ug10 + qd[3] * ug11;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
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
            const CeedScalar Dxu0 = qd[0] * ug00 + qd[3] * ug01;
            const CeedScalar Dyu0 = qd[1] * ug00 + qd[4] * ug01;
            const CeedScalar Dzu0 = qd[2] * ug00 + qd[5] * ug01;
            const CeedScalar Dxu1 = qd[0] * ug10 + qd[3] * ug11;
            const CeedScalar Dyu1 = qd[1] * ug10 + qd[4] * ug11;
            const CeedScalar Dzu1 = qd[2] * ug10 + qd[5] * ug11;
            const CeedScalar Dxu2 = qd[0] * ug20 + qd[3] * ug21;
            const CeedScalar Dyu2 = qd[1] * ug20 + qd[4] * ug21;
            const CeedScalar Dzu2 = qd[2] * ug20 + qd[5] * ug21;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
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
            const CeedScalar Dxu0 = qd[0] * ug00 + qd[3] * ug01 + qd[6] * ug02;
            const CeedScalar Dyu0 = qd[1] * ug00 + qd[4] * ug01 + qd[7] * ug02;
            const CeedScalar Dzu0 = qd[2] * ug00 + qd[5] * ug01 + qd[8] * ug02;
            const CeedScalar Dxu1 = qd[0] * ug10 + qd[3] * ug11 + qd[6] * ug12;
            const CeedScalar Dyu1 = qd[1] * ug10 + qd[4] * ug11 + qd[7] * ug12;
            const CeedScalar Dzu1 = qd[2] * ug10 + qd[5] * ug11 + qd[8] * ug12;
            const CeedScalar Dxu2 = qd[0] * ug20 + qd[3] * ug21 + qd[6] * ug22;
            const CeedScalar Dyu2 = qd[1] * ug20 + qd[4] * ug21 + qd[7] * ug22;
            const CeedScalar Dzu2 = qd[2] * ug20 + qd[5] * ug21 + qd[8] * ug22;
            vg[i + Q * 0] = u0 * Dxu0 + u1 * Dyu0 + u2 * Dzu0;
            vg[i + Q * 1] = u0 * Dxu1 + u1 * Dyu1 + u2 * Dzu1;
            vg[i + Q * 2] = u0 * Dxu2 + u1 * Dyu2 + u2 * Dzu2;
         }
         break;
   }
   return 0;
}

#endif // MFEM_LIBCEED_NLCONV_QF_H
