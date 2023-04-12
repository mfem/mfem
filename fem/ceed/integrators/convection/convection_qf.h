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

#ifndef MFEM_LIBCEED_CONV_QF_H
#define MFEM_LIBCEED_CONV_QF_H

#include "../qf_utils.h"

#define LIBCEED_CONV_COEFF_COMP_MAX 3

/// A structure used to pass additional data to f_build_conv and f_apply_conv
struct ConvectionContext {
   CeedInt dim, space_dim;
   CeedScalar coeff[LIBCEED_CONV_COEFF_COMP_MAX];
   CeedScalar alpha;
};

/// libCEED Q-function for building quadrature data for a convection operator
/// with a constant coefficient
CEED_QFUNCTION(f_build_conv_const)(void *ctx, CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out)
{
   ConvectionContext *bc = (ConvectionContext *)ctx;
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute and store qw * α * c^T adj(J)^T.
   const CeedScalar alpha  = bc->alpha;
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            qd[i] = alpha * coeff0 * qw[i] * J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultCtAdjJt21(J + i, Q, coeff, 1, alpha * qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultCtAdjJt22(J + i, Q, coeff, 1, alpha * qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultCtAdjJt32(J + i, Q, coeff, 1, alpha * qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultCtAdjJt33(J + i, Q, coeff, 1, alpha * qw[i], Q, qd + i);
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
   ConvectionContext *bc = (ConvectionContext *)ctx;
   // in[0] is coefficients with shape [ncomp=space_dim, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute and store qw * α * c^T adj(J)^T.
   const CeedScalar alpha  = bc->alpha;
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff = c[i];
            qd[i] = alpha * coeff * qw[i] * J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultCtAdjJt21(J + i, Q, c + i, Q, alpha * qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultCtAdjJt22(J + i, Q, c + i, Q, alpha * qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultCtAdjJt32(J + i, Q, c + i, Q, alpha * qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultCtAdjJt33(J + i, Q, c + i, Q, alpha * qw[i], Q, qd + i);
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
   ConvectionContext *bc = (ConvectionContext *)ctx;
   // in[0] has shape [dim, ncomp=1, Q]
   // out[0] has shape [ncomp=1, Q]
   const CeedScalar *ug = in[0], *qd = in[1];
   CeedScalar *vg = out[0];
   switch (bc->dim)
   {
      case 1:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            vg[i] = ug[i] * qd[i];
         }
         break;
      case 2:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i] = qd[i + Q * 0] * ug0 + qd[i + Q * 1] * ug1;
         }
         break;
      case 3:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i] = qd[i + Q * 0] * ug0 + qd[i + Q * 1] * ug1 + qd[i + Q * 2] * ug2;
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
   ConvectionContext *bc = (ConvectionContext *)ctx;
   // in[0] has shape [dim, ncomp=1, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   // out[0] has shape [ncomp=1, Q]
   //
   // At every quadrature point, compute qw * α * c^T adj(J)^T.
   const CeedScalar alpha  = bc->alpha;
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *ug = in[0], *J = in[1], *qw = in[2];
   CeedScalar *vg = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            const CeedScalar qd = alpha * coeff0 * qw[i] * J[i];
            vg[i] = ug[i] * qd;
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultCtAdjJt21(J + i, Q, coeff, 1, alpha * qw[i], 1, &qd);
            vg[i] = qd * ug[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[2];
            MultCtAdjJt22(J + i, Q, coeff, 1, alpha * qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i] = qd[0] * ug0 + qd[1] * ug1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[2];
            MultCtAdjJt32(J + i, Q, coeff, 1, alpha * qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i] = qd[0] * ug0 + qd[1] * ug1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultCtAdjJt33(J + i, Q, coeff, 1, alpha * qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i] = qd[0] * ug0 + qd[1] * ug1 + qd[2] * ug2;
         }
         break;
   }
   return 0;
}

CEED_QFUNCTION(f_apply_conv_mf_quad)(void *ctx, CeedInt Q,
                                     const CeedScalar *const *in,
                                     CeedScalar *const *out)
{
   ConvectionContext *bc = (ConvectionContext *)ctx;
   // in[0] has shape [dim, ncomp=1, Q]
   // in[1] is coefficients with shape [ncomp=space_dim, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   // out[0] has shape [ncomp=1, Q]
   //
   // At every quadrature point, compute qw * α * c^T adj(J)^T.
   const CeedScalar alpha  = bc->alpha;
   const CeedScalar *ug = in[0], *c = in[1], *J = in[2], *qw = in[3];
   CeedScalar *vg = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = alpha * c[i] * qw[i] * J[i];
            vg[i] = ug[i] * qd;
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultCtAdjJt21(J + i, Q, c + i, Q, alpha * qw[i], 1, &qd);
            vg[i] = qd * ug[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[2];
            MultCtAdjJt22(J + i, Q, c + i, Q, alpha * qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i] = qd[0] * ug0 + qd[1] * ug1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[2];
            MultCtAdjJt32(J + i, Q, c + i, Q, alpha * qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i] = qd[0] * ug0 + qd[1] * ug1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultCtAdjJt33(J + i, Q, c + i, Q, alpha * qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i] = qd[0] * ug0 + qd[1] * ug1 + qd[2] * ug2;
         }
         break;
   }
   return 0;
}

#endif // MFEM_LIBCEED_CONV_QF_H
