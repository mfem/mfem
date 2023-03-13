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

#ifndef MFEM_LIBCEED_DIFF_QF_H
#define MFEM_LIBCEED_DIFF_QF_H

#include "../qf_utils.h"

#define LIBCEED_DIFF_COEFF_COMP_MAX 6

/// A structure used to pass additional data to f_build_diff and f_apply_diff
struct DiffusionContext
{
   CeedInt dim, space_dim, vdim, coeff_comp;
   CeedScalar coeff[LIBCEED_DIFF_COEFF_COMP_MAX];
};

/// libCEED Q-function for building quadrature data for a diffusion operator
/// with a constant coefficient
CEED_QFUNCTION(f_build_diff_const)(void *ctx, CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out)
{
   DiffusionContext *bc = (DiffusionContext *)ctx;
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T and store
   // the symmetric part of the result.
   const CeedInt coeff_comp = bc->coeff_comp;
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            qd[i] = coeff0 * qw[i] / J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt21(J + i, Q, coeff, 1, coeff_comp, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, coeff, 1, coeff_comp, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, coeff, 1, coeff_comp, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, coeff, 1, coeff_comp, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED Q-function for building quadrature data for a diffusion operator
/// coefficient evaluated at quadrature points.
CEED_QFUNCTION(f_build_diff_quad)(void *ctx, CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out)
{
   DiffusionContext *bc = (DiffusionContext *)ctx;
   // in[0] is coefficients with shape [ncomp=coeff_comp, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T and store
   // the symmetric part of the result.
   const CeedInt coeff_comp = bc->coeff_comp;
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = c[i] * qw[i] / J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt21(J + i, Q, c + i, Q, coeff_comp, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, c + i, Q, coeff_comp, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, c + i, Q, coeff_comp, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, c + i, Q, coeff_comp, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED Q-function for applying a diff operator
CEED_QFUNCTION(f_apply_diff)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out)
{
   DiffusionContext *bc = (DiffusionContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=vdim, Q]
   const CeedScalar *ug = in[0], *qd = in[1];
   CeedScalar *vg = out[0];
   switch (10 * bc->dim + bc->vdim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            vg[i] = ug[i] * qd[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i + Q * 0] = qd[i + Q * 0] * ug0 + qd[i + Q * 1] * ug1;
            vg[i + Q * 1] = qd[i + Q * 1] * ug0 + qd[i + Q * 2] * ug1;
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd00 = qd[i + Q * 0];
            const CeedScalar qd01 = qd[i + Q * 1];
            const CeedScalar qd10 = qd01;
            const CeedScalar qd11 = qd[i + Q * 2];
            CeedPragmaSIMD for (CeedInt d = 0; d < 2; d++)
            {
               const CeedScalar ug0 = ug[i + Q * (d + 2 * 0)];
               const CeedScalar ug1 = ug[i + Q * (d + 2 * 1)];
               vg[i + Q * (d + 2 * 0)] = qd00 * ug0 + qd01 * ug1;
               vg[i + Q * (d + 2 * 1)] = qd10 * ug0 + qd11 * ug1;
            }
         }
         break;
      case 31:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i + Q * 0] = qd[i + Q * 0] * ug0 + qd[i + Q * 1] * ug1 + qd[i + Q * 2] * ug2;
            vg[i + Q * 1] = qd[i + Q * 1] * ug0 + qd[i + Q * 3] * ug1 + qd[i + Q * 4] * ug2;
            vg[i + Q * 2] = qd[i + Q * 2] * ug0 + qd[i + Q * 4] * ug1 + qd[i + Q * 5] * ug2;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd00 = qd[i + Q * 0];
            const CeedScalar qd01 = qd[i + Q * 1];
            const CeedScalar qd02 = qd[i + Q * 2];
            const CeedScalar qd10 = qd01;
            const CeedScalar qd11 = qd[i + Q * 3];
            const CeedScalar qd12 = qd[i + Q * 4];
            const CeedScalar qd20 = qd02;
            const CeedScalar qd21 = qd12;
            const CeedScalar qd22 = qd[i + Q * 5];
            CeedPragmaSIMD for (CeedInt d = 0; d < 3; d++)
            {
               const CeedScalar ug0 = ug[i + Q * (d + 3 * 0)];
               const CeedScalar ug1 = ug[i + Q * (d + 3 * 1)];
               const CeedScalar ug2 = ug[i + Q * (d + 3 * 2)];
               vg[i + Q * (d + 3 * 0)] = qd00 * ug0 + qd01 * ug1 + qd02 * ug2;
               vg[i + Q * (d + 3 * 1)] = qd10 * ug0 + qd11 * ug1 + qd12 * ug2;
               vg[i + Q * (d + 3 * 2)] = qd20 * ug0 + qd21 * ug1 + qd22 * ug2;
            }
         }
         break;
   }
   return 0;
}

/// libCEED Q-function for applying a diff operator
CEED_QFUNCTION(f_apply_diff_mf_const)(void *ctx, CeedInt Q,
                                      const CeedScalar *const *in,
                                      CeedScalar *const *out)
{
   DiffusionContext *bc = (DiffusionContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=vdim, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T
   const CeedInt coeff_comp = bc->coeff_comp;
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *ug = in[0], *J = in[1], *qw = in[2];
   CeedScalar *vg = out[0];
   switch (100 * bc->space_dim + 10 * bc->dim + bc->vdim)
   {
      case 111:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            const CeedScalar qd = coeff0 * qw[i] / J[i];
            vg[i] = ug[i] * qd;
         }
         break;
      case 211:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultAdjJCAdjJt21(J + i, Q, coeff, 1, coeff_comp, qw[i], 1, &qd);
            vg[i] = ug[i] * qd;
         }
         break;
      case 212:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultAdjJCAdjJt21(J + i, Q, coeff, 1, coeff_comp, qw[i], 1, &qd);
            CeedPragmaSIMD for (CeedInt d = 0; d < 2; d++)
            {
               vg[i + Q * d] = ug[i + Q * d] * qd;
            }
         }
         break;
      case 221:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt22(J + i, Q, coeff, 1, coeff_comp, qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i + Q * 0] = qd[0] * ug0 + qd[1] * ug1;
            vg[i + Q * 1] = qd[1] * ug0 + qd[2] * ug1;
         }
         break;
      case 222:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt22(J + i, Q, coeff, 1, coeff_comp, qw[i], 1, qd);
            CeedPragmaSIMD for (CeedInt d = 0; d < 2; d++)
            {
               const CeedScalar ug0 = ug[i + Q * (d + 2 * 0)];
               const CeedScalar ug1 = ug[i + Q * (d + 2 * 1)];
               vg[i + Q * (d + 2 * 0)] = qd[0] * ug0 + qd[1] * ug1;
               vg[i + Q * (d + 2 * 1)] = qd[1] * ug0 + qd[2] * ug1;
            }
         }
         break;
      case 321:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt32(J + i, Q, coeff, 1, coeff_comp, qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i + Q * 0] = qd[0] * ug0 + qd[1] * ug1;
            vg[i + Q * 1] = qd[1] * ug0 + qd[2] * ug1;
         }
         break;
      case 323:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt32(J + i, Q, coeff, 1, coeff_comp, qw[i], 1, qd);
            CeedPragmaSIMD for (CeedInt d = 0; d < 3; d++)
            {
               const CeedScalar ug0 = ug[i + Q * (d + 3 * 0)];
               const CeedScalar ug1 = ug[i + Q * (d + 3 * 1)];
               vg[i + Q * (d + 3 * 0)] = qd[0] * ug0 + qd[1] * ug1;
               vg[i + Q * (d + 3 * 1)] = qd[1] * ug0 + qd[2] * ug1;
            }
         }
         break;
      case 331:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultAdjJCAdjJt33(J + i, Q, coeff, 1, coeff_comp, qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i + Q * 0] = qd[0] * ug0 + qd[1] * ug1 + qd[2] * ug2;
            vg[i + Q * 1] = qd[1] * ug0 + qd[3] * ug1 + qd[4] * ug2;
            vg[i + Q * 2] = qd[2] * ug0 + qd[4] * ug1 + qd[5] * ug2;
         }
         break;
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultAdjJCAdjJt33(J + i, Q, coeff, 1, coeff_comp, qw[i], 1, qd);
            CeedPragmaSIMD for (CeedInt d = 0; d < 3; d++)
            {
               const CeedScalar ug0 = ug[i + Q * (d + 3 * 0)];
               const CeedScalar ug1 = ug[i + Q * (d + 3 * 1)];
               const CeedScalar ug2 = ug[i + Q * (d + 3 * 2)];
               vg[i + Q * (d + 3 * 0)] = qd[0] * ug0 + qd[1] * ug1 + qd[2] * ug2;
               vg[i + Q * (d + 3 * 1)] = qd[1] * ug0 + qd[3] * ug1 + qd[4] * ug2;
               vg[i + Q * (d + 3 * 2)] = qd[2] * ug0 + qd[4] * ug1 + qd[5] * ug2;
            }
         }
   }
   return 0;
}

CEED_QFUNCTION(f_apply_diff_mf_quad)(void *ctx, CeedInt Q,
                                     const CeedScalar *const *in,
                                     CeedScalar *const *out)
{
   DiffusionContext *bc = (DiffusionContext *)ctx;
   // in[0] is coefficients with shape [ncomp=coeff_comp, Q]
   // in[1], out[0] have shape [dim, ncomp=vdim, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T
   const CeedInt coeff_comp = bc->coeff_comp;
   const CeedScalar *c = in[0], *ug = in[1], *J = in[2], *qw = in[3];
   CeedScalar *vg = out[0];
   switch (100 * bc->space_dim + 10 * bc->dim + bc->vdim)
   {
      case 111:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = c[i] * qw[i] / J[i];
            vg[i] = ug[i] * qd;
         }
         break;
      case 211:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultAdjJCAdjJt21(J + i, Q, c + i, Q, coeff_comp, qw[i], 1, &qd);
            vg[i] = ug[i] * qd;
         }
         break;
      case 212:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultAdjJCAdjJt21(J + i, Q, c + i, Q, coeff_comp, qw[i], 1, &qd);
            CeedPragmaSIMD for (CeedInt d = 0; d < 2; d++)
            {
               vg[i + Q * d] = ug[i + Q * d] * qd;
            }
         }
      case 221:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt22(J + i, Q, c + i, Q, coeff_comp, qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i + Q * 0] = qd[0] * ug0 + qd[1] * ug1;
            vg[i + Q * 1] = qd[1] * ug0 + qd[2] * ug1;
         }
         break;
      case 222:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt22(J + i, Q, c + i, Q, coeff_comp, qw[i], 1, qd);
            CeedPragmaSIMD for (CeedInt d = 0; d < 2; d++)
            {
               const CeedScalar ug0 = ug[i + Q * (d + 2 * 0)];
               const CeedScalar ug1 = ug[i + Q * (d + 2 * 1)];
               vg[i + Q * (d + 2 * 0)] = qd[0] * ug0 + qd[1] * ug1;
               vg[i + Q * (d + 2 * 1)] = qd[1] * ug0 + qd[2] * ug1;
            }
         }
         break;
      case 321:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt32(J + i, Q, c + i, Q, coeff_comp, qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i + Q * 0] = qd[0] * ug0 + qd[1] * ug1;
            vg[i + Q * 1] = qd[1] * ug0 + qd[2] * ug1;
         }
         break;
      case 323:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt32(J + i, Q, c + i, Q, coeff_comp, qw[i], 1, qd);
            CeedPragmaSIMD for (CeedInt d = 0; d < 3; d++)
            {
               const CeedScalar ug0 = ug[i + Q * (d + 3 * 0)];
               const CeedScalar ug1 = ug[i + Q * (d + 3 * 1)];
               vg[i + Q * (d + 3 * 0)] = qd[0] * ug0 + qd[1] * ug1;
               vg[i + Q * (d + 3 * 1)] = qd[1] * ug0 + qd[2] * ug1;
            }
         }
         break;
      case 331:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultAdjJCAdjJt33(J + i, Q, c + i, Q, coeff_comp, qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i + Q * 0] = qd[0] * ug0 + qd[1] * ug1 + qd[2] * ug2;
            vg[i + Q * 1] = qd[1] * ug0 + qd[3] * ug1 + qd[4] * ug2;
            vg[i + Q * 2] = qd[2] * ug0 + qd[4] * ug1 + qd[5] * ug2;
         }
         break;
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultAdjJCAdjJt33(J + i, Q, c + i, Q, coeff_comp, qw[i], 1, qd);
            CeedPragmaSIMD for (CeedInt d = 0; d < 3; d++)
            {
               const CeedScalar ug0 = ug[i + Q * (d + 3 * 0)];
               const CeedScalar ug1 = ug[i + Q * (d + 3 * 1)];
               const CeedScalar ug2 = ug[i + Q * (d + 3 * 2)];
               vg[i + Q * (d + 3 * 0)] = qd[0] * ug0 + qd[1] * ug1 + qd[2] * ug2;
               vg[i + Q * (d + 3 * 1)] = qd[1] * ug0 + qd[3] * ug1 + qd[4] * ug2;
               vg[i + Q * (d + 3 * 2)] = qd[2] * ug0 + qd[4] * ug1 + qd[5] * ug2;
            }
         }
         break;
   }
   return 0;
}

#endif // MFEM_LIBCEED_DIFF_QF_H
