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

#ifndef MFEM_LIBCEED_DIFF_MASS_QF_H
#define MFEM_LIBCEED_DIFF_MASS_QF_H

#include "../util/util_qf.h"

struct DiffusionMassContext
{
   CeedInt dim, space_dim;
};

/// libCEED QFunction for building quadrature data for a diffusion + mass
/// operator with a scalar coefficient evaluated at quadrature points
CEED_QFUNCTION(f_build_diff_mass_quad_scalar)(void *ctx, CeedInt Q,
                                              const CeedScalar *const *in,
                                              CeedScalar *const *out)
{
   DiffusionMassContext *bc = (DiffusionMassContext *)ctx;
   // in[0] is diffusion coefficients with shape [ncomp=1, Q]
   // in[1] is mass coefficients with shape [ncomp=1, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T and
   // qw * c * det(J) and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0], *qdm = out[0] + bc->dim * (bc->dim + 1) / 2 * Q;
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / J[i];
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt21(J + i, Q, cd + i, Q, 1, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * DetJ21(J + i, Q);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, cd + i, Q, 1, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * DetJ22(J + i, Q);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, cd + i, Q, 1, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * DetJ32(J + i, Q);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, cd + i, Q, 1, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * DetJ33(J + i, Q);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a diffusion + mass
/// operator with a vector coefficient evaluated at quadrature points
CEED_QFUNCTION(f_build_diff_mass_quad_vector)(void *ctx, CeedInt Q,
                                              const CeedScalar *const *in,
                                              CeedScalar *const *out)
{
   DiffusionMassContext *bc = (DiffusionMassContext *)ctx;
   // in[0] is diffusion coefficients with shape [ncomp=space_dim, Q]
   // in[1] is mass coefficients with shape [ncomp=1, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T and
   // qw * c * det(J) and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0], *qdm = out[0] + bc->dim * (bc->dim + 1) / 2 * Q;
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt21(J + i, Q, cd + i, Q, 2, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * DetJ21(J + i, Q);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, cd + i, Q, 2, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * DetJ22(J + i, Q);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, cd + i, Q, 3, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * DetJ32(J + i, Q);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, cd + i, Q, 3, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * DetJ33(J + i, Q);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a diffusion + mass
/// operator with a matrix coefficient evaluated at quadrature points
CEED_QFUNCTION(f_build_diff_mass_quad_matrix)(void *ctx, CeedInt Q,
                                              const CeedScalar *const *in,
                                              CeedScalar *const *out)
{
   DiffusionMassContext *bc = (DiffusionMassContext *)ctx;
   // in[0] is diffusion coefficients with shape [ncomp=space_dim*(space_dim+1)/2, Q]
   // in[1] is mass coefficients with shape [ncomp=1, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T and
   // qw * c * det(J) and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0], *qdm = out[0] + bc->dim * (bc->dim + 1) / 2 * Q;
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt21(J + i, Q, cd + i, Q, 3, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * DetJ21(J + i, Q);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, cd + i, Q, 3, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * DetJ22(J + i, Q);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, cd + i, Q, 6, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * DetJ32(J + i, Q);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, cd + i, Q, 6, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdm[i] = qw[i] * cm[i] * DetJ33(J + i, Q);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a diffusion + mass operator
CEED_QFUNCTION(f_apply_diff_mass)(void *ctx, CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out)
{
   DiffusionMassContext *bc = (DiffusionMassContext *)ctx;
   // in[0], out[0] have shape [ncomp=1, Q]
   // in[1], out[1] have shape [dim, ncomp=1, Q]
   const CeedScalar *u = in[0], *ug = in[1], *qdd = in[2],
                     *qdm = in[2] + bc->dim * (bc->dim + 1) / 2 * Q;
   CeedScalar *v = out[0], *vg = out[1];
   switch (bc->dim)
   {
      case 1:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            vg[i] = qdd[i] * ug[i];
         }
         break;
      case 2:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i + Q * 0] = qdd[i + Q * 0] * ug0 + qdd[i + Q * 1] * ug1;
            vg[i + Q * 1] = qdd[i + Q * 1] * ug0 + qdd[i + Q * 2] * ug1;
         }
         break;
      case 3:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i + Q * 0] =
               qdd[i + Q * 0] * ug0 + qdd[i + Q * 1] * ug1 + qdd[i + Q * 2] * ug2;
            vg[i + Q * 1] =
               qdd[i + Q * 1] * ug0 + qdd[i + Q * 3] * ug1 + qdd[i + Q * 4] * ug2;
            vg[i + Q * 2] =
               qdd[i + Q * 2] * ug0 + qdd[i + Q * 4] * ug1 + qdd[i + Q * 5] * ug2;
         }
         break;
   }
   CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
   {
      v[i] = qdm[i] * u[i];
   }
   return 0;
}

#endif // MFEM_LIBCEED_DIFF_MASS_QF_H
