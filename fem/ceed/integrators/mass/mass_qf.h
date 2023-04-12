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

#ifndef MFEM_LIBCEED_MASS_QF_H
#define MFEM_LIBCEED_MASS_QF_H

#include "../qf_utils.h"

/// A structure used to pass additional data to f_build_mass and f_apply_mass
struct MassContext
{
   CeedInt dim, space_dim, vdim;
   CeedScalar coeff;
};

/// libCEED Q-function for building quadrature data for a mass operator with a
/// constant coefficient
CEED_QFUNCTION(f_build_mass_const)(void *ctx, CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out)
{
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   MassContext *bc = (MassContext *)ctx;
   const CeedScalar coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = coeff * J[i] * qw[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = coeff * qw[i] * DetJ21(J + i, Q);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = coeff * qw[i] * DetJ22(J + i, Q);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = coeff * qw[i] * DetJ32(J + i, Q);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = coeff * qw[i] * DetJ33(J + i, Q);
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
   // in[0] is coefficients, size (Q)
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   MassContext *bc = (MassContext *)ctx;
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = c[i] * J[i] * qw[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = c[i] * qw[i] * DetJ21(J + i, Q);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = c[i] * qw[i] * DetJ22(J + i, Q);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = c[i] * qw[i] * DetJ32(J + i, Q);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = c[i] * qw[i] * DetJ33(J + i, Q);
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
   MassContext *bc = (MassContext *)ctx;
   const CeedScalar *u = in[0], *qd = in[1];
   CeedScalar *v = out[0];
   switch (bc->vdim)
   {
      case 1:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            v[i] = qd[i] * u[i];
         }
         break;
      case 2:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qdi = qd[i];
            CeedPragmaSIMD for (CeedInt d = 0; d < 2; d++)
            {
               v[i + d * Q] = qdi * u[i + d * Q];
            }
         }
         break;
      case 3:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qdi = qd[i];
            CeedPragmaSIMD for (CeedInt d = 0; d < 3; d++)
            {
               v[i + d * Q] = qdi * u[i + d * Q];
            }
         }
         break;
   }
   return 0;
}

/// libCEED Q-function for applying a mass operator
CEED_QFUNCTION(f_apply_mass_mf_const)(void *ctx, CeedInt Q,
                                      const CeedScalar *const *in, CeedScalar *const *out)
{
   MassContext *bc = (MassContext *)ctx;
   const CeedScalar coeff = bc->coeff;
   const CeedScalar *u = in[0], *J = in[1], *qw = in[2];
   CeedScalar *v = out[0];
   switch (100 * bc->space_dim + 10 * bc->dim + bc->vdim)
   {
      case 111:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = coeff * qw[i] * J[i];
            v[i] = qd * u[i];
         }
         break;
      case 211:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = coeff * qw[i] * DetJ21(J + i, Q);
            v[i] = qd * u[i];
         }
         break;
      case 212:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = coeff * qw[i] * DetJ21(J + i, Q);
            CeedPragmaSIMD for (CeedInt d = 0; d < 2; d++)
            {
               v[i + d * Q] = qd * u[i + d * Q];
            }
         }
         break;
      case 221:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = coeff * qw[i] * DetJ22(J + i, Q);
            v[i] = qd * u[i];
         }
         break;
      case 222:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = coeff * qw[i] * DetJ22(J + i, Q);
            CeedPragmaSIMD for (CeedInt d = 0; d < 2; d++)
            {
               v[i + d * Q] = qd * u[i + d * Q];
            }
         }
         break;
      case 321:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = coeff * qw[i] * DetJ32(J + i, Q);
            v[i] = qd * u[i];
         }
         break;
      case 323:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = coeff * qw[i] * DetJ32(J + i, Q);
            CeedPragmaSIMD for (CeedInt d = 0; d < 3; d++)
            {
               v[i + d * Q] = qd * u[i + d * Q];
            }
         }
         break;
      case 331:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = coeff * qw[i] * DetJ33(J + i, Q);
            v[i] = qd * u[i];
         }
         break;
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = coeff * qw[i] * DetJ33(J + i, Q);
            CeedPragmaSIMD for (CeedInt d = 0; d < 3; d++)
            {
               v[i + d * Q] = qd * u[i + d * Q];
            }
         }
         break;
   }
   return 0;
}

CEED_QFUNCTION(f_apply_mass_mf_quad)(void *ctx, CeedInt Q,
                                     const CeedScalar *const *in, CeedScalar *const *out)
{
   MassContext *bc = (MassContext *)ctx;
   const CeedScalar *u = in[0], *c = in[1], *J = in[2], *qw = in[3];
   CeedScalar *v = out[0];
   switch (100 * bc->space_dim + 10 * bc->dim + bc->vdim)
   {
      case 111:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = c[i] * J[i] * qw[i];
            v[i] = qd * u[i];
         }
         break;
      case 211:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = c[i] * qw[i] * DetJ21(J + i, Q);
            v[i] = qd * u[i];
         }
         break;
      case 212:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = c[i] * qw[i] * DetJ21(J + i, Q);
            CeedPragmaSIMD for (CeedInt d = 0; d < 2; d++)
            {
               v[i + d * Q] = qd * u[i + d * Q];
            }
         }
         break;
      case 221:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = c[i] * qw[i] * DetJ22(J + i, Q);
            v[i] = qd * u[i];
         }
         break;
      case 222:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = c[i] * qw[i] * DetJ22(J + i, Q);
            CeedPragmaSIMD for (CeedInt d = 0; d < 2; d++)
            {
               v[i + d * Q] = qd * u[i + d * Q];
            }
         }
         break;
      case 321:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = c[i] * qw[i] * DetJ32(J + i, Q);
            v[i] = qd * u[i];
         }
         break;
      case 323:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = c[i] * qw[i] * DetJ32(J + i, Q);
            CeedPragmaSIMD for (CeedInt d = 0; d < 3; d++)
            {
               v[i + d * Q] = qd * u[i + d * Q];
            }
         }
         break;
      case 331:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = c[i] * qw[i] * DetJ33(J + i, Q);
            v[i] = qd * u[i];
         }
         break;
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = c[i] * qw[i] * DetJ33(J + i, Q);
            CeedPragmaSIMD for (CeedInt d = 0; d < 3; d++)
            {
               v[i + d * Q] = qd * u[i + d * Q];
            }
         }
         break;
   }
   return 0;
}

#endif // MFEM_LIBCEED_MASS_QF_H
