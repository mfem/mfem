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

#ifndef MFEM_LIBCEED_DIVDIV_MASS_QF_H
#define MFEM_LIBCEED_DIVDIV_MASS_QF_H

#include "../util/util_qf.h"

struct DivDivMassContext
{
   CeedInt dim, space_dim;
};

/// libCEED QFunction for building quadrature data for a div-div + mass
/// operator with a scalar coefficient evaluated at quadrature points
CEED_QFUNCTION(f_build_divdiv_mass_quad_scalar)(void *ctx, CeedInt Q,
                                                const CeedScalar *const *in,
                                                CeedScalar *const *out)
{
   DivDivMassContext *bc = (DivDivMassContext *)ctx;
   // in[0] is div-div coefficients with shape [ncomp=1, Q]
   // in[1] is mass coefficients with shape [ncomp=1, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw * c / det(J) and
   // qw/det(J) J^T C J and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0], *qdm = out[0] + Q;
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
            qdd[i] = qw[i] * cd[i] / DetJ21(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ21(J + i, Q, cm + i, Q, 1, qw[i], Q, qdm + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ22(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ22(J + i, Q, cm + i, Q, 1, qw[i], Q, qdm + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ32(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ32(J + i, Q, cm + i, Q, 1, qw[i], Q, qdm + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ33(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, cm + i, Q, 1, qw[i], Q, qdm + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a div-div + mass
/// operator with a vector coefficient evaluated at quadrature points
CEED_QFUNCTION(f_build_divdiv_mass_quad_vector)(void *ctx, CeedInt Q,
                                                const CeedScalar *const *in,
                                                CeedScalar *const *out)
{
   DivDivMassContext *bc = (DivDivMassContext *)ctx;
   // in[0] is div-div coefficients with shape [ncomp=1, Q]
   // in[1] is mass coefficients with shape [ncomp=space_dim, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw * c / det(J) and
   // qw/det(J) J^T C J and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0], *qdm = out[0] + Q;
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ21(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ21(J + i, Q, cm + i, Q, 2, qw[i], Q, qdm + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ22(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ22(J + i, Q, cm + i, Q, 2, qw[i], Q, qdm + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ32(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ32(J + i, Q, cm + i, Q, 3, qw[i], Q, qdm + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ33(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, cm + i, Q, 3, qw[i], Q, qdm + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a div-div + mass
/// operator with a matrix coefficient evaluated at quadrature points
CEED_QFUNCTION(f_build_divdiv_mass_quad_matrix)(void *ctx, CeedInt Q,
                                                const CeedScalar *const *in,
                                                CeedScalar *const *out)
{
   DivDivMassContext *bc = (DivDivMassContext *)ctx;
   // in[0] is div-div coefficients with shape [ncomp=1, Q]
   // in[1] is mass coefficients with shape [ncomp=space_dim*(space_dim+1)/2, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw * c / det(J) and
   // qw/det(J) J^T C J and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0], *qdm = out[0] + Q;
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ21(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ21(J + i, Q, cm + i, Q, 3, qw[i], Q, qdm + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ22(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ22(J + i, Q, cm + i, Q, 3, qw[i], Q, qdm + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ32(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ32(J + i, Q, cm + i, Q, 6, qw[i], Q, qdm + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ33(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, cm + i, Q, 6, qw[i], Q, qdm + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a div-div + mass operator
CEED_QFUNCTION(f_apply_divdiv_mass)(void *ctx, CeedInt Q,
                                    const CeedScalar *const *in,
                                    CeedScalar *const *out)
{
   DivDivMassContext *bc = (DivDivMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1], out[1] have shape [ncomp=1, Q]
   const CeedScalar *u = in[0], *ud = in[1], *qdd = in[2], *qdm = in[2] + Q;
   CeedScalar *v = out[0], *vd = out[1];
   CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
   {
      vd[i] = qdd[i] * ud[i];
   }
   switch (bc->dim)
   {
      case 1:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            v[i] = qdm[i] * u[i];
         }
         break;
      case 2:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qdm[i + Q * 0] * u0 + qdm[i + Q * 1] * u1;
            v[i + Q * 1] = qdm[i + Q * 1] * u0 + qdm[i + Q * 2] * u1;
         }
         break;
      case 3:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qdm[i + Q * 0] * u0 + qdm[i + Q * 1] * u1 + qdm[i + Q * 2] * u2;
            v[i + Q * 1] = qdm[i + Q * 1] * u0 + qdm[i + Q * 3] * u1 + qdm[i + Q * 4] * u2;
            v[i + Q * 2] = qdm[i + Q * 2] * u0 + qdm[i + Q * 4] * u1 + qdm[i + Q * 5] * u2;
         }
         break;
   }
   return 0;
}

#endif // MFEM_LIBCEED_DIVDIV_MASS_QF_H
