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

#ifndef MFEM_LIBCEED_DIVDIV_QF_H
#define MFEM_LIBCEED_DIVDIV_QF_H

#include "../util/util_qf.h"

/// A structure used to pass additional data to f_build_divdiv and
/// f_apply_divdiv
struct DivDivContext
{
   CeedInt dim, space_dim;
   CeedScalar coeff;
};

/// libCEED QFunction for building quadrature data for a div-div operator with
/// a constant coefficient
CEED_QFUNCTION(f_build_divdiv_const)(void *ctx, CeedInt Q,
                                     const CeedScalar *const *in,
                                     CeedScalar *const *out)
{
   DivDivContext *bc = (DivDivContext *)ctx;
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute and store qw * c / det(J).
   const CeedScalar coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * coeff / J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * coeff / DetJ21(J + i, Q);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * coeff / DetJ22(J + i, Q);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * coeff / DetJ32(J + i, Q);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * coeff / DetJ33(J + i, Q);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a div-div operator with
/// a coefficient evaluated at quadrature points.
CEED_QFUNCTION(f_build_divdiv_quad)(void *ctx, CeedInt Q,
                                    const CeedScalar *const *in,
                                    CeedScalar *const *out)
{
   DivDivContext *bc = (DivDivContext *)ctx;
   // in[0] is coefficients, size (Q)
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute and store qw * c / det(J).
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * c[i] / J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * c[i] / DetJ21(J + i, Q);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * c[i] / DetJ22(J + i, Q);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * c[i] / DetJ32(J + i, Q);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * c[i] / DetJ33(J + i, Q);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a div-div operator
CEED_QFUNCTION(f_apply_divdiv)(void *ctx, CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out)
{
   // in[0], out[0] have shape [ncomp=1, Q]
   const CeedScalar *ud = in[0], *qd = in[1];
   CeedScalar *vd = out[0];
   CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
   {
      vd[i] = qd[i] * ud[i];
   }
   return 0;
}

/// libCEED QFunction for applying a div-div operator
CEED_QFUNCTION(f_apply_divdiv_mf_const)(void *ctx, CeedInt Q,
                                        const CeedScalar *const *in, CeedScalar *const *out)
{
   DivDivContext *bc = (DivDivContext *)ctx;
   // in[0], out[0] have shape [ncomp=1, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw * c / det(J).
   const CeedScalar coeff = bc->coeff;
   const CeedScalar *ud = in[0], *J = in[1], *qw = in[2];
   CeedScalar *vd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * coeff / J[i];
            vd[i] = qd * ud[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * coeff / DetJ21(J + i, Q);
            vd[i] = qd * ud[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * coeff / DetJ22(J + i, Q);
            vd[i] = qd * ud[i];
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * coeff / DetJ32(J + i, Q);
            vd[i] = qd * ud[i];
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * coeff / DetJ33(J + i, Q);
            vd[i] = qd * ud[i];
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a div-div operator
CEED_QFUNCTION(f_apply_divdiv_mf_quad)(void *ctx, CeedInt Q,
                                       const CeedScalar *const *in, CeedScalar *const *out)
{
   DivDivContext *bc = (DivDivContext *)ctx;
   // in[0], out[0] have shape [ncomp=1, Q]
   // in[0] is coefficients, size (Q)
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw * c / det(J).
   const CeedScalar *ud = in[0], *c = in[1], *J = in[2], *qw = in[3];
   CeedScalar *vd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * c[i] / J[i];
            vd[i] = qd * ud[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * c[i] / DetJ21(J + i, Q);
            vd[i] = qd * ud[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * c[i] / DetJ22(J + i, Q);
            vd[i] = qd * ud[i];
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * c[i] / DetJ32(J + i, Q);
            vd[i] = qd * ud[i];
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * c[i] / DetJ33(J + i, Q);
            vd[i] = qd * ud[i];
         }
         break;
   }
   return 0;
}

#endif // MFEM_LIBCEED_DIVDIV_QF_H
