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

#ifndef MFEM_LIBCEED_CURLCURL_QF_H
#define MFEM_LIBCEED_CURLCURL_QF_H

#include "../util/util_qf.h"

#define LIBCEED_CURLCURL_COEFF_COMP_MAX 6

/// A structure used to pass additional data to f_build_curlcurl and
/// f_apply_curlcurl
struct CurlCurlContext
{
   CeedInt dim, space_dim, curl_dim, coeff_comp;
   CeedScalar coeff[LIBCEED_CURLCURL_COEFF_COMP_MAX];
};

/// libCEED QFunction for building quadrature data for a curl-curl
/// operator with a constant coefficient
CEED_QFUNCTION(f_build_curlcurl_const)(void *ctx, CeedInt Q,
                                       const CeedScalar *const *in,
                                       CeedScalar *const *out)
{
   CurlCurlContext *bc = (CurlCurlContext *)ctx;
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J and store the
   // symmetric part of the result. In 2D, compute and store qw * c / det(J).
   const CeedInt coeff_comp = bc->coeff_comp;
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 221:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            qd[i] = qw[i] * coeff0 / DetJ22(J + i, Q);
         }
         break;
      case 321:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            qd[i] = qw[i] * coeff0 / DetJ32(J + i, Q);
         }
         break;
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, coeff, 1, coeff_comp, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a curl-curl
/// operator with a coefficient evaluated at quadrature points.
CEED_QFUNCTION(f_build_curlcurl_quad)(void *ctx, CeedInt Q,
                                      const CeedScalar *const *in,
                                      CeedScalar *const *out)
{
   CurlCurlContext *bc = (CurlCurlContext *)ctx;
   // in[0] is coefficients with shape [ncomp=coeff_comp, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J and store the
   // symmetric part of the result. In 2D, compute and store qw * c / det(J).
   const CeedInt coeff_comp = bc->coeff_comp;
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 221:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * c[i] / DetJ22(J + i, Q);
         }
         break;
      case 321:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * c[i] / DetJ32(J + i, Q);
         }
         break;
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, c + i, Q, coeff_comp, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a curl-curl operator
CEED_QFUNCTION(f_apply_curlcurl)(void *ctx, CeedInt Q,
                                 const CeedScalar *const *in,
                                 CeedScalar *const *out)
{
   CurlCurlContext *bc = (CurlCurlContext *)ctx;
   // in[0], out[0] have shape [curl_dim, ncomp=1, Q]
   const CeedScalar *uc = in[0], *qd = in[1];
   CeedScalar *vc = out[0];
   switch (10 * bc->dim + bc->curl_dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            vc[i] = qd[i] * uc[i];
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar uc0 = uc[i + Q * 0];
            const CeedScalar uc1 = uc[i + Q * 1];
            const CeedScalar uc2 = uc[i + Q * 2];
            vc[i + Q * 0] = qd[i + Q * 0] * uc0 + qd[i + Q * 1] * uc1 + qd[i + Q * 2] * uc2;
            vc[i + Q * 1] = qd[i + Q * 1] * uc0 + qd[i + Q * 3] * uc1 + qd[i + Q * 4] * uc2;
            vc[i + Q * 2] = qd[i + Q * 2] * uc0 + qd[i + Q * 4] * uc1 + qd[i + Q * 5] * uc2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a curl-curl operator
CEED_QFUNCTION(f_apply_curlcurl_mf_const)(void *ctx, CeedInt Q,
                                          const CeedScalar *const *in,
                                          CeedScalar *const *out)
{
   CurlCurlContext *bc = (CurlCurlContext *)ctx;
   // in[0], out[0] have shape [curl_dim, ncomp=1, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J.
   const CeedInt coeff_comp = bc->coeff_comp;
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *uc = in[0], *J = in[1], *qw = in[2];
   CeedScalar *vc = out[0];
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 221:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            const CeedScalar qd = qw[i] * coeff0 / DetJ22(J + i, Q);
            vc[i] = qd * uc[i];
         }
         break;
      case 321:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            const CeedScalar qd = qw[i] * coeff0 / DetJ32(J + i, Q);
            vc[i] = qd * uc[i];
         }
         break;
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultJtCJ33(J + i, Q, coeff, 1, coeff_comp, qw[i], 1, qd);
            const CeedScalar uc0 = uc[i + Q * 0];
            const CeedScalar uc1 = uc[i + Q * 1];
            const CeedScalar uc2 = uc[i + Q * 2];
            vc[i + Q * 0] = qd[0] * uc0 + qd[1] * uc1 + qd[2] * uc2;
            vc[i + Q * 1] = qd[1] * uc0 + qd[3] * uc1 + qd[4] * uc2;
            vc[i + Q * 2] = qd[2] * uc0 + qd[4] * uc1 + qd[5] * uc2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a curl-curl operator
CEED_QFUNCTION(f_apply_curlcurl_mf_quad)(void *ctx, CeedInt Q,
                                         const CeedScalar *const *in,
                                         CeedScalar *const *out)
{
   CurlCurlContext *bc = (CurlCurlContext *)ctx;
   // in[0], out[0] have shape [curl_dim, ncomp=1, Q]
   // in[1] is coefficients with shape [ncomp=coeff_comp, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J.
   const CeedInt coeff_comp = bc->coeff_comp;
   const CeedScalar *uc = in[0], *c = in[1], *J = in[2], *qw = in[3];
   CeedScalar *vc = out[0];
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 221:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * c[i] / DetJ22(J + i, Q);
            vc[i] = qd * uc[i];
         }
         break;
      case 321:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * c[i] / DetJ32(J + i, Q);
            vc[i] = qd * uc[i];
         }
         break;
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultJtCJ33(J + i, Q, c + i, Q, coeff_comp, qw[i], 1, qd);
            const CeedScalar uc0 = uc[i + Q * 0];
            const CeedScalar uc1 = uc[i + Q * 1];
            const CeedScalar uc2 = uc[i + Q * 2];
            vc[i + Q * 0] = qd[0] * uc0 + qd[1] * uc1 + qd[2] * uc2;
            vc[i + Q * 1] = qd[1] * uc0 + qd[3] * uc1 + qd[4] * uc2;
            vc[i + Q * 2] = qd[2] * uc0 + qd[4] * uc1 + qd[5] * uc2;
         }
         break;
   }
   return 0;
}

#endif // MFEM_LIBCEED_CURLCURL_QF_H
