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

#ifndef MFEM_LIBCEED_CURLCURL_MASS_QF_H
#define MFEM_LIBCEED_CURLCURL_MASS_QF_H

#include "../util/util_qf.h"

struct CurlCurlMassContext
{
   CeedInt dim, space_dim, curl_dim;
};

/// libCEED QFunction for building quadrature data for a curl-curl + mass
/// operator with scalar coefficients evaluated at quadrature points
CEED_QFUNCTION(f_build_curlcurl_mass_quad_scalar_scalar)(void *ctx, CeedInt Q,
                                                         const CeedScalar *const *in,
                                                         CeedScalar *const *out)
{
   CurlCurlMassContext *bc = (CurlCurlMassContext *)ctx;
   // in[0] is curl-curl coefficients with shape [ncomp=1, Q]
   // in[1] is mass coefficients with shape [ncomp=1, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J (3D) or
   // qw * c / det(J) (2D) and qw/det(J) adj(J) C adj(J)^T and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0],
               *qdm = out[0] + bc->curl_dim * (bc->curl_dim + 1) / 2 * Q;
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 221:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ22(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, cm + i, Q, 1, qw[i], Q, qdm + i);
         }
         break;
      case 321:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ32(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, cm + i, Q, 1, qw[i], Q, qdm + i);
         }
         break;
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, cd + i, Q, 1, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, cm + i, Q, 1, qw[i], Q, qdm + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a curl-curl + mass
/// operator with scalar and vector coefficients evaluated at quadrature points
CEED_QFUNCTION(f_build_curlcurl_mass_quad_scalar_vector)(void *ctx, CeedInt Q,
                                                         const CeedScalar *const *in,
                                                         CeedScalar *const *out)
{
   CurlCurlMassContext *bc = (CurlCurlMassContext *)ctx;
   // in[0] is curl-curl coefficients with shape [ncomp=1, Q]
   // in[1] is mass coefficients with shape [ncomp=space_dim, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J (3D) or
   // qw * c / det(J) (2D) and qw/det(J) adj(J) C adj(J)^T and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0],
               *qdm = out[0] + bc->curl_dim * (bc->curl_dim + 1) / 2 * Q;
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 221:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ22(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, cm + i, Q, 2, qw[i], Q, qdm + i);
         }
         break;
      case 321:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ32(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, cm + i, Q, 3, qw[i], Q, qdm + i);
         }
         break;
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, cd + i, Q, 1, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, cm + i, Q, 3, qw[i], Q, qdm + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a curl-curl + mass
/// operator with scalar and matrix coefficients evaluated at quadrature points
CEED_QFUNCTION(f_build_curlcurl_mass_quad_scalar_matrix)(void *ctx, CeedInt Q,
                                                         const CeedScalar *const *in,
                                                         CeedScalar *const *out)
{
   CurlCurlMassContext *bc = (CurlCurlMassContext *)ctx;
   // in[0] is curl-curl coefficients with shape [ncomp=1, Q]
   // in[1] is mass coefficients with shape [ncomp=space_dim*(space_dim+1)/2, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J (3D) or
   // qw * c / det(J) (2D) and qw/det(J) adj(J) C adj(J)^T and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0],
               *qdm = out[0] + bc->curl_dim * (bc->curl_dim + 1) / 2 * Q;
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 221:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ22(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, cm + i, Q, 3, qw[i], Q, qdm + i);
         }
         break;
      case 321:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qdd[i] = qw[i] * cd[i] / DetJ32(J + i, Q);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, cm + i, Q, 6, qw[i], Q, qdm + i);
         }
         break;
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, cd + i, Q, 1, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, cm + i, Q, 6, qw[i], Q, qdm + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a curl-curl + mass
/// operator with vector and scalar coefficients evaluated at quadrature points
CEED_QFUNCTION(f_build_curlcurl_mass_quad_vector_scalar)(void *ctx, CeedInt Q,
                                                         const CeedScalar *const *in,
                                                         CeedScalar *const *out)
{
   CurlCurlMassContext *bc = (CurlCurlMassContext *)ctx;
   // in[0] is curl-curl coefficients with shape [ncomp=space_dim, Q]
   // in[1] is mass coefficients with shape [ncomp=1, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J (3D) or
   // qw * c / det(J) (2D) and qw/det(J) adj(J) C adj(J)^T and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0],
               *qdm = out[0] + bc->curl_dim * (bc->curl_dim + 1) / 2 * Q;
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, cd + i, Q, 3, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, cm + i, Q, 1, qw[i], Q, qdm + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a curl-curl + mass
/// operator with vector coefficients evaluated at quadrature points
CEED_QFUNCTION(f_build_curlcurl_mass_quad_vector_vector)(void *ctx, CeedInt Q,
                                                         const CeedScalar *const *in,
                                                         CeedScalar *const *out)
{
   CurlCurlMassContext *bc = (CurlCurlMassContext *)ctx;
   // in[0] is curl-curl coefficients with shape [ncomp=space_dim, Q]
   // in[1] is mass coefficients with shape [ncomp=space_dim, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J (3D) or
   // qw * c / det(J) (2D) and qw/det(J) adj(J) C adj(J)^T and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0],
               *qdm = out[0] + bc->curl_dim * (bc->curl_dim + 1) / 2 * Q;
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, cd + i, Q, 3, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, cm + i, Q, 3, qw[i], Q, qdm + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a curl-curl + mass
/// operator with vector and matrix coefficients evaluated at quadrature points
CEED_QFUNCTION(f_build_curlcurl_mass_quad_vector_matrix)(void *ctx, CeedInt Q,
                                                         const CeedScalar *const *in,
                                                         CeedScalar *const *out)
{
   CurlCurlMassContext *bc = (CurlCurlMassContext *)ctx;
   // in[0] is curl-curl coefficients with shape [ncomp=space_dim, Q]
   // in[1] is mass coefficients with shape [ncomp=space_dim*(space_dim+1)/2, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J (3D) or
   // qw * c / det(J) (2D) and qw/det(J) adj(J) C adj(J)^T and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0],
               *qdm = out[0] + bc->curl_dim * (bc->curl_dim + 1) / 2 * Q;
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, cd + i, Q, 3, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, cm + i, Q, 6, qw[i], Q, qdm + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a curl-curl + mass
/// operator with matrix and scalar coefficients evaluated at quadrature points
CEED_QFUNCTION(f_build_curlcurl_mass_quad_matrix_scalar)(void *ctx, CeedInt Q,
                                                         const CeedScalar *const *in,
                                                         CeedScalar *const *out)
{
   CurlCurlMassContext *bc = (CurlCurlMassContext *)ctx;
   // in[0] is curl-curl coefficients with shape [ncomp=space_dim*(space_dim+1)/2, Q]
   // in[1] is mass coefficients with shape [ncomp=1, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J (3D) or
   // qw * c / det(J) (2D) and qw/det(J) adj(J) C adj(J)^T and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0],
               *qdm = out[0] + bc->curl_dim * (bc->curl_dim + 1) / 2 * Q;
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, cd + i, Q, 6, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, cm + i, Q, 1, qw[i], Q, qdm + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a curl-curl + mass
/// operator with matrix and vector coefficients evaluated at quadrature points
CEED_QFUNCTION(f_build_curlcurl_mass_quad_matrix_vector)(void *ctx, CeedInt Q,
                                                         const CeedScalar *const *in,
                                                         CeedScalar *const *out)
{
   CurlCurlMassContext *bc = (CurlCurlMassContext *)ctx;
   // in[0] is curl-curl coefficients with shape [ncomp=space_dim*(space_dim+1)/2, Q]
   // in[1] is mass coefficients with shape [ncomp=space_dim, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J (3D) or
   // qw * c / det(J) (2D) and qw/det(J) adj(J) C adj(J)^T and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0],
               *qdm = out[0] + bc->curl_dim * (bc->curl_dim + 1) / 2 * Q;
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, cd + i, Q, 6, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, cm + i, Q, 3, qw[i], Q, qdm + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a curl-curl + mass
/// operator with matrix coefficients evaluated at quadrature points
CEED_QFUNCTION(f_build_curlcurl_mass_quad_matrix_matrix)(void *ctx, CeedInt Q,
                                                         const CeedScalar *const *in,
                                                         CeedScalar *const *out)
{
   CurlCurlMassContext *bc = (CurlCurlMassContext *)ctx;
   // in[0] is curl-curl coefficients with shape [ncomp=space_dim*(space_dim+1)/2, Q]
   // in[1] is mass coefficients with shape [ncomp=space_dim*(space_dim+1)/2, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) J^T C J (3D) or
   // qw * c / det(J) (2D) and qw/det(J) adj(J) C adj(J)^T and store the result
   const CeedScalar *cd = in[0], *cm = in[1], *J = in[2], *qw = in[3];
   CeedScalar *qdd = out[0],
               *qdm = out[0] + bc->curl_dim * (bc->curl_dim + 1) / 2 * Q;
   switch (100 * bc->space_dim + 10 * bc->dim + bc->curl_dim)
   {
      case 333:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, cd + i, Q, 6, qw[i], Q, qdd + i);
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, cm + i, Q, 6, qw[i], Q, qdm + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a curl-curl + mass operator
CEED_QFUNCTION(f_apply_curlcurl_mass)(void *ctx, CeedInt Q,
                                      const CeedScalar *const *in,
                                      CeedScalar *const *out)
{
   CurlCurlMassContext *bc = (CurlCurlMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1], out[1] have shape [curl_dim, ncomp=1, Q]
   const CeedScalar *u = in[0], *uc = in[1], *qdd = in[2],
                     *qdm = in[2] + bc->curl_dim * (bc->curl_dim + 1) / 2 * Q;
   CeedScalar *v = out[0], *vc = out[1];
   switch (10 * bc->dim + bc->curl_dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            vc[i] = qdd[i] * uc[i];
         }
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qdm[i + Q * 0] * u0 + qdm[i + Q * 1] * u1;
            v[i + Q * 1] = qdm[i + Q * 1] * u0 + qdm[i + Q * 2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar uc0 = uc[i + Q * 0];
            const CeedScalar uc1 = uc[i + Q * 1];
            const CeedScalar uc2 = uc[i + Q * 2];
            vc[i + Q * 0] =
               qdd[i + Q * 0] * uc0 + qdd[i + Q * 1] * uc1 + qdd[i + Q * 2] * uc2;
            vc[i + Q * 1] =
               qdd[i + Q * 1] * uc0 + qdd[i + Q * 3] * uc1 + qdd[i + Q * 4] * uc2;
            vc[i + Q * 2] =
               qdd[i + Q * 2] * uc0 + qdd[i + Q * 4] * uc1 + qdd[i + Q * 5] * uc2;
         }
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

#endif // MFEM_LIBCEED_CURLCURL_MASS_QF_H
