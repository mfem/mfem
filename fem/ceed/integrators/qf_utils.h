// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LIBCEED_QF_UTILS_H
#define MFEM_LIBCEED_QF_UTILS_H

#include <math.h>

CEED_QFUNCTION_HELPER CeedScalar DetJ22(const CeedScalar *J,
                                        const CeedInt J_stride)
{
   // J: 0 2
   //    1 3
   return J[J_stride * 0] * J[J_stride * 3] - J[J_stride * 1] * J[J_stride * 2];
}

CEED_QFUNCTION_HELPER CeedScalar DetJ21(const CeedScalar *J,
                                        const CeedInt J_stride)
{
   // J: 0
   //    1
   return sqrt(J[J_stride * 0] * J[J_stride * 0] - J[J_stride * 1] * J[J_stride * 1]);
}

CEED_QFUNCTION_HELPER CeedScalar DetJ33(const CeedScalar *J,
                                        const CeedInt J_stride)
{
   // J: 0 3 6
   //    1 4 7
   //    2 5 8
   return J[J_stride * 0] * (J[J_stride * 4] * J[J_stride * 8] -
                             J[J_stride * 5] * J[J_stride * 7]) -
          J[J_stride * 1] * (J[J_stride * 3] * J[J_stride * 8] -
                             J[J_stride * 5] * J[J_stride * 6]) +
          J[J_stride * 2] * (J[J_stride * 3] * J[J_stride * 7] -
                             J[J_stride * 4] * J[J_stride * 6]);
}

CEED_QFUNCTION_HELPER CeedScalar DetJ32(const CeedScalar *J,
                                        const CeedInt J_stride)
{
   // J: 0 3
   //    1 4
   //    2 5
   const CeedScalar E = J[J_stride * 0] * J[J_stride * 0] +
                        J[J_stride * 1] * J[J_stride * 1] +
                        J[J_stride * 2] * J[J_stride * 2];
   const CeedScalar G = J[J_stride * 3] * J[J_stride * 3] +
                        J[J_stride * 4] * J[J_stride * 4] +
                        J[J_stride * 5] * J[J_stride * 5];
   const CeedScalar F = J[J_stride * 0] * J[J_stride * 3] +
                        J[J_stride * 1] * J[J_stride * 4] +
                        J[J_stride * 2] * J[J_stride * 5];
   return sqrt(E * G - F * F);
}

CEED_QFUNCTION_HELPER void MultAdjJCAdjJt22(const CeedScalar *J,
                                            const CeedInt J_stride,
                                            const CeedScalar *c,
                                            const CeedInt c_stride,
                                            const CeedInt c_comp,
                                            const CeedScalar qw,
                                            const CeedInt qd_stride,
                                            CeedScalar *qd)
{
   // compute qw/det(J) adj(J) C adj(J)^T and store the symmetric part of the result.
   // J: 0 2   adj(J):  J22 -J12   qd: 0 1
   //    1 3           -J21  J11       1 2
   const CeedScalar J11 = J[J_stride * 0];
   const CeedScalar J21 = J[J_stride * 1];
   const CeedScalar J12 = J[J_stride * 2];
   const CeedScalar J22 = J[J_stride * 3];
   const CeedScalar w = qw / (J11 * J22 - J21 * J12);
   if (c_comp == 3)  // Matrix coefficient (symmetric)
   {
      // First compute entries of R = C adj(J)^T
      // c: 0 1
      //    1 2
      const CeedScalar R11 =  c[c_stride * 0] * J22 - c[c_stride * 1] * J12;
      const CeedScalar R21 =  c[c_stride * 1] * J22 - c[c_stride * 2] * J12;
      const CeedScalar R12 = -c[c_stride * 0] * J21 + c[c_stride * 1] * J11;
      const CeedScalar R22 = -c[c_stride * 1] * J21 + c[c_stride * 2] * J11;
      qd[qd_stride * 0] = w * (J22 * R11 - J12 * R21);
      qd[qd_stride * 1] = w * (J11 * R21 - J21 * R11);
      qd[qd_stride * 2] = w * (J11 * R22 - J21 * R12);
   }
   else if (c_comp == 2)  // Vector coefficient
   {
      qd[qd_stride * 0] =  w * (c[c_stride * 0] * J12 * J12 +
                                c[c_stride * 1] * J22 * J22);
      qd[qd_stride * 1] = -w * (c[c_stride * 1] * J11 * J12 +
                                c[c_stride * 0] * J21 * J22);
      qd[qd_stride * 2] =  w * (c[c_stride * 1] * J11 * J11 +
                                c[c_stride * 0] * J21 * J21);
   }
   else  // Scalar coefficient
   {
      qd[qd_stride * 0] =  w * c[c_stride * 0] * (J12 * J12 + J22 * J22);
      qd[qd_stride * 1] = -w * c[c_stride * 0] * (J11 * J12 + J21 * J22);
      qd[qd_stride * 2] =  w * c[c_stride * 0] * (J11 * J11 + J21 * J21);
   }
}

CEED_QFUNCTION_HELPER void MultAdjJCAdjJt21(const CeedScalar *J,
                                            const CeedInt J_stride,
                                            const CeedScalar *c,
                                            const CeedInt c_stride,
                                            const CeedInt c_comp,
                                            const CeedScalar qw,
                                            const CeedInt qd_stride,
                                            CeedScalar *qd)
{
   // compute qw/det(J) adj(J) C adj(J)^T and store the symmetric part of the result.
   // J: 0   adj(J): 1/sqrt(J^T J) J^T   qd: 0
   //    1
   const CeedScalar J11 = J[J_stride * 0];
   const CeedScalar J21 = J[J_stride * 1];
   const CeedScalar d = J11 * J11 + J21 * J21;
   const CeedScalar w = qw / sqrt(d);
   if (c_comp == 3)  // Matrix coefficient (symmetric)
   {
      // First compute entries of R = C adj(J)^T
      // c: 0 1
      //    1 2
      const CeedScalar R11 = c[c_stride * 0] * J11 + c[c_stride * 1] * J21;
      const CeedScalar R21 = c[c_stride * 1] * J11 + c[c_stride * 2] * J21;
      qd[qd_stride * 0] = w * (J11 * R11 + J21 * R21) / d;
   }
   else if (c_comp == 2)  // Vector coefficient
   {
      // First compute entries of R = C adj(J)^T
      // c: 0
      //      1
      const CeedScalar R11 = c[c_stride * 0] * J11;
      const CeedScalar R21 = c[c_stride * 1] * J21;
      qd[qd_stride * 0] = w * (J11 * R11 + J21 * R21) / d;
   }
   else  // Scalar coefficient
   {
      qd[qd_stride * 0] = w * c[c_stride * 0];
   }
}

CEED_QFUNCTION_HELPER void MultAdjJCAdjJt33(const CeedScalar *J,
                                            const CeedInt J_stride,
                                            const CeedScalar *c,
                                            const CeedInt c_stride,
                                            const CeedInt c_comp,
                                            const CeedScalar qw,
                                            const CeedInt qd_stride,
                                            CeedScalar *qd)
{
   // compute qw/det(J) adj(J) C adj(J)^T and store the symmetric part of the result.
   // J: 0 3 6   qd: 0 1 2
   //    1 4 7       1 3 4
   //    2 5 8       2 4 5
   const CeedScalar J11 = J[J_stride * 0];
   const CeedScalar J21 = J[J_stride * 1];
   const CeedScalar J31 = J[J_stride * 2];
   const CeedScalar J12 = J[J_stride * 3];
   const CeedScalar J22 = J[J_stride * 4];
   const CeedScalar J32 = J[J_stride * 5];
   const CeedScalar J13 = J[J_stride * 6];
   const CeedScalar J23 = J[J_stride * 7];
   const CeedScalar J33 = J[J_stride * 8];
   const CeedScalar A11 = J22 * J33 - J23 * J32;
   const CeedScalar A12 = J13 * J32 - J12 * J33;
   const CeedScalar A13 = J12 * J23 - J13 * J22;
   const CeedScalar A21 = J23 * J31 - J21 * J33;
   const CeedScalar A22 = J11 * J33 - J13 * J31;
   const CeedScalar A23 = J13 * J21 - J11 * J23;
   const CeedScalar A31 = J21 * J32 - J22 * J31;
   const CeedScalar A32 = J12 * J31 - J11 * J32;
   const CeedScalar A33 = J11 * J22 - J12 * J21;
   const CeedScalar w = qw / (J11 * A11 + J21 * A12 + J31 * A13);
   if (c_comp == 6)  // Matrix coefficient (symmetric)
   {
      // First compute entries of R = C adj(J)^T
      // c: 0 1 2
      //    1 3 4
      //    2 4 5
      const CeedScalar R11 = c[c_stride * 0] * A11 +
                             c[c_stride * 1] * A12 +
                             c[c_stride * 2] * A13;
      const CeedScalar R12 = c[c_stride * 0] * A21 +
                             c[c_stride * 1] * A22 +
                             c[c_stride * 2] * A23;
      const CeedScalar R13 = c[c_stride * 0] * A31 +
                             c[c_stride * 1] * A32 +
                             c[c_stride * 2] * A33;
      const CeedScalar R21 = c[c_stride * 1] * A11 +
                             c[c_stride * 3] * A12 +
                             c[c_stride * 4] * A13;
      const CeedScalar R22 = c[c_stride * 1] * A21 +
                             c[c_stride * 3] * A22 +
                             c[c_stride * 4] * A23;
      const CeedScalar R23 = c[c_stride * 1] * A31 +
                             c[c_stride * 3] * A32 +
                             c[c_stride * 4] * A33;
      const CeedScalar R31 = c[c_stride * 2] * A11 +
                             c[c_stride * 4] * A12 +
                             c[c_stride * 5] * A13;
      const CeedScalar R32 = c[c_stride * 2] * A21 +
                             c[c_stride * 4] * A22 +
                             c[c_stride * 5] * A23;
      const CeedScalar R33 = c[c_stride * 2] * A31 +
                             c[c_stride * 4] * A32 +
                             c[c_stride * 5] * A33;
      qd[qd_stride * 0] = w * (A11 * R11 + A12 * R21 + A13 * R31);
      qd[qd_stride * 1] = w * (A11 * R12 + A12 * R22 + A13 * R32);
      qd[qd_stride * 2] = w * (A11 * R13 + A12 * R23 + A13 * R33);
      qd[qd_stride * 3] = w * (A21 * R12 + A22 * R22 + A23 * R32);
      qd[qd_stride * 4] = w * (A21 * R13 + A22 * R23 + A23 * R33);
      qd[qd_stride * 5] = w * (A31 * R13 + A32 * R23 + A33 * R33);
   }
   else if (c_comp == 3)  // Vector coefficient
   {
      qd[qd_stride * 0] = w * (c[c_stride * 0] * A11 * A11 +
                               c[c_stride * 1] * A12 * A12 +
                               c[c_stride * 2] * A13 * A13);
      qd[qd_stride * 1] = w * (c[c_stride * 0] * A11 * A21 +
                               c[c_stride * 1] * A12 * A22 +
                               c[c_stride * 2] * A13 * A23);
      qd[qd_stride * 2] = w * (c[c_stride * 0] * A11 * A31 +
                               c[c_stride * 1] * A12 * A32 +
                               c[c_stride * 2] * A13 * A33);
      qd[qd_stride * 3] = w * (c[c_stride * 0] * A21 * A21 +
                               c[c_stride * 1] * A22 * A22 +
                               c[c_stride * 2] * A23 * A23);
      qd[qd_stride * 4] = w * (c[c_stride * 0] * A21 * A31 +
                               c[c_stride * 1] * A22 * A32 +
                               c[c_stride * 2] * A23 * A33);
      qd[qd_stride * 5] = w * (c[c_stride * 0] * A31 * A31 +
                               c[c_stride * 1] * A32 * A32 +
                               c[c_stride * 2] * A33 * A33);
   }
   else  // Scalar coefficient
   {
      qd[qd_stride * 0] =
         w * c[c_stride * 0] * (A11 * A11 + A12 * A12 + A13 * A13);
      qd[qd_stride * 1] =
         w * c[c_stride * 0] * (A11 * A21 + A12 * A22 + A13 * A23);
      qd[qd_stride * 2] =
         w * c[c_stride * 0] * (A11 * A31 + A12 * A32 + A13 * A33);
      qd[qd_stride * 3] =
         w * c[c_stride * 0] * (A21 * A21 + A22 * A22 + A23 * A23);
      qd[qd_stride * 4] =
         w * c[c_stride * 0] * (A21 * A31 + A22 * A32 + A23 * A33);
      qd[qd_stride * 5] =
         w * c[c_stride * 0] * (A31 * A31 + A32 * A32 + A33 * A33);
   }
}

CEED_QFUNCTION_HELPER void MultAdjJCAdjJt32(const CeedScalar *J,
                                            const CeedInt J_stride,
                                            const CeedScalar *c,
                                            const CeedInt c_stride,
                                            const CeedInt c_comp,
                                            const CeedScalar qw,
                                            const CeedInt qd_stride,
                                            CeedScalar *qd)
{
   // compute qw/det(J) adj(J) C adj(J)^T and store the symmetric part of the result.
   // J: 0 3   qd: 0 1
   //    1 4       1 2
   //    2 5
   const CeedScalar J11 = J[J_stride * 0];
   const CeedScalar J21 = J[J_stride * 1];
   const CeedScalar J31 = J[J_stride * 2];
   const CeedScalar J12 = J[J_stride * 3];
   const CeedScalar J22 = J[J_stride * 4];
   const CeedScalar J32 = J[J_stride * 5];
   const CeedScalar E = J11 * J11 + J21 * J21 + J31 * J31;
   const CeedScalar G = J12 * J12 + J22 * J22 + J32 * J32;
   const CeedScalar F = J11 * J12 + J21 * J22 + J31 * J32;
   const CeedScalar d = E * G - F * F;
   const CeedScalar w = qw / sqrt(d);
   if (c_comp == 6)  // Matrix coefficient (symmetric)
   {
      // First compute entries of R = C adj(J)^T
      // c: 0 1 2
      //    1 3 4
      //    2 4 5
      const CeedScalar R11 = G * (c[c_stride * 0] * J11 +
                                  c[c_stride * 1] * J21 +
                                  c[c_stride * 2] * J31) -
                             F * (c[c_stride * 0] * J12 +
                                  c[c_stride * 1] * J22 +
                                  c[c_stride * 2] * J32);
      const CeedScalar R21 = G * (c[c_stride * 1] * J11 +
                                  c[c_stride * 3] * J21 +
                                  c[c_stride * 4] * J31) -
                             F * (c[c_stride * 1] * J12 +
                                  c[c_stride * 3] * J22 +
                                  c[c_stride * 4] * J32);
      const CeedScalar R31 = G * (c[c_stride * 2] * J11 +
                                  c[c_stride * 4] * J21 +
                                  c[c_stride * 5] * J31) -
                             F * (c[c_stride * 2] * J12 +
                                  c[c_stride * 4] * J22 +
                                  c[c_stride * 5] * J32);
      const CeedScalar R12 = E * (c[c_stride * 0] * J11 +
                                  c[c_stride * 1] * J22 +
                                  c[c_stride * 2] * J32) -
                             F * (c[c_stride * 0] * J11 +
                                  c[c_stride * 1] * J21 +
                                  c[c_stride * 2] * J31);
      const CeedScalar R22 = E * (c[c_stride * 1] * J11 +
                                  c[c_stride * 3] * J22 +
                                  c[c_stride * 4] * J32) -
                             F * (c[c_stride * 1] * J11 +
                                  c[c_stride * 3] * J21 +
                                  c[c_stride * 4] * J31);
      const CeedScalar R32 = E * (c[c_stride * 2] * J11 +
                                  c[c_stride * 4] * J22 +
                                  c[c_stride * 5] * J32) -
                             F * (c[c_stride * 2] * J11 +
                                  c[c_stride * 4] * J21 +
                                  c[c_stride * 5] * J31);
      qd[qd_stride * 0] = w * (G * (J11 * R11 + J21 * R21 + J31 * R31) -
                               F * (J12 * R11 + J22 * R21 + J32 * R31)) / d;
      qd[qd_stride * 1] = w * (G * (J11 * R12 + J21 * R22 + J31 * R32) -
                               F * (J12 * R12 + J22 * R22 + J32 * R32)) / d;
      qd[qd_stride * 2] = w * (E * (J12 * R12 + J22 * R22 + J32 * R32) -
                               F * (J11 * R12 + J21 * R22 + J32 * R31)) / d;
   }
   else if (c_comp == 3)  // Vector coefficient
   {
      // First compute entries of R = C adj(J)^T
      // c: 0
      //      1
      //        2
      const CeedScalar R11 = c[c_stride * 0] * (G * J11 - F * J12);
      const CeedScalar R21 = c[c_stride * 1] * (G * J21 - F * J22);
      const CeedScalar R31 = c[c_stride * 2] * (G * J31 - F * J32);
      const CeedScalar R12 = c[c_stride * 0] * (E * J12 - F * J11);
      const CeedScalar R22 = c[c_stride * 1] * (E * J22 - F * J21);
      const CeedScalar R32 = c[c_stride * 2] * (E * J32 - F * J31);
      qd[qd_stride * 0] = w * (G * (J11 * R11 + J21 * R21 + J31 * R31) -
                               F * (J12 * R11 + J22 * R21 + J32 * R31)) / d;
      qd[qd_stride * 1] = w * (G * (J11 * R12 + J21 * R22 + J31 * R32) -
                               F * (J12 * R12 + J22 * R22 + J32 * R32)) / d;
      qd[qd_stride * 2] = w * (E * (J12 * R12 + J22 * R22 + J32 * R32) -
                               F * (J11 * R12 + J21 * R22 + J32 * R31)) / d;
   }
   else  // Scalar coefficient
   {
      qd[qd_stride * 0] =  w * c[c_stride * 0] * G;
      qd[qd_stride * 1] = -w * c[c_stride * 0] * F;
      qd[qd_stride * 2] =  w * c[c_stride * 0] * E;
   }
}

CEED_QFUNCTION_HELPER void MultCtAdjJt22(const CeedScalar *J,
                                         const CeedInt J_stride,
                                         const CeedScalar *c,
                                         const CeedInt c_stride,
                                         const CeedScalar qw,
                                         const CeedInt qd_stride,
                                         CeedScalar *qd)
{
   // compute qw c^T adj(J)^T and store the result vector.
   // J: 0 2   adj(J):  J22 -J12
   //    1 3           -J21  J11
   const CeedScalar J11 = J[J_stride * 0];
   const CeedScalar J21 = J[J_stride * 1];
   const CeedScalar J12 = J[J_stride * 2];
   const CeedScalar J22 = J[J_stride * 3];
   const CeedScalar w1 = qw * c[c_stride * 0];
   const CeedScalar w2 = qw * c[c_stride * 1];
   qd[qd_stride * 0] =  w1 * J22 - w2 * J12;
   qd[qd_stride * 1] = -w1 * J21 + w2 * J11;
}

CEED_QFUNCTION_HELPER void MultCtAdjJt21(const CeedScalar *J,
                                         const CeedInt J_stride,
                                         const CeedScalar *c,
                                         const CeedInt c_stride,
                                         const CeedScalar qw,
                                         const CeedInt qd_stride,
                                         CeedScalar *qd)
{
   // compute qw c^T adj(J)^T and store the result vector.
   // J: 0   adj(J): 1/sqrt(J^T J) J^T
   //    1
   const CeedScalar J11 = J[J_stride * 0];
   const CeedScalar J21 = J[J_stride * 1];
   const CeedScalar w = qw / sqrt(J11 * J11 + J21 * J21);
   const CeedScalar w1 = w * c[c_stride * 0];
   const CeedScalar w2 = w * c[c_stride * 1];
   qd[qd_stride * 0] =  w1 * J11 + w2 * J21;
}

CEED_QFUNCTION_HELPER void MultCtAdjJt33(const CeedScalar *J,
                                         const CeedInt J_stride,
                                         const CeedScalar *c,
                                         const CeedInt c_stride,
                                         const CeedScalar qw,
                                         const CeedInt qd_stride,
                                         CeedScalar *qd)
{
   // compute qw c^T adj(J)^T and store the result vector.
   // J: 0 3 6
   //    1 4 7
   //    2 5 8
   const CeedScalar J11 = J[J_stride * 0];
   const CeedScalar J21 = J[J_stride * 1];
   const CeedScalar J31 = J[J_stride * 2];
   const CeedScalar J12 = J[J_stride * 3];
   const CeedScalar J22 = J[J_stride * 4];
   const CeedScalar J32 = J[J_stride * 5];
   const CeedScalar J13 = J[J_stride * 6];
   const CeedScalar J23 = J[J_stride * 7];
   const CeedScalar J33 = J[J_stride * 8];
   const CeedScalar A11 = J22 * J33 - J23 * J32;
   const CeedScalar A12 = J13 * J32 - J12 * J33;
   const CeedScalar A13 = J12 * J23 - J13 * J22;
   const CeedScalar A21 = J23 * J31 - J21 * J33;
   const CeedScalar A22 = J11 * J33 - J13 * J31;
   const CeedScalar A23 = J13 * J21 - J11 * J23;
   const CeedScalar A31 = J21 * J32 - J22 * J31;
   const CeedScalar A32 = J12 * J31 - J11 * J32;
   const CeedScalar A33 = J11 * J22 - J12 * J21;
   const CeedScalar w1 = qw * c[c_stride * 0];
   const CeedScalar w2 = qw * c[c_stride * 1];
   const CeedScalar w3 = qw * c[c_stride * 2];
   qd[qd_stride * 0] = w1 * A11 + w2 * A12 + w3 * A13;
   qd[qd_stride * 1] = w1 * A21 + w2 * A22 + w3 * A23;
   qd[qd_stride * 2] = w1 * A31 + w2 * A32 + w3 * A33;
}

CEED_QFUNCTION_HELPER void MultCtAdjJt32(const CeedScalar *J,
                                        const CeedInt J_stride,
                                        const CeedScalar *c,
                                        const CeedInt c_stride,
                                        const CeedScalar qw,
                                        const CeedInt qd_stride,
                                        CeedScalar *qd)
{
   // compute qw c^T adj(J)^T and store the result vector.
   // J: 0 3
   //    1 4
   //    2 5
   const CeedScalar J11 = J[J_stride * 0];
   const CeedScalar J21 = J[J_stride * 1];
   const CeedScalar J31 = J[J_stride * 2];
   const CeedScalar J12 = J[J_stride * 3];
   const CeedScalar J22 = J[J_stride * 4];
   const CeedScalar J32 = J[J_stride * 5];
   const CeedScalar E = J11 * J11 + J21 * J21 + J31 * J31;
   const CeedScalar G = J12 * J12 + J22 * J22 + J32 * J32;
   const CeedScalar F = J11 * J12 + J21 * J22 + J31 * J32;
   const CeedScalar A11 = G * J11 - F * J12;
   const CeedScalar A21 = E * J12 - F * J11;
   const CeedScalar A12 = G * J21 - F * J22;
   const CeedScalar A22 = E * J22 - F * J21;
   const CeedScalar A13 = G * J31 - F * J32;
   const CeedScalar A23 = E * J32 - F * J31;
   const CeedScalar w = qw / sqrt(E * G - F * F);
   const CeedScalar w1 = w * c[c_stride * 0];
   const CeedScalar w2 = w * c[c_stride * 1];
   const CeedScalar w3 = w * c[c_stride * 2];
   qd[qd_stride * 0] = w1 * A11 + w2 * A12 + w3 * A13;
   qd[qd_stride * 1] = w1 * A21 + w2 * A22 + w3 * A23;
}

CEED_QFUNCTION_HELPER void MultAdjJt22(const CeedScalar *J,
                                       const CeedInt J_stride,
                                       const CeedScalar qw,
                                       const CeedInt qd_stride,
                                       CeedScalar *qd)
{
   // compute qw adj(J)^T and store the result matrix.
   // J: 0 2   adj(J):  J22 -J12   qd: 0 2
   //    1 3           -J21  J11       1 3
   const CeedScalar J11 = J[J_stride * 0];
   const CeedScalar J21 = J[J_stride * 1];
   const CeedScalar J12 = J[J_stride * 2];
   const CeedScalar J22 = J[J_stride * 3];
   qd[qd_stride * 0] =  qw * J22;
   qd[qd_stride * 1] = -qw * J12;
   qd[qd_stride * 2] = -qw * J21;
   qd[qd_stride * 3] =  qw * J11;
}

CEED_QFUNCTION_HELPER void MultAdjJt21(const CeedScalar *J,
                                       const CeedInt J_stride,
                                       const CeedScalar qw,
                                       const CeedInt qd_stride,
                                       CeedScalar *qd)
{
   // compute qw adj(J)^T and store the result matrix.
   // J: 0   adj(J):  1/sqrt(J^T J) J^T   qd: 0
   //    1                                    1
   const CeedScalar J11 = J[J_stride * 0];
   const CeedScalar J21 = J[J_stride * 1];
   const CeedScalar w = qw / sqrt(J11 * J11 + J21 * J21);
   qd[qd_stride * 0] = w * J11;
   qd[qd_stride * 1] = w * J21;
}

CEED_QFUNCTION_HELPER void MultAdjJt33(const CeedScalar *J,
                                       const CeedInt J_stride,
                                       const CeedScalar qw,
                                       const CeedInt qd_stride,
                                       CeedScalar *qd)
{
   // compute qw adj(J)^T and store the result matrix.
   // J: 0 3 6   qd: 0 3 6
   //    1 4 7       1 4 7
   //    2 5 8       2 5 8
   const CeedScalar J11 = J[J_stride * 0];
   const CeedScalar J21 = J[J_stride * 1];
   const CeedScalar J31 = J[J_stride * 2];
   const CeedScalar J12 = J[J_stride * 3];
   const CeedScalar J22 = J[J_stride * 4];
   const CeedScalar J32 = J[J_stride * 5];
   const CeedScalar J13 = J[J_stride * 6];
   const CeedScalar J23 = J[J_stride * 7];
   const CeedScalar J33 = J[J_stride * 8];
   const CeedScalar A11 = J22 * J33 - J23 * J32;
   const CeedScalar A12 = J13 * J32 - J12 * J33;
   const CeedScalar A13 = J12 * J23 - J13 * J22;
   const CeedScalar A21 = J23 * J31 - J21 * J33;
   const CeedScalar A22 = J11 * J33 - J13 * J31;
   const CeedScalar A23 = J13 * J21 - J11 * J23;
   const CeedScalar A31 = J21 * J32 - J22 * J31;
   const CeedScalar A32 = J12 * J31 - J11 * J32;
   const CeedScalar A33 = J11 * J22 - J12 * J21;
   qd[qd_stride * 0] = qw * A11;
   qd[qd_stride * 1] = qw * A12;
   qd[qd_stride * 2] = qw * A13;
   qd[qd_stride * 3] = qw * A21;
   qd[qd_stride * 4] = qw * A22;
   qd[qd_stride * 5] = qw * A23;
   qd[qd_stride * 6] = qw * A31;
   qd[qd_stride * 7] = qw * A32;
   qd[qd_stride * 8] = qw * A33;
}

CEED_QFUNCTION_HELPER void MultAdjJt32(const CeedScalar *J,
                                       const CeedInt J_stride,
                                       const CeedScalar qw,
                                       const CeedInt qd_stride,
                                       CeedScalar *qd)
{
   // compute qw adj(J)^T and store the result matrix.
   // J: 0 3   qd: 0 3
   //    1 4       1 4
   //    2 5       2 5
   const CeedScalar J11 = J[J_stride * 0];
   const CeedScalar J21 = J[J_stride * 1];
   const CeedScalar J31 = J[J_stride * 2];
   const CeedScalar J12 = J[J_stride * 3];
   const CeedScalar J22 = J[J_stride * 4];
   const CeedScalar J32 = J[J_stride * 5];
   const CeedScalar E = J11 * J11 + J21 * J21 + J31 * J31;
   const CeedScalar G = J12 * J12 + J22 * J22 + J32 * J32;
   const CeedScalar F = J11 * J12 + J21 * J22 + J31 * J32;
   const CeedScalar A11 = G * J11 - F * J12;
   const CeedScalar A21 = E * J12 - F * J11;
   const CeedScalar A12 = G * J21 - F * J22;
   const CeedScalar A22 = E * J22 - F * J21;
   const CeedScalar A13 = G * J31 - F * J32;
   const CeedScalar A23 = E * J32 - F * J31;
   const CeedScalar w = qw / sqrt(E * G - F * F);
   qd[qd_stride * 0] = w * A11;
   qd[qd_stride * 1] = w * A12;
   qd[qd_stride * 2] = w * A13;
   qd[qd_stride * 3] = w * A21;
   qd[qd_stride * 4] = w * A22;
   qd[qd_stride * 5] = w * A23;
}

#endif // MFEM_LIBCEED_QF_UTILS_H
