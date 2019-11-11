// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

/// A structure used to pass additional data to f_build_diff and f_apply_diff
struct BuildContext { CeedInt dim, space_dim; CeedScalar coeff; };

/// libCEED Q-function for building quadrature data for a diffusion operator with a constant coefficient
CEED_QFUNCTION(f_build_mech)(void *ctx, CeedInt Q,
                                   const CeedScalar *const *in, CeedScalar *const *out)
{
   BuildContext *bc = (BuildContext*)ctx;
   // in[0] is ktan
   // in[1] is Jacobians with shape [dim, nc=dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
   // the symmetric part of the result.
   const CeedScalar *ktan = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (bc->dim + 10 * bc->space_dim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = ktan[i] * qw[i] / J[i];
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 1   adj(J):  J22 -J12
            //    1 3       1 2           -J21  J11
            // const CeedScalar J11 = J[i + Q * 0];
            // const CeedScalar J21 = J[i + Q * 1];
            // const CeedScalar J12 = J[i + Q * 2];
            // const CeedScalar J22 = J[i + Q * 3];
            // const CeedScalar w = qw[i] / (J11 * J22 - J21 * J12);
            // qd[i + Q * 0] =   coeff * w * (J12 * J12 + J22 * J22);
            // qd[i + Q * 1] = - coeff * w * (J11 * J12 + J21 * J22);
            // qd[i + Q * 2] =   coeff * w * (J11 * J11 + J21 * J21);
            //TODO
         }
         break;
      case 33:
         for (CeedInt q = 0; q < Q; q++)
         {
            // J: 0 3 6   qd: 0 1 2
            //    1 4 7       1 3 4
            //    2 5 8       2 4 5
            const CeedScalar J11 = J[q + Q * 0];
            const CeedScalar J21 = J[q + Q * 1];
            const CeedScalar J31 = J[q + Q * 2];
            const CeedScalar J12 = J[q + Q * 3];
            const CeedScalar J22 = J[q + Q * 4];
            const CeedScalar J32 = J[q + Q * 5];
            const CeedScalar J13 = J[q + Q * 6];
            const CeedScalar J23 = J[q + Q * 7];
            const CeedScalar J33 = J[q + Q * 8];
            const CeedScalar A11 = J22 * J33 - J23 * J32;
            const CeedScalar A12 = J13 * J32 - J12 * J33;
            const CeedScalar A13 = J12 * J23 - J13 * J22;
            const CeedScalar A21 = J23 * J31 - J21 * J33;
            const CeedScalar A22 = J11 * J33 - J13 * J31;
            const CeedScalar A23 = J13 * J21 - J11 * J23;
            const CeedScalar A31 = J21 * J32 - J22 * J31;
            const CeedScalar A32 = J12 * J31 - J11 * J32;
            const CeedScalar A33 = J11 * J22 - J12 * J21;
            const CeedScalar w = qw[q] / (J11 * A11 + J21 * A12 + J31 * A13);
            // Load ktan
            CeedScalar K[3][3][3][3];
            for (int j = 0; j < 3; ++j) {
               for (int k = 0; k < 3; ++k) {
                  for (int l = 0; l < 3; ++l) {
                     for (int m = 0; m < 3; ++m) {
                        K[j][k][l][m] = ktan [q + (j + k*3 + l*3*3 + m*3*3*3) * Q];
                     }
                  }
               }
            }
            // ktan*J^-1
            CeedScalar tmp[3][3][3][3];
            for (int j = 0; j < 3; ++j) {
               for (int k = 0; k < 3; ++k) {
                  for (int l = 0; l < 3; ++l) {
                     tmp[j][k][l][0] = K[j][k][l][0] * A11 + K[j][k][l][1] * A21 + K[j][k][l][2] * A31;
                     tmp[j][k][l][1] = K[j][k][l][0] * A12 + K[j][k][l][1] * A22 + K[j][k][l][2] * A32;
                     tmp[j][k][l][2] = K[j][k][l][0] * A13 + K[j][k][l][1] * A23 + K[j][k][l][2] * A33;
                  }
               }
            }
            // J^-T*ktan*J^-1
            for (int k = 0; k < 3; ++k) {
               for (int l = 0; l < 3; ++l) {
                  for (int n = 0; n < 3; ++n) {
                     qd[q + (0 + k*3 + l*3*3 + n*3*3*3)*Q] = w * (A11 * tmp[0][k][l][n] + A21 * tmp[1][k][l][n] + A31 * tmp[2][k][l][n]);
                     qd[q + (1 + k*3 + l*3*3 + n*3*3*3)*Q] = w * (A12 * tmp[0][k][l][n] + A22 * tmp[1][k][l][n] + A32 * tmp[2][k][l][n]);
                     qd[q + (2 + k*3 + l*3*3 + n*3*3*3)*Q] = w * (A13 * tmp[0][k][l][n] + A23 * tmp[1][k][l][n] + A33 * tmp[2][k][l][n]);
                  }
               }
            }
         }
         break;
   }
   return 0;
}

/// libCEED Q-function for applying a diff operator
CEED_QFUNCTION(f_apply_mech)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in, CeedScalar *const *out)
{
   BuildContext *bc = (BuildContext *)ctx;
   // in[0], out[0] have shape [dim, nc=1, Q]
   const CeedScalar *ug = in[0], *qd = in[1];
   CeedScalar *vg = out[0];
   switch (bc->dim)
   {
      case 1:
         for (CeedInt i = 0; i < Q; i++)
         {
            vg[i] = ug[i] * qd[i];
         }
         break;
      case 2:
         // for (CeedInt i = 0; i < Q; i++)
         // {
         //    const CeedScalar ug0 = ug[i + Q * 0];
         //    const CeedScalar ug1 = ug[i + Q * 1];
         //    vg[i + Q * 0] = qd[i + Q * 0] * ug0 + qd[i + Q * 1] * ug1;
         //    vg[i + Q * 1] = qd[i + Q * 1] * ug0 + qd[i + Q * 2] * ug1;
         // }
         break;
      case 3:
         for (CeedInt q = 0; q < Q; q++)
         {
            // Read spatial derivatives of u components
            const CeedScalar uJ[3][3]        = {{ug[q+(0+0*3)*Q],
                                                 ug[q+(0+1*3)*Q],
                                                 ug[q+(0+2*3)*Q]},
                                                {ug[q+(1+0*3)*Q],
                                                 ug[q+(1+1*3)*Q],
                                                 ug[q+(1+2*3)*Q]},
                                                {ug[q+(2+0*3)*Q],
                                                 ug[q+(2+1*3)*Q],
                                                 ug[q+(2+2*3)*Q]}
                                                };
            // Load quadrature data
            CeedScalar K[3][3][3][3];
            for (int j = 0; j < 3; ++j) {
               for (int k = 0; k < 3; ++k) {
                  for (int l = 0; l < 3; ++l) {
                     for (int m = 0; m < 3; ++m) {
                        K[j][k][l][m] = qd [q + (j + k*3 + l*3*3 + m*3*3*3) * Q];
                     }
                  }
               }
            }
            // double contraction
            for (int j = 0; j < 3; ++j) {
               for (int k = 0; k < 3; ++k) {
                  vg[q + (j+k*3)*Q] = 0.0;
                  for (int l = 0; l < 3; ++l) {
                     for (int m = 0; m < 3; ++m) {
                        vg[q + (j+k*3)*Q] += K[j][k][l][m] * uJ[m][l];
                     }
                  }
               }
            }
         }
         break;
   }
   return 0;
}

