// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

/// A structure used to pass additional data to f_build_diff and f_apply_diff
struct BuildContext { CeedInt dim, space_dim; CeedScalar coeff; };

/// libCEED Q-function for building quadrature data for a diffusion operator
/// with a constant coefficient
CEED_QFUNCTION(f_build_diff_const)(void *ctx, CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out)
{
   BuildContext *bc = (BuildContext*)ctx;
   // in[0] is Jacobians with shape [dim, nc=dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
   // the symmetric part of the result.
   const CeedScalar coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (bc->dim + 10 * bc->space_dim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = coeff * qw[i] / J[i];
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 1   adj(J):  J22 -J12
            //    1 3       1 2           -J21  J11
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J12 = J[i + Q * 2];
            const CeedScalar J22 = J[i + Q * 3];
            const CeedScalar w = qw[i] / (J11 * J22 - J21 * J12);
            qd[i + Q * 0] =   coeff * w * (J12 * J12 + J22 * J22);
            qd[i + Q * 1] = - coeff * w * (J11 * J12 + J21 * J22);
            qd[i + Q * 2] =   coeff * w * (J11 * J11 + J21 * J21);
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 3 6   qd: 0 1 2
            //    1 4 7       1 3 4
            //    2 5 8       2 4 5
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J31 = J[i + Q * 2];
            const CeedScalar J12 = J[i + Q * 3];
            const CeedScalar J22 = J[i + Q * 4];
            const CeedScalar J32 = J[i + Q * 5];
            const CeedScalar J13 = J[i + Q * 6];
            const CeedScalar J23 = J[i + Q * 7];
            const CeedScalar J33 = J[i + Q * 8];
            const CeedScalar A11 = J22 * J33 - J23 * J32;
            const CeedScalar A12 = J13 * J32 - J12 * J33;
            const CeedScalar A13 = J12 * J23 - J13 * J22;
            const CeedScalar A21 = J23 * J31 - J21 * J33;
            const CeedScalar A22 = J11 * J33 - J13 * J31;
            const CeedScalar A23 = J13 * J21 - J11 * J23;
            const CeedScalar A31 = J21 * J32 - J22 * J31;
            const CeedScalar A32 = J12 * J31 - J11 * J32;
            const CeedScalar A33 = J11 * J22 - J12 * J21;
            const CeedScalar w = qw[i] / (J11 * A11 + J21 * A12 + J31 * A13);
            qd[i + Q * 0] = coeff * w * (A11 * A11 + A12 * A12 + A13 * A13);
            qd[i + Q * 1] = coeff * w * (A11 * A21 + A12 * A22 + A13 * A23);
            qd[i + Q * 2] = coeff * w * (A11 * A31 + A12 * A32 + A13 * A33);
            qd[i + Q * 3] = coeff * w * (A21 * A21 + A22 * A22 + A23 * A23);
            qd[i + Q * 4] = coeff * w * (A21 * A31 + A22 * A32 + A23 * A33);
            qd[i + Q * 5] = coeff * w * (A31 * A31 + A32 * A32 + A33 * A33);
         }
         break;
   }
   return 0;
}

/// libCEED Q-function for building quadrature data for a diffusion operator
/// with a grid function coefficient
CEED_QFUNCTION(f_build_diff_grid)(void *ctx, CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out)
{
   BuildContext *bc = (BuildContext *)ctx;
   // in[1] is Jacobians with shape [dim, nc=dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
   // the symmetric part of the result.
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (bc->dim + 10 * bc->space_dim)
   {
      case 11:
         for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = c[i] * qw[i] / J[i];
         }
         break;
      case 22:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 2   qd: 0 1   adj(J):  J22 -J12
            //    1 3       1 2           -J21  J11
            const CeedScalar coeff = c[i];
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J12 = J[i + Q * 2];
            const CeedScalar J22 = J[i + Q * 3];
            const CeedScalar w = qw[i] / (J11 * J22 - J21 * J12);
            qd[i + Q * 0] =   coeff * w * (J12 * J12 + J22 * J22);
            qd[i + Q * 1] = - coeff * w * (J11 * J12 + J21 * J22);
            qd[i + Q * 2] =   coeff * w * (J11 * J11 + J21 * J21);
         }
         break;
      case 33:
         for (CeedInt i = 0; i < Q; i++)
         {
            // J: 0 3 6   qd: 0 1 2
            //    1 4 7       1 3 4
            //    2 5 8       2 4 5
            const CeedScalar coeff = c[i];
            const CeedScalar J11 = J[i + Q * 0];
            const CeedScalar J21 = J[i + Q * 1];
            const CeedScalar J31 = J[i + Q * 2];
            const CeedScalar J12 = J[i + Q * 3];
            const CeedScalar J22 = J[i + Q * 4];
            const CeedScalar J32 = J[i + Q * 5];
            const CeedScalar J13 = J[i + Q * 6];
            const CeedScalar J23 = J[i + Q * 7];
            const CeedScalar J33 = J[i + Q * 8];
            const CeedScalar A11 = J22 * J33 - J23 * J32;
            const CeedScalar A12 = J13 * J32 - J12 * J33;
            const CeedScalar A13 = J12 * J23 - J13 * J22;
            const CeedScalar A21 = J23 * J31 - J21 * J33;
            const CeedScalar A22 = J11 * J33 - J13 * J31;
            const CeedScalar A23 = J13 * J21 - J11 * J23;
            const CeedScalar A31 = J21 * J32 - J22 * J31;
            const CeedScalar A32 = J12 * J31 - J11 * J32;
            const CeedScalar A33 = J11 * J22 - J12 * J21;
            const CeedScalar w = qw[i] / (J11 * A11 + J21 * A12 + J31 * A13);
            qd[i + Q * 0] = coeff * w * (A11 * A11 + A12 * A12 + A13 * A13);
            qd[i + Q * 1] = coeff * w * (A11 * A21 + A12 * A22 + A13 * A23);
            qd[i + Q * 2] = coeff * w * (A11 * A31 + A12 * A32 + A13 * A33);
            qd[i + Q * 3] = coeff * w * (A21 * A21 + A22 * A22 + A23 * A23);
            qd[i + Q * 4] = coeff * w * (A21 * A31 + A22 * A32 + A23 * A33);
            qd[i + Q * 5] = coeff * w * (A31 * A31 + A32 * A32 + A33 * A33);
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
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i + Q * 0] = qd[i + Q * 0] * ug0 + qd[i + Q * 1] * ug1;
            vg[i + Q * 1] = qd[i + Q * 1] * ug0 + qd[i + Q * 2] * ug1;
         }
         break;
      case 3:
         for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i + Q * 0] = qd[i + Q * 0] * ug0 + qd[i + Q * 1] * ug1 + qd[i + Q * 2] * ug2;
            vg[i + Q * 1] = qd[i + Q * 1] * ug0 + qd[i + Q * 3] * ug1 + qd[i + Q * 4] * ug2;
            vg[i + Q * 2] = qd[i + Q * 2] * ug0 + qd[i + Q * 4] * ug1 + qd[i + Q * 5] * ug2;
         }
         break;
   }
   return 0;
}
