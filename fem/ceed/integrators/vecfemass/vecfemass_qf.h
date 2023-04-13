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

#ifndef MFEM_LIBCEED_VECFEMASS_QF_H
#define MFEM_LIBCEED_VECFEMASS_QF_H

#include "../util/util_qf.h"

#define LIBCEED_VECFEMASS_COEFF_COMP_MAX 6

/// A structure used to pass additional data to f_build_vecfemass and
/// f_apply_vecfemass
struct VectorFEMassContext
{
   CeedInt dim, space_dim, coeff_comp;
   bool is_hdiv;
   CeedScalar coeff[LIBCEED_VECFEMASS_COEFF_COMP_MAX];
};

/// libCEED QFunction for building quadrature data for a vector FE mass
/// operator with a constant coefficient
CEED_QFUNCTION(f_build_vecfemass_const)(void *ctx, CeedInt Q,
                                        const CeedScalar *const *in,
                                        CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result.
   const bool is_hdiv = bc->is_hdiv;
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
            qd[i] = qw[i] * coeff0 * (is_hdiv ? J[i] : 1.0 / J[i]);
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            (is_hdiv ? MultJtCJ21 : MultAdjJCAdjJt21)(J + i, Q, coeff, 1, coeff_comp,
                                                      qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            (is_hdiv ? MultJtCJ22 : MultAdjJCAdjJt22)(J + i, Q, coeff, 1, coeff_comp,
                                                      qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            (is_hdiv ? MultJtCJ32 : MultAdjJCAdjJt32)(J + i, Q, coeff, 1, coeff_comp,
                                                      qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            (is_hdiv ? MultJtCJ33 : MultAdjJCAdjJt33)(J + i, Q, coeff, 1, coeff_comp,
                                                      qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for a vector FE mass
/// operator with a coefficient evaluated at quadrature points.
CEED_QFUNCTION(f_build_vecfemass_quad)(void *ctx, CeedInt Q,
                                       const CeedScalar *const *in,
                                       CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is coefficients with shape [ncomp=coeff_comp, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result.
   const bool is_hdiv = bc->is_hdiv;
   const CeedInt coeff_comp = bc->coeff_comp;
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * c[i] * (is_hdiv ? J[i] : 1.0 / J[i]);
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            (is_hdiv ? MultJtCJ21 : MultAdjJCAdjJt21)(J + i, Q, c + i, Q, coeff_comp,
                                                      qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            (is_hdiv ? MultJtCJ22 : MultAdjJCAdjJt22)(J + i, Q, c + i, Q, coeff_comp,
                                                      qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            (is_hdiv ? MultJtCJ32 : MultAdjJCAdjJt32)(J + i, Q, c + i, Q, coeff_comp,
                                                      qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            (is_hdiv ? MultJtCJ33 : MultAdjJCAdjJt33)(J + i, Q, c + i, Q, coeff_comp,
                                                      qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a vector FE mass operator
CEED_QFUNCTION(f_apply_vecfemass)(void *ctx, CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   const CeedScalar *ug = in[0], *qd = in[1];
   CeedScalar *vg = out[0];
   switch (bc->dim)
   {
      case 1:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            vg[i] = qd[i] * ug[i];
         }
         break;
      case 2:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i + Q * 0] = qd[i + Q * 0] * ug0 + qd[i + Q * 1] * ug1;
            vg[i + Q * 1] = qd[i + Q * 1] * ug0 + qd[i + Q * 2] * ug1;
         }
         break;
      case 3:
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
   }
   return 0;
}

/// libCEED QFunction for applying a vector FE mass operator
CEED_QFUNCTION(f_apply_vecfemass_mf_const)(void *ctx, CeedInt Q,
                                           const CeedScalar *const *in,
                                           CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)).
   const bool is_hdiv = bc->is_hdiv;
   const CeedInt coeff_comp = bc->coeff_comp;
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *ug = in[0], *J = in[1], *qw = in[2];
   CeedScalar *vg = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            const CeedScalar qd = qw[i] * coeff0 * (is_hdiv ? J[i] : 1.0 / J[i]);
            vg[i] = qd * ug[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            (is_hdiv ? MultJtCJ21 : MultAdjJCAdjJt21)(J + i, Q, coeff, 1, coeff_comp,
                                                      qw[i], 1, &qd);
            vg[i] = qd * ug[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            (is_hdiv ? MultJtCJ22 : MultAdjJCAdjJt22)(J + i, Q, coeff, 1, coeff_comp,
                                                      qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i + Q * 0] = qd[0] * ug0 + qd[1] * ug1;
            vg[i + Q * 1] = qd[1] * ug0 + qd[2] * ug1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            (is_hdiv ? MultJtCJ32 : MultAdjJCAdjJt32)(J + i, Q, coeff, 1, coeff_comp,
                                                      qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i + Q * 0] = qd[0] * ug0 + qd[1] * ug1;
            vg[i + Q * 1] = qd[1] * ug0 + qd[2] * ug1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            (is_hdiv ? MultJtCJ33 : MultAdjJCAdjJt33)(J + i, Q, coeff, 1, coeff_comp,
                                                      qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i + Q * 0] = qd[0] * ug0 + qd[1] * ug1 + qd[2] * ug2;
            vg[i + Q * 1] = qd[1] * ug0 + qd[3] * ug1 + qd[4] * ug2;
            vg[i + Q * 2] = qd[2] * ug0 + qd[4] * ug1 + qd[5] * ug2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a vector FE mass operator
CEED_QFUNCTION(f_apply_vecfemass_mf_quad)(void *ctx, CeedInt Q,
                                          const CeedScalar *const *in,
                                          CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is coefficients with shape [ncomp=coeff_comp, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)).
   const bool is_hdiv = bc->is_hdiv;
   const CeedInt coeff_comp = bc->coeff_comp;
   const CeedScalar *ug = in[0], *c = in[1], *J = in[2], *qw = in[3];
   CeedScalar *vg = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * c[i] * (is_hdiv ? J[i] : 1.0 / J[i]);
            vg[i] = qd * ug[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            (is_hdiv ? MultJtCJ21 : MultAdjJCAdjJt21)(J + i, Q, c + i, Q, coeff_comp,
                                                      qw[i], 1, &qd);
            vg[i] = qd * ug[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            (is_hdiv ? MultJtCJ22 : MultAdjJCAdjJt22)(J + i, Q, c + i, Q, coeff_comp,
                                                      qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i + Q * 0] = qd[0] * ug0 + qd[1] * ug1;
            vg[i + Q * 1] = qd[1] * ug0 + qd[2] * ug1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            (is_hdiv ? MultJtCJ32 : MultAdjJCAdjJt32)(J + i, Q, c + i, Q, coeff_comp,
                                                      qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            vg[i + Q * 0] = qd[0] * ug0 + qd[1] * ug1;
            vg[i + Q * 1] = qd[1] * ug0 + qd[2] * ug1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            (is_hdiv ? MultJtCJ33 : MultAdjJCAdjJt33)(J + i, Q, c + i, Q, coeff_comp,
                                                      qw[i], 1, qd);
            const CeedScalar ug0 = ug[i + Q * 0];
            const CeedScalar ug1 = ug[i + Q * 1];
            const CeedScalar ug2 = ug[i + Q * 2];
            vg[i + Q * 0] = qd[0] * ug0 + qd[1] * ug1 + qd[2] * ug2;
            vg[i + Q * 1] = qd[1] * ug0 + qd[3] * ug1 + qd[4] * ug2;
            vg[i + Q * 2] = qd[2] * ug0 + qd[4] * ug1 + qd[5] * ug2;
         }
         break;
   }
   return 0;
}

#endif // MFEM_LIBCEED_VECFEMASS_QF_H
