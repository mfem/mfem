// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../gslib.hpp"
#include "findpts_2.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../../linalg/dinvariants.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{

static MFEM_HOST_DEVICE void lagrange_eval(double *p0, double x,
                                           int i, int p_Nq,
                                           double *z, double *lagrangeCoeff)
{
   double p_i = (1 << (p_Nq - 1));
   for (int j = 0; j < p_Nq; ++j)
   {
      double d_j = x - z[j];
      p_i *= j == i ? 1 : d_j;
   }
   p0[i] = lagrangeCoeff[i] * p_i;
}

static void InterpolateLocal2D_Kernel(const double *const gf_in,
                                      int *const el,
                                      double *const r,
                                      double *const int_out,
                                      const int npt,
                                      const int ncomp,
                                      const int nel,
                                      const int dof1Dsol,
                                      const int gf_offset,
                                      double *gll1D,
                                      double *lagcoeff,
                                      double *infok)
{
   const int p_Nq = dof1Dsol;
   const int Nfields = ncomp;
   const int fieldOffset = gf_offset;
   const int p_Np = p_Nq*p_Nq;
   const int pMax = 12;
   mfem::forall_2D(npt, dof1Dsol, 1, [=] MFEM_HOST_DEVICE (int i)
   {
      MFEM_SHARED double wtr[pMax];
      MFEM_SHARED double wts[pMax];
      MFEM_SHARED double sums[pMax];

      // Evaluate basis functions at the reference space coordinates
      MFEM_FOREACH_THREAD(j,x,p_Nq)
      {
         lagrange_eval(wtr, r[2 * i + 0], j, p_Nq, gll1D, lagcoeff);
         lagrange_eval(wts, r[2 * i + 1], j, p_Nq, gll1D, lagcoeff);
      }
      MFEM_SYNC_THREAD;

      for (int fld = 0; fld < Nfields; ++fld)
      {

         // field is (N^2 X NEL X VDIM)
         // const int elemOffset = el[i] * p_Np + fld * fieldOffset;
         // field is (N^2 X VDIM X NEL)
         const int elemOffset = el[i] * p_Np * Nfields + fld * p_Np;

         MFEM_FOREACH_THREAD(j,x,p_Nq)
         {
            double sum_j = 0;
            for (int k = 0; k < p_Nq; ++k)
            {
               sum_j += gf_in[elemOffset + j + k * p_Nq] * wts[k];
            }
            sums[j] = wtr[j] * sum_j;
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(j,x,p_Nq)
         {
            if (j == 0)
            {
               double sumv = 0.0;
               for (int jj = 0; jj < p_Nq; ++jj)
               {
                  sumv += sums[jj];
               }
               int_out[i + fld * npt] = sumv;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void FindPointsGSLIB::InterpolateLocal2(const Vector &field_in,
                                        Array<int> &gsl_elem_dev_l,
                                        Vector &gsl_ref_l,
                                        Vector &field_out,
                                        int npt, int ncomp,
                                        int nel, int dof1Dsol)
{
   const int gf_offset = field_in.Size()/ncomp;
   MFEM_VERIFY(dim == 2,"Kernel for 2D only.");
   InterpolateLocal2D_Kernel(field_in.Read(),
                             gsl_elem_dev_l.ReadWrite(),
                             gsl_ref_l.ReadWrite(),
                             field_out.Write(),
                             npt, ncomp, nel, dof1Dsol, gf_offset,
                             DEV.gll1dsol.ReadWrite(),
                             DEV.lagcoeffsol.ReadWrite(),
                             DEV.info.ReadWrite());
}

} // namespace mfem

#endif //ifdef MFEM_USE_GSLIB
