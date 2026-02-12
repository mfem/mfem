// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

#ifdef MFEM_USE_GSLIB

#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "gslib.h"
#ifndef GSLIB_RELEASE_VERSION //gslib v1.0.7
#define GSLIB_RELEASE_VERSION 10007
#endif
#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic pop
#endif
namespace mfem
{
#if GSLIB_RELEASE_VERSION >= 10009
#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2

static MFEM_HOST_DEVICE void lagrange_eval(double *p0, double x,
                                           int i, int p_Nq,
                                           double *z, double *lagrangeCoeff)
{
   double p_i = (1 << (p_Nq - 1));
   for (int j=0; j<p_Nq; ++j)
   {
      p_i *= j==i ? 1 : x-z[j];
   }
   p0[i] = lagrangeCoeff[i] * p_i;
}

template<int T_D1D = 0>
static void InterpolateLocal1DKernel(const double *const gf_in,
                                     int *const el,
                                     double *const r,
                                     double *const int_out,
                                     const int npt,
                                     const int nfields,
                                     const int nel,
                                     const int gf_offset,
                                     double *gll1D,
                                     double *lagcoeff,
                                     const int pN = 0)
{
   const int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
   const int D1D = T_D1D ? T_D1D : pN;
   const int p_Nq = D1D;
   MFEM_VERIFY(MD1 <= DofQuadLimits::MAX_D1D,
               "Increase Max allowable polynomial order.");
   MFEM_VERIFY(pN<=DofQuadLimits::MAX_D1D,
               "Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D != 0, "Polynomial order not specified.");
   // for each point of the npt points, create a thread block of size dof1Dsol
   mfem::forall_2D(npt, D1D, 1, [=] MFEM_HOST_DEVICE (int i)
   {
      MFEM_SHARED double wtr[MD1];
      MFEM_SHARED double sums[MD1];

      // Evaluate basis functions at the reference space coordinates
      MFEM_FOREACH_THREAD(j,x,D1D)
      {
         lagrange_eval(wtr, r[i], j, p_Nq, gll1D, lagcoeff);
      }
      MFEM_SYNC_THREAD;

      for (int fld=0; fld<nfields; ++fld)
      {
         // If using GetNodalValues, ordering is NDOFSxNELxVDIM
         // const int elemOffset = el[i] * p_Np + fld * gf_offset;
         //if using R->Mult for L -> E-Vec use below: NDOFSxVDIMxNEL
         const int elemOffset = el[i]*nfields*p_Nq + fld*p_Nq;
         MFEM_FOREACH_THREAD(j,x,D1D)
         {
            sums[j] = wtr[j] * gf_in[elemOffset + j];
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(j,x,D1D)
         {
            if (j==0)
            {
               double sumv = 0.0;
               // sum the contributions of each lagrange polynomial
               for (int jj=0; jj<D1D; ++jj)
               {
                  sumv += sums[jj];
               }
               int_out[fld*npt + i] = sumv;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void FindPointsGSLIB::InterpolateLocal1( const Vector &field_in,
                                         Array<int> &gsl_elem_dev_l,
                                         Vector &gsl_ref_l,
                                         Vector &field_out,
                                         int npt,
                                         int ncomp,
                                         int nel,
                                         int dof1Dsol )
{
   MFEM_VERIFY(dim == 1, "Kernel for edges only.");
   if (npt == 0) { return; }
   const int gf_offset = field_in.Size()/ncomp;
   bool use_dev = field_in.UseDevice();
   auto pfin = field_in.Read(use_dev);
   auto pgsl = gsl_elem_dev_l.ReadWrite(use_dev);
   auto pgslr = gsl_ref_l.ReadWrite(use_dev);
   auto pfout = field_out.Write(use_dev);
   auto pgll = DEV.gll1d_sol.ReadWrite(use_dev);
   auto plcf = DEV.lagcoeff_sol.ReadWrite(use_dev);
   switch (dof1Dsol)
   {
      case 2: return InterpolateLocal1DKernel<2>(pfin, pgsl, pgslr, pfout,
                                                    npt, ncomp, nel, gf_offset,
                                                    pgll, plcf);
      case 3: return InterpolateLocal1DKernel<3>(pfin, pgsl, pgslr, pfout,
                                                    npt, ncomp, nel, gf_offset,
                                                    pgll, plcf);
      case 4: return InterpolateLocal1DKernel<4>(pfin, pgsl, pgslr, pfout,
                                                    npt, ncomp, nel, gf_offset,
                                                    pgll, plcf);
      case 5: return InterpolateLocal1DKernel<5>(pfin, pgsl, pgslr, pfout,
                                                    npt, ncomp, nel, gf_offset,
                                                    pgll, plcf);
      default: return InterpolateLocal1DKernel(pfin, pgsl, pgslr, pfout,
                                                  npt, ncomp, nel, gf_offset,
                                                  pgll, plcf, dof1Dsol);
   }
}
#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND
#else
void FindPointsGSLIB::InterpolateLocal1(const Vector &field_in,
                                        Array<int> &gsl_elem_dev_l,
                                        Vector &gsl_ref_l,
                                        Vector &field_out,
                                        int npt, int ncomp,
                                        int nel, int dof1Dsol) {};
#endif

} // namespace mfem

#endif //ifdef MFEM_USE_GSLIB
