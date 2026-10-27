// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
#include "gslib_kernel_helpers.hpp"

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

using gslib::lagrange_eval;

template<int T_D1D = 0>
static void InterpolateLocal1DKernel(const bool use_dev,
                                     const double *const gf_in,
                                     const unsigned int *const el,
                                     const double *const r,
                                     double *const int_out,
                                     const int npt,
                                     const int nfields,
                                     double *gll1D,
                                     double *lagcoeff,
                                     const unsigned int *const code,
                                     const unsigned int *const proc,
                                     const int my_rank,
                                     const int out_ordering,
                                     const int pN = 0)
{
   const int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
   const int D1D = T_D1D ? T_D1D : pN;
   const int p_Nq = D1D;
   MFEM_VERIFY(MD1 <= DofQuadLimits::MAX_D1D,
               "Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D > 0, "Polynomial order not specified.");
   MFEM_VERIFY(D1D <= MD1, "D1D exceeds the allocated workspace size.");
   // for each point of the npt points, create a thread block of size dof1Dsol
   mfem::forall_2D_switch(use_dev, npt, D1D, 1, [=] MFEM_HOST_DEVICE (int i)
   {
      // Optional mask: skip points that were not found or that belong to
      // another rank. The whole thread block returns together.
      if (code && (code[i] == CODE_NOT_FOUND || proc[i] != my_rank)) { return; }

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
         // If using GetNodalValues, ordering is NDOFS x NEL x VDIM and the
         // offset would be `el[i] * p_Nq + fld * gf_offset`.
         // R->Mult produces element vectors in NDOFS x VDIM x NEL layout.
         const int elemOffset = el[i]*nfields*p_Nq + fld*p_Nq;
         MFEM_FOREACH_THREAD(j,x,D1D)
         {
            sums[j] = wtr[j] * gf_in[elemOffset + j];
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(j,x,1)
         {
            double sumv = 0.0;
            // sum the contributions of each lagrange polynomial
            for (int jj=0; jj<D1D; ++jj)
            {
               sumv += sums[jj];
            }
            const int oidx = (out_ordering == Ordering::byNODES) ?
                             i + fld*npt : i*nfields + fld;
            int_out[oidx] = sumv;
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void FindPointsGSLIB::InterpolateLocal1(const Vector &field_in,
                                        const Array<unsigned int> &elem,
                                        const Vector &ref,
                                        Vector &field_out,
                                        int npt, int ncomp, int dof1Dsol,
                                        int out_ordering,
                                        const Array<unsigned int> *code,
                                        const Array<unsigned int> *proc,
                                        int my_rank)
{
   MFEM_VERIFY(dim == 1, "Kernel for edges only.");
   if (npt == 0) { return; }
   MFEM_ASSERT((code == nullptr) == (proc == nullptr),
               "code and proc must be given together.");
   // All arrays are accessed in the memory space selected by field_in.
   const bool use_dev = field_in.UseDevice();
   auto pfin  = field_in.Read(use_dev);
   auto pelem = elem.Read(use_dev);
   auto pref  = ref.Read(use_dev);
   // With a mask, the entries of field_out that are not evaluated are kept.
   auto pout  = code ? field_out.ReadWrite(use_dev) : field_out.Write(use_dev);
   auto pcode = code ? code->Read(use_dev) : nullptr;
   auto pproc = proc ? proc->Read(use_dev) : nullptr;
   auto pgll  = DEV.gll1d_sol.ReadWrite(use_dev);
   auto plcf  = DEV.lagcoeff_sol.ReadWrite(use_dev);
   switch (dof1Dsol)
   {
      case 2:
         InterpolateLocal1DKernel<2>(use_dev, pfin, pelem, pref, pout, npt,
                                     ncomp, pgll, plcf, pcode, pproc, my_rank,
                                     out_ordering);
         break;
      case 3:
         InterpolateLocal1DKernel<3>(use_dev, pfin, pelem, pref, pout, npt,
                                     ncomp, pgll, plcf, pcode, pproc, my_rank,
                                     out_ordering);
         break;
      case 4:
         InterpolateLocal1DKernel<4>(use_dev, pfin, pelem, pref, pout, npt,
                                     ncomp, pgll, plcf, pcode, pproc, my_rank,
                                     out_ordering);
         break;
      case 5:
         InterpolateLocal1DKernel<5>(use_dev, pfin, pelem, pref, pout, npt,
                                     ncomp, pgll, plcf, pcode, pproc, my_rank,
                                     out_ordering);
         break;
      default:
         InterpolateLocal1DKernel(use_dev, pfin, pelem, pref, pout, npt,
                                  ncomp, pgll, plcf, pcode, pproc, my_rank,
                                  out_ordering, dof1Dsol);
         break;
   }
}

#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND
#else
void FindPointsGSLIB::InterpolateLocal1(const Vector &field_in,
                                        const Array<unsigned int> &elem,
                                        const Vector &ref,
                                        Vector &field_out,
                                        int npt, int ncomp, int dof1Dsol,
                                        int out_ordering,
                                        const Array<unsigned int> *code,
                                        const Array<unsigned int> *proc,
                                        int my_rank) { }
#endif

} // namespace mfem

#endif //ifdef MFEM_USE_GSLIB
