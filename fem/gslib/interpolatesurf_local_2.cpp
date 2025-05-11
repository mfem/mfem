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
#include "findptssurf_2.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../../linalg/dinvariants.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{

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

static void InterpolateSurfLocal2D_Kernel(const double *const gf_in,
                                          int *const el,        // element id containing each point
                                          double *const r,      // local (ref) coord of each point
                                          double *const int_out,
                                          const int npt,        // total number of points
                                          const int ncomp,      // number of components of the field
                                          const int nel,
                                          const int dof1Dsol,   // number of dofs per element in gf_in
                                          const int gf_offset,  // offset for each field component
                                          double *gll1D,        // GLL nodes of the dof1Dsol dofs
                                          double *lagcoeff,     // Lagrange coefficients of the dof1Dsol dofs
                                          double *infok)
{
   const int p_Nq = dof1Dsol;
   const int Nfields = ncomp;
   const int fieldOffset = gf_offset;
   const int pMax = 12;

   MFEM_VERIFY(p_Nq<=pMax, "Increase Max allowable polynomial order.");

   // for each point of the npt points, create a thread block of size dof1Dsol
   mfem::forall_2D(npt, dof1Dsol, 1, [=] MFEM_HOST_DEVICE (int i)
   {
      MFEM_SHARED double wtr[pMax];
      MFEM_SHARED double sums[pMax];

      // evaluate the Lagrange polynomials at the ref coord r[i]
      MFEM_FOREACH_THREAD(j,x,p_Nq)
      {
         lagrange_eval(wtr, r[i], j, p_Nq, gll1D, lagcoeff);
      }
      MFEM_SYNC_THREAD;

      // for each field component, sum the contributions of the dofs
      for (int fld=0; fld<Nfields; ++fld)
      {
         // offset to the nodal values of element el[i] of its component fld
         const int elemOffset = el[i]*p_Nq*Nfields + fld*p_Nq;
         // lagrange polynomial j contributes wtr[j]*gf_in[elemOffset + j] to
         // the field value at point i
         MFEM_FOREACH_THREAD(j,x,p_Nq)
         {
            sums[j] = wtr[j] * gf_in[elemOffset + j];
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(j,x,p_Nq)
         {
            if (j==0)
            {
               double sumv = 0.0;
               // sum the contributions of each lagrange polynomial
               for (int jj=0; jj<p_Nq; ++jj)
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

void FindPointsGSLIB::InterpolateSurfLocal2(const Vector &field_in,
                                            Array<int> &gsl_elem_dev_l,
                                            Vector &gsl_ref_l,
                                            Vector &field_out,
                                            int npt, int ncomp,
                                            int nel, int dof1Dsol)
{
   const int gf_offset = field_in.Size()/ncomp;
   MFEM_VERIFY(spacedim==2,"Kernel for 2D space only.");
   InterpolateSurfLocal2D_Kernel(field_in.Read(),
                                 gsl_elem_dev_l.ReadWrite(),
                                 gsl_ref_l.ReadWrite(),
                                 field_out.Write(),
                                 npt,
                                 ncomp,
                                 nel,
                                 dof1Dsol,
                                 gf_offset,
                                 DEV.gll1d_sol.ReadWrite(),
                                 DEV.lagcoeff_sol.ReadWrite(),
                                 DEV.info.ReadWrite());
}


#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND
} // namespace mfem

#endif //ifdef MFEM_USE_GSLIB
