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

#include "../gslib.hpp"
#include "findpts_2.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../../linalg/dinvariants.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{
#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2
#define dlong int
#define dfloat double


////// OBBOX //////
static MFEM_HOST_DEVICE inline void lagrange_eval_first_derivative(dfloat *p0,
                                                                   dfloat x, dlong i,
                                                                   const dfloat *z, const dfloat *lagrangeCoeff, dlong p_Nr)
{
   dfloat u0 = 1, u1 = 0;
   for (dlong j = 0; j < p_Nr; ++j)
   {
      if (i != j)
      {
         dfloat d_j = 2 * (x - z[j]);
         u1 = d_j * u1 + u0;
         u0 = d_j * u0;
      }
   }
   dfloat *p1 = p0 + p_Nr;
   p0[i] = lagrangeCoeff[i] * u0;
   p1[i] = 2.0 * lagrangeCoeff[i] * u1;
}

static MFEM_HOST_DEVICE inline void lagrange_eval_second_derivative(dfloat *p0,
                                                                    dfloat x, dlong i,
                                                                    const dfloat *z, const dfloat *lagrangeCoeff, dlong p_Nr)
{
   dfloat u0 = 1, u1 = 0, u2 = 0;
   //#pragma unroll p_Nq
   for (dlong j = 0; j < p_Nr; ++j)
   {
      if (i != j)
      {
         dfloat d_j = 2 * (x - z[j]);
         u2 = d_j * u2 + u1;
         u1 = d_j * u1 + u0;
         u0 = d_j * u0;
      }
   }
   dfloat *p1 = p0 + p_Nr, *p2 = p0 + 2 * p_Nr;
   p0[i] = lagrangeCoeff[i] * u0;
   p1[i] = 2.0 * lagrangeCoeff[i] * u1;
   p2[i] = 8.0 * lagrangeCoeff[i] * u2;
}

/* positive when possibly inside */
static MFEM_HOST_DEVICE inline dfloat obbox_axis_test(const obbox_t *const b,
                                                      const dfloat x[2])
{
   dfloat test = 1;
   for (dlong d = 0; d < 2; ++d)
   {
      dfloat b_d = (x[d] - b->x[d].min) * (b->x[0].max - x[0]);
      test = test < 0 ? test : b_d;
   }
   return test;
}

/* positive when possibly inside */
static MFEM_HOST_DEVICE inline dfloat obbox_test(const obbox_t *const b,
                                                 const dfloat x[2])
{
   const dfloat bxyz = obbox_axis_test(b, x);
   if (bxyz < 0)
   {
      return bxyz;
   }
   else
   {
      dfloat dxyz[2];
      for (dlong d = 0; d < 2; ++d)
      {
         dxyz[d] = x[d] - b->c0[d];
      }
      dfloat test = 1;
      for (dlong d = 0; d < 2; ++d)
      {
         dfloat rst = 0;
         for (dlong e = 0; e < 2; ++e)
         {
            rst += b->A[d * 2 + e] * dxyz[e];
         }
         dfloat brst = (rst + 1) * (1 - rst);
         test = test < 0 ? test : brst;
      }
      return test;
   }
}

////// HASH //////
static MFEM_HOST_DEVICE inline dlong hash_index(const findptsLocalHashData_t *p,
                                                const dfloat x[2])
{
   const dlong n = p->hash_n;
   dlong sum = 0;
   for (dlong d = 2 - 1; d >= 0; --d)
   {
      sum *= n;
      dlong i = (dlong)floor((x[d] - p->bnd[d].min) * p->fac[d]);
      sum += i < 0 ? 0 : (n - 1 < i ? n - 1 : i);
   }
   return sum;
}

//// Linear algebra ////
/* A is row-major */

/* A is row-major */

static MFEM_HOST_DEVICE inline void lin_solve_2(dfloat x[2], const dfloat A[4],
                                                const dfloat y[2])
{
   const double idet = 1/(A[0]*A[3] - A[1]*A[2]);
   x[0] = idet*(A[3]*y[0] - A[1]*y[1]);
   x[1] = idet*(A[0]*y[1] - A[2]*y[0]);
}

static MFEM_HOST_DEVICE inline void lin_solve_sym_2(dfloat x[2],
                                                    const dfloat A[3], const dfloat y[2])
{
   const dfloat idet = 1 / (A[0] * A[2] - A[1] * A[1]);
   x[0] = idet * (A[2] * y[0] - A[1] * y[1]);
   x[1] = idet * (A[0] * y[1] - A[1] * y[0]);
}

static MFEM_HOST_DEVICE inline dfloat norm2(const dfloat x[2]) { return x[0] * x[0] + x[1] * x[1]; }

/* the bit structure of flags is CSSRR
   the C bit --- 1<<4 --- is set when the point is converged
   RR is 0 = 00b if r is unconstrained,
         1 = 01b if r is constrained at -1
         2 = 10b if r is constrained at +1
   SS is similarly for s constraints
*/

#define CONVERGED_FLAG (1u<<4)
#define FLAG_MASK 0x1fu

static MFEM_HOST_DEVICE inline dlong num_constrained(const dlong flags)
{
   const dlong y = flags | flags >> 1;
   return (y&1u) + (y>>2 & 1u);
}

/* assumes x = 0, 1, or 2 */
static MFEM_HOST_DEVICE inline dlong plus_1_mod_3(const dlong x)
{
   return ((x | x >> 1) + 1) & 3u;
}

static MFEM_HOST_DEVICE inline dlong plus_2_mod_3(const dlong x)
{
   const dlong y = (x - 1) & 3u;
   return y ^ (y >> 1);
}

static MFEM_HOST_DEVICE inline dlong plus_1_mod_2(const dlong x)
{
   return x ^ 1u;
}

/* assumes x = 1 << i, with i < 4, returns i+1 */
static MFEM_HOST_DEVICE inline dlong which_bit(const dlong x)
{
   const dlong y = x & 7u;
   return (y-(y>>2)) | ((x-1)&4u);
}

static MFEM_HOST_DEVICE inline dlong edge_index(const dlong x)
{
   return which_bit(x)-1;
}

static MFEM_HOST_DEVICE inline dlong point_index(const dlong x)
{
   return ((x>>1)&1u) | ((x>>2)&2u);
}

// compute (x,y) and (dxdn, dydx) data for all DOFs along the edge based on
// edge index.
static MFEM_HOST_DEVICE inline findptsElementGEdge_t
get_edge(const dfloat *elx[2], const dfloat *wtend[2], dlong ei,
         dfloat *workspace,
         dlong &side_init, dlong j, dlong p_Nr)
{
   findptsElementGEdge_t edge;
   const dlong dn = ei>>1, de = plus_1_mod_2(dn);
   const dlong jidx = ei >= 2 ? j : ei*(p_Nr-1);
   const dlong kidx = ei >= 2 ? (ei-2)*(p_Nr-1) : j;

   // location of derivatives based on whether we want at r/s=-1 or r/s=+1
   // ei == 0 and 2 are constrained at -1, 1 and 3 are constrained at +1.
   const dfloat *wt1 = wtend[0] + (ei%2==0 ? 0 : 1)* p_Nr * 3 + p_Nr;

   for (dlong d = 0; d < 2; ++d)
   {
      edge.x[d] = workspace + d * p_Nr; //x & y coordinates of DOFS along edge
      edge.dxdn[d] = workspace + (2 + d) * p_Nr; //dxdn and dydn at DOFs along edge
   }

   const dlong mask = 1u << (ei / 2);
   if ((side_init & mask) == 0)
   {
      if (j < p_Nr)
      {
#define ELX(d, j, k) elx[d][j + k * p_Nr] // assumes lexicographic ordering
         for (dlong d = 0; d < 2; ++d)
         {
            // copy nodal coordinates along the constrained edge
            edge.x[d][j] = ELX(d, jidx, kidx);

            // compute derivative in normal direction.
            dfloat sums_k = 0.0;
            for (dlong k = 0; k < p_Nr; ++k)
            {
               if (ei >= 2)
               {
                  sums_k += wt1[k] * ELX(d, j, k);
               }
               else
               {
                  sums_k += wt1[k] * ELX(d, k, j);
               }
            }
            edge.dxdn[d][j] = sums_k;
         }
#undef ELX
      }
      side_init = mask;
   }
   return edge;
}

//pi=0, r=-1,s=-1
//pi=1, r=+1,s=-1
//pi=2, r=-1,s=+1
//pi=3, r=+1,s=+1
static MFEM_HOST_DEVICE inline findptsElementGPT_t get_pt(const dfloat *elx[2],
                                                          const dfloat *wtend[2],
                                                          dlong pi, dlong p_Nr)
{
   findptsElementGPT_t pt;

#define ELX(d, j, k) elx[d][j + k * p_Nr]

   dlong r_g_wt_offset = pi % 2 == 0 ? 0 : 1; //wtend offset for gradient
   dlong s_g_wt_offset = pi < 2 ? 0 : 1;
   dlong jidx = pi % 2 == 0 ? 0 : p_Nr-1;
   dlong kidx = pi < 2 ? 0 : p_Nr-1;

   pt.x[0] = ELX(0, jidx, kidx);
   pt.x[1] = ELX(1, jidx, kidx);

   pt.jac[0] = 0.0;
   pt.jac[1] = 0.0;
   pt.jac[2] = 0.0;
   pt.jac[3] = 0.0;
   for (dlong j = 0; j < p_Nr; ++j)
   {
      //dx/dr
      pt.jac[0] += wtend[0][3 * r_g_wt_offset * p_Nr + p_Nr + j] * ELX(0, j, kidx);

      // dy/dr
      pt.jac[1] += wtend[0][3 * r_g_wt_offset * p_Nr + p_Nr + j] * ELX(1, j, kidx);

      // dx/ds
      pt.jac[2] += wtend[0][3 * s_g_wt_offset * p_Nr + p_Nr + j] * ELX(0, kidx, j);

      // dy/ds
      pt.jac[3] += wtend[0][3 * s_g_wt_offset * p_Nr + p_Nr + j] * ELX(1, kidx, j);
   }

   pt.hes[0] = 0.0;
   pt.hes[1] = 0.0;
   pt.hes[2] = 0.0;
   pt.hes[3] = 0.0;
   for (dlong j = 0; j < p_Nr; ++j)
   {
      //d2x/dr2
      pt.hes[0] += wtend[0][3 * r_g_wt_offset * p_Nr + 2*p_Nr + j] * ELX(0, j, kidx);

      // d2y/dr2
      pt.hes[2] += wtend[0][3 * r_g_wt_offset * p_Nr + 2*p_Nr + j] * ELX(1, j, kidx);

      // d2x/ds2
      pt.hes[1] += wtend[0][3 * s_g_wt_offset * p_Nr + 2*p_Nr + j] * ELX(0, kidx, j);

      // d2y/ds2
      pt.hes[3] += wtend[0][3 * s_g_wt_offset * p_Nr + 2*p_Nr + j] * ELX(1, kidx, j);
   }
#undef ELX
   return pt;
}

/* check reduction in objective against prediction, and adjust
   trust region radius (p->tr) accordingly;
   may reject the prior step, returning 1; otherwise returns 0
   sets out->dist2, out->index, out->x, out->oldr in any event,
   leaving out->r, out->dr, out->flags to be set when returning 0 */
static MFEM_HOST_DEVICE bool reject_prior_step_q(findptsElementPoint_t *out,
                                                 const dfloat resid[2],
                                                 const findptsElementPoint_t *p,
                                                 const dfloat tol)
{
   const dfloat dist2 = norm2(resid);
   const dfloat decr = p->dist2 - dist2;
   const dfloat pred = p->dist2p;
   for (dlong d = 0; d < 2; ++d)
   {
      out->x[d] = p->x[d];
      out->oldr[d] = p->r[d];
   }
   out->dist2 = dist2;
   if (decr >= 0.01 * pred)
   {
      if (decr >= 0.9 * pred)
      {
         // very good iteration
         out->tr = p->tr * 2;
      }
      else
      {
         // good iteration
         out->tr = p->tr;
      }
      return false;
   }
   else
   {
      /* reject step; note: the point will pass through this routine
         again, and we set things up here so it gets classed as a
         "very good iteration" --- this doubles the trust radius,
         which is why we divide by 4 below */
      dfloat v0 = fabs(p->r[0] - p->oldr[0]);
      dfloat v1 = fabs(p->r[1] - p->oldr[1]);
      out->tr = (v0 > v1 ? v0 : v1)/4;
      out->dist2 = p->dist2;
      for (dlong d = 0; d < 2; ++d)
      {
         out->r[d] = p->oldr[d];
      }
      out->flags = p->flags >> 5;
      out->dist2p = -DBL_MAX;
      if (pred < dist2 * tol)
      {
         out->flags |= CONVERGED_FLAG;
      }
      return true;
   }
}

/* minimize ||resid - jac * dr||_2, with |dr| <= tr, |r0+dr|<=1
   (exact solution of trust region problem) */
static MFEM_HOST_DEVICE void newton_area(findptsElementPoint_t *const out,
                                         const dfloat jac[4],
                                         const dfloat resid[2],
                                         const findptsElementPoint_t *const p,
                                         const dfloat tol)
{
   const dfloat tr = p->tr;
   dfloat bnd[4] = {-1, 1, -1, 1};
   dfloat r0[2];
   dfloat dr[2], fac;
   dlong d, mask, flags;
   r0[0] = p->r[0], r0[1] = p->r[1];

   mask = 0xfu;
   for (d = 0; d < 2; ++d)
   {
      if (r0[d] - tr > -1)
      {
         bnd[2 * d] = r0[d] - tr, mask ^= 1u << (2 * d);
      }
      if (r0[d] + tr < 1)
      {
         bnd[2 * d + 1] = r0[d] + tr, mask ^= 2u << (2 * d);
      }
   }

   lin_solve_2(dr, jac, resid);

   fac = 1, flags = 0;
   for (d = 0; d < 2; ++d)
   {
      dfloat nr = r0[d] + dr[d];
      if ((nr - bnd[2 * d]) * (bnd[2 * d + 1] - nr) >= 0)
      {
         continue;
      }
      if (nr < bnd[2 * d])
      {
         dfloat f = (bnd[2 * d] - r0[d]) / dr[d];
         if (f < fac)
         {
            fac = f, flags = 1u << (2 * d);
         }
      }
      else
      {
         dfloat f = (bnd[2 * d + 1] - r0[d]) / dr[d];
         if (f < fac)
         {
            fac = f, flags = 2u << (2 * d);
         }
      }
   }

   if (flags == 0)
   {
      goto newton_area_fin;
   }

   for (d = 0; d < 2; ++d)
   {
      dr[d] *= fac;
   }

newton_area_edge :
   {
      const dlong ei = edge_index(flags);
      const dlong dn = ei>>1, de = plus_1_mod_2(dn);
      dfloat fac = 1;
      dlong new_flags = 0;
      dfloat res[2], y, JtJ, drc;
      res[0] = resid[0] - (jac[0] * dr[0] + jac[1] * dr[1]),
               res[1] = resid[1] - (jac[2] * dr[0] + jac[3] * dr[1]);
      /* y = J_u^T res */
      y = jac[de] * res[0] + jac[2+de] * res[1];
      /* JtJ = J_u^T J_u */
      JtJ = jac[de] * jac[de] + jac[2+de] * jac[2+de];
      drc = y / JtJ;
      {
         const double rz = r0[de] + dr[de], lb = bnd[2*de], ub = bnd[2*de+1];
         const double nr = r0[de]+(dr[de]+drc);
         if ((nr-lb) * (ub-nr) < 0)
         {
            if (nr < lb)
            {
               double f = (lb-rz)/drc;
               if (f < fac)
               {
                  fac=f;
                  new_flags = 1u<<(2*de);
               }
            }
            else
            {
               double f = (ub-rz)/drc;
               if (f < fac)
               {
                  fac=f;
                  new_flags = 2u<<(2*de);
               }
            }
         }
      }

      dr[de] += fac * drc;
      flags |= new_flags;
      goto newton_area_relax;
   }

   /* check and possibly relax constraints */
newton_area_relax :
   {
      const dlong old_flags = flags;
      dfloat res[2], y[2];
      /* res := res_0 - J dr */
      res[0] = resid[0] - (jac[0] * dr[0] + jac[1] * dr[1]);
      res[1] = resid[1] - (jac[2] * dr[0] + jac[3] * dr[1]);
      /* y := J^T res */
      y[0] = jac[0] * res[0] + jac[2] * res[1];
      y[1] = jac[1] * res[0] + jac[3] * res[1];
      for (dlong d = 0; d < 2; ++d)
      {
         dlong f = flags >> (2 * d) & 3u;
         if (f)
         {
            dr[d] = bnd[2 * d + (f - 1)] - r0[d];
            if (dr[d] * y[d] < 0)
            {
               flags &= ~(3u << (2 * d));
            }
         }
      }
      if (flags == old_flags)
      {
         goto newton_area_fin;
      }
      switch (num_constrained(flags))
      {
         case 1:
            goto newton_area_edge;
      }
   }

newton_area_fin:
   flags &= mask;
   if (fabs(dr[0]) + fabs(dr[1]) < tol)
   {
      flags |= CONVERGED_FLAG;
   }
   {
      const dfloat res0 = resid[0] - (jac[0] * dr[0] + jac[1] * dr[1]);
      const dfloat res1 = resid[1] - (jac[2] * dr[0] + jac[3] * dr[1]);
      out->dist2p = resid[0] * resid[0] + resid[1] * resid[1] -
                    (res0 * res0 + res1 * res1);
   }
   for (dlong d = 0; d < 2; ++d)
   {
      dlong f = flags >> (2 * d) & 3u;
      out->r[d] = f == 0 ? r0[d] + dr[d] : (f == 1 ? -1 : 1);
   }
   out->flags = flags | (p->flags << 5);
}

static MFEM_HOST_DEVICE inline void newton_edge(findptsElementPoint_t *const
                                                out,
                                                const dfloat jac[4],
                                                const dfloat rhes,
                                                const dfloat resid[2],
                                                const dlong de,
                                                const dlong dn,
                                                dlong flags,
                                                const findptsElementPoint_t *const p,
                                                const dfloat tol)
{
   const dfloat tr = p->tr;
   /* A = J^T J - resid_d H_d */
   const dfloat A = jac[de] * jac[de] + jac[2 + de] * jac[2 + de] - rhes;
   /* y = J^T r */
   const dfloat y = jac[de] * resid[0] + jac[2 + de] * resid[1];

   const dfloat oldr = p->r[de];
   dfloat dr, nr, tdr, tnr;
   dfloat v, tv;
   dlong new_flags = 0, tnew_flags = 0;

#define EVAL(dr) (dr * A - 2 * y) * dr

   /* if A is not SPD, quadratic model has no minimum */
   if (A > 0)
   {
      dr = y / A, nr = oldr + dr;
      if (fabs(dr) < tr && fabs(nr) < 1)
      {
         v = EVAL(dr);
         goto newton_edge_fin;
      }
   }

   if ((nr = oldr - tr) > -1)
   {
      dr = -tr;
   }
   else
   {
      nr = -1, dr = -1 - oldr, new_flags = flags | 1u << (2 * de);
   }
   v = EVAL(dr);

   if ((tnr = oldr + tr) < 1)
   {
      tdr = tr;
   }
   else
   {
      tnr = 1, tdr = 1 - oldr, tnew_flags = flags | 2u << (2 * de);
   }
   tv = EVAL(tdr);

   if (tv < v)
   {
      nr = tnr, dr = tdr, v = tv, new_flags = tnew_flags;
   }

newton_edge_fin:
   /* check convergence */
   if (fabs(dr) < tol)
   {
      new_flags |= CONVERGED_FLAG;
   }
   out->r[de] = nr;
   out->r[dn]=p->r[dn];
   out->dist2p = -v;
   out->flags = flags | new_flags | (p->flags << 5);
}

static MFEM_HOST_DEVICE void seed_j(const dfloat *elx[2],
                                    const dfloat x[2],
                                    const dfloat *z, //GLL point locations [-1, 1]
                                    dfloat *dist2,
                                    dfloat *r[2],
                                    const int j,
                                    const dlong p_Nr)
{
   if (j >= p_Nr)
   {
      return;
   }

   dist2[j] = DBL_MAX;

   dfloat zr = z[j];
   for (dlong k = 0; k < p_Nr; ++k)
   {
      dfloat zs = z[k];
      const dlong jk = j + k * p_Nr;
      dfloat dx[2];
      for (dlong d = 0; d < 2; ++d)
      {
         dx[d] = x[d] - elx[d][jk];
      }
      const dfloat dist2_jkl = norm2(dx);
      if (dist2[j] > dist2_jkl)
      {
         dist2[j] = dist2_jkl;
         r[0][j] = zr;
         r[1][j] = zs;
      }
   }
}

static MFEM_HOST_DEVICE dfloat tensor_ig2_j(dfloat *g_partials,
                                            const dfloat *Jr,
                                            const dfloat *Dr,
                                            const dfloat *Js,
                                            const dfloat *Ds,
                                            const dfloat *u,
                                            const dlong j,
                                            const dlong p_Nr)
{
   dfloat uJs = 0.0;
   dfloat uDs = 0.0;
   for (dlong k = 0; k < p_Nr; ++k)
   {
      uJs += u[j + k * p_Nr] * Js[k];
      uDs += u[j + k * p_Nr] * Ds[k];
   }

   g_partials[0] = uJs * Dr[j];
   g_partials[1] = uDs * Jr[j];
   return uJs * Jr[j];
}

static void FindPointsLocal2D_Kernel(const int npt,
                                     const dfloat tol,
                                     const dfloat *x,
                                     const dlong point_pos_ordering,
                                     const dfloat *xElemCoord,
                                     const dfloat *yElemCoord,
                                     const dfloat *wtend_x,
                                     const dfloat *wtend_y,
                                     const dfloat *c,
                                     const dfloat *A,
                                     const dfloat *minBound,
                                     const dfloat *maxBound,
                                     const dlong hash_n,
                                     const dfloat *hashMin,
                                     const dfloat *hashFac,
                                     dlong *hashOffset,
                                     dlong *const code_base,
                                     dlong *const el_base,
                                     dfloat *const r_base,
                                     dfloat *const dist2_base,
                                     const dfloat *gll1D,
                                     const double *lagcoeff,
                                     dfloat *infok,
                                     const dlong p_Nr)
{
#define MAX_CONST(a, b) (((a) > (b)) ? (a) : (b))
#define p_innerSize 32
   const int dim = 2;
   const int dim2 = dim*dim;
   const dlong p_NE = p_Nr*p_Nr;
   const int p_Nr_Max = 8;
   //    mfem::forall_1D(npt, p_innerSize, [=] MFEM_HOST_DEVICE (int i)
   //   mfem::forall_2D(npt, p_innerSize, 1, [=] MFEM_HOST_DEVICE (int i)
   mfem::forall(npt, [=] MFEM_HOST_DEVICE (int i)
   {
      constexpr int size1 = MAX_CONST(4,
                                      p_Nr_Max + 1) * (3 * 3 + 2 * 3) + 3 * 2 * p_Nr_Max + 5;
      constexpr int size2 = MAX_CONST(p_Nr_Max *p_Nr_Max * 6, p_Nr_Max * 3 * 3);
      MFEM_SHARED dfloat r_workspace[size1];
      MFEM_SHARED findptsElementPoint_t el_pts[2];

      MFEM_SHARED dfloat constraint_workspace[size2];
      // dlong constraint_init;
      MFEM_SHARED dlong constraint_init_t[p_innerSize];

      dfloat *r_workspace_ptr;
      findptsElementPoint_t *fpt, *tmp;
      MFEM_FOREACH_THREAD(j,x,p_innerSize)
      {
         r_workspace_ptr = r_workspace;
         fpt = el_pts + 0;
         tmp = el_pts + 1;
      }
      MFEM_SYNC_THREAD;

      dlong id_x = point_pos_ordering == 0 ? i : i*dim;
      dlong id_y = point_pos_ordering == 0 ? i+npt : i*dim+1;
      dfloat x_i[2] = {x[id_x], x[id_y]};
      const dfloat *wtend[2] = {&wtend_x[0], &wtend_y[0]};

      dlong *code_i = code_base + i;
      dlong *el_i = el_base + i;
      dfloat *r_i = r_base + dim * i;
      dfloat *dist2_i = dist2_base + i;

      //// map_points_to_els ////
      findptsLocalHashData_t hash;
      for (int d = 0; d < dim; ++d)
      {
         hash.bnd[d].min = hashMin[d];
         hash.fac[d] = hashFac[d];
      }
      hash.hash_n = hash_n;
      hash.offset = hashOffset;
      const dlong hi = hash_index(&hash, x_i);
      const dlong *elp = hash.offset + hash.offset[hi],
                   *const ele = hash.offset + hash.offset[hi + 1];
      *code_i = CODE_NOT_FOUND;
      *dist2_i = DBL_MAX;

      for (; elp != ele; ++elp)
      {
         //elp

         const dlong el = *elp;

         // construct obbox_t on the fly from data
         obbox_t box;

         for (int idx = 0; idx < dim; ++idx)
         {
            box.c0[idx] = c[dim * el + idx];
            box.x[idx].min = minBound[dim * el + idx];
            box.x[idx].max = maxBound[dim * el + idx];
         }

         for (int idx = 0; idx < dim*dim; ++idx)
         {
            box.A[idx] = A[dim*dim * el + idx];
         }

         if (obbox_test(&box, x_i) >= 0)
         {
            //// findpts_local ////
            {
               const dfloat *elx[dim];

               elx[0] = xElemCoord + el * p_NE;
               elx[1] = yElemCoord + el * p_NE;

               //// findpts_el ////
               {
                  MFEM_SYNC_THREAD;
                  //                  MFEM_FOREACH_THREAD(j,x,p_innerSize)
                  for (dlong j = 0; j < p_innerSize; ++j) //inner
                  {
                     if (j == 0)
                     {
                        fpt->dist2 = DBL_MAX;
                        fpt->dist2p = 0;
                        fpt->tr = 1;
                     }
                     if (j < dim) { fpt->x[j] = x_i[j]; }
                     constraint_init_t[j] = 0;
                  }
                  MFEM_SYNC_THREAD;
                  //// seed ////
                  {
                     dfloat *dist2_temp = r_workspace_ptr;
                     dfloat *r_temp[dim];
                     for (dlong d = 0; d < dim; ++d)
                     {
                        r_temp[d] = dist2_temp + (1 + d) * p_Nr;
                     }

                     //                     MFEM_FOREACH_THREAD(j,x,p_innerSize)
                     for (dlong j = 0; j < p_innerSize; ++j)  //inner
                     {
                        seed_j(elx, x_i, gll1D, dist2_temp, r_temp, j, p_Nr);
                     }
                     MFEM_SYNC_THREAD;

                     //                     MFEM_FOREACH_THREAD(j,x,p_innerSize)
                     for (dlong j = 0; j < p_innerSize; ++j) //inner
                     {
                        if (j == 0)
                        {
                           fpt->dist2 = DBL_MAX;
                           for (dlong jj = 0; jj < p_Nr; ++jj)
                           {
                              if (dist2_temp[jj] < fpt->dist2)
                              {
                                 fpt->dist2 = dist2_temp[jj];
                                 for (dlong d = 0; d < dim; ++d)
                                 {
                                    fpt->r[d] = r_temp[d][jj];
                                 }
                              }
                           }
                        }
                     }
                     MFEM_SYNC_THREAD;
                  } //seed done


                  //                  MFEM_FOREACH_THREAD(j,x,p_innerSize)
                  for (dlong j = 0; j < p_innerSize; ++j)  //inner
                  {
                     if (j == 0)
                     {
                        tmp->dist2 = DBL_MAX;
                        tmp->dist2p = 0;
                        tmp->tr = 1;
                        tmp->flags = 0;
                     }
                     if (j < dim)
                     {
                        tmp->x[j] = fpt->x[j];
                        tmp->r[j] = fpt->r[j];
                     }
                  }
                  MFEM_SYNC_THREAD;

                  for (dlong step = 0; step < 50; step++)
                  {
                     switch (num_constrained(tmp->flags & FLAG_MASK))
                     {
                        case 0:   // findpt_area
                        {
                           dfloat *wtr = r_workspace_ptr;
                           dfloat *wts = wtr + 2 * p_Nr;

                           dfloat *resid = wts + 2 * p_Nr;
                           dfloat *jac = resid + 2;
                           dfloat *resid_temp = jac + 4;
                           dfloat *jac_temp = resid_temp + 2 * p_Nr;

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           //                           for (dlong j = 0; j < p_innerSize; ++j)
                           {
                              if (j < p_Nr)
                              {
                                 lagrange_eval_first_derivative(wtr, tmp->r[0], j, gll1D, lagcoeff, p_Nr);
                                 lagrange_eval_first_derivative(wts, tmp->r[1], j, gll1D, lagcoeff, p_Nr);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           //                           for (dlong j = 0; j < p_innerSize; ++j)
                           {
                              if (j < p_Nr * 2)
                              {
                                 const int qp = j / 2;
                                 const int d = j % 2;
                                 resid_temp[d + qp * 2] = tensor_ig2_j(jac_temp + 2 * d + 4 * qp,
                                                                       wtr,
                                                                       wtr + p_Nr,
                                                                       wts,
                                                                       wts + p_Nr,
                                                                       elx[d],
                                                                       qp,
                                                                       p_Nr);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(l,x,p_innerSize)
                           //                           for (dlong l = 0; l < p_innerSize; ++l) //inner
                           {
                              if (l < 2)
                              {
                                 resid[l] = tmp->x[l];
                                 for (dlong j = 0; j < p_Nr; ++j)
                                 {
                                    resid[l] -= resid_temp[l + j * 2];
                                 }
                              }
                              if (l < 4)
                              {
                                 jac[l] = 0;
                                 for (dlong j = 0; j < p_Nr; ++j)
                                 {
                                    jac[l] += jac_temp[l + j * 4];
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(l,x,p_innerSize)
                           //                           for (dlong l = 0; l < p_innerSize; ++l)
                           {
                              if (l == 0)
                              {
                                 if (!reject_prior_step_q(fpt, resid, tmp, tol))
                                 {
                                    newton_area(fpt, jac, resid, tmp, tol);
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        } //case 0
                        case 1:   // findpt_edge
                        {
                           const dlong ei = edge_index(tmp->flags & FLAG_MASK);
                           const dlong dn = ei>>1, de = plus_1_mod_2(dn);
                           dlong d_j[2];
                           d_j[0] = de;
                           d_j[1] = dn;

                           dfloat *wt = r_workspace_ptr;
                           dfloat *resid = wt + 3 * p_Nr;
                           dfloat *jac = resid + 2; //jac will be row-major
                           dfloat *hes_T = jac + 2 * 2;
                           dfloat *hess = hes_T + 3;
                           findptsElementGEdge_t edge;

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           {
                              edge = get_edge(elx, wtend, ei, constraint_workspace, constraint_init_t[j], j,
                                              p_Nr);
                           }
                           MFEM_SYNC_THREAD;

                           const dfloat *const *e_x[2] = {edge.x, edge.dxdn};

                           // compute basis function info upto 2nd derivative for tangential components
                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           {
                              if (j < p_Nr)
                              {
                                 lagrange_eval_second_derivative(wt, tmp->r[de], j, gll1D, lagcoeff, p_Nr);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           {
                              if (j < dim)
                              {
                                 resid[j] = tmp->x[j];
                                 jac[2*j] = 0.0;
                                 jac[2*j + 1] = 0.0;
                                 hess[j] = 0.0;
                                 for (dlong k = 0; k < p_Nr; ++k)
                                 {
                                    resid[j] -= wt[k]*edge.x[j][k];
                                    jac[2*j] += wt[k]*edge.dxdn[j][k];
                                    jac[2*j+1] += wt[k+p_Nr]*edge.x[j][k];
                                    hess[j] += wt[k+2*p_Nr]*edge.x[j][k];
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;

                           // at this point, the Jacobian will be out of
                           // order for edge index 2 and 3 so we need to swap
                           // columns
                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           {
                              if (j == 0)
                              {
                                 if (ei >= 2)
                                 {
                                    dfloat temp1 = jac[1],
                                           temp2 = jac[3];
                                    jac[1] = jac[0];
                                    jac[3] = jac[2];
                                    jac[0] = temp1;
                                    jac[2] = temp2;
                                 }
                                 hess[2] = resid[0]*hess[0] + resid[1]*hess[1];
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(l,x,p_innerSize)
                           {
                              if (l == 0)
                              {
                                 // check prior step //
                                 if (!reject_prior_step_q(fpt, resid, tmp, tol))
                                 {
                                    double steep = resid[0] * jac[  dn]
                                                   + resid[1] * jac[2+dn];

                                    if (steep * tmp->r[dn] < 0) /* relax constraint */
                                    {
                                       newton_area(fpt, jac, resid, tmp, tol);
                                    }
                                    else
                                    {
                                       newton_edge(fpt, jac, hess[2], resid, de, dn, tmp->flags & FLAG_MASK,
                                                   tmp, tol);
                                    }
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        }
                        case 2:   // findpts_pt
                        {
                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           {
                              if (j == 0)
                              {
                                 dlong de = 0;
                                 dlong dn = 0;
                                 const dlong pi = point_index(tmp->flags & FLAG_MASK);
                                 const findptsElementGPT_t gpt = get_pt(elx, wtend, pi, p_Nr);

                                 const dfloat *const pt_x = gpt.x,
                                                     *const jac = gpt.jac,
                                                            *const hes = gpt.hes;

                                 dfloat resid[dim], steep[dim], sr[dim];
                                 for (dlong d = 0; d < dim; ++d)
                                 {
                                    resid[d] = fpt->x[d] - pt_x[d];
                                 }
                                 steep[0] = jac[0]*resid[0] + jac[2]*resid[1],
                                            steep[1] = jac[1]*resid[0] + jac[3]*resid[1];

                                 sr[0] = steep[0]*tmp->r[0];
                                 sr[1] = steep[1]*tmp->r[1];
                                 if (!reject_prior_step_q(fpt, resid, tmp, tol))
                                 {
                                    if (sr[0]<0)
                                    {
                                       if (sr[1]<0)
                                       {
                                          newton_area(fpt, jac, resid, tmp, tol);
                                       }
                                       else
                                       {
                                          de=0;
                                          dn=1;
                                          const double rh = resid[0]*hes[de]+
                                                            resid[1]*hes[2+de];
                                          newton_edge(fpt, jac, rh,
                                                      resid, de,
                                                      dn,
                                                      tmp->flags & FLAG_MASK,
                                                      tmp, tol);
                                       }
                                    }
                                    else if (sr[1]<0)
                                    {
                                       de=1;
                                       dn=0;
                                       const double rh = resid[0]*hes[de]+
                                                         resid[1]*hes[2+de];
                                       newton_edge(fpt, jac, rh,
                                                   resid, de,
                                                   dn,
                                                   tmp->flags & FLAG_MASK,
                                                   tmp, tol);
                                    }
                                    else
                                    {
                                       fpt->r[0] = tmp->r[0];
                                       fpt->r[1] = tmp->r[1];
                                       fpt->dist2p = 0;
                                       fpt->flags = tmp->flags | CONVERGED_FLAG;
                                    }
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        } //case 3
                     } //switch
                     if (fpt->flags & CONVERGED_FLAG)
                     {
                        break;
                     }
                     MFEM_SYNC_THREAD;
                     MFEM_FOREACH_THREAD(j,x,p_innerSize)
                     //                     for (dlong j = 0; j < p_innerSize; ++j)
                     if (j == 0)
                     {
                        *tmp = *fpt;
                     }
                     MFEM_SYNC_THREAD;
                  } //for dlong step < 50
               } //findpts_el

               bool converged_internal = (fpt->flags & FLAG_MASK) == CONVERGED_FLAG;
               if (*code_i == CODE_NOT_FOUND || converged_internal || fpt->dist2 < *dist2_i)
               {
                  //                  MFEM_FOREACH_THREAD(j,x,p_innerSize)
                  for (dlong j = 0; j < p_innerSize; ++j)
                  {
                     if (j == 0)
                     {
                        *el_i = el;
                        *code_i = converged_internal ? CODE_INTERNAL : CODE_BORDER;
                        *dist2_i = fpt->dist2;
                     }
                     if (j < 3)
                     {
                        r_i[j] = fpt->r[j];
                     }
                  }
                  MFEM_SYNC_THREAD;
                  if (converged_internal)
                  {
                     break;
                  }
               }
            } //findpts_local
         } //obbox_test
      } //elp
   });
}

void FindPointsGSLIB::FindPointsLocal2(const Vector &point_pos,
                                       int point_pos_ordering,
                                       Array<int> &gsl_code_dev_l,
                                       Array<int> &gsl_elem_dev_l,
                                       Vector &gsl_ref_l,
                                       Vector &gsl_dist_l,
                                       int npt)
{
   if (npt == 0) { return; }
   MFEM_VERIFY(dim == 2,"Function for 2D only");
   FindPointsLocal2D_Kernel(npt, DEV.tol,
                            point_pos.Read(), point_pos_ordering,
                            DEV.o_x.Read(),
                            DEV.o_y.Read(),
                            DEV.o_wtend_x.Read(),
                            DEV.o_wtend_y.Read(),
                            DEV.o_c.Read(),
                            DEV.o_A.Read(),
                            DEV.o_min.Read(),
                            DEV.o_max.Read(),
                            DEV.hash_n,
                            DEV.o_hashMin.Read(),
                            DEV.o_hashFac.Read(),
                            DEV.o_offset.ReadWrite(),
                            gsl_code_dev_l.Write(),
                            gsl_elem_dev_l.Write(),
                            gsl_ref_l.Write(),
                            gsl_dist_l.Write(),
                            DEV.gll1d.ReadWrite(),
                            DEV.lagcoeff.Read(),
                            DEV.info.ReadWrite(),
                            DEV.dof1d);
}


#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND
#undef dlong
#undef dfloat

} // namespace mfem

#endif //ifdef MFEM_USE_GSLIB
