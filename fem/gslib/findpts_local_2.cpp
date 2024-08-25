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
#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2

static MFEM_HOST_DEVICE inline void lagrange_eval_first_derivative(double *p0,
                                                                   double x, int i,
                                                                   const double *z, const double *lagrangeCoeff, int pN)
{
   double u0 = 1, u1 = 0;
   for (int j = 0; j < pN; ++j)
   {
      if (i != j)
      {
         double d_j = 2 * (x - z[j]);
         u1 = d_j * u1 + u0;
         u0 = d_j * u0;
      }
   }
   double *p1 = p0 + pN;
   p0[i] = lagrangeCoeff[i] * u0;
   p1[i] = 2.0 * lagrangeCoeff[i] * u1;
}

static MFEM_HOST_DEVICE inline void lagrange_eval_second_derivative(double *p0,
                                                                    double x, int i,
                                                                    const double *z, const double *lagrangeCoeff, int pN)
{
   double u0 = 1, u1 = 0, u2 = 0;
   for (int j = 0; j < pN; ++j)
   {
      if (i != j)
      {
         double d_j = 2 * (x - z[j]);
         u2 = d_j * u2 + u1;
         u1 = d_j * u1 + u0;
         u0 = d_j * u0;
      }
   }
   double *p1 = p0 + pN, *p2 = p0 + 2 * pN;
   p0[i] = lagrangeCoeff[i] * u0;
   p1[i] = 2.0 * lagrangeCoeff[i] * u1;
   p2[i] = 8.0 * lagrangeCoeff[i] * u2;
}

/* positive when possibly inside */
static MFEM_HOST_DEVICE inline double obbox_axis_test(const obbox_t *const b,
                                                      const double x[2])
{
   double test = 1;
   for (int d = 0; d < 2; ++d)
   {
      double b_d = (x[d] - b->x[d].min) * (b->x[d].max - x[d]);
      test = test < 0 ? test : b_d;
   }
   return test;
}

/* positive when possibly inside */
static MFEM_HOST_DEVICE inline double obbox_test(const obbox_t *const b,
                                                 const double x[2])
{
   const double bxyz = obbox_axis_test(b, x);
   if (bxyz < 0)
   {
      return bxyz;
   }
   else
   {
      double dxyz[2];
      for (int d = 0; d < 2; ++d)
      {
         dxyz[d] = x[d] - b->c0[d];
      }
      double test = 1;
      for (int d = 0; d < 2; ++d)
      {
         double rst = 0;
         for (int e = 0; e < 2; ++e)
         {
            rst += b->A[d * 2 + e] * dxyz[e];
         }
         double brst = (rst + 1) * (1 - rst);
         test = test < 0 ? test : brst;
      }
      return test;
   }
}

////// HASH //////
static MFEM_HOST_DEVICE inline int hash_index(const findptsLocalHashData_t *p,
                                              const double x[2])
{
   const int n = p->hash_n;
   int sum = 0;
   for (int d = 2 - 1; d >= 0; --d)
   {
      sum *= n;
      int i = (int)floor((x[d] - p->bnd[d].min) * p->fac[d]);
      sum += i < 0 ? 0 : (n - 1 < i ? n - 1 : i);
   }
   return sum;
}

/* A is row-major */
static MFEM_HOST_DEVICE inline void lin_solve_2(double x[2], const double A[4],
                                                const double y[2])
{
   const double idet = 1/(A[0]*A[3] - A[1]*A[2]);
   x[0] = idet*(A[3]*y[0] - A[1]*y[1]);
   x[1] = idet*(A[0]*y[1] - A[2]*y[0]);
}

static MFEM_HOST_DEVICE inline double norm2(const double x[2]) { return x[0] * x[0] + x[1] * x[1]; }

/* the bit structure of flags is CSSRR
   the C bit --- 1<<4 --- is set when the point is converged
   RR is 0 = 00b if r is unconstrained,
         1 = 01b if r is constrained at -1
         2 = 10b if r is constrained at +1
   SS is similarly for s constraints
   SSRR = smax,smin,rmax,rmin
*/

#define CONVERGED_FLAG (1u<<4)
#define FLAG_MASK 0x1fu

static MFEM_HOST_DEVICE inline int num_constrained(const int flags)
{
   const int y = flags | flags >> 1;
   return (y&1u) + (y>>2 & 1u);
}

/* assumes x = 0, 1, or 2 */
static MFEM_HOST_DEVICE inline int plus_1_mod_2(const int x)
{
   return x ^ 1u;
}

/* assumes x = 1 << i, with i < 4, returns i+1 */
static MFEM_HOST_DEVICE inline int which_bit(const int x)
{
   const int y = x & 7u;
   return (y-(y>>2)) | ((x-1)&4u);
}

static MFEM_HOST_DEVICE inline int edge_index(const int x)
{
   return which_bit(x)-1;
}

static MFEM_HOST_DEVICE inline int point_index(const int x)
{
   return ((x>>1)&1u) | ((x>>2)&2u);
}

// compute (x,y) and (dxdn, dydn) data for all DOFs along the edge based on
// edge index. ei=0..3 corresponding to rmin, rmax, smin, smax.
static MFEM_HOST_DEVICE inline findptsElementGEdge_t
get_edge(const double *elx[2], const double *wtend, int ei,
         double *workspace,
         int &side_init, int j, int pN)
{
   findptsElementGEdge_t edge;
   const int jidx = ei >= 2 ? j : ei*(pN-1);
   const int kidx = ei >= 2 ? (ei-2)*(pN-1) : j;

   // location of derivatives based on whether we want at r/s=-1 or r/s=+1
   // ei == 0 and 2 are constrained at -1, 1 and 3 are constrained at +1.
   const double *wt1 = wtend + (ei%2==0 ? 0 : 1)* pN * 3 + pN;

   for (int d = 0; d < 2; ++d)
   {
      edge.x[d] = workspace + d * pN; //x & y coordinates of DOFS along edge
      edge.dxdn[d] = workspace + (2 + d) * pN; //dxdn and dydn at DOFs along edge
   }

   const int mask = 1u << ei;
   if ((side_init & mask) == 0)
   {
      if (j < pN)
      {
#define ELX(d, j, k) elx[d][j + k * pN] // assumes lexicographic ordering
         for (int d = 0; d < 2; ++d)
         {
            // copy nodal coordinates along the constrained edge
            edge.x[d][j] = ELX(d, jidx, kidx);

            // compute derivative in normal direction.
            double sums_k = 0.0;
            for (int k = 0; k < pN; ++k)
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
static MFEM_HOST_DEVICE inline findptsElementGPT_t get_pt(const double *elx[2],
                                                          const double *wtend,
                                                          int pi, int pN)
{
   findptsElementGPT_t pt;

#define ELX(d, j, k) elx[d][j + k * pN]

   int r_g_wt_offset = pi % 2 == 0 ? 0 : 1; //wtend offset for gradient
   int s_g_wt_offset = pi < 2 ? 0 : 1;
   int jidx = pi % 2 == 0 ? 0 : pN-1;
   int kidx = pi < 2 ? 0 : pN-1;

   pt.x[0] = ELX(0, jidx, kidx);
   pt.x[1] = ELX(1, jidx, kidx);

   pt.jac[0] = 0.0;
   pt.jac[1] = 0.0;
   pt.jac[2] = 0.0;
   pt.jac[3] = 0.0;
   for (int j = 0; j < pN; ++j)
   {
      //dx/dr
      pt.jac[0] += wtend[3 * r_g_wt_offset * pN + pN + j] * ELX(0, j, kidx);

      // dy/dr
      pt.jac[2] += wtend[3 * r_g_wt_offset * pN + pN + j] * ELX(1, j, kidx);

      // dx/ds
      pt.jac[1] += wtend[3 * s_g_wt_offset * pN + pN + j] * ELX(0, kidx, j);

      // dy/ds
      pt.jac[3] += wtend[3 * s_g_wt_offset * pN + pN + j] * ELX(1, kidx, j);
   }

   pt.hes[0] = 0.0;
   pt.hes[1] = 0.0;
   pt.hes[2] = 0.0;
   pt.hes[3] = 0.0;
   for (int j = 0; j < pN; ++j)
   {
      //d2x/dr2
      pt.hes[0] += wtend[3 * r_g_wt_offset * pN + 2*pN + j] * ELX(0, j, kidx);

      // d2y/dr2
      pt.hes[2] += wtend[3 * r_g_wt_offset * pN + 2*pN + j] * ELX(1, j, kidx);

      // d2x/ds2
      pt.hes[1] += wtend[3 * s_g_wt_offset * pN + 2*pN + j] * ELX(0, kidx, j);

      // d2y/ds2
      pt.hes[3] += wtend[3 * s_g_wt_offset * pN + 2*pN + j] * ELX(1, kidx, j);
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
                                                 const double resid[2],
                                                 const findptsElementPoint_t *p,
                                                 const double tol)
{
   const double dist2 = norm2(resid);
   const double decr = p->dist2 - dist2;
   const double pred = p->dist2p;
   for (int d = 0; d < 2; ++d)
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
      double v0 = fabs(p->r[0] - p->oldr[0]);
      double v1 = fabs(p->r[1] - p->oldr[1]);
      out->tr = (v0 > v1 ? v0 : v1)/4;
      out->dist2 = p->dist2;
      for (int d = 0; d < 2; ++d)
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
                                         const double jac[4],
                                         const double resid[2],
                                         const findptsElementPoint_t *const p,
                                         const double tol)
{
   const double tr = p->tr;
   double bnd[4] = {-1, 1, -1, 1};
   double r0[2];
   double dr[2], fac;
   int d, mask, flags;
   r0[0] = p->r[0], r0[1] = p->r[1];

   mask = 0xfu; // 1111 - MSB to LSB - smax,smin,rmax,rmin
   // bnd is initialized to be a box [-1, 1]^2, but if the limits change based
   // on the initial guess (r0) and trust region, the mask is modified.
   // Example: r0 = [0.2, -0.3].. r0[0]-tr = 0.2-1 = -0.8.
   // In this case the bounding box will be shortened and the bit corresponding
   // to rmin will be changed.
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

   // dr = Jac^-1*resid where resid = x^* - x(r)
   lin_solve_2(dr, jac, resid);

   fac = 1, flags = 0;
   for (d = 0; d < 2; ++d)
   {
      double nr = r0[d] + dr[d];
      if ((nr - bnd[2 * d]) * (bnd[2 * d + 1] - nr) >= 0)
      {
         continue;
      }
      if (nr < bnd[2 * d])
      {
         double f = (bnd[2 * d] - r0[d]) / dr[d];
         if (f < fac)
         {
            fac = f, flags = 1u << (2 * d);
         }
      }
      else
      {
         double f = (bnd[2 * d + 1] - r0[d]) / dr[d];
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
      const int ei = edge_index(flags);
      const int dn = ei>>1, de = plus_1_mod_2(dn);
      double fac = 1;
      int new_flags = 0;
      double res[2], y, JtJ, drc;
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
      const int old_flags = flags;
      double res[2], y[2];
      /* res := res_0 - J dr */
      res[0] = resid[0] - (jac[0] * dr[0] + jac[1] * dr[1]);
      res[1] = resid[1] - (jac[2] * dr[0] + jac[3] * dr[1]);
      /* y := J^T res */
      y[0] = jac[0] * res[0] + jac[2] * res[1];
      y[1] = jac[1] * res[0] + jac[3] * res[1];
      for (int d = 0; d < 2; ++d)
      {
         int f = flags >> (2 * d) & 3u;
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
      const double res0 = resid[0] - (jac[0] * dr[0] + jac[1] * dr[1]);
      const double res1 = resid[1] - (jac[2] * dr[0] + jac[3] * dr[1]);
      out->dist2p = resid[0] * resid[0] + resid[1] * resid[1] -
                    (res0 * res0 + res1 * res1);
   }
   for (int d = 0; d < 2; ++d)
   {
      int f = flags >> (2 * d) & 3u;
      out->r[d] = f == 0 ? r0[d] + dr[d] : (f == 1 ? -1 : 1);
   }
   out->flags = flags | (p->flags << 5);
}

static MFEM_HOST_DEVICE inline void newton_edge(findptsElementPoint_t *const
                                                out,
                                                const double jac[4],
                                                const double rhes,
                                                const double resid[2],
                                                const int de,
                                                const int dn,
                                                int flags,
                                                const findptsElementPoint_t *const p,
                                                const double tol)
{
   const double tr = p->tr;
   /* A = J^T J - resid_d H_d */
   const double A = jac[de] * jac[de] + jac[2 + de] * jac[2 + de] - rhes;
   /* y = J^T r */
   const double y = jac[de] * resid[0] + jac[2 + de] * resid[1];

   const double oldr = p->r[de];
   double dr, nr, tdr, tnr;
   double v, tv;
   int new_flags = 0, tnew_flags = 0;

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

static MFEM_HOST_DEVICE void seed_j(const double *elx[2],
                                    const double x[2],
                                    const double *z, //GLL point locations [-1, 1]
                                    double *dist2,
                                    double *r[2],
                                    const int j,
                                    const int pN)
{
   if (j >= pN)
   {
      return;
   }

   dist2[j] = DBL_MAX;

   double zr = z[j];
   for (int k = 0; k < pN; ++k)
   {
      double zs = z[k];
      const int jk = j + k * pN;
      double dx[2];
      for (int d = 0; d < 2; ++d)
      {
         dx[d] = x[d] - elx[d][jk];
      }
      const double dist2_jkl = norm2(dx);
      if (dist2[j] > dist2_jkl)
      {
         dist2[j] = dist2_jkl;
         r[0][j] = zr;
         r[1][j] = zs;
      }
   }
}

static MFEM_HOST_DEVICE double tensor_ig2_j(double *g_partials,
                                            const double *Jr,
                                            const double *Dr,
                                            const double *Js,
                                            const double *Ds,
                                            const double *u,
                                            const int j,
                                            const int pN)
{
   double uJs = 0.0;
   double uDs = 0.0;
   for (int k = 0; k < pN; ++k)
   {
      uJs += u[j + k * pN] * Js[k];
      uDs += u[j + k * pN] * Ds[k];
   }

   g_partials[0] = uJs * Dr[j];
   g_partials[1] = uDs * Jr[j];
   return uJs * Jr[j];
}

template<int T_D1D = 0>
static void FindPointsLocal2D_Kernel(const int npt,
                                     const double tol,
                                     const double *x,
                                     const int point_pos_ordering,
                                     const double *xElemCoord,
                                     const int nel,
                                     const double *wtend,
                                     const double *c,
                                     const double *A,
                                     const double *minBound,
                                     const double *maxBound,
                                     const int hash_n,
                                     const double *hashMin,
                                     const double *hashFac,
                                     unsigned int *hashOffset,
                                     unsigned int *const code_base,
                                     unsigned int *const el_base,
                                     double *const r_base,
                                     double *const dist2_base,
                                     const double *gll1D,
                                     const double *lagcoeff,
                                     int *newton,
                                     double *infok,
                                     const int pN = 0)
{
#define MAX_CONST(a, b) (((a) > (b)) ? (a) : (b))
   const int dim = 2;
   const int dim2 = dim*dim;
   const int MD1 = T_D1D ? T_D1D : 14;
   const int D1D = T_D1D ? T_D1D : pN;
   const int p_NE = D1D*D1D;
   const int p_NEL = nel*p_NE;
   MFEM_VERIFY(MD1 <= 14,"Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D != 0, "Polynomial order not specified.");
   const int nThreads = MAX_CONST(2*MD1, 4);

   mfem::forall_2D(npt, nThreads, 1, [=] MFEM_HOST_DEVICE (int i)
   {
      constexpr int size1 = MAX_CONST(4, MD1 + 1) *
                            (3 * 3 + 2 * 3) + 3 * 2 * MD1 + 5;
      constexpr int size2 = MAX_CONST(MD1 *MD1 * 6, MD1 * 3 * 3);
      MFEM_SHARED double r_workspace[size1];
      MFEM_SHARED findptsElementPoint_t el_pts[2];

      MFEM_SHARED double constraint_workspace[size2];
      MFEM_SHARED int constraint_init_t[nThreads];

      double *r_workspace_ptr;
      findptsElementPoint_t *fpt, *tmp;
      MFEM_FOREACH_THREAD(j,x,nThreads)
      {
         r_workspace_ptr = r_workspace;
         fpt = el_pts + 0;
         tmp = el_pts + 1;
      }
      MFEM_SYNC_THREAD;

      int id_x = point_pos_ordering == 0 ? i : i*dim;
      int id_y = point_pos_ordering == 0 ? i+npt : i*dim+1;
      double x_i[2] = {x[id_x], x[id_y]};

      unsigned int *code_i = code_base + i;
      unsigned int *el_i = el_base + i;
      double *r_i = r_base + dim * i;
      double *dist2_i = dist2_base + i;
      int *newton_i = newton + i;

      //// map_points_to_els ////
      findptsLocalHashData_t hash;
      for (int d = 0; d < dim; ++d)
      {
         hash.bnd[d].min = hashMin[d];
         hash.fac[d] = hashFac[d];
      }
      hash.hash_n = hash_n;
      hash.offset = hashOffset;
      const int hi = hash_index(&hash, x_i);
      const unsigned int        *elp = hash.offset + hash.offset[hi];
      const unsigned int *const  ele = hash.offset + hash.offset[hi + 1];
      *code_i = CODE_NOT_FOUND;
      *dist2_i = DBL_MAX;

      for (; elp != ele; ++elp)
      {
         //elp

         const unsigned int el = *elp;

         // construct obbox_t on the fly from data
         obbox_t box;

         for (int idx = 0; idx < dim; ++idx)
         {
            box.c0[idx] = c[dim * el + idx];
            box.x[idx].min = minBound[dim * el + idx];
            box.x[idx].max = maxBound[dim * el + idx];
         }

         for (int idx = 0; idx < dim2; ++idx)
         {
            box.A[idx] = A[dim2 * el + idx];
         }

         if (obbox_test(&box, x_i) >= 0)
         {
            //// findpts_local ////
            {
               const double *elx[dim];

               for (int d = 0; d < dim; d++)
               {
                  elx[d] = xElemCoord + d*p_NEL + el * p_NE;
               }

               //// findpts_el ////
               {
                  MFEM_SYNC_THREAD;
                  MFEM_FOREACH_THREAD(j,x,nThreads)
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
                     double *dist2_temp = r_workspace_ptr;
                     double *r_temp[dim];
                     for (int d = 0; d < dim; ++d)
                     {
                        r_temp[d] = dist2_temp + (1 + d) * D1D;
                     }

                     MFEM_FOREACH_THREAD(j,x,nThreads)
                     {
                        seed_j(elx, x_i, gll1D, dist2_temp, r_temp, j, D1D);
                     }
                     MFEM_SYNC_THREAD;

                     MFEM_FOREACH_THREAD(j,x,nThreads)
                     {
                        if (j == 0)
                        {
                           fpt->dist2 = DBL_MAX;
                           for (int jj = 0; jj < D1D; ++jj)
                           {
                              if (dist2_temp[jj] < fpt->dist2)
                              {
                                 fpt->dist2 = dist2_temp[jj];
                                 for (int d = 0; d < dim; ++d)
                                 {
                                    fpt->r[d] = r_temp[d][jj];
                                 }
                              }
                           }
                        }
                     }
                     MFEM_SYNC_THREAD;
                  } //seed done

                  MFEM_FOREACH_THREAD(j,x,nThreads)
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

                  for (int step = 0; step < 50; step++)
                  {
                     switch (num_constrained(tmp->flags & FLAG_MASK))
                     {
                        case 0:   // findpt_area
                        {
                           double *wtr = r_workspace_ptr;
                           double *wts = wtr + 2 * D1D;

                           double *resid = wts + 2 * D1D;
                           double *jac = resid + 2;
                           double *resid_temp = jac + 4;
                           double *jac_temp = resid_temp + 2 * D1D;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j < D1D * 2)
                              {
                                 const int qp = j % D1D;
                                 const int d = j / D1D;
                                 lagrange_eval_first_derivative(wtr + 2*d*D1D,
                                                                tmp->r[d], qp,
                                                                gll1D, lagcoeff,
                                                                D1D);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j < D1D * 2)
                              {
                                 const int qp = j % D1D;
                                 const int d = j / D1D;
                                 resid_temp[d + qp * 2] = tensor_ig2_j(jac_temp + 2 * d + 4 * qp,
                                                                       wtr,
                                                                       wtr + D1D,
                                                                       wts,
                                                                       wts + D1D,
                                                                       elx[d],
                                                                       qp,
                                                                       D1D);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(l,x,nThreads)
                           {
                              if (l < 2)
                              {
                                 resid[l] = tmp->x[l];
                                 for (int j = 0; j < D1D; ++j)
                                 {
                                    resid[l] -= resid_temp[l + j * 2];
                                 }
                              }
                              if (l < 4)
                              {
                                 jac[l] = 0;
                                 for (int j = 0; j < D1D; ++j)
                                 {
                                    jac[l] += jac_temp[l + j * 4];
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(l,x,nThreads)
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
                           const int ei = edge_index(tmp->flags & FLAG_MASK);
                           const int dn = ei>>1, de = plus_1_mod_2(dn);

                           double *wt = r_workspace_ptr;
                           double *resid = wt + 3 * D1D;
                           double *jac = resid + 2; //jac will be row-major
                           double *hess = jac + 2 * 2;
                           findptsElementGEdge_t edge;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              edge = get_edge(elx, wtend, ei, constraint_workspace, constraint_init_t[j], j,
                                              D1D);
                           }
                           MFEM_SYNC_THREAD;

                           // compute basis function info upto 2nd derivative for tangential components
                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j < D1D)
                              {
                                 lagrange_eval_second_derivative(wt, tmp->r[de], j, gll1D, lagcoeff, D1D);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(d,x,nThreads)
                           {
                              if (d < dim)
                              {
                                 resid[d] = tmp->x[d];
                                 jac[2*d] = 0.0;
                                 jac[2*d + 1] = 0.0;
                                 hess[d] = 0.0;
                                 for (int k = 0; k < D1D; ++k)
                                 {
                                    resid[d] -= wt[k]*edge.x[d][k];
                                    jac[2*d] += wt[k]*edge.dxdn[d][k];
                                    jac[2*d+1] += wt[k+D1D]*edge.x[d][k];
                                    hess[d] += wt[k+2*D1D]*edge.x[d][k];
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;

                           // at this point, the Jacobian will be out of
                           // order for edge index 2 and 3 so we need to swap
                           // columns
                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j == 0)
                              {
                                 if (ei >= 2)
                                 {
                                    double temp1 = jac[1],
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

                           MFEM_FOREACH_THREAD(l,x,nThreads)
                           {
                              if (l == 0)
                              {
                                 // check prior step //
                                 if (!reject_prior_step_q(fpt, resid, tmp, tol))
                                 {
                                    // steep is negative of the gradient of the objective,
                                    // so steeps tells direction of decrease.
                                    // since we are only doing in normal direction (in reference space)
                                    // steep tells whether the objective is decreasing towards
                                    // the interior of the element or exterior (normal to the edge). If it is interior,
                                    // we relax the constraint and go search inside the element,
                                    // otherwise we minimize along the edge.
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
                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j == 0)
                              {
                                 int de = 0;
                                 int dn = 0;
                                 const int pi = point_index(tmp->flags & FLAG_MASK);
                                 const findptsElementGPT_t gpt = get_pt(elx, wtend, pi, D1D);

                                 const double *const pt_x = gpt.x,
                                                     *const jac = gpt.jac,
                                                            *const hes = gpt.hes;

                                 double resid[dim], steep[dim], sr[dim];
                                 for (int d = 0; d < dim; ++d)
                                 {
                                    resid[d] = fpt->x[d] - pt_x[d];
                                 }
                                 steep[0] = jac[0]*resid[0] + jac[2]*resid[1];
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
                                                      tmp->flags & FLAG_MASK & (3u<<(2*dn)),
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
                                                   tmp->flags & FLAG_MASK & (3u<<(2*dn)),
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
                        *newton_i = step+1;
                        break;
                     }
                     MFEM_SYNC_THREAD;
                     MFEM_FOREACH_THREAD(j,x,nThreads)
                     if (j == 0)
                     {
                        *tmp = *fpt;
                     }
                     MFEM_SYNC_THREAD;
                  } //for int step < 50
               } //findpts_el

               bool converged_internal = (fpt->flags & FLAG_MASK) == CONVERGED_FLAG;
               if (*code_i == CODE_NOT_FOUND || converged_internal || fpt->dist2 < *dist2_i)
               {
                  MFEM_FOREACH_THREAD(j,x,nThreads)
                  {
                     if (j == 0)
                     {
                        *el_i = el;
                        *code_i = converged_internal ? CODE_INTERNAL : CODE_BORDER;
                        *dist2_i = fpt->dist2;
                     }
                     if (j < dim)
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
                                       Array<unsigned int> &code,
                                       Array<unsigned int> &elem,
                                       Vector &ref,
                                       Vector &dist,
                                       Array<int> &newton,
                                       int npt)
{
   if (npt == 0) { return; }
   MFEM_VERIFY(dim == 2,"Function for 2D only");
   switch (DEV.dof1d)
   {
      case 3: return FindPointsLocal2D_Kernel<3>(npt, DEV.tol,
                                                    point_pos.Read(), point_pos_ordering,
                                                    gsl_mesh.Read(), NE_split_total,
                                                    DEV.o_wtend.Read(),
                                                    DEV.o_c.Read(), DEV.o_A.Read(),
                                                    DEV.o_min.Read(), DEV.o_max.Read(),
                                                    DEV.hash_n, DEV.o_hashMin.Read(),
                                                    DEV.o_hashFac.Read(),
                                                    DEV.ou_offset.ReadWrite(),
                                                    code.Write(),
                                                    elem.Write(),
                                                    ref.Write(),
                                                    dist.Write(),
                                                    DEV.gll1d.ReadWrite(),
                                                    DEV.lagcoeff.Read(),
                                                    newton.ReadWrite(),
                                                    DEV.info.ReadWrite());
      default: return FindPointsLocal2D_Kernel(npt, DEV.tol,
                                                  point_pos.Read(), point_pos_ordering,
                                                  gsl_mesh.Read(), NE_split_total,
                                                  DEV.o_wtend.Read(),
                                                  DEV.o_c.Read(),
                                                  DEV.o_A.Read(),
                                                  DEV.o_min.Read(),
                                                  DEV.o_max.Read(),
                                                  DEV.hash_n,
                                                  DEV.o_hashMin.Read(),
                                                  DEV.o_hashFac.Read(),
                                                  DEV.ou_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.ReadWrite(),
                                                  DEV.lagcoeff.Read(),
                                                  newton.ReadWrite(),
                                                  DEV.info.ReadWrite(),
                                                  DEV.dof1d);
   }
}


#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND

} // namespace mfem

#endif //ifdef MFEM_USE_GSLIB
