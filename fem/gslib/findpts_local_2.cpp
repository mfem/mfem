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
#define DIM 2
#define DIM2 4

struct findptsElementPoint_t
{
   double x[DIM], r[DIM], oldr[DIM], dist2, dist2p, tr;
   int flags;
};

struct findptsElementGEdge_t
{
   double *x[DIM], *dxdn[2];
};

struct findptsElementGPT_t
{
   double x[DIM], jac[DIM * DIM], hes[4];
};

struct dbl_range_t
{
   double min, max;
};
struct obbox_t
{
   double c0[DIM], A[DIM * DIM];
   dbl_range_t x[DIM];
};

struct findptsLocalHashData_t
{
   int hash_n;
   dbl_range_t bnd[DIM];
   double fac[DIM];
   unsigned int *offset;
   int max;
};

// Eval the ith Lagrange interpolant and its first derivative at x.
// Note: lCoeff stores pre-computed coefficients for fast evaluation.
static MFEM_HOST_DEVICE inline void lag_eval_first_der(double *p0, double x,
                                                       int i, const double *z,
                                                       const double *lCoeff,
                                                       int pN)
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
   p0[i] = lCoeff[i] * u0;
   p0[pN+i] = 2.0 * lCoeff[i] * u1;
}

// Eval the ith Lagrange interpolant and its first and second derivative at x.
// Note: lCoeff stores pre-computed coefficients for fast evaluation.
static MFEM_HOST_DEVICE inline void lag_eval_second_der(double *p0, double x,
                                                        int i, const double *z,
                                                        const double *lCoeff,
                                                        int pN)
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
   p0[i] = lCoeff[i] * u0;
   p0[pN+i] = 2.0 * lCoeff[i] * u1;
   p0[2*pN+i] = 8.0 * lCoeff[i] * u2;
}

// Axis-aligned bounding box test.
static MFEM_HOST_DEVICE inline double AABB_test(const obbox_t *const b,
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

// Axis-aligned bounding box test followed by oriented bounding-box test.
static MFEM_HOST_DEVICE inline double bbox_test(const obbox_t *const b,
                                                const double x[2])
{
   const double bxyz = AABB_test(b, x);
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

// Element index corresponding to hash mesh that the point is located in.
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

/*Solve Ax=y. A is row-major */
static MFEM_HOST_DEVICE inline void lin_solve_2(double x[2], const double A[4],
                                                const double y[2])
{
   const double idet = 1/(A[0]*A[3] - A[1]*A[2]);
   x[0] = idet*(A[3]*y[0] - A[1]*y[1]);
   x[1] = idet*(A[0]*y[1] - A[2]*y[0]);
}

/* L2 norm squared. */
static MFEM_HOST_DEVICE inline double l2norm2(const double x[2])
{
   return x[0] * x[0] + x[1] * x[1];
}

/* the bit structure of flags is CSSRR
   the C bit --- 1<<4 --- is set when the point is converged
   RR is 0 = 00b if r is unconstrained,
         1 = 01b if r is constrained at -1
         2 = 10b if r is constrained at +1
   SS is similarly for s constraints
   SSRR = smax,smin,rmax,rmin
*/
#define CONVERGED_FLAG (1u<<4)
#define FLAG_MASK 0x1fu // set all 5 bits to 1

/* Determine number of constraints based on the CTTSSRR bits. */
static MFEM_HOST_DEVICE inline int num_constrained(const int flags)
{
   const int y = flags | flags >> 1;
   return (y&1u) + (y>>2 & 1u);
}

// Helper functions. Assumes x = 0, 1, or 2.
static MFEM_HOST_DEVICE inline int plus_1_mod_2(const int x)
{
   return x ^ 1u;
}
// assumes x = 1 << i, with i < 4, returns i+1.
static MFEM_HOST_DEVICE inline int which_bit(const int x)
{
   const int y = x & 7u;
   return (y-(y>>2)) | ((x-1)&4u);
}

// Get edge index based on the SSRR bits.
static MFEM_HOST_DEVICE inline int edge_index(const int x)
{
   return which_bit(x)-1;
}

// Gets vertex index based SSRR bits.
static MFEM_HOST_DEVICE inline int point_index(const int x)
{
   return ((x>>1)&1u) | ((x>>2)&2u);
}

// compute (x,y) and (dxdn, dydn) data for all DOFs along the edge based on
// edge index. ei=0..3 corresponding to rmin, rmax, smin, smax.
static MFEM_HOST_DEVICE inline findptsElementGEdge_t
get_edge(const double *elx[2], const double *wtend, int ei,
         double *workspace,
         int &side_init, int j, int pN) // Assumes j < pN
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

   if (side_init != (1u << ei))
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

/* Check reduction in objective against prediction, and adjust trust region
   radius (p->tr) accordingly. May reject the prior step, returning 1; otherwise
   returns 0 sets res->dist2, res->index, res->x, res->oldr in any event,
   leaving res->r, res->dr, res->flags to be set when returning 0 */
static MFEM_HOST_DEVICE bool reject_prior_step_q(findptsElementPoint_t *res,
                                                 const double resid[2],
                                                 const findptsElementPoint_t *p,
                                                 const double tol)
{
   const double dist2 = l2norm2(resid);
   const double decr = p->dist2 - dist2;
   const double pred = p->dist2p;
   for (int d = 0; d < 2; ++d)
   {
      res->x[d] = p->x[d];
      res->oldr[d] = p->r[d];
   }
   res->dist2 = dist2;
   if (decr >= 0.01 * pred)
   {
      if (decr >= 0.9 * pred)
      {
         // very good iteration
         res->tr = p->tr * 2;
      }
      else
      {
         // good iteration
         res->tr = p->tr;
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
      res->tr = (v0 > v1 ? v0 : v1)/4;
      res->dist2 = p->dist2;
      for (int d = 0; d < 2; ++d)
      {
         res->r[d] = p->oldr[d];
      }
      res->flags = p->flags >> 5;
      res->dist2p = -std::numeric_limits<double>::max();
      if (pred < dist2 * tol)
      {
         res->flags |= CONVERGED_FLAG;
      }
      return true;
   }
}

/* Minimize 0.5||x* - x(r)||^2_2 using gradient-descent, with
   |dr| <= tr and |r0+dr|<=1*/
static MFEM_HOST_DEVICE void newton_area(findptsElementPoint_t *const res,
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
      double facc = 1;
      int new_flags = 0;
      double ress[2], y, JtJ, drc;
      ress[0] = resid[0] - (jac[0] * dr[0] + jac[1] * dr[1]);
      ress[1] = resid[1] - (jac[2] * dr[0] + jac[3] * dr[1]);
      /* y = J_u^T res */
      y = jac[de] * ress[0] + jac[2+de] * ress[1];
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
               if (f < facc)
               {
                  facc=f;
                  new_flags = 1u<<(2*de);
               }
            }
            else
            {
               double f = (ub-rz)/drc;
               if (f < facc)
               {
                  facc=f;
                  new_flags = 2u<<(2*de);
               }
            }
         }
      }

      dr[de] += facc * drc;
      flags |= new_flags;
      goto newton_area_relax;
   }

   /* check and possibly relax constraints */
newton_area_relax :
   {
      const int old_flags = flags;
      double ress[2], y[2];
      /* res := res_0 - J dr */
      ress[0] = resid[0] - (jac[0] * dr[0] + jac[1] * dr[1]);
      ress[1] = resid[1] - (jac[2] * dr[0] + jac[3] * dr[1]);
      /* y := J^T res */
      y[0] = jac[0] * ress[0] + jac[2] * ress[1];
      y[1] = jac[1] * ress[0] + jac[3] * ress[1];
      for (int dd = 0; dd < 2; ++dd)
      {
         int f = flags >> (2 * dd) & 3u;
         if (f)
         {
            dr[dd] = bnd[2 * dd + (f - 1)] - r0[dd];
            if (dr[dd] * y[dd] < 0)
            {
               flags &= ~(3u << (2 * dd));
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
      res->dist2p = resid[0] * resid[0] + resid[1] * resid[1] -
                    (res0 * res0 + res1 * res1);
   }
   for (int dd = 0; dd < 2; ++dd)
   {
      int f = flags >> (2 * dd) & 3u;
      res->r[dd] = f == 0 ? r0[dd] + dr[dd] : (f == 1 ? -1 : 1);
   }
   res->flags = flags | (p->flags << 5);
}

// Full Newton solve on the face. One of r/s/t is constrained.
static MFEM_HOST_DEVICE inline void newton_edge(findptsElementPoint_t *const
                                                res,
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
   res->r[de] = nr;
   res->r[dn]=p->r[dn];
   res->dist2p = -v;
   res->flags = flags | new_flags | (p->flags << 5);
}

// Find closest mesh node to the sought point.
static MFEM_HOST_DEVICE void seed_j(const double *elx[2],
                                    const double x[2],
                                    const double *z, //GLL point locations [-1, 1]
                                    double *dist2,
                                    double *r[2],
                                    const int j,
                                    const int pN)
{
   dist2[j] = std::numeric_limits<double>::max();

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
      const double dist2_jkl = l2norm2(dx);
      if (dist2[j] > dist2_jkl)
      {
         dist2[j] = dist2_jkl;
         r[0][j] = zr;
         r[1][j] = zs;
      }
   }
}

/* Compute contribution towards function value and its derivatives in each
   reference direction. */
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
                                     const double *boxinfo,
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
                                     const int pN = 0)
{
#define MAX_CONST(a, b) (((a) > (b)) ? (a) : (b))
   const int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
   const int D1D = T_D1D ? T_D1D : pN;
   const int p_NE = D1D*D1D;
   const int p_NEL = nel*p_NE;
   MFEM_VERIFY(MD1 <= DofQuadLimits::MAX_D1D,
               "Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D != 0, "Polynomial order not specified.");
   const int nThreads = D1D*DIM;

   mfem::forall_2D(npt, nThreads, 1, [=] MFEM_HOST_DEVICE (int i)
   {
      // 3D1D for seed, 10D1D+6 for area, 3D1D+9 for edge
      constexpr int size1 = 10*MD1 + 6;
      constexpr int size2 = MD1*4;            // edge constraints
      constexpr int size3 = MD1*MD1*MD1*DIM;  // local element coordinates

      MFEM_SHARED double r_workspace[size1];
      MFEM_SHARED findptsElementPoint_t el_pts[2];

      MFEM_SHARED double constraint_workspace[size2];
      MFEM_SHARED int edge_init;

      MFEM_SHARED double elem_coords[MD1 <= 6 ? size3 : 1];

      double *r_workspace_ptr;
      findptsElementPoint_t *fpt, *tmp;
      MFEM_FOREACH_THREAD(j,x,nThreads)
      {
         r_workspace_ptr = r_workspace;
         fpt = el_pts + 0;
         tmp = el_pts + 1;
      }
      MFEM_SYNC_THREAD;

      int id_x = point_pos_ordering == 0 ? i : i*DIM;
      int id_y = point_pos_ordering == 0 ? i+npt : i*DIM+1;
      double x_i[2] = {x[id_x], x[id_y]};

      unsigned int *code_i = code_base + i;
      unsigned int *el_i = el_base + i;
      double *r_i = r_base + DIM * i;
      double *dist2_i = dist2_base + i;

      // Initialize the code and dist
      *code_i = CODE_NOT_FOUND;
      *dist2_i = std::numeric_limits<double>::max();

      //// map_points_to_els ////
      findptsLocalHashData_t hash;
      for (int d = 0; d < DIM; ++d)
      {
         hash.bnd[d].min = hashMin[d];
         hash.fac[d] = hashFac[d];
      }
      hash.hash_n = hash_n;
      hash.offset = hashOffset;
      const int hi = hash_index(&hash, x_i);
      const unsigned int *elp = hash.offset+hash.offset[hi],
                          *const ele = hash.offset+hash.offset[hi+1];

      for (; elp != ele; ++elp)
      {
         //elp
         const unsigned int el = *elp;

         // construct obbox_t on the fly from data
         obbox_t box;
         int n_box_ents = 3*DIM + DIM2;

         for (int idx = 0; idx < DIM; ++idx)
         {
            box.c0[idx] = boxinfo[n_box_ents*el + idx];
            box.x[idx].min = boxinfo[n_box_ents*el + DIM + idx];
            box.x[idx].max = boxinfo[n_box_ents*el + 2*DIM + idx];
         }

         for (int idx = 0; idx < DIM2; ++idx)
         {
            box.A[idx] = boxinfo[n_box_ents*el + 3*DIM + idx];
         }

         if (bbox_test(&box, x_i) < 0) { continue; }

         //// findpts_local ////
         {
            // read element coordinates into shared memory
            if (MD1 <= 6)
            {
               MFEM_FOREACH_THREAD(j,x,nThreads)
               {
                  const int qp = j % D1D;
                  const int d = j / D1D;
                  for (int k = 0; k < D1D; ++k)
                  {
                     const int jk = qp + k * D1D;
                     elem_coords[jk + d*p_NE] =
                        xElemCoord[jk + el*p_NE + d*p_NEL];
                  }
               }
               MFEM_SYNC_THREAD;
            }

            const double *elx[DIM];
            for (int d = 0; d < DIM; d++)
            {
               elx[d] = MD1<= 6 ? &elem_coords[d*p_NE] :
                        xElemCoord + d*p_NEL + el * p_NE;
            }

            //// findpts_el ////
            {
               MFEM_FOREACH_THREAD(j,x,1)
               {
                  fpt->dist2 = std::numeric_limits<double>::max();
                  fpt->dist2p = 0;
                  fpt->tr = 1;
                  edge_init = 0;
               }
               MFEM_FOREACH_THREAD(j,x,DIM)
               {
                  fpt->x[j] = x_i[j];
               }
               MFEM_SYNC_THREAD;

               //// seed ////
               {
                  double *dist2_temp = r_workspace_ptr;
                  double *r_temp[DIM];
                  for (int d = 0; d < DIM; ++d)
                  {
                     r_temp[d] = dist2_temp + (1 + d) * D1D;
                  }

                  MFEM_FOREACH_THREAD(j,x,D1D)
                  {
                     seed_j(elx, x_i, gll1D, dist2_temp, r_temp, j, D1D);
                  }
                  MFEM_SYNC_THREAD;

                  MFEM_FOREACH_THREAD(j,x,1)
                  {
                     fpt->dist2 = std::numeric_limits<double>::max();
                     for (int jj = 0; jj < D1D; ++jj)
                     {
                        if (dist2_temp[jj] < fpt->dist2)
                        {
                           fpt->dist2 = dist2_temp[jj];
                           for (int d = 0; d < DIM; ++d)
                           {
                              fpt->r[d] = r_temp[d][jj];
                           }
                        }
                     }
                  }
                  MFEM_SYNC_THREAD;
               } //seed done

               MFEM_FOREACH_THREAD(j,x,1)
               {
                  tmp->dist2 = std::numeric_limits<double>::max();
                  tmp->dist2p = 0;
                  tmp->tr = 1;
                  tmp->flags = 0;
               }
               MFEM_FOREACH_THREAD(j,x,DIM)
               {
                  tmp->x[j] = fpt->x[j];
                  tmp->r[j] = fpt->r[j];
               }
               MFEM_SYNC_THREAD;

               for (int step = 0; step < 50; step++)
               {
                  switch (num_constrained(tmp->flags & FLAG_MASK))
                  {
                     case 0:   // findpt_area
                     {
                        double *wtr = r_workspace_ptr;
                        double *resid = wtr + 4 * D1D;
                        double *jac = resid + 2;
                        double *resid_temp = jac + 4;
                        double *jac_temp = resid_temp + 2 * D1D;

                        MFEM_FOREACH_THREAD(j,x,nThreads)
                        {
                           const int qp = j % D1D;
                           const int d = j / D1D;
                           lag_eval_first_der(wtr + 2*d*D1D, tmp->r[d], qp,
                                              gll1D, lagcoeff, D1D);
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(j,x,nThreads)
                        {
                           const int qp = j % D1D;
                           const int d = j / D1D;
                           double *idx = jac_temp+2*d+4*qp;
                           resid_temp[d+qp*2] = tensor_ig2_j(idx, wtr,
                                                             wtr+D1D,
                                                             wtr+2*D1D,
                                                             wtr+3*D1D,
                                                             elx[d], qp,
                                                             D1D);
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(l,x,2)
                        {
                           resid[l] = tmp->x[l];
                           for (int j = 0; j < D1D; ++j)
                           {
                              resid[l] -= resid_temp[l + j * 2];
                           }
                        }
                        MFEM_FOREACH_THREAD(l,x,4)
                        {
                           jac[l] = 0;
                           for (int j = 0; j < D1D; ++j)
                           {
                              jac[l] += jac_temp[l + j * 4];
                           }
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(l,x,1)
                        {
                           if (!reject_prior_step_q(fpt, resid, tmp, tol))
                           {
                              newton_area(fpt, jac, resid, tmp, tol);
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

                        MFEM_FOREACH_THREAD(j,x,D1D)
                        {
                           edge = get_edge(elx, wtend, ei,
                                           constraint_workspace,
                                           edge_init, j,
                                           D1D);
                        }
                        MFEM_SYNC_THREAD;

                        // compute basis function info upto 2nd derivative.
                        MFEM_FOREACH_THREAD(j,x,D1D)
                        {
                           if (j == 0) { edge_init = 1u << ei; }
                           lag_eval_second_der(wt, tmp->r[de], j, gll1D,
                                               lagcoeff, D1D);
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(d,x,DIM)
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
                        MFEM_SYNC_THREAD;

                        // at this point, the Jacobian will be out of
                        // order for edge index 2 and 3 so we need to swap
                        // columns
                        MFEM_FOREACH_THREAD(j,x,1)
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
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(l,x,1)
                        {
                           // check prior step //
                           if (!reject_prior_step_q(fpt, resid, tmp, tol))
                           {
                              // steep is negative of the gradient of the
                              // objective, so it tells direction of
                              // decrease.
                              double steep = resid[0] * jac[  dn]
                                             + resid[1] * jac[2+dn];

                              if (steep * tmp->r[dn] < 0)
                              {
                                 newton_area(fpt, jac, resid, tmp, tol);
                              }
                              else
                              {
                                 newton_edge(fpt, jac, hess[2], resid, de,
                                             dn, tmp->flags & FLAG_MASK,
                                             tmp, tol);
                              }
                           }
                        }
                        MFEM_SYNC_THREAD;
                        break;
                     }
                     case 2:   // findpts_pt
                     {
                        MFEM_FOREACH_THREAD(j,x,1)
                        {
                           int de = 0;
                           int dn = 0;
                           const int pi = point_index(tmp->flags & FLAG_MASK);
                           const findptsElementGPT_t gpt =
                              get_pt(elx, wtend, pi, D1D);

                           const double *const pt_x = gpt.x;
                           const double *const jac = gpt.jac;
                           const double *const hes = gpt.hes;

                           double resid[DIM], steep[DIM], sr[DIM];
                           for (int d = 0; d < DIM; ++d)
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
                                    newton_edge(fpt, jac, rh, resid, de, dn,
                                                tmp->flags &
                                                FLAG_MASK &
                                                (3u<<(2*dn)),
                                                tmp, tol);
                                 }
                              }
                              else if (sr[1]<0)
                              {
                                 de=1;
                                 dn=0;
                                 const double rh = resid[0]*hes[de]+
                                                   resid[1]*hes[2+de];
                                 newton_edge(fpt, jac, rh, resid, de, dn,
                                             tmp->flags &
                                             FLAG_MASK &
                                             (3u<<(2*dn)),
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
                        MFEM_SYNC_THREAD;
                        break;
                     } //case 3
                  } //switch
                  if (fpt->flags & CONVERGED_FLAG)
                  {
                     break;
                  }
                  MFEM_SYNC_THREAD;
                  MFEM_FOREACH_THREAD(j,x,1)
                  {
                     *tmp = *fpt;
                  }
                  MFEM_SYNC_THREAD;
               } //for int step < 50
            } //findpts_el

            bool converged_internal = (fpt->flags&FLAG_MASK)==CONVERGED_FLAG;
            if (*code_i == CODE_NOT_FOUND || converged_internal ||
                fpt->dist2 < *dist2_i)
            {
               MFEM_FOREACH_THREAD(j,x,1)
               {
                  *el_i = el;
                  *code_i = converged_internal ? CODE_INTERNAL :
                            CODE_BORDER;
                  *dist2_i = fpt->dist2;
               }
               MFEM_FOREACH_THREAD(j,x,DIM)
               {
                  r_i[j] = fpt->r[j];
               }
               MFEM_SYNC_THREAD;
               if (converged_internal)
               {
                  break;
               }
            }
         } //findpts_local
      } //elp
   });
}

void FindPointsGSLIB::FindPointsLocal2(const Vector &point_pos,
                                       int point_pos_ordering,
                                       Array<unsigned int> &code,
                                       Array<unsigned int> &elem,
                                       Vector &ref,
                                       Vector &dist,
                                       int npt)
{
   if (npt == 0) { return; }
   auto pp = point_pos.Read();
   auto pgslm = gsl_mesh.Read();
   auto pwt = DEV.wtend.Read();
   auto pbb = DEV.bb.Read();
   auto plhm = DEV.loc_hash_min.Read();
   auto plhf = DEV.loc_hash_fac.Read();
   auto plho = DEV.loc_hash_offset.ReadWrite();
   auto pcode = code.Write();
   auto pelem = elem.Write();
   auto pref = ref.Write();
   auto pdist = dist.Write();
   auto pgll1d = DEV.gll1d.ReadWrite();
   auto plc = DEV.lagcoeff.Read();

   switch (DEV.dof1d)
   {
      case 2: return FindPointsLocal2D_Kernel<2>(npt, DEV.newt_tol,
                                                    pp, point_pos_ordering,
                                                    pgslm, NE_split_total,
                                                    pwt, pbb,
                                                    DEV.h_nx, plhm, plhf, plho,
                                                    pcode, pelem, pref, pdist,
                                                    pgll1d, plc);
      case 3: return FindPointsLocal2D_Kernel<3>(npt, DEV.newt_tol,
                                                    pp, point_pos_ordering,
                                                    pgslm, NE_split_total,
                                                    pwt, pbb,
                                                    DEV.h_nx, plhm, plhf, plho,
                                                    pcode, pelem, pref, pdist,
                                                    pgll1d, plc);
      case 4: return FindPointsLocal2D_Kernel<4>(npt, DEV.newt_tol,
                                                    pp, point_pos_ordering,
                                                    pgslm, NE_split_total,
                                                    pwt, pbb,
                                                    DEV.h_nx, plhm, plhf, plho,
                                                    pcode, pelem, pref, pdist,
                                                    pgll1d, plc);
      case 5: return FindPointsLocal2D_Kernel<5>(npt, DEV.newt_tol,
                                                    pp, point_pos_ordering,
                                                    pgslm, NE_split_total,
                                                    pwt, pbb,
                                                    DEV.h_nx, plhm, plhf, plho,
                                                    pcode, pelem, pref, pdist,
                                                    pgll1d, plc);
      default: return FindPointsLocal2D_Kernel(npt, DEV.newt_tol,
                                                  pp, point_pos_ordering,
                                                  pgslm, NE_split_total,
                                                  pwt, pbb,
                                                  DEV.h_nx, plhm, plhf, plho,
                                                  pcode, pelem, pref, pdist,
                                                  pgll1d, plc, DEV.dof1d);
   }
}
#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND
#else
void FindPointsGSLIB::FindPointsLocal2(const Vector &point_pos,
                                       int point_pos_ordering,
                                       Array<unsigned int> &code,
                                       Array<unsigned int> &elem,
                                       Vector &ref,
                                       Vector &dist,
                                       int npt) {};
#endif
} // namespace mfem

#endif //ifdef MFEM_USE_GSLIB
