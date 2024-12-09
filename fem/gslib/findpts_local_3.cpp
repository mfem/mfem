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
#define DIM 3
#define DIM2 DIM*DIM
#define pMax 10

struct findptsPt
{
   double x[DIM], r[DIM], oldr[DIM], dist2, dist2p, tr;
   int flags;
};

struct findptsElemFace
{
   double *x[DIM], *dxdn[DIM];
};

struct findptsElemEdge
{
   double *x[DIM], *dxdn1[DIM], *dxdn2[DIM], *d2xdn1[DIM], *d2xdn2[DIM];
};

struct findptsElemPt
{
   double x[DIM], jac[DIM * DIM], hes[18];
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
   // int max;
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
         double d_j = 2*(x-z[j]);
         u1 = d_j*u1+u0;
         u0 = d_j*u0;
      }
   }
   p0[i] = lCoeff[i]*u0;
   p0[pN+i] = 2.0*lCoeff[i]*u1;
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
         double d_j = 2*(x-z[j]);
         u2 = d_j*u2+u1;
         u1 = d_j*u1+u0;
         u0 = d_j*u0;
      }
   }
   p0[i] = lCoeff[i]*u0;
   p0[pN+i] = 2.0*lCoeff[i]*u1;
   p0[2*pN+i] = 8.0*lCoeff[i]*u2;
}

// Axis-aligned bounding box test.
static MFEM_HOST_DEVICE inline double AABB_test(const obbox_t *const b,
                                                const double x[3])
{
   double b_d;
   for (int d = 0; d < 3; ++d)
   {
      b_d = (x[d]-b->x[d].min)*(b->x[d].max-x[d]);
      if (b_d < 0) { return b_d; }
   }
   return b_d;
}

// Axis-aligned bounding box test followed by oriented bounding-box test.
static MFEM_HOST_DEVICE inline double bbox_test(const obbox_t *const b,
                                                const double x[3])
{
   const double bxyz = AABB_test(b, x);
   if (bxyz < 0)
   {
      return bxyz;
   }
   else
   {
      double dxyz[3];
      for (int d = 0; d < 3; ++d)
      {
         dxyz[d] = x[d]-b->c0[d];
      }
      double test = 1;
      for (int d = 0; d < 3; ++d)
      {
         double rst = 0;
         for (int e = 0; e < 3; ++e)
         {
            rst += b->A[d*3+e]*dxyz[e];
         }
         double brst = (rst+1)*(1-rst);
         test = test < 0 ? test : brst;
      }
      return test;
   }
}

// Element index corresponding to hash mesh that the point is located in.
static MFEM_HOST_DEVICE inline int hash_index(const findptsLocalHashData_t *p,
                                              const double x[3])
{
   const int n = p->hash_n;
   int sum = 0;
   for (int d = 3-1; d >= 0; --d)
   {
      sum *= n;
      int i = (int)floor((x[d]-p->bnd[d].min)*p->fac[d]);
      sum += i < 0 ? 0 : (n-1 < i ? n-1 : i);
   }
   return sum;
}

// Solve Ax=y. A is row-major.
static MFEM_HOST_DEVICE inline void lin_solve_3(double x[3], const double A[9],
                                                const double y[3])
{
   const double a = A[4]*A[8]-A[5]*A[7], b = A[5]*A[6]-A[3]*A[8],
                c = A[3]*A[7]-A[4]*A[6],
                idet = 1 / (A[0]*a+A[1]*b+A[2]*c);
   const double inv0 = a, inv1 = A[2]*A[7]-A[1]*A[8],
                inv2 = A[1]*A[5]-A[2]*A[4], inv3 = b,
                inv4 = A[0]*A[8]-A[2]*A[6], inv5 = A[2]*A[3]-A[0]*A[5],
                inv6 = c, inv7 = A[1]*A[6]-A[0]*A[7],
                inv8 = A[0]*A[4]-A[1]*A[3];
   x[0] = idet*(inv0*y[0]+inv1*y[1]+inv2*y[2]);
   x[1] = idet*(inv3*y[0]+inv4*y[1]+inv5*y[2]);
   x[2] = idet*(inv6*y[0]+inv7*y[1]+inv8*y[2]);
}

// Solve Ax=y. A is a symmetric 2x2 matrix.
static MFEM_HOST_DEVICE inline void lin_solve_sym_2(double x[2],
                                                    const double A[3],
                                                    const double y[2])
{
   const double idet = 1 / (A[0]*A[2]-A[1]*A[1]);
   x[0] = idet*(A[2]*y[0]-A[1]*y[1]);
   x[1] = idet*(A[0]*y[1]-A[1]*y[0]);
}

// L2 norm.
static MFEM_HOST_DEVICE inline double l2norm2(const double x[3])
{
   return x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
}

/* the bit structure of flags is CTTSSRR
   the C bit --- 1<<6 --- is set when the point is converged
   RR is 0 = 00b if r is unconstrained,
         1 = 01b if r is constrained at -1
         2 = 10b if r is constrained at +1
   SS, TT are similarly for s and t constraints
   TT is ignored, but treated as set when 3==2
*/
#define CONVERGED_FLAG (1u << 6)
#define FLAG_MASK 0x7fu // set all 7 bits to 1

// Determine number of constraints based on the CTTSSRR bits.
static MFEM_HOST_DEVICE inline int num_constrained(const int flags)
{
   const int y = flags | flags >> 1;
   return (y & 1u)+(y >> 2 & 1u)+(((3 == 2) | y >> 4) & 1u);
}

// Helper functions. Assumes x = 0, 1, or 2.
static MFEM_HOST_DEVICE inline int plus_1_mod_3(const int x)
{
   return ((x | x >> 1)+1) & 3u;
}
static MFEM_HOST_DEVICE inline int plus_2_mod_3(const int x)
{
   const int y = (x-1) & 3u;
   return y ^ (y >> 1);
}

// Assumes x = 1 << i, with i < 6, returns i+1.
static MFEM_HOST_DEVICE inline int which_bit(const int x)
{
   const int y = x & 7u;
   return (y-(y >> 2)) | ((x-1) & 4u) | (x >> 4);
}

// Get face index based on the TTSSRR bits.
static MFEM_HOST_DEVICE inline int face_index(const int x)
{
   return which_bit(x)-1;
}

// Get edge index based on the TTSSRR bits.
static MFEM_HOST_DEVICE inline int edge_index(const int x)
{
   const int y = ~((x >> 1) | x);
   const int RTSR = ((x >> 1) & 1u) | ((x >> 2) & 2u) |
                    ((x >> 3) & 4u) | ((x << 2) & 8u);
   const int re = RTSR >> 1;
   const int se = 4u | RTSR >> 2;
   const int te = 8u | (RTSR & 3u);
   return ((0u-(y & 1u)) & re) | ((0u-((y >> 2) & 1u)) & se) |
          ((0u-((y >> 4) & 1u)) & te);
}

// Get point index based on the TTSSRR bits.
static MFEM_HOST_DEVICE inline int point_index(const int x)
{
   return ((x >> 1) & 1u) | ((x >> 2) & 2u) | ((x >> 3) & 4u);
}

// Gets mesh nodal coordinates and normal derivative in reference-space at the
// given face.
static MFEM_HOST_DEVICE inline findptsElemFace
get_face(const double *elx[3], const double *wtend, int fi, double *workspace,
         int &side_init, int jidx, int pN) // jidx < 3*pN
{
   const int dn = fi >> 1, d1 = plus_1_mod_3(dn), d2 = plus_2_mod_3(dn);
   const int side_n = fi & 1;
   const int p_Nfr = pN*pN;
   findptsElemFace face;
   const int jj = jidx % pN;
   const int dd = jidx / pN;
   for (int d = 0; d < 3; ++d)
   {
      face.x[d] = workspace+d*p_Nfr;
      face.dxdn[d] = workspace+(3+d)*p_Nfr;
   }

   if (side_init != (1u << fi))
   {
      const int e_stride[3] = {1, pN, pN*pN};
#define ELX(d, j, k, l) elx[d][j*e_stride[d1]+k*e_stride[d2]+l*e_stride[dn]]
      for (int k = 0; k < pN; ++k)
      {
         // copy first/last entries in normal direction
         face.x[dd][jj+k*pN] = ELX(dd, jj, k, side_n*(pN-1));

         // tensor product between elx and derivative in normal direction
         double sum_l = 0;
         for (int l = 0; l < pN; ++l)
         {
            sum_l += wtend[pN+l]*ELX(dd, jj, k, l);
         }
         face.dxdn[dd][jj+k*pN] = sum_l;
      }
#undef ELX
   }
   return face;
}

// Gets mesh nodal coordinates and normal derivatives in reference-space at the
// given edge.
static MFEM_HOST_DEVICE inline findptsElemEdge
get_edge(const double *elx[3], const double *wtend, int ei, double *workspace,
         int &side_init, int jidx, int pN)
{
   findptsElemEdge edge;
   const int de = ei >> 2, dn1 = plus_1_mod_3(de), dn2 = plus_2_mod_3(de);
   const int side_n1 = ei & 1, side_n2 = (ei & 2) >> 1;

   const int in1 = side_n1*(pN-1), in2 = side_n2*(pN-1);
   const double *wt1 = wtend+side_n1*pN*3;
   const double *wt2 = wtend+side_n2*pN*3;
   const int jj = jidx % pN;
   const int dd = jidx / pN;
   for (int d = 0; d < 3; ++d)
   {
      edge.x[d] = workspace+d*pN;
      edge.dxdn1[d] = workspace+(3+d)*pN;
      edge.dxdn2[d] = workspace+(6+d)*pN;
      edge.d2xdn1[d] = workspace+(9+d)*pN;
      edge.d2xdn2[d] = workspace+(12+d)*pN;
   }

   if (jidx >= 3*pN) { return edge; }

   if (side_init != (64u << ei))
   {
      const int e_stride[3] = {1, pN, pN*pN};
#define ELX(d, j, k, l) elx[d][j*e_stride[de]+k*e_stride[dn1]+l*e_stride[dn2]]
      // copy first/last entries in normal directions
      edge.x[dd][jj] = ELX(dd, jj, in1, in2);
      // tensor product between elx (w/ first/last entries in second direction)
      // and the derivatives in the first normal direction
      double sums_k[2] = {0, 0};
      for (int k = 0; k < pN; ++k)
      {
         sums_k[0] += wt1[pN+k]*ELX(dd, jj, k, in2);
         sums_k[1] += wt1[2*pN+k]*ELX(dd, jj, k, in2);
      }
      edge.dxdn1[dd][jj] = sums_k[0];
      edge.d2xdn1[dd][jj] = sums_k[1];
      // tensor product between elx (w/ first/last entries in first direction)
      // and the derivatives in the second normal direction
      sums_k[0] = 0, sums_k[1] = 0;
      for (int k = 0; k < pN; ++k)
      {
         sums_k[0] += wt2[pN+k]*ELX(dd, jj, in1, k);
         sums_k[1] += wt2[2*pN+k]*ELX(dd, jj, in1, k);
      }
      edge.dxdn2[dd][jj] = sums_k[0];
      edge.d2xdn2[dd][jj] = sums_k[1];
#undef ELX
   }
   return edge;
}

// Gets nodal coordinate and derivatives at the given vertex of the element.
static MFEM_HOST_DEVICE inline findptsElemPt get_pt(const double *elx[3],
                                                    const double *wtend,
                                                    int pi, int pN)
{
   const int side_n1 = pi & 1, side_n2 = (pi >> 1) & 1, side_n3 = (pi >> 2) & 1;
   const int in1 = side_n1*(pN-1), in2 = side_n2*(pN-1),
             in3 = side_n3*(pN-1);
   const int hes_stride = (3+1)*3 / 2;
   findptsElemPt pt;

#define ELX(d, j, k, l) elx[d][j+k*pN+l*pN*pN]
   for (int d = 0; d < 3; ++d)
   {
      pt.x[d] = ELX(d, side_n1*(pN-1), side_n2*(pN-1),
                    side_n3*(pN-1));

      const double *wt1 = wtend+pN*(1+3*side_n1);
      const double *wt2 = wtend+pN*(1+3*side_n2);
      const double *wt3 = wtend+pN*(1+3*side_n3);

      for (int i = 0; i < 3; ++i)
      {
         pt.jac[3*d+i] = 0;
      }
      for (int i = 0; i < hes_stride; ++i)
      {
         pt.hes[hes_stride*d+i] = 0;
      }

      for (int j = 0; j < pN; ++j)
      {
         pt.jac[3*d+0] += wt1[j]*ELX(d, j, in2, in3);
         pt.hes[hes_stride*d] += wt1[pN+j]*ELX(d, j, in2, in3);
      }

      const int hes_off = hes_stride*d+hes_stride / 2;
      for (int k = 0; k < pN; ++k)
      {
         pt.jac[3*d+1] += wt2[k]*ELX(d, in1, k, in3);
         pt.hes[hes_off] += wt2[pN+k]*ELX(d, in1, k, in3);
      }

      for (int l = 0; l < pN; ++l)
      {
         pt.jac[3*d+2] += wt3[l]*ELX(d, in1, in2, l);
         pt.hes[hes_stride*d+5] += wt3[pN+l]*ELX(d, in1, in2, l);
      }

      for (int l = 0; l < pN; ++l)
      {
         double sum_k = 0, sum_j = 0;
         for (int k = 0; k < pN; ++k)
         {
            sum_k += wt2[k]*ELX(d, in1, k, l);
         }
         for (int j = 0; j < pN; ++j)
         {
            sum_j += wt1[j]*ELX(d, j, in2, l);
         }
         pt.hes[hes_stride*d+2] += wt3[l]*sum_j;
         pt.hes[hes_stride*d+4] += wt3[l]*sum_k;
      }
      for (int k = 0; k < pN; ++k)
      {
         double sum_j = 0;
         for (int j = 0; j < pN; ++j)
         {
            sum_j += wt1[j]*ELX(d, j, k, in3);
         }
         pt.hes[hes_stride*d+1] += wt2[k]*sum_j;
      }
#undef ELX
   }
   return pt;
}

/* Check reduction in objective against prediction, and adjust trust region
   radius (p->tr) accordingly. May reject the prior step, returning 1; otherwise
   returns 0 sets res->dist2, res->index, res->x, res->oldr in any event,
   leaving res->r, res->dr, res->flags to be set when returning 0 */
static MFEM_HOST_DEVICE bool reject_prior_step_q(findptsPt *res,
                                                 const double resid[3],
                                                 const findptsPt *p,
                                                 const double tol)
{
   const double dist2 = l2norm2(resid);
   const double decr = p->dist2-dist2;
   const double pred = p->dist2p;
   for (int d = 0; d < 3; ++d)
   {
      res->x[d] = p->x[d];
      res->oldr[d] = p->r[d];
   }
   res->dist2 = dist2;
   if (decr >= 0.01*pred)
   {
      if (decr >= 0.9*pred)
      {
         // very good iteration
         res->tr = p->tr*2;
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
      double v0 = fabs(p->r[0]-p->oldr[0]);
      double v1 = fabs(p->r[1]-p->oldr[1]);
      double v2 = fabs(p->r[2]-p->oldr[2]);
      res->tr = (v1 > v2 ? (v0 > v1 ? v0 : v1) : (v0 > v2 ? v0 : v2)) / 4;
      res->dist2 = p->dist2;
      for (int d = 0; d < 3; ++d)
      {
         res->r[d] = p->oldr[d];
      }
      res->flags = p->flags >> 7;
      res->dist2p = -std::numeric_limits<double>::max();
      if (pred < dist2*tol)
      {
         res->flags |= CONVERGED_FLAG;
      }
      return true;
   }
}

/* minimize 0.5||x* - x(r)||^2_2 using gradient-descent, with
   |dr| <= tr and |r0+dr|<=1*/
static MFEM_HOST_DEVICE void newton_vol(findptsPt *const res,
                                        const double jac[9],
                                        const double resid[3],
                                        const findptsPt *const p,
                                        const double tol)
{
   const double tr = p->tr;
   double bnd[6] = {-1, 1, -1, 1, -1, 1};
   double r0[3];
   double dr[3], fac;
   int d, mask, flags;
   r0[0] = p->r[0], r0[1] = p->r[1], r0[2] = p->r[2];

   mask = 0x3fu;
   for (d = 0; d < 3; ++d)
   {
      if (r0[d]-tr > -1)
      {
         bnd[2*d] = r0[d]-tr, mask ^= 1u << (2*d);
      }
      if (r0[d]+tr < 1)
      {
         bnd[2*d+1] = r0[d]+tr, mask ^= 2u << (2*d);
      }
   }

   lin_solve_3(dr, jac, resid);

   fac = 1, flags = 0;
   for (d = 0; d < 3; ++d)
   {
      double nr = r0[d]+dr[d];
      if ((nr-bnd[2*d])*(bnd[2*d+1]-nr) >= 0)
      {
         continue;
      }
      if (nr < bnd[2*d])
      {
         double f = (bnd[2*d]-r0[d]) / dr[d];
         if (f < fac)
         {
            fac = f, flags = 1u << (2*d);
         }
      }
      else
      {
         double f = (bnd[2*d+1]-r0[d]) / dr[d];
         if (f < fac)
         {
            fac = f, flags = 2u << (2*d);
         }
      }
   }

   if (flags == 0)
   {
      goto newton_vol_fin;
   }

   for (d = 0; d < 3; ++d)
   {
      dr[d] *= fac;
   }

newton_vol_face :
   {
      const int fi = face_index(flags);
      const int dn = fi >> 1, d1 = plus_1_mod_3(dn), d2 = plus_2_mod_3(dn);
      double drc[2], facc = 1;
      int new_flags = 0;
      double ress[3], y[2], JtJ[3];
      ress[0] = resid[0]-(jac[0]*dr[0]+jac[1]*dr[1]+jac[2]*dr[2]);
      ress[1] = resid[1]-(jac[3]*dr[0]+jac[4]*dr[1]+jac[5]*dr[2]);
      ress[2] = resid[2]-(jac[6]*dr[0]+jac[7]*dr[1]+jac[8]*dr[2]);
      /* y = J_u^T res */
      y[0] = jac[d1]*ress[0]+jac[3+d1]*ress[1]+jac[6+d1]*ress[2];
      y[1] = jac[d2]*ress[0]+jac[3+d2]*ress[1]+jac[6+d2]*ress[2];
      /* JtJ = J_u^T J_u */
      JtJ[0] = jac[d1]*jac[d1]+jac[3+d1]*jac[3+d1] +
               jac[6+d1]*jac[6 +d1];
      JtJ[1] = jac[d1]*jac[d2]+jac[3+d1]*jac[3+d2] +
               jac[6+d1]*jac[6+d2];
      JtJ[2] = jac[d2]*jac[d2]+jac[3+d2]*jac[3+d2] +
               jac[6+d2]*jac[6+d2];
      lin_solve_sym_2(drc, JtJ, y);
#define CHECK_CONSTRAINT(drcd, d3)                                             \
{                                                                              \
   const double rz = r0[d3]+dr[d3], lb = bnd[2*d3], ub = bnd[2*d3+1];          \
   const double delta = drcd, nr = r0[d3]+(dr[d3]+delta);                      \
   if ((nr-lb)*(ub-nr) < 0) {                                                  \
      if (nr < lb) {                                                           \
         double f = (lb-rz) / delta;                                           \
         if (f < fac) {                                                        \
         fac = f; new_flags = 1u << (2*d3);                                    \
         }                                                                     \
      }                                                                        \
      else {                                                                   \
         double f = (ub-rz) / delta;                                           \
         if (f < fac) {                                                        \
         fac = f; new_flags = 2u << (2*d3);                                    \
         }                                                                     \
      }                                                                        \
   }                                                                           \
}
      CHECK_CONSTRAINT(drc[0], d1);
      CHECK_CONSTRAINT(drc[1], d2);
      dr[d1] += facc*drc[0], dr[d2] += facc*drc[1];
      if (new_flags == 0)
      {
         goto newton_vol_fin;
      }
      flags |= new_flags;
   }

newton_vol_edge :
   {
      const int ei = edge_index(flags);
      const int de = ei >> 2;
      double facc = 1;
      int new_flags = 0;
      double ress[3], y, JtJ, drc;
      ress[0] = resid[0]-(jac[0]*dr[0]+jac[1]*dr[1]+jac[2]*dr[2]);
      ress[1] = resid[1]-(jac[3]*dr[0]+jac[4]*dr[1]+jac[5]*dr[2]);
      ress[2] = resid[2]-(jac[6]*dr[0]+jac[7]*dr[1]+jac[8]*dr[2]);
      /* y = J_u^T res */
      y = jac[de]*ress[0]+jac[3+de]*ress[1]+jac[6+de]*ress[2];
      /* JtJ = J_u^T J_u */
      JtJ = jac[de]*jac[de]+jac[3+de]*jac[3+de] +
            jac[6+de]*jac[6+de];
      drc = y / JtJ;
      CHECK_CONSTRAINT(drc, de);
#undef CHECK_CONSTRAINT
      dr[de] += facc*drc;
      flags |= new_flags;
      goto newton_vol_relax;
   }

   /* check and possibly relax constraints */
newton_vol_relax :
   {
      const int old_flags = flags;
      double ress[3], y[3];
      /* res := res_0-J dr */
      ress[0] = resid[0]-(jac[0]*dr[0]+jac[1]*dr[1]+jac[2]*dr[2]);
      ress[1] = resid[1]-(jac[3]*dr[0]+jac[4]*dr[1]+jac[5]*dr[2]);
      ress[2] = resid[2]-(jac[6]*dr[0]+jac[7]*dr[1]+jac[8]*dr[2]);
      /* y := J^T res */
      y[0] = jac[0]*ress[0]+jac[3]*ress[1]+jac[6]*ress[2];
      y[1] = jac[1]*ress[0]+jac[4]*ress[1]+jac[7]*ress[2];
      y[2] = jac[2]*ress[0]+jac[5]*ress[1]+jac[8]*ress[2];
      for (int dd = 0; dd < 3; ++dd)
      {
         int f = flags >> (2*dd) & 3u;
         if (f)
         {
            dr[dd] = bnd[2*dd+(f-1)]-r0[dd];
            if (dr[dd]*y[dd] < 0)
            {
               flags &= ~(3u << (2*dd));
            }
         }
      }
      if (flags == old_flags)
      {
         goto newton_vol_fin;
      }
      switch (num_constrained(flags))
      {
         case 1:
            goto newton_vol_face;
         case 2:
            goto newton_vol_edge;
      }
   }

newton_vol_fin:
   flags &= mask;
   if (fabs(dr[0])+fabs(dr[1])+fabs(dr[2]) < tol)
   {
      flags |= CONVERGED_FLAG;
   }
   {
      const double res0 = resid[0]-(jac[0]*dr[0]+jac[1]*dr[1] +
                                    jac[2]*dr[2]);
      const double res1 = resid[1]-(jac[3]*dr[0]+jac[4]*dr[1] +
                                    jac[5]*dr[2]);
      const double res2 = resid[2]-(jac[6]*dr[0]+jac[7]*dr[1] +
                                    jac[8]*dr[2]);
      res->dist2p = resid[0]*resid[0]+resid[1]*resid[1] +
                    resid[2]*resid[2] -
                    (res0*res0+res1*res1+res2*res2);
   }
   for (int dd = 0; dd < 3; ++dd)
   {
      int f = flags >> (2*dd) & 3u;
      res->r[dd] = f == 0 ? r0[dd]+dr[dd] : (f == 1 ? -1 : 1);
   }
   res->flags = flags | (p->flags << 7);
}

// Full Newton solve on the face. One of r/s/t is constrained.
static MFEM_HOST_DEVICE void newton_face(findptsPt *const res,
                                         const double jac[9],
                                         const double rhes[3],
                                         const double resid[3],
                                         const int d1,
                                         const int d2,
                                         const int dn,
                                         const int flags,
                                         const findptsPt *const p,
                                         const double tol)
{
   const double tr = p->tr;
   double bnd[4];
   double r[2], dr[2] = {0, 0};
   int mask, new_flags;
   double v, tv;
   int i;
   double A[3], y[2], r0[2];
   /* A = J^T J-resid_d H_d */
   A[0] = jac[d1]*jac[d1]+jac[3+d1]*jac[3+d1] +
          jac[6+d1]*jac[6+d1]-rhes[0];
   A[1] = jac[d1]*jac[d2]+jac[3+d1]*jac[3+d2] +
          jac[6+d1]*jac[6+d2]-rhes[1];
   A[2] = jac[d2]*jac[d2]+jac[3+d2]*jac[3+d2] +
          jac[6+d2]*jac[6+d2]-rhes[2];
   /* y = J^T r */
   y[0] = jac[d1]*resid[0]+jac[3+d1]*resid[1]+jac[6+d1]*resid[2];
   y[1] = jac[d2]*resid[0]+jac[3+d2]*resid[1]+jac[6+d2]*resid[2];
   r0[0] = p->r[d1];
   r0[1] = p->r[d2];

   new_flags = flags;
   mask = 0x3fu;
   if (r0[0]-tr > -1)
   {
      bnd[0] = -tr;
      mask ^= 1u;
   }
   else
   {
      bnd[0] = -1-r0[0];
   }
   if (r0[0]+tr < 1)
   {
      bnd[1] = tr;
      mask ^= 2u;
   }
   else
   {
      bnd[1] = 1-r0[0];
   }
   if (r0[1]-tr > -1)
   {
      bnd[2] = -tr;
      mask ^= 1u << 2;
   }
   else
   {
      bnd[2] = -1-r0[1];
   }
   if (r0[1]+tr < 1)
   {
      bnd[3] = tr;
      mask ^= 2u << 2;
   }
   else
   {
      bnd[3] = 1-r0[1];
   }

   if (A[0]+A[2] <= 0 || A[0]*A[2] <= A[1]*A[1])
   {
      goto newton_face_constrained;
   }
   lin_solve_sym_2(dr, A, y);

#define EVAL(r, s) -(y[0]*r+y[1]*s)+(r*A[0]*r+(2*r*A[1]+s*A[2])*s) / 2
   if ((dr[0]-bnd[0])*(bnd[1]-dr[0]) >= 0 &&
       (dr[1]-bnd[2])*(bnd[3]-dr[1]) >= 0)
   {
      r[0] = r0[0]+dr[0], r[1] = r0[1]+dr[1];
      v = EVAL(dr[0], dr[1]);
      goto newton_face_fin;
   }
newton_face_constrained:
   v = EVAL(bnd[0], bnd[2]);
   i = 1u | (1u << 2);
   tv = EVAL(bnd[1], bnd[2]);
   if (tv < v)
   {
      v = tv;
      i = 2u | (1u << 2);
   }
   tv = EVAL(bnd[0], bnd[3]);
   if (tv < v)
   {
      v = tv;
      i = 1u | (2u << 2);
   }
   tv = EVAL(bnd[1], bnd[3]);
   if (tv < v)
   {
      v = tv;
      i = 2u | (2u << 2);
   }
   if (A[0] > 0)
   {
      double drc;
      drc = (y[0]-A[1]*bnd[2]) / A[0];
      if ((drc-bnd[0])*(bnd[1]-drc) >= 0 && (tv = EVAL(drc, bnd[2])) < v)
      {
         v = tv;
         i = 1u << 2;
         dr[0] = drc;
      }
      drc = (y[0]-A[1]*bnd[3]) / A[0];
      if ((drc-bnd[0])*(bnd[1]-drc) >= 0 && (tv = EVAL(drc, bnd[3])) < v)
      {
         v = tv;
         i = 2u << 2;
         dr[0] = drc;
      }
   }
   if (A[2] > 0)
   {
      double drc;
      drc = (y[1]-A[1]*bnd[0]) / A[2];
      if ((drc-bnd[2])*(bnd[3]-drc) >= 0 && (tv = EVAL(bnd[0], drc)) < v)
      {
         v = tv;
         i = 1u;
         dr[1] = drc;
      }
      drc = (y[1]-A[1]*bnd[1]) / A[2];
      if ((drc-bnd[2])*(bnd[3]-drc) >= 0 && (tv = EVAL(bnd[1], drc)) < v)
      {
         v = tv;
         i = 2u;
         dr[1] = drc;
      }
   }
#undef EVAL

   {
      int dir[2];
      dir[0] = d1;
      dir[1] = d2;
      for (int d = 0; d < 2; ++d)
      {
         const int f = i >> (2*d) & 3u;
         if (f == 0)
         {
            r[d] = r0[d]+dr[d];
         }
         else
         {
            if ((f & (mask >> (2*d))) == 0)
            {
               r[d] = r0[d]+(f == 1 ? -tr : tr);
            }
            else
            {
               r[d] = (f == 1 ? -1 : 1);
               new_flags |= f << (2*dir[d]);
            }
         }
      }
   }
newton_face_fin:
   res->dist2p = -2*v;
   dr[0] = r[0]-p->r[d1];
   dr[1] = r[1]-p->r[d2];
   if (fabs(dr[0])+fabs(dr[1]) < tol)
   {
      new_flags |= CONVERGED_FLAG;
   }
   res->r[dn] = p->r[dn];
   res->r[d1] = r[0];
   res->r[d2] = r[1];
   res->flags = new_flags | (p->flags << 7);
}

// Full Newton solve on the edge. Two of r/s/t are constrained.
static MFEM_HOST_DEVICE inline void newton_edge(findptsPt *const res,
                                                const double jac[9],
                                                const double rhes,
                                                const double resid[3],
                                                const int de,
                                                const int dn1,
                                                const int dn2,
                                                int flags,
                                                const findptsPt *const p,
                                                const double tol)
{
   const double tr = p->tr;
   /* A = J^T J-resid_d H_d */
   const double A = jac[de]*jac[de]+jac[3+de]*jac[3+de]+jac[6+de] *
                    jac[6+de]-rhes;
   /* y = J^T r */
   const double y = jac[de]*resid[0]+jac[3+de]*resid[1]+jac[6+de] *
                    resid[2];

   const double oldr = p->r[de];
   double dr, nr, tdr, tnr;
   double v, tv;
   int new_flags = 0, tnew_flags = 0;

#define EVAL(dr) (dr*A-2*y)*dr

   /* if A is not SPD, quadratic model has no minimum */
   if (A > 0)
   {
      dr = y / A;
      nr = oldr+dr;
      if (fabs(dr) < tr && fabs(nr) < 1)
      {
         v = EVAL(dr);
         goto newton_edge_fin;
      }
   }

   if ((nr = oldr-tr) > -1)
   {
      dr = -tr;
   }
   else
   {
      nr = -1;
      dr = -1-oldr;
      new_flags = flags | 1u << (2*de);
   }
   v = EVAL(dr);

   if ((tnr = oldr+tr) < 1)
   {
      tdr = tr;
   }
   else
   {
      tnr = 1;
      tdr = 1-oldr;
      tnew_flags = flags | 2u << (2*de);
   }
   tv = EVAL(tdr);

   if (tv < v)
   {
      nr = tnr;
      dr = tdr;
      v = tv;
      new_flags = tnew_flags;
   }

newton_edge_fin:
   /* check convergence */
   if (fabs(dr) < tol)
   {
      new_flags |= CONVERGED_FLAG;
   }
   res->r[de] = nr;
   res->r[dn1] = p->r[dn1];
   res->r[dn2] = p->r[dn2];
   res->dist2p = -v;
   res->flags = flags | new_flags | (p->flags << 7);
}

// Find closest mesh node to the sought point.
static MFEM_HOST_DEVICE void seed_j(const double *elx[3],
                                    const double x[3],
                                    const double *z,//GLL point locations [-1, 1]
                                    double *dist2,
                                    double *r[3],
                                    const int j,
                                    const int pN) // assumes j < pN
{
   dist2[j] = std::numeric_limits<double>::max();

   double zr = z[j];
   for (int l = 0; l < pN; ++l)
   {
      const double zt = z[l];
      for (int k = 0; k < pN; ++k)
      {
         double zs = z[k];

         const int jkl = j+k*pN+l*pN*pN;
         double dx[3];
         for (int d = 0; d < 3; ++d)
         {
            dx[d] = x[d]-elx[d][jkl];
         }
         const double dist2_jkl = l2norm2(dx);
         if (dist2[j] > dist2_jkl)
         {
            dist2[j] = dist2_jkl;
            r[0][j] = zr;
            r[1][j] = zs;
            r[2][j] = zt;
         }
      }
   }
}

/* Compute contribution towards function value and its derivatives in each
   reference direction. */
static MFEM_HOST_DEVICE double tensor_ig3_j(double *g_partials,
                                            const double *Jr,
                                            const double *Dr,
                                            const double *Js,
                                            const double *Ds,
                                            const double *Jt,
                                            const double *Dt,
                                            const double *u,
                                            const int j,
                                            const int pN)
{
   double uJtJs = 0.0;
   double uDtJs = 0.0;
   double uJtDs = 0.0;
   for (int k = 0; k < pN; ++k)
   {
      double uJt = 0.0;
      double uDt = 0.0;
      for (int l = 0; l < pN; ++l)
      {
         uJt += u[j+k*pN+l*pN*pN]*Jt[l];
         uDt += u[j+k*pN+l*pN*pN]*Dt[l];
      }

      uJtJs += uJt*Js[k];
      uJtDs += uJt*Ds[k];
      uDtJs += uDt*Js[k];
   }

   g_partials[0] = uJtJs*Dr[j];
   g_partials[1] = uJtDs*Jr[j];
   g_partials[2] = uDtJs*Jr[j];
   return uJtJs*Jr[j];
}

template<int T_D1D = 0>
static void FindPointsLocal3DKernel(const int npt,
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
   const int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
   const int D1D = T_D1D ? T_D1D : pN;
   const int p_NE = D1D*D1D*D1D;
   MFEM_VERIFY(MD1 <= DofQuadLimits::MAX_D1D,
               "Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D != 0, "Polynomial order not specified.");
#define MAXC(a, b) (((a) > (b)) ? (a) : (b))
   const int nThreads = MAXC(D1D*DIM, 15);

   mfem::forall_2D(npt, nThreads, 1, [=] MFEM_HOST_DEVICE (int i)
   {
      // 4 D1D for seed, 18D1D+12 for vol, 21D1D+15 for face, 3D1D+32 for edge.
      constexpr int size1 = 21*MD1+15;
      // 6D1D^2 for face, 15D1D for edge.
      constexpr int size2 = MAXC(MD1*MD1*6, MD1*3*5);
      //size depends on max of info for faces and edges
      constexpr int size3 = MD1*MD1*MD1*DIM;  // local element coordinates

      MFEM_SHARED double r_workspace[size1];
      MFEM_SHARED findptsPt el_pts[2];

      MFEM_SHARED double constraint_workspace[size2];
      MFEM_SHARED int face_edge_init;

      MFEM_SHARED double elem_coords[MD1 <= 6 ? size3 : 1];

      double *r_workspace_ptr;
      findptsPt *fpt, *tmp;
      MFEM_FOREACH_THREAD(j,x,nThreads)
      {
         r_workspace_ptr = r_workspace;
         fpt = el_pts+0;
         tmp = el_pts+1;
      }
      MFEM_SYNC_THREAD;

      int id_x = point_pos_ordering == 0 ? i : i*DIM;
      int id_y = point_pos_ordering == 0 ? i+npt : i*DIM+1;
      int id_z = point_pos_ordering == 0 ? i+2*npt : i*DIM+2;
      double x_i[3] = {x[id_x], x[id_y], x[id_z]};

      unsigned int *code_i = code_base+i;
      double *dist2_i = dist2_base+i;

      //// map_points_to_els ////
      findptsLocalHashData_t hash;
      for (int d = 0; d < DIM; ++d)
      {
         hash.bnd[d].min = hashMin[d];
         hash.fac[d] = hashFac[d];
      }
      hash.hash_n = hash_n;
      hash.offset = hashOffset;
      const unsigned int hi = hash_index(&hash, x_i);
      const unsigned int *elp = hash.offset+hash.offset[hi],
                          *const ele = hash.offset+hash.offset[hi+1];
      *code_i = CODE_NOT_FOUND;
      *dist2_i = std::numeric_limits<double>::max();

      for (; elp != ele; ++elp)
      {
         //elp

         const int el = *elp;

         // construct obbox_t on the fly from data
         obbox_t box;
         int n_box_ents = 3*DIM+DIM2;
         for (int idx = 0; idx < DIM; ++idx)
         {
            box.c0[idx] = boxinfo[n_box_ents*el+idx];
            box.x[idx].min = boxinfo[n_box_ents*el+DIM+idx];
            box.x[idx].max = boxinfo[n_box_ents*el+2*DIM+idx];
         }

         for (int idx = 0; idx < DIM2; ++idx)
         {
            box.A[idx] = boxinfo[n_box_ents*el+3*DIM+idx];
         }

         if (bbox_test(&box, x_i) < 0) { continue; }

         //// findpts_local ////
         {
            // read element coordinates into shared memory
            if (MD1 <= 6)
            {
               MFEM_FOREACH_THREAD(j,x,D1D*DIM)
               {
                  const int qp = j % D1D;
                  const int d = j / D1D;
                  for (int l = 0; l < D1D; ++l)
                  {
                     for (int k = 0; k < D1D; ++k)
                     {
                        const int jkl = qp+k*D1D+l*D1D*D1D;
                        elem_coords[jkl+d*p_NE] =
                           xElemCoord[jkl+el*p_NE+d*nel*p_NE];
                     }
                  }
               }
               MFEM_SYNC_THREAD;
            }

            const double *elx[DIM];
            for (int d = 0; d < DIM; d++)
            {
               elx[d] = MD1<= 6 ? &elem_coords[d*p_NE] :
                        xElemCoord+d*nel*p_NE+el*p_NE;
            }

            //// findpts_el ////
            {
               MFEM_SYNC_THREAD;
               MFEM_FOREACH_THREAD(j,x,1)
               {
                  fpt->dist2 = std::numeric_limits<double>::max();
                  fpt->dist2p = 0;
                  fpt->tr = 1;
                  face_edge_init = 0;
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
                     r_temp[d] = dist2_temp+(1+d)*D1D;
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
                  tmp->flags = 0; // we do newton_vol regardless of seed.
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
                     case 0: // findpt_vol
                     {
                        double *wtr = r_workspace_ptr;

                        double *resid = wtr+6*D1D;
                        double *jac = resid+3;
                        double *resid_temp = jac+9;
                        double *jac_temp = resid_temp+3*D1D;

                        MFEM_FOREACH_THREAD(j,x,D1D*DIM)
                        {
                           const int qp = j % D1D;
                           const int d = j / D1D;
                           lag_eval_first_der(wtr+2*d*D1D, tmp->r[d], qp,
                                              gll1D, lagcoeff, D1D);
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(j,x,D1D*DIM)
                        {
                           const int qp = j % D1D;
                           const int d = j / D1D;
                           double *idx = jac_temp+3*d+9*qp;
                           resid_temp[d+qp*3] = tensor_ig3_j(idx,
                                                             wtr,
                                                             wtr+D1D,
                                                             wtr+2*D1D,
                                                             wtr+3*D1D,
                                                             wtr+4*D1D,
                                                             wtr+5*D1D,
                                                             elx[d],
                                                             qp, D1D);
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(l,x,3)
                        {
                           resid[l] = tmp->x[l];
                           for (int j = 0; j < D1D; ++j)
                           {
                              resid[l] -= resid_temp[l+j*3];
                           }
                        }
                        MFEM_FOREACH_THREAD(l,x,9)
                        {
                           jac[l] = 0;
                           for (int j = 0; j < D1D; ++j)
                           {
                              jac[l] += jac_temp[l+j*9];
                           }
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(l,x,1)
                        {
                           // if (l == 0)
                           {
                              if (!reject_prior_step_q(fpt, resid, tmp, tol))
                              {
                                 newton_vol(fpt, jac, resid, tmp, tol);
                              }
                           }
                        }
                        MFEM_SYNC_THREAD;
                        break;
                     } //case 0
                     case 1: // findpt_face
                     {
                        const int fi = face_index(tmp->flags & FLAG_MASK);
                        const int dn = fi >> 1;
                        const int d1 = plus_1_mod_3(dn), d2 = plus_2_mod_3(dn);

                        double *wt1 = r_workspace_ptr;
                        double *resid = wt1+6*D1D;
                        double *jac = resid+3;
                        double *resid_temp = jac+9;
                        double *jac_temp = resid_temp+3*D1D;
                        double *hes = jac_temp+9*D1D;
                        double *hes_temp = hes+3;
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(j,x,D1D*2)
                        {
                           int dd[2];
                           dd[0] = d1;
                           dd[1] = d2;
                           const int d = j / D1D;
                           const int qp = j % D1D;
                           lag_eval_second_der(wt1+3*d*D1D, tmp->r[dd[d]],
                                               qp, gll1D, lagcoeff, D1D);
                        }
                        MFEM_SYNC_THREAD;

                        double *J1 = wt1, *D1 = wt1+D1D;
                        double *J2 = wt1+3*D1D, *D2 = J2+D1D;
                        double *DD1 = D1+D1D, *DD2 = D2+D1D;
                        findptsElemFace face;

                        MFEM_FOREACH_THREAD(j,x,D1D*DIM)
                        {
                           // utilizes first 3*D1D threads
                           face = get_face(elx, wtend, fi, constraint_workspace,
                                           face_edge_init, j, D1D);
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(j,x,D1D*DIM)
                        {
                           if (j == 0) { face_edge_init = (1 << fi); }
                           const int qp = j % D1D;
                           const int d = j / D1D;
                           const double *u = face.x[d];
                           const double *du = face.dxdn[d];
                           double sums_k[4] = {0.0, 0.0, 0.0, 0.0};
                           for (int k = 0; k < D1D; ++k)
                           {
                              sums_k[0] += u[qp+k*D1D]*J2[k];
                              sums_k[1] += u[qp+k*D1D]*D2[k];
                              sums_k[2] += u[qp+k*D1D]*DD2[k];
                              sums_k[3] += du[qp+k*D1D]*J2[k];
                           }

                           resid_temp[3*qp+d] = sums_k[0]*J1[qp];
                           jac_temp[9*qp+3*d+d1] = sums_k[0]*D1[qp];
                           jac_temp[9*qp+3*d+d2] = sums_k[1]*J1[qp];
                           jac_temp[9*qp+3*d+dn] = sums_k[3]*J1[qp];
                           if (d == 0)
                           {
                              hes_temp[3*qp] = sums_k[0]*DD1[qp];
                              hes_temp[3*qp+1] = sums_k[1]*D1[qp];
                              hes_temp[3*qp+2] = sums_k[2]*J1[qp];
                           }
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(l,x,3)
                        {
                           resid[l] = fpt->x[l];
                           hes[l] = 0;
                           for (int j = 0; j < D1D; ++j)
                           {
                              resid[l] -= resid_temp[l+j*3];
                              hes[l] += hes_temp[l+3*j];
                           }
                           hes[l] *= resid[l];
                        }

                        MFEM_FOREACH_THREAD(l,x,9)
                        {
                           jac[l] = 0;
                           for (int j = 0; j < D1D; ++j)
                           {
                              jac[l] += jac_temp[l+j*9];
                           }
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(l,x,1)
                        {
                           if (!reject_prior_step_q(fpt, resid, tmp, tol))
                           {
                              const double steep = resid[0]*jac[dn]+
                                                   resid[1]*jac[3+dn]+
                                                   resid[2]*jac[6+dn];
                              if (steep*tmp->r[dn] < 0)
                              {
                                 // relax constraint //
                                 newton_vol(fpt, jac, resid, tmp, tol);
                              }
                              else
                              {
                                 newton_face(fpt, jac, hes, resid, d1,
                                             d2, dn, tmp->flags&FLAG_MASK,
                                             tmp, tol);
                              }
                           }
                        }
                        MFEM_SYNC_THREAD;
                        break;
                     }
                     case 2: // findpt_edge
                     {
                        const int ei = edge_index(tmp->flags & FLAG_MASK);
                        const int de = ei >> 2,
                                  dn1 = plus_1_mod_3(de),
                                  dn2 = plus_2_mod_3(de);
                        int d_j[3];
                        d_j[0] = de;
                        d_j[1] = dn1;
                        d_j[2] = dn2;
                        const int hes_count = 2*3-1;

                        double *wt = r_workspace_ptr;
                        double *resid = wt+3*D1D;
                        double *jac = resid+3;
                        double *hes_T = jac+9;
                        double *hes = hes_T+hes_count*3;
                        findptsElemEdge edge;

                        MFEM_FOREACH_THREAD(j,x,nThreads)
                        {
                           // utilizes first 3*D1D threads f
                           edge = get_edge(elx, wtend, ei,
                                           constraint_workspace,
                                           face_edge_init, j,
                                           D1D);
                        }
                        MFEM_SYNC_THREAD;

                        const double *const *e_x[3+3] = {edge.x, edge.x,
                                                         edge.dxdn1,
                                                         edge.dxdn2,
                                                         edge.d2xdn1,
                                                         edge.d2xdn2
                                                        };

                        MFEM_FOREACH_THREAD(j,x,D1D)
                        {
                           if (j == 0) { face_edge_init = (64 << ei); }
                           lag_eval_second_der(wt, tmp->r[de], j, gll1D,
                                               lagcoeff, D1D);
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(j,x,hes_count*3)
                        {
                           const int d = j % 3;
                           const int row = j / 3;
                           if (j < (3+1)*3)
                           {
                              // resid and jac_T
                              // [0, 1, 0, 0]
                              double *wt_j = wt+(row == 1 ? D1D : 0);
                              const double *x = e_x[row][d];
                              double sum = 0.0;
                              for (int k = 0; k < D1D; ++k)
                              {
                                 // resid+3 == jac_T
                                 sum += wt_j[k]*x[k];
                              }
                              if (j < 3)
                              {
                                 resid[j] = tmp->x[j]-sum;
                              }
                              else
                              {
                                 jac[d*3+d_j[row-1]] = sum;
                              }
                           }

                           {
                              // Hes_T is transposed version (i.e. in col major)
                              // n1*[2, 1, 1, 0, 0]
                              // j==1 => wt_j = wt+n1
                              double *wt_j = wt+D1D*(2-(row+1) / 2);
                              const double *x = e_x[row+1][d];
                              hes_T[j] = 0.0;
                              for (int k = 0; k < D1D; ++k)
                              {
                                 hes_T[j] += wt_j[k]*x[k];
                              }
                           }
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(j,x,hes_count)
                        {
                           hes[j] = 0.0;
                           for (int d = 0; d < 3; ++d)
                           {
                              hes[j] += resid[d]*hes_T[j*3+d];
                           }
                        }

                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(l,x,1)
                        {
                           // check prior step //
                           if (!reject_prior_step_q(fpt, resid, tmp, tol))
                           {
                              // check constraint //
                              double steep[3-1];
                              for (int k = 0; k < 3-1; ++k)
                              {
                                 int dn = d_j[k+1];
                                 steep[k] = 0;
                                 for (int d = 0; d < 3; ++d)
                                 {
                                    steep[k] += jac[dn+d*3]*resid[d];
                                 }
                                 steep[k] *= tmp->r[dn];
                              }
                              if (steep[0] < 0)
                              {
                                 if (steep[1] < 0)
                                 {
                                    newton_vol(fpt, jac, resid, tmp, tol);
                                 }
                                 else
                                 {
                                    double rh[3];
                                    rh[0] = hes[0];
                                    rh[1] = hes[1];
                                    rh[2] = hes[3];
                                    newton_face(fpt, jac, rh, resid, de,
                                                dn1, dn2,
                                                tmp->flags&(3u<<(dn2*2)),
                                                tmp, tol);
                                 }
                              }
                              else
                              {
                                 if (steep[1] < 0)
                                 {
                                    double rh[3];
                                    rh[0] = hes[4];
                                    rh[1] = hes[2];
                                    rh[2] = hes[0];
                                    newton_face(fpt, jac, rh, resid, dn2,
                                                de, dn1,
                                                tmp->flags&(3u<<(dn1*2)),
                                                tmp, tol);
                                 }
                                 else
                                 {
                                    newton_edge(fpt, jac, hes[0], resid,
                                                de, dn1, dn2,
                                                tmp->flags & FLAG_MASK,
                                                tmp, tol);
                                 }
                              }
                           }
                        }
                        MFEM_SYNC_THREAD;
                        break;
                     }
                     case 3:   // findpts_pt
                     {
                        MFEM_FOREACH_THREAD(j,x,1)
                        {
                           // if (j == 0)
                           const int pi=point_index(tmp->flags & FLAG_MASK);
                           const findptsElemPt gpt=get_pt(elx,wtend,pi,D1D);
                           const double *const pt_x = gpt.x;
                           const double *const jac = gpt.jac;
                           const double *const hes = gpt.hes;

                           double resid[3], steep[3];
                           for (int d = 0; d < 3; ++d)
                           {
                              resid[d] = fpt->x[d]-pt_x[d];
                           }
                           if (!reject_prior_step_q(fpt, resid, tmp, tol))
                           {
                              for (int d = 0; d < 3; ++d)
                              {
                                 steep[d] = 0;
                                 for (int e = 0; e < 3; ++e)
                                 {
                                    steep[d] += jac[d+e*3]*resid[e];
                                 }
                                 steep[d] *= tmp->r[d];
                              }
                              int de, dn1, dn2, d1, d2, dn, hi0, hi1, hi2;
                              if (steep[0] < 0)
                              {
                                 if (steep[1] < 0)
                                 {
                                    if (steep[2] < 0)
                                    {
                                       newton_vol(fpt,jac,resid,tmp,tol);
                                    }
                                    else
                                    {
                                       d1 = 0; d2 = 1; dn = 2; hi0 = 0;
                                       hi1 = 1, hi2 = 3;
                                       double rh[3];
                                       rh[0] = resid[0]*hes[hi0] +
                                               resid[1]*hes[6+hi0] +
                                               resid[2]*hes[12+hi0];
                                       rh[1] = resid[0]*hes[hi1] +
                                               resid[1]*hes[6+hi1] +
                                               resid[2]*hes[12+hi1];
                                       rh[2] = resid[0]*hes[hi2] +
                                               resid[1]*hes[6+hi2] +
                                               resid[2]*hes[12+hi2];
                                       newton_face(fpt, jac, rh, resid,
                                                   d1, d2, dn,
                                                   (tmp->flags)&(3u<<(2*dn)),
                                                   tmp, tol);
                                    }
                                 }
                                 else
                                 {
                                    if (steep[2] < 0)
                                    {
                                       d1 = 2; d2 = 0; dn = 1; hi0 = 5;
                                       hi1 = 2, hi2 = 0;
                                       double rh[3];
                                       rh[0] = resid[0]*hes[hi0] +
                                               resid[1]*hes[6+hi0] +
                                               resid[2]*hes[12+hi0];
                                       rh[1] = resid[0]*hes[hi1] +
                                               resid[1]*hes[6+hi1] +
                                               resid[2]*hes[12+hi1];
                                       rh[2] = resid[0]*hes[hi2] +
                                               resid[1]*hes[6+hi2] +
                                               resid[2]*hes[12+hi2];
                                       newton_face(fpt, jac, rh, resid,
                                                   d1, d2, dn,
                                                   (tmp->flags)&(3u<<(2*dn)),
                                                   tmp, tol);
                                    }
                                    else
                                    {
                                       de = 0, dn1 = 1, dn2 = 2, hi0 = 0;
                                       const double rh =
                                          resid[0]*hes[hi0] +
                                          resid[1]*hes[6+hi0] +
                                          resid[2]*hes[12+hi0];
                                       newton_edge(fpt, jac, rh, resid,
                                                   de, dn1, dn2,
                                                   tmp->flags&(~(3u<<(2*de))),
                                                   tmp, tol);
                                    }
                                 }
                              }
                              else
                              {
                                 if (steep[1] < 0)
                                 {
                                    if (steep[2] < 0)
                                    {
                                       d1 = 1, d2 = 2, dn = 0;
                                       hi0 = 3, hi1 = 4, hi2 = 5;
                                       double rh[3];
                                       rh[0] = resid[0]*hes[hi0] +
                                               resid[1]*hes[6+hi0] +
                                               resid[2]*hes[12+hi0];
                                       rh[1] = resid[0]*hes[hi1] +
                                               resid[1]*hes[6+hi1] +
                                               resid[2]*hes[12+hi1];
                                       rh[2] = resid[0]*hes[hi2] +
                                               resid[1]*hes[6+hi2] +
                                               resid[2]*hes[12+hi2];
                                       newton_face(fpt, jac, rh, resid,
                                                   d1, d2, dn,
                                                   (tmp->flags)&(3u<<(2*dn)),
                                                   tmp, tol);
                                    }
                                    else
                                    {
                                       de = 1, dn1 = 2, dn2 = 0, hi0 = 3;
                                       const double rh =
                                          resid[0]*hes[hi0] +
                                          resid[1]*hes[6+hi0] +
                                          resid[2]*hes[12+hi0];
                                       newton_edge(fpt, jac, rh, resid,
                                                   de, dn1, dn2,
                                                   tmp->flags&(~(3u<<(2*de))),
                                                   tmp, tol);
                                    }
                                 }
                                 else
                                 {
                                    if (steep[2] < 0)
                                    {
                                       de = 2, dn1 = 0, dn2 = 1, hi0 = 5;
                                       const double rh =
                                          resid[0]*hes[hi0] +
                                          resid[1]*hes[6+hi0] +
                                          resid[2]*hes[12+hi0];
                                       newton_edge(fpt, jac, rh, resid,
                                                   de, dn1, dn2,
                                                   tmp->flags&(~(3u<<(2*de))),
                                                   tmp, tol);
                                    }
                                    else
                                    {
                                       fpt->r[0] = tmp->r[0];
                                       fpt->r[1] = tmp->r[1];
                                       fpt->r[2] = tmp->r[2];
                                       fpt->dist2p = 0;
                                       fpt->flags =tmp->flags|CONVERGED_FLAG;
                                    }
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
                  *(el_base+i) = el;
                  *code_i = converged_internal ?  CODE_INTERNAL :
                            CODE_BORDER;
                  *dist2_i = fpt->dist2;
               }
               MFEM_FOREACH_THREAD(j,x,DIM)
               {
                  *(r_base+DIM*i+j) = fpt->r[j];
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

void FindPointsGSLIB::FindPointsLocal3(const Vector &point_pos,
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
      case 2: FindPointsLocal3DKernel<2>(npt, DEV.newt_tol,
                                            pp, point_pos_ordering,
                                            pgslm, NE_split_total,
                                            pwt, pbb,
                                            DEV.h_nx, plhm, plhf, plho,
                                            pcode, pelem, pref, pdist,
                                            pgll1d, plc);
         break;
      case 3: FindPointsLocal3DKernel<3>(npt, DEV.newt_tol,
                                            pp, point_pos_ordering,
                                            pgslm, NE_split_total,
                                            pwt, pbb,
                                            DEV.h_nx, plhm, plhf, plho,
                                            pcode, pelem, pref, pdist,
                                            pgll1d, plc);
         break;
      case 4: FindPointsLocal3DKernel<4>(npt, DEV.newt_tol,
                                            pp, point_pos_ordering,
                                            pgslm, NE_split_total,
                                            pwt, pbb,
                                            DEV.h_nx, plhm, plhf, plho,
                                            pcode, pelem, pref, pdist,
                                            pgll1d, plc);
         break;
      case 5: FindPointsLocal3DKernel<5>(npt, DEV.newt_tol,
                                            pp, point_pos_ordering,
                                            pgslm, NE_split_total,
                                            pwt, pbb,
                                            DEV.h_nx, plhm, plhf, plho,
                                            pcode, pelem, pref, pdist,
                                            pgll1d, plc);
         break;
      default: FindPointsLocal3DKernel(npt, DEV.newt_tol,
                                          pp, point_pos_ordering,
                                          pgslm, NE_split_total,
                                          pwt, pbb,
                                          DEV.h_nx, plhm, plhf, plho,
                                          pcode, pelem, pref, pdist,
                                          pgll1d, plc, DEV.dof1d);
   }
}
#undef pMax
#undef DIM2
#undef DIM
#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND
#else
void FindPointsGSLIB::FindPointsLocal3(const Vector &point_pos,
                                       int point_pos_ordering,
                                       Array<unsigned int> &code,
                                       Array<unsigned int> &elem,
                                       Vector &ref,
                                       Vector &dist,
                                       int npt) {};
#endif
} // namespace mfem
#endif //ifdef MFEM_USE_GSLIB
