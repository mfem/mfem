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

#include <climits>

namespace mfem
{
#if GSLIB_RELEASE_VERSION >= 10009
#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2
#define sDIM 3
#define rDIM 2
#define sDIM2 9

struct findptsElementPoint_t
{
   double x[sDIM], r[rDIM], oldr[rDIM], dist2, dist2p, tr;
   int flags;
};

struct findptsElementGEdge_t
{
   double *x[sDIM], *dxdn[sDIM], *d2xdn[sDIM];
};

struct findptsElementGPT_t
{
   double x[sDIM], jac[sDIM*rDIM], hes[sDIM*(rDIM+1)];
};

struct dbl_range_t
{
   double min, max;
};

struct obbox_t
{
   double c0[sDIM], A[sDIM*sDIM];
   dbl_range_t x[sDIM];
};

struct findptsLocalHashData_t
{
   int hash_n;
   dbl_range_t bnd[sDIM];
   double fac[sDIM];
   unsigned int *offset;
};

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

/* positive when possibly inside */
static MFEM_HOST_DEVICE inline double AABB_test(const obbox_t *const b,
                                                const double x[sDIM])
{
   double b_d;
   for (int d=0; d<sDIM; ++d)
   {
      b_d = (x[d] - b->x[d].min) * (b->x[d].max - x[d]);
      if (b_d < 0)   // if outside in any dimension
      {
         return b_d;
      }
   }
   return b_d;       // only positive if inside in all dimensions
}

/* positive when possibly inside */
static MFEM_HOST_DEVICE inline double bbox_test(const obbox_t *const b,
                                                 const double x[sDIM])
{
   const double bxyz = AABB_test(b, x);
   if (bxyz<0)
   {
      return bxyz;
   }
   else
   {
      double dxyz[3];
      // dxyz: distance of the point from the center of the OBB
      for (int d=0; d<sDIM; ++d)
      {
         dxyz[d] = x[d] - b->c0[d];
      }
      // tranform dxyz to the local coordinate system of the OBB,
      // and check if the point is inside the OBB [-1,1]^sDIM
      double test = 1;
      for (int d=0; d<sDIM; ++d)
      {
         double rst = 0;
         for (int e=0; e<sDIM; ++e)
         {
            rst += b->A[d*sDIM + e] * dxyz[e];
         }
         double brst = (rst+1)*(1-rst);
         test = test<0 ? test : brst;
      }
      return test;
   }
}

/* Hash index in the hash table to the elements that possibly contain the point x */
static MFEM_HOST_DEVICE inline int hash_index(const findptsLocalHashData_t *p,
                                              const double x[sDIM])
{
   const int n = p->hash_n;
   int sum = 0;
   for (int d=sDIM-1; d>=0; --d)
   {
      sum *= n;
      int i = (int)floor((x[d] - p->bnd[d].min) * p->fac[d]);
      sum += i<0 ? 0 : (n-1 < i ? n-1 : i);
   }
   return sum;
}

static MFEM_HOST_DEVICE inline void lin_solve_sym_2(double x[2],
                                                    const double A[3],
                                                    const double y[2])
{
   const double idet = 1 / (A[0] * A[2] - A[1] * A[1]);
   x[0] = idet * (A[2] * y[0] - A[1] * y[1]);
   x[1] = idet * (A[0] * y[1] - A[1] * y[0]);
}

static MFEM_HOST_DEVICE inline double l2norm2(const double x[sDIM])
{
   return ( x[0]*x[0] + x[1]*x[1] + x[2]*x[2] );
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
#define FLAG_MASK 0x1fu

/* returns the number of constrained reference coordinates, max 2
*/
static MFEM_HOST_DEVICE inline int num_constrained(const int flags)
{
   const int y = (flags | flags>>1);
   return (y & 1u) + (y>>2 & 1u);
}

/* returns (x+1)%2
 */
static MFEM_HOST_DEVICE inline int plus_1_mod_2(const int x)
{
   return x^1u;
}

/* assumes x = 1<<i, with i<4, returns i+1
 * Gives index of the first bit set in x
 */
static MFEM_HOST_DEVICE inline int which_bit(const int x)
{
   const int y = x & 7u;
   return (y-(y>>2)) | ((x-1)&4u);
}

/* Returns an index representing the edge:
 * 0 for rmin, 1 for rmax, 2 for smin, 3 for smax
 */
static MFEM_HOST_DEVICE inline int edge_index(const int x)
{
   return which_bit(x) - 1;
}

static MFEM_HOST_DEVICE inline int point_index(const int x)
{
   return ((x>>1)&1u) | ((x>>2)&2u);
}

/* Compute (x,y) and (dxdn, dydn) data for all DOFs along the edge based on
 * edge index. ei=0..3 corresponding to rmin, rmax, smin, smax.
 */
static MFEM_HOST_DEVICE inline findptsElementGEdge_t
get_edge(const double *elx[3], const double *wtend, int ei,
               double *workspace, int &side_init, int j, int pN)
{
   findptsElementGEdge_t edge;

   if (j >= pN) { return edge; }

   const int jidx = ei >= 2 ? j : ei*(pN-1);
   const int kidx = ei >= 2 ? (ei-2)*(pN-1) : j;

   // location of derivatives based on whether we want at r/s=-1 or r/s=+1
   // ei == 0 and 2 are constrained at -1, 1 and 3 are constrained at +1.
   const double *wt1 = wtend + (ei%2==0 ? 0 : 1)* pN * 3 + pN;

   for (int d=0; d<sDIM; ++d)
   {
      edge.x[d]     = workspace             + d*pN; //x,y,z
      edge.dxdn[d]  = workspace +   sDIM*pN + d*pN; //dxdn,dydn,dzdn
      edge.d2xdn[d] = workspace + 2*sDIM*pN + d*pN; //d2xdn2,d2ydn2,d2zdn2
   }

   if (side_init != (1u << ei))
   {
#define ELX(d, j, k) elx[d][j + k * pN] // assumes lexicographic ordering
      for (int d = 0; d < sDIM; ++d)
      {
         // copy nodal coordinates along the constrained edge
         edge.x[d][j] = ELX(d, jidx, kidx);

         // compute derivatives in normal direction.
         double sums_k = 0.0,
                sums2_k = 0.0;
         for (int k = 0; k < pN; ++k)
         {
            if (ei >= 2)
            {
               sums_k += wt1[k] * ELX(d, j, k);
               sums2_k += wt1[k + pN] * ELX(d, j, k);
            }
            else
            {
               sums_k += wt1[k] * ELX(d, k, j);
               sums2_k += wt1[k + pN] * ELX(d, k, j);
            }
         }
         edge.dxdn[d][j] = sums_k;
         edge.d2xdn[d][j] = sums2_k;
      }
   }
#undef ELX
   return edge;
}

static MFEM_HOST_DEVICE inline findptsElementGEdge_t
get_edge2(const double *elx[3], const double *wtend, int ei,
                 double *workspace, int &side_init, int jidx, int pN)
{
   findptsElementGEdge_t edge;
   for (int d=0; d<sDIM; ++d)
   {
      edge.x[d]     = workspace             + d*pN;
      edge.dxdn[d]  = workspace +   sDIM*pN + d*pN;
      edge.d2xdn[d] = workspace + 2*sDIM*pN + d*pN;
   }

   // given edge index, compute normal and tangential directions
   const int dn = ei>>1, //0 for rmin/rmax, 1 for smin/smax
             de = plus_1_mod_2(dn); // 1 for rmin/rmax, 0 for smin/smax
   const int side_n        = ei&1,          // 0 for rmin/smin, 1 for rmax/smax
             side_n_offset = side_n*(pN-1); // 0 from rmin/smin, pN-1 for rmax/smax
   const double *wt1       = wtend + 3*pN*side_n;

   const int jj = jidx%pN;
   const int dd = jidx/pN;
   if (side_init != (1u << ei))
   {
         const int elx_stride[2] = {1,pN};
#define ELX(d,j,k) elx[d][j*elx_stride[de] + k*elx_stride[dn]]

         edge.x[dd][jj] = ELX(dd, jj, side_n_offset);
         double sums_k[2] = {0,0};
         for (int k=0; k<pN; ++k)
         {
            sums_k[0] += wt1[pN+k]   * ELX(dd,jj,k);
            sums_k[1] += wt1[2*pN+k] * ELX(dd,jj,k);
         }
         edge.dxdn[dd][jj]  = sums_k[0];
         edge.d2xdn[dd][jj] = sums_k[1];
#undef ELX
   }
   return edge;
}

static MFEM_HOST_DEVICE inline findptsElementGEdge_t
get_edge_old(const double *elx[3], const double *wtend, int ei,
         double *workspace, int &side_init, int jidx, int pN)
{
   findptsElementGEdge_t edge;
   for (int d=0; d<sDIM; ++d)
   {
      edge.x[d]     = workspace             + d*pN;
      edge.dxdn[d]  = workspace +   sDIM*pN + d*pN;
      edge.d2xdn[d] = workspace + 2*sDIM*pN + d*pN;
   }

   // given edge index, compute normal and tangential directions
   const int dn = ei>>1, //0 for rmin/rmax, 1 for smin/smax
             de = plus_1_mod_2(dn); // 1 for rmin/rmax, 0 for smin/smax
   const int side_n        = ei&1,          // 0 for rmin/smin, 1 for rmax/smax
             side_n_offset = side_n*(pN-1); // 0 from rmin/smin, pN-1 for rmax/smax
   const double *wt1       = wtend + 3*pN*side_n;
   // First 3*pN entries in wtend for rmin/smin, next 3*pN entries for rmax/smax.
   // stores lagfunc and its 1st and 2nd derivatives values

   const int jj = jidx%pN;
   const int dd = jidx/pN;
   if (side_init != (1u << ei))
   {
      /* As j runs from 0 to 3*pN - 1
         jj = 0,1,...,pN-1, 0,1,...,pN-1, 0,1,......,pN-1
         dd = 0,0,......,0, 1,1,......,1, 1,1,......,1
         Essentially, first everything for dd=0 is computed, then for dd=1, and
         so on. dd here means the spatial dimension and is used to access
         corresponding memory in elx, x, dxdn.
      */
      if (jidx<3*pN)
      {
         // If de=0, then nodes along the edge follow contiguously and hence are strided by 1.
         // If de=1, then nodes along the edge are separated by pN nodes, and hence are strided by pN.
         const int elx_stride[2] = {1,pN};

#define ELX(d,j,k) elx[d][j*elx_stride[de] + k*elx_stride[dn]]

         edge.x[dd][jj] = ELX(dd, jj, side_n_offset);

         double sums_k[2] = {0,0};

         for (int k=0; k<pN; ++k)
         {
            sums_k[0] += wt1[pN+k]   * ELX(dd,jj,k);
            sums_k[1] += wt1[2*pN+k] * ELX(dd,jj,k);
         }
         edge.dxdn[dd][jj]  = sums_k[0];
         edge.d2xdn[dd][jj] = sums_k[1];
#undef ELX
      }
      // side_init = (1u << ei);
   }
   return edge;
}

static MFEM_HOST_DEVICE inline findptsElementGPT_t get_pt(const double *elx[3],
                                                          const double *wtend,
                                                          int pi,
                                                          int pN)
{
   const int side_n1    = pi&1,
             side_n2    = (pi>>1)&1;
   const int in1        = side_n1*(pN-1),
             in2        = side_n2*(pN-1);
   const int hes_stride = rDIM + 1;  // rDIM + C^rDIM_2

   findptsElementGPT_t pt;

#define ELX(d,j,k) elx[d][j + k*pN]
   for (int d=0; d<sDIM; ++d)
   {
      pt.x[d] = ELX(d,in1,in2);

      // point to the start of 1st derivatives corresponding to whether r/s is constrained at -1 or 1.
      const double *wt1 = wtend + pN + side_n1*3*pN;
      const double *wt2 = wtend + pN + side_n2*3*pN;

      for (int i=0; i<rDIM; ++i)
      {
         pt.jac[rDIM*d + i] = 0;
      }
      for (int i=0; i<hes_stride; ++i)
      {
         pt.hes[hes_stride*d + i] = 0;
      }

      for (int j=0; j<pN; ++j)
      {
         pt.jac[rDIM*d+0] += wt1[j] * ELX(d,j,in2);
         pt.jac[rDIM*d+1] += wt2[j] * ELX(d,in1,j);

         double sum_k = 0;
         for (int k=0; k<pN; ++k)
         {
            sum_k += wt1[k] * ELX(d,k,j);
         }
         pt.hes[hes_stride*d+0] += wt1[pN+j] * ELX(d,j,in2);
         pt.hes[hes_stride*d+1] += wt2[j]    * sum_k;
         pt.hes[hes_stride*d+2] += wt2[pN+j] * ELX(d,in1,j);
      }
#undef ELX
   }
   return pt;
}

/* check reduction in objective against prediction, and adjust
   trust region radius (p->tr) accordingly;
   may reject the prior step, returning 1; otherwise returns 0
   sets out->dist2, out->index, out->x, out->oldr in any event,
   leaving out->r, out->dr, out->flags to be set when returning 0 */
static MFEM_HOST_DEVICE bool reject_prior_step_q(findptsElementPoint_t *out,
                                                 const double resid[3],
                                                 const findptsElementPoint_t *p,
                                                 const double tol)
{
   const double dist2 = l2norm2(resid);
   const double decr  = p->dist2 - dist2;
   const double pred  = p->dist2p;
   for (int d=0; d<sDIM; ++d)
   {
      out->x[d] = p->x[d];
   }
   for (int d=0; d<rDIM; ++d)
   {
      out->oldr[d] = p->r[d];
   }
   out->dist2 = dist2;
   if (decr>=0.01*pred)
   {
      if (decr>=0.9*pred)   // very good iteration
      {
         out->tr = 2*p->tr;
      }
      else   // good iteration
      {
         out->tr = p->tr;
      }
      return false;
   }
   else   // if the iteration in not good
   {
      /* reject step; note: the point will pass through this routine
         again, and we set things up here so it gets classed as a
         "very good iteration" --- this doubles the trust radius,
         which is why we divide by 4 below */
      double v0 = fabs(p->r[0] - p->oldr[0]),
             v1 = fabs(p->r[1] - p->oldr[1]);
      out->tr     = ( v0>v1 ? v0 : v1 )/4;
      out->dist2  = p->dist2;
      out->flags   = p->flags >> 5;
      out->dist2p = -DBL_MAX;
      for (int d=0; d<rDIM; ++d)
      {
         out->r[d] = p->oldr[d];
      }
      if (pred<dist2*tol)
      {
         out->flags |= CONVERGED_FLAG;
      }
      return true;
   }
}

/* minimize ||resid - jac * dr||_2, with |dr| <= tr, |r0+dr|<=1
   (exact solution of trust region problem) */
static MFEM_HOST_DEVICE void newton_face( findptsElementPoint_t *const out,
                                          const double jac[sDIM*rDIM],
                                          const double rhes[3],
                                          const double resid[sDIM],
                                          const int flags,
                                          const findptsElementPoint_t *const p,
                                          const double tol )
{
   const double tr = p->tr;
   double bnd[4];
   double r[2], dr[2] = {0, 0};
   int mask, new_flags;
   double v, tv;
   int i;
   double A[3], y[2], r0[2];

   /* A = J^T J - resid_d H_d
      Technically A has one more term, but it is same as A[1],
      since A is symmetric (both J^T J and H_d are symmetric )
   */
   A[0] = jac[0]*jac[0] + jac[2]*jac[2] + jac[4]*jac[4] - rhes[0];
   A[1] = jac[0]*jac[1] + jac[2]*jac[3] + jac[4]*jac[5] - rhes[1];
   A[2] = jac[1]*jac[1] + jac[3]*jac[3] + jac[5]*jac[5] - rhes[2];

   /* y = J^T r */
   y[0] = jac[0]*resid[0] + jac[2]*resid[1] + jac[4]*resid[2];
   y[1] = jac[1]*resid[0] + jac[3]*resid[1] + jac[5]*resid[2];

   r0[0] = p->r[0];
   r0[1] = p->r[1];

   new_flags = flags;
   mask = 0xfu; // 1111 - MSB to LSB - smax,smin,rmax,rmin

   // bnd stores limits of r based on the initial guess (r0) and trust region,
   // and the mask is modified correspondingly.
   // Example: r0 = [0.2, -0.3].. r0[0]-tr = 0.2-1 = -0.8.
   // In this case the bounding box will be set to -0.8 for r=-1 edge of the face
   // and the bit corresponding to rmin will be changed.

   if (r0[0]-tr > -1)   // unconstrained at r=-1, mask's 1st bit is set to 0
   {
      bnd[0] = -tr, mask ^= 1u;
   }
   else
   {
      bnd[0] = -1-r0[0];
   }
   if (r0[0]+tr < 1)   // unconstrained at r=1, mask's 2nd bit is set to 0
   {
      bnd[1] = tr, mask ^= 2u;
   }
   else
   {
      bnd[1] = 1-r0[0];
   }
   if (r0[1]-tr > -1)   // unconstrained at s=-1, mask's 3rd bit is set to 0
   {
      bnd[2] = -tr, mask ^= 1u<<2;
   }
   else
   {
      bnd[2] = -1-r0[1];
   }
   if (r0[1]+tr < 1)   // unconstrained at s=1, mask's 4th bit is set to 0
   {
      bnd[3] = tr, mask ^= 2u << 2;
   }
   else
   {
      bnd[3] = 1 - r0[1];
   }
   // At this stage, mask has information on if the search space is constrained,
   // and the specific edge of the face it is constrained to.
   // bnd has the corresponding limits of the search space.

   if (A[0]+A[2]<=0 || A[0]*A[2]<=A[1]*A[1])
   {
      goto newton_face_constrained;
   }

   lin_solve_sym_2(dr, A, y);

#define EVAL(r,s) -(y[0]*r + y[1]*s) + (r*A[0]*r + (2*r*A[1] + s*A[2])*s)/2
   if ((dr[0]-bnd[0])*(bnd[1]-dr[0])>=0 && (dr[1]-bnd[2])*(bnd[3]-dr[1])>=0)
   {
      r[0] = r0[0] + dr[0], r[1] = r0[1] + dr[1];
      v = EVAL(dr[0], dr[1]);
      goto newton_face_fin;
   }

newton_face_constrained:
   v  = EVAL(bnd[0], bnd[2]); // bound at r=-1 and s=-1
   i  = 1u|(1u<<2);           // 0101b
   tv = EVAL(bnd[1], bnd[2]); // bound at r=1 and s=-1
   if (tv<v)
   {
      v = tv, i = 2u|(1u<<2); // i = 0110b
   }
   tv = EVAL(bnd[0], bnd[3]); // bound at r=-1 and s=1
   if (tv<v)
   {
      v = tv, i = 1u|(2u<<2); // i = 1001b
   }
   tv = EVAL(bnd[1], bnd[3]); // bound at r=1 and s=1
   if (tv<v)
   {
      v = tv, i = 2u|(2u<<2); // i = 1010b
   }

   if (A[0]>0)   // for r[0] (i.e., r) ref coord
   {
      double drc;
      drc = (y[0] - A[1]*bnd[2])/A[0];
      if ( (drc-bnd[0])*(bnd[1]-drc)>=0 && // if drc lies within r=-1 and r=1
           (tv=EVAL(drc,bnd[2]))<v )
      {
         // i = 0100b, relieve constrainsts at r=-1 or 1, and set constraints at s=-1
         v = tv, i = 1u<<2, dr[0] = drc;
      }
      drc = (y[0] - A[1]*bnd[3])/A[0];
      if ( (drc-bnd[0])*(bnd[1]-drc)>=0 && // if drc lies within r=-1 and r=1
           (tv=EVAL(drc,bnd[3]))<v )
      {
         // i = 1000b, relieve constraints at r=-1 or 1, and set constraints at s=1
         v = tv, i = 2u<<2, dr[0] = drc;
      }
   }
   if (A[2]>0)   // for r[1] (i.e., s) ref coord
   {
      double drc;
      drc = (y[1] - A[1]*bnd[0])/A[2];
      if ( (drc-bnd[2])*(bnd[3]-drc)>=0 && // if drc lies within s=-1 and s=1
           (tv = EVAL(bnd[0], drc)) < v)
      {
         v = tv, i = 1u, dr[1] = drc;      // i = 0001b, set constraints at r=-1
      }
      drc = (y[1] - A[1]*bnd[1])/A[2];
      if ((drc-bnd[2])*(bnd[3]-drc)>=0 &&  // if drc lies within s=-1 and s=1
          (tv = EVAL(bnd[1], drc))<v)
      {
         v = tv, i = 2u, dr[1] = drc;      // i = 0010b, set constraints at r=1
      }
   }
#undef EVAL

   {
      for (int d=0; d<rDIM; ++d)
      {
         // For d=0, f=0 if r is unconstrained; f=1 if r is constrained at -1; f=2 if r is constrained at 1
         // For d=1, f=0 if s is unconstrained; f=1 if s is constrained at -1; f=2 if s is constrained at 1
         const int f = (i>>2*d) & 3u;
         if (f==0)    // if r (or s) is unconstrained
         {
            r[d] = r0[d] + dr[d];
         }
         else         // if r (or s) is constrained
         {
            if ( ( f&(mask>>(2*d)) ) == 0 )
            {
               r[d] = r0[d] + (f==1 ? -tr : tr);
            }
            else
            {
               r[d] = (f==1 ? -1 : 1), new_flags |= f<<(2*d);
            }
         }
      }
   }

newton_face_fin:
   out->dist2p = -2*v;
   dr[0] = r[0] - p->r[0];
   dr[1] = r[1] - p->r[1];
   if ( fabs(dr[0])+fabs(dr[1]) < tol)
   {
      new_flags |= CONVERGED_FLAG;
   }
   out->r[0] = r[0], out->r[1] = r[1];
   out->flags = new_flags | (p->flags<<5);
}

static MFEM_HOST_DEVICE inline void newton_edge(findptsElementPoint_t *const
                                                out,
                                                const double jac[sDIM*rDIM],
                                                const double rhes,
                                                const double resid[sDIM],
                                                const int de,
                                                const int dn,
                                                int flags,
                                                const findptsElementPoint_t *const p,
                                                const double tol)
{
   const double tr = p->tr;
   /* A = J^T J - resid_d H_d */
   const double A = jac[de]       *jac[de]
                    + jac[de+rDIM]  *jac[de+rDIM]
                    + jac[de+2*rDIM]*jac[de+2*rDIM]
                    - rhes;
   /* y = J^T r */
   const double y = jac[de]       *resid[0]
                    + jac[de+rDIM]  *resid[1]
                    + jac[de+2*rDIM]*resid[2];

   const double oldr = p->r[de];
   double dr, nr, tdr, tnr;
   double v, tv;
   int new_flags = 0, tnew_flags = 0;

#define EVAL(dr) (dr*A - 2*y)*dr

   /* if A is not SPD, quadratic model has no minimum */
   if (A>0)
   {
      dr = y/A;
      // if dr is too small, set it to 0. Required since roundoff dr could cause
      // fabs(newr)<1 to succeed when it shouldn't.
      // FIXME: This check might be redundant since for 3d surface meshes, we have
      // normal derivatives available and hence dr=0 truly means we are converged.
      //  we also check for dist2<dist2tol in newton iterations loop, which is a
      //  sureshot safeguard against false converged flag sets.
      if (fabs(dr)<tol)
      {
         dr=0.0;
         nr = oldr;
      }
      else
      {
         nr = oldr+dr;
      }
      if ( fabs(dr)<tr && fabs(nr)<1 )
      {
         v = EVAL(dr);
         goto newton_edge_fin;
      }
   }

   if ( (nr=oldr-tr)>-1 )
   {
      dr = -tr;
   }
   else
   {
      nr = -1, dr = -1-oldr, new_flags = flags | 1u<<2*de;
   }
   v = EVAL(dr);

   if ( (tnr = oldr+tr)<1 )
   {
      tdr = tr;
   }
   else
   {
      tnr = 1, tdr = 1-oldr, tnew_flags = flags | 2u<<2*de;
   }
   tv = EVAL(tdr);

   if (tv<v)
   {
      nr = tnr, dr = tdr, v = tv, new_flags = tnew_flags;
   }

newton_edge_fin:
   /* check convergence */
   if ( fabs(dr)<tol )
   {
      new_flags |= CONVERGED_FLAG;
   }
   out->r[de] = nr;
   out->r[dn] = p->r[dn];
   out->dist2p = -v;
   out->flags = flags | new_flags | (p->flags<<5);
}

static MFEM_HOST_DEVICE void seed_j(const double *elx[sDIM],
                                    const double x[sDIM],
                                    const double *z,
                                    double *dist2,
                                    double *r[rDIM],
                                    const int j,
                                    const int pN)
{
   dist2[j] = HUGE_VAL;
   double zr = z[j];
   for (int k=0; k<pN; ++k)
   {
      double zs = z[k];
      const int jk = j + k*pN; // dof index
      double dx[sDIM];
      for (int d=0; d<sDIM; ++d)
      {
         dx[d] = x[d] - elx[d][jk];
      }
      const double dist2_jk = l2norm2(dx);
      if (dist2[j]>dist2_jk)
      {
         dist2[j] = dist2_jk;
         r[0][j] = zr;
         r[1][j] = zs;
      }
   }
}

// global memory access of element coordinates.
// Are the structs being stored in "local memory" or registers?
template<int T_D1D = 0>
static void FindPointsSurfLocal3D_Kernel(const int npt,
                                          const double tol,
                                          const double dist2tol,
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
                                          double *infok,
                                          const int pN = 0)
{
#define MAXC(a, b) (((a) > (b)) ? (a) : (b))
   const int MD1   = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
   const int D1D   = T_D1D ? T_D1D : pN;
   const int p_NE  = D1D*D1D;  // total nos. points in an element
   MFEM_VERIFY(MD1<=DofQuadLimits::MAX_D1D,
              "Increase Max allowable polynomial order.");
   MFEM_VERIFY(pN<=DofQuadLimits::MAX_D1D,
              "Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D!=0, "Polynomial order not specified.");
   const int nThreads = MAXC(D1D*sDIM, 9);

   mfem::forall_2D(npt, nThreads, 1, [=] MFEM_HOST_DEVICE (int i)
   {
      constexpr int size1 = 18*MD1 + 12;
      constexpr int size2 = 9*MD1;
      constexpr int size3 = MD1*MD1*sDIM;  // local element coordinates

      MFEM_SHARED double r_workspace[size1];
      MFEM_SHARED findptsElementPoint_t el_pts[2];

      MFEM_SHARED double constraint_workspace[size2];
      MFEM_SHARED int edge_init;

      MFEM_SHARED double elem_coords[MD1 <= 6 ? size3 : 1];

      double *r_workspace_ptr = r_workspace;
      findptsElementPoint_t *fpt, *tmp;
      fpt = el_pts + 0;
      tmp = el_pts + 1;

      int id_x = point_pos_ordering==0 ?       i :   i*sDIM;
      int id_y = point_pos_ordering==0 ?   npt+i : 1+i*sDIM;
      int id_z = point_pos_ordering==0 ? 2*npt+i : 2+i*sDIM;
      double x_i[3] = {x[id_x], x[id_y], x[id_z]};

      unsigned int *code_i = code_base  + i;
      double *dist2_i      = dist2_base + i;

      //// map_points_to_els ////
      findptsLocalHashData_t hash;
      for (int d=0; d<sDIM; ++d)
      {
         hash.bnd[d].min = hashMin[d];
         hash.fac[d]     = hashFac[d];
      }
      hash.hash_n = hash_n;
      hash.offset = hashOffset;
      const unsigned int hi   = hash_index(&hash, x_i);
      const unsigned int *elp = hash.offset + hash.offset[hi],
                          *const ele = hash.offset + hash.offset[hi+1];
      *code_i  = CODE_NOT_FOUND;
      *dist2_i = DBL_MAX;

      for (; elp!=ele; ++elp)
      {
         const unsigned int el = *elp;

         // construct obbox on the fly
         obbox_t box;
         int n_box_ents = 3*sDIM + sDIM2;
         for (int idx = 0; idx < sDIM; ++idx)
         {
            box.c0[idx] = boxinfo[n_box_ents*el + idx];
            box.x[idx].min = boxinfo[n_box_ents*el + sDIM + idx];
            box.x[idx].max = boxinfo[n_box_ents*el + 2*sDIM + idx];
         }

         for (int idx = 0; idx < sDIM2; ++idx)
         {
            box.A[idx] = boxinfo[n_box_ents*el + 3*sDIM + idx];
         }

         if (bbox_test(&box, x_i) < 0) { continue; }

         //// findpts_local ////
         {
            if (MD1 <= 6)
            {
               MFEM_FOREACH_THREAD(j,x,D1D*sDIM)
               {
                  const int qp = j % D1D;
                  const int d = j / D1D;
                  for (int k = 0; k < D1D; ++k)
                  {
                     const int jk = qp + k * D1D;
                     elem_coords[jk + d*p_NE] =
                        xElemCoord[jk + el*p_NE + d*p_NE*nel];
                  }
               }
               MFEM_SYNC_THREAD;
            }

            const double *elx[sDIM];
            for (int d=0; d<sDIM; d++)
            {
               elx[d] = MD1<= 6 ? &elem_coords[d*p_NE] :
                        xElemCoord + d*nel*p_NE + el*p_NE;
            }

            MFEM_SYNC_THREAD;
            //// findpts_el ////
            {
               MFEM_FOREACH_THREAD(j,x,1)
               {

                  fpt->dist2 = DBL_MAX;
                  fpt->dist2p = 0;
                  fpt->tr = 1.0;
                  edge_init = 0;
               }
               MFEM_FOREACH_THREAD(j,x,sDIM)
               {
                  fpt->x[j] = x_i[j];
               }
               MFEM_SYNC_THREAD;

               //// seed ////
               {
                  double *dist2_temp = r_workspace_ptr;
                  double *r_temp[rDIM];
                  for (int d=0; d<rDIM; ++d)
                  {
                     r_temp[d] = dist2_temp+(1+d)*D1D;
                  }
                  MFEM_FOREACH_THREAD(j,x,D1D)
                  {
                     // if (j == 0)
                     // {
                     //    printf("Found element %u %u\n", i, *elp);
                     // }
                     seed_j(elx, x_i, gll1D, dist2_temp, r_temp, j, D1D);
                  }
                  MFEM_SYNC_THREAD;

                  MFEM_FOREACH_THREAD(j,x,1)
                  {
                     fpt->dist2 = HUGE_VAL;
                     for (int jj = 0; jj < D1D; ++jj)
                     {
                        //  printf("%u %u %f dist2\n", i, jj, dist2_temp[jj]);
                        if (dist2_temp[jj] < fpt->dist2)
                        {
                           fpt->dist2 = dist2_temp[jj];
                           for (int d = 0; d < rDIM; ++d)
                           {
                              fpt->r[d] = r_temp[d][jj];
                           }
                        }
                     }
                     // printf("%u %f dist2\n", i, fpt->dist2);
                  }
                  MFEM_SYNC_THREAD;
               } //seed done

               MFEM_FOREACH_THREAD(j,x,1)
               {
                  tmp->dist2 = HUGE_VAL;
                  tmp->dist2p = 0;
                  tmp->tr = 1;
                  tmp->flags = 0; // we do newton_vol regardless of seed.
               }
               MFEM_FOREACH_THREAD(j,x,rDIM)
               {
                  tmp->r[j] = fpt->r[j];
               }
               MFEM_FOREACH_THREAD(j,x,sDIM)
               {
                  tmp->x[j] = fpt->x[j];
               }
               MFEM_SYNC_THREAD;

               for (int step=0; step<50; step++)
               {
                  switch (num_constrained(tmp->flags & FLAG_MASK))
                  {
                     case 0:    // findpt_area
                     {
                        double *wt1   = r_workspace_ptr;
                        double *resid = wt1 + 6*D1D;
                        double *jac   = resid + sDIM;
                        double *resid_temp = jac + sDIM*rDIM;
                        double *jac_temp   = resid_temp + sDIM*D1D;

                        // size 3 for sDIM=3, d2f/dr2, d2f/drds, and d2f/ds2
                        double *hes      = jac_temp + sDIM*rDIM*D1D;
                        double *hes_temp = hes + 3;
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(j,x,D1D*rDIM)
                        {
                           const int qp = j % D1D;
                           const int d = j / D1D;
                           lag_eval_second_der(wt1+3*d*D1D, tmp->r[d], qp,
                                               gll1D, lagcoeff, D1D);
                        }
                        MFEM_SYNC_THREAD;

                        double *J1 = wt1, *D1 = wt1+D1D, *DD1 = D1+D1D;
                        double *J2 = wt1 + 3*D1D, *D2 = J2+D1D, *DD2 = D2+D1D;

                        MFEM_FOREACH_THREAD(j,x,D1D*sDIM)
                        {
                           const int qp = j % D1D;
                           const int d = j / D1D;
                           const double *u  = elx[d];
                           double sums_k[3] = {0.0, 0.0, 0.0};
                           for (int k=0; k<D1D; ++k)
                           {
                              sums_k[0] += u[qp + k*D1D] * J2[k];
                              sums_k[1] += u[qp + k*D1D] * D2[k];
                              sums_k[2] += u[qp + k*D1D] * DD2[k];
                           }

                           resid_temp[sDIM*qp+d] = sums_k[0] * J1[qp];
                           jac_temp[sDIM*rDIM*qp+rDIM*d+0] = sums_k[0]*D1[qp];
                           jac_temp[sDIM*rDIM*qp+rDIM*d+1] = sums_k[1]*J1[qp];
                           if (d==0)
                           {
                              hes_temp[3*qp + 0] = sums_k[0] * DD1[qp];
                              hes_temp[3*qp + 1] = sums_k[1] * D1[qp];
                              hes_temp[3*qp + 2] = sums_k[2] * J1[qp];
                           }
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(l,x,sDIM)
                        {
                           resid[l] = fpt->x[l];
                           for (int j=0; j<D1D; ++j)
                           {
                              resid[l] -= resid_temp[l + j*sDIM];
                           }
                        }
                        MFEM_FOREACH_THREAD(l,x,sDIM*rDIM)
                        {
                           jac[l] = 0;
                           for (int j=0; j<D1D; ++j)
                           {
                              jac[l] += jac_temp[l + j*sDIM*rDIM];
                           }
                           if (l<sDIM)   // d2f/dr2, d2f/ds2, and d2f/drds
                           {
                              hes[l] = 0;
                              for (int j=0; j<D1D; ++j)
                              {
                                 hes[l] += hes_temp[l + sDIM*j];
                              }
                              hes[l] *= resid[l];
                           }
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(l,x,1)
                        {
                           if (!reject_prior_step_q(fpt,resid,tmp,tol))
                           {
                              newton_face(fpt,jac,hes,resid,(tmp->flags & CONVERGED_FLAG),tmp,tol);
                           }
                        }
                        MFEM_SYNC_THREAD;
                        break;
                     }
                     case 1:    // findpt_edge
                     {
                        const int ei = edge_index(tmp->flags & FLAG_MASK);
                        const int dn = ei>>1;
                        const int de = plus_1_mod_2(dn);
                        const int d_j[2] = {de,dn};
                        const int hes_count = 3;

                        double *wt = r_workspace_ptr;
                        double *resid = wt + 3*D1D;
                        double *jac   = resid + sDIM;
                        double *hes_T = jac + sDIM*rDIM;
                        double *hes   = hes_T + hes_count*sDIM;
                        findptsElementGEdge_t edge;

                        MFEM_FOREACH_THREAD(j,x,D1D*sDIM)
                        {
                           // utilized first D1D threads
                           edge = get_edge2(elx, wtend, ei,
                                           constraint_workspace, edge_init, j,
                                           D1D);
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(j,x,D1D)
                        {
                           if (j == 0) { edge_init = (1u << ei); }
                           lag_eval_second_der(wt,tmp->r[de],j,gll1D,
                                              lagcoeff,D1D);
                        }
                        MFEM_SYNC_THREAD;

                        const double *const *e_x[4] = {edge.x, edge.x,
                                                       edge.dxdn, edge.d2xdn};
                        MFEM_FOREACH_THREAD(j,x,hes_count*3)
                        {
                           const int d   = j%sDIM; //0,1,2
                           const int row = j/sDIM; //0,1,2
                           {
                              double *wt_j = wt + (row==1 ? D1D : 0);
                              const double *x = e_x[row][d];
                              double sum = 0.0;
                              for (int k=0; k<D1D; ++k)
                              {
                                 sum += wt_j[k]*x[k];
                              }
                              if (row==0)   // j<sDIM
                              {
                                 resid[j] = tmp->x[j] - sum;
                              }
                              else   // row = 1, 2
                              {
                                 jac[ d*rDIM + d_j[row-1] ] = sum;
                              }
                           }

                           {
                              // Hes_T is transposed version (i.e. in col major)
                              double *wt_j  = wt + (2-row)*D1D;
                              hes_T[j] = 0.0;
                              for (int k=0; k<D1D; ++k)
                              {
                                 hes_T[j] += wt_j[k] * e_x[row+1][d][k];
                              }
                           }
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(j,x,hes_count)
                        {
                           hes[j] = 0.0;
                           for (int d=0; d<sDIM; ++d)
                           {
                              hes[j] += resid[d] * hes_T[hes_count*j + d];
                           }
                        }
                        MFEM_SYNC_THREAD;

                        MFEM_FOREACH_THREAD(l,x,1)
                        {
                           if ( !reject_prior_step_q(fpt,resid,tmp,tol))
                           {
                              double steep = 0;
                              for (int d=0; d<sDIM; ++d)
                              {
                                 steep += jac[d*rDIM + dn] * resid[d];
                              }
                              steep *= tmp->r[dn];
                              if (steep<0)
                              {
                                 newton_face( fpt,jac,hes,resid,tmp->flags&CONVERGED_FLAG,tmp,tol);
                              }
                              else
                              {
                                 newton_edge(fpt,jac,hes[0],resid,de,dn,tmp->flags&FLAG_MASK,tmp,tol);
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
                           const int pi=point_index(tmp->flags & FLAG_MASK);
                           const findptsElementGPT_t gpt=get_pt(elx,wtend,pi,D1D);
                           const double *const pt_x = gpt.x;
                           const double *const jac = gpt.jac;
                           const double *const hes = gpt.hes;

                           double resid[sDIM], steep[rDIM];
                           for (int d=0; d<sDIM; ++d)
                           {
                              resid[d] = fpt->x[d]-pt_x[d];
                           }
                           if (!reject_prior_step_q(fpt,resid,tmp,tol))
                           {
                              for (int d=0; d<rDIM; ++d)
                              {
                                 steep[d] = 0;
                                 for (int e=0; e<sDIM; ++e)
                                 {
                                    steep[d] += jac[e*rDIM+d] * resid[e];
                                 }
                                 steep[d] *= tmp->r[d];
                              }

                              int de, dn;
                              if (steep[0]<0)
                              {
                                 if (steep[1]<0)
                                 {
                                    double rh[3];
                                    for (int rd=0; rd<3; ++rd)
                                    {
                                       rh[rd] = 0;
                                       for (int d=0; d<sDIM; ++d)
                                       {
                                          rh[rd] += resid[d] * hes[3*d + rd];
                                       }
                                    }
                                    newton_face(fpt,jac,rh,resid,
                                                (tmp->flags & CONVERGED_FLAG),
                                                tmp,tol);
                                 }
                                 else
                                 {
                                    de = 0, dn = 1;
                                    // hes index 0 is for d2x/dr2
                                    const double rh=resid[0] * hes[0] +
                                                    resid[1] * hes[3+ 0] +
                                                    resid[2] * hes[6 + 0];
                                    newton_edge(fpt,jac,rh,resid,de,dn,
                                    (tmp->flags & ~(3u<<2*de)),tmp,tol);
                                 }
                              }
                              else
                              {
                                 if (steep[1]<0)
                                 {
                                    de = 1, dn = 0;
                                    // hes index 2 is for d2x/ds2
                                    const double rh = resid[0] * hes[2] +
                                                      resid[1] * hes[5] +
                                                      resid[2] * hes[8];
                                    newton_edge(fpt,jac,rh,resid,de,dn,
                                    (tmp->flags & ~(3u<<2*de)),tmp,tol);
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
                     } // case 2
                  } // switch
                  if (fpt->flags & CONVERGED_FLAG)
                  {
                     break;
                  }
                  MFEM_SYNC_THREAD;

                  MFEM_FOREACH_THREAD(j,x,nThreads)
                  {
                     if (j==0)
                     {
                        *tmp = *fpt;
                     }
                  }
                  MFEM_SYNC_THREAD;
               } // for step<50
            } // findpts_el

            bool converged_internal =
                           ((fpt->flags&FLAG_MASK)==CONVERGED_FLAG ) &&
                           fpt->dist2<dist2tol;
            if (*code_i==CODE_NOT_FOUND || converged_internal ||
               fpt->dist2<*dist2_i)
            {
               MFEM_FOREACH_THREAD(j,x,1)
               {
                  *(el_base+i) = el;
                  *code_i = converged_internal ?  CODE_INTERNAL :
                              CODE_BORDER;
                  *dist2_i = fpt->dist2;
               }
               MFEM_FOREACH_THREAD(j,x,rDIM)
               {
                  *(r_base+rDIM*i+j) = fpt->r[j];
               }
               MFEM_SYNC_THREAD;
               if (converged_internal)
               {
                  break;
               }
            }
         } // findpts_local
      } // elp
   });
}

void FindPointsGSLIB::FindPointsSurfLocal3(const Vector &point_pos,
                                            int point_pos_ordering,
                                            Array<unsigned int> &code,
                                            Array<unsigned int> &elem,
                                            Vector &ref,
                                            Vector &dist,
                                            Array<int> &newton,
                                            int npt)
{
   double dist_tol = 1e-10;

   if (npt == 0)
   {
      return;
   }
   MFEM_VERIFY(spacedim==3,"Function for 3D only");
   switch (DEV.dof1d)
   {
      case 1: FindPointsSurfLocal3D_Kernel<1>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.wtend.Read(),
                                                  DEV.bb.Read(),
                                                  DEV.lh_nx,
                                                  DEV.lh_min.Read(),
                                                  DEV.lh_fac.Read(),
                                                  DEV.lh_offset.ReadWrite(),
                                                  code.Write(), elem.Write(),
                                                  ref.Write(), dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  DEV.info.ReadWrite());
         break;
      case 2: FindPointsSurfLocal3D_Kernel<2>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.wtend.Read(),
                                                  DEV.bb.Read(),
                                                  DEV.lh_nx, DEV.lh_min.Read(),
                                                  DEV.lh_fac.Read(),
                                                  DEV.lh_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 3: FindPointsSurfLocal3D_Kernel<3>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.wtend.Read(),
                                                  DEV.bb.Read(),
                                                  DEV.lh_nx, DEV.lh_min.Read(),
                                                  DEV.lh_fac.Read(),
                                                  DEV.lh_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 4: FindPointsSurfLocal3D_Kernel<4>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.wtend.Read(),
                                                  DEV.bb.Read(),
                                                  DEV.lh_nx,
                                                  DEV.lh_min.Read(),
                                                  DEV.lh_fac.Read(),
                                                  DEV.lh_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 5: FindPointsSurfLocal3D_Kernel<5>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.wtend.Read(),
                                                  DEV.bb.Read(),
                                                  DEV.lh_nx,
                                                  DEV.lh_min.Read(),
                                                  DEV.lh_fac.Read(),
                                                  DEV.lh_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 6: FindPointsSurfLocal3D_Kernel<6>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.wtend.Read(),
                                                  DEV.bb.Read(),
                                                  DEV.lh_nx,
                                                  DEV.lh_min.Read(),
                                                  DEV.lh_fac.Read(),
                                                  DEV.lh_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  // newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      default: FindPointsSurfLocal3D_Kernel(npt,
                                                DEV.tol,
                                                DEV.tol,
                                                point_pos.Read(),
                                                point_pos_ordering,
                                                gsl_mesh.Read(),
                                                NE_split_total,
                                                DEV.wtend.Read(),
                                                DEV.bb.Read(),
                                                DEV.lh_nx,
                                                DEV.lh_min.Read(),
                                                DEV.lh_fac.Read(),
                                                DEV.lh_offset.ReadWrite(),
                                                code.Write(),
                                                elem.Write(),
                                                ref.Write(),
                                                dist.Write(),
                                                DEV.gll1d.Read(),
                                                DEV.lagcoeff.Read(),
                                                // newton.ReadWrite(),
                                                DEV.info.ReadWrite(),
                                                DEV.dof1d);
   }
}

#undef sDIM2
#undef rDIM
#undef sDIM
#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND
#else
void FindPointsGSLIB::FindPointsSurfLocal3( const Vector &point_pos,
                                            int point_pos_ordering,
                                            Array<unsigned int> &code,
                                            Array<unsigned int> &elem,
                                            Vector &ref,
                                            Vector &dist,
                                            Array<int> &newton,
                                            int npt ) {} ;
#endif
} // namespace mfem

#endif //ifdef MFEM_USE_GSLIB
