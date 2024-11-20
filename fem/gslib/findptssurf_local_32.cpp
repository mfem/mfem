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
#include "findptssurf_3.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../../linalg/dinvariants.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{

#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2

// #define DEBUG_PRINT
#define sDIM 3
#define rDIM 2
#define sDIM2 (sDIM*sDIM)
#define rDIM2 (rDIM*rDIM)
#define pMax 10
#define nThreads 32

/* p0 is the pointer where the value of the Lagrange polynomial is stored.
   p1 is the pointer where the value of the first derivative of the Lagrange polynomial is stored.
   p2 is the pointer where the value of the second derivative of the Lagrange polynomial is stored.
   x is the point (reference coords here!) at which the derivatives are evaluated.
   i is the index of the Lagrange polynomial.
   z is the array of GLL points.
   lagrangeCoeff: the denominator term in the Lagrange polynomial.
   pN is the number of GLL points, i.e., the number of Lagrange polynomials.
*/
static MFEM_HOST_DEVICE inline void lagrange_eval_second_derivative(
   double *p0,                  // 0 to pN-1: p0, pN to 2*pN-1: p1, 2*pN to 3*pN-1: p2
   double x,                    // ref. coords of the point of interest
   int i,                       // index of the Lagrange polynomial
   const double *z,             // GLL points
   const double *lagrangeCoeff, // Lagrange polynomial denominator term
   int pN)                      // number of GLL points
{
   double u0 = 1, u1 = 0, u2 = 0;
   for (int j=0; j<pN; ++j)
   {
      if (i!=j)
      {
         double d_j = 2 * (x-z[j]);
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
static MFEM_HOST_DEVICE inline double obbox_test(const obbox_t *const b,
                                                 const double x[sDIM])
{
   const double bxyz = obbox_axis_test(b, x);
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

static MFEM_HOST_DEVICE inline double norm2(const double x[sDIM])
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
static MFEM_HOST_DEVICE inline findptsElementGEdge_t get_edge(
   const double *elx[3],
   const double *wtend,
   int ei,
   double *workspace,
   int &side_init,
   int jidx,
   int pN )
{
   findptsElementGEdge_t edge;
   for (int d=0; d<sDIM; ++d)
   {
      edge.x[d]     = workspace             + d*pN;
      edge.dxdn[d]  = workspace +   sDIM*pN + d*pN;
      edge.d2xdn[d] = workspace + 2*sDIM*pN + d*pN;
   }

   // given edge index, compute normal and tangential directions
   const int dn            = ei>>1,
             de            = plus_1_mod_2(dn);
   const int side_n        = ei&1,          // 0 for rmin/smin, 1 for rmax/smax
             side_n_offset = side_n*(pN-1); // offset from rmin/smin to access side_n
   const double *wt1       = wtend + 3*pN*side_n;
   // First 3*pN entries in wtend for rmin/smin, next 3*pN entries for rmax/smax.
   // stores lagfunc and its 1st and 2nd derivatives values

   const int jj = jidx%pN;
   const int dd = jidx/pN;
   const int mask = 1u << ei;
   if ((side_init&mask) == 0)
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

         // This macro only works for lexico-graphically ordered nodes!
#define ELX(d,j,k) elx[d][j*elx_stride[de] + k*elx_stride[dn]]

         // copy first/last entries in normal directions; note side_n_offset is
         // used to access the nodes of only the side of interest.
         edge.x[dd][jj] = ELX(dd, jj, side_n_offset);

         // sum of the product of nodal positions and corresponding 1st and 2nd
         // derivatives, along all nodes in the normal direction.
         double sums_k[2] = {0,0};

         // note how k replaces side_n_offset in the following loop (compared to edge.x assignment);
         // allows us to loop through all nodes in the normal direction.
         for (int k=0; k<pN; ++k)
         {
            sums_k[0] += wt1[pN+k]   * ELX(dd,jj,k); // 1st derivative times nodal position
            sums_k[1] += wt1[2*pN+k] * ELX(dd,jj,k); // 2nd derivative times nodal position
         }
         edge.dxdn[dd][jj]  = sums_k[0];
         edge.d2xdn[dd][jj] = sums_k[1];

#undef ELX
      }
      side_init = mask;
   }
   return edge;  // edge struct containing pointers pointing to constraint workspace are returned!
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
   const double dist2 = norm2(resid);
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
                                    const double *z, //GLL point locations [-1, 1]
                                    double *dist2,
                                    double *r[rDIM],
                                    const int ir,
                                    const int pN)
{
   if (ir>=pN)
   {
      return;
   }

   dist2[ir] = DBL_MAX;
   double zr =
      z[ir];             // r gll coord for ir-th thread, i.e., ir-th dof in r-direction
   for (int is=0; is<pN;
        ++is)    // loop through all "s" gll coords for ir-th dof in s-direction
   {
      double zs = z[is];
      const int irs = ir + is*pN; // dof index
      double dx[sDIM];
      for (int d=0; d<sDIM; ++d)
      {
         dx[d] = x[d] - elx[d][irs];
      }
      const double dist2_rs = norm2(dx);
      if (dist2[ir]>dist2_rs)
      {
         dist2[ir] = dist2_rs;
         r[0][ir] = zr;
         r[1][ir] = zs;
      }
   }
}

// global memory access of element coordinates.
// Are the structs being stored in "local memory" or registers?
template<int T_D1D = 0>
static void FindPointsSurfLocal32D_Kernel(const int npt,
                                          const double tol,
                                          const double dist2tol,
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
                                          //   int *newton,
                                          double *infok,
                                          const int pN = 0)
{
#define MAX_CONST(a, b) (((a) > (b)) ? (a) : (b))
   const int MD1   = T_D1D ? T_D1D : pMax;
   const int D1D   = T_D1D ? T_D1D : pN;
   const int p_NE  = D1D*D1D;  // total nos. points in an element
   const int p_NEL = p_NE*nel; // total nos. points in all elements
   MFEM_VERIFY(MD1<=pMax, "Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D!=0, "Polynomial order not specified.");
   std::cout << std::setprecision(9);

   mfem::forall_2D(npt, nThreads, 1, [=] MFEM_HOST_DEVICE (int i)
   {
      // adi: understand the sizes! and how that changes when 1 ref dim is removed.
      //size depends on max of info for faces and edges
      constexpr int size1 = MAX_CONST(4,MD1+1) * (3*3 + 2*3) + 3*2*MD1 + 5;
      constexpr int size2 = MAX_CONST(MD1*MD1*6,
                                      3*sDIM*MD1); // 2nd 3 for storing x, dxdn, d2xdn

      MFEM_SHARED double r_workspace[size1];
      MFEM_SHARED findptsElementPoint_t
      el_pts[2];  // two pts structs for saving previous iteration pts data
      MFEM_SHARED double constraint_workspace[size2];
      MFEM_SHARED int constraint_init_t[nThreads];

      double *r_workspace_ptr = r_workspace;
      findptsElementPoint_t *fpt, *tmp;
      fpt = el_pts + 0;
      tmp = el_pts + 1;

      int id_x = point_pos_ordering==0 ?       i :   i*sDIM;
      int id_y = point_pos_ordering==0 ?   npt+i : 1+i*sDIM;
      int id_z = point_pos_ordering==0 ? 2*npt+i : 2+i*sDIM;
      double x_i[3] = {x[id_x], x[id_y], x[id_z]};

      unsigned int *code_i = code_base  + i;
      unsigned int *el_i   = el_base    + i;
      double *r_i          = r_base     + i*rDIM;
      double *dist2_i      = dist2_base + i;
      // int *newton_i = newton + i;

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

      for (; elp!=ele; ++elp)   // note we are incrementing pointers here!!
      {
         const unsigned int el = *elp;   // element ID in the hash
         obbox_t box;
         // construct obbox_t on the fly from data
         for (int d=0; d<sDIM; ++d)
         {
            box.c0[d]    = c[sDIM*el + d];
            box.x[d].min = minBound[sDIM*el + d];
            box.x[d].max = maxBound[sDIM*el + d];
         }

         for (int d2=0; d2<sDIM2; ++d2)
         {
            box.A[d2] = A[sDIM2*el + d2];
         }

         if (obbox_test(&box, x_i)>=0)
         {
            //// findpts_local ////
            {
               const double *elx[sDIM];
               for (int d=0; d<sDIM; d++)
               {
                  elx[d] = xElemCoord + d*p_NEL + el*p_NE;
               }

               MFEM_SYNC_THREAD;
               //// findpts_el ////
               {
                  MFEM_FOREACH_THREAD(j,x,nThreads)
                  {
                     if (j==0)
                     {
                        fpt->dist2 = DBL_MAX;
                        fpt->dist2p = 0;
                        fpt->tr = 1.0;
                     }
                     if (j<sDIM)
                     {
                        fpt->x[j] = x_i[j];
                     }
                     constraint_init_t[j] = 0;
                  }
                  MFEM_SYNC_THREAD;

                  //// seed ////
                  {
                     double *dist2_temp = r_workspace_ptr; // size: D1D
                     double *r_temp[rDIM];
                     for (int d=0; d<rDIM; ++d)
                     {
                        r_temp[d] = r_workspace_ptr + D1D + d*D1D;
                     }

                     // look for the node closest to x_i in the element among
                     // all nodes with r-coord = ir-th GLL point.
                     // r_temp stores the r,s coords of the closest node.
                     // dist2_temp stores the distance^2 of the closest node.
                     MFEM_FOREACH_THREAD(j,x,nThreads)
                     {
                        seed_j(elx, x_i, gll1D, dist2_temp, r_temp, j, D1D);
                     }
                     MFEM_SYNC_THREAD;

                     // The closest node (smallest dist2_temp) found across all threads is stored in fpt.
                     MFEM_FOREACH_THREAD(j,x,nThreads)
                     {
                        if (j==0)
                        {
                           fpt->dist2 = DBL_MAX;
                           // for (int ir=D1D/2; ir<D1D/2+1; ++ir)
                           for (int ir=0; ir<D1D; ++ir)
                           // loop through all r-th dof data obtained from seed_j
                           {
                              if (dist2_temp[ir] < fpt->dist2)
                              {
                                 fpt->dist2 = dist2_temp[ir];
                                 for (int d=0; d<rDIM; ++d)
                                 {
                                    fpt->r[d] = r_temp[d][ir];
                                 }
                              }
                           }
                        }

                     }
                     MFEM_SYNC_THREAD;
                  } //seed done
                  // std::cout << " Step: " << -1 <<
                  //           " xyz: " << x_i[0] << " " << x_i[1] << " " << x_i[2] <<
                  //           " el: " << el <<
                  //           " fpt->r: " << fpt->r[0] << " " <<  fpt->r[1] << " " << fpt->r[2] <<
                  //           " fpt->dist2: " << fpt->dist2 << " k10-seed\n";

                  MFEM_FOREACH_THREAD(j,x,nThreads)
                  {
                     if (j<sDIM)
                     {
                        tmp->x[j] = fpt->x[j];
                     }
                     else if (j==sDIM)
                     {
                        tmp->dist2  = DBL_MAX;
                        tmp->dist2p = 0;
                        tmp->tr     = 1.0; //0.25;
                        tmp->flags   = 0;
                        tmp->r[0]   = fpt->r[0];
                        tmp->r[1]   = fpt->r[1];
                     }
                  }
                  MFEM_SYNC_THREAD;

                  for (int step=0; step<50; step++)
                  {
                     int nc = num_constrained(tmp->flags &
                                              FLAG_MASK); // number of constrained reference directions
                     switch (nc)
                     {
                        case 0:    // findpt_area
                        {
                           double *wt1   =
                              r_workspace_ptr; // value, 1st, and 2nd derivative at r ref coord
                           double *wt2   = wt1 +
                                           3*D1D;     // value, 1st, and 2nd derivative at s ref coord
                           double *resid = wt2 + 3*D1D;     // 3 residuals for 3 phy. coords
                           double *jac   = resid +
                                           sDIM;    // sDIM*rDIM elements in jacobian matrix dimensions

                           // see their calculation loops to understand their sizes!
                           double *resid_temp = jac + sDIM*rDIM;
                           double *jac_temp   = resid_temp + sDIM*D1D;

                           // size 3 for sDIM=3, d2f/dr2, d2f/drds, and d2f/ds2
                           double *hes      = jac_temp + sDIM*rDIM*D1D;
                           double *hes_temp = hes + 3;
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j<D1D)
                              {
                                 lagrange_eval_second_derivative(wt1, tmp->r[0], j, gll1D, lagcoeff, D1D);
                              }
                              else if (j<2*D1D)
                              {
                                 lagrange_eval_second_derivative(wt2, tmp->r[1], j-D1D, gll1D, lagcoeff, D1D);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           double *J1 = wt1, *D1 = wt1+D1D,
                                   *DD1 = wt1+2*D1D; // for r ref coord, value, 1st, and 2nd derivative
                           double *J2 = wt2, *D2 = wt2+D1D,
                                   *DD2 = wt2+2*D1D; // for s ref coord, value, 1st, and 2nd derivative

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              // Each thread works on a specific sDIM physical direction and all s in s
                              // direction for a specific r among D1D r's in r direction
                              // Hence we need to utilize sDIM*D1D theads for this task
                              if (j<D1D*sDIM)
                              {
                                 const int d      = j%sDIM; // phy.coord index
                                 const int qp     = j/sDIM; // dof id qp, remains constant for all d=0 to sDIM-1
                                 const double *u  = elx[d]; // one of the phy. coord. components, depending on d
                                 double sums_k[3] = {0.0, 0.0, 0.0};

                                 // loop through all nodes in s direction for the qp-th dof in r,
                                 // and sum their contributions to jacobians and hessians
                                 for (int k=0; k<D1D; ++k)
                                 {
                                    sums_k[0] += u[qp + k*D1D] * J2[k];  // coefficient*lagfuncvalue in s direction
                                    sums_k[1] += u[qp + k*D1D] *
                                                 D2[k];  // coefficient*lagfunc1stderivative in s direction
                                    sums_k[2] += u[qp + k*D1D] *
                                                 DD2[k]; // coefficient*lagfunc2ndderivative in s direction
                                 }

                                 resid_temp[sDIM*qp + d]             = sums_k[0] *
                                                                       J1[qp]; // (coefficient*lagfuncvalue in s direction) * (lagfuncvalue in r direction)
                                 jac_temp[sDIM*rDIM*qp + rDIM*d + 0] = sums_k[0] *
                                                                       D1[qp]; // (coefficient*lagfuncvalue in s direction) * (lagfunc1stderivative in r direction)
                                 jac_temp[sDIM*rDIM*qp + rDIM*d + 1] = sums_k[1] *
                                                                       J1[qp]; // (coefficient*lagfunc1stderivative in s direction) * (lagfuncval in r direction)

                                 if (d==0)
                                 {
                                    hes_temp[3*qp]     = sums_k[0] *
                                                         DD1[qp]; // (coefficient*lagfuncvalue in s direction) * (lagfunc2ndderivative in r direction)
                                    hes_temp[3*qp + 1] = sums_k[1] *
                                                         D1[qp];  // (coefficient*lagfunc1stderivative in s direction) * (lagfunc1stderivative in r direction)
                                    hes_temp[3*qp + 2] = sums_k[2] *
                                                         J1[qp];  // (coefficient*lagfunc2ndderivative in s direction) * (lagfuncval in r direction)
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(l,x,nThreads)
                           {
                              if (l<sDIM)
                              {
                                 resid[l] = fpt->x[l];
                                 for (int j=0; j<D1D; ++j)
                                 {
                                    resid[l] -= resid_temp[l + j*sDIM];
                                 }
                              }
                              if (l<sDIM*rDIM)
                              {
                                 jac[l] = 0;
                                 for (int j=0; j<D1D; ++j)
                                 {
                                    jac[l] += jac_temp[l + j*sDIM*rDIM];
                                 }
                              }
                              if (l<3)   // d2f/dr2, d2f/ds2, and d2f/drds
                              {
                                 hes[l] = 0;
                                 for (int j=0; j<D1D; ++j)
                                 {
                                    hes[l] += hes_temp[l + 3*j];
                                 }
                                 hes[l] *= resid[l];
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(l,x,nThreads)
                           {
                              if (l==0)
                              {
                                 if (!reject_prior_step_q(fpt,resid,tmp,tol))
                                 {
                                    newton_face(fpt,jac,hes,resid,(tmp->flags & CONVERGED_FLAG),tmp,tol);
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        }
                        case 1:    // findpt_edge
                        {
                           const int ei        = edge_index(tmp->flags & FLAG_MASK);
                           const int dn        = ei>>1;
                           const int de        = plus_1_mod_2(dn);
                           const int d_j[2]    = {de,dn};
                           const int hes_count = 3; // phi''.edge.x, phi'.edge.dxdn, phi.edge.d2xdn

                           double *wt    =
                              r_workspace_ptr;       // 3*D1D to store lagrange values, 1st and 2nd derivatives
                           double *resid = wt + 3*D1D;            // sDIM residuals for sDIM phy. coords
                           double *jac   = resid +
                                           sDIM;          // sDIM*rDIM elements in jacobian for an edge
                           double *hes_T = jac + sDIM*rDIM;       // hes_count*3
                           double *hes   = hes_T +
                                           hes_count*sDIM;// <hes_count> stuff for each of the sDIM dimensions
                           findptsElementGEdge_t edge;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              // utilized first 3*D1D threads
                              edge = get_edge(elx, wtend, ei, constraint_workspace, constraint_init_t[j], j,
                                              D1D);
                           }
                           MFEM_SYNC_THREAD;

                           // Now, edge points to all edge related data (nodal values, derivatives,
                           // 2nd derivatives, etc) that has been stored in constraint_workspace.

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j<D1D)
                              {
                                 lagrange_eval_second_derivative(wt,tmp->r[de],j,gll1D,lagcoeff,D1D);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           const double *const *e_x[4] = {edge.x, edge.x, edge.dxdn, edge.d2xdn};
                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              const int d           = j%sDIM;
                              const int iactive_e_x = j/sDIM;
                              if ( j<(sDIM*rDIM + sDIM) )    // sDIM*rDIM(jac) + sDIM(resid)
                              {
                                 // wt     = [vals, 1st derivatives, 2nd derivatives]
                                 // wt+0   = vals,            for iactive_e_x!=1
                                 // wt+D1D = 1st derivatives, for iactive_e_x==1
                                 double *wt_j    = wt + (iactive_e_x==1 ? D1D : 0); // for all j
                                 const double *x = e_x[iactive_e_x][d];             // for all j
                                 double sum = 0.0;
                                 // for iactive_e_x=0, sum = lagfuncvals     * edge.x
                                 // for iactive_e_x=1, sum = lagfunc1stderiv * edge.x
                                 // for iactive_e_x=2, sum = lagfuncvals     * edge.dxdn = actual dxdn
                                 for (int k=0; k<D1D; ++k)
                                 {
                                    sum += wt_j[k]*x[k];
                                 }
                                 if (iactive_e_x==0)   // j<sDIM
                                 {
                                    resid[j] = tmp->x[j] - sum;
                                 }
                                 else   // iactive_e_x = 1, 2
                                 {
                                    jac[ d*rDIM + d_j[iactive_e_x-1] ] = sum;
                                 }
                              }

                              if (j<hes_count*sDIM)
                              {
                                 // Hes_T is transposed version (i.e. in col major)
                                 double *wt_j  = wt + (2-iactive_e_x)*D1D; // 2ndderiv,1stderiv,val
                                 hes_T[j] = 0.0;
                                 for (int k=0; k<D1D; ++k)
                                 {
                                    // iactive_e_x+1 to start at the second edge.x in e_x array
                                    hes_T[j] += wt_j[k] * e_x[iactive_e_x+1][d][k];
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j<hes_count)
                              {
                                 hes[j] = 0.0;
                                 for (int d=0; d<sDIM; ++d)
                                 {
                                    hes[j] += resid[d] * hes_T[hes_count*j + d];
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(l,x,nThreads)
                           {
                              if (l==0)   // check prior step
                              {
                                 if ( !reject_prior_step_q(fpt,resid,tmp,tol) )   // check constraint
                                 {
                                    double steep = 0;
                                    for (int d=0; d<sDIM; ++d)
                                    {
                                       steep += jac[d*rDIM + dn] * resid[d];
                                    }
                                    steep *= tmp->r[dn];
                                    if (steep<0)
                                    {
                                       // no constraints on ref-dims anymore! Flags sent to
                                       // newton_face reflects this.
                                       newton_face( fpt,jac,hes,resid,tmp->flags&CONVERGED_FLAG,tmp,tol);
                                    }
                                    else
                                    {
                                       newton_edge(fpt,jac,hes[0],resid,de,dn,tmp->flags&FLAG_MASK,tmp,tol);
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
                              if (j==0)
                              {
                                 // 0,1,2,3 for (-1,-1), (1,-1), (-1,1), (1,1)
                                 const int pi                 = point_index(tmp->flags & FLAG_MASK);
                                 const findptsElementGPT_t gpt = get_pt(elx,wtend,pi,D1D);
                                 const double *const pt_x     = gpt.x,
                                                     *const jac      = gpt.jac,
                                                            *const hes      = gpt.hes;
                                 const int hes_count = 3;

                                 double resid[sDIM],
                                        steep[rDIM];
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
                                          for (int rd=0; rd<hes_count; ++rd)
                                          {
                                             rh[rd] = 0;
                                             for (int d=0; d<sDIM; ++d)
                                             {
                                                rh[rd] += resid[d] * hes[hes_count*d + rd];
                                             }
                                          }
                                          // no constraints on ref-dims anymore! Flags sent to
                                          // newton_face reflect this.
                                          newton_face(fpt,jac,rh,resid,(tmp->flags & CONVERGED_FLAG),tmp,tol);
                                       }
                                       else
                                       {
                                          de = 0, dn = 1;
                                          // hes index 0 is for d2x/dr2
                                          const double rh = resid[0] * hes[              0] +
                                                            resid[1] * hes[hes_count   + 0] +
                                                            resid[2] * hes[hes_count*2 + 0];
                                          newton_edge(fpt,jac,rh,resid,de,dn,(tmp->flags & ~(3u<<2*de)),tmp,tol);
                                       }
                                    }
                                    else
                                    {
                                       if (steep[1]<0)
                                       {
                                          de = 1, dn = 0;
                                          // hes index 2 is for d2x/ds2
                                          const double rh = resid[0] * hes[              2] +
                                                            resid[1] * hes[hes_count   + 2] +
                                                            resid[2] * hes[hes_count*2 + 2];
                                          newton_edge(fpt,jac,rh,resid,de,dn,(tmp->flags & ~(3u<<2*de)),tmp,tol);
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
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        } // case 2
                     } // switch

                     if (fpt->flags & CONVERGED_FLAG)
                     {
                        // *newton_i = step+1;
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

               bool converged_internal = ( (fpt->flags&FLAG_MASK)==CONVERGED_FLAG ) &&
                                         fpt->dist2<dist2tol;
               if (*code_i==CODE_NOT_FOUND || converged_internal || fpt->dist2<*dist2_i)
               {
                  MFEM_FOREACH_THREAD(j,x,nThreads)
                  {
                     if (j==0)
                     {
                        *el_i    = el;
                        *code_i  = converged_internal ? CODE_INTERNAL : CODE_BORDER;
                        *dist2_i = fpt->dist2;
                     }
                     if (j<rDIM)
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
            } // findpts_local
         } // obbox_test
      } // elp
   });
}

void FindPointsGSLIB::FindPointsSurfLocal32(Vector &point_pos,
                                            int point_pos_ordering,
                                            Array<unsigned int> &code,
                                            Array<unsigned int> &elem,
                                            Vector &ref,
                                            Vector &dist,
                                            Array<int> &newton,
                                            int npt)
{
   point_pos.HostReadWrite();
   gsl_mesh.HostReadWrite();
   DEV.o_wtend.HostReadWrite();
   DEV.o_c.HostReadWrite();
   DEV.o_A.HostReadWrite();
   DEV.o_min.HostReadWrite();
   DEV.o_max.HostReadWrite();
   DEV.o_hashMin.HostReadWrite();
   DEV.o_hashFac.HostReadWrite();
   DEV.ou_offset.HostReadWrite();
   DEV.gll1d.HostReadWrite();
   DEV.lagcoeff.HostReadWrite();
   DEV.info.HostReadWrite();

   double dist_tol = 1e-14;

   if (npt == 0)
   {
      return;
   }
   MFEM_VERIFY(spacedim==3,"Function for 3D only");
   switch (DEV.dof1d)
   {
      case 1: FindPointsSurfLocal32D_Kernel<1>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.o_wtend.Read(),
                                                  DEV.o_c.Read(),
                                                  DEV.o_A.Read(),
                                                  DEV.o_min.Read(),
                                                  DEV.o_max.Read(),
                                                  DEV.hash_n,
                                                  DEV.o_hashMin.Read(),
                                                  DEV.o_hashFac.Read(),
                                                  DEV.ou_offset.ReadWrite(),
                                                  code.Write(), elem.Write(),
                                                  ref.Write(), dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 2: FindPointsSurfLocal32D_Kernel<2>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.o_wtend.Read(),
                                                  DEV.o_c.Read(),
                                                  DEV.o_A.Read(),
                                                  DEV.o_min.Read(),
                                                  DEV.o_max.Read(),
                                                  DEV.hash_n, DEV.o_hashMin.Read(),
                                                  DEV.o_hashFac.Read(),
                                                  DEV.ou_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 3: FindPointsSurfLocal32D_Kernel<3>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.o_wtend.Read(),
                                                  DEV.o_c.Read(),
                                                  DEV.o_A.Read(),
                                                  DEV.o_min.Read(),
                                                  DEV.o_max.Read(),
                                                  DEV.hash_n, DEV.o_hashMin.Read(),
                                                  DEV.o_hashFac.Read(),
                                                  DEV.ou_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 4: FindPointsSurfLocal32D_Kernel<4>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
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
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 5: FindPointsSurfLocal32D_Kernel<5>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
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
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 6: FindPointsSurfLocal32D_Kernel<6>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
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
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  // newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      default: FindPointsSurfLocal32D_Kernel(npt,
                                                DEV.tol,
                                                DEV.tol,
                                                point_pos.Read(),
                                                point_pos_ordering,
                                                gsl_mesh.Read(),
                                                NE_split_total,
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
                                                DEV.gll1d.Read(),
                                                DEV.lagcoeff.Read(),
                                                // newton.ReadWrite(),
                                                DEV.info.ReadWrite(),
                                                DEV.dof1d);
   }
}

// Polynomial order = p
// Let's assume q = p+1
// Dimension = dim = 3
// Number of points = npt
// Number of elements = nel

// Data being read
// Doubles (3): npt, DEV.tol, point_pos_ordering
// Ints (3): NE_split_total, DEV.hash_n, DEV.dof1d
// Arrays/Vectors:
// point_pos [double] = (npt*dim)
// gsl_mesh [double] = (nel*(p+1)^dim)*dim
// DEV.o_wtend [double] =  6*(p+1)
// DEV.o_c [double] = nel*dim
// DEV.o_A [double] = nel*dim*dim
// DEV.o_min [double] = nel*dim
// DEV.o_max [double] = nel*dim
// DEV.o_hashMin [double] = dim
// DEV.o_hashFac [double] = dim
// DEV.ou_offset [int] = hash_data size is at-msot (nel*(p+1)^dim).
// DEV.lagcoeff [double] = (p+1)
// DEV.gll1d [double] = (p+1)

// Writing
// code (int) = npt
// elem (int)= npt
// dist (double)= npt
// ref (double)= npt*dim

// Reading
// [int/unsigned int] 3 + (nel*Q^D)
// [doubles] 3 + npt*D + (nel*Q^D)*D + 6Q + nel*D +
//           nel*D*D + nel*D + nel*D + dim + dim + Q + Q

// Writing
// [int] npt + npt
// [double] npt + npt*dim

#undef nThreads
#undef pMax
#undef rDIM2
#undef sDIM2
#undef rDIM
#undef sDIM

#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND

} // namespace mfem

#endif //ifdef MFEM_USE_GSLIB
