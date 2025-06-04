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
#include "findptsedge_3.hpp"
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
#define rDIM 1
#define sDIM2 (sDIM*sDIM)
#define rDIM2 (rDIM*rDIM)
#define pMax 20
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
   return bxyz;
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
   out->oldr = p->r;
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
      double v0 = fabs(p->r - p->oldr);
      out->tr = v0/4.0;
      out->dist2 = p->dist2;
      out->r = p->oldr;
      out->flags = p->flags>>3;
      out->dist2p = -DBL_MAX;
      if (pred<dist2*tol)
      {
         out->flags |= CONVERGED_FLAG;
      }
      return true;
   }
}

static MFEM_HOST_DEVICE inline void newton_edge(findptsElementPoint_t *const
                                                out,
                                                const double jac[sDIM*rDIM],
                                                const double rhes,
                                                const double resid[sDIM],
                                                int flags,
                                                const findptsElementPoint_t *const p,
                                                const double tol)
{
   const double tr = p->tr;
   /* A = J^T J - resid_d H_d */
   const double A = jac[0]*jac[0]+ jac[1] * jac[1] + jac[2] * jac[2]
                    - rhes;
   /* y = J^T r */
   const double y = jac[0]*resid[0] + jac[1]*resid[1] + jac[0+2]*resid[2];

   const double oldr = p->r;
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
      nr = -1, dr = -1-oldr, new_flags = flags | 1u;
   }
   v = EVAL(dr);

   if ( (tnr = oldr+tr)<1 )
   {
      tdr = tr;
   }
   else
   {
      tnr = 1, tdr = 1-oldr, tnew_flags = flags | 2u;
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
   out->r = nr;
   out->dist2p = -v;
   out->flags = flags | new_flags | (p->flags<<5);
}

static MFEM_HOST_DEVICE void seed_j(const double *elx[sDIM],
                                    const double x[sDIM],
                                    const double *z,
                                    double *dist2,
                                    double *r,
                                    const int ir,
                                    const int pN)
{
   if (ir>=pN)
   {
      return;
   }

   double dx[sDIM];
   for (int d=0; d<sDIM; ++d)
   {
      dx[d] = x[d] - elx[d][ir];
   }
   dist2[ir] = norm2(dx);;
   r[ir] = z[ir];
}

// global memory access of element coordinates.
// Are the structs being stored in "local memory" or registers?
template<int T_D1D = 0>
static void FindPointsEdgeLocal32D_Kernel(const int npt,
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
                                          //   int *newton,
                                          double *infok,
                                          const int pN = 0)
{
#define MAX_CONST(a, b) (((a) > (b)) ? (a) : (b))
   const int MD1   = T_D1D ? T_D1D : pMax;
   const int D1D   = T_D1D ? T_D1D : pN;
   const int p_NE  = D1D;  // total nos. points in an element
   const int p_NEL = p_NE*nel; // total nos. points in all elements
   MFEM_VERIFY(MD1<=pMax, "Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D<=pMax, "Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D!=0, "Polynomial order not specified.");

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
                     double *r_temp = dist2_temp + D1D;

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
                           for (int ir=0; ir<D1D; ++ir)
                           {
                              if (dist2_temp[ir] < fpt->dist2)
                              {
                                 fpt->dist2 = dist2_temp[ir];
                                 fpt->r = r_temp[ir];
                              }
                           }
                        }
                     }
                     MFEM_SYNC_THREAD;
                  }

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
                        tmp->tr     = 1.0;
                        tmp->flags   = 0;
                        tmp->r   = fpt->r;
                     }
                  }
                  MFEM_SYNC_THREAD;

                  for (int step=0; step<50; step++)
                  {
                     switch (num_constrained(tmp->flags & FLAG_MASK))
                     {
                        case 0:    // findpt_edge
                        {
                           double *wt = r_workspace_ptr;
                           double *resid = wt + 3*D1D;
                           double *jac = resid + sDIM;
                           double *hess = jac + sDIM*rDIM;

                           findptsElementGEdge_t edge;
                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              // pointers to memory where to store the x & y coordinates of DOFS along the edge
                              for (int d=0; d<sDIM; ++d)
                              {
                                 edge.x[d] = constraint_workspace + d*D1D;
                              }
                              if (j<D1D)
                              {
                                 for (int d=0; d<sDIM; ++d)
                                 {
                                    edge.x[d][j] = elx[d][j];  // copy nodal coordinates along the constrained edge
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;

                           // compute basis function info upto 2nd derivative for tangential components
                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j<D1D)
                              {
                                 lagrange_eval_second_derivative(wt, tmp->r, j, gll1D, lagcoeff, D1D);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j<sDIM)
                              {
                                 resid[j] = tmp->x[j];
                                 jac[j] = 0.0;
                                 hess[j] = 0.0;
                                 for (int k=0; k<D1D; ++k)
                                 {
                                    resid[j] -= wt[      k]*edge.x[j][k];
                                    // wt[k] = value of the basis function
                                    jac[j]   += wt[D1D+k]*edge.x[j][k];
                                    // wt[k+D1D] = derivative of the basis in r
                                    hess[j]  += wt[2*D1D+k]*edge.x[j][k];
                                    // wt[k+2*D1D] = 2nd derivative of the basis function
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j==0)
                              {
                                 hess[3] = resid[0]*hess[0] + resid[1]*hess[1] +
                                           resid[2]*hess[2];
                              }
                           }

                           MFEM_FOREACH_THREAD(l,x,nThreads)
                           {
                              if (l==0)   // check prior step
                              {
                                 if ( !reject_prior_step_q(fpt,resid,tmp,tol) )   // check constraint
                                 {
                                    newton_edge(fpt,jac,hess[3],resid,tmp->flags&FLAG_MASK,tmp,tol);
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        }
                        case 1:   // findpts_pt
                        {
                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j==0)
                              {
                                 const int pi = point_index(tmp->flags & FLAG_MASK);
                                 const double *wt   = wtend +
                                                      pi*3*D1D;  // 3*D1D: basis function values, their derivatives and 2nd derivatives
                                 findptsElementGPT_t gpt;
                                 for (int d=0; d<sDIM; ++d)
                                 {
                                    gpt.x[d] = elx[d][pi*(D1D-1)];
                                    gpt.jac[d] = 0.0;
                                    gpt.hes[d] = 0.0;
                                    for (int k=0; k<D1D; ++k)
                                    {
                                       gpt.jac[d] += wt[D1D  +k]*elx[d][k];  // dx_d/dr
                                       gpt.hes[d] += wt[2*D1D+k]*elx[d][k];  // d2x_d/dr2
                                    }
                                 }

                                 const double *const pt_x = gpt.x,
                                                     *const jac = gpt.jac,
                                                            *const hes = gpt.hes;
                                 double resid[sDIM], steep, sr;
                                 resid[0] = fpt->x[0] - pt_x[0];
                                 resid[1] = fpt->x[1] - pt_x[1];
                                 resid[2] = fpt->x[2] - pt_x[2];
                                 steep = jac[0]*resid[0] + jac[1]*resid[1] + jac[2]*resid[2];
                                 sr = steep*tmp->r;
                                 if ( !reject_prior_step_q(fpt, resid, tmp, tol) )
                                 {
                                    if (sr<0)
                                    {
                                       // adi: hessian 4 or 3 size? compare to case 0
                                       const double rhess = resid[0]*hes[0] + resid[1]*hes[1] + resid[2]*hes[2];
                                       newton_edge( fpt, jac, rhess, resid, 0, tmp, tol );
                                    }
                                    else   // sr==0
                                    {
                                       fpt->r = tmp->r;
                                       fpt->dist2p = 0;
                                       // Bitwise OR is important to retain the setting of 1st or 2nd bit!
                                       fpt->flags = tmp->flags | CONVERGED_FLAG;
                                    }
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        } // case 1
                     } //switch

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
                     r_i[0] = fpt->r;
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

void FindPointsGSLIB::FindPointsEdgeLocal32(const Vector &point_pos,
                                            int point_pos_ordering,
                                            Array<unsigned int> &code,
                                            Array<unsigned int> &elem,
                                            Vector &ref,
                                            Vector &dist,
                                            Array<int> &newton,
                                            int npt)
{
   double dist_tol = 1e-14;

   if (npt == 0)
   {
      return;
   }
   MFEM_VERIFY(spacedim==3 && dim == 1,"Function for 3D only");
   switch (DEV.dof1d)
   {
      case 1: FindPointsEdgeLocal32D_Kernel<1>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.wtend.Read(),
                                                  DEV.bb.Read(),
                                                  DEV.h_nx,
                                                  DEV.loc_hash_min.Read(),
                                                  DEV.loc_hash_fac.Read(),
                                                  DEV.loc_hash_offset.ReadWrite(),
                                                  code.Write(), elem.Write(),
                                                  ref.Write(), dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 2: FindPointsEdgeLocal32D_Kernel<2>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.wtend.Read(),
                                                  DEV.bb.Read(),
                                                  DEV.h_nx, DEV.loc_hash_min.Read(),
                                                  DEV.loc_hash_fac.Read(),
                                                  DEV.loc_hash_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 3: FindPointsEdgeLocal32D_Kernel<3>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.wtend.Read(),
                                                  DEV.bb.Read(),
                                                  DEV.h_nx, DEV.loc_hash_min.Read(),
                                                  DEV.loc_hash_fac.Read(),
                                                  DEV.loc_hash_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 4: FindPointsEdgeLocal32D_Kernel<4>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.wtend.Read(),
                                                  DEV.bb.Read(),
                                                  DEV.h_nx,
                                                  DEV.loc_hash_min.Read(),
                                                  DEV.loc_hash_fac.Read(),
                                                  DEV.loc_hash_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 5: FindPointsEdgeLocal32D_Kernel<5>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.wtend.Read(),
                                                  DEV.bb.Read(),
                                                  DEV.h_nx,
                                                  DEV.loc_hash_min.Read(),
                                                  DEV.loc_hash_fac.Read(),
                                                  DEV.loc_hash_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  //   newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      case 6: FindPointsEdgeLocal32D_Kernel<6>(npt,
                                                  DEV.tol,
                                                  dist_tol,
                                                  point_pos.Read(),
                                                  point_pos_ordering,
                                                  gsl_mesh.Read(),
                                                  NE_split_total,
                                                  DEV.wtend.Read(),
                                                  DEV.bb.Read(),
                                                  DEV.h_nx,
                                                  DEV.loc_hash_min.Read(),
                                                  DEV.loc_hash_fac.Read(),
                                                  DEV.loc_hash_offset.ReadWrite(),
                                                  code.Write(),
                                                  elem.Write(),
                                                  ref.Write(),
                                                  dist.Write(),
                                                  DEV.gll1d.Read(),
                                                  DEV.lagcoeff.Read(),
                                                  // newton.ReadWrite(),
                                                  DEV.info.ReadWrite());
         break;
      default: FindPointsEdgeLocal32D_Kernel(npt,
                                                DEV.tol,
                                                DEV.tol,
                                                point_pos.Read(),
                                                point_pos_ordering,
                                                gsl_mesh.Read(),
                                                NE_split_total,
                                                DEV.wtend.Read(),
                                                DEV.bb.Read(),
                                                DEV.h_nx,
                                                DEV.loc_hash_min.Read(),
                                                DEV.loc_hash_fac.Read(),
                                                DEV.loc_hash_offset.ReadWrite(),
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
