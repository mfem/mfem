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

#ifndef sDIM
#define sDIM 2
#endif

#ifndef rDIM
#define rDIM 1
#endif

namespace mfem
{
#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2

/* p0 is the pointer where the value of the Lagrange polynomial is stored.
   p1 is the pointer where the value of the derivative of the Lagrange polynomial is stored.
*/
static MFEM_HOST_DEVICE inline void lagrange_eval_first_derivative( double *p0,                 // 0 to pN-1: p0, pN to 2*pN-1: p1
                                                                   double x,                   // ref. coords of the point of interest
                                                                   int i,                      // index of the Lagrange polynomial
                                                                   const double *z,            // GLL points
                                                                   const double *lagrangeCoeff,// Lagrange polynomial denominator term
                                                                   int pN )                    // number of GLL points
{
   double u0 = 1, u1 = 0;
   for (int j = 0; j < pN; ++j) {
      if (i != j) {
         double d_j = 2 * (x-z[j]); // note 2x scaling, also see lagrangecoeff calculation for the same factor
         u1 = d_j * u1 + u0;
         u0 = d_j * u0;
      }
   }
   double *p1 = p0 + pN;
   p0[i] = lagrangeCoeff[i] * u0;
   p1[i] = 2.0 * lagrangeCoeff[i] * u1;
}

/* p0 is the pointer where the value of the Lagrange polynomial is stored.
   p1 is the pointer where the value of the first derivative of the Lagrange polynomial is stored.
   p2 is the pointer where the value of the second derivative of the Lagrange polynomial is stored.
   x is the point (reference coords here!) at which the derivatives are evaluated.
   i is the index of the Lagrange polynomial.
   z is the array of GLL points.
   lagrangeCoeff: the denominator term in the Lagrange polynomial.
   pN is the number of GLL points, i.e., the number of Lagrange polynomials.
*/
static MFEM_HOST_DEVICE inline void lagrange_eval_second_derivative( double *p0,                  // 0 to pN-1: p0, pN to 2*pN-1: p1, 2*pN to 3*pN-1: p2
                                                                     double x,                    // ref. coords of the point of interest
                                                                     int i,                       // index of the Lagrange polynomial
                                                                     const double *z,             // GLL points
                                                                     const double *lagrangeCoeff, // Lagrange polynomial denominator term
                                                                     int pN)                      // number of GLL points
{
   double u0 = 1, u1 = 0, u2 = 0;
   for (int j = 0; j < pN; ++j) {
      if (i != j) {
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
   double test = 1;
   for (int d=0; d<sDIM; ++d) {
      double b_d = (x[d] - b->x[d].min) * (b->x[d].max - x[d]);
      test = test<0 ? test : b_d;
   }
   return test;
}

/* positive when given point is possibly inside given obbox b */
static MFEM_HOST_DEVICE inline double obbox_test(const obbox_t *const b,
                                                 const double x[sDIM])
{
   const double bxyz = obbox_axis_test(b,x);  // test if point is in AABB
   return bxyz;
   // if (bxyz<0)
   //   return bxyz;
   // else                         // test OBB only if inside AABB
   // {
   //    double dxyz[sDIM];
   //    for (int d=0; d<sDIM; ++d)
   //      dxyz[d] = x[d] - b->c0[d];
   //    double test = 1;
   //    for (int d=0; d<sDIM; ++d)
   //    {
   //       double rst = 0;
   //       for (int e=0; e<2; ++e)
   //         rst += b->A[d*2 + e] * dxyz[e];
   //       double brst = (rst + 1) * (1 - rst);
   //       test = test<0 ? test : brst;
   //    }
   //    return test;
   // }
}

/* Hash cell ID that contains the point x */
static MFEM_HOST_DEVICE inline int hash_index(const findptsLocalHashData_t *p,
                                              const double x[2])
{
   const int n = p->hash_n;
   int sum = 0;
   for (int d=sDIM-1; d>=0; --d) {
      sum *= n;
      int i = (int)floor((x[d] - p->bnd[d].min) * p->fac[d]);
      sum += i<0 ? 0 : (n-1 < i ? n-1 : i);
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

/* the bit structure of flags is CRR
   the C bit --- 1<<2 --- is set when the point is converged
   RR is 0 = 00b if r is unconstrained,
         1 = 01b if r is constrained at -1, i.e., rmin
         2 = 10b if r is constrained at +1, i.e., rmax
*/

// 1u<<2 = 100b, i.e., 3rd bit is set.
#define CONVERGED_FLAG (1u<<2)
// 0x prefix indicates hexadecimal, 07 = 7 in decimal, u suffix indicates unsigned int
// Used to mask flag variable to ensure only first 3 bits are used.
#define FLAG_MASK 0x07u // = 111b

/* returns 1 if r direction (the only free direction in 2D) is constrained.
   returns 1 if either 1st or 2nd bit of flags is set.
*/
static MFEM_HOST_DEVICE inline int num_constrained(const int flags)
{
   return ((flags | flags>>1) & 1u);
}

/* (x>>1)&1u = discards 1st bit, hence tests if the 2nd bit of x is set.
   (x>>2)&2u = discards first 2 bits, hence tests if the 4th bit of x is set.
   * So, if either 2nd or 4th bit of x is set, return 1, else return 0. *
   pi=0, r=-1
   pi=1, r=+1
*/
static MFEM_HOST_DEVICE inline int point_index(const int x)
{
   return ((x>>1) & 1u);
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
   out->x[0] = p->x[0];
   out->x[1] = p->x[1];
   out->oldr = p->r;
   out->dist2 = dist2;
   if (decr >= 0.01*pred) {
      if (decr >= 0.9*pred) { // very good iteration
         out->tr = p->tr*2;
      }
      else {                  // somewhat good iteration
         out->tr = p->tr;
      }
      return false;
   }
   else {
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
      if (pred < dist2*tol) {
         out->flags |= CONVERGED_FLAG;
      }
      return true;
   }
}

static MFEM_HOST_DEVICE inline void newton_edge( findptsElementPoint_t *const out,
                                                 const double jac[2],
                                                 const double rhess,
                                                 const double resid[2],
                                                 int flags,
                                                 const findptsElementPoint_t *const p,
                                                 const double tol )
{
   const double tr = p->tr;
   const double A = jac[0] * jac[0] + jac[1] * jac[1] - rhess; // A = J^T J - resid_d H_d
   const double y = jac[0]*resid[0] + jac[1]*resid[1];        // y = J^T resid

   const double oldr = p->r;
   double dr, newr, tdr, tnewr, v, tv;
   int new_flags=0, tnew_flags=0;

#define EVAL(dr) ( (dr*A - 2*y) * dr )
   /* if A is not SPD, quadratic model has no minimum */
   if (A>0) {
      dr = y/A;
      // if dr is too small, set it to 0. Required since roundoff dr could cause
      // fabs(newr)<1 to succeed when it shouldn't.
      if (fabs(dr)<tol) {
         dr=0.0;
         newr = oldr;
      }
      else {
         newr = oldr+dr;
      }
      // test if new dr and r result in an internal point based on one ref coord.
      // For such a point, tests on whether the point is on the boundary are not needed.
      // If the point is internal, we send it to newton_edge_fin for convergence checks.
      if (fabs(dr)<tr && fabs(newr)<1) {
         // std::cout << "oldr: " << oldr << ", dr: " << dr << ", fabs(dr)<tr: " << (fabs(dr)<tr) << ", tr: " << tr
         //           << ", newr: " << newr << ", fabs(newr)<1: " << (fabs(newr)<1) << std::endl;
         v = EVAL(dr);
         goto newton_edge_fin;
      }
   }

   if ((newr=oldr-tr) > -1) { // if oldr is at least tr distance away from r=-1
      dr = -tr;
   }                          // else set newr=-1 and set dr that supports this newr 
   else {
      newr = -1, dr = -1-oldr, new_flags = flags|1u;
   }
   v = EVAL(dr);

   if ((tnewr=oldr+tr) < 1) { // same as above, but for r=1; tdr is a temporary dr
      tdr = tr;
   }
   else {
      tnewr = 1, tdr = 1-oldr, tnew_flags = flags|2u;
   }
   tv = EVAL(tdr);          // new v value if dr is chosen based on r=1

   if (tv<v) {              // compare the two possible dr values, based on r=-1 and r=1
      newr = tnewr, dr = tdr, v = tv, new_flags = tnew_flags;
   }
#undef EVAL

newton_edge_fin:
   // check convergence by testing if change in r is less than tol
   if (fabs(dr)<tol) {
      // std::cout << "newton_edge_fin dr: " << dr << ", tol: " << tol << std::endl;
      new_flags |= CONVERGED_FLAG;
   }
   out->r = newr;
   out->dist2p = -v;
   out->flags = flags | new_flags | (p->flags<<3);
   // std::cout << "flags: " << flags << ", new_flags: " << new_flags << ", out->flags: " << out->flags << ", p->flags<<3: " << (p->flags<<3) << std::endl;
}

static MFEM_HOST_DEVICE void seed_j( const double *elx[sDIM],
                                     const double x[sDIM],
                                     const double *z, //GLL point locations [-1, 1]
                                     double       *dist2,
                                     double       *r,
                                     const int    ir,
                                     const int    pN )
{
   if (ir>=pN) {
      return;
   }

   double dx[sDIM];
   for (int d=0; d<sDIM; ++d) {
      dx[d] = x[d] - elx[d][ir];
   }
   dist2[ir] = DBL_MAX;
   const double dist2_rs = norm2(dx);
   if (dist2[ir]>dist2_rs) {
      dist2[ir] = dist2_rs;
      r[ir] = z[ir];
   }
}

template<int T_D1D = 0>
static void FindPointsSurfLocal2D_Kernel( const int     npt,
                                          const double  tol,
                                          const double  dist2tol,
                                          const double  *x,                  // point_pos
                                          const int     point_pos_ordering,  // default: byNodes
                                          const double  *xElemCoord,         // gsl_mesh
                                          const int     nel,                 // NE_split_total
                                          const double  *wtend,
                                          const double  *c,
                                          const double  *A,
                                          const double  *minBound,
                                          const double  *maxBound,
                                          const int     hash_n,
                                          const double  *hashMin,
                                          const double  *hashFac,
                                          int           *hashOffset,
                                          int *const    code_base,
                                          int *const    el_base,
                                          double *const r_base,
                                          double *const dist2_base,
                                          const double  *gll1D,
                                          const double  *lagcoeff,
                                          int           *newton,
                                          double        *infok,
                                          const int     pN = 0 )
{
#define MAX_CONST(a, b) (((a) > (b)) ? (a) : (b))
   const int MD1 = T_D1D ? T_D1D : 14;
   const int D1D = T_D1D ? T_D1D : pN;
   const int p_NE = D1D;
   const int p_NEL = nel*p_NE;

   MFEM_VERIFY(MD1<=14,"Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D!=0, "Polynomial order not specified.");
   // const int nThreads = MAX_CONST(2*MD1, 4);
   const int nThreads = 32;  // adi: npoints numbers can be quite big, especially for 3d cases
   // std::cout << "pN, D1D, MD1, p_NE, p_NEL: " << pN    << ", " << D1D  << ", "
   //                                            << MD1   << ", " << p_NE << ", "
   //                                            << p_NEL << "\n";

   /* A macro expansion that for
      1) CPU: expands to a standard for loop
      2) GPU: assigns thread blocks of size nThreads for npt total threads and
              then executes all statements within the loop in parallel.
   */
   mfem::forall_2D(npt, nThreads, 1, [=] MFEM_HOST_DEVICE (int i)
   {
      // i = point index, every thread is assigned work of a point
      // adi: find size1 and size2's dependence on rDIM and sDIM.
      constexpr int size1 = MAX_CONST(4, MD1+1) * (3*3 + 2*3) + 3*2*MD1 + 5;
      constexpr int size2 = MAX_CONST(MD1*MD1*6, MD1*3*3);

      // MFEM_SHARED: thread block level shared memory required for all points
      // in that block
      MFEM_SHARED findptsElementPoint_t el_pts[2];
      MFEM_SHARED double r_workspace[size1];
      MFEM_SHARED double constraint_workspace[size2];
      MFEM_SHARED int    constraint_init_t[nThreads];

      // pointers to shared memory for convenience
      findptsElementPoint_t *fpt, *tmp;
      double *r_workspace_ptr;

      fpt = el_pts + 0;
      tmp = el_pts + 1;
      r_workspace_ptr = r_workspace;

      /* MFEM_FOREACH_THREAD: a for loop that behaves slightly differently
         depending on if code is run on GPU or CPU.
         On GPU, this is a for loop whose every iteration is executed by a different thread.
         On CPU, this a normal for loop.
         Note that all statements outside of MFEM_FOREACH_THREAD are executed by
         all threads too.
      */

      // x and y coord index within point_pos for point i
      int id_x = point_pos_ordering == 0 ? i     : i*sDIM;
      int id_y = point_pos_ordering == 0 ? i+npt : i*sDIM+1;
      double x_i[2] = {x[id_x], x[id_y]};

      int *code_i = code_base + i;
      int *el_i = el_base + i;
      int *newton_i = newton + i;
      double *r_i = r_base + rDIM*i;  // ref coords. of point i
      double *dist2_i = dist2_base + i;

      //---------------- map_points_to_els --------------------
      // adi: why do this before forall? can't we just do it once?
      // should these arrays be in shared memory?
      findptsLocalHashData_t hash;
      for (int d=0; d<sDIM; ++d) { // hash data is spacedim dimensional
         hash.bnd[d].min = hashMin[d];
         hash.fac[d] = hashFac[d];
      }
      hash.hash_n = hash_n;
      hash.offset = hashOffset;
      
      const int hi = hash_index(&hash, x_i);
      const int *elp = hash.offset + hash.offset[hi];    // start of possible elements containing x_i
      const int *const ele = hash.offset + hash.offset[hi+1];  // end of possible elements containing x_i
      *code_i = CODE_NOT_FOUND;
      *dist2_i = DBL_MAX;

      // const int sdim2 = sDIM*sDIM;

      // Search through all elements that could contain x_i
      for (; elp!=ele; ++elp) {
         // NOTE: the pointer elp is being incremented, to the next index
         const int el = *elp;
         // MFEM_FOREACH_THREAD(j,x,nThreads)
         // {
         //    if (j==0) {
         //       std::cout << "xknown: " << x_i[0] << ", " << x_i[1]
         //                 << ", hi: " << hi
         //                 << ", el: " << el << std::endl;
         //    }
         // }
         // MFEM_SYNC_THREAD;

         // construct obbox_t on the fly for element el
         obbox_t box;
         for (int idx=0; idx<sDIM; ++idx) {
            // box.c0[idx] = c[spacedim*el + idx];
            box.x[idx].min = minBound[sDIM*el + idx];
            box.x[idx].max = maxBound[sDIM*el + idx];
         }
         // for (int idx = 0; idx<sdim2; ++idx) {
         //    box.A[idx] = A[sdim2 * el + idx];
         // }

         if (obbox_test(&box,x_i)>=0) {
            //------------ findpts_local ------------------
            {
               // elx is a pointer to the coordinates of the nodes in element el
               const double *elx[sDIM];
               for (int d=0; d<sDIM; d++) {
                  elx[d] = xElemCoord + d*p_NEL + el*p_NE;
               }

               //--------------- findpts_el ----------------
               {
                  MFEM_SYNC_THREAD;
                  // Initialize findptsElementPoint_t struct for point i
                  MFEM_FOREACH_THREAD(j,x,nThreads)
                  {
                     if (j==0) {
                        // fpt is MFEM_SHARED, so this initialization is shared
                        // across all threads and hence only one thread needs to
                        // do this. Same for constraint_init_t.
                        fpt->dist2 = DBL_MAX;
                        fpt->dist2p = 0;
                        fpt->tr = 1;
                     }
                     if (j<sDIM) {
                        fpt->x[j] = x_i[j];
                     }
                     constraint_init_t[j] = 0;
                  }
                  MFEM_SYNC_THREAD;
                  //---------------- seed ------------------
                  {
                     // pointers to shared memory for convenience, note r_temp's address
                     double *dist2_temp = r_workspace_ptr;
                     double *r_temp = dist2_temp + D1D;

                     // Each thread finds the closest point in the element for a
                     // specific r values.
                     // Then we minimize the distance across all threads.
                     MFEM_FOREACH_THREAD(j,x,nThreads)
                     {
                        seed_j(elx, x_i, gll1D, dist2_temp, r_temp, j, D1D);
                     }
                     MFEM_SYNC_THREAD;

                     MFEM_FOREACH_THREAD(j,x,nThreads)
                     {
                        if (j==0) {
                           for (int ir=0; ir<D1D; ++ir) {
                              if (dist2_temp[ir]<fpt->dist2) {
                                 fpt->dist2 = dist2_temp[ir];
                                 fpt->r = r_temp[ir];
                              }
                           }
                        }
                     }
                     MFEM_SYNC_THREAD;
                  } //seed done

                  // Initialize tmp struct with fpt values before starting Newton iterations
                  MFEM_FOREACH_THREAD(j,x,nThreads)
                  {
                     if (j==0) {
                        tmp->dist2 = DBL_MAX;
                        tmp->dist2p = 0;
                        tmp->tr = 1;
                        tmp->flags = 0;
                        tmp->r = fpt->r;
                     }
                     if (j<sDIM) {
                        tmp->x[j] = fpt->x[j];
                     }
                  }
                  MFEM_SYNC_THREAD;


                  for (int step=0; step<50; step++) {
                     int nc = num_constrained(tmp->flags & FLAG_MASK); // number of constrained reference directions
                     switch (nc) {
                        // r is unconstrained
                        case 0:
                        {
                           double *wt = r_workspace_ptr;    // 3*D1D: value, derivative and 2nd derivative
                           double *resid = wt + 3*D1D;      // sdim coord components, so sdim residuals
                           double *jac = resid + sDIM*rDIM; // sdim, dx/dr, dy/dr
                           double *hess = jac + sDIM*rDIM;  // 3, 2nd derivative of two phy. coords in r 

                           findptsElementGEdge_t edge;
                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              const int mask = 2u;
                              if ((constraint_init_t[j] & mask) == 0) {
                                 // pointers to memory where to store the x & y coordinates of DOFS along the edge
                                 for (int d=0; d<sDIM; ++d) {
                                    edge.x[d] = constraint_workspace + d*D1D;
                                 }
                                 if (j<pN) {
                                    for (int d=0; d<sDIM; ++d) {
                                       edge.x[d][j] = elx[d][j];  // copy nodal coordinates along the constrained edge
                                    }
                                 }
                                 constraint_init_t[j] = mask;
                              }
                           }
                           MFEM_SYNC_THREAD;

                           // compute basis function info upto 2nd derivative for tangential components
                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j<D1D) {
                                 lagrange_eval_second_derivative(wt, tmp->r, j, gll1D, lagcoeff, D1D);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j<sDIM) {
                                 resid[j] = tmp->x[j];
                                 jac[j] = 0.0;
                                 hess[j] = 0.0;
                                 for (int k=0; k<D1D; ++k) {
                                    resid[j] -= wt[      k]*edge.x[j][k]; // wt[k] = value of the basis function
                                    jac[j]   += wt[D1D  +k]*edge.x[j][k]; // wt[k+D1D] = derivative of the basis in r
                                    hess[j]  += wt[2*D1D+k]*edge.x[j][k]; // wt[k+2*D1D] = 2nd derivative of the basis function
                                 }
                                 // std::cout << "jac[" << j << "]: " << jac[j]
                                 //           << ", resid[" << j << "]: " << resid[j] << std::endl;
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j==0) {
                                 hess[2] = resid[0]*hess[0] + resid[1]*hess[1];
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j==0) {
                                 if ( !reject_prior_step_q(fpt, resid, tmp, tol) ) {
                                    newton_edge( fpt, jac, hess[2], resid, tmp->flags & FLAG_MASK, tmp, tol );
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        }
                        // r is constrained to either -1 or 1
                        case 1:
                        {
                           double *wt = r_workspace_ptr;  // 3*D1D: basis functions, their derivatives and 2nd derivatives

                           // compute basis function info upto 2nd derivative for tangential components
                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j<D1D) {
                                 lagrange_eval_second_derivative(wt, tmp->r, j, gll1D, lagcoeff, D1D);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j==0) {
                                 const int pi = point_index(tmp->flags & FLAG_MASK);
                                 findptsElementGPT_t gpt;
                                 for (int d=0; d<sDIM; ++d) {
                                    gpt.x[d] = elx[d][pi*(pN-1)];
                                    gpt.jac[d] = 0.0;
                                    gpt.hes[d] = 0.0;
                                    for (int k=0; k<D1D; ++k) {
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
                                 steep = jac[0]*resid[0] + jac[1]*resid[1];
                                 sr = steep*tmp->r;
                                 if ( !reject_prior_step_q(fpt, resid, tmp, tol) ) {
                                    if (sr<0) {
                                       // adi: hessian 4 or 3 size? compare to case 0
                                       const double rhess = resid[0]*hes[0] + resid[1]*hes[1];
                                       newton_edge( fpt, jac, rhess, resid, 0, tmp, tol );
                                    }
                                    else { // sr==0
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
                     if (fpt->flags & CONVERGED_FLAG) {
                        // std::cout << "Newton search converged: " << fpt->flags << std::endl;
                        *newton_i = step+1;
                        break;
                     }
                     MFEM_SYNC_THREAD;
                     MFEM_FOREACH_THREAD(j,x,nThreads)
                     if (j==0) {
                        *tmp = *fpt;
                     }
                     MFEM_SYNC_THREAD;
                     // MFEM_FOREACH_THREAD(j,x,nThreads)
                     // {
                     //    if (j==0) {
                     //       std::cout << "step: " << step
                     //                 << ", num_constrained: " << nc
                     //                 << ", flags: " << fpt->flags << ", " << tmp->flags
                     //                 << ", dist2: " << fpt->dist2
                     //                 << ", r: " << fpt->r
                     //                 << ", threadID: " << j << std::endl;
                     //    }
                     // }
                     // MFEM_SYNC_THREAD;
                  } //for int step<50
                  // MFEM_FOREACH_THREAD(j,x,nThreads)
                  // {
                  //    if (j==0) {
                  //       std::cout << "final flags: " << fpt->flags << ", " << tmp->flags
                  //                 << ", FLAG_MASK: " << FLAG_MASK
                  //                 << ", CONVERGED_FLAG: " << CONVERGED_FLAG
                  //                 << ", dist2: " << fpt->dist2
                  //                 << ", r: " << fpt->r
                  //                 << ", threadID: " << j << "\n" << std::endl;
                  //    }
                  // }
                  // MFEM_SYNC_THREAD;
               } //findpts_el

               // flags has to EXACTLY match CONVERGED_FLAG for the point to be considered converged
               // So cases where flags has 1st or 2nd bit set are not considered converged.
               // Important since newton_point case would lead to flags having 1st or 2nd bit set.
               bool converged_internal = ((fpt->flags&FLAG_MASK) == CONVERGED_FLAG) && (fpt->dist2<dist2tol);
               if (*code_i == CODE_NOT_FOUND || converged_internal || fpt->dist2 < *dist2_i)
               {
                  MFEM_FOREACH_THREAD(j,x,nThreads)
                  {
                     if (j == 0) {
                        *el_i = el;
                        *code_i = converged_internal ? CODE_INTERNAL : CODE_BORDER;
                        *dist2_i = fpt->dist2;
                        *r_i = fpt->r;
                     }
                  }
                  MFEM_SYNC_THREAD;
                  // std::cout << "converged_internal: " << converged_internal
                  //           << ", code: " << *code_i
                  //           << ", dist2: " << *dist2_i
                  //           << ", r: " << *r_i << "\n\n";
                  if (converged_internal) {
                     break;
                  }
               }
            } //findpts_local
         } //obbox_test
      } //elp
   }
   );
}

void FindPointsGSLIB::FindPointsSurfLocal2( const Vector &point_pos,
                                             int point_pos_ordering,
                                                   Array<int> &code,
                                                   Array<int> &elem,
                                                        Vector &ref,
                                                       Vector &dist,
                                                 Array<int> &newton,
                                                            int npt )
{
   if (npt == 0) { return; }
   MFEM_VERIFY(spacedim==2,"Function for 2D only");
   switch (DEV.dof1d)
   {
      case 3: return FindPointsSurfLocal2D_Kernel<3>(                      npt,
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
                                                      DEV.o_offset.ReadWrite(),
                                                                  code.Write(),
                                                                  elem.Write(),
                                                                   ref.Write(),
                                                                  dist.Write(),
                                                         DEV.gll1d.ReadWrite(),
                                                           DEV.lagcoeff.Read(),
                                                            newton.ReadWrite(),
                                                          DEV.info.ReadWrite() );
      default: return FindPointsSurfLocal2D_Kernel(                      npt,
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
                                                    DEV.o_offset.ReadWrite(),
                                                                code.Write(),
                                                                elem.Write(),  // element ID
                                                                 ref.Write(),  // Final Ref coords.
                                                                dist.Write(),  // Final dist from target physical point
                                                       DEV.gll1d.ReadWrite(),  // GLL points if already calculated
                                                         DEV.lagcoeff.Read(),  // Corresponding Lagrange coefficients
                                                          newton.ReadWrite(),  // Number of Newton iterations
                                                        DEV.info.ReadWrite(),  
                                                                   DEV.dof1d ); // (Max for split meshes) Polynomial order
   }
}

#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND

} // namespace mfem

#endif //ifdef MFEM_USE_GSLIB
