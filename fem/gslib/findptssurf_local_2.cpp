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
   for (int d=0; d<sDIM; ++d)
   {
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

/* the bit structure of flags is CSSRR
   the C bit --- 1<<4 --- is set when the point is converged
   RR is 0 = 00b if r is unconstrained,
         1 = 01b if r is constrained at -1
         2 = 10b if r is constrained at +1
   SS is similarly for s constraints
   SSRR = smax,smin,rmax,rmin
*/

// 1u<<4 = 10000b, i.e., 5th bit is set.
#define CONVERGED_FLAG (1u<<4)
// 0x prefix indicates hexadecimal, 1f = 11111b, u suffix indicates unsigned int
// Used to mask flag to ensure only first 5 bits are used.
#define FLAG_MASK 0x1fu

/* returns the number of constrained reference directions */
static MFEM_HOST_DEVICE inline int num_constrained(const int flags)
{
   // OR the 1st and 3rd bits with the 2nd and 4th bits of flag, respectively.
   // flag hence will have 1 or (and) 3 set if direction r or (and) s is
   // constrained.
   const int y = flags | flags >> 1;
   // (y & 1u)    = 1 if 1st bit is set, i.e., r is constrained
   // (y>>2 & 1u) = 1 if 3rd bit is set, i.e., s is constrained.
   return (y&1u) + (y>>2 & 1u);
}

/* (x>>1)&1u = discards 1st bit, and tests if the result has its 1st bit set.
               Effectively tests 2nd bit of x.
   (x>>2)&2u = discards first 2 bits, and tests if the result has its 2nd bit set.
               Effectively tests 4th bit of x.
   So, if either 2nd or 4th bit of x is set, return 1, else return 0.
*/
static MFEM_HOST_DEVICE inline int point_index(const int x)
{
   return ((x>>1)&1u) | ((x>>2)&2u);
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
      pt.jac[1] += wtend[3 * r_g_wt_offset * pN + pN + j] * ELX(1, j, kidx);

      // dx/ds
      pt.jac[2] += wtend[3 * s_g_wt_offset * pN + pN + j] * ELX(0, kidx, j);

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
   out->x[0] = p->x[0];
   out->x[1] = p->x[1];
   out->oldr[0] = p->r[0];
   out->dist2 = dist2;
   if (decr >= 0.01 * pred) {
      if (decr >= 0.9 * pred) { // very good iteration
         out->tr = p->tr*2;
      }
      else {                    // somewhat good iteration
         out->tr = p->tr;
      }
      return false;
   }
   else {
      /* reject step; note: the point will pass through this routine
         again, and we set things up here so it gets classed as a
         "very good iteration" --- this doubles the trust radius,
         which is why we divide by 4 below */
      double v0 = fabs(p->r[0] - p->oldr[0]);
      out->tr = v0/4;
      out->dist2 = p->dist2;
      out->r[0] = p->oldr[0];
      out->flags = p->flags >> 5;
      out->dist2p = -DBL_MAX;
      if (pred < dist2*tol) {
         out->flags |= CONVERGED_FLAG;
      }
      return true;
   }
}

static MFEM_HOST_DEVICE inline void newton_edge(findptsElementPoint_t *const out,
                                                const double jac[4],
                                                const double rhes,
                                                const double resid[2],
                                                int flags,
                                                const findptsElementPoint_t *const p,
                                                const double tol)
{
   const double tr = p->tr;
   const double A = jac[0] * jac[0] + jac[2] * jac[2] - rhes; // A = J^T J - resid_d H_d
   const double y = jac[0]*resid[0] + jac[2]*resid[1];        // y = J^T resid

   const double oldr = p->r[0];
   double dr, nr, tdr, tnr;
   double v, tv;
   int new_flags=0, tnew_flags=0;

#define EVAL(dr) (dr*A - 2*y) * dr
   /* if A is not SPD, quadratic model has no minimum */
   if (A>0) {
      dr = y/A, nr = oldr+dr;
      if (fabs(dr)<tr && fabs(nr)<1) {
         v = EVAL(dr);
         goto newton_edge_fin;
      }
   }

   if ((nr=oldr-tr) > -1) {
      dr = -tr;
   }
   else {
      nr = -1, dr = -1-oldr, new_flags = flags|1u;
   }
   v = EVAL(dr);

   if ((tnr=oldr+tr) < 1) {
      tdr = tr;
   }
   else {
      tnr = 1, tdr = 1-oldr, tnew_flags = flags|2u;
   }
   tv = EVAL(tdr);

   if (tv<v) {
      nr = tnr, dr = tdr, v = tv, new_flags = tnew_flags;
   }
#undef EVAL

newton_edge_fin:
   /* check convergence */
   if (fabs(dr)<tol) {
      new_flags |= CONVERGED_FLAG;
   }
   out->r[0] = nr;
   out->dist2p = -v;
   out->flags = flags | new_flags | (p->flags << 5);
}

static MFEM_HOST_DEVICE void seed_j( const double *elx[sDIM],
                                     const double x[sDIM],
                                     const double *z, //GLL point locations [-1, 1]
                                     double       *dist2,
                                     double       *r[rDIM],
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
   if (dist2[ir]>dist2_rs) {  // adi: why this check and not just set dist2[ir] = dist2_rs?
      dist2[ir] = dist2_rs;
      r[0][ir] = z[ir];
   }
}

template<int T_D1D = 0>
static void FindPointsSurfLocal2D_Kernel( const int     npt,
                                          const double  tol,
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
   const int sdim2 = sDIM*sDIM;
   const int   MD1 = T_D1D ? T_D1D : 14;  // max polynomial degree = T_D1D
   const int   D1D = T_D1D ? T_D1D : pN;  // Polynomial degree = T_D1D
   const int  p_NE = D1D;
   const int p_NEL = nel*p_NE;

   MFEM_VERIFY(MD1<=14,"Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D!=0, "Polynomial order not specified.");
   // const int nThreads = MAX_CONST(2*MD1, 4);
   const int nThreads = 32;  // adi: npoints numbers can be quite big, especially for 3d cases
   std::cout << "pN, D1D, MD1, p_NE, p_NEL: " << pN    << ", " << D1D  << ", "
                                              << MD1   << ", " << p_NE << ", "
                                              << p_NEL << "\n";

   std::cout << "gll1D: ";
   for (int i=0; i<D1D; ++i) {
      std::cout << gll1D[i] << " ";
   }
   std::cout << std::endl;

   /* A macro expansion that
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
      // MFEM_FOREACH_THREAD(j,x,nThreads)
      // {
      // }
      // MFEM_SYNC_THREAD;

      // x and y coord index within point_pos for point i
      int id_x = point_pos_ordering == 0 ? i     : i*sDIM;
      int id_y = point_pos_ordering == 0 ? i+npt : i*sDIM+1;
      double x_i[2] = {x[id_x], x[id_y]};

      int     *code_i = code_base + i;
      int       *el_i = el_base + i;
      int   *newton_i = newton + i;
      double     *r_i = r_base + rDIM*i;  // ref coords. of point i
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
      
      const int         hi = hash_index(&hash, x_i);
      const int       *elp = hash.offset + hash.offset[hi];    // start of possible elements containing x_i
      const int *const ele = hash.offset + hash.offset[hi+1];  // end of possible elements containing x_i
      *code_i = CODE_NOT_FOUND;
      *dist2_i = DBL_MAX;

      // Search through all elements that could contain x_i
      for (; elp!=ele; ++elp) {
         // NOTE: elp is a pointer, which is being incremented
         const int el = *elp;

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


         MFEM_FOREACH_THREAD(j,x,nThreads)
         {
            if (j==0) {
               std::cout << "el | hash_index | obbox_xbnd | obbox_ybnd | threadID: "
                        << el << " | "
                        << hi << " | "
                        << box.x[0].min << ", " << box.x[0].max << " | "
                        << box.x[1].min << ", " << box.x[1].max << " | "
                        << j << std::endl;
            }
         }
         MFEM_SYNC_THREAD;

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
                  // Initialize findptsElementPoint_t struct for point i
                  MFEM_SYNC_THREAD;
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
                     double *r_temp[rDIM];

                     r_temp[0] = dist2_temp + D1D;

                     // Each thread finds the closest point in the element for a
                     // specific r and all s.
                     // Then we minimize the distance across all threads.
                     MFEM_FOREACH_THREAD(j,x,nThreads)
                     {
                        seed_j(elx, x_i, gll1D, dist2_temp, r_temp, j, D1D);
                     }
                     MFEM_SYNC_THREAD;

                     MFEM_FOREACH_THREAD(j,x,nThreads)
                     {
                        if (j==0) {
                           fpt->dist2 = DBL_MAX;
                           for (int ir=0; ir<D1D; ++ir) {
                              if (dist2_temp[ir]<fpt->dist2) {
                                 fpt->dist2 = dist2_temp[ir];
                                 fpt->r[0] = r_temp[0][ir];
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
                     }
                     if (j<sDIM) tmp->x[j] = fpt->x[j];
                     if (j==0)   tmp->r[0] = fpt->r[0];
                  }
                  MFEM_SYNC_THREAD;

                  MFEM_FOREACH_THREAD(j,x,nThreads)
                  {
                     if (j==0) {
                        std::cout << "el_pass | point | seeddist | seedref | threadID: "
                                  << el << " | "
                                  << fpt->x[0] << ", " << fpt->x[1] << " | "
                                  << fpt->dist2 << " | "
                                  << fpt->r[0]  << " | "
                                  << j << std::endl;
                     }
                  }
                  MFEM_SYNC_THREAD;

                  for (int step=0; step < 50; step++) {
                     std::cout << "step: " << step << std::endl;
                     // number of constrained reference directions
                     int nc = num_constrained(tmp->flags & FLAG_MASK);
                     std::cout << "num_constrained: " << nc << std::endl;
                     nc = 1;
                     switch (nc) {
                        case 1:   // the point is constrained to 1 and only 1 edge
                        {
                           double *wt = r_workspace_ptr;  // 2*D1D, value, derivative and 2nd derivative
                           double *resid = wt + 3*D1D;    // sdim coord components, so sdim residuals
                           double *jac = resid + sDIM;    // sdim*sdim, jac will be row-major
                           double *hess = jac + sDIM*sDIM;// 2, 2nd derivative of two phy. coords in r 

                           findptsElementGEdge_t edge;
                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              const int mask = 2u;
                              if ((constraint_init_t[j] & mask) == 0) {
                                 for (int d=0; d<sDIM; ++d) {
                                    edge.x[d] = constraint_workspace + d*D1D; // pointers to memory where to store the x & y coordinates of DOFS along the edge
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
                                 lagrange_eval_second_derivative(wt, tmp->r[0], j, gll1D, lagcoeff, D1D);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j<sDIM) {
                                 resid[j] = tmp->x[j];
                                 jac[2*j] = 0.0;
                                 jac[2*j + 1] = 0.0;
                                 hess[j] = 0.0;
                                 for (int k=0; k<D1D; ++k) {
                                    resid[j] -= wt[k]*edge.x[j][k];      // wt[k] = value of the basis function
                                    jac[2*j] += wt[k+D1D]*edge.x[j][k];  // wt[k+D1D] = derivative of the basis in r
                                    jac[2*j+1] += 0.0;                   // derivative of basis in s is 0
                                    hess[j] += wt[k+2*D1D]*edge.x[j][k]; // wt[k+2*D1D] = 2nd derivative of the basis function
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,nThreads)
                           {
                              if (j==0) {
                                 hess[2] = resid[0]*hess[0] + resid[1]*hess[1];  // adi: what is this?
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(l,x,nThreads)
                           {
                              if (l==0) {
                                 // check prior step //
                                 if (!reject_prior_step_q(fpt, resid, tmp, tol)) {
                                    newton_edge( fpt,
                                                 jac,
                                                 hess[2],
                                                 resid,
                                                 tmp->flags & FLAG_MASK,
                                                 tmp,
                                                 tol );
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        }
                        case 2:   // Constrained to 2 edges, i.e., at a corner
                        {
               //             MFEM_FOREACH_THREAD(j,x,nThreads)
               //             {
               //                if (j == 0)
               //                {
               //                   int de = 0;
               //                   int dn = 0;
               //                   const int pi = point_index(tmp->flags & FLAG_MASK);
               //                   const findptsElementGPT_t gpt = get_pt(elx, wtend, pi, D1D);

               //                   const double *const pt_x = gpt.x,
               //                                       *const jac = gpt.jac,
               //                                              *const hes = gpt.hes;

               //                   double resid[dim], steep[dim], sr[dim];
               //                   for (int d = 0; d < dim; ++d)
               //                   {
               //                      resid[d] = fpt->x[d] - pt_x[d];
               //                   }
               //                   steep[0] = jac[0]*resid[0] + jac[2]*resid[1],
               //                              steep[1] = jac[1]*resid[0] + jac[3]*resid[1];

               //                   sr[0] = steep[0]*tmp->r[0];
               //                   sr[1] = steep[1]*tmp->r[1];
               //                   if (!reject_prior_step_q(fpt, resid, tmp, tol))
               //                   {
               //                      if (sr[0]<0)
               //                      {
               //                         if (sr[1]<0)
               //                         {
               //                            newton_area(fpt, jac, resid, tmp, tol);
               //                         }
               //                         else
               //                         {
               //                            de=0;
               //                            dn=1;
               //                            const double rh = resid[0]*hes[de]+
               //                                              resid[1]*hes[2+de];
               //                            newton_edge(fpt, jac, rh,
               //                                        resid, de,
               //                                        dn,
               //                                        tmp->flags & FLAG_MASK,
               //                                        tmp, tol);
               //                         }
               //                      }
               //                      else if (sr[1]<0)
               //                      {
               //                         de=1;
               //                         dn=0;
               //                         const double rh = resid[0]*hes[de]+
               //                                           resid[1]*hes[2+de];
               //                         newton_edge(fpt, jac, rh,
               //                                     resid, de,
               //                                     dn,
               //                                     tmp->flags & FLAG_MASK,
               //                                     tmp, tol);
               //                      }
               //                      else
               //                      {
               //                         fpt->r[0] = tmp->r[0];
               //                         fpt->r[1] = tmp->r[1];
               //                         fpt->dist2p = 0;
               //                         fpt->flags = tmp->flags | CONVERGED_FLAG;
               //                      }
               //                   }
               //                }
               //             }
               //             MFEM_SYNC_THREAD;
                           break;
                        } // case 2
                     } //switch
                     if (fpt->flags & CONVERGED_FLAG) {
                        *newton_i = step+1;
                        break;
                     }
                     MFEM_SYNC_THREAD;
                     MFEM_FOREACH_THREAD(j,x,nThreads)
                     if (j == 0) {
                        *tmp = *fpt;
                     }
                     MFEM_SYNC_THREAD;
                  } //for int step < 50
                  MFEM_FOREACH_THREAD(j,x,nThreads)
                  {
                     if (j==0) {
                        std::cout << "el_conv | truepoint | finaldist | finalref | threadID: "
                                  << el << " | "
                                  << fpt->x[0] << ", " << fpt->x[1] << " | "
                                  << fpt->dist2 << " | "
                                  << fpt->r[0]  << " | "
                                  << j << std::endl;
                     }
                  }
                  MFEM_SYNC_THREAD;
               } //findpts_el

   //             bool converged_internal = (fpt->flags & FLAG_MASK) == CONVERGED_FLAG;
   //             if (*code_i == CODE_NOT_FOUND || converged_internal || fpt->dist2 < *dist2_i)
   //             {
   //                MFEM_FOREACH_THREAD(j,x,nThreads)
   //                {
   //                   if (j == 0)
   //                   {
   //                      *el_i = el;
   //                      *code_i = converged_internal ? CODE_INTERNAL : CODE_BORDER;
   //                      *dist2_i = fpt->dist2;
   //                   }
   //                   if (j < dim)
   //                   {
   //                      r_i[j] = fpt->r[j];
   //                   }
   //                }
   //                MFEM_SYNC_THREAD;
   //                if (converged_internal)
   //                {
   //                   break;
   //                }
   //             }
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
