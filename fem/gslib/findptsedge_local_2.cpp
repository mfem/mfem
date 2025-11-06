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
#define sDIM 2
#define sDIM2 4
#define rDIM 1

struct findptsElementPoint_t
{
   double x[sDIM], r, oldr, dist2, dist2p, tr;
   int flags;
};

struct findptsElementGEdge_t
{
   double *x[sDIM];
};

struct findptsElementGPT_t
{
   double x[sDIM], jac[sDIM*rDIM], hes[sDIM*rDIM];
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
         double d_j = 2 * (x-z[j]);
         u2 = d_j * u2 + u1;
         u1 = d_j * u1 + u0;
         u0 = d_j * u0;
      }
   }
   double *p1 = p0 + pN, *p2 = p0 + 2 * pN;
   p0[i] = lCoeff[i] * u0;
   p1[i] = 2.0 * lCoeff[i] * u1;
   p2[i] = 8.0 * lCoeff[i] * u2;
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
   return b_d;       // only positive if inside
}

/* positive when given point is possibly inside given obbox b */
static MFEM_HOST_DEVICE inline double obbox_test(const obbox_t *const b,
                                                 const double x[sDIM])
{
   const double bxyz = obbox_axis_test(b,x);
   if (bxyz<0)   // test if point is in AABB
   {
      return bxyz;
   }
   else   // test OBB only if inside AABB
   {
      double dxyz[sDIM];
      for (int d=0; d<sDIM; ++d)
      {
         dxyz[d] = x[d] - b->c0[d];
      }
      double test = 1;
      for (int d=0; d<sDIM; ++d)
      {
         double rst = 0;
         for (int e=0; e<sDIM; ++e)
         {
            rst += b->A[d*2 + e] * dxyz[e];
         }
         double brst = (rst+1)*(1-rst);
         test = test<0 ? test : brst;
      }
      return test;
   }
}

/* Hash index in the hash table to the elements that possibly contain the point x */
static MFEM_HOST_DEVICE inline int hash_index(const findptsLocalHashData_t *p,
                                              const double x[2])
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

static MFEM_HOST_DEVICE inline double l2norm2(const double x[2])
{
   return x[0] * x[0] + x[1] * x[1];
}

/* the bit structure of flags is CRR
   the C bit --- 1<<2 --- is set when the point is converged
   RR is 0 = 00b if r is unconstrained,
         1 = 01b if r is constrained at -1, i.e., rmin
         2 = 10b if r is constrained at +1, i.e., rmax
*/

#define CONVERGED_FLAG (1u<<2)
#define FLAG_MASK 0x07u // = 111b

/* returns 1 if r direction (the only free direction in 2D) is constrained.
   returns 1 if either 1st or 2nd bit of flags is set.
*/
static MFEM_HOST_DEVICE inline int num_constrained(const int flags)
{
   return ((flags | flags>>1) & 1u);
}

/* pi=0, r=-1; pi=1, r=+1 */
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
   const double dist2 = l2norm2(resid);
   const double decr = p->dist2 - dist2;
   const double pred = p->dist2p;
   out->x[0] = p->x[0];
   out->x[1] = p->x[1];
   out->oldr = p->r;
   out->dist2 = dist2;
   if (decr >= 0.01*pred)
   {
      if (decr >= 0.9*pred)   // very good iteration
      {
         out->tr = p->tr*2;
      }
      else                    // somewhat good iteration
      {
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
      double v0 = fabs(p->r - p->oldr);
      out->tr = v0/4.0;
      out->dist2 = p->dist2;
      out->r = p->oldr;
      out->flags = p->flags>>3;
      out->dist2p = -DBL_MAX;
      if (pred < dist2*tol)
      {
         out->flags |= CONVERGED_FLAG;
      }
      return true;
   }
}

static MFEM_HOST_DEVICE inline void newton_edge( findptsElementPoint_t *const
                                                 out,
                                                 const double jac[2],
                                                 const double rhess,
                                                 const double resid[2],
                                                 int flags,
                                                 const findptsElementPoint_t *const p,
                                                 const double tol )
{
   const double tr = p->tr;
   const double A = jac[0] * jac[0] + jac[1] * jac[1] -
                    rhess; // A = J^T J - resid_d H_d
   const double y = jac[0]*resid[0] + jac[1]*resid[1];        // y = J^T resid

   const double oldr = p->r;
   double dr, newr, tdr, tnewr, v, tv;
   int new_flags=0, tnew_flags=0;

#define EVAL(dr) ( (dr*A - 2*y) * dr )
   if (A>0)
   {
      dr = y/A;
      if (fabs(dr)<tol)
      {
         dr=0.0;
         newr = oldr;
      }
      else
      {
         newr = oldr+dr;
      }

      if (fabs(dr)<tr && fabs(newr)<1)
      {
         v = EVAL(dr);
         goto newton_edge_fin;
      }
   }

   if ((newr=oldr-tr) > -1)
   {
      dr = -tr;
   }
   else
   {
      newr = -1, dr = -1-oldr, new_flags = flags|1u;
   }
   v = EVAL(dr);

   if ((tnewr=oldr+tr) < 1)
   {
      tdr = tr;
   }
   else
   {
      tnewr = 1, tdr = 1-oldr, tnew_flags = flags|2u;
   }
   tv = EVAL(tdr);

   if (tv<v)
   {
      newr = tnewr, dr = tdr, v = tv, new_flags = tnew_flags;
   }
#undef EVAL

newton_edge_fin:
   // check convergence by testing if change in r is less than tol
   if (fabs(dr)<tol)
   {
      new_flags |= CONVERGED_FLAG;
   }
   out->r = newr;
   out->dist2p = -v;
   out->flags = flags | new_flags | (p->flags<<3);
}

static MFEM_HOST_DEVICE void seed_j( const double *elx[sDIM],
                                     const double x[sDIM],
                                     const double *z,
                                     double       *dist2,
                                     double       *r,
                                     const int    ir,
                                     const int    pN )
{
   double dx[sDIM];
   for (int d=0; d<sDIM; ++d)
   {
      dx[d] = x[d] - elx[d][ir];
   }
   dist2[ir] = DBL_MAX;
   const double dist2_rs = l2norm2(dx);
   if (dist2[ir]>dist2_rs)
   {
      dist2[ir] = dist2_rs;
      r[ir] = z[ir];
   }
}

template<int T_D1D = 0>
static void FindPointsEdgeLocal2D_Kernel( const int     npt,
                                          const double  tol,
                                          const double  dist2tol,
                                          const double  *x,
                                          const int     point_pos_ordering,
                                          const double  *xElemCoord,
                                          const int     nel,
                                          const double  *wtend,
                                          const double  *boxinfo,
                                          const int     hash_n,
                                          const double  *hashMin,
                                          const double  *hashFac,
                                          unsigned int  *hashOffset,
                                          unsigned int *const code_base,
                                          unsigned int *const el_base,
                                          double *const r_base,
                                          double *const dist2_base,
                                          const double  *gll1D,
                                          const double  *lagcoeff,
                                          const int     pN = 0 )
{
#define MAXC(a, b) (((a) > (b)) ? (a) : (b))
   const int MD1   = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
   const int D1D   = T_D1D ? T_D1D : pN;
   const int p_NEL = nel*D1D;
   MFEM_VERIFY(MD1<=DofQuadLimits::MAX_D1D,
              "Increase Max allowable polynomial order.");
   MFEM_VERIFY(pN<=DofQuadLimits::MAX_D1D,
              "Increase Max allowable polynomial order.");
   MFEM_VERIFY(D1D!=0, "Polynomial order not specified.");
   const int nThreads = D1D*sDIM;

   mfem::forall_2D(npt, nThreads, 1, [=] MFEM_HOST_DEVICE (int i)
   {
      // 2D1D for seed, 3D1D + 7 for edge
      constexpr int size1 = 3*MD1 + 7;
      // edge coordinates = D1D*2
      constexpr int size2 = 2*MD1;
      // local element coordinates in shared memory
      constexpr int size3 = MD1*sDIM;

      MFEM_SHARED findptsElementPoint_t el_pts[2];
      MFEM_SHARED double r_workspace[size1];

      MFEM_SHARED double constraint_workspace[size2];

      MFEM_SHARED double elem_coords[MD1 <= 6 ? size3 : 1];

      double *r_workspace_ptr = r_workspace;
      findptsElementPoint_t *fpt, *tmp;
      fpt = el_pts + 0;
      tmp = el_pts + 1;

      // x and y coord index within point_pos for point i
      int id_x = point_pos_ordering == 0 ? i     : i*sDIM;
      int id_y = point_pos_ordering == 0 ? i+npt : i*sDIM+1;
      double x_i[2] = {x[id_x], x[id_y]};

      unsigned int *code_i = code_base  + i;
      double *dist2_i      = dist2_base + i;

      //---------------- map_points_to_els --------------------
      findptsLocalHashData_t hash;
      for (int d=0; d<sDIM; ++d)
      {
         hash.bnd[d].min = hashMin[d];
         hash.fac[d] = hashFac[d];
      }
      hash.hash_n = hash_n;
      hash.offset = hashOffset;

      const int hi                  = hash_index(&hash, x_i);
      const unsigned int *elp       = hash.offset + hash.offset[hi];
      const unsigned int *const ele = hash.offset + hash.offset[hi+1];
      *code_i  = CODE_NOT_FOUND;
      *dist2_i = DBL_MAX;

      for (; elp!=ele; ++elp)
      {
         const unsigned int el = *elp;

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

         if (obbox_test(&box,x_i)>=0)
         {
            //------------ findpts_local ------------------
            {
               if (MD1 <= 6)
               {
                  MFEM_FOREACH_THREAD(j,x,D1D*sDIM)
                  {
                     const int qp = j % D1D;
                     const int d = j / D1D;
                     elem_coords[qp + d*D1D] =
                           xElemCoord[qp + el*D1D + d*p_NEL];
                  }
                  MFEM_SYNC_THREAD;
               }

               const double *elx[sDIM];
               for (int d=0; d<sDIM; d++)
               {
                  elx[d] = MD1<= 6 ? &elem_coords[d*D1D] :
                           xElemCoord + d*p_NEL + el*D1D;
               }
               MFEM_SYNC_THREAD;
               //// findpts_el ////
               {
                  MFEM_FOREACH_THREAD(j,x,1)
                  {
                     fpt->dist2 = HUGE_VAL;
                     fpt->dist2p = 0;
                     fpt->tr = 1;
                  }
                  MFEM_FOREACH_THREAD(j,x,sDIM)
                  {
                     fpt->x[j] = x_i[j];
                  }
                  MFEM_SYNC_THREAD;

                  {
                     double *dist2_temp = r_workspace_ptr;
                     double *r_temp = dist2_temp + D1D;
                     MFEM_FOREACH_THREAD(j,x,D1D)
                     {
                        seed_j(elx, x_i, gll1D, dist2_temp, r_temp, j, D1D);
                     }
                     MFEM_SYNC_THREAD;

                     MFEM_FOREACH_THREAD(j,x,1)
                     {
                        for (int ir=0; ir<D1D; ++ir)
                        {
                           if (dist2_temp[ir]<fpt->dist2)
                           {
                              fpt->dist2 = dist2_temp[ir];
                              fpt->r = r_temp[ir];
                           }
                        }
                     }
                     MFEM_SYNC_THREAD;
                  } //seed done

                  // Initialize tmp struct with fpt values before starting Newton iterations
                  MFEM_FOREACH_THREAD(j,x,1)
                  {
                     tmp->dist2 = HUGE_VAL;
                     tmp->dist2p = 0;
                     tmp->tr = 1;
                     tmp->flags = 0;
                     tmp->r = fpt->r;
                  }
                  MFEM_FOREACH_THREAD(j,x,sDIM)
                  {
                     tmp->x[j] = fpt->x[j];
                  }
                  MFEM_SYNC_THREAD;


                  for (int step=0; step<50; step++)
                  {
                     int nc = num_constrained(tmp->flags & FLAG_MASK);
                     switch (nc)
                     {
                        case 0:
                        {
                           double *wt = r_workspace_ptr;
                           double *resid = wt + 3*D1D;
                           double *jac = resid + sDIM;
                           double *hess = jac + sDIM*rDIM;

                           findptsElementGEdge_t edge;
                           MFEM_FOREACH_THREAD(j,x,D1D)
                           {
                              for (int d=0; d<sDIM; ++d)
                              {
                                 edge.x[d] = constraint_workspace + d*D1D;
                                 edge.x[d][j] = elx[d][j];
                              }
                           }
                           MFEM_SYNC_THREAD;

                           // compute basis function info upto 2nd derivative
                           MFEM_FOREACH_THREAD(j,x,D1D)
                           {
                              lag_eval_second_der(wt, tmp->r, j, gll1D,
                                                  lagcoeff, D1D);
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,sDIM)
                           {
                              resid[j] = tmp->x[j];
                              jac[j] = 0.0;
                              hess[j] = 0.0;
                              for (int k=0; k<D1D; ++k)
                              {
                                 resid[j] -= wt[      k]*edge.x[j][k];
                                 jac[j]   += wt[D1D+k]*edge.x[j][k];
                                 hess[j]  += wt[2*D1D+k]*edge.x[j][k];
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,1)
                           {
                              hess[2] = resid[0]*hess[0] + resid[1]*hess[1];
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,1)
                           {
                              if (!reject_prior_step_q(fpt, resid, tmp, tol))
                              {
                                 newton_edge(fpt, jac, hess[2], resid,
                                             tmp->flags & FLAG_MASK, tmp, tol);
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        }
                        case 1:    // r is constrained to either -1 or 1
                        {
                           MFEM_FOREACH_THREAD(j,x,1)
                           {
                              const int pi = point_index(tmp->flags &
                                                            FLAG_MASK);
                              const double *wt   = wtend + pi*3*D1D;
                              findptsElementGPT_t gpt;
                              for (int d=0; d<sDIM; ++d)
                              {
                                 gpt.x[d] = elx[d][pi*(D1D-1)];
                                 gpt.jac[d] = 0.0;
                                 gpt.hes[d] = 0.0;
                                 for (int k=0; k<D1D; ++k)
                                 {
                                    gpt.jac[d] += wt[D1D  +k]*elx[d][k];
                                    gpt.hes[d] += wt[2*D1D+k]*elx[d][k];
                                 }
                              }

                              const double *const pt_x = gpt.x;
                              const double *const jac = gpt.jac;
                              const double *const hes = gpt.hes;
                              double resid[sDIM], steep, sr;
                              resid[0] = fpt->x[0] - pt_x[0];
                              resid[1] = fpt->x[1] - pt_x[1];
                              steep = jac[0]*resid[0] + jac[1]*resid[1];
                              sr = steep*tmp->r;
                              if ( !reject_prior_step_q(fpt, resid, tmp, tol) )
                              {
                                 if (sr<0)
                                 {
                                    const double rhess = resid[0]*hes[0] +
                                                         resid[1]*hes[1];
                                    newton_edge(fpt, jac, rhess,
                                                resid, 0, tmp, tol);
                                 }
                                 else // sr==0
                                 {
                                    fpt->r = tmp->r;
                                    fpt->dist2p = 0;
                                    fpt->flags = tmp->flags | CONVERGED_FLAG;
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        } // case 1
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
                  } //for int step<50
               } //findpts_el

               bool converged_internal =
                                 ((fpt->flags&FLAG_MASK) == CONVERGED_FLAG) &&
                                  (fpt->dist2<dist2tol);

               if (*code_i == CODE_NOT_FOUND || converged_internal ||
                  fpt->dist2 < *dist2_i)
               {
                  MFEM_FOREACH_THREAD(j,x,1)
                  {
                     *(el_base+i) = el;
                     *code_i = converged_internal ? CODE_INTERNAL : CODE_BORDER;
                     *dist2_i = fpt->dist2;
                     *(r_base+i) = fpt->r;
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

void FindPointsGSLIB::FindPointsEdgeLocal2( const Vector &point_pos,
                                            int point_pos_ordering,
                                            Array<unsigned int> &code,
                                            Array<unsigned int> &elem,
                                            Vector &ref,
                                            Vector &dist,
                                            Array<int> &newton,
                                            int npt )
{
   if (npt==0)
   {
      return;
   }
   MFEM_VERIFY(dim==1 && spacedim==2,"Function for 2D edges only");
   auto pp = point_pos.Read();
   auto pgslm = gsl_mesh.Read();
   auto pwt = DEV.wtend.Read();
   auto pbb = DEV.bb.Read();
   auto plhm = DEV.lh_min.Read();
   auto plhf = DEV.lh_fac.Read();
   auto plho = DEV.lh_offset.ReadWrite();
   auto pcode = code.Write();
   auto pelem = elem.Write();
   auto pref = ref.Write();
   auto pdist = dist.Write();
   auto pgll1d = DEV.gll1d.ReadWrite();
   auto plc = DEV.lagcoeff.Read();
   double dist2tol = DEV.surf_dist_tol;
   switch (DEV.dof1d)
   {
      case 2:
         return FindPointsEdgeLocal2D_Kernel<2>(
            npt, DEV.tol, dist2tol, pp, point_pos_ordering, pgslm,
            NE_split_total, pwt, pbb, DEV.lh_nx, plhm, plhf,
            plho, pcode, pelem, pref, pdist, pgll1d, plc);
      case 3:
         return FindPointsEdgeLocal2D_Kernel<3>(
            npt, DEV.tol, dist2tol, pp, point_pos_ordering, pgslm,
            NE_split_total, pwt, pbb, DEV.lh_nx, plhm, plhf,
            plho, pcode, pelem, pref, pdist, pgll1d, plc);
      case 4:
         return FindPointsEdgeLocal2D_Kernel<4>(
            npt, DEV.tol, dist2tol, pp, point_pos_ordering, pgslm,
            NE_split_total, pwt, pbb, DEV.lh_nx, plhm, plhf,
            plho, pcode, pelem, pref, pdist, pgll1d, plc);
      default:
         return FindPointsEdgeLocal2D_Kernel(
            npt, DEV.tol, dist2tol, pp, point_pos_ordering, pgslm,
            NE_split_total, pwt, pbb, DEV.lh_nx, plhm, plhf,
            plho, pcode, pelem, pref, pdist, pgll1d, plc, DEV.dof1d);
   }
}
#undef sDIM
#undef rDIM
#undef sDIM2
#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND
#else
void FindPointsGSLIB::FindPointsEdgeLocal2( const Vector &point_pos,
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
