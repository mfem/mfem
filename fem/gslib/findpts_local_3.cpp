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
#include "findpts_3.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../../linalg/dinvariants.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{

//findptsKernelInfo["includes"].asArray();
//    const dlong Nq = N + 1;
//  findptsKernelInfo["defines/p_D"] = 3;
//  findptsKernelInfo["defines/p_Nq"] = Nq;
//  findptsKernelInfo["defines/p_Np"] = Nq * Nq * Nq;
//  findptsKernelInfo["defines/p_nptsBlock"] = 4;

//  unsigned int Nq2 = Nq * Nq;
//  const auto blockSize = nearestPowerOfTwo(Nq2);

//  findptsKernelInfo["defines/p_blockSize"] = blockSize;
//  findptsKernelInfo["defines/p_Nfp"] = Nq * Nq;
//  findptsKernelInfo["defines/dlong"] = dlongString;
//  findptsKernelInfo["defines/hlong"] = hlongString;
//  findptsKernelInfo["defines/dfloat"] = dfloatString;
//  findptsKernelInfo["defines/DBL_MAX"] = std::numeric_limits<dfloat>::max();


#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2
#define dlong int
#define dfloat double


////// OBBOX //////

static MFEM_HOST_DEVICE inline void lagrange_eval_first_derivative(dfloat *p0,
                                                                   dfloat x, dlong i,
                                                                   dfloat *z, dfloat *lagrangeCoeff, dlong p_Nr)
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
                                                                    dfloat *z, dfloat *lagrangeCoeff, dlong p_Nr)
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
                                                      const dfloat x[3])
{
   dfloat test = 1;
   for (dlong d = 0; d < 3; ++d)
   {
      dfloat b_d = (x[d] - b->x[d].min) * (b->x[0].max - x[0]);
      test = test < 0 ? test : b_d;
   }
   return test;
}

/* positive when possibly inside */
static MFEM_HOST_DEVICE inline dfloat obbox_test(const obbox_t *const b,
                                                 const dfloat x[3])
{
   const dfloat bxyz = obbox_axis_test(b, x);
   if (bxyz < 0)
   {
      return bxyz;
   }
   else
   {
      dfloat dxyz[3];
      for (dlong d = 0; d < 3; ++d)
      {
         dxyz[d] = x[d] - b->c0[d];
      }
      dfloat test = 1;
      for (dlong d = 0; d < 3; ++d)
      {
         dfloat rst = 0;
         for (dlong e = 0; e < 3; ++e)
         {
            rst += b->A[d * 3 + e] * dxyz[e];
         }
         dfloat brst = (rst + 1) * (1 - rst);
         test = test < 0 ? test : brst;
      }
      return test;
   }
}

////// HASH //////

static MFEM_HOST_DEVICE inline dlong hash_index(const findptsLocalHashData_t *p,
                                                const dfloat x[3])
{
   const dlong n = p->hash_n;
   dlong sum = 0;
   for (dlong d = 3 - 1; d >= 0; --d)
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
static MFEM_HOST_DEVICE inline void lin_solve_3(dfloat x[3], const dfloat A[9],
                                                const dfloat y[3])
{
   const dfloat a = A[4] * A[8] - A[5] * A[7], b = A[5] * A[6] - A[3] * A[8],
                c = A[3] * A[7] - A[4] * A[6],
                idet = 1 / (A[0] * a + A[1] * b + A[2] * c);
   const dfloat inv0 = a, inv1 = A[2] * A[7] - A[1] * A[8],
                inv2 = A[1] * A[5] - A[2] * A[4], inv3 = b,
                inv4 = A[0] * A[8] - A[2] * A[6], inv5 = A[2] * A[3] - A[0] * A[5],
                inv6 = c, inv7 = A[1] * A[6] - A[0] * A[7],
                inv8 = A[0] * A[4] - A[1] * A[3];
   x[0] = idet * (inv0 * y[0] + inv1 * y[1] + inv2 * y[2]);
   x[1] = idet * (inv3 * y[0] + inv4 * y[1] + inv5 * y[2]);
   x[2] = idet * (inv6 * y[0] + inv7 * y[1] + inv8 * y[2]);
}

static MFEM_HOST_DEVICE inline void lin_solve_sym_2(dfloat x[2],
                                                    const dfloat A[3], const dfloat y[2])
{
   const dfloat idet = 1 / (A[0] * A[2] - A[1] * A[1]);
   x[0] = idet * (A[2] * y[0] - A[1] * y[1]);
   x[1] = idet * (A[0] * y[1] - A[1] * y[0]);
}

static MFEM_HOST_DEVICE inline dfloat norm2(const dfloat x[3]) { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; }

/* the bit structure of flags is CTTSSRR
   the C bit --- 1<<6 --- is set when the point is converged
   RR is 0 = 00b if r is unconstrained,
         1 = 01b if r is constrained at -1
         2 = 10b if r is constrained at +1
   SS, TT are similarly for s and t constraints
   TT is ignored, but treated as set when 3==2
*/

#define CONVERGED_FLAG (1u << 6)
#define FLAG_MASK 0x7fu

static MFEM_HOST_DEVICE inline dlong num_constrained(const dlong flags)
{
   const dlong y = flags | flags >> 1;
   return (y & 1u) + (y >> 2 & 1u) + (((3 == 2) | y >> 4) & 1u);
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

/* assumes x = 1 << i, with i < 6, returns i+1 */
static MFEM_HOST_DEVICE inline dlong which_bit(const dlong x)
{
   const dlong y = x & 7u;
   return (y - (y >> 2)) | ((x - 1) & 4u) | (x >> 4);
}

static MFEM_HOST_DEVICE inline dlong face_index(const dlong x)
{
   return which_bit(x) - 1;
}

static MFEM_HOST_DEVICE inline dlong edge_index(const dlong x)
{
   const dlong y = ~((x >> 1) | x);
   const dlong RTSR = ((x >> 1) & 1u) | ((x >> 2) & 2u) |
                      ((x >> 3) & 4u) | ((x << 2) & 8u);
   const dlong re = RTSR >> 1;
   const dlong se = 4u | RTSR >> 2;
   const dlong te = 8u | (RTSR & 3u);
   return ((0u - (y & 1u)) & re) | ((0u - ((y >> 2) & 1u)) & se) |
          ((0u - ((y >> 4) & 1u)) & te);
}

static MFEM_HOST_DEVICE inline dlong point_index(const dlong x)
{
   return ((x >> 1) & 1u) | ((x >> 2) & 2u) | ((x >> 3) & 4u);
}

// gets face info
// Must be called within an inner loop, with the final argument being the loop index
// workspace is a shared workspace
// side_init indicates the mode the workspace is set to
static MFEM_HOST_DEVICE inline findptsElementGFace_t
get_face(const dfloat *elx[3], dfloat *wtend[3], dlong fi, dfloat *workspace,
         dlong &side_init, dlong j, dlong p_Nr)
{
   const dlong dn = fi >> 1, d1 = plus_1_mod_3(dn), d2 = plus_2_mod_3(dn);
   const dlong side_n = fi & 1;
   const dlong p_Nfr = p_Nr*p_Nr;
   findptsElementGFace_t face;
   for (dlong d = 0; d < 3; ++d)
   {
      face.x[d] = workspace + d * p_Nfr;
      face.dxdn[d] = workspace + (3 + d) * p_Nfr;
   }

   const dlong mask = 1u << (fi / 2);
   if ((side_init & mask) == 0)
   {
      const dlong elx_stride[3] = {1, p_Nr, p_Nr * p_Nr};
#define ELX(d, j, k, l) elx[d][j * elx_stride[d1] + k * elx_stride[d2] + l * elx_stride[dn]]
      if (j < p_Nr)
      {
         for (dlong d = 0; d < 3; ++d)
         {
            for (dlong k = 0; k < p_Nr; ++k)
            {
               // copy first/last entries in normal direction
               face.x[d][j + k * p_Nr] = ELX(d, j, k, side_n * (p_Nr - 1));

               // tensor product between elx and the derivative in the normal direction
               dfloat sum_l = 0;
               for (dlong l = 0; l < p_Nr; ++l)
               {
                  sum_l += wtend[dn][p_Nr + l] * ELX(d, j, k, l);
               }
               face.dxdn[d][j + k * p_Nr] = sum_l;
            }
         }
      }
#undef ELX
      side_init = mask;
   }
   return face;
}

static MFEM_HOST_DEVICE inline findptsElementGEdge_t
get_edge(const dfloat *elx[3], dfloat *wtend[3], dlong ei, dfloat *workspace,
         dlong &side_init, dlong j, dlong p_Nr)
{
   findptsElementGEdge_t edge;
   const dlong de = ei >> 2, dn1 = plus_1_mod_3(de), dn2 = plus_2_mod_3(de);
   const dlong side_n1 = ei & 1, side_n2 = (ei & 2) >> 1;

   const dlong in1 = side_n1 * (p_Nr - 1), in2 = side_n2 * (p_Nr - 1);
   const dfloat *wt1 = wtend[dn1] + side_n1 * p_Nr * 3;
   const dfloat *wt2 = wtend[dn2] + side_n2 * p_Nr * 3;
   for (dlong d = 0; d < 3; ++d)
   {
      edge.x[d] = workspace + d * p_Nr;
      edge.dxdn1[d] = workspace + (3 + d) * p_Nr;
      edge.dxdn2[d] = workspace + (6 + d) * p_Nr;
      edge.d2xdn1[d] = workspace + (7 + d) * p_Nr;
      edge.d2xdn2[d] = workspace + (8 + d) * p_Nr;
   }

   const dlong mask = 8u << (ei / 2);
   if ((side_init & mask) == 0)
   {
      if (j < p_Nr)
      {
         const dlong elx_stride[3] = {1, p_Nr, p_Nr * p_Nr};
#define ELX(d, j, k, l) elx[d][j * elx_stride[de] + k * elx_stride[dn1] + l * elx_stride[dn2]]
         for (dlong d = 0; d < 3; ++d)
         {
            // copy first/last entries in normal directions
            edge.x[d][j] = ELX(d, j, in1, in2);
            // tensor product between elx (w/ first/last entries in second direction)
            // and the derivatives in the first normal direction
            dfloat sums_k[2] = {0, 0};
            for (dlong k = 0; k < p_Nr; ++k)
            {
               sums_k[0] += wt1[p_Nr + k] * ELX(d, j, k, in2);
               sums_k[1] += wt1[2 * p_Nr + k] * ELX(d, j, k, in2);
            }
            edge.dxdn1[d][j] = sums_k[0];
            edge.d2xdn1[d][j] = sums_k[1];
            // tensor product between elx (w/ first/last entries in first direction)
            // and the derivatives in the second normal direction
            sums_k[0] = 0, sums_k[1] = 0;
            for (dlong k = 0; k < p_Nr; ++k)
            {
               sums_k[0] += wt2[p_Nr + k] * ELX(d, j, in1, k);
               sums_k[1] += wt2[2 * p_Nr + k] * ELX(d, j, in1, k);
            }
            edge.dxdn2[d][j] = sums_k[0];
            edge.d2xdn2[d][j] = sums_k[1];
         }
#undef ELX
      }
      side_init = mask;
   }
   return edge;
}
static MFEM_HOST_DEVICE inline findptsElementGPT_t get_pt(const dfloat *elx[3],
                                                          dfloat *wtend[3], dlong pi, dlong p_Nr)
{
   const dlong side_n1 = pi & 1, side_n2 = (pi >> 1) & 1, side_n3 = (pi >> 2) & 1;
   const dlong in1 = side_n1 * (p_Nr - 1), in2 = side_n2 * (p_Nr - 1),
               in3 = side_n3 * (p_Nr - 1);
   const dlong hes_stride = (3 + 1) * 3 / 2;
   findptsElementGPT_t pt;

#define ELX(d, j, k, l) elx[d][j + k * p_Nr + l * p_Nr * p_Nr]
   for (dlong d = 0; d < 3; ++d)
   {
      pt.x[d] = ELX(d, side_n1 * (p_Nr - 1), side_n2 * (p_Nr - 1),
                    side_n3 * (p_Nr - 1));

      dfloat *wt1 = wtend[0] + p_Nr * (1 + 3 * side_n1);
      dfloat *wt2 = wtend[1] + p_Nr * (1 + 3 * side_n2);
      dfloat *wt3 = wtend[2] + p_Nr * (1 + 3 * side_n3);

      for (dlong i = 0; i < 3; ++i)
      {
         pt.jac[3 * d + i] = 0;
      }
      for (dlong i = 0; i < hes_stride; ++i)
      {
         pt.hes[hes_stride * d + i] = 0;
      }

      for (dlong j = 0; j < p_Nr; ++j)
      {
         pt.jac[3 * d + 0] += wt1[j] * ELX(d, j, in2, in3);
         pt.hes[hes_stride * d] += wt1[p_Nr + j] * ELX(d, j, in2, in3);
      }

      const dlong hes_off = hes_stride * d + hes_stride / 2;
      for (dlong k = 0; k < p_Nr; ++k)
      {
         pt.jac[3 * d + 1] += wt2[k] * ELX(d, in1, k, in3);
         pt.hes[hes_off] += wt2[p_Nr + k] * ELX(d, in1, k, in3);
      }

      for (dlong l = 0; l < p_Nr; ++l)
      {
         pt.jac[3 * d + 2] += wt3[l] * ELX(d, in1, in2, l);
         pt.hes[hes_stride * d + 5] += wt3[p_Nr + l] * ELX(d, in1, in2, l);
      }

      for (dlong l = 0; l < p_Nr; ++l)
      {
         dfloat sum_k = 0, sum_j = 0;
         for (dlong k = 0; k < p_Nr; ++k)
         {
            sum_k += wt2[k] * ELX(d, in1, k, l);
         }
         for (dlong j = 0; j < p_Nr; ++j)
         {
            sum_j += wt1[j] * ELX(d, j, in2, l);
         }
         pt.hes[hes_stride * d + 2] += wt3[l] * sum_j;
         pt.hes[hes_stride * d + 4] += wt3[l] * sum_k;
      }
      for (dlong k = 0; k < p_Nr; ++k)
      {
         dfloat sum_j = 0;
         for (dlong j = 0; j < p_Nr; ++j)
         {
            sum_j += wt1[j] * ELX(d, j, k, in3);
         }
         pt.hes[hes_stride * d + 1] += wt2[k] * sum_j;
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
                                                 const dfloat resid[3],
                                                 const findptsElementPoint_t *p,
                                                 const dfloat tol)
{
   const dfloat dist2 = norm2(resid);
   const dfloat decr = p->dist2 - dist2;
   const dfloat pred = p->dist2p;
   for (dlong d = 0; d < 3; ++d)
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
      dfloat v2 = fabs(p->r[2] - p->oldr[2]);
      out->tr = (v1 > v2 ? (v0 > v1 ? v0 : v1) : (v0 > v2 ? v0 : v2)) / 4;
      out->dist2 = p->dist2;
      for (dlong d = 0; d < 3; ++d)
      {
         out->r[d] = p->oldr[d];
      }
      out->flags = p->flags >> 7;
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
static MFEM_HOST_DEVICE void newton_vol(findptsElementPoint_t *const out,
                                        const dfloat jac[9],
                                        const dfloat resid[3],
                                        const findptsElementPoint_t *const p,
                                        const dfloat tol)
{
   const dfloat tr = p->tr;
   dfloat bnd[6] = {-1, 1, -1, 1, -1, 1};
   dfloat r0[3];
   dfloat dr[3], fac;
   dlong d, mask, flags;
   r0[0] = p->r[0], r0[1] = p->r[1], r0[2] = p->r[2];

   mask = 0x3fu;
   for (d = 0; d < 3; ++d)
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

   lin_solve_3(dr, jac, resid);

   fac = 1, flags = 0;
   for (d = 0; d < 3; ++d)
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
      goto newton_vol_fin;
   }

   for (d = 0; d < 3; ++d)
   {
      dr[d] *= fac;
   }

newton_vol_face :
   {
      const dlong fi = face_index(flags);
      const dlong dn = fi >> 1, d1 = plus_1_mod_3(dn), d2 = plus_2_mod_3(dn);
      dfloat drc[2], fac = 1;
      dlong new_flags = 0;
      dfloat res[3], y[2], JtJ[3];
      res[0] = resid[0] - (jac[0] * dr[0] + jac[1] * dr[1] + jac[2] * dr[2]);
      res[1] = resid[1] - (jac[3] * dr[0] + jac[4] * dr[1] + jac[5] * dr[2]);
      res[2] = resid[2] - (jac[6] * dr[0] + jac[7] * dr[1] + jac[8] * dr[2]);
      /* y = J_u^T res */
      y[0] = jac[d1] * res[0] + jac[3 + d1] * res[1] + jac[6 + d1] * res[2];
      y[1] = jac[d2] * res[0] + jac[3 + d2] * res[1] + jac[6 + d2] * res[2];
      /* JtJ = J_u^T J_u */
      JtJ[0] = jac[d1] * jac[d1] + jac[3 + d1] * jac[3 + d1] +
               jac[6 + d1] * jac[6 +d1];
      JtJ[1] = jac[d1] * jac[d2] + jac[3 + d1] * jac[3 + d2] +
               jac[6 + d1] * jac[6 + d2];
      JtJ[2] = jac[d2] * jac[d2] + jac[3 + d2] * jac[3 + d2] +
               jac[6 + d2] * jac[6 + d2];
      lin_solve_sym_2(drc, JtJ, y);
#define CHECK_CONSTRAINT(drcd, d3)                                                                           \
{                                                                                                            \
const dfloat rz = r0[d3] + dr[d3], lb = bnd[2 * d3], ub = bnd[2 * d3 + 1];                                   \
const dfloat delta = drcd, nr = r0[d3] + (dr[d3] + delta);                                                   \
if ((nr - lb) * (ub - nr) < 0) {                                                                             \
if (nr < lb) {                                                                                               \
dfloat f = (lb - rz) / delta;                                                                                \
if (f < fac)                                                                                                 \
fac = f, new_flags = 1u << (2 * d3);                                                                         \
}                                                                                                            \
else {                                                                                                       \
dfloat f = (ub - rz) / delta;                                                                                \
if (f < fac)                                                                                                 \
fac = f, new_flags = 2u << (2 * d3);                                                                         \
}                                                                                                            \
}                                                                                                            \
}
      CHECK_CONSTRAINT(drc[0], d1);
      CHECK_CONSTRAINT(drc[1], d2);
      dr[d1] += fac * drc[0], dr[d2] += fac * drc[1];
      if (new_flags == 0)
      {
         goto newton_vol_fin;
      }
      flags |= new_flags;
   }

newton_vol_edge :
   {
      const dlong ei = edge_index(flags);
      const dlong de = ei >> 2;
      dfloat fac = 1;
      dlong new_flags = 0;
      dfloat res[3], y, JtJ, drc;
      res[0] = resid[0] - (jac[0] * dr[0] + jac[1] * dr[1] + jac[2] * dr[2]);
      res[1] = resid[1] - (jac[3] * dr[0] + jac[4] * dr[1] + jac[5] * dr[2]);
      res[2] = resid[2] - (jac[6] * dr[0] + jac[7] * dr[1] + jac[8] * dr[2]);
      /* y = J_u^T res */
      y = jac[de] * res[0] + jac[3 + de] * res[1] + jac[6 + de] * res[2];
      /* JtJ = J_u^T J_u */
      JtJ = jac[de] * jac[de] + jac[3 + de] * jac[3 + de] +
            jac[6 + de] * jac[6 + de];
      drc = y / JtJ;
      CHECK_CONSTRAINT(drc, de);
#undef CHECK_CONSTRAINT
      dr[de] += fac * drc;
      flags |= new_flags;
      goto newton_vol_relax;
   }

   /* check and possibly relax constraints */
newton_vol_relax :
   {
      const dlong old_flags = flags;
      dfloat res[3], y[3];
      /* res := res_0 - J dr */
      res[0] = resid[0] - (jac[0] * dr[0] + jac[1] * dr[1] + jac[2] * dr[2]);
      res[1] = resid[1] - (jac[3] * dr[0] + jac[4] * dr[1] + jac[5] * dr[2]);
      res[2] = resid[2] - (jac[6] * dr[0] + jac[7] * dr[1] + jac[8] * dr[2]);
      /* y := J^T res */
      y[0] = jac[0] * res[0] + jac[3] * res[1] + jac[6] * res[2];
      y[1] = jac[1] * res[0] + jac[4] * res[1] + jac[7] * res[2];
      y[2] = jac[2] * res[0] + jac[5] * res[1] + jac[8] * res[2];
      for (dlong d = 0; d < 3; ++d)
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
   if (fabs(dr[0]) + fabs(dr[1]) + fabs(dr[2]) < tol)
   {
      flags |= CONVERGED_FLAG;
   }
   {
      const dfloat res0 = resid[0] - (jac[0] * dr[0] + jac[1] * dr[1] +
                                      jac[2] * dr[2]);
      const dfloat res1 = resid[1] - (jac[3] * dr[0] + jac[4] * dr[1] +
                                      jac[5] * dr[2]);
      const dfloat res2 = resid[2] - (jac[6] * dr[0] + jac[7] * dr[1] +
                                      jac[8] * dr[2]);
      out->dist2p = resid[0] * resid[0] + resid[1] * resid[1] +
                    resid[2] * resid[2] -
                    (res0 * res0 + res1 * res1 + res2 * res2);
   }
   for (dlong d = 0; d < 3; ++d)
   {
      dlong f = flags >> (2 * d) & 3u;
      out->r[d] = f == 0 ? r0[d] + dr[d] : (f == 1 ? -1 : 1);
   }
   out->flags = flags | (p->flags << 7);
}
static MFEM_HOST_DEVICE void newton_face(findptsElementPoint_t *const out,
                                         const dfloat jac[9],
                                         const dfloat rhes[3],
                                         const dfloat resid[3],
                                         const dlong d1,
                                         const dlong d2,
                                         const dlong dn,
                                         const dlong flags,
                                         const findptsElementPoint_t *const p,
                                         const dfloat tol)
{
   const dfloat tr = p->tr;
   dfloat bnd[4];
   dfloat r[2], dr[2] = {0, 0};
   dlong mask, new_flags;
   dfloat v, tv;
   dlong i;
   dfloat A[3], y[2], r0[2];
   /* A = J^T J - resid_d H_d */
   A[0] = jac[d1] * jac[d1] + jac[3 + d1] * jac[3 + d1] +
          jac[6 + d1] * jac[6 + d1] - rhes[0];
   A[1] = jac[d1] * jac[d2] + jac[3 + d1] * jac[3 + d2] +
          jac[6 + d1] * jac[6 + d2] - rhes[1];
   A[2] = jac[d2] * jac[d2] + jac[3 + d2] * jac[3 + d2] +
          jac[6 + d2] * jac[6 + d2] - rhes[2];
   /* y = J^T r */
   y[0] = jac[d1] * resid[0] + jac[3 + d1] * resid[1] + jac[6 + d1] * resid[2];
   y[1] = jac[d2] * resid[0] + jac[3 + d2] * resid[1] + jac[6 + d2] * resid[2];
   r0[0] = p->r[d1], r0[1] = p->r[d2];

   new_flags = flags;
   mask = 0x3fu;
   if (r0[0] - tr > -1)
   {
      bnd[0] = -tr, mask ^= 1u;
   }
   else
   {
      bnd[0] = -1 - r0[0];
   }
   if (r0[0] + tr < 1)
   {
      bnd[1] = tr, mask ^= 2u;
   }
   else
   {
      bnd[1] = 1 - r0[0];
   }
   if (r0[1] - tr > -1)
   {
      bnd[2] = -tr, mask ^= 1u << 2;
   }
   else
   {
      bnd[2] = -1 - r0[1];
   }
   if (r0[1] + tr < 1)
   {
      bnd[3] = tr, mask ^= 2u << 2;
   }
   else
   {
      bnd[3] = 1 - r0[1];
   }

   if (A[0] + A[2] <= 0 || A[0] * A[2] <= A[1] * A[1])
   {
      goto newton_face_constrained;
   }
   lin_solve_sym_2(dr, A, y);

#define EVAL(r, s) -(y[0] * r + y[1] * s) + (r * A[0] * r + (2 * r * A[1] + s * A[2]) * s) / 2
   if ((dr[0] - bnd[0]) * (bnd[1] - dr[0]) >= 0 &&
       (dr[1] - bnd[2]) * (bnd[3] - dr[1]) >= 0)
   {
      r[0] = r0[0] + dr[0], r[1] = r0[1] + dr[1];
      v = EVAL(dr[0], dr[1]);
      goto newton_face_fin;
   }
newton_face_constrained:
   v = EVAL(bnd[0], bnd[2]);
   i = 1u | (1u << 2);
   tv = EVAL(bnd[1], bnd[2]);
   if (tv < v)
   {
      v = tv, i = 2u | (1u << 2);
   }
   tv = EVAL(bnd[0], bnd[3]);
   if (tv < v)
   {
      v = tv, i = 1u | (2u << 2);
   }
   tv = EVAL(bnd[1], bnd[3]);
   if (tv < v)
   {
      v = tv, i = 2u | (2u << 2);
   }
   if (A[0] > 0)
   {
      dfloat drc;
      drc = (y[0] - A[1] * bnd[2]) / A[0];
      if ((drc - bnd[0]) * (bnd[1] - drc) >= 0 && (tv = EVAL(drc, bnd[2])) < v)
      {
         v = tv, i = 1u << 2, dr[0] = drc;
      }
      drc = (y[0] - A[1] * bnd[3]) / A[0];
      if ((drc - bnd[0]) * (bnd[1] - drc) >= 0 && (tv = EVAL(drc, bnd[3])) < v)
      {
         v = tv, i = 2u << 2, dr[0] = drc;
      }
   }
   if (A[2] > 0)
   {
      dfloat drc;
      drc = (y[1] - A[1] * bnd[0]) / A[2];
      if ((drc - bnd[2]) * (bnd[3] - drc) >= 0 && (tv = EVAL(bnd[0], drc)) < v)
      {
         v = tv, i = 1u, dr[1] = drc;
      }
      drc = (y[1] - A[1] * bnd[1]) / A[2];
      if ((drc - bnd[2]) * (bnd[3] - drc) >= 0 && (tv = EVAL(bnd[1], drc)) < v)
      {
         v = tv, i = 2u, dr[1] = drc;
      }
   }
#undef EVAL

   {
      dlong dir[2];
      dir[0] = d1;
      dir[1] = d2;
      for (dlong d = 0; d < 2; ++d)
      {
         const dlong f = i >> (2 * d) & 3u;
         if (f == 0)
         {
            r[d] = r0[d] + dr[d];
         }
         else
         {
            if ((f & (mask >> (2 * d))) == 0)
            {
               r[d] = r0[d] + (f == 1 ? -tr : tr);
            }
            else
            {
               r[d] = (f == 1 ? -1 : 1), new_flags |= f << (2 * dir[d]);
            }
         }
      }
   }
newton_face_fin:
   out->dist2p = -2 * v;
   dr[0] = r[0] - p->r[d1];
   dr[1] = r[1] - p->r[d2];
   if (fabs(dr[0]) + fabs(dr[1]) < tol)
   {
      new_flags |= CONVERGED_FLAG;
   }
   out->r[dn] = p->r[dn], out->r[d1] = r[0], out->r[d2] = r[1];
   out->flags = new_flags | (p->flags << 7);
}

static MFEM_HOST_DEVICE inline void newton_edge(findptsElementPoint_t *const
                                                out,
                                                const dfloat jac[9],
                                                const dfloat rhes,
                                                const dfloat resid[3],
                                                const dlong de,
                                                const dlong dn1,
                                                const dlong dn2,
                                                dlong flags,
                                                const findptsElementPoint_t *const p,
                                                const dfloat tol)
{
   const dfloat tr = p->tr;
   /* A = J^T J - resid_d H_d */
   const dfloat A = jac[de] * jac[de] + jac[3 + de] * jac[3 + de] + jac[6 + de] *
                    jac[6 + de] - rhes;
   /* y = J^T r */
   const dfloat y = jac[de] * resid[0] + jac[3 + de] * resid[1] + jac[6 + de] *
                    resid[2];

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
   out->r[dn1] = p->r[dn1];
   out->r[dn2] = p->r[dn2];
   out->dist2p = -v;
   out->flags = flags | new_flags | (p->flags << 7);
}

static MFEM_HOST_DEVICE void seed_j(const dfloat *elx[3],
                                    const dfloat x[3],
                                    dfloat *z, //GLL point locations [-1, 1]
                                    dfloat *dist2,
                                    dfloat *r[3],
                                    const int j,
                                    const dlong p_Nr)
{
   if (j >= p_Nr)
   {
      return;
   }

   dist2[j] = DBL_MAX;

   dfloat zr = z[j];
   for (dlong l = 0; l < p_Nr; ++l)
   {
      const dfloat zt = z[l];
      for (dlong k = 0; k < p_Nr; ++k)
      {
         dfloat zs = z[k];

         const dlong jkl = j + k * p_Nr + l * p_Nr * p_Nr;
         dfloat dx[3];
         for (dlong d = 0; d < 3; ++d)
         {
            dx[d] = x[d] - elx[d][jkl];
         }
         const dfloat dist2_jkl = norm2(dx);
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

static MFEM_HOST_DEVICE dfloat tensor_ig3_j(dfloat *g_partials,
                                            const dfloat *Jr,
                                            const dfloat *Dr,
                                            const dfloat *Js,
                                            const dfloat *Ds,
                                            const dfloat *Jt,
                                            const dfloat *Dt,
                                            const dfloat *u,
                                            const dlong j,
                                            const dlong p_Nr)
{
   dfloat uJtJs = 0.0;
   dfloat uDtJs = 0.0;
   dfloat uJtDs = 0.0;
   for (dlong k = 0; k < p_Nr; ++k)
   {
      dfloat uJt = 0.0;
      dfloat uDt = 0.0;
      for (dlong l = 0; l < p_Nr; ++l)
      {
         uJt += u[j + k * p_Nr + l * p_Nr * p_Nr] * Jt[l];
         uDt += u[j + k * p_Nr + l * p_Nr * p_Nr] * Dt[l];
      }

      uJtJs += uJt * Js[k];
      uJtDs += uJt * Ds[k];
      uDtJs += uDt * Js[k];
   }

   g_partials[0] = uJtJs * Dr[j];
   g_partials[1] = uJtDs * Jr[j];
   g_partials[2] = uDtJs * Jr[j];
   return uJtJs * Jr[j];
}

static void FindPointsLocal3D_Kernel(const int npt,
                                     const dfloat tol,
                                     const dfloat *x,
                                     const dlong point_pos_ordering,
                                     const dfloat *xElemCoord,
                                     const dfloat *yElemCoord,
                                     const dfloat *zElemCoord,
                                     dfloat *wtend_x,
                                     dfloat *wtend_y,
                                     dfloat *wtend_z,
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
                                     dfloat *gll1D,
                                     double *lagcoeff,
                                     dfloat *infok,
                                     const dlong p_Nr)
{
#define MAX_CONST(a, b) (((a) > (b)) ? (a) : (b))
#define p_innerSize 32
   const int dim = 3;
   const dlong p_NE = p_Nr*p_Nr*p_Nr;
   const int p_Nr_Max = 8;
   //    mfem::forall_1D(npt, p_innerSize, [=] MFEM_HOST_DEVICE (int i)
   mfem::forall_2D(npt, p_innerSize, 1, [=] MFEM_HOST_DEVICE (int i)
                   //   mfem::forall(npt, [=] MFEM_HOST_DEVICE (int i)
   {
      constexpr int size1 = MAX_CONST(4,
                                      p_Nr_Max + 1) * (3 * 3 + 2 * 3) + 3 * 2 * p_Nr_Max + 5;
      constexpr int size2 = MAX_CONST(p_Nr_Max *p_Nr_Max * 6, p_Nr_Max * 3 * 3);
      MFEM_SHARED dfloat r_workspace[size1];
      MFEM_SHARED findptsElementPoint_t el_pts[2];

      MFEM_SHARED dfloat constraint_workspace[size2];
      //   infok[0] = size1;
      //   infok[1] = size2;
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
      dlong id_z = point_pos_ordering == 0 ? i+2*npt : i*dim+2;
      dfloat x_i[3] = {x[id_x], x[id_y], x[id_z]};
      dfloat *wtend[3] = {&wtend_x[0], &wtend_y[0], &wtend_z[0]};

      dlong *code_i = code_base + i;
      dlong *el_i = el_base + i;
      dfloat *r_i = r_base + 3 * i;
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

         for (int idx = 0; idx < 3; ++idx)
         {
            box.c0[idx] = c[3 * el + idx];
            box.x[idx].min = minBound[3 * el + idx];
            box.x[idx].max = maxBound[3 * el + idx];
         }

         for (int idx = 0; idx < 9; ++idx)
         {
            box.A[idx] = A[9 * el + idx];
         }

         if (obbox_test(&box, x_i) >= 0)
         {
            //// findpts_local ////
            {
               const dfloat *elx[3];

               elx[0] = xElemCoord + el * p_NE;
               elx[1] = yElemCoord + el * p_NE;
               elx[2] = zElemCoord + el * p_NE;

               //// findpts_el ////
               {
                  MFEM_SYNC_THREAD;
                  MFEM_FOREACH_THREAD(j,x,p_innerSize)
                  //                  for (dlong j = 0; j < p_innerSize; ++j) //inner
                  {
                     if (j == 0)
                     {
                        fpt->dist2 = DBL_MAX;
                        fpt->dist2p = 0;
                        fpt->tr = 1;
                     }
                     if (j < 3) { fpt->x[j] = x_i[j]; }
                     constraint_init_t[j] = 0;
                  }
                  MFEM_SYNC_THREAD;
                  //// seed ////
                  {
                     dfloat *dist2_temp = r_workspace_ptr;
                     dfloat *r_temp[3];
                     for (dlong d = 0; d < 3; ++d)
                     {
                        r_temp[d] = dist2_temp + (1 + d) * p_Nr;
                     }

                     MFEM_FOREACH_THREAD(j,x,p_innerSize)
                     //                     for (dlong j = 0; j < p_innerSize; ++j)  //inner
                     {
                        seed_j(elx, x_i, gll1D, dist2_temp, r_temp, j, p_Nr);
                        //seed_j(elx, x_i, dist2_temp, r_temp, j);
                     }
                     MFEM_SYNC_THREAD;

                     MFEM_FOREACH_THREAD(j,x,p_innerSize)
                     //                     for (dlong j = 0; j < p_innerSize; ++j) //inner
                     {
                        if (j == 0)
                        {
                           fpt->dist2 = DBL_MAX;
                           for (dlong jj = 0; jj < p_Nr; ++jj)
                           {
                              if (dist2_temp[jj] < fpt->dist2)
                              {
                                 fpt->dist2 = dist2_temp[jj];
                                 for (dlong d = 0; d < 3; ++d)
                                 {
                                    fpt->r[d] = r_temp[d][jj];
                                 }
                              }
                           }
                        }
                     }
                     MFEM_SYNC_THREAD;
                  } //seed done


                  MFEM_FOREACH_THREAD(j,x,p_innerSize)
                  //                  for (dlong j = 0; j < p_innerSize; ++j)  //inner
                  {
                     if (j == 0)
                     {
                        tmp->dist2 = DBL_MAX;
                        tmp->dist2p = 0;
                        tmp->tr = 1;
                        tmp->flags = 0;
                     }
                     if (j < 3)
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
                        case 0:   // findpt_vol
                        {
                           // need 3 dimensions to have a volume
                           dfloat *wtr = r_workspace_ptr;
                           dfloat *wts = wtr + 2 * p_Nr;
                           dfloat *wtt = wts + 2 * p_Nr;

                           dfloat *resid = wtt + 2 * p_Nr;
                           dfloat *jac = resid + 3;
                           dfloat *resid_temp = jac + 9;
                           dfloat *jac_temp = resid_temp + 3 * p_Nr;

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           //                           for (dlong j = 0; j < p_innerSize; ++j)
                           {
                              if (j < p_Nr)
                              {
                                 lagrange_eval_first_derivative(wtr, tmp->r[0], j, gll1D, lagcoeff, p_Nr);
                                 lagrange_eval_first_derivative(wts, tmp->r[1], j, gll1D, lagcoeff, p_Nr);
                                 lagrange_eval_first_derivative(wtt, tmp->r[2], j, gll1D, lagcoeff, p_Nr);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           //                           for (dlong j = 0; j < p_innerSize; ++j)
                           {
                              if (j < p_Nr * 3)
                              {
                                 const int qp = j / 3;
                                 const int d = j % 3;
                                 resid_temp[d + qp * 3] = tensor_ig3_j(jac_temp + 3 * d + 9 * qp,
                                                                       wtr,
                                                                       wtr + p_Nr,
                                                                       wts,
                                                                       wts + p_Nr,
                                                                       wtt,
                                                                       wtt + p_Nr,
                                                                       elx[d],
                                                                       qp,
                                                                       p_Nr);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(l,x,p_innerSize)
                           //                           for (dlong l = 0; l < p_innerSize; ++l) //inner
                           {
                              if (l < 3)
                              {
                                 resid[l] = tmp->x[l];
                                 for (dlong j = 0; j < p_Nr; ++j)
                                 {
                                    resid[l] -= resid_temp[l + j * 3];
                                 }
                              }
                              if (l < 9)
                              {
                                 jac[l] = 0;
                                 for (dlong j = 0; j < p_Nr; ++j)
                                 {
                                    jac[l] += jac_temp[l + j * 9];
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
                                    newton_vol(fpt, jac, resid, tmp, tol);
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        } //case 0
                        case 1:   // findpt_face / findpt_area
                        {
                           const dlong fi = face_index(tmp->flags & FLAG_MASK);
                           const dlong dn = fi >> 1;
                           const dlong d1 = plus_1_mod_3(dn), d2 = plus_2_mod_3(dn);

                           dfloat *wt1 = r_workspace_ptr;
                           dfloat *wt2 = wt1 + 3 * p_Nr;
                           dfloat *resid = wt2 + 3 * p_Nr;
                           dfloat *jac = resid + 3;
                           dfloat *resid_temp = jac + 3 * 3;
                           dfloat *jac_temp = resid_temp + 3 * p_Nr;
                           dfloat *hes = jac_temp + 3 * 3 * p_Nr;
                           dfloat *hes_temp = hes + 3;
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           //                           for (dlong j = 0; j < p_innerSize; ++j)
                           {
                              if (j < p_Nr)
                              {
                                 lagrange_eval_second_derivative(wt1, tmp->r[d1], j, gll1D, lagcoeff, p_Nr);
                                 lagrange_eval_second_derivative(wt2, tmp->r[d2], j, gll1D, lagcoeff, p_Nr);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           dfloat *J1 = wt1, *D1 = wt1 + p_Nr;
                           dfloat *J2 = wt2, *D2 = wt2 + p_Nr;
                           dfloat *DD1 = D1 + p_Nr, *DD2 = D2 + p_Nr;
                           findptsElementGFace_t face;

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           //                           for (dlong j = 0; j < p_innerSize; j++)
                           {
                              //                              dlong constraint_init_dummy = constraint_init;
                              face = get_face(elx, wtend, fi, constraint_workspace, constraint_init_t[j], j,
                                              p_Nr);
                              //                              if (j==p_innerSize-1)
                              //                              {
                              //                                 constraint_init = constraint_init_dummy;
                              //                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           //                           for (dlong j = 0; j < p_innerSize; ++j)
                           {
                              if (j < p_Nr * 3)
                              {
                                 const int d = j % 3;
                                 const int qp = j / 3;
                                 const dfloat *u = face.x[d];
                                 const dfloat *du = face.dxdn[d];
                                 dfloat sums_k[4] = {0.0, 0.0, 0.0, 0.0};
                                 for (dlong k = 0; k < p_Nr; ++k)
                                 {
                                    sums_k[0] += u[qp + k * p_Nr] * J2[k];
                                    sums_k[1] += u[qp + k * p_Nr] * D2[k];
                                    sums_k[2] += u[qp + k * p_Nr] * DD2[k];
                                    sums_k[3] += du[qp + k * p_Nr] * J2[k];
                                 }

                                 resid_temp[3 * qp + d] = sums_k[0] * J1[qp];
                                 jac_temp[3 * 3 * qp + 3 * d + d1] = sums_k[0] * D1[qp];
                                 jac_temp[3 * 3 * qp + 3 * d + d2] = sums_k[1] * J1[qp];
                                 jac_temp[3 * 3 * qp + 3 * d + dn] = sums_k[3] * J1[qp];
                                 if (d == 0)
                                 {
                                    hes_temp[3 * qp] = sums_k[0] * DD1[qp];
                                    hes_temp[3 * qp + 1] = sums_k[1] * D1[qp];
                                    hes_temp[3 * qp + 2] = sums_k[2] * J1[qp];
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(l,x,p_innerSize)
                           //                           for (dlong l = 0; l < p_innerSize; ++l)
                           {
                              if (l < 3)
                              {
                                 resid[l] = fpt->x[l];
                                 for (dlong j = 0; j < p_Nr; ++j)
                                 {
                                    resid[l] -= resid_temp[l + j * 3];
                                 }
                              }

                              if (l < 3 * 3)
                              {
                                 jac[l] = 0;
                                 for (dlong j = 0; j < p_Nr; ++j)
                                 {
                                    jac[l] += jac_temp[l + j * 3 * 3];
                                 }
                              }
                              if (l < 3)
                              {
                                 hes[l] = 0;
                                 for (dlong j = 0; j < p_Nr; ++j)
                                 {
                                    hes[l] += hes_temp[l + 3 * j];
                                 }
                                 hes[l] *= resid[l];
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
                                    const dfloat steep =
                                       resid[0] * jac[dn] + resid[1] * jac[3 + dn] + resid[2] * jac[6 + dn];
                                    if (steep * tmp->r[dn] < 0)
                                    {
                                       // relax constraint //
                                       newton_vol(fpt, jac, resid, tmp, tol);
                                    }
                                    else
                                    {
                                       newton_face(fpt, jac, hes, resid, d1, d2, dn, tmp->flags & FLAG_MASK, tmp, tol);
                                    }
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        }
                        case 2:   // findpt_edge
                        {
                           const dlong ei = edge_index(tmp->flags & FLAG_MASK);
                           const dlong de = ei >> 2, dn1 = plus_1_mod_3(de), dn2 = plus_2_mod_3(de);
                           dlong d_j[3];
                           d_j[0] = de;
                           d_j[1] = dn1;
                           d_j[2] = dn2;
                           const dlong hes_count = 2 * 3 - 1; // 3=3 ? 5 : 1;

                           dfloat *wt = r_workspace_ptr;
                           dfloat *resid = wt + 3 * p_Nr;
                           dfloat *jac = resid + 3;
                           dfloat *hes_T = jac + 3 * 3;
                           dfloat *hes = hes_T + hes_count * 3;
                           findptsElementGEdge_t edge;

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           //                           for (dlong j = 0; j < p_innerSize; ++j)
                           {
                              //                              dlong constraint_init_dummy = constraint_init;
                              edge = get_edge(elx, wtend, ei, constraint_workspace, constraint_init_t[j], j,
                                              p_Nr);
                              //                              if (j==p_innerSize-1)
                              //                              {
                              //                                 constraint_init = constraint_init_dummy;
                              //                              }
                           }
                           MFEM_SYNC_THREAD;

                           const dfloat *const *e_x[3 + 3] =
                           {edge.x, edge.x, edge.dxdn1, edge.dxdn2, edge.d2xdn1, edge.d2xdn2};

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           //                           for (dlong j = 0; j < p_innerSize; ++j)
                           {
                              if (j < p_Nr)
                              {
                                 lagrange_eval_second_derivative(wt, tmp->r[de], j, gll1D, lagcoeff, p_Nr);
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           //                           for (dlong j = 0; j < p_innerSize; ++j)
                           {
                              const dlong d = j % 3;
                              const dlong row = j / 3;
                              if (j < (3 + 1) * 3)
                              {
                                 // resid and jac_T
                                 // [0, 1, 0, 0]
                                 dfloat *wt_j = wt + (row == 1 ? p_Nr : 0);
                                 const dfloat *x = e_x[row][d];
                                 dfloat sum = 0.0;
                                 for (dlong k = 0; k < p_Nr; ++k)
                                 {
                                    // resid+3 == jac_T
                                    sum += wt_j[k] * x[k];
                                 }
                                 if (j < 3)
                                 {
                                    resid[j] = tmp->x[j] - sum;
                                 }
                                 else
                                 {
                                    jac[d * 3 + d_j[row - 1]] = sum;
                                 }
                              }

                              if (j < hes_count * 3)
                              {
                                 // Hes_T is transposed version (i.e. in col major)

                                 // n1*[2, 1, 1, 0, 0]
                                 // j==1 => wt_j = wt+n1
                                 dfloat *wt_j = wt + p_Nr * (2 - (row + 1) / 2);
                                 const dfloat *x = e_x[row + 1][d];
                                 hes_T[j] = 0.0;
                                 for (dlong k = 0; k < p_Nr; ++k)
                                 {
                                    hes_T[j] += wt_j[k] * x[k];
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           //                           for (dlong j = 0; j < p_innerSize; ++j)
                           {
                              if (j < hes_count)
                              {
                                 hes[j] = 0.0;
                                 for (dlong d = 0; d < 3; ++d)
                                 {
                                    hes[j] += resid[d] * hes_T[j * 3 + d];
                                 }
                              }
                           }

                           MFEM_SYNC_THREAD;

                           MFEM_FOREACH_THREAD(l,x,p_innerSize)
                           //                           for (dlong l = 0; l < p_innerSize; ++l)
                           {
                              if (l == 0)
                              {
                                 // check prior step //
                                 if (!reject_prior_step_q(fpt, resid, tmp, tol))
                                 {
                                    // check constraint //
                                    dfloat steep[3 - 1];
                                    for (dlong k = 0; k < 3 - 1; ++k)
                                    {
                                       dlong dn = d_j[k + 1];
                                       steep[k] = 0;
                                       for (dlong d = 0; d < 3; ++d)
                                       {
                                          steep[k] += jac[dn + d * 3] * resid[d];
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
                                          dfloat rh[3];
                                          rh[0] = hes[0], rh[1] = hes[1], rh[2] = hes[3];
                                          newton_face(fpt,
                                                      jac,
                                                      rh,
                                                      resid,
                                                      de,
                                                      dn1,
                                                      dn2,
                                                      tmp->flags & (3u << (dn2 * 2)),
                                                      tmp,
                                                      tol);
                                       }
                                    }
                                    else
                                    {
                                       if (steep[1] < 0)
                                       {
                                          dfloat rh[3];
                                          rh[0] = hes[4], rh[1] = hes[2], rh[2] = hes[0];
                                          newton_face(fpt,
                                                      jac,
                                                      rh,
                                                      resid,
                                                      dn2,
                                                      de,
                                                      dn1,
                                                      tmp->flags & (3u << (dn1 * 2)),
                                                      tmp,
                                                      tol);
                                       }
                                       else
                                       {
                                          newton_edge(fpt,
                                                      jac,
                                                      hes[0],
                                                      resid,
                                                      de,
                                                      dn1,
                                                      dn2,
                                                      tmp->flags & FLAG_MASK,
                                                      tmp,
                                                      tol);
                                       }
                                    }
                                 }
                              }
                           }
                           MFEM_SYNC_THREAD;
                           break;
                        }
                        case 3:   // findpts_pt
                        {
                           MFEM_FOREACH_THREAD(j,x,p_innerSize)
                           //                           for (dlong j = 0; j < p_innerSize; ++j)
                           {
                              if (j == 0)
                              {
                                 const dlong pi = point_index(tmp->flags & FLAG_MASK);
                                 const findptsElementGPT_t gpt = get_pt(elx, wtend, pi, p_Nr);
                                 const dfloat *const pt_x = gpt.x, *const jac = gpt.jac, *const hes = gpt.hes;

                                 dfloat resid[3], steep[3];
                                 for (dlong d = 0; d < 3; ++d)
                                 {
                                    resid[d] = fpt->x[d] - pt_x[d];
                                 }
                                 if (!reject_prior_step_q(fpt, resid, tmp, tol))
                                 {
                                    for (dlong d = 0; d < 3; ++d)
                                    {
                                       steep[d] = 0;
                                       for (dlong e = 0; d < 3; ++d)
                                       {
                                          steep[d] += jac[d + e * 3] * resid[d];
                                       }
                                       steep[d] *= tmp->r[d];
                                    }
                                    dlong de, dn1, dn2, d1, d2, dn, hi0, hi1, hi2;
                                    if (steep[0] < 0)
                                    {
                                       if (steep[1] < 0)
                                       {
                                          if (steep[2] < 0)
                                          {
                                             newton_vol(fpt, jac, resid, tmp, tol);
                                          }
                                          else
                                          {
                                             d1 = 0, d2 = 1, dn = 2, hi0 = 0, hi1 = 1, hi2 = 3;
                                             dfloat rh[3];
                                             rh[0] = resid[0] * hes[hi0] + resid[1] * hes[6 + hi0] + resid[2] * hes[12 +
                                                                                                                    hi0],
                                                     rh[1] = resid[0] * hes[hi1] + resid[1] * hes[6 + hi1] + resid[2] * hes[12 +
                                                                                                                            hi1],
                                                             rh[2] = resid[0] * hes[hi2] + resid[1] * hes[6 + hi2] + resid[2] * hes[12 +
                                                                     hi2];
                                             newton_face(fpt,
                                                         jac,
                                                         rh,
                                                         resid,
                                                         d1,
                                                         d2,
                                                         dn,
                                                         (tmp->flags) & (3u << (2 * dn)),
                                                         tmp,
                                                         tol);
                                          }
                                       }
                                       else
                                       {
                                          if (steep[2] < 0)
                                          {
                                             d1 = 2, d2 = 0, dn = 1, hi0 = 5, hi1 = 2, hi2 = 0;
                                             dfloat rh[3];
                                             rh[0] = resid[0] * hes[hi0] +
                                                     resid[1] * hes[6 + hi0] +
                                                     resid[2] * hes[12 + hi0];
                                             rh[1] = resid[0] * hes[hi1] +
                                                     resid[1] * hes[6 + hi1] +
                                                     resid[2] * hes[12 + hi1];
                                             rh[2] = resid[0] * hes[hi2] +
                                                     resid[1] * hes[6 + hi2] +
                                                     resid[2] * hes[12 + hi2];
                                             newton_face(fpt,
                                                         jac,
                                                         rh,
                                                         resid,
                                                         d1,
                                                         d2,
                                                         dn,
                                                         (tmp->flags) & (3u << (2 * dn)),
                                                         tmp,
                                                         tol);
                                          }
                                          else
                                          {
                                             de = 0, dn1 = 1, dn2 = 2, hi0 = 0;
                                             const dfloat rh =
                                                resid[0] * hes[hi0] + resid[1] * hes[6 + hi0] + resid[2] * hes[12 + hi0];
                                             newton_edge(fpt,
                                                         jac,
                                                         rh,
                                                         resid,
                                                         de,
                                                         dn1,
                                                         dn2,
                                                         tmp->flags & (~(3u << (2 * de))),
                                                         tmp,
                                                         tol);
                                          }
                                       }
                                    }
                                    else
                                    {
                                       if (steep[1] < 0)
                                       {
                                          if (steep[2] < 0)
                                          {
                                             d1 = 1, d2 = 2, dn = 0, hi0 = 3, hi1 = 4, hi2 = 5;
                                             dfloat rh[3];
                                             rh[0] = resid[0] * hes[hi0] + resid[1] * hes[6 + hi0] + resid[2] * hes[12 +
                                                                                                                    hi0],
                                                     rh[1] = resid[0] * hes[hi1] + resid[1] * hes[6 + hi1] + resid[2] * hes[12 +
                                                                                                                            hi1],
                                                             rh[2] = resid[0] * hes[hi2] + resid[1] * hes[6 + hi2] + resid[2] * hes[12 +
                                                                     hi2];
                                             newton_face(fpt,
                                                         jac,
                                                         rh,
                                                         resid,
                                                         d1,
                                                         d2,
                                                         dn,
                                                         (tmp->flags) & (3u << (2 * dn)),
                                                         tmp,
                                                         tol);
                                          }
                                          else
                                          {
                                             de = 1, dn1 = 2, dn2 = 0, hi0 = 3;
                                             const dfloat rh =
                                                resid[0] * hes[hi0] + resid[1] * hes[6 + hi0] + resid[2] * hes[12 + hi0];
                                             newton_edge(fpt,
                                                         jac,
                                                         rh,
                                                         resid,
                                                         de,
                                                         dn1,
                                                         dn2,
                                                         tmp->flags & (~(3u << (2 * de))),
                                                         tmp,
                                                         tol);
                                          }
                                       }
                                       else
                                       {
                                          if (steep[2] < 0)
                                          {
                                             de = 2, dn1 = 0, dn2 = 1, hi0 = 5;
                                             const dfloat rh =
                                                resid[0] * hes[hi0] + resid[1] * hes[6 + hi0] + resid[2] * hes[12 + hi0];
                                             newton_edge(fpt,
                                                         jac,
                                                         rh,
                                                         resid,
                                                         de,
                                                         dn1,
                                                         dn2,
                                                         tmp->flags & (~(3u << (2 * de))),
                                                         tmp,
                                                         tol);
                                          }
                                          else
                                          {
                                             fpt->r[0] = tmp->r[0], fpt->r[1] = tmp->r[1], fpt->r[2] = tmp->r[2];
                                             fpt->dist2p = 0;
                                             fpt->flags = tmp->flags | CONVERGED_FLAG;
                                          }
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
                  MFEM_FOREACH_THREAD(j,x,p_innerSize)
                  //                  for (dlong j = 0; j < p_innerSize; ++j)
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

void FindPointsGSLIB::FindPointsLocal(const Vector &point_pos,
                                      int point_pos_ordering,
                                      Array<int> &gsl_code_dev_l,
                                      Array<int> &gsl_elem_dev_l,
                                      Vector &gsl_ref_l,
                                      Vector &gsl_dist_l,
                                      int npt)
{
   if (npt == 0) { return; }
   if (dim == 3)
   {
      FindPointsLocal3D_Kernel(npt, DEV.tol,
                               point_pos.Read(), point_pos_ordering,
                               DEV.o_x.ReadWrite(),
                               DEV.o_y.ReadWrite(),
                               DEV.o_z.ReadWrite(),
                               DEV.o_wtend_x.ReadWrite(),
                               DEV.o_wtend_y.ReadWrite(),
                               DEV.o_wtend_z.ReadWrite(),
                               DEV.o_c.Read(),
                               DEV.o_A.Read(),
                               DEV.o_min.Read(),
                               DEV.o_max.Read(),
                               DEV.hash_n,
                               DEV.o_hashMin.Read(),
                               DEV.o_hashFac.Read(),
                               DEV.o_offset.ReadWrite(),
                               gsl_code_dev_l.ReadWrite(),
                               gsl_elem_dev_l.ReadWrite(),
                               gsl_ref_l.ReadWrite(),
                               gsl_dist_l.ReadWrite(),
                               DEV.gll1d.ReadWrite(),
                               DEV.lagcoeff.ReadWrite(),
                               DEV.info.ReadWrite(),
                               DEV.dof1d);
   }
   else
   {
      MFEM_ABORT("Device implementation only for 3D yet.");
   }
}


#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND
#undef dlong
#undef dfloat

} // namespace mfem

#endif //ifdef MFEM_USE_GSLIB
