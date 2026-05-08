// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BOUNDS
#define MFEM_BOUNDS

#include "../config/config.hpp"
#include "../general/forall.hpp"
#include "fespace.hpp"

namespace mfem
{

/** @name Piecewise linear bounds of bases
    \brief Piecewise linear bounds of bases can be used to compute bounds on
    the grid function in each element. The bounds for the bases are constructed
    based on the following parameters:

    (i) @b nb: number of bases/nodes in 1D (i.e. polynomial order+1),

    (ii) @b b_type: bases type, 0 - Lagrange interpolants on Gauss-Legendre
    nodes, 1 - Lagrange interpolants on Gauss-Lobatto-Legendre nodes, and
    2 - Positive/Bernstein bases on uniformly distributed nodes,

    (iii) @b ncp: number of control points used to construct the piecewise
    linear bounds

    (iv) @b cp_type: control point distribution. 0 - GL + end-points,
                    1 - Chebyshev.

    Note: @b nb and @b b_type are inferred directly from the grid-function.

    If the user does not specify @b ncp and @b cp_type, the minimum value of
    @b ncp is used that would bound the bases for the @b cp_type. We default
    to @b cp_type = 0 as it requires fewer number of points to bound the bases.
    Typically, @b ncp = 2 @b nb is sufficient to get fairly compact bounds, and
    increasing @b ncp results in tighter bounds.

    Finally, only tensor-product elements are currently supported.

    For more technical details see:
    Mittal et al., "General Field Evaluation in High-Order Meshes on GPUs" &
    Dzanic et al., "A method for bounding high-order finite element
    functions: Applications to mesh validity and bounds-preserving limiters".
*/
class PLBound
{
private:
   int nb; // #mesh nodes in 1D
   int ncp; // #control points in 1D
   int b_type; // bases type: 0 - GL, 1 - GLL, 2 - Bernstein
   int cp_type; // control points type: 0 - GL+Ends, 1 - Chebyshev
   bool proj = true; // Use linear projection to compute bounds.
   real_t tol = 0.0; // offset bounds to avoid round-off errors
   Vector nodes, weights, control_points;
   Vector xhat, what, cphat;
   DenseMatrix lbound, ubound; // ncp x nb matrices with bounds of all bases
   DenseMatrix lbound_t, ubound_t; // nb x ncp transposes for device kernel
   // Some auxillary storage for computing the bounds with Bernstein
   DenseMatrix basisMatNodes; // Bernstein bases at equispaced nodes
   DenseMatrix basisMatInt;   // Bernstein bases at GLL nodes
   Vector nodes_int, weights_int; // Integration nodes and weights
   DenseMatrix basisMatLU;    // Used to compute LU factors for Bernstein
   mutable Array<int> lu_ip;

   // stores min_ncp for nb = 2..12 for Lagrange interpolants on GL nodes
   // with GL+end points and Chebyshev points as control points
   static constexpr int min_ncp_gl_x[2][11]= {{3,5,6,8,9,10,11,11,12,13,14},
      {3,5,8,9,11,12,14,15,17,18,20}
   };

   // stores min_ncp for nb = 2..12 for Lagrange interpolants on GLL nodes
   // with GL+end points and Chebyshev points as control points
   static constexpr int min_ncp_gll_x[2][11]= {{3,5,7,8,9,10,12,13,14,15,16},
      {3,5,8,10,12,13,15,17,19,21,22}
   };

   // stores min_ncp for nb = 2..12 for Bernstein bases with GL+end points
   // and Chebyshev points as control points
   static constexpr int min_ncp_pos_x[2][11]= {{3,5,7,8,8,9,10,10,11,12,13},
      {3,5,8,9,11,12,13,13,14,15,16}
   };

   /// Helper function to extract lower or upper bounding matrix
   DenseMatrix GetBoundingMatrix(int dim, bool is_lower) const;

public:
   // Constructor
   PLBound(const int nb_i, const int ncp_i, const int b_type_i,
           const int cp_type_i, const real_t tol_i)
   {
      Setup(nb_i, ncp_i, b_type_i, cp_type_i, tol_i);
   }

   // Constructor
   PLBound(const FiniteElementSpace *fes,
           const int ncp_i = -1, const int cp_type_i = 0);

   /// Get minimum number of control points needed to bound the given bases
   int GetMinimumPointsForGivenBases(int nb_i, int b_type_i,
                                     int cp_type_i) const;

   /// Print information about the bounds
   void Print(std::ostream &outp = mfem::out) const;

   /** @brief Enable (default) or disable linear projection before bounding.
    *
    *  @details This projection increases the computational cost but results in
    *  tighter bounds.
    */
   void SetProjectionFlagForBounding(bool proj_)
   {
      proj = proj_;
   }

   /** @brief Compute piecewise linear bounds for the lexicographically-ordered
    *  nodal coefficients in @a coeff in 1D/2D/3D.
    *
    *  @param[in] rdim    The spatial dimension of the element (1, 2, or 3).
    *  @param[in] coeff   The vector of lexicographically-ordered coefficients.
    *                     Should be of size nb^rdim, where nb is the number of
    *                     bases/nodes in 1D. These coefficients must correspond
    *                     to the bases type and number of bases, used in the
    *                     constructor of PLBound.
    *
    *  @param[out] intmin The vector of minimum bound for all control points.
    *  @param[out] intmax The vector of maximum bound for all control points.
    *                     Both intmin and intmax are of size ncp^rdim, where
    *                     ncp is the number of control points in 1D, and are
    *                     ordered lexicographically.
    */
   void GetNDBounds(const int rdim, const Vector &coeff,
                    Vector &intmin, Vector &intmax) const;

   /// Get number of control points used to compute the bounds.
   int GetNControlPoints() const { return ncp; }

   /// Get the underlying 1D basis type.
   int GetBasisType() const { return b_type; }

   /// Get 1D control point locations (lexicographic order) in [0,1].
   const Vector &GetControlPoints() const { return control_points; }

   /** @brief Compute element-wise bounds from a lexicographic E-vector.
    *
    *  @details The expected layout of @a e_vec is `ND x VDIM x NE`, where
    *  `ND = nb^rdim`, `VDIM = fes_vdim`, and `NE` is the number of elements.
    *  The output layout matches GridFunction::GetElementBounds:
    *  `NE x active_vdim`, with the element index varying fastest.
    */
   void GetElementBoundsKernel(const int rdim, const int fes_vdim,
                               const Vector &e_vec, Vector &lower,
                               Vector &upper, const int vdim = 0) const;

   /** @brief Get lower and upper bounding matrix (ncp^dim x nb^dim)
    *
    *  @details The matrices can be used to compute the bounds at control points
    *           by a simple matrix-vector product with the
    *           lexicographically-ordered nodal coefficients.
    *           The resulting output is also lexicographically-ordered.
    *
    *  @note These matrices do not account for the linear projection step that
    *        is optionally done in GetNDBounds before bounding the function.
    */
   ///@{
   DenseMatrix GetLowerBoundMatrix(int dim = 1) const;
   DenseMatrix GetUpperBoundMatrix(int dim = 1) const;
   ///@}

private:
   /** @brief Compute piecewise linear bounds for the lexicographically-ordered
    *  nodal coefficients in @a coeff in 1D.
    *  See GetNDBounds for details of the input and output parameters.
    */
   void Get1DBounds(const Vector &coeff, Vector &intmin, Vector &intmax) const;

   /** @brief Compute piecewise linear bounds for the lexicographically-ordered
    *  nodal coefficients in @a coeff in 2D.
    *  See GetNDBounds for details of the input and output parameters.
    */
   void Get2DBounds(const Vector &coeff, Vector &intmin, Vector &intmax) const;

   /** @brief Compute piecewise linear bounds for the lexicographically-ordered
    *  nodal coefficients in @a coeff in 3D.
    *  See GetNDBounds for details of the input and output parameters.
    */
   void Get3DBounds(const Vector &coeff, Vector &intmin, Vector &intmax) const;

   /** @brief Setup matrix used to compute values at given 1D locations in [0,1]
    *  for Bernstein bases.
    */
   void SetupBernsteinBasisMat(DenseMatrix &basisMat, Vector &nodesBern) const;

   void Setup(const int nb_i, const int ncp_i, const int b_type_i,
              const int cp_type_i, const real_t tol_i);
};

namespace internal
{

struct PLBoundDeviceData
{
   int nb;
   int ncp;
   bool proj;
   const real_t *xhat;
   const real_t *what;
   const real_t *cphat;
   const real_t *lbound;
   const real_t *ubound;
};

template<int T_NB = 0>
inline void GetElementBoundsKernel1D(const PLBoundDeviceData &data,
                                     const int fes_vdim,
                                     const int ne,
                                     const Vector &e_vec,
                                     Vector &lower,
                                     Vector &upper,
                                     const int comp0,
                                     const int ncomp)
{
   constexpr int GENERIC_MAX_ND = 32;
   constexpr int MAX_ND = T_NB ? T_NB : GENERIC_MAX_ND;
   constexpr int BLOCK_X = 2*MAX_ND;

   const int nd = T_NB ? T_NB : data.nb;
   MFEM_VERIFY(nd <= MAX_ND,
               "Device element bounds kernel supports up to 32 "
               "1D degrees of freedom.");

   const auto E = Reshape(e_vec.Read(), nd, fes_vdim, ne);
   auto L = Reshape(lower.Write(), ne, ncomp);
   auto U = Reshape(upper.Write(), ne, ncomp);

   mfem::forall_2D<BLOCK_X>(ne*ncomp, BLOCK_X, 1,
                            [=] MFEM_HOST_DEVICE (int ec)
   {
      const int e = ec % ne;
      const int c = ec / ne;
      const int vc = comp0 + c;
      const real_t *coeff = &E(0, vc, e);
      const int tid = MFEM_THREAD_ID(x);

      MFEM_SHARED real_t sproj[MAX_ND];
      MFEM_SHARED real_t ssum0[MAX_ND];
      MFEM_SHARED real_t ssum1[MAX_ND];
      MFEM_SHARED real_t smin[BLOCK_X];
      MFEM_SHARED real_t smax[BLOCK_X];
      MFEM_SHARED real_t sa0;
      MFEM_SHARED real_t sa1;

      MFEM_FOREACH_THREAD(i, x, nd)
      {
         if (data.proj)
         {
            const real_t x = data.xhat[i];
            const real_t w = data.what[i];
            ssum0[i] = 0.5*coeff[i]*w;
            ssum1[i] = 1.5*coeff[i]*w*x;
         }
         else
         {
            ssum0[i] = 0.0;
            ssum1[i] = 0.0;
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(ii, x, 1)
      {
         sa0 = 0.0;
         sa1 = 0.0;
         for (int i = 0; i < nd; i++)
         {
            sa0 += ssum0[i];
            sa1 += ssum1[i];
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(i, x, nd)
      {
         if (data.proj)
         {
            const real_t x = data.xhat[i];
            sproj[i] = coeff[i] - sa0 - sa1*x;
         }
         else
         {
            sproj[i] = coeff[i];
         }
      }
      MFEM_SYNC_THREAD;

      real_t lower_local = HUGE_VAL;
      real_t upper_local = -HUGE_VAL;
      MFEM_FOREACH_THREAD(j, x, data.ncp)
      {
         real_t lo = 0.0;
         real_t hi = 0.0;
         if (data.proj)
         {
            const real_t xcp = data.cphat[j];
            lo = sa0 + sa1*xcp;
            hi = lo;
         }

         for (int i = 0; i < nd; i++)
         {
            const real_t val = sproj[i];
            const real_t lv = data.lbound[j + i*data.ncp]*val;
            const real_t uv = data.ubound[j + i*data.ncp]*val;
            lo += lv < uv ? lv : uv;
            hi += lv > uv ? lv : uv;
         }
         lower_local = lower_local < lo ? lower_local : lo;
         upper_local = upper_local > hi ? upper_local : hi;
      }

      smin[tid] = lower_local;
      smax[tid] = upper_local;
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(ii, x, 1)
      {
         real_t lower_ec = smin[0];
         real_t upper_ec = smax[0];
         const int nthreads = MFEM_THREAD_SIZE(x);
         const int nactive = data.ncp < nthreads ? data.ncp : nthreads;
         for (int t = 1; t < nactive; t++)
         {
            lower_ec = lower_ec < smin[t] ? lower_ec : smin[t];
            upper_ec = upper_ec > smax[t] ? upper_ec : smax[t];
         }
         L(e, c) = lower_ec;
         U(e, c) = upper_ec;
      }
   });
}

template<int T_NB = 0, int T_NCP = 0>
inline void GetElementBoundsKernel2D(const PLBoundDeviceData &data,
                                     const int fes_vdim,
                                     const int ne,
                                     const Vector &e_vec,
                                     Vector &lower,
                                     Vector &upper,
                                     const int comp0,
                                     const int ncomp)
{
   constexpr int DEFAULT_MAX_NB = 8;
   constexpr int DEFAULT_MAX_CP = 3*DEFAULT_MAX_NB;
   constexpr int MAX_NB = T_NB ? T_NB : DEFAULT_MAX_NB;
   constexpr int MAX_CP = T_NCP ? T_NCP : DEFAULT_MAX_CP;
   constexpr int MAX_THREADS = MAX_CP*MAX_CP;

   const int nb = data.nb;
   const int ncp = data.ncp;
   const int nd = nb*nb;
   MFEM_VERIFY(nb <= MAX_NB,
               "Device 2D element bounds kernel exceeds its compile-time "
               "1D degree bound.");
   MFEM_VERIFY(ncp <= MAX_CP,
               "Device 2D element bounds kernel exceeds its compile-time "
               "control-point bound.");
   MFEM_VERIFY(ncp*ncp <= MAX_THREADS,
               "Device 2D element bounds kernel exceeds its compile-time "
               "thread-block bound.");

   const auto E = Reshape(e_vec.Read(), nd, fes_vdim, ne);
   auto L = Reshape(lower.Write(), ne, ncomp);
   auto U = Reshape(upper.Write(), ne, ncomp);

   mfem::forall_2D<MAX_THREADS>(ne*ncomp, ncp, ncp,
                                [=] MFEM_HOST_DEVICE (int ec)
   {
      const int e = ec % ne;
      const int c = ec / ne;
      const int vc = comp0 + c;
      const real_t *coeff = &E(0, vc, e);
      const int tx = MFEM_THREAD_ID(x);
      const int ty = MFEM_THREAD_ID(y);

      MFEM_SHARED real_t sproj[MAX_NB*MAX_NB];
      MFEM_SHARED real_t srow_min[MAX_NB*MAX_CP];
      MFEM_SHARED real_t srow_max[MAX_NB*MAX_CP];
      MFEM_SHARED real_t srow_a0[MAX_NB];
      MFEM_SHARED real_t srow_a1[MAX_NB];
      MFEM_SHARED real_t sa0[MAX_CP];
      MFEM_SHARED real_t sa1[MAX_CP];
      MFEM_SHARED real_t smin[MAX_THREADS];
      MFEM_SHARED real_t smax[MAX_THREADS];

      // Stage 1a: for each nodal row, form the per-node contributions to the
      // row-wise linear fit used by the first 1D bounding solve.
      MFEM_FOREACH_THREAD(jrow, y, nb)
      {
         const real_t *row_coeff = coeff + jrow*nb;
         const int row_ncp_off = jrow*MAX_CP;
         MFEM_FOREACH_THREAD(i, x, nb)
         {
            if (data.proj)
            {
               const real_t x = data.xhat[i];
               const real_t w = data.what[i];
               srow_min[row_ncp_off + i] = 0.5*row_coeff[i]*w;
               srow_max[row_ncp_off + i] = 1.5*row_coeff[i]*w*x;
            }
            else
            {
               srow_min[row_ncp_off + i] = 0.0;
               srow_max[row_ncp_off + i] = 0.0;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Stage 1b: reduce the row-wise projection coefficients a0/a1.
      if (data.proj)
      {
         MFEM_FOREACH_THREAD(jrow, y, nb)
         {
            const int row_ncp_off = jrow*MAX_CP;
            real_t a0 = 0.0;
            real_t a1 = 0.0;
            MFEM_FOREACH_THREAD(ii, x, 1)
            {
               for (int i = 0; i < nb; i++)
               {
                  a0 += srow_min[row_ncp_off + i];
                  a1 += srow_max[row_ncp_off + i];
               }
               srow_a0[jrow] = a0;
               srow_a1[jrow] = a1;
            }
         }
         MFEM_SYNC_THREAD;
      }

      // Stage 1c: subtract the row-wise linear fit once and cache the
      // projected row coefficients for reuse across all x-control points.
      MFEM_FOREACH_THREAD(jrow, y, nb)
      {
         const real_t *row_coeff = coeff + jrow*nb;
         MFEM_FOREACH_THREAD(i, x, nb)
         {
            if (data.proj)
            {
               const real_t x = data.xhat[i];
               sproj[jrow*MAX_NB + i] = row_coeff[i]
                                        - srow_a0[jrow] - srow_a1[jrow]*x;
            }
            else
            {
               sproj[jrow*MAX_NB + i] = row_coeff[i];
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Stage 1d: solve the first 1D bounding problem along each nodal row and
      // store bounds at every x-direction control point.
      MFEM_FOREACH_THREAD(icp, x, ncp)
      {
         MFEM_FOREACH_THREAD(jrow, y, nb)
         {
            const int row_cp_off = jrow*ncp;
            real_t lo = 0.0;
            real_t hi = 0.0;
            if (data.proj)
            {
               const real_t xcp = data.cphat[icp];
               lo = srow_a0[jrow] + srow_a1[jrow]*xcp;
               hi = lo;
            }
            for (int i = 0; i < nb; i++)
            {
               const real_t val = sproj[jrow*MAX_NB + i];
               const real_t lv = data.lbound[icp + i*data.ncp]*val;
               const real_t uv = data.ubound[icp + i*data.ncp]*val;
               lo += lv < uv ? lv : uv;
               hi += lv > uv ? lv : uv;
            }
            srow_min[row_cp_off + icp] = lo;
            srow_max[row_cp_off + icp] = hi;
         }
      }
      MFEM_SYNC_THREAD;

      // Stage 2a: from the row bounds, form the per-row contributions to the
      // second 1D projection solve in the y-direction.
      MFEM_FOREACH_THREAD(icp, x, ncp)
      {
         MFEM_FOREACH_THREAD(jrow, y, nb)
         {
            const int row_cp_off = jrow*ncp;
            if (data.proj)
            {
               const real_t x = data.xhat[jrow];
               const real_t w = data.what[jrow];
               const real_t t = 0.5*(srow_min[row_cp_off + icp] +
                                     srow_max[row_cp_off + icp]);
               smin[row_cp_off + icp] = 0.5*t*w;
               smax[row_cp_off + icp] = 1.5*t*w*x;
            }
            else
            {
               smin[row_cp_off + icp] = 0.0;
               smax[row_cp_off + icp] = 0.0;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Stage 2b: reduce the y-direction projection coefficients for each
      // x-control-point column.
      MFEM_FOREACH_THREAD(jj, y, 1)
      {
         MFEM_FOREACH_THREAD(icp, x, ncp)
         {
            real_t a0 = 0.0;
            real_t a1 = 0.0;
            for (int jrow = 0; jrow < nb; jrow++)
            {
               a0 += smin[jrow*ncp + icp];
               a1 += smax[jrow*ncp + icp];
            }
            sa0[icp] = a0;
            sa1[icp] = a1;
         }
      }
      MFEM_SYNC_THREAD;

      // Stage 2c: subtract the y-direction linear fit from the intermediate
      // row bounds so the final tensor-product bound uses the perturbation.
      if (data.proj)
      {
         MFEM_FOREACH_THREAD(icp, x, ncp)
         {
            MFEM_FOREACH_THREAD(jrow, y, nb)
            {
               const int row_cp_off = jrow*ncp;
               const real_t x = data.xhat[jrow];
               const real_t t = sa0[icp] + sa1[icp]*x;
               srow_min[row_cp_off + icp] -= t;
               srow_max[row_cp_off + icp] -= t;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Stage 3: each thread now owns one 2D control point (icp, kcp) and
      // accumulates its final lower/upper bound from the row-bound data.
      MFEM_FOREACH_THREAD(icp, x, ncp)
      {
         MFEM_FOREACH_THREAD(kcp, y, ncp)
         {
            real_t lo = 0.0;
            real_t hi = 0.0;
            if (data.proj)
            {
               const real_t xcp = data.cphat[kcp];
               lo = sa0[icp] + sa1[icp]*xcp;
               hi = lo;
            }
            for (int jrow = 0; jrow < nb; jrow++)
            {
               const real_t w0 = srow_min[jrow*ncp + icp];
               const real_t w1 = srow_max[jrow*ncp + icp];
               const real_t lb = data.lbound[kcp + jrow*data.ncp];
               const real_t ub = data.ubound[kcp + jrow*data.ncp];
               const real_t v0 = lb*w0;
               const real_t v1 = ub*w0;
               const real_t v2 = lb*w1;
               const real_t v3 = ub*w1;
               real_t vlo = v0 < v1 ? v0 : v1;
               real_t vhi = v0 > v1 ? v0 : v1;
               vlo = vlo < v2 ? vlo : v2;
               vlo = vlo < v3 ? vlo : v3;
               vhi = vhi > v2 ? vhi : v2;
               vhi = vhi > v3 ? vhi : v3;
               lo += vlo;
               hi += vhi;
            }
            const int slot = kcp*ncp + icp;
            smin[slot] = lo;
            smax[slot] = hi;
         }
      }
      MFEM_SYNC_THREAD;

      const int lane = ty*ncp + tx;
      const int nactive = ncp*ncp;
      const int nthreads = MFEM_THREAD_SIZE(x)*MFEM_THREAD_SIZE(y);

      // Reduce all 2D control-point bounds to one lower/upper pair per
      // (element, component).
      if (nthreads == 1)
      {
         if (tx == 0 && ty == 0)
         {
            real_t lower_ec = smin[0];
            real_t upper_ec = smax[0];
            for (int t = 1; t < nactive; t++)
            {
               lower_ec = lower_ec < smin[t] ? lower_ec : smin[t];
               upper_ec = upper_ec > smax[t] ? upper_ec : smax[t];
            }
            L(e, c) = lower_ec;
            U(e, c) = upper_ec;
         }
      }
      else
      {
         for (int stride = (nactive + 1)/2; stride > 0;
              stride = (stride + 1)/2)
         {
            if (lane < stride && lane + stride < nactive)
            {
               smin[lane] = smin[lane] < smin[lane + stride] ?
                            smin[lane] : smin[lane + stride];
               smax[lane] = smax[lane] > smax[lane + stride] ?
                            smax[lane] : smax[lane + stride];
            }
            MFEM_SYNC_THREAD;
            if (stride == 1) { break; }
         }

         if (lane == 0)
         {
            L(e, c) = smin[0];
            U(e, c) = smax[0];
         }
      }
   });
}

} // namespace internal

inline void PLBound::GetElementBoundsKernel(const int rdim, const int fes_vdim,
                                            const Vector &e_vec,
                                            Vector &lower, Vector &upper,
                                            const int vdim) const
{
   MFEM_VERIFY(b_type != BasisType::Positive,
               "Bernstein device bounds are not implemented.");
   if (rdim == 3)
   {
      MFEM_ABORT("Device element bounds kernel currently only supports 1D/2D.");
   }
   MFEM_VERIFY(rdim == 1 || rdim == 2, "Invalid element dimension.");
   MFEM_VERIFY(vdim >= -1 && vdim <= fes_vdim, "Invalid vector component.");
   const int nd = static_cast<int>(std::pow(nb, rdim));
   const int ne = e_vec.Size()/(nd*fes_vdim);
   const int ncomp = (vdim > 0) ? 1 : fes_vdim;

   lower.SetSize(ne*ncomp, e_vec);
   upper.SetSize(ne*ncomp, e_vec);
   lower.UseDevice(true);
   upper.UseDevice(true);

   const real_t *dxhat = xhat.Read();
   const real_t *dwhat = what.Read();
   const real_t *dcphat = cphat.Read();
   const real_t *dlbound = lbound.Read();
   const real_t *dubound = ubound.Read();

   internal::PLBoundDeviceData data
   {
      nb,
      ncp,
      proj,
      dxhat,
      dwhat,
      dcphat,
      dlbound,
      dubound
   };

   const int comp0 = (vdim > 0) ? (vdim - 1) : 0;

   if (rdim == 1)
   {
      switch (nb)
      {
         case 2: return internal::GetElementBoundsKernel1D<2>(data, fes_vdim, ne,
                                                                 e_vec, lower, upper,
                                                                 comp0, ncomp);
         case 3: return internal::GetElementBoundsKernel1D<3>(data, fes_vdim, ne,
                                                                 e_vec, lower, upper,
                                                                 comp0, ncomp);
         case 4: return internal::GetElementBoundsKernel1D<4>(data, fes_vdim, ne,
                                                                 e_vec, lower, upper,
                                                                 comp0, ncomp);
         case 5: return internal::GetElementBoundsKernel1D<5>(data, fes_vdim, ne,
                                                                 e_vec, lower, upper,
                                                                 comp0, ncomp);
         case 6: return internal::GetElementBoundsKernel1D<6>(data, fes_vdim, ne,
                                                                 e_vec, lower, upper,
                                                                 comp0, ncomp);
         case 7: return internal::GetElementBoundsKernel1D<7>(data, fes_vdim, ne,
                                                                 e_vec, lower, upper,
                                                                 comp0, ncomp);
         case 8: return internal::GetElementBoundsKernel1D<8>(data, fes_vdim, ne,
                                                                 e_vec, lower, upper,
                                                                 comp0, ncomp);
         case 9: return internal::GetElementBoundsKernel1D<9>(data, fes_vdim, ne,
                                                                 e_vec, lower, upper,
                                                                 comp0, ncomp);
         case 10: return internal::GetElementBoundsKernel1D<10>(data, fes_vdim, ne,
                                                                   e_vec, lower, upper,
                                                                   comp0, ncomp);
         default: return internal::GetElementBoundsKernel1D<>(data, fes_vdim, ne,
                                                                 e_vec, lower, upper,
                                                                 comp0, ncomp);
      }
   }
#define MFEM_PLBOUND_2D_DISPATCH(NB, NCP) \
   return internal::GetElementBoundsKernel2D<NB, NCP>(data, fes_vdim, ne, \
                                                      e_vec, lower, upper, \
                                                      comp0, ncomp)
   switch (nb)
   {
      case 2:
         switch (ncp)
         {
            case 4: MFEM_PLBOUND_2D_DISPATCH(2, 4);
            case 6: MFEM_PLBOUND_2D_DISPATCH(2, 6);
            case 8: MFEM_PLBOUND_2D_DISPATCH(2, 8);
         }
         break;
      case 3:
         switch (ncp)
         {
            case 6: MFEM_PLBOUND_2D_DISPATCH(3, 6);
            case 9: MFEM_PLBOUND_2D_DISPATCH(3, 9);
            case 12: MFEM_PLBOUND_2D_DISPATCH(3, 12);
         }
         break;
      case 4:
         switch (ncp)
         {
            case 8: MFEM_PLBOUND_2D_DISPATCH(4, 8);
            case 12: MFEM_PLBOUND_2D_DISPATCH(4, 12);
            case 16: MFEM_PLBOUND_2D_DISPATCH(4, 16);
         }
         break;
      case 5:
         switch (ncp)
         {
            case 10: MFEM_PLBOUND_2D_DISPATCH(5, 10);
            case 15: MFEM_PLBOUND_2D_DISPATCH(5, 15);
            case 20: MFEM_PLBOUND_2D_DISPATCH(5, 20);
         }
         break;
      case 6:
         switch (ncp)
         {
            case 12: MFEM_PLBOUND_2D_DISPATCH(6, 12);
            case 18: MFEM_PLBOUND_2D_DISPATCH(6, 18);
            case 24: MFEM_PLBOUND_2D_DISPATCH(6, 24);
         }
         break;
      case 7:
         switch (ncp)
         {
            case 14: MFEM_PLBOUND_2D_DISPATCH(7, 14);
            case 21: MFEM_PLBOUND_2D_DISPATCH(7, 21);
            case 28: MFEM_PLBOUND_2D_DISPATCH(7, 28);
         }
         break;
      case 8:
         switch (ncp)
         {
            case 16: MFEM_PLBOUND_2D_DISPATCH(8, 16);
            case 24: MFEM_PLBOUND_2D_DISPATCH(8, 24);
            case 32: MFEM_PLBOUND_2D_DISPATCH(8, 32);
         }
         break;
   }
#undef MFEM_PLBOUND_2D_DISPATCH
   return internal::GetElementBoundsKernel2D<>(data, fes_vdim, ne,
                                               e_vec, lower, upper,
                                               comp0, ncomp);
}

} // namespace mfem

#endif // MFEM_BOUNDS
