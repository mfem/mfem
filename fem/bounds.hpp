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
   void SetProjectionFlagForBounding(bool proj_) { proj = proj_; }

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
   const real_t *nodes;
   const real_t *weights;
   const real_t *control_points;
   const real_t *lbound_t;
   const real_t *ubound_t;

   MFEM_HOST_DEVICE inline real_t Min2(const real_t a, const real_t b) const
   {
      return (a < b) ? a : b;
   }

   MFEM_HOST_DEVICE inline real_t Max2(const real_t a, const real_t b) const
   {
      return (a > b) ? a : b;
   }

   MFEM_HOST_DEVICE inline void GetFinalBounds1D(const real_t *coeff,
                                                 real_t &lower,
                                                 real_t &upper) const
   {
      real_t a0 = 0.0;
      real_t a1 = 0.0;
      if (proj)
      {
         for (int i = 0; i < nb; i++)
         {
            const real_t x = 2.0*nodes[i] - 1.0;
            const real_t w = 2.0*weights[i];
            a0 += 0.5*coeff[i]*w;
            a1 += 1.5*coeff[i]*w*x;
         }
      }

      lower = 0.0;
      upper = 0.0;
      for (int j = 0; j < ncp; j++)
      {
         real_t lo = 0.0;
         real_t hi = 0.0;
         if (proj)
         {
            const real_t xcp = 2.0*control_points[j] - 1.0;
            lo = a0 + a1*xcp;
            hi = lo;
         }

         for (int i = 0; i < nb; i++)
         {
            real_t c = coeff[i];
            if (proj)
            {
               const real_t x = 2.0*nodes[i] - 1.0;
               c -= a0 + a1*x;
            }
            const real_t lv = lbound_t[i + j*nb]*c;
            const real_t uv = ubound_t[i + j*nb]*c;
            lo += Min2(lv, uv);
            hi += Max2(lv, uv);
         }
         if (j == 0)
         {
            lower = lo;
            upper = hi;
         }
         else
         {
            lower = Min2(lower, lo);
            upper = Max2(upper, hi);
         }
      }
   }
};

} // namespace internal

inline void PLBound::GetElementBoundsKernel(const int rdim, const int fes_vdim,
                                            const Vector &e_vec,
                                            Vector &lower, Vector &upper,
                                            const int vdim) const
{
   constexpr int BOUNDS_MAX_ND = 32;

   MFEM_VERIFY(b_type != BasisType::Positive,
               "Bernstein device bounds are not implemented.");
   if (rdim == 2 || rdim == 3)
   {
      MFEM_ABORT("Device element bounds kernel currently only supports 1D.");
   }
   MFEM_VERIFY(rdim == 1, "Invalid element dimension.");
   MFEM_VERIFY(vdim >= -1 && vdim <= fes_vdim, "Invalid vector component.");
   const int nd = nb;
   const int ne = e_vec.Size()/(nd*fes_vdim);
   const int ncomp = (vdim > 0) ? 1 : fes_vdim;
   MFEM_VERIFY(nd <= BOUNDS_MAX_ND,
               "Device element bounds kernel supports up to 32 "
               "1D degrees of freedom.");

   lower.SetSize(ne*ncomp, e_vec);
   upper.SetSize(ne*ncomp, e_vec);
   lower.UseDevice(true);
   upper.UseDevice(true);

   internal::PLBoundDeviceData data
   {
      nb,
      ncp,
      proj,
      nodes.Read(),
      weights.Read(),
      control_points.Read(),
      lbound_t.Read(),
      ubound_t.Read()
   };

   const auto E = Reshape(e_vec.Read(), nd, fes_vdim, ne);
   auto L = Reshape(lower.Write(), ne, ncomp);
   auto U = Reshape(upper.Write(), ne, ncomp);
   const int comp0 = (vdim > 0) ? (vdim - 1) : 0;

   mfem::forall_2D<BOUNDS_MAX_ND>(ne*ncomp, nd, 1,
                                  [=] MFEM_HOST_DEVICE (int ec)
   {
      const int e = ec % ne;
      const int c = ec / ne;
      const int vc = comp0 + c;
      const real_t *coeff = &E(0, vc, e);
      const int tid = MFEM_THREAD_ID(x);

      MFEM_SHARED real_t s0[BOUNDS_MAX_ND];
      MFEM_SHARED real_t s1[BOUNDS_MAX_ND];
      MFEM_SHARED real_t slo[BOUNDS_MAX_ND];
      MFEM_SHARED real_t shi[BOUNDS_MAX_ND];
      MFEM_SHARED real_t sa0;
      MFEM_SHARED real_t sa1;

      MFEM_FOREACH_THREAD(i, x, nd)
      {
         if (data.proj)
         {
            const real_t x = 2.0*data.nodes[i] - 1.0;
            const real_t w = 2.0*data.weights[i];
            s0[i] = 0.5*coeff[i]*w;
            s1[i] = 1.5*coeff[i]*w*x;
         }
         else
         {
            s0[i] = 0.0;
            s1[i] = 0.0;
         }
      }
      MFEM_SYNC_THREAD;

      if (tid == 0)
      {
         sa0 = 0.0;
         sa1 = 0.0;
         for (int i = 0; i < nd; i++)
         {
            sa0 += s0[i];
            sa1 += s1[i];
         }
      }
      MFEM_SYNC_THREAD;

      // Reuse s0 to cache the projected coefficients for this block.
      MFEM_FOREACH_THREAD(i, x, nd)
      {
         if (data.proj)
         {
            const real_t x = 2.0*data.nodes[i] - 1.0;
            s0[i] = coeff[i] - sa0 - sa1*x;
         }
         else
         {
            s0[i] = coeff[i];
         }
      }
      MFEM_SYNC_THREAD;

      real_t lower_ec = HUGE_VAL;
      real_t upper_ec = -HUGE_VAL;

      for (int j = 0; j < data.ncp; j++)
      {
         MFEM_FOREACH_THREAD(i, x, nd)
         {
            const real_t val = s0[i];
            const real_t lv = data.lbound_t[i + j*data.nb]*val;
            const real_t uv = data.ubound_t[i + j*data.nb]*val;
            slo[i] = lv < uv ? lv : uv;
            shi[i] = lv > uv ? lv : uv;
         }
         MFEM_SYNC_THREAD;

         if (tid == 0)
         {
            real_t lo = 0.0;
            real_t hi = 0.0;
            if (data.proj)
            {
               const real_t xcp = 2.0*data.control_points[j] - 1.0;
               lo = sa0 + sa1*xcp;
               hi = lo;
            }

            for (int i = 0; i < nd; i++)
            {
               lo += slo[i];
               hi += shi[i];
            }

            lower_ec = lower_ec < lo ? lower_ec : lo;
            upper_ec = upper_ec > hi ? upper_ec : hi;
         }
         MFEM_SYNC_THREAD;
      }

      if (tid == 0)
      {
         L(e, c) = lower_ec;
         U(e, c) = upper_ec;
      }
   });
}

} // namespace mfem

#endif // MFEM_BOUNDS
