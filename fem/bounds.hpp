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
#include "fespace.hpp"

namespace mfem
{

/** @name Piecewise linear bounds of bases
    \brief Piecewise linear bounds of bases can be used to compute bounds on the grid function in each element. The bounds for the bases are constructed based on the following parameters:

    (i) @b nb: number of bases/nodes in 1D (i.e. polynomial order+1),

    (ii) @b b_type: bases type, 0 - Lagrange interpolants on Gauss-Legendre nodes, 1 - Lagrange interpolants on Gauss-Lobatto-Legendre nodes, and
    2 - Positive/Bernstein bases on uniformly distributed nodes,

    (iii) @b ncp: number of control points used to construct the piecewise linear bounds

    (iv) @b cp_type: control point distribution. 0 - GL + end-points,
                    1 - Chebyshev.

    Note: @b nb and @b b_type are inferred directly from the grid-function.

    If the user does not specify @b ncp and @b cp_type, the minimum value of
    @b ncp is used that would bound the bases for the @b cp_type. We default
    to @b cp_type = 0 as it requires fewer number of points to bound the bases. Typically, @b ncp = 2 @b nb is sufficient to get fairly compact bounds, and increasing @b ncp results in tighter bounds.

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
   DenseMatrix lbound, ubound; // nb x ncp matrices with bounds of all bases
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

   // Get minimum number of control points needed to bound the given bases
   int GetMinimumPointsForGivenBases(int nb_i, int b_type_i,
                                     int cp_type_i) const;

   // Print information about the bounds
   void Print(std::ostream &outp = mfem::out) const;

   // Enable (default) or disable linear projection before bounding.
   // This projection increases the computational cost but results in tighter
   // bounds.
   void SetProjectionFlagForBounding(bool proj_) { proj = proj_; }

   /// Compute piecewise linear bounds for the lexicographically-ordered
   /// coefficients in @a coeff in 1D/2D/3D.
   void GetNDBounds(const int rdim, const Vector &coeff,
                    Vector &intmin, Vector &intmax) const;

   /// Get number of control points used to compute the bounds.
   int GetNControlPoints() const { return ncp; }
private:
   /// Compute piecewise linear bounds for the lexicographically-ordered
   /// coefficients in @a coeff in 1D.
   void Get1DBounds(const Vector &coeff, Vector &intmin, Vector &intmax) const;

   /// Compute piecewise linear bounds for the lexicographically-ordered
   /// coefficients in @a coeff in 2D.
   void Get2DBounds(const Vector &coeff, Vector &intmin, Vector &intmax) const;

   /// Compute piecewise linear bounds for the lexicographically-ordered
   /// coefficients in @a coeff in 3D.
   void Get3DBounds(const Vector &coeff, Vector &intmin, Vector &intmax) const;

   /// Setup matrix used to compute values at given 1D locations in [0,1]
   /// for Bernstein bases.
   void SetupBernsteinBasisMat(DenseMatrix &basisMat, Vector &nodesBern) const;

   void Setup(const int nb_i, const int ncp_i, const int b_type_i,
              const int cp_type_i, const real_t tol_i);
};

} // namespace mfem

#endif // MFEM_BOUNDS
