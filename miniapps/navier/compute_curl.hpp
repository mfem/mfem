// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_COMPUTE_CURL_HPP
#define MFEM_COMPUTE_CURL_HPP

#include "mfem.hpp"

namespace mfem
{
namespace navier
{

/// Class to evaluate the curl and curl-curl of grid functions.
class CurlEvaluator
{
protected:
   /// Vector valued finite element space (the domain of the curl operation).
   ParFiniteElementSpace &fes;
   /// Corresponding scalar-valued space in 2D, NULL in 3D.
   ParFiniteElementSpace *scalar_fes;
   /// Spatial dimension, either 2 or 3.
   int dim;
   /// Internal vector used when computing the curl-curl.
   mutable Vector u_curl_tmp;
   /// Is partial assembly enabled?
   bool partial_assembly = false;

   /// @name Partial assembly
   ///@{

   /// Number of elements (across all MPI ranks) containing each DOF.
   mutable Array<int> els_per_dof;

   /// Nodal points in lexicographic ordering.
   mutable IntegrationRule ir_lex;
   ///@{
   /// @name Quadrature interpolators used for PA computations
   mutable QuadratureInterpolator *vector_quad_interp = nullptr;
   mutable QuadratureInterpolator *scalar_quad_interp = nullptr;
   ///@}
   ///@{
   /// @name Internal vectors used for PA computations
   mutable Vector u_evec, du_evec, curl_u_evec;
   mutable ParGridFunction curl_u_gf;
   mutable ParGridFunction u_gf;
   ///@}

   ///@}

   /// @brief Used internally to compute the curl and perpendicular gradient.
   ///
   /// In 3D, @a perp_grad must be false. In 2D, if @a perp_grad is true, the
   /// result is the perpendicular gradient of a scalar field. If @a perp_grad
   /// is false, the result is the (scalar) curl of a vector field.
   ///
   /// This function uses the "legacy" algorithm that is @b not GPU compatible.
   void ComputeCurlLegacy_(const Vector &u, Vector &curl_u, bool perp_grad) const;
public:
   /// @brief Create an object to evaluate the curl and curl-curl of grid
   /// functions in @a fes.
   CurlEvaluator(ParFiniteElementSpace &fes_);

   /// @brief Return the finite element space containing the curl.
   ///
   /// In 3D, the original vector field and its curl belong to the same space,
   /// and so this function returns the same space that was used to construct
   /// this object.
   ///
   /// In 2D, the curl of a vector field is a scalar field, and so this function
   /// returns a scalar (vdim = 1) version of the space that was used to
   /// construct this object.
   ParFiniteElementSpace &GetCurlSpace();

   /// @a const version of GetCurlSpace().
   const ParFiniteElementSpace &GetCurlSpace() const;

   /// @brief Compute the perpendicular gradient in 2D of @a u and place the
   /// result in @a perp_grad_u.
   ///
   /// The input vector @a u should be a scalar-valued T-DOF vector, and the
   /// output vector should be a vector-valued T-DOF vector.
   void ComputePerpGrad(const Vector &u, Vector &perp_grad_u) const;

   /// @brief Compute the curl of @a u and place the result in @a curl_u.
   ///
   /// The input vector @a u should be a vector-valued T-DOF vector. In 3D, the
   /// output vector should be a vector-valued T-DOF vector, and in 2D it should
   /// be a scalar-valued TDOF-vector.
   void ComputeCurl(const Vector &u, Vector &curl_u) const;

   /// @brief Compute curl(curl(u)) and place the result in
   /// @a curl_curl_u.
   ///
   /// The input and output vectors should be vector-valued T-DOF vectors
   /// belonging to the space used to construct this object.
   void ComputeCurlCurl(const Vector &u, Vector &curl_curl_u) const;

   /// @brief Enable or disable partial assembly (required for device support,
   /// e.g. on GPU), disabled by default.
   void EnablePA(bool enable_pa) { partial_assembly = enable_pa; }

   /// @name Not part of the public API
   ///@{
   /// These functions contain MFEM_FORALL kernels and so because of @c nvcc
   /// restrictions, they cannot be @a private or @a protected. However they
   /// should not be considered part of the public API.

   /// Count the number of elements containing each DOF. Used for averaging.
   void CountElementsPerDof();

   /// @brief Used internally to compute the curl and perpendicular gradient.
   ///
   /// In 3D, @a perp_grad must be false. In 2D, if @a perp_grad is true, the
   /// result is the perpendicular gradient of a scalar field. If @a perp_grad
   /// is false, the result is the (scalar) curl of a vector field.
   void ComputeCurlPA_(const Vector &u, Vector &curl_u, bool perp_grad) const;
   ///@}

   ~CurlEvaluator();
};

} // namespace navier
} // namespace mfem

#endif
