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
   /// Vector valued finite element space (the domain of the curl operation)
   ParFiniteElementSpace &fes;
   /// Corresponding scalar-valued space in 2D, NULL in 3D.
   ParFiniteElementSpace *scalar_fes;
   /// Spatial dimension, either 2 or 3.
   int dim;
   /// Internal grid function used when computing the curl-curl.
   mutable ParGridFunction curl_u;
   /// @name Partial assembly
   ///@{

   /// Nodal points in lexicographic ordering
   mutable IntegrationRule ir_lex;
   ///@{
   /// @name Quadrature interpolators used for PA computations
   mutable QuadratureInterpolator *vector_quad_interp = nullptr;
   mutable QuadratureInterpolator *scalar_quad_interp = nullptr;
   ///@}
   ///@{
   /// @name Internal vectors used for PA computations
   mutable Vector u_evec, du_evec, curl_u_evec;
   ///@}

   ///@}

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

   /// @brief Compute the curl of @a u and place the result in @a curl_u.
   ///
   /// In 3D, the grid functions @a u and @a curl_u should belong to the space used to construct
   /// this object.
   ///
   /// In 2D, there are two options:
   ///
   /// 1. The grid function @a u belongs to the vector-valued space @a fes, and
   ///    @a curl_u belongs to its scalar counterpart @a scalar_fes.
   /// 2. The grid function @a u belongs to the scalar space @a scalar_fes, and
   ///    @a curl_u belongs to the vector-valued space @a fes.
   void ComputeCurl(const ParGridFunction &u, ParGridFunction &curl_u) const;

   void ComputeCurlPA(const ParGridFunction &u, ParGridFunction &curl_u) const;

   void ComputeCurlCurlPA(
      const ParGridFunction &u, ParGridFunction &curl_curl_u) const;

   /// @brief Compute the curl-curl of @a u and place the result in
   /// @a curl_curl_u.
   ///
   /// The grid function @a u should belong to the space used to construct this
   /// object.
   void ComputeCurlCurl(const ParGridFunction &u,
                        ParGridFunction &curl_curl_u) const;

   ~CurlEvaluator();
};

} // namespace navier
} // namespace mfem

#endif
