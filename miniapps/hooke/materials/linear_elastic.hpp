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

#ifndef MFEM_ELASTICITY_MAT_LIN_ELAST_HPP
#define MFEM_ELASTICITY_MAT_LIN_ELAST_HPP

#include "linalg/tensor.hpp"
#include "mfem.hpp"

using mfem::internal::tensor;

/** @brief Linear elastic material.
 *
 * Defines a linear elastic material response. It satisfies the material_type
 * interface for ElasticityOperator::SetMaterial.
 */
template <int dim> struct LinearElasticMaterial
{
   /**
    * @brief Compute the stress response.
    *
    * @param[in] dudx derivative of the displacement
    * @return tensor<double, dim, dim>
    */
   tensor<mfem::real_t, dim, dim>
   MFEM_HOST_DEVICE stress(const tensor<mfem::real_t, dim, dim> &dudx) const
   {
      constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto epsilon = sym(dudx);
      return lambda * tr(epsilon) * I + 2.0 * mu * epsilon;
   }

   /**
    * @brief Apply the gradient of the stress.
    *
    */
   tensor<mfem::real_t, dim, dim> MFEM_HOST_DEVICE
   action_of_gradient(const tensor<mfem::real_t, dim, dim> & /* dudx */,
                      const tensor<mfem::real_t, dim, dim> &ddudx) const
   {
      return stress(ddudx);
   }

   /**
    * @brief Compute the gradient.
    *
    * This method is used in the ElasticityDiagonalPreconditioner type to
    * compute the gradient matrix entries of the current quadrature point,
    * instead of the action.
    *
    * @return tensor<double, dim, dim, dim, dim>
    */
   tensor<mfem::real_t, dim, dim, dim, dim>
   MFEM_HOST_DEVICE gradient(tensor<mfem::real_t, dim, dim> /* dudx */) const
   {
      return mfem::internal::make_tensor<dim, dim, dim, dim>([&](int i, int j, int k,
                                                                 int l)
      {
         return lambda * (i == j) * (k == l) +
                mu * ((i == l) * (j == k) + (i == k) * (j == l));
      });
   }

   /// First Lame parameter
   mfem::real_t lambda = 100;
   /// Second Lame parameter
   mfem::real_t mu = 50;
};

#endif
