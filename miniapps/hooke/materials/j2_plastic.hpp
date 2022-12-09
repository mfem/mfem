// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#pragma once

#include "linalg/dual_tensor.hpp"
#include "material_traits.hpp"
#include "mfem.hpp"

struct J2Material
{
   /// this material is written for 3D
   static constexpr int dim = 3;

   double E;        ///< Young's modulus
   double nu;       ///< Poisson's ratio
   double Hi;       ///< isotropic hardening constant
   double Hk;       ///< kinematic hardening constant
   double sigma_y;  ///< yield stress
   double density;  ///< mass density

   /// @brief variables required to characterize the hysteresis response
   struct State
   {
      /// back-stress tensor
      mfem::internal::tensor<double, dim, dim> beta;

      /// plastic strain
      mfem::internal::tensor<double, dim, dim> plastic_strain;

      /// accumulated plastic strain
      double accumulated_plastic_strain;
   };

   /** @brief calculate the Cauchy stress, given the displacement gradient and previous material state */
   template <typename T>
   auto stress(State& state, const T du_dx) const
   {
      using std::sqrt;
      constexpr auto I = mfem::internal::IsotropicIdentity<3>();
      const double   K = E / (3.0 * (1.0 - 2.0 * nu));
      const double   G = 0.5 * E / (1.0 + nu);

      //
      // see pg. 260, box 7.5,
      // in "Computational Methods for Plasticity"
      //

      // (i) elastic predictor
      auto el_strain = mfem::internal::sym(du_dx) - state.plastic_strain;
      auto p         = K * mfem::internal::tr(el_strain);
      auto s         = 2.0 * G * mfem::internal::dev(el_strain);
      auto eta       = s - state.beta;
      auto q         = sqrt(3.0 / 2.0) * mfem::internal::norm(eta);
      auto phi       = q - (sigma_y + Hi * state.accumulated_plastic_strain);

      // (ii) admissibility
      if (phi > 0.0)
      {
         // see (7.207) on pg. 261
         auto plastic_strain_inc = phi / (3 * G + Hk + Hi);

         // from here on, only normalize(eta) is required
         // so we overwrite eta with its normalized version
         eta = mfem::internal::normalize(eta);

         // (iii) return mapping
         s = s - sqrt(6.0) * G * plastic_strain_inc * eta;
         state.accumulated_plastic_strain += mfem::internal::get_value(
                                                plastic_strain_inc);
         state.plastic_strain += sqrt(3.0 / 2.0) * mfem::internal::get_value(
                                    plastic_strain_inc) *
                                 mfem::internal::get_value(eta);
         state.beta = state.beta + sqrt(2.0 / 3.0) * Hk * mfem::internal::get_value(
                         plastic_strain_inc) * mfem::internal::get_value(eta);
      }

      return s + p * I;
   }

   /**
    * @brief Apply the gradient of the stress.
    *
    */
   MFEM_HOST_DEVICE
   template <typename T> auto
   action_of_gradient(State &state, const T &dudx, const T &ddudx) const
   {
      auto sigma = stress(state, make_tensor<dim, dim>([&](int i, int j)
      {
         return mfem::internal::dual<double, double> {dudx[i][j], ddudx[i][j]};
      }));
      return make_tensor<dim, dim>(
      [&](int i, int j) { return sigma[i][j].gradient; });
   }

   MFEM_HOST_DEVICE tensor<double, dim, dim, dim, dim>
   gradient(State &state, tensor<double, dim, dim> dudx) const
   {
      using std::sqrt;
      constexpr auto I = mfem::internal::IsotropicIdentity<3>();
      const double   K = E / (3.0 * (1.0 - 2.0 * nu));
      const double   G = 0.5 * E / (1.0 + nu);

      //
      // see pg. 260, box 7.5,
      // in "Computational Methods for Plasticity"
      //

      // (i) elastic predictor
      auto el_strain = mfem::internal::sym(dudx) - state.plastic_strain;
      auto p         = K * mfem::internal::tr(el_strain);
      auto s         = 2.0 * G * mfem::internal::dev(el_strain);
      auto eta       = s - state.beta;
      auto q         = sqrt(3.0 / 2.0) * mfem::internal::norm(eta);
      auto phi       = q - (sigma_y + Hi * state.accumulated_plastic_strain);

      // (ii) admissibility
      if (phi > 0.0)
      {
         // see (7.207) on pg. 261
         auto plastic_strain_inc = phi / (3 * G + Hk + Hi);

         // from here on, only normalize(eta) is required
         // so we overwrite eta with its normalized version
         eta = mfem::internal::normalize(eta);

         // (iii) return mapping
         s = s - sqrt(6.0) * G * plastic_strain_inc * eta;
         state.accumulated_plastic_strain += mfem::internal::get_value(
                                                plastic_strain_inc);
         state.plastic_strain += sqrt(3.0 / 2.0) * mfem::internal::get_value(
                                    plastic_strain_inc) *
                                 mfem::internal::get_value(eta);
         state.beta = state.beta + sqrt(2.0 / 3.0) * Hk * mfem::internal::get_value(
                         plastic_strain_inc) * mfem::internal::get_value(eta);
      }

      // return s + p * I;

      double A1 = 2.0 * G;
      double A2 = 0.0;

      tensor<double, 3, 3> N{};

      auto pl_strain_inc = fmax(0.0, phi / (3 * G + Hk + Hi));

      if (pl_strain_inc > 0.0)
      {
         tensor<double, 3, 3> s = 2.0 * G * mfem::internal::dev(el_strain);
         N = mfem::internal::normalize(s - state.beta);

         A1 -= 6 * G * G * pl_strain_inc / q;
         A2 = 6 * G * G * ((pl_strain_inc / q) - (1.0 /
                                                  (3.0 * G + Hi + Hk)));
      }

      return make_tensor<3, 3, 3, 3>([&](auto i, auto j, auto k, auto l)
      {
         double I4    = (i == j) * (k == l);
         double I4sym = 0.5 * ((i == k) * (j == l) + (i == l) * (j == k));
         double I4dev = I4sym - (i == j) * (k == l) / 3.0;
         return K * I4 + A1 * I4dev + A2 * N(i, j) * N(k, l);
      });
   }

};

template<>
struct material_has_state<J2Material>
{
   static const bool value = true;
};
