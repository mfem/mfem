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

#ifndef MFEM_ELASTICITY_MAT_NEOHOOKEAN_HPP
#define MFEM_ELASTICITY_MAT_NEOHOOKEAN_HPP

#include "general/enzyme.hpp"
#include "gradient_type.hpp"
#include "linalg/tensor.hpp"
#include "mfem.hpp"

using mfem::internal::tensor;
using mfem::internal::make_tensor;

/**
 * @brief Neo-Hookean material
 *
 * Defines a Neo-Hookean material response. It satisfies the material_type
 * interface for ElasticityOperator::SetMaterial. This material type allows
 * choosing the method of derivative calculation in `action_of_gradient`.
 * Choices include methods derived by hand using symbolic calculation and a
 * variety of automatically computed gradient applications, like
 * - Enzyme forward mode
 * - Enzyme reverse mode
 * - Dual number type forward mode
 * - Finite difference mode
 *
 * @tparam dim
 * @tparam stress_gradient_type How to compute the derivative of the stress
 * @tparam energy_gradient_type How to compute the derivative of the energy, which is the stress
 */
template <int dim = 3, GradientType stress_gradient_type = GradientType::Symbolic, GradientType energy_gradient_type = GradientType::Symbolic>
struct NeoHookeanMaterial
{
   static_assert(dim == 3, "NeoHookean model only defined in 3D");

   /**
    * @brief Compute strain energy density
    * @param[in] dudx displacement gradient
    * @return W strain energy density in reference configuration
    */
   template <typename T>
   MFEM_HOST_DEVICE T
   strain_energy_density(const tensor<T, dim, dim> & dudx) const
   {
      constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto F = I + dudx;
      auto J = det(F);
      T Jm23 = pow(J, -2.0/3.0);
      T Wvol = D1*(J - 1.0)*(J - 1.0);
      T Wdev = C1*(Jm23*inner(F,F) - 3.0);
      return Wdev + Wvol;
   }

   /**
    * @brief Compute the stress response.
    *
    * @param[in] dudx derivative of the displacement
    * @return
    */
   template <typename T>
   MFEM_HOST_DEVICE tensor<T, dim, dim>
   stress(const tensor<T, dim, dim> & dudx) const
   {
      if (energy_gradient_type == GradientType::Symbolic)
      {
         return stress_symbolic(dudx);
      }
      else if (energy_gradient_type == GradientType::EnzymeRev)
      {
         return stress_enzyme_rev(dudx);
      }
      else if (energy_gradient_type == GradientType::FiniteDiff)
      {
         return stress_fd(dudx);
      }
      else
      {
         return 0.0*dudx;
      }
   }

   /**
    * @brief A method to wrap the strain energy density calculation into a static function.
    *
    * This is necessary for Enzyme to access the class pointer (self).
    *
    * @param[in] self the class pointer
    * @param[in] dudx the displacement gradient
    * @param[out] W the strain energy density 
    */
   MFEM_HOST_DEVICE static void
   strain_energy_density_wrapper(NeoHookeanMaterial<dim, stress_gradient_type> *self,
                                 tensor<double, dim, dim> &dudx, double &W)
   {
      W = self->strain_energy_density(dudx);
   }

   /**
    * @brief Evaluate the stress symbolically
    *
    * @param[in] dudx
    * @return Piola stress
    */
   template <typename T>
   MFEM_HOST_DEVICE tensor<T, dim, dim>
   stress_symbolic(const tensor<T, dim, dim> &__restrict__ dudx) const
   {
      constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      T J = det(I + dudx);
      T p = -2.0 * D1 * J * (J - 1);
      auto devB = dev(dudx + transpose(dudx) + dot(dudx, transpose(dudx)));
      auto sigma = -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;
      return sigma;
   }

#ifdef MFEM_USE_ENZYME
   /**
    * @brief Evaluate the stress with reverse mode differentiation
    *
    * @param[in] dudx
    * @return Piola stress
    */
   template <typename T>
   MFEM_HOST_DEVICE tensor<T, dim, dim>
   stress_enzyme_rev(const tensor<T, dim, dim> &dudx) const
   {
      T W;
      T seed{1.0};
      tensor<T, 3, 3> P{};
      __enzyme_autodiff<void>(strain_energy_density_wrapper, enzyme_const, this,
                              enzyme_dup, &dudx, &P, enzyme_dupnoneed, &W, &seed);
      return P;
   }
#endif

   template <typename T>
   MFEM_HOST_DEVICE tensor<T, dim, dim>
   stress_fd(const tensor<T, dim, dim> &dudx) const
   {
      auto H = dudx;
      auto P = 0.0*dudx;
      T h{1e-6};
      for (int i = 0; i < dim; ++i) {
         for (int j = 0; j < dim; ++j) {
            H[i][j] += h;
            auto Wp = strain_energy_density(H);
            H[i][j] -= 2.0*h;
            auto Wm = strain_energy_density(H);
            H[i][j] += h;
            P[i][j] = (Wp - Wm) / (2.0 * h);
         }
      }
      return P;
   }

   /**
    * @brief A method to wrap the stress calculation into a static function.
    *
    * This is necessary for Enzyme to access the class pointer (self).
    *
    * @param[in] self
    * @param[in] dudx
    * @param[out] sigma
    * @return stress
    */
   MFEM_HOST_DEVICE static void
   stress_wrapper(NeoHookeanMaterial<dim, stress_gradient_type> *self,
                  tensor<double, dim, dim> &dudx,
                  tensor<double, dim, dim> &sigma)
   {
      sigma = self->stress(dudx);
   }

   /**
    * @brief Compute the gradient.
    *
    * This method is used in the ElasticityDiagonalPreconditioner type to
    * compute the gradient matrix entries of the current quadrature point,
    * instead of the action.
    *
    * @param[in] dudx
    * @return
    */
   MFEM_HOST_DEVICE tensor<double, dim, dim, dim, dim>
   gradient(tensor<double, dim, dim> dudx) const
   {
      constexpr auto I = mfem::internal::IsotropicIdentity<dim>();

      tensor<double, dim, dim> F = I + dudx;
      tensor<double, dim, dim> invF = inv(F);
      tensor<double, dim, dim> devB =
         dev(dudx + transpose(dudx) + dot(dudx, transpose(dudx)));
      double J = det(F);
      double coef = (C1 / pow(J, 5.0 / 3.0));
      return make_tensor<dim, dim, dim, dim>([&](int i, int j, int k,
                                                 int l)
      {
         return 2.0 * (D1 * J * (i == j) - (5.0 / 3.0) * coef * devB[i][j]) *
                invF[l][k] +
                2.0 * coef *
                ((i == k) * F[j][l] + F[i][l] * (j == k) -
                 (2.0 / 3.0) * ((i == j) * F[k][l]));
      });
   }

   /**
    * @brief Apply the gradient of the stress.
    *
    * @param[in] dudx
    * @param[in] ddudx
    * @return
    */
   MFEM_HOST_DEVICE tensor<double, dim, dim>
   action_of_gradient(const tensor<double, dim, dim> &dudx,
                      const tensor<double, dim, dim> &ddudx) const
   {
      if (stress_gradient_type == GradientType::Symbolic)
      {
         return action_of_gradient_symbolic(dudx, ddudx);
      }
#ifdef MFEM_USE_ENZYME
      else if (stress_gradient_type == GradientType::EnzymeFwd)
      {
         return action_of_gradient_enzyme_fwd(dudx, ddudx);
      }
      else if (stress_gradient_type == GradientType::EnzymeRev)
      {
         return action_of_gradient_enzyme_rev(dudx, ddudx);
      }
#endif
      else if (stress_gradient_type == GradientType::FiniteDiff)
      {
         return action_of_gradient_fd(dudx, ddudx);
      }
      else if (stress_gradient_type == GradientType::DualNumbers)
      {
         return action_of_gradient_dual(dudx, ddudx);
      }
   }

   MFEM_HOST_DEVICE tensor<double, dim, dim>
   action_of_gradient_dual(const tensor<double, dim, dim> &dudx,
                           const tensor<double, dim, dim> &ddudx) const
   {
      auto sigma = stress(make_tensor<dim, dim>([&](int i, int j)
      {
         return mfem::internal::dual<double, double> {dudx[i][j], ddudx[i][j]};
      }));
      return make_tensor<dim, dim>(
      [&](int i, int j) { return sigma[i][j].gradient; });
   }

#ifdef MFEM_USE_ENZYME
   MFEM_HOST_DEVICE tensor<double, dim, dim>
   action_of_gradient_enzyme_fwd(const tensor<double, dim, dim> &dudx,
                                 const tensor<double, dim, dim> &ddudx) const
   {
      tensor<double, dim, dim> sigma{};
      tensor<double, dim, dim> dsigma{};

      __enzyme_fwddiff<void>(stress_wrapper, enzyme_const, this, enzyme_dup,
                             &dudx, &ddudx, enzyme_dupnoneed, &sigma, &dsigma);
      return dsigma;
   }

   MFEM_HOST_DEVICE tensor<double, dim, dim>
   action_of_gradient_enzyme_rev(const tensor<double, dim, dim> &dudx,
                                 const tensor<double, dim, dim> &ddudx) const
   {
      tensor<double, dim, dim, dim, dim> gradient{};
      tensor<double, dim, dim> sigma{};
      tensor<double, dim, dim> dir{};

      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            dir[i][j] = 1;
            __enzyme_autodiff<void>(stress_wrapper, enzyme_const, this, enzyme_dup,
                                    &dudx, &gradient[i][j], enzyme_dupnoneed,
                                    &sigma, &dir);
            dir[i][j] = 0;
         }
      }
      return ddot(gradient, ddudx);
   }
#endif

   MFEM_HOST_DEVICE tensor<double, dim, dim>
   action_of_gradient_fd(const tensor<double, dim, dim> &dudx,
                         const tensor<double, dim, dim> &ddudx) const
   {
      return (stress(dudx + 1.0e-8 * ddudx) - stress(dudx - 1.0e-8 * ddudx)) /
             2.0e-8;
   }

   // d(stress)_{ij} := (d(stress)_ij / d(du_dx)_{kl}) * d(du_dx)_{kl}
   // Only works with 3D stress
   MFEM_HOST_DEVICE tensor<double, dim, dim>
   action_of_gradient_symbolic(const tensor<double, dim, dim> &du_dx,
                               const tensor<double, dim, dim> &ddu_dx) const
   {
      constexpr auto I = mfem::internal::IsotropicIdentity<dim>();

      tensor<double, dim, dim> F = I + du_dx;
      tensor<double, dim, dim> invFT = inv(transpose(F));
      tensor<double, dim, dim> devB =
         dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
      double J = det(F);
      double coef = (C1 / pow(J, 5.0 / 3.0));
      double a1 = ddot(invFT, ddu_dx);
      double a2 = ddot(F, ddu_dx);

      return (2.0 * D1 * J * a1 - (4.0 / 3.0) * coef * a2) * I -
             ((10.0 / 3.0) * coef * a1) * devB +
             (2 * coef) * (dot(ddu_dx, transpose(F)) + dot(F, transpose(ddu_dx)));
   }

   // Parameters
   double D1 = 100.0;
   double C1 = 50.0;
};

#endif
