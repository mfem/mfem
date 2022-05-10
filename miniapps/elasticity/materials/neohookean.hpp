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
 * choosing the method of derivative calculation for the stress (which is
 * the derivative of the strain energy density) and for the 
 * `action_of_gradient`, which applies the tangent elastic tensor operator
 * as a directional derivative.
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
      // Incorrect answer with IsotropicIdentity and enzyme fwddiff mode.
      // type deduction is not what I expect - why?
      // Ask Julian
      auto I = mfem::internal::Identity<dim>();
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
      else if (energy_gradient_type == GradientType::EnzymeFwd)
      {
         return stress_enzyme_fwd(dudx);
      }
      else if (energy_gradient_type == GradientType::EnzymeRev)
      {
         return stress_enzyme_rev(dudx);
      }
      else if (energy_gradient_type == GradientType::FiniteDiff)
      {
         return stress_fd(dudx);
      }
      else if (energy_gradient_type == GradientType::DualNumbers)
      {
         return stress_dual(dudx);
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
      auto I = mfem::internal::Identity<dim>();
      T J = det(I + dudx);
      T p = -2.0 * D1 * J * (J - 1);
      auto devB = dev(dudx + transpose(dudx) + dot(dudx, transpose(dudx)));
      auto sigma = -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;
      auto F = dudx + I;
      return J*sigma*transpose(inv(F));
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

   /**
    * @brief Evaluate the stress with forward mode differentiation
    *
    * @param[in] dudx
    * @return Piola stress
    */
   template <typename T>
   MFEM_HOST_DEVICE tensor<T, dim, dim>
   stress_enzyme_fwd(const tensor<T, dim, dim> &dudx) const
   {
      tensor<T, dim, dim> direction{};
      tensor<T, dim, dim> P{};
      T W;
      for (int i = 0; i < dim; ++i) {
         for (int j = 0; j < dim; ++j) {
            direction[i][j] = 1;
            __enzyme_fwddiff<void>(strain_energy_density_wrapper, enzyme_const, this,
                                   enzyme_dupnoneed, &dudx, &direction,
                                   enzyme_dup, &W, &(P[i][j]));
            direction[i][j] = 0;
         }
      }
      return P;
   }
#endif

   /**
    * @brief Evaluate the stress with finite differences
    *
    * @param[in] dudx
    * @return Piola stress
    */
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
    * @brief Evaluate the stress with built-in dual number forward mode
    *
    * @param[in] dudx
    * @return Piola stress
    */
   template <typename T>
   MFEM_HOST_DEVICE tensor<T, dim, dim>
   stress_dual(const tensor<T, dim, dim> &dudx) const
   {
      tensor<T, dim, dim> dir{};
      tensor<T, dim, dim> P{};
      for (int k = 0; k < dim; ++k) {
         for (int l = 0; l < dim; ++l) {
            dir[k][l] = 1;
            auto H(make_tensor<dim, dim>([&](int i, int j)
            {
               return mfem::internal::dual<T, T> {dudx[i][j], dir[i][j]};
            }));
            auto W = strain_energy_density(H);
            P[k][l] = get_gradient(W);
            dir[k][l] = 0;
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
   stress_wrapper(NeoHookeanMaterial<dim, stress_gradient_type, energy_gradient_type> *self,
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
      double J = det(F);
      double Jm23 = pow(J, -2.0/3.0);
      double I1 = inner(F, F);
      auto invF = inv(F);

      double dWvol = 2.0*D1*(J - 1.0);
      double ddWvol = 2.0*D1;

      tensor<double, dim, dim, dim, dim> Avol = make_tensor<dim, dim, dim, dim>(
          [&](int i, int j, int k, int l)
          {
             return (dWvol*J*(invF[j][i]*invF[l][k] - invF[j][k]*invF[l][i])
                     + J*J*ddWvol*invF[j][i]*invF[l][k]);
          });

      tensor<double, 3, 3, 3, 3> Adev = make_tensor<dim, dim, dim, dim>(
          [&](int i, int j, int k, int l)
          {
             return 2.0*C1*Jm23*((i==k)*(j==l)
                                 - 2.0/3.0*(invF[j][i]*F[k][l] + invF[l][k]*F[i][j])
                                 + I1/3.0*invF[j][k]*invF[l][i] + 2.0/9.0*I1*invF[l][k]*invF[j][i]);
          });
      return Adev + Avol;
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
      const auto I = mfem::internal::Identity<dim>();

      const auto F = I + du_dx;
      const double J = det(F);
      const double I1_3 = ddot(F, F)/3.0;
      const auto invF = inv(F);
      const auto invFT = transpose(invF);
      const double fac = pow(J, -2.0/3.0) * 2.0 * C1;
      const double a1 = ddot(invFT, ddu_dx);
      const double a2 = ddot(F, ddu_dx);
      const auto M = dot(invF, dot(ddu_dx, invF));
      const auto Pdev = fac*(F - I1_3*invFT);
      
      auto dPdev = ddu_dx - (2.0/3.0*a2)*invFT + I1_3*transpose(M);
      dPdev = fac * dPdev;
      dPdev -= (2.0/3.0*a1) * Pdev;

      const double dWvol = 2.0*D1*(J - 1.0);
      const double ddWvol = 2.0*D1;
      auto dPvol = (J*ddWvol + dWvol)*a1*invFT - dWvol*transpose(M);
      dPvol = J*dPvol;
      
      return dPdev + dPvol;
   }

   // Parameters
   double D1 = 100.0;
   double C1 = 50.0;
};

#endif
