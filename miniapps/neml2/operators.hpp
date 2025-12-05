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

#pragma once

#include "constitutive.hpp"
#include "mfem.hpp"

namespace op {

using namespace mfem;
using namespace mfem::future;

// Forward declaration
class LinearMomentumBalanceJacobian;

/**
 * @brief Linear momentum balance operator
 *
 * This operator computes the residual vector given the unknown displacements
 *
 * It consists of three sub-operators:
 *   - disp -> strain: A DifferentiableOperator which computes strain from
 *                     displacements. The computed strain is stored in a
 *                     ParameterFunction.
 *   - strain -> stress: A NEML2 material model which computes stress from
 *                       strain. The strain is first mapped from MFEM to NEML2
 *                       tensors, then the stress is computed, and finally
 *                       mapped back to MFEM as a ParameterFunction.
 *   - stress -> residual: A DifferentiableOperator which computes the
 *                         residual given stress.
 */
class LinearMomentumBalance : public Operator {
public:
  /**
   * @brief Construct a new Linear Momentum Balance object
   *
   * @param fe_space Finite element space for the displacements
   * @param ir Integration rule for the quadrature
   * @param cmodel NEML2 constitutive model for the material
   */
  LinearMomentumBalance(ParFiniteElementSpace &fe_space,
                        const IntegrationRule &ir,
                        std::shared_ptr<neml2::Model> cmodel);

  /**
   * @brief Set the essential boundary conditions
   *
   * @param ebcs Map of arrays of boundary attributes to VectorCoefficients
   */
  void
  SetEssBdrConditions(const std::map<Array<int> *, VectorCoefficient *> &ebcs);

  /**
   * @brief Perform residual evaluation
   *
   * @param x Input displacement L-vector
   * @param y Output residual L-vector
   */
  void Mult(const Vector &X, Vector &R) const override;

  /**
   * @brief Get the gradient operator which defines the action of the Jacobian
   *
   * @param x Input displacement L-vector
   * @return Operator& The gradient operator
   */
  Operator &GetGradient(const Vector &x) const override;

  ///@{
  /// @brief Enumeration of the operator fields (required by
  /// DifferentiableOperator)
  constexpr static int SOLUTION = 1;
  constexpr static int NODAL_COORDS = 2;
  constexpr static int STRAIN = 3;
  constexpr static int STRESS = 4;
  constexpr static int RESIDUAL = 5;
  ///@}

  /// The strain operator
  std::unique_ptr<DifferentiableOperator> strain_op;
  /// The constitutive model "operator"
  std::unique_ptr<ConstitutiveModel> constit_op;
  /// The stress divergence operator
  std::unique_ptr<DifferentiableOperator> stressdiv_op;

  /// The finite element space for the displacements
  ParFiniteElementSpace &fe_space;
  /// The integration rule for the quadrature
  const IntegrationRule &ir;
  /// The parallel mesh
  ParMesh &pmesh;
  /// The nodes of the mesh
  ParGridFunction &nodes;
  /// The finite element space for the coordinates
  ParFiniteElementSpace &coord_space;
  /// The quadrature space for symmetric 2nd order tensors
  UniformParameterSpace q_space_symr2;

  /// Essential true dofs
  Array<int> ess_tdofs;
  /// Essential dof values
  Vector ess_dof_vals;

private:
  /// The strain storage
  mutable ParameterFunction _strain;
  /// The stress storage
  mutable ParameterFunction _stress;
  /// The Jacobian operator (at the given state)
  mutable std::unique_ptr<LinearMomentumBalanceJacobian> _deriv_op;
};

struct Strain {
  /**
   * @brief Compute the strain from the displacement gradient and the Jacobian
   *
   * For small strain, $\varepsilon = \frac{1}{2} \left( \nabla u + \nabla u^T
   * \right)$, with $\nabla u = \frac{\partial u}{\partial x} = \frac{\partial
   * u}{\partial \xi} \cdot \frac{\partial \xi}{\partial x} = \frac{\partial
   * u}{\partial \xi} \cdot J^{-1}$.
   *
   * @param dudxi Displacement gradient in parametric coordinates
   * @param J Jacobian of the parametric to physical mapping
   */
  MFEM_HOST_DEVICE auto operator()(const tensor<real_t, 3, 3> &dudxi,
                                   const tensor<real_t, 3, 3> &J) const {
    const auto dudx = dudxi * inv(J);
    const auto e = real_t(0.5) * (dudx + transpose(dudx));
    // NEML2 uses Mandel notation for symmetric 2nd order tensors.
    constexpr real_t sqrt2 = 1.4142135623730951_r;
    const auto e_mandel =
        tensor<real_t, 6>{e(0, 0),         e(1, 1),         e(2, 2),
                          sqrt2 * e(1, 2), sqrt2 * e(0, 2), sqrt2 * e(0, 1)};
    return tuple{e_mandel};
  }
};

struct StressDivergence {
  /**
   * @brief Compute the stress divergence contribution to the residual
   *
   * Given the stress tensor $\sigma$, the contribution to the residual at
   * quadrature point is given by $\nabla \phi \cdot \sigma$.
   *
   * @note The definition here appears incomplete, but it's not; multiplication
   * by B^T is handled by the DifferentiableOperator's output operator.
   *
   * @param stress Stress tensor in Mandel notation at the quadrature point
   * @param J Jacobian of the parametric to physical mapping
   * @param w Quadrature weight
   */
  MFEM_HOST_DEVICE auto operator()(const tensor<real_t, 6> &stress,
                                   const tensor<real_t, 3, 3> &J,
                                   const real_t &w) const {
    constexpr real_t sqrt2 = 1.4142135623730951_r;
    const auto stress_tensor = tensor<real_t, 3, 3>{
        stress(0),         stress(5) / sqrt2, stress(4) / sqrt2,
        stress(5) / sqrt2, stress(1),         stress(3) / sqrt2,
        stress(4) / sqrt2, stress(3) / sqrt2, stress(2)};
    const auto div_sigma = stress_tensor * transpose(inv(J)) * det(J) * w;
    return tuple{div_sigma};
  }
};

/**
 * @brief Jacobian operator for the linear momentum balance equation
 *
 */
class LinearMomentumBalanceJacobian : public Operator {
public:
  LinearMomentumBalanceJacobian(const LinearMomentumBalance &r,
                                const Vector &X);

  /**
   * @brief Perform Jacobian evaluation
   *
   * @param dX Input delta displacement L-vector
   * @param dR Output delta residual L-vector
   */
  void Mult(const Vector &dX, Vector &dR) const override;

private:
  /// The underlying linear momentum balance operator
  const LinearMomentumBalance &_op;
  /// The quadrature space for symmetric 2nd order tensors
  UniformParameterSpace _q_space_symr2;
  /// Temporary storage for delta strain
  mutable ParameterFunction _delta_strain;
  /// Temporary storage for delta stress
  mutable ParameterFunction _delta_stress;
  /// Material tangent (dstress/dstrain) obtained from NEML2 evaluated at the
  /// current strain
  neml2::Tensor _tangent;
};

} // namespace op
