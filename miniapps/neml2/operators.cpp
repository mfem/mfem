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

#include "operators.hpp"
#include <torch/torch.h>

namespace op {

template <typename dscalar_t>
LinearMomentumBalance<dscalar_t>::LinearMomentumBalance(
    ParFiniteElementSpace &fe_space, const IntegrationRule &ir,
    std::shared_ptr<neml2::Model> cmodel, DerivativeType deriv_type)
    : Operator(fe_space.GetTrueVSize(), fe_space.GetTrueVSize()),
      fe_space(fe_space), ir(ir), derivative_type(deriv_type),
      pmesh(*fe_space.GetParMesh()), u(&fe_space),
      nodes(*static_cast<ParGridFunction *>(pmesh.GetNodes())),
      coord_space(*nodes.ParFESpace()), q_space_symr2(pmesh, ir, 6),
      _strain(q_space_symr2), _stress(q_space_symr2) {
  // Subdomains
  Array<int> everywhere(fe_space.GetMesh()->attributes.Max());
  everywhere = 1;

  // Put strain and stress storage on device
  _strain.UseDevice(true);
  _stress.UseDevice(true);

  // Fields and parameters
  std::vector<FieldDescriptor> fields, params;

  // Set up the strain operator
  fields = {{SOLUTION, &fe_space}};
  params = {{NODAL_COORDS, &coord_space}, {STRAIN, &q_space_symr2}};
  strain_op = std::make_unique<DifferentiableOperator>(fields, params, pmesh);
  Strain strain_qfunc;
  strain_op->SetParameters({&nodes, &_strain});
  const auto strain_derivatives =
      std::integer_sequence<std::size_t, SOLUTION>{};
  strain_op->AddDomainIntegrator(
      strain_qfunc, tuple{Gradient<SOLUTION>{}, Gradient<NODAL_COORDS>{}},
      tuple{Identity<STRAIN>{}}, ir, everywhere, strain_derivatives);

  // Set up the constitutive operator
  constit_op = std::make_unique<ConstitutiveModel>(cmodel);

  // Set up the stress divergence operator
  fields = {{SOLUTION, &fe_space}};
  params = {{NODAL_COORDS, &coord_space}, {STRESS, &q_space_symr2}};
  stressdiv_op =
      std::make_unique<DifferentiableOperator>(fields, params, pmesh);
  StressDivergence balance_qfunc;
  const auto residual_derivatives =
      std::integer_sequence<std::size_t, STRESS>{};
  stressdiv_op->AddDomainIntegrator(
      balance_qfunc,
      tuple{Identity<STRESS>{}, Gradient<NODAL_COORDS>{}, Weight{}},
      tuple{Gradient<SOLUTION>{}}, ir, everywhere, residual_derivatives);
}

template <typename dscalar_t>
void LinearMomentumBalance<dscalar_t>::SetEssBdrConditions(
    const std::map<Array<int> *, VectorCoefficient *> &ebcs) {
  ParGridFunction ug(&fe_space);
  for (const auto [attr, coef] : ebcs) {
    ug.ProjectBdrCoefficient(*coef, *attr);
    Array<int> curr_ess_tdofs;
    fe_space.GetEssentialTrueDofs(*attr, curr_ess_tdofs);
    ess_tdofs.Append(curr_ess_tdofs);
  }
  ug.GetSubVector(ess_tdofs, ess_dof_vals);
}

template <typename dscalar_t>
void LinearMomentumBalance<dscalar_t>::Mult(const Vector &X, Vector &R) const {
  // displacement -> strain
  strain_op->Mult(X, _strain);

  // strain -> stress via NEML2
  constit_op->Mult(_strain, _stress);

  // stress -> residual
  stressdiv_op->SetParameters({&nodes, &_stress});
  stressdiv_op->Mult(_stress, R);

  // essential boundary conditions
  Vector X_ess;
  X.GetSubVector(ess_tdofs, X_ess);
  X_ess -= ess_dof_vals;
  R.SetSubVector(ess_tdofs, X_ess);
}

template <typename dscalar_t>
Operator &LinearMomentumBalance<dscalar_t>::GetGradient(const Vector &X) const {
  switch (derivative_type) {
  case HANDCODED:
    _man_deriv_op = std::make_unique<LinearMomentumBalanceJacobian>(*this, X);
    return *_man_deriv_op;
  case AUTODIFF:
  default:
    _autodiff_deriv_op =
        std::make_unique<AutodiffLinearMomentumBalanceJacobian>(this, X);
    return *_autodiff_deriv_op;
  }
}

template <typename dscalar_t>
LinearMomentumBalance<dscalar_t>::LinearMomentumBalanceJacobian::
    LinearMomentumBalanceJacobian(const LinearMomentumBalance &op,
                                  const Vector &X)
    : Operator(op.Height(), op.Width()), _op(op),
      _q_space_symr2(op.pmesh, op.ir, 6), _delta_strain(_q_space_symr2),
      _delta_stress(_q_space_symr2) {
  _delta_strain.UseDevice(true);
  _delta_stress.UseDevice(true);

  // Evaluate the tangent at the current state
  ParameterFunction strain(_q_space_symr2);
  _op.strain_op->SetParameters({&_op.nodes, &strain});
  _op.strain_op->Mult(X, strain);
  _op.constit_op->Tangent(strain, _tangent);
}

template <typename dscalar_t>
void LinearMomentumBalance<dscalar_t>::LinearMomentumBalanceJacobian::Mult(
    const Vector &dX, Vector &dR) const {
  // essential boundary conditions
  Vector dX_non_ess, dX_ess;
  dX_non_ess = dX;
  dX_non_ess.SetSubVector(_op.ess_tdofs, 0.0);
  dX.GetSubVector(_op.ess_tdofs, dX_ess);

  // dε = strain_op(dX)
  _op.strain_op->SetParameters({&_op.nodes, &_delta_strain});
  _op.strain_op->Mult(dX_non_ess, _delta_strain);

  // dσ = C(ε) : dε
  _op.constit_op->ApplyTangent(_tangent, _delta_strain, _delta_stress);

  // dR = stressdiv_op(dσ)
  _op.stressdiv_op->SetParameters({&_op.nodes, &_delta_stress});
  _op.stressdiv_op->Mult(_delta_stress, dR);

  // essential boundary conditions
  dR.SetSubVector(_op.ess_tdofs, dX_ess);
}

template <typename dscalar_t>
LinearMomentumBalance<dscalar_t>::AutodiffLinearMomentumBalanceJacobian::
    AutodiffLinearMomentumBalanceJacobian(
        const LinearMomentumBalance *momentum_balance, const Vector &x)
    : Operator(momentum_balance->Height()), momentum_balance(momentum_balance),
      _q_space_symr2(momentum_balance->pmesh, momentum_balance->ir, 6),
      _strain(_q_space_symr2), _stress(_q_space_symr2),
      _delta_strain(_q_space_symr2), _delta_stress(_q_space_symr2) {
  momentum_balance->u.SetFromTrueDofs(x);

  // Evaluate the tangent at the current state
  momentum_balance->strain_op->SetParameters(
      {&momentum_balance->nodes, &_strain});
  momentum_balance->strain_op->Mult(x, _strain);
  momentum_balance->constit_op->Tangent(_strain, _tangent);
  momentum_balance->constit_op->Mult(_strain, _stress);

  // One can retrieve the derivative of a DifferentiableOperator wrt a
  // field variable if the derivative has been requested during the
  // DifferentiableOperator::AddDomainIntegrator call.
  dstrain_du = momentum_balance->strain_op->GetDerivative(
      LinearMomentumBalance::SOLUTION, {&momentum_balance->u},
      {&momentum_balance->nodes, &_strain});
  dres_dstress = momentum_balance->stressdiv_op->GetDerivative(
      LinearMomentumBalance::STRESS, {&momentum_balance->u},
      {&momentum_balance->nodes, &_stress});
}

template <typename dscalar_t>
void LinearMomentumBalance<
    dscalar_t>::AutodiffLinearMomentumBalanceJacobian::Mult(const Vector &dX,
                                                            Vector &dR) const {
  Vector dX_non_ess = dX, dX_ess;
  dX_non_ess.SetSubVector(momentum_balance->ess_tdofs, 0.0);
  dX.GetSubVector(momentum_balance->ess_tdofs, dX_ess);

  dstrain_du->Mult(dX_non_ess, _delta_strain);
  momentum_balance->constit_op->ApplyTangent(_tangent, _delta_strain,
                                             _delta_stress);
  dres_dstress->Mult(_delta_stress, dR);

  // essential boundary conditions
  dR.SetSubVector(momentum_balance->ess_tdofs, dX_ess);
}
#ifdef MFEM_USE_ENZYME
template class LinearMomentumBalance<real_t>
#else
using mfem::future::dual;
using dual_t = dual<real_t, real_t>;
template class LinearMomentumBalance<dual_t>;
#endif
} // namespace op
