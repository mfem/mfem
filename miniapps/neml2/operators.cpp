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

LinearMomentumBalance::LinearMomentumBalance(
    ParFiniteElementSpace &fe_space, const IntegrationRule &ir,
    std::shared_ptr<neml2::Model> cmodel)
    : Operator(fe_space.GetTrueVSize(), fe_space.GetTrueVSize()),
      fe_space(fe_space), ir(ir), pmesh(*fe_space.GetParMesh()),
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
  strain_op->AddDomainIntegrator(
      strain_qfunc, tuple{Gradient<SOLUTION>{}, Gradient<NODAL_COORDS>{}},
      tuple{Identity<STRAIN>{}}, ir, everywhere);

  // Set up the constitutive operator
  constit_op = std::make_unique<ConstitutiveModel>(cmodel);

  // Set up the stress divergence operator
  fields = {{RESIDUAL, &fe_space}};
  params = {{NODAL_COORDS, &coord_space}, {STRESS, &q_space_symr2}};
  stressdiv_op =
      std::make_unique<DifferentiableOperator>(fields, params, pmesh);
  StressDivergence balance_qfunc;
  stressdiv_op->AddDomainIntegrator(
      balance_qfunc,
      tuple{Identity<STRESS>{}, Gradient<NODAL_COORDS>{}, Weight{}},
      tuple{Gradient<RESIDUAL>{}}, ir, everywhere);
}

void LinearMomentumBalance::SetEssBdrConditions(
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

void LinearMomentumBalance::Mult(const Vector &X, Vector &R) const {
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

Operator &LinearMomentumBalance::GetGradient(const Vector &X) const {
  _deriv_op = std::make_unique<LinearMomentumBalanceJacobian>(*this, X);
  return *_deriv_op;
}

LinearMomentumBalanceJacobian::LinearMomentumBalanceJacobian(
    const LinearMomentumBalance &op, const Vector &X)
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

void LinearMomentumBalanceJacobian::Mult(const Vector &dX, Vector &dR) const {
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

} // namespace op
