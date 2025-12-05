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

#include "mfem.hpp"
#include "neml2/models/Model.h"

namespace op {

using namespace mfem;
using namespace mfem::future;

/// @brief Constitutive model wrapper for NEML2 models
class ConstitutiveModel {
public:
  ConstitutiveModel(std::shared_ptr<neml2::Model> cmodel);

  /**
   * @brief Perform stress update
   *
   * @param strain Input strain
   * @param stress Output stress
   */
  void Mult(ParameterFunction &strain, ParameterFunction &stress) const;

  /**
   * @brief Compute material tangent
   *
   * @param strain Input strain
   * @param tangent Output tangent (dstress/dstrain) evaluated at strain
   */
  void Tangent(ParameterFunction &strain, neml2::Tensor &tangent) const;

  /**
   * @brief Apply material tangent to delta strain (or anything in the strain
   * space)
   *
   * @param tangent Material tangent (dstress/dstrain)
   * @param dstrain Input delta strain
   * @param dstress Output delta stress
   */
  void ApplyTangent(const neml2::Tensor &tangent, ParameterFunction &dstrain,
                    ParameterFunction &dstress) const;

private:
  /// The NEML2 constitutive model being wrapped
  std::shared_ptr<neml2::Model> _cmodel;

  /// Name of the strain variable in the NEML2 model
  const neml2::VariableName _strain_name;

  /// Name of the stress variable in the NEML2 model
  const neml2::VariableName _stress_name;
};

} // namespace op
