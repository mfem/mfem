// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

#include "transformation.hpp"
#include <cmath>
#include <functional>

namespace mfem
{
namespace materials
{

/// This function effectively implements equation 19 of the paper (see header).
///  `\Phi [y(x)]`
double TransformToUniform(double x) {
  return std::erfc(-x/std::sqrt(2))/2;
}

void UniformGRFTransformer::Transform(ParGridFunction &x) const {
  GridFunctionCoefficient gf_coeff(&x);
  ConstantCoefficient factor(max_-min_);
  ConstantCoefficient summand(min_);
  TransformedCoefficient transformation_coeff(&gf_coeff, TransformToUniform);
  ProductCoefficient product_coeff(transformation_coeff, factor);
  SumCoefficient sum_coeff(product_coeff, summand);
  ParGridFunction xx(x);
  xx.ProjectCoefficient(sum_coeff);
  x = xx;
}

void OffsetTransformer::Transform(ParGridFunction &x) const {
  ConstantCoefficient offset(offset_);
  ParGridFunction xx(x);
  xx.ProjectCoefficient(offset);
  x += xx;
}

void ScaleTransformer::Transform(ParGridFunction &x) const {
  x *= scale_;
}

} // namespace materials
} // namespace mfem
