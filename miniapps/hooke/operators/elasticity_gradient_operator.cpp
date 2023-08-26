// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "elasticity_gradient_operator.hpp"
#include "elasticity_operator.hpp"

namespace mfem
{
ElasticityGradientOperator::ElasticityGradientOperator(ElasticityOperator &op)
   : Operator(op.Height()), elasticity_op_(op) {}

void ElasticityGradientOperator::Mult(const Vector &x, Vector &y) const
{
   elasticity_op_.GradientMult(x, y);
}

void ElasticityGradientOperator::AssembleGradientDiagonal(
   Vector &Ke_diag, Vector &K_diag_local, Vector &K_diag) const
{
   static_cast<ElasticityOperator &>(elasticity_op_)
   .AssembleGradientDiagonal(Ke_diag, K_diag_local, K_diag);
}

} // namespace mfem
