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

#ifndef MFEM_ELASTICITY_GRADIENT_OP_HPP
#define MFEM_ELASTICITY_GRADIENT_OP_HPP

#include "elasticity_operator.hpp"
#include "../materials/gradient_type.hpp"

namespace mfem
{
/**
 * @brief ElasticityGradientOperator is a wrapper class to pass
 * ElasticityOperator::AssembleGradientDiagonal and
 * ElasticityOperator::GradientMult as a separate object through NewtonSolver.
 */
class ElasticityGradientOperator : public Operator
{
public:
   ElasticityGradientOperator(ElasticityOperator &op);

   void AssembleGradientDiagonal(Vector &Ke_diag, Vector &K_diag_local,
                                 Vector &K_diag) const;

   void Mult(const Vector &x, Vector &y) const override;

   ElasticityOperator &elasticity_op_;
};

} // namespace mfem

#endif
