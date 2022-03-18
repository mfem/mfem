#ifndef MFEM_ELASTICITY_GRADIENT_OP_HPP
#define MFEM_ELASTICITY_GRADIENT_OP_HPP

#include "elasticity_operator.hpp"
#include "materials/gradient_type.hpp"

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
}
#endif