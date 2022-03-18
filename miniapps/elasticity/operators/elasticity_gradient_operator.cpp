#include "elasticity_gradient_operator.hpp"
#include "elasticity_operator.hpp"

namespace mfem
{
ElasticityGradientOperator::ElasticityGradientOperator(ElasticityOperator &op)
   : Operator(op.Height()), elasticity_op_(op)
{

}

void ElasticityGradientOperator::Mult(const Vector &x, Vector &y) const
{
   elasticity_op_.GradientMult(x, y);
}

void ElasticityGradientOperator::AssembleGradientDiagonal(Vector &Ke_diag,
                                                          Vector &K_diag_local,
                                                          Vector &K_diag) const
{
   static_cast<ElasticityOperator &>(elasticity_op_).AssembleGradientDiagonal(
      Ke_diag, K_diag_local, K_diag);
}

}