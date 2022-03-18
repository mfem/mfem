#ifndef MFEM_ELASTICITY_DIAGONAL_PC_HPP
#define MFEM_ELASTICITY_DIAGONAL_PC_HPP

#include "operators/elasticity_gradient_operator.hpp"

namespace mfem
{
/**
 * @brief ElasticityDiagonalPreconditioner acts as a matrix-free preconditioner
 * for ElasticityOperator.
 *
 * @note There are two types to choose from
 * - Diagonal: A classic Jacobi type preconditioner
 * - BlockDiagonal: A Jacobi type preconditioner which calculates the diagonal
 *   contribution of ElasticityOperator on each diagonal element and applies
 *   it's inverted submatrix.
 */
class ElasticityDiagonalPreconditioner : public Solver
{
   static constexpr int dim = 3;

public:
   enum Type
   {
      Diagonal = 0,
      BlockDiagonal
   };

   ElasticityDiagonalPreconditioner(Type type = Type::Diagonal)
      : Solver(), type_(type) {}

   void SetOperator(const Operator &op) override;

   void Mult(const Vector &x, Vector &y) const override;

private:
   const ElasticityGradientOperator *gradient_operator_;
   int num_submats_, submat_height_;
   Vector Ke_diag_, K_diag_local_, K_diag_;
   Type type_;
};
}
#endif