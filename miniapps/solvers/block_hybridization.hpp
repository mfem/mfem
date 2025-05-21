#ifndef MFEM_BH_SOLVER_HPP
#define MFEM_BH_SOLVER_HPP

#include "darcy_solver.hpp"

namespace mfem
{
namespace blocksolvers
{

class BlockHybridizationSolver : public DarcySolver
{
   ParFiniteElementSpace *hdiv_space;
   ParFiniteElementSpace *l2_space;
   ParFiniteElementSpace *multiplier_space;

   CGSolver solver_;

   DenseMatrix *saved_hdiv_matrices;
   DenseMatrix *saved_mixed_matrices;
   DenseMatrix *saved_l2_matrices;
   Array<int> *interior_indices;

   HypreParMatrix *pH;
   HypreBoomerAMG *preconditioner;

public:
   BlockHybridizationSolver(ParBilinearForm *mVarf,
                            ParMixedBilinearForm *bVarf,
                            IterSolveParameters param);
   ~BlockHybridizationSolver();
   void Mult(const Vector &x, Vector&y) const override { }
   void SetOperator(const Operator &op) override { }
   int GetNumIterations() const override { return solver_.GetNumIterations(); }
};

} // namespace blocksolvers
} // namespace mfem

#endif // MFEM_BH_SOLVER_HPP
