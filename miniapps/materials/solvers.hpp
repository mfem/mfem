#ifndef MATERIALS_SOLVERS_HPP
#define MATERIALS_SOLVERS_HPP

#include "mfem.hpp"

namespace mfem {
namespace materials {

/// PDE solver for equations of type (div \Theta grad + \alpha I) u = \beta f.
class PDESolver {
public:
  /// Constructor. PDE solver for equations of type
  /// (div \Theta grad + c I) u = f.
  /// @param diff_coefficient The diffusion coefficient \Theta.
  /// @param ess_tdof_list Boundary conditions.
  /// @param fespace Finite element space.
  PDESolver(MatrixConstantCoefficient &diff_coefficient,
            const Array<int> &ess_tdof_list, ParFiniteElementSpace *fespace);

  /// Solve the PDE (div \Theta grad + \alpha I) x = \beta b.
  void Solve(const LinearForm &b, GridFunction &x, double alpha, double beta,
             int exponent = 1);

  /// Writes the solution of the PDE from the previous call to Solve() to the
  /// linear from b (with appropriate transformations).
  void UpdateRHS(LinearForm &b);

  /// Activate repeated solve capabilities. E.g. if the PDE is of the form
  /// A^N x = b. This method solves the PDE A x = b for the first time, and
  /// then uses the solution as RHS for the next solve and so forth.
  void ActivateRepeatedSolve() { repeated_solve_ = true; }

  /// Single solve only.
  void DeactivateRepeatedSolve() { repeated_solve_ = false; }

private:
  // Solve the linear system Op_ X_ = B_ with a PCG solver and hypre's
  // BoomerAMG implementation as pre-conditioner.
  void SolveLinearSystem();

  // Bilinear forms and corresponding matrices for the solver.
  ParBilinearForm k_;
  ParBilinearForm m_;
  HypreParMatrix stiffness_;
  HypreParMatrix mass_bc_;
  HypreParMatrix mass_0_;

  // Transformation matrices (needed to construct the linear systems and
  // solutions)
  const SparseMatrix *restriction_matrix_;
  const Operator *prolongation_matrix_;

  // Members to solve the linear system.
  Vector X_;
  Vector B_;
  HypreParMatrix *Op_;

  // Information of the finite element space.
  const Array<int> &ess_tdof_list_;
  ParFiniteElementSpace *fespace_ptr_;

  // Member to switch to repeated solve capabilities.
  bool repeated_solve_ = false;
};

} // namespace materials
} // namespace mfem

#endif // MATERIALS_SOLVERS_HPP
