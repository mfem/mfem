#ifndef MFEM_DIFFUSION_MASS_GMG_HPP
#define MFEM_DIFFUSION_MASS_GMG_HPP

#include "mfem.hpp"

namespace mfem
{

/// mfem::Solver-compatible geometric multigrid preconditioner for
///     -div(diffusion_coefficient grad u) + mass_coefficient u.
///
/// The operator on every level is assembled with partial assembly.  The level
/// smoothers are Jacobi smoothers wrapped by SymmetrizedSmoother, and the
/// coarsest level uses the same symmetrized smoother as an inexpensive coarse
/// solver.  The class derives from GeometricMultigrid, and therefore from
/// mfem::Solver, so it can be passed directly to MFEM iterative solvers as a
/// preconditioner.  Coefficients are referenced, not owned; callers must keep
/// them alive while the solver is used.
class DiffusionMassGeometricMultigrid : public GeometricMultigrid
{
public:
   enum class CoarseSolver
   {
      SymmetrizedJacobi,
      BoomerAMG
   };

   DiffusionMassGeometricMultigrid(ParFiniteElementSpaceHierarchy &hierarchy,
                                   const Array<int> &ess_bdr,
                                   Coefficient &diffusion_coefficient,
                                   Coefficient &mass_coefficient,
                                   real_t jacobi_damping = 1.0,
                                   CoarseSolver coarse_solver =
                                      CoarseSolver::SymmetrizedJacobi);

   /// Rebuild all level operators and smoothers with a new diffusion coefficient.
   void SetDiffusionCoefficient(Coefficient &coefficient);

   /// Rebuild all level operators and smoothers with a new mass coefficient.
   void SetMassCoefficient(Coefficient &coefficient);

   /// Return the finite element hierarchy used to define the geometric levels.
   ParFiniteElementSpaceHierarchy &GetHierarchy();
   const ParFiniteElementSpaceHierarchy &GetHierarchy() const;

   /// Access the partial-assembly bilinear form at a multigrid level.
   ParBilinearForm &GetLevelBilinearForm(int level);
   const ParBilinearForm &GetLevelBilinearForm(int level) const;

   /// Access the essential true dofs used at a multigrid level.
   const Array<int> &GetEssentialTrueDofsAtLevel(int level) const;

   /// Reassemble the hierarchy after external coefficient state changes.
   void Rebuild();

   /// Select the solver used on the coarsest multigrid level and rebuild.
   void SetCoarseSolver(CoarseSolver coarse_solver);

   /// Return the currently selected coarsest-level solver type.
   CoarseSolver GetCoarseSolver() const { return coarse_solver_; }

   /// IterativeSolver calls SetOperator on its preconditioner.  The geometric
   /// hierarchy already owns its level operators, so this validates dimensions
   /// and otherwise leaves the hierarchy unchanged.
   void SetOperator(const Operator &op) override;

   /// Form/recover helpers that also support empty essential boundary lists.
   void FormFineLinearSystem(Vector &x, Vector &b, OperatorHandle &A,
                             Vector &X, Vector &B);
   void RecoverFineFEMSolution(const Vector &X, const Vector &b, Vector &x);

private:
   const Array<int> &LevelEssentialTrueDofs(int level) const;
   void ClearLevelData();
   void ConstructLevelOperatorAndSmoother(ParFiniteElementSpace &fespace,
                                          int level);
   Solver *ConstructSymmetrizedJacobiSmoother(ParBilinearForm &form,
                                              Operator &op,
                                              const Array<int> &ess_tdofs);
   Solver *ConstructBoomerAMGCoarseSolver(ParFiniteElementSpace &fespace,
                                          const Array<int> &ess_tdofs);

   ParFiniteElementSpaceHierarchy &hierarchy_;
   Coefficient *diffusion_coefficient_;
   Coefficient *mass_coefficient_;
   real_t jacobi_damping_;
   CoarseSolver coarse_solver_;
   bool level_operator_owner_;
   Array<int> empty_ess_tdofs_;
};

} // namespace mfem

#endif
