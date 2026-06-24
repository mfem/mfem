#pragma once

#include "mfem.hpp"

namespace mfem
{

/// @brief Parallel mass matrix solver using partial assembly and a diagonal
///        (Jacobi) preconditioner.
///
/// Solves  M x = b  where M is the (optionally weighted) L2/H1 mass matrix
/// assembled on a ParFiniteElementSpace.
///
/// Two construction paths:
///  1. General  – pass an arbitrary Coefficient* (caller retains ownership).
///  2. Constant – pass a real_t scalar; the solver owns the internal
///               ConstantCoefficient it creates.
///
/// The CG solver uses partial assembly (no explicit sparse matrix) and an
/// OperatorJacobiSmoother (PA diagonal) as preconditioner.
class MassMatrixSolver
{
public:
   // -----------------------------------------------------------------------
   // Constructor 1 – general coefficient (caller owns coeff)
   // -----------------------------------------------------------------------
   /// @param pfes         Parallel FE space (not owned).
   /// @param coeff        Mass coefficient (not owned, may be nullptr for unit
   ///                     weight).
   /// @param rel_tol      Relative CG residual tolerance  (default 1e-6).
   /// @param max_iter     Maximum CG iterations           (default 500).
   /// @param print_level  IterativeSolver verbosity level  (default 0).
   MassMatrixSolver(ParFiniteElementSpace *pfes,
                    Coefficient           *coeff,
                    real_t                 rel_tol     = 1e-6,
                    int                    max_iter    = 500,
                    int                    print_level = 0);

   // -----------------------------------------------------------------------
   // Constructor 2 – constant coefficient (solver owns internal coeff)
   // -----------------------------------------------------------------------
   /// @param pfes         Parallel FE space (not owned).
   /// @param coeff_val    Constant mass coefficient value.
   /// @param rel_tol      Relative CG residual tolerance  (default 1e-6).
   /// @param max_iter     Maximum CG iterations           (default 500).
   /// @param print_level  IterativeSolver verbosity level  (default 0).
   MassMatrixSolver(ParFiniteElementSpace *pfes,
                    real_t                 coeff_val,
                    real_t                 rel_tol     = 1e-6,
                    int                    max_iter    = 500,
                    int                    print_level = 0);

   ~MassMatrixSolver();

   // Non-copyable
   MassMatrixSolver(const MassMatrixSolver &)            = delete;
   MassMatrixSolver &operator=(const MassMatrixSolver &) = delete;

   // -----------------------------------------------------------------------
   // Solve  M x = b  (true-dof vectors)
   // -----------------------------------------------------------------------
   /// @param b  RHS vector in true-dof layout.
   /// @param x  Solution vector (also used as initial guess when iterative_mode
   ///           is enabled).
   void Solve(const Vector &b, Vector &x) const;

   // -----------------------------------------------------------------------
   // Diagnostics
   // -----------------------------------------------------------------------
   int    GetNumIterations()  const { return last_num_iter_;  }
   real_t GetFinalResidual()  const { return last_final_res_; }
   bool   GetConverged()      const { return last_converged_; }

private:
   /// Shared setup called by both constructors.
   void Init(ParFiniteElementSpace *pfes,
             Coefficient           *coeff,
             real_t                 rel_tol,
             int                    max_iter,
             int                    print_level);

   // Ownership
   bool         owns_coeff_    {false};
   Coefficient *owned_coeff_ptr_{nullptr}; ///< Non-null only when owns_coeff_

   // MFEM objects (all owned by this class)
   ParBilinearForm        *pblf_ {nullptr};
   OperatorPtr             mass_oper_;
   OperatorJacobiSmoother *prec_ {nullptr};
   CGSolver               *cg_   {nullptr};

   // Diagnostics (mutable so Solve() can be const)
   mutable int    last_num_iter_  {0};
   mutable real_t last_final_res_ {0.0};
   mutable bool   last_converged_ {false};
};

} // namespace mfem
