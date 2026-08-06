#ifndef PAR_MASS_MATRIX_SOLVER_HPP
#define PAR_MASS_MATRIX_SOLVER_HPP

#include "mfem.hpp"

#include <memory>

/**
 * @brief Parallel mass-matrix solver with an mfem::Solver interface.
 *
 * The solver applies conjugate gradients to the partially assembled operator
 *
 *     M_ij = integral_Omega rho phi_i phi_j dx
 *
 * using an OperatorJacobiSmoother built from the partial-assembly diagonal.
 * Both arguments to Mult() use parallel true-DOF layout.
 *
 * When MFEM is configured with a GPU backend and the application creates an
 * mfem::Device before constructing the finite element objects, the
 * partial-assembly kernels, Jacobi application, vector operations, and CG
 * operator applications execute through MFEM's device backend. Mult() remains
 * a host-callable orchestration method because it performs MPI collectives; it
 * is not a function that can be invoked from inside a device kernel.
 *
 * Construction, ReassembleOperator(), Mult(), and destruction must occur on
 * every rank in the finite-element-space communicator, in the same order.
 * Constructor solver parameters and the inherited iterative_mode flag must
 * also agree on every rank. Diagnostic getters are local and non-collective.
 */
class ParMassMatrixSolver : public mfem::Solver
{
public:
   /**
    * @brief Construct with a caller-owned mass coefficient.
    *
    * The finite element space and coefficient must outlive this solver.
    * Construction is collective over @a fes.GetComm().
    *
    * @param fes Parallel finite element space. It must outlive the solver.
    * @param mass_coefficient Borrowed coefficient. It must outlive the solver.
    * @param rel_tol Relative CG convergence tolerance.
    * @param max_iter Maximum number of CG iterations.
    * @param print_level MFEM iterative-solver print level.
    */
   ParMassMatrixSolver(mfem::ParFiniteElementSpace &fes,
                       mfem::Coefficient &mass_coefficient,
                       mfem::real_t rel_tol = 1e-12,
                       int max_iter = 500,
                       int print_level = 0);

   /**
    * @brief Construct by sharing ownership of a mass coefficient.
    *
    * The shared pointer is copied into the solver, keeping the coefficient
    * alive until both the solver and all other owners release it.
    * Construction is collective over @a fes.GetComm().
    *
    * @param fes Parallel finite element space. It must outlive the solver.
    * @param mass_coefficient Non-null shared coefficient ownership.
    * @param rel_tol Relative CG convergence tolerance.
    * @param max_iter Maximum number of CG iterations.
    * @param print_level MFEM iterative-solver print level.
    */
   ParMassMatrixSolver(
      mfem::ParFiniteElementSpace &fes,
      std::shared_ptr<mfem::Coefficient> mass_coefficient,
      mfem::real_t rel_tol = 1e-12,
      int max_iter = 500,
      int print_level = 0);

   /**
    * @brief Construct with an internally owned constant mass coefficient.
    *
    * Construction is collective over @a fes.GetComm().
    *
    * @param fes Parallel finite element space. It must outlive the solver.
    * @param coefficient_value Constant mass coefficient value.
    * @param rel_tol Relative CG convergence tolerance.
    * @param max_iter Maximum number of CG iterations.
    * @param print_level MFEM iterative-solver print level.
    */
   ParMassMatrixSolver(mfem::ParFiniteElementSpace &fes,
                       mfem::real_t coefficient_value,
                       mfem::real_t rel_tol = 1e-12,
                       int max_iter = 500,
                       int print_level = 0);

   /**
    * @brief Release the Krylov solver, preconditioner, operator, form, and any
    *        shared coefficient ownership.
    *
    * The destructor itself performs local releases. Solver lifetimes must still
    * end in a consistent phase on all ranks to avoid later unmatched calls.
    */
   ~ParMassMatrixSolver() override;

   /// Copy construction is disabled because the solver owns parallel state.
   ParMassMatrixSolver(const ParMassMatrixSolver &) = delete;

   /// Copy assignment is disabled because the solver owns parallel state.
   ParMassMatrixSolver &operator=(const ParMassMatrixSolver &) = delete;

   /**
    * @brief Solve M x = rhs on parallel true DOFs.
    *
    * This call is collective. If iterative_mode is false, x is resized and a
    * zero initial guess is used. If iterative_mode is true, x must already have
    * the local true-DOF size and is used as the initial guess.
    *
    * @param rhs Input true-DOF right-hand-side vector.
    * @param x Output true-DOF solution vector, or initial guess and output when
    *          iterative_mode is true.
    */
   void Mult(const mfem::Vector &rhs, mfem::Vector &x) const override;

   /**
    * @brief Return the preferred memory class of the assembled mass operator.
    *
    * MFEM iterative solvers use this value to allocate their work vectors in a
    * memory space compatible with the configured CPU or GPU backend.
    *
    * @return Memory class reported by the current true-DOF mass operator.
    */
   mfem::MemoryClass GetMemoryClass() const override;

   /**
    * @brief Rebuild the mass operator and preconditioner.
    *
    * This call is collective and is required after the borrowed coefficient
    * changes. Rebuild the solver instead if the finite element space changes.
    * Shared and internally owned coefficients may also be modified in place
    * before this call.
    */
   void ReassembleOperator();

   /**
    * @brief Reject replacement through the generic mfem::Solver interface.
    *
    * @param op Ignored external operator.
    */
   void SetOperator(const mfem::Operator &op) override;

   /// Return the CG iteration count from the most recent Mult() call.
   int GetNumIterations() const { return last_num_iterations_; }

   /// Return the final CG residual norm from the most recent Mult() call.
   mfem::real_t GetFinalResidual() const { return last_final_residual_; }

   /// Return whether the most recent Mult() call converged.
   bool GetConverged() const { return last_converged_; }

   /// Return the currently assembled true-DOF mass operator.
   const mfem::Operator &GetMassOperator() const { return *mass_operator_; }

private:
   /// Validate constructor state and collectively assemble the initial operator.
   void Initialize();

   /**
    * @brief Verify that all ranks selected the same constructor overload.
    *
    * @param constructor_kind Integer identifier assigned to an overload.
    */
   void VerifyConstructorKind(int constructor_kind) const;

   /// Verify rank-invariant CG parameters and their local validity.
   void VerifyCollectiveConfiguration() const;

   /**
    * @brief Convert a local precondition into a communicator-wide assertion.
    *
    * @param local_condition Condition evaluated independently on each rank.
    * @param message Error message used when any rank reports false.
    */
   void CollectiveVerify(bool local_condition, const char *message) const;

   mfem::ParFiniteElementSpace &fes_;
   MPI_Comm comm_;

   std::shared_ptr<mfem::Coefficient> owned_coefficient_;
   mfem::Coefficient *mass_coefficient_ = nullptr;

   mfem::real_t rel_tol_;
   int max_iter_;
   int print_level_;

   // The declaration order is intentional. Default destruction runs in reverse:
   // CG -> preconditioner -> system operator -> form -> marker -> coefficient.
   mfem::Array<int> empty_ess_tdofs_;
   std::unique_ptr<mfem::ParBilinearForm> mass_form_;
   mfem::OperatorPtr mass_operator_;
   std::unique_ptr<mfem::OperatorJacobiSmoother> preconditioner_;
   mutable mfem::CGSolver cg_;

   mutable int last_num_iterations_ = 0;
   mutable mfem::real_t last_final_residual_ = 0.0;
   mutable bool last_converged_ = false;
};

#endif // PAR_MASS_MATRIX_SOLVER_HPP
