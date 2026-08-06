#ifndef PAR_DIFFUSION_SOLVER_PA_HPP
#define PAR_DIFFUSION_SOLVER_PA_HPP

#include "mfem.hpp"

#include <memory>
#include <vector>

/**
 * @brief Device-capable partial-assembly replacement for ParDiffusionSolver.
 *
 * The solver implements the same construction/add-boundary/Assemble/Mult
 * workflow as ParDiffusionSolver, but stores an MFEM partial-assembly operator
 * instead of HypreParMatrix objects. Dirichlet conditions are imposed by an
 * mfem::ConstrainedOperator. CG is preconditioned with an assembled
 * low-order-refined operator and HypreBoomerAMG. With no Dirichlet conditions,
 * the right-hand side
 * is projected to the compatible pure-Neumann range and the solution is shifted
 * to zero integral mean.
 *
 * Mult() is host-called because it coordinates MPI reductions and Krylov
 * iterations. High-order operator, vector, and boundary-elimination kernels
 * use the MFEM device selected by the application; the LOR auxiliary matrix is
 * handled by HypreBoomerAMG.
 */
class ParDiffusionSolverPA : public mfem::Solver
{
public:
   /**
    * @brief Construct with a constant diffusion coefficient owned by the solver.
    * @param fes Parallel scalar H1 finite element space; must outlive the solver.
    * @param diffusion_coefficient Positive constant diffusion coefficient.
    * @param rel_tol Relative CG tolerance.
    * @param max_iter Maximum CG iterations.
    * @param print_level CG and LOR-AMG verbosity.
    */
   ParDiffusionSolverPA(mfem::ParFiniteElementSpace &fes,
                        mfem::real_t diffusion_coefficient,
                        mfem::real_t rel_tol = 1e-12,
                        int max_iter = 500,
                        int print_level = 0);

   /**
    * @brief Construct with a borrowed coefficient.
    * @param fes Parallel scalar H1 finite element space; must outlive the solver.
    * @param diffusion_coefficient Borrowed coefficient; must outlive the solver.
    * @param rel_tol Relative CG tolerance.
    * @param max_iter Maximum CG iterations.
    * @param print_level CG and LOR-AMG verbosity.
    */
   ParDiffusionSolverPA(mfem::ParFiniteElementSpace &fes,
                        mfem::Coefficient &diffusion_coefficient,
                        mfem::real_t rel_tol = 1e-12,
                        int max_iter = 500,
                        int print_level = 0);

   /**
    * @brief Construct while sharing ownership of a coefficient.
    * @param fes Parallel scalar H1 finite element space; must outlive the solver.
    * @param diffusion_coefficient Non-null shared coefficient.
    * @param rel_tol Relative CG tolerance.
    * @param max_iter Maximum CG iterations.
    * @param print_level CG and LOR-AMG verbosity.
    */
   ParDiffusionSolverPA(
      mfem::ParFiniteElementSpace &fes,
      std::shared_ptr<mfem::Coefficient> diffusion_coefficient,
      mfem::real_t rel_tol = 1e-12,
      int max_iter = 500,
      int print_level = 0);

   /// @brief Release PA, LOR, AMG, Krylov, and shared coefficient resources.
   ~ParDiffusionSolverPA() override = default;

   /// @brief Disable copying of communicator-bound solver state.
   ParDiffusionSolverPA(const ParDiffusionSolverPA &) = delete;

   /// @brief Disable assignment of communicator-bound solver state.
   ParDiffusionSolverPA &operator=(const ParDiffusionSolverPA &) = delete;

   /**
    * @brief Reject external replacement of the internally assembled operator.
    * @param op Ignored operator required by the mfem::Solver interface.
    */
   void SetOperator(const mfem::Operator &op) override;

   /// @brief Return the memory class used by the current CPU/GPU system operator.
   mfem::MemoryClass GetMemoryClass() const override;

   /**
    * @brief Add or replace borrowed coefficient-valued Dirichlet data.
    * @param boundary_attribute One-based MFEM boundary attribute.
    * @param coefficient Borrowed boundary coefficient, needed through Assemble().
    */
   void AddBoundaryCondition(int boundary_attribute,
                             mfem::Coefficient &coefficient);

   /**
    * @brief Add or replace an internally owned constant Dirichlet value.
    * @param boundary_attribute One-based MFEM boundary attribute.
    * @param value Constant boundary value.
    */
   void AddBoundaryCondition(int boundary_attribute, mfem::real_t value);

   /// @brief Remove all recorded Dirichlet boundary conditions.
   void ClearBoundaryConditions();

   /// @brief Rebuild boundary data, constrained operator, LOR-AMG, and CG.
   void Assemble();

   /// @brief Rebuild partial-assembly diffusion data after coefficient changes.
   void ReassembleOperator();

   /**
    * @brief Solve using true-DOF vectors.
    * @param rhs Input distributed true-DOF right-hand side.
    * @param x Output distributed true-DOF solution.
    */
   void Mult(const mfem::Vector &rhs, mfem::Vector &x) const override;

   /**
    * @brief Form the eliminated Dirichlet or compatible Neumann system RHS.
    * @param rhs Input true-DOF load.
    * @param system_rhs Output load used by the current system operator.
    */
   void FormSystemRHS(const mfem::Vector &rhs,
                      mfem::Vector &system_rhs) const;

   /**
    * @brief Project a true-DOF load into the pure-Neumann compatible range.
    * @param rhs Input true-DOF load.
    * @param projected_rhs Output load with zero total constant-mode component.
    */
   void ProjectRHS(const mfem::Vector &rhs,
                   mfem::Vector &projected_rhs) const;

   /// @brief Return true when Assemble() selected mean-free Neumann mode.
   bool UsesMeanFreeMode() const;

   /// @brief Return true when essential true DOFs are currently assembled.
   bool HasEssentialBoundaryConditions() const;

   /// @brief Return true when Assemble() is required before Mult().
   bool NeedsAssembly() const { return needs_assembly_; }

   /// @brief Return the number of recorded boundary-attribute conditions.
   int GetNumBoundaryConditions() const;

   /// @brief Return the current constrained or unconstrained true-DOF PA operator.
   const mfem::Operator &GetSystemOperator() const;

   /// @brief Return the unconstrained true-DOF PA diffusion operator.
   const mfem::Operator &GetFullOperator() const;

   /// @brief Return the current essential true-DOF list.
   const mfem::Array<int> &GetEssentialTrueDofs() const;

   /// @brief Return the current boundary-attribute marker.
   const mfem::Array<int> &GetBoundaryAttributeMarker() const;

   /// @brief Return prescribed values in true-DOF layout.
   const mfem::Vector &GetEssentialTrueDofValues() const;

   /// @brief Return the true-DOF vector representing the constant function one.
   const mfem::Vector &GetConstantMode() const { return z_; }

   /// @brief Return the true-DOF vector representing integration against one.
   const mfem::Vector &GetMassVector() const { return m_; }

   /// @brief Return the global domain measure.
   mfem::real_t GetVolume() const { return volume_; }

   /**
    * @brief Return the integral mean of a true-DOF finite-element function.
    * @param x Distributed true-DOF coefficient vector.
    */
   mfem::real_t Mean(const mfem::Vector &x) const;

   /**
    * @brief Return the total load against the constant function.
    * @param rhs Distributed true-DOF load vector.
    */
   mfem::real_t TotalLoad(const mfem::Vector &rhs) const;

   /// @brief Return iterations from the most recent solve.
   int GetNumIterations() const { return last_num_iterations_; }

   /// @brief Return the final residual norm from the most recent solve.
   mfem::real_t GetFinalResidual() const { return last_final_residual_; }

   /// @brief Return convergence status from the most recent solve.
   bool GetConverged() const { return last_converged_; }

private:
   /// @brief Stored description and optional ownership for one boundary value.
   struct BoundaryConditionEntry
   {
      int boundary_attribute = 0;
      mfem::Coefficient *coefficient = nullptr;
      std::shared_ptr<mfem::Coefficient> owned_coefficient;
   };

   /**
    * @brief Validate common state and build the initial Neumann solver.
    * @param constructor_kind Identifier used to check overload agreement.
    */
   void Initialize(int constructor_kind);

   /// @brief Verify collectively that all ranks selected the same constructor.
   void VerifyConstructorKind(int constructor_kind) const;

   /// @brief Verify collectively that Krylov parameters agree on all ranks.
   void VerifyCollectiveConfiguration() const;

   /**
    * @brief Convert a rank-local condition into a communicator-wide assertion.
    * @param condition Rank-local condition.
    * @param message Failure message if any rank reports false.
    */
   void CollectiveVerify(bool condition, const char *message) const;

   /// @brief Return the maximum one-based boundary attribute, or zero.
   int MaxBoundaryAttribute() const;

   /// @brief Verify that a one-based boundary attribute exists on the mesh.
   void ValidateBoundaryAttribute(int boundary_attribute) const;

   /// @brief Remove any recorded condition for one boundary attribute.
   void RemoveBoundaryCondition(int boundary_attribute);

   /// @brief Project boundary values and build markers and essential true DOFs.
   void BuildBoundaryValuesAndMarkers();

   /// @brief Build LOR-AMG and configure the high-order PA CG solver.
   void ConfigureLinearSolver();

   /// @brief Build the constant null mode, integration vector, and volume.
   void BuildConstantModeAndMassVector();

   /// @brief Remove the incompatible constant-load component in place.
   void MakeCompatible(mfem::Vector &rhs) const;

   /// @brief Shift a Neumann solution to zero integral mean.
   void SetZeroMean(mfem::Vector &x) const;

   /// @brief Overwrite essential true DOFs with prescribed values on device.
   void CopyEssentialValues(mfem::Vector &x) const;

   mfem::ParFiniteElementSpace &fes_;
   MPI_Comm comm_;
   std::shared_ptr<mfem::Coefficient> owned_diffusion_coefficient_;
   mfem::Coefficient *diffusion_coefficient_ = nullptr;
   mfem::real_t rel_tol_;
   int max_iter_;
   int print_level_;

   std::vector<BoundaryConditionEntry> boundary_conditions_;
   mfem::Array<int> bdr_attr_marker_;
   mfem::Array<int> ess_tdof_list_;
   mfem::Array<int> empty_tdof_list_;

   std::unique_ptr<mfem::ParBilinearForm> diffusion_form_;
   mfem::OperatorPtr full_operator_;
   mfem::OperatorPtr system_operator_;
   std::unique_ptr<mfem::ParLORDiscretization> lor_discretization_;
   std::unique_ptr<mfem::HypreBoomerAMG> lor_amg_;
   std::unique_ptr<mfem::Solver> preconditioner_;
   mutable mfem::CGSolver cg_;

   mutable mfem::Vector rhs_work_;
   mfem::Vector z_;
   mfem::Vector m_;
   mfem::Vector x_bc_;
   mfem::real_t volume_ = 0.0;

   bool needs_assembly_ = true;
   bool use_mean_free_mode_ = true;

   mutable int last_num_iterations_ = 0;
   mutable mfem::real_t last_final_residual_ = 0.0;
   mutable bool last_converged_ = false;
};

#endif
