// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.

#ifndef MFEM_MTOP_LINEAR_ELASTICITY_SOLVER_HPP
#define MFEM_MTOP_LINEAR_ELASTICITY_SOLVER_HPP

#include "mfem.hpp"

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace mfem
{

/// Lazy partial-assembly solver for isotropic linear elasticity.
///
/// The supplied space must be an H1 vector space with vector dimension equal
/// to the mesh dimension. Mult() accepts an unconstrained true-dof load vector
/// and applies the configured displacement data. MultTranspose() is the
/// corresponding adjoint solve and uses homogeneous data on constrained dofs.
/// Solve() assembles all configured volume and boundary loads internally.
class LinearElasticitySolver : public Solver
{
public:
   /// Available preconditioners for the high-order PA elasticity operator.
   enum class PreconditionerType
   {
      Jacobi,
      AMG,
      LORDiagonalAMG,
      LORMonolithicAMG
   };

   /// Construct a solver on @a fespace without taking ownership.
   explicit LinearElasticitySolver(ParFiniteElementSpace &fespace);

   /// Construct a solver and keep shared ownership of @a fespace.
   explicit LinearElasticitySolver(
      std::shared_ptr<ParFiniteElementSpace> fespace);

   /// Set the first Lame parameter to a constant and mark the solver dirty.
   void SetLambda(real_t value);

   /// Set lambda from a reference, optionally taking ownership of it.
   void SetLambda(Coefficient &coefficient,
                  bool transfer_ownership = false);

   /// Set lambda from a shared coefficient and mark the solver dirty.
   void SetLambda(std::shared_ptr<Coefficient> coefficient);

   /// Set the shear modulus to a constant and mark the solver dirty.
   void SetMu(real_t value);

   /// Set mu from a reference, optionally taking ownership of it.
   void SetMu(Coefficient &coefficient, bool transfer_ownership = false);

   /// Set mu from a shared coefficient and mark the solver dirty.
   void SetMu(std::shared_ptr<Coefficient> coefficient);

   /// Mark a one-based boundary attribute as homogeneous Dirichlet.
   void AddBoundaryID(int id);

   /// Prescribe a constant displacement component on a boundary attribute.
   void AddDisplacementBC(int id, int component, real_t value);

   /// Prescribe a displacement component from a referenced scalar coefficient.
   void AddDisplacementBC(int id, int component, Coefficient &coefficient,
                          bool transfer_ownership = false);

   /// Prescribe a displacement component from a shared scalar coefficient.
   void AddDisplacementBC(int id, int component,
                          std::shared_ptr<Coefficient> coefficient);

   /// Prescribe all displacement components from a vector coefficient.
   void AddDisplacementBC(int id, VectorCoefficient &coefficient,
                          bool transfer_ownership = false);

   /// Prescribe all displacement components from a shared vector coefficient.
   void AddDisplacementBC(int id,
                          std::shared_ptr<VectorCoefficient> coefficient);

   /// Add a vector-valued volume load on a domain attribute.
   void AddVolumeLoad(int id, VectorCoefficient &coefficient,
                      bool transfer_ownership = false);

   /// Add a shared vector-valued volume load on a domain attribute.
   void AddVolumeLoad(int id,
                      std::shared_ptr<VectorCoefficient> coefficient);

   /// Add a vector-valued traction on a boundary attribute.
   void AddBoundaryLoad(int id, VectorCoefficient &coefficient,
                        bool transfer_ownership = false);

   /// Add a shared vector-valued traction on a boundary attribute.
   void AddBoundaryLoad(int id,
                        std::shared_ptr<VectorCoefficient> coefficient);

   /// Remove all volume and boundary loads.
   void ClearLoads();

   /// Remove all homogeneous and prescribed displacement boundary conditions.
   void ClearBoundaryConditions();

   /// Return true when Assemble() must rebuild the PA system.
   bool NeedsAssembly() const { return needs_assembly_; }

   /// Force or clear the lazy assembly flag.
   void SetNeedsAssembly(bool needs_assembly = true) const;

   /// Assemble the PA operator, constrained system, preconditioner, and CG.
   void Assemble() const;

   /// Set the CG relative tolerance.
   void SetRelTol(real_t rel_tol);

   /// Set the CG absolute tolerance.
   void SetAbsTol(real_t abs_tol);

   /// Set the maximum number of CG iterations.
   void SetMaxIter(int max_iter);

   /// Set the CG print level.
   void SetPrintLevel(int print_level);

   /// Select the preconditioner and mark the solver dirty.
   ///
   /// Diagonal LOR/AMG requires Ordering::byNODES. Plain AMG and monolithic
   /// LOR/AMG support either auxiliary ordering and enable rigid-body
   /// near-nullspace vectors only for Ordering::byVDIM.
   void SetPreconditionerType(PreconditionerType type);

   /// Select the vector ordering for plain AMG and monolithic LOR/AMG.
   /// Ordering::byVDIM enables BoomerAMG rigid-body near-nullspace vectors;
   /// Ordering::byNODES uses generic systems AMG without rigid-body modes.
   void SetMonolithicLOROrdering(Ordering::Type ordering);

   /// Return the CG relative tolerance.
   real_t GetRelTol() const { return rel_tol_; }

   /// Return the CG absolute tolerance.
   real_t GetAbsTol() const { return abs_tol_; }

   /// Return the maximum number of CG iterations.
   int GetMaxIter() const { return max_iter_; }

   /// Return the CG print level.
   int GetPrintLevel() const { return print_level_; }

   /// Return the selected preconditioner type.
   PreconditionerType GetPreconditionerType() const
   { return preconditioner_type_; }

   /// Return the selected plain or monolithic auxiliary AMG ordering.
   Ordering::Type GetMonolithicLOROrdering() const
   { return monolithic_lor_ordering_; }

   /// Return the iteration count from the most recent CG solve.
   int GetNumIterations() const;

   /// Return seconds spent in PA and constrained-system assembly.
   double GetAssemblyTime() const { return assembly_time_; }

   /// Return seconds spent building the selected preconditioner.
   ///
   /// For Jacobi this is diagonal assembly and inversion; for an AMG option it
   /// includes auxiliary matrix assembly and AMG construction.
   double GetPrecAssemblyTime() const { return prec_assembly_time_; }

   /// Solve a true-dof load vector while applying prescribed displacements.
   void Mult(const Vector &rhs, Vector &solution) const override;

   /// Apply Mult() without performing the lazy assembly check.
   void MultAssembled(const Vector &rhs, Vector &solution) const;

   /// Apply the adjoint solve with zero data on all constrained true dofs.
   void MultTranspose(const Vector &rhs, Vector &solution) const override;

   /// Apply MultTranspose() without performing the lazy assembly check.
   void MultTransposeAssembled(const Vector &rhs, Vector &solution) const;

   /// Assemble configured loads and solve, reusing the previous state as a
   /// warm start when available.
   void Solve(ParGridFunction &solution) const;

   /// Check that an externally supplied operator has compatible dimensions.
   void SetOperator(const Operator &op) override;

   /// Return the finite element space supplied by the caller.
   ParFiniteElementSpace &GetFESpace() { return fespace_; }

   /// Return the finite element space supplied by the caller.
   const ParFiniteElementSpace &GetFESpace() const { return fespace_; }

   /// Return the current component-aware essential true-dof list.
   const Array<int> &GetEssentialTrueDofs() const;

   /// Return the current constrained PA system operator.
   const Operator *GetOperator() const;

   /// Return the currently selected preconditioner.
   const Solver *GetPreconditioner() const;

private:
   /// Validate and dereference a shared finite element space.
   static ParFiniteElementSpace &CheckedFESpace(
      const std::shared_ptr<ParFiniteElementSpace> &fespace);

   /// Wrap a coefficient reference in an owning or non-owning shared pointer.
   static std::shared_ptr<Coefficient> ShareCoefficient(
      Coefficient &coefficient, bool transfer_ownership);

   /// Wrap a vector coefficient in an owning or non-owning shared pointer.
   static std::shared_ptr<VectorCoefficient> ShareVectorCoefficient(
      VectorCoefficient &coefficient, bool transfer_ownership);

   /// Verify that the supplied FE space is supported by elasticity PA.
   void ValidateSpace() const;

   /// Build essential true dofs from homogeneous and component-wise data.
   void BuildEssentialTrueDofs() const;

   /// Build a boundary marker for one displacement component.
   void BuildComponentBoundaryMarker(int component, Array<int> &marker) const;

   /// Build the selected Jacobi, assembled AMG, or LOR/AMG preconditioner.
   void BuildPreconditioner() const;

   /// Build AMG on a fully assembled high-order auxiliary elasticity matrix.
   void BuildAMG() const;

   /// Build essential true dofs on an auxiliary vector space.
   void BuildAuxiliaryEssentialTrueDofs(ParFiniteElementSpace &space,
                                        Array<int> &ess_tdofs) const;

   /// Build independent scalar LOR/AMG blocks for displacement components.
   void BuildLORDiagonalAMG() const;

   /// Build AMG on the fully coupled LOR elasticity matrix.
   void BuildLORMonolithicAMG() const;

   /// Project all prescribed displacement components into @a solution.
   void ProjectBoundaryValues(ParGridFunction &solution) const;

   /// Build the prescribed displacement vector in true-dof coordinates.
   void BuildBoundaryTrueVector(Vector &values) const;

   /// Set every constrained entry of @a vector to zero.
   void ZeroConstrainedDofs(Vector &vector) const;

   /// Apply the forward solve with optional reuse of @a solution as a guess.
   void SolveForward(const Vector &rhs, Vector &solution,
                     bool use_initial_guess) const;

   /// Validate an attribute ID and vector coefficient dimension.
   void ValidateVectorCoefficient(int id,
                                  VectorCoefficient &coefficient) const;

   std::shared_ptr<ParFiniteElementSpace> fespace_owner_;
   ParFiniteElementSpace &fespace_;
   std::shared_ptr<Coefficient> lambda_;
   std::shared_ptr<Coefficient> mu_;
   std::set<int> boundary_ids_;
   std::map<std::pair<int, int>, std::shared_ptr<Coefficient> >
   displacement_bcs_;
   std::map<int, std::shared_ptr<VectorCoefficient> > vector_displacement_bcs_;
   std::map<int, std::shared_ptr<VectorCoefficient> > volume_loads_;
   std::map<int, std::shared_ptr<VectorCoefficient> > boundary_loads_;

   mutable bool needs_assembly_ = true;
   real_t rel_tol_ = 1.0e-12;
   real_t abs_tol_ = 0.0;
   int max_iter_ = 500;
   int print_level_ = -1;
   PreconditionerType preconditioner_type_ = PreconditionerType::Jacobi;
   Ordering::Type monolithic_lor_ordering_ = Ordering::byNODES;

   mutable Array<int> ess_tdofs_;
   mutable std::unique_ptr<ParBilinearForm> form_;
   mutable OperatorHandle system_operator_;

   mutable std::unique_ptr<ParLORDiscretization> lor_disc_;
   mutable std::unique_ptr<ParFiniteElementSpace> amg_fespace_;
   mutable std::unique_ptr<ParBilinearForm> amg_form_;
   mutable std::unique_ptr<HypreParMatrix> amg_matrix_;
   mutable std::unique_ptr<ParFiniteElementSpace> lor_scalar_fespace_;
   mutable std::unique_ptr<ElasticityIntegrator> lor_integrator_;
   mutable std::vector<std::unique_ptr<ParBilinearForm> > lor_forms_;
   mutable std::vector<std::unique_ptr<HypreParMatrix> > lor_blocks_;
   mutable std::vector<std::unique_ptr<HypreBoomerAMG> > lor_amg_blocks_;
   mutable std::unique_ptr<ParFiniteElementSpace> lor_monolithic_fespace_;
   mutable std::unique_ptr<ParBilinearForm> lor_monolithic_form_;
   mutable std::unique_ptr<HypreParMatrix> lor_monolithic_matrix_;
   mutable Array<int> lor_block_offsets_;
   mutable std::unique_ptr<Solver> preconditioner_;
   mutable std::unique_ptr<CGSolver> cg_;
   mutable Vector boundary_true_values_;
   mutable Vector solve_rhs_;
   mutable Vector previous_solution_;
   mutable bool has_previous_solution_ = false;
   mutable double assembly_time_ = 0.0;
   mutable double prec_assembly_time_ = 0.0;
};

} // namespace mfem

#endif
