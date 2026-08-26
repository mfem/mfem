// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.

#ifndef MFEM_MTOP_FREQUENCY_DOMAIN_ELASTICITY_SOLVER_HPP
#define MFEM_MTOP_FREQUENCY_DOMAIN_ELASTICITY_SOLVER_HPP

#include "frequency_domain_elasticity.hpp"
#include "frequency_domain_preconditioners.hpp"

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace mfem
{

/// Lazy, GPU-ready solver for damped frequency-domain linear elasticity.
///
/// Public vectors contain contiguous [real;imaginary] true-dof blocks. The
/// iterative paths use a high-order partial-assembly operator and a monolithic
/// LOR auxiliary matrix. MUMPS is an explicitly host-resident reference path.
/// The PRESB and block-diagonal choices are isolated behind a builder boundary
/// so split preconditioners can later consume W1, W2, T, W, and H without
/// changing problem assembly or load handling.
///
/// @see PRESBPreconditioner for the PRESB factorization and literature.
/// @see mfem::ParLORDiscretization for the low-order refined auxiliary space.
///
/// @warning One instance is not safe for concurrent solves because its
/// Krylov methods, preconditioners, and boundary lifting reuse work vectors.
class FrequencyDomainLinearElasticitySolver : public Solver
{
public:
   /// Available outer linear solvers.
   enum class LinearSolverType
   {
      Automatic,
      GMRES,
      FGMRES,
      MINRES,
      MUMPS
   };

   /// Available real-block preconditioners.
   enum class PreconditionerType
   {
      PRESB,
      BlockDiagonal
   };

   /// Available inverse actions for H=W+T.
   enum class HInverseType
   {
      LORMonolithicAMG,
      LORMonolithicCGAMG,
      MUMPS
   };

   /// Construct a solver without taking ownership of @a fespace.
   explicit FrequencyDomainLinearElasticitySolver(
      ParFiniteElementSpace &fespace);

   /// Construct a solver retaining shared ownership of @a fespace.
   explicit FrequencyDomainLinearElasticitySolver(
      std::shared_ptr<ParFiniteElementSpace> fespace);

   /// Destroy all PA, LOR, Krylov, and direct-solver state.
   ~FrequencyDomainLinearElasticitySolver() override;

   /// Set the first Lame coefficient to a constant.
   void SetLambda(real_t value);

   /// Set the first Lame coefficient from a reference.
   void SetLambda(Coefficient &coefficient,
                  bool transfer_ownership = false);

   /// Set the first Lame coefficient from a shared pointer.
   void SetLambda(std::shared_ptr<Coefficient> coefficient);

   /// Set the shear modulus to a constant.
   void SetMu(real_t value);

   /// Set the shear modulus from a reference.
   void SetMu(Coefficient &coefficient, bool transfer_ownership = false);

   /// Set the shear modulus from a shared pointer.
   void SetMu(std::shared_ptr<Coefficient> coefficient);

   /// Set the mass density to a constant.
   void SetDensity(real_t value);

   /// Set the mass density from a reference.
   void SetDensity(Coefficient &coefficient,
                   bool transfer_ownership = false);

   /// Set the mass density from a shared pointer.
   void SetDensity(std::shared_ptr<Coefficient> coefficient);

   /// Set lambda, mu, and density together from coefficient references.
   void SetLameMaterial(Coefficient &lambda, Coefficient &mu,
                        Coefficient &density,
                        bool transfer_ownership = false);

   /// Set lambda, mu, and density together from shared coefficients.
   void SetLameMaterial(std::shared_ptr<Coefficient> lambda,
                        std::shared_ptr<Coefficient> mu,
                        std::shared_ptr<Coefficient> density);

   /// Set isotropic material through E, nu, and density references.
   void SetEngineeringMaterial(Coefficient &young_modulus,
                               Coefficient &poisson_ratio,
                               Coefficient &density,
                               bool transfer_ownership = false);

   /// Set isotropic material through shared E, nu, and density coefficients.
   void SetEngineeringMaterial(
      std::shared_ptr<Coefficient> young_modulus,
      std::shared_ptr<Coefficient> poisson_ratio,
      std::shared_ptr<Coefficient> density);

   /// Use distributed grid functions as lambda, mu, and density fields.
   /// The caller retains ownership and must keep all fields alive.
   void SetLameMaterialFields(ParGridFunction &lambda,
                              ParGridFunction &mu,
                              ParGridFunction &density);

   /// Use distributed grid functions as E, nu, and density fields.
   /// The caller retains ownership and must keep all fields alive.
   void SetEngineeringMaterialFields(ParGridFunction &young_modulus,
                                     ParGridFunction &poisson_ratio,
                                     ParGridFunction &density);

   /// Set the angular excitation frequency.
   ///
   /// Iterative methods require this frequency to remain below the first
   /// eigenfrequency; this is a caller contract and is not checked internally.
   void SetFrequency(real_t omega);

   /// Select Rayleigh damping C=alpha M+beta K.
   void SetRayleighDamping(real_t alpha, real_t beta);

   /// Select independent mass, lambda, and mu damping coefficients.
   void SetDampingCoefficients(Coefficient &mass_damping,
                               Coefficient &damping_lambda,
                               Coefficient &damping_mu,
                               bool transfer_ownership = false);

   /// Select shared independent mass, lambda, and mu damping coefficients.
   void SetDampingCoefficients(
      std::shared_ptr<Coefficient> mass_damping,
      std::shared_ptr<Coefficient> damping_lambda,
      std::shared_ptr<Coefficient> damping_mu);

   /// Use distributed grid functions as c_M, lambda_C, and mu_C damping.
   /// The caller retains ownership and must keep all fields alive.
   void SetDampingCoefficientFields(ParGridFunction &mass_damping,
                                    ParGridFunction &damping_lambda,
                                    ParGridFunction &damping_mu);

   /// Mark material and damping coefficient values as changed in place.
   void MaterialChanged();

   /// Mark a one-based boundary attribute as zero complex displacement.
   void AddBoundaryID(int id);

   /// Prescribe a constant complex displacement component.
   void AddDisplacementBC(int id, int component,
                          real_t real_value, real_t imaginary_value = 0.0);

   /// Prescribe a complex displacement component from coefficient references.
   void AddDisplacementBC(int id, int component,
                          Coefficient &real_coefficient,
                          Coefficient &imaginary_coefficient,
                          bool transfer_ownership = false);

   /// Prescribe a complex displacement component from shared coefficients.
   void AddDisplacementBC(
      int id, int component,
      std::shared_ptr<Coefficient> real_coefficient,
      std::shared_ptr<Coefficient> imaginary_coefficient);

   /// Prescribe a real vector displacement from a coefficient reference.
   void AddDisplacementBC(int id, VectorCoefficient &real_coefficient,
                          bool transfer_ownership = false);

   /// Prescribe a complex vector displacement from coefficient references.
   void AddDisplacementBC(int id, VectorCoefficient &real_coefficient,
                          VectorCoefficient &imaginary_coefficient,
                          bool transfer_ownership = false);

   /// Prescribe a complex vector displacement from shared coefficients.
   ///
   /// Either part may be null to prescribe purely real or imaginary data;
   /// at least one part must be supplied.
   void AddDisplacementBC(
      int id, std::shared_ptr<VectorCoefficient> real_coefficient,
      std::shared_ptr<VectorCoefficient> imaginary_coefficient = nullptr);

   /// Add a real-valued volume load on one domain attribute.
   void AddVolumeLoad(int id, VectorCoefficient &real_coefficient,
                      bool transfer_ownership = false);

   /// Add a complex volume load on one domain attribute.
   void AddVolumeLoad(int id, VectorCoefficient &real_coefficient,
                      VectorCoefficient &imaginary_coefficient,
                      bool transfer_ownership = false);

   /// Add a complex volume load through shared coefficients.
   ///
   /// Either part may be null to define a purely real or imaginary load;
   /// at least one part must be supplied.
   void AddVolumeLoad(
      int id, std::shared_ptr<VectorCoefficient> real_coefficient,
      std::shared_ptr<VectorCoefficient> imaginary_coefficient = nullptr);

   /// Add a real-valued traction on one boundary attribute.
   void AddBoundaryLoad(int id, VectorCoefficient &real_coefficient,
                        bool transfer_ownership = false);

   /// Add a complex traction on one boundary attribute.
   void AddBoundaryLoad(int id, VectorCoefficient &real_coefficient,
                        VectorCoefficient &imaginary_coefficient,
                        bool transfer_ownership = false);

   /// Add a complex traction through shared coefficients.
   ///
   /// Either part may be null to define a purely real or imaginary traction;
   /// at least one part must be supplied.
   void AddBoundaryLoad(
      int id, std::shared_ptr<VectorCoefficient> real_coefficient,
      std::shared_ptr<VectorCoefficient> imaginary_coefficient = nullptr);

   /// Remove all configured volume and boundary loads.
   void ClearLoads();

   /// Remove all homogeneous and prescribed displacement conditions.
   void ClearBoundaryConditions();

   /// Select the outer linear solver and mark setup state stale.
   void SetLinearSolverType(LinearSolverType type);

   /// Select the real-block preconditioner and mark setup state stale.
   void SetPreconditionerType(PreconditionerType type);

   /// Select the inverse action for H=W+T and mark setup state stale.
   void SetHInverseType(HInverseType type);

   /// Select the vector ordering of the monolithic LOR auxiliary space.
   void SetLOROrdering(Ordering::Type ordering);

   /// Set the outer iterative relative tolerance.
   void SetRelTol(real_t rel_tol);

   /// Set the outer iterative absolute tolerance.
   void SetAbsTol(real_t abs_tol);

   /// Set the outer iterative maximum iteration count.
   void SetMaxIter(int max_iter);

   /// Set the GMRES or FGMRES restart dimension.
   void SetKDim(int kdim);

   /// Set the outer solver print level.
   void SetPrintLevel(int print_level);

   /// Set the nested CG relative tolerance.
   ///
   /// This setting applies to LORMonolithicCGAMG. A fixed AMG inverse always
   /// applies one cycle, while MUMPS performs a direct solve.
   void SetPreconditionerRelTol(real_t rel_tol);

   /// Set the nested CG absolute tolerance.
   void SetPreconditionerAbsTol(real_t abs_tol);

   /// Set the nested CG maximum iteration count.
   void SetPreconditionerMaxIter(int max_iter);

   /// Set the nested CG, AMG, or H-MUMPS print level.
   void SetPreconditionerPrintLevel(int print_level);

   /// Return whether lazy assembly is required.
   bool NeedsAssembly() const { return needs_assembly_; }

   /// Force or clear the lazy assembly flag; forcing also releases stale setup.
   void SetNeedsAssembly(bool needs_assembly = true) const;

   /// Assemble the PA operator, auxiliary inverse, preconditioner, and solver.
   void Assemble() const;

   /// Solve a supplied standard complex true-dof right-hand side.
   void Mult(const Vector &rhs, Vector &solution) const override;

   /// Solve without performing the lazy assembly check.
   void MultAssembled(const Vector &rhs, Vector &solution) const;

   /// Solve the transposed standard complex system with homogeneous BC data.
   void MultTranspose(const Vector &rhs, Vector &solution) const override;

   /// Solve the transpose without performing the lazy assembly check.
   void MultTransposeAssembled(const Vector &rhs,
                               Vector &solution) const;

   /// Assemble configured complex loads and solve into @a solution.
   void Solve(ParComplexGridFunction &solution) const;

   /// Validate dimensions of an externally supplied standard complex operator.
   void SetOperator(const Operator &op) override;

   /// Return the current outer relative tolerance.
   real_t GetRelTol() const { return rel_tol_; }

   /// Return the current outer absolute tolerance.
   real_t GetAbsTol() const { return abs_tol_; }

   /// Return the current outer maximum iteration count.
   int GetMaxIter() const { return max_iter_; }

   /// Return the GMRES or FGMRES restart dimension.
   int GetKDim() const { return kdim_; }

   /// Return the outer solver print level.
   int GetPrintLevel() const { return print_level_; }

   /// Return the current nested H-solve relative tolerance.
   real_t GetPreconditionerRelTol() const { return preconditioner_rel_tol_; }

   /// Return the current nested H-solve absolute tolerance.
   real_t GetPreconditionerAbsTol() const { return preconditioner_abs_tol_; }

   /// Return the current nested H-solve maximum iteration count.
   int GetPreconditionerMaxIter() const { return preconditioner_max_iter_; }

   /// Return the nested H-solve and AMG print level.
   int GetPreconditionerPrintLevel() const
   { return preconditioner_print_level_; }

   /// Return the requested outer solver type.
   LinearSolverType GetLinearSolverType() const { return linear_solver_type_; }

   /// Return the selected block preconditioner type.
   PreconditionerType GetPreconditionerType() const
   { return preconditioner_type_; }

   /// Return the selected H inverse type.
   HInverseType GetHInverseType() const { return h_inverse_type_; }

   /// Return the vector ordering used by the monolithic LOR space.
   Ordering::Type GetLOROrdering() const { return lor_ordering_; }

   /// Return the effective outer solver after automatic selection.
   LinearSolverType GetActiveLinearSolverType() const;

   /// Return the outer iteration count from the most recent solve.
   int GetNumIterations() const;

   /// Return seconds spent assembling the PA system and constraints.
   double GetAssemblyTime() const { return assembly_time_; }

   /// Return seconds spent constructing H^{-1} and the block preconditioner.
   double GetPreconditionerAssemblyTime() const
   { return preconditioner_assembly_time_; }

   /// Return seconds spent constructing the outer iterative or direct solver.
   double GetSolverSetupTime() const { return solver_setup_time_; }

   /// Return the finite element space supplied by the caller.
   ParFiniteElementSpace &GetFESpace() { return fespace_; }

   /// Return the finite element space supplied by the caller.
   const ParFiniteElementSpace &GetFESpace() const { return fespace_; }

   /// Return the frequency-domain component operator.
   const FrequencyDomainElasticityOperator &
   GetFrequencyDomainOperator() const;

   /// Return the current component-aware essential true-dof list.
   const Array<int> &GetEssentialTrueDofs() const;

   /// Return the active PA real-block operator, or null for MUMPS.
   const Operator *GetOperator() const;

   /// Return the currently active real-block preconditioner, if any.
   const Solver *GetPreconditioner() const;

private:
   struct ComplexScalarData
   {
      std::shared_ptr<Coefficient> real;
      std::shared_ptr<Coefficient> imaginary;
   };

   struct ComplexVectorData
   {
      std::shared_ptr<VectorCoefficient> real;
      std::shared_ptr<VectorCoefficient> imaginary;
   };

   struct PreconditionerTraits
   {
      ComplexOperator::Convention convention = ComplexOperator::HERMITIAN;
      bool variable = false;
      bool symmetric_positive_definite = false;
   };

   /// Validate and dereference a shared finite element space.
   static ParFiniteElementSpace &CheckedFESpace(
      const std::shared_ptr<ParFiniteElementSpace> &fespace);

   /// Wrap a scalar coefficient with optional ownership transfer.
   static std::shared_ptr<Coefficient> ShareCoefficient(
      Coefficient &coefficient, bool transfer_ownership);

   /// Wrap a vector coefficient with optional ownership transfer.
   static std::shared_ptr<VectorCoefficient> ShareVectorCoefficient(
      VectorCoefficient &coefficient, bool transfer_ownership);

   /// Verify the finite element space required by elasticity PA.
   void ValidateSpace() const;

   /// Verify that an attribute ID is present in the requested mesh attribute
   /// set.
   void ValidateAttribute(int id, bool boundary) const;

   /// Verify an attribute ID and vector coefficient dimensions.
   void ValidateVectorCoefficient(int id, VectorCoefficient &coefficient,
                                  bool boundary) const;

   /// Push the stored Lame material into the component operator.
   void UpdateMaterial();

   /// Build essential true dofs from all displacement conditions.
   void BuildEssentialTrueDofs() const;

   /// Build essential true dofs on an auxiliary LOR vector space.
   void BuildAuxiliaryEssentialTrueDofs(
      ParFiniteElementSpace &space, Array<int> &ess_tdofs) const;

   /// Project prescribed real and imaginary displacement values.
   void BuildBoundaryTrueVector(Vector &values) const;

   /// Assemble configured real and imaginary volume and boundary loads.
   void BuildLoadTrueVector(Vector &values) const;

   /// Construct the monolithic LOR matrix approximating H=W+T.
   void BuildLORHMatrix() const;

   /// Construct the selected inverse action for H=W+T.
   void BuildHInverse() const;

   /// Construct PRESB or block diagonal and return its algebraic traits.
   PreconditionerTraits BuildBlockPreconditioner() const;

   /// Resolve Automatic and validate an explicit iterative solver choice.
   LinearSolverType ResolveLinearSolver(
      const PreconditionerTraits &traits) const;

   /// Construct and configure one outer MFEM iterative solver.
   std::unique_ptr<IterativeSolver> BuildIterativeSolver(
      LinearSolverType type, const Operator &system,
      Solver &preconditioner) const;

   /// Construct the full block-symmetric MUMPS factorization.
   void BuildDirectSolver() const;

   /// Apply the forward solve with optional reuse of @a solution.
   void SolveForward(const Vector &rhs, Vector &solution,
                     bool use_initial_guess) const;

   /// Negate the imaginary block on the active device.
   static void NegateImaginaryBlock(Vector &vector);

   /// Set constrained solution entries from standard complex boundary data.
   void InsertBoundaryValues(const Vector &boundary_values,
                             Vector &solution) const;

   std::shared_ptr<ParFiniteElementSpace> fespace_owner_;
   ParFiniteElementSpace &fespace_;
   mutable FrequencyDomainElasticityOperator operator_;
   std::shared_ptr<Coefficient> lambda_;
   std::shared_ptr<Coefficient> mu_;
   std::shared_ptr<Coefficient> density_;

   std::set<int> boundary_ids_;
   std::map<std::pair<int, int>, ComplexScalarData> displacement_bcs_;
   std::map<int, ComplexVectorData> vector_displacement_bcs_;
   std::map<int, ComplexVectorData> volume_loads_;
   std::map<int, ComplexVectorData> boundary_loads_;

   LinearSolverType linear_solver_type_ = LinearSolverType::Automatic;
   PreconditionerType preconditioner_type_ = PreconditionerType::PRESB;
   HInverseType h_inverse_type_ = HInverseType::LORMonolithicAMG;
   Ordering::Type lor_ordering_ = Ordering::byNODES;
   real_t rel_tol_ = 1.0e-12;
   real_t abs_tol_ = 0.0;
   int max_iter_ = 500;
   int kdim_ = 50;
   int print_level_ = -1;
   real_t preconditioner_rel_tol_ = 1.0e-2;
   real_t preconditioner_abs_tol_ = 0.0;
   int preconditioner_max_iter_ = 50;
   int preconditioner_print_level_ = -1;

   mutable bool needs_assembly_ = true;
   mutable LinearSolverType active_solver_type_ = LinearSolverType::Automatic;
   mutable ComplexOperator::Convention active_convention_ =
      ComplexOperator::HERMITIAN;
   mutable Array<int> ess_tdofs_;
   mutable std::unique_ptr<ComplexOperator> system_operator_;
   mutable std::unique_ptr<TransposeOperator> transpose_operator_;
   mutable std::unique_ptr<Solver> h_auxiliary_preconditioner_;
   mutable std::unique_ptr<Solver> h_inverse_;
   mutable std::unique_ptr<Solver> preconditioner_;
   mutable std::unique_ptr<Solver> transpose_preconditioner_;
   mutable std::unique_ptr<IterativeSolver> iterative_solver_;
   mutable std::unique_ptr<IterativeSolver> transpose_solver_;
   mutable std::unique_ptr<ParLORDiscretization> lor_discretization_;
   mutable std::unique_ptr<ParFiniteElementSpace> lor_fespace_;
   mutable std::unique_ptr<ParBilinearForm> lor_h_form_;
   mutable std::unique_ptr<HypreParMatrix> lor_h_matrix_;
   mutable std::vector<std::unique_ptr<Coefficient> > lor_h_coefficients_;
   mutable std::unique_ptr<HypreParMatrix> h_matrix_;

   mutable std::unique_ptr<ComplexHypreParMatrix> assembled_complex_;
   mutable std::unique_ptr<HypreParMatrix> direct_matrix_;
   mutable std::unique_ptr<Solver> direct_solver_;

   mutable Vector boundary_true_values_;
   mutable bool boundary_values_stale_ = true;
   mutable Vector solve_rhs_;
   mutable Vector previous_solution_;
   mutable bool has_previous_solution_ = false;
   mutable int num_iterations_ = 0;
   mutable double assembly_time_ = 0.0;
   mutable double preconditioner_assembly_time_ = 0.0;
   mutable double solver_setup_time_ = 0.0;
};

} // namespace mfem

#endif
