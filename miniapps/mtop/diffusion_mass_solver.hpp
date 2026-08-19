#ifndef MFEM_DIFFUSION_MASS_SOLVER_HPP
#define MFEM_DIFFUSION_MASS_SOLVER_HPP

#include "mfem.hpp"

#include <map>
#include <memory>
#include <set>

namespace mfem
{

/// Riesz map on a parallel finite element space using the L2 inner product.
///
/// For a primal true vector u, this operator returns the dual true vector
/// b = M u, where M is the finite element mass matrix on the supplied space.
/// The mass bilinear form is assembled with partial assembly and no essential
/// boundary elimination.  The class owns no finite element space unless it is
/// constructed from a shared pointer.
class RieszMapOperator : public Operator
{
public:
   /// Construct the L2 Riesz map on @a fespace without taking ownership.
   explicit RieszMapOperator(ParFiniteElementSpace &fespace);

   /// Construct the L2 Riesz map and keep shared ownership of @a fespace.
   explicit RieszMapOperator(std::shared_ptr<ParFiniteElementSpace> fespace);

   /// Return true when Assemble() must rebuild the mass operator.
   bool NeedsAssembly() const { return needs_assembly_; }

   /// Force or clear the lazy assembly flag.  This method is collective when
   /// followed by Assemble(), Mult(), or MultTranspose().
   void SetNeedsAssembly(bool needs_assembly = true) const;

   /// Assemble the partial-assembly mass operator if it is dirty.
   void Assemble() const;

   /// Map primal true dofs to dual true dofs: dual = M primal.
   void Mult(const Vector &primal, Vector &dual) const override;

   /// Apply Mult() without checking assembly state.  The caller must first call
   /// Assemble() collectively.
   void MultAssembled(const Vector &primal, Vector &dual) const;

   /// Apply the transpose map.  The L2 Riesz map is symmetric, so this is the
   /// same operation as Mult().
   void MultTranspose(const Vector &dual, Vector &primal) const override;

   /// Apply MultTranspose() without checking assembly state.
   void MultTransposeAssembled(const Vector &dual, Vector &primal) const;

   /// Return the finite element space on which the map is defined.
   ParFiniteElementSpace &GetFESpace() { return fespace_; }

   /// Return the finite element space on which the map is defined.
   const ParFiniteElementSpace &GetFESpace() const { return fespace_; }

   /// Return the current true-dof mass operator, assembling first if needed.
   const Operator *GetOperator() const;

private:
   /// Optional shared owner for fespace_.
   std::shared_ptr<ParFiniteElementSpace> fespace_owner_;

   /// Finite element space defining the primal and dual coordinates.
   ParFiniteElementSpace &fespace_;

   /// True when the PA mass form and system operator must be rebuilt.
   mutable bool needs_assembly_ = true;

   /// Partial-assembly mass bilinear form.
   mutable std::unique_ptr<ParBilinearForm> form_;

   /// True-dof mass operator produced by form_.
   mutable OperatorHandle mass_operator_;
};

/// Inverse L2 Riesz map on a parallel finite element space.
///
/// This operator maps a dual true vector b to the primal true vector u solving
/// M u = b, where M is the partial-assembly mass operator from RieszMapOperator.
/// The inverse is applied by CG preconditioned with an OperatorJacobiSmoother
/// built from the operator diagonal.  The finite element space is not owned
/// unless the shared-pointer constructor is used.
class InverseRieszMapOperator : public Operator
{
public:
   /// Construct the inverse L2 Riesz map on @a fespace without taking ownership.
   explicit InverseRieszMapOperator(ParFiniteElementSpace &fespace);

   /// Construct the inverse L2 Riesz map and keep shared ownership of @a fespace.
   explicit InverseRieszMapOperator(
      std::shared_ptr<ParFiniteElementSpace> fespace);

   /// Set the relative tolerance used by the internal CG solver.
   void SetRelTol(real_t rel_tol);

   /// Set the absolute tolerance used by the internal CG solver.
   void SetAbsTol(real_t abs_tol);

   /// Set the maximum number of CG iterations.
   void SetMaxIter(int max_iter);

   /// Set the print level used by the internal CG solver.
   void SetPrintLevel(int print_level);

   /// Return true when Assemble() must rebuild the map/preconditioner.
   bool NeedsAssembly() const { return needs_assembly_ || riesz_.NeedsAssembly(); }

   /// Force or clear the lazy assembly flag.
   void SetNeedsAssembly(bool needs_assembly = true) const;

   /// Assemble the Riesz map, diagonal preconditioner, and CG solver if dirty.
   void Assemble() const;

   /// Map dual true dofs to primal true dofs by solving M primal = dual.
   void Mult(const Vector &dual, Vector &primal) const override;

   /// Apply Mult() without checking assembly state.  The caller must first call
   /// Assemble() collectively.
   void MultAssembled(const Vector &dual, Vector &primal) const;

   /// Apply the transpose inverse map.  The mass matrix is symmetric, so this
   /// is the same operation as Mult().
   void MultTranspose(const Vector &dual, Vector &primal) const override;

   /// Apply MultTranspose() without checking assembly state.
   void MultTransposeAssembled(const Vector &dual, Vector &primal) const;

   /// Return the finite element space on which the inverse map is defined.
   ParFiniteElementSpace &GetFESpace() { return riesz_.GetFESpace(); }

   /// Return the finite element space on which the inverse map is defined.
   const ParFiniteElementSpace &GetFESpace() const { return riesz_.GetFESpace(); }

   /// Return the underlying forward Riesz map.
   RieszMapOperator &GetRieszMap() { return riesz_; }

   /// Return the underlying forward Riesz map.
   const RieszMapOperator &GetRieszMap() const { return riesz_; }

private:
   /// Forward Riesz map whose operator is inverted.
   RieszMapOperator riesz_;

   /// True when the preconditioner and CG solver must be rebuilt.
   mutable bool needs_assembly_ = true;

   /// Relative tolerance for CG.
   real_t rel_tol_ = 1.0e-12;

   /// Absolute tolerance for CG.
   real_t abs_tol_ = 0.0;

   /// Maximum CG iterations.
   int max_iter_ = 200;

   /// Print level for CG.
   int print_level_ = -1;

   /// Diagonal Jacobi preconditioner for the PA mass operator.
   mutable std::unique_ptr<OperatorJacobiSmoother> jacobi_;

   /// Cached CG solver used to apply the inverse map.
   mutable std::unique_ptr<CGSolver> cg_solver_;
};

/// True-dof mass map between two parallel scalar finite element spaces.
///
/// Given an input/trial space V_in and output/test space V_out, this operator
/// represents the rectangular mass matrix
///     y_i = int_Omega weight u phi_i^out dx,
/// i.e. `output = M input` on true vectors.  `MultTranspose()` applies M^T and
/// is intended for adjoint/gradient propagation from the output space back to
/// the input space.  The form is assembled lazily with no essential boundary
/// elimination.  MixedScalarMassIntegrator does not provide partial assembly in
/// this MFEM tree, so the map stores an assembled HypreParMatrix.
class TrueMassMapOperator : public Operator
{
public:
   /// Construct a lazy mass map from @a input_space to @a output_space.
   TrueMassMapOperator(ParFiniteElementSpace &input_space,
                       ParFiniteElementSpace &output_space);

   /// Construct a lazy mass map and keep shared ownership of both spaces.
   TrueMassMapOperator(std::shared_ptr<ParFiniteElementSpace> input_space,
                       std::shared_ptr<ParFiniteElementSpace> output_space);

   /// Set the scalar weight to a constant value and mark the map dirty.
   void SetWeightCoefficient(real_t value);

   /// Set the scalar weight from a coefficient reference.  If
   /// @a transfer_ownership is true, the operator deletes the coefficient when
   /// it is no longer referenced.
   void SetWeightCoefficient(Coefficient &coefficient,
                             bool transfer_ownership = false);

   /// Set the scalar weight from a shared coefficient and mark the map dirty.
   void SetWeightCoefficient(std::shared_ptr<Coefficient> coefficient);

   /// Return true when Assemble() must rebuild the mixed mass operator.
   bool NeedsAssembly() const { return needs_assembly_; }

   /// Force or clear the lazy assembly flag for the mixed mass operator.
   void SetNeedsAssembly(bool needs_assembly = true) const;

   /// Assemble the mixed mass operator if it is dirty.
   void Assemble() const;

   /// Apply `output_true = M input_true`.
   void Mult(const Vector &input, Vector &output) const override;

   /// Apply `output_true = M input_true` without checking assembly state.
   /// The caller must first call Assemble() collectively.
   void MultAssembled(const Vector &input, Vector &output) const;

   /// Apply `input_true = M^T output_true`.
   void MultTranspose(const Vector &output, Vector &input) const override;

   /// Apply `input_true = M^T output_true` without checking assembly state.
   /// The caller must first call Assemble() collectively.
   void MultTransposeAssembled(const Vector &output, Vector &input) const;

   /// Return the input/trial finite element space.
   ParFiniteElementSpace &GetInputFESpace() { return input_space_; }

   /// Return the input/trial finite element space.
   const ParFiniteElementSpace &GetInputFESpace() const { return input_space_; }

   /// Return the output/test finite element space.
   ParFiniteElementSpace &GetOutputFESpace() { return output_space_; }

   /// Return the output/test finite element space.
   const ParFiniteElementSpace &GetOutputFESpace() const { return output_space_; }

   /// Return the current true-dof mass operator, assembling first if needed.
   const Operator *GetOperator() const;

private:
   /// Wrap a coefficient reference in a shared pointer with optional ownership.
   std::shared_ptr<Coefficient> ShareCoefficient(Coefficient &coefficient,
                                                 bool transfer_ownership);

   /// Optional shared owner for input_space_.
   std::shared_ptr<ParFiniteElementSpace> input_space_owner_;

   /// Optional shared owner for output_space_.
   std::shared_ptr<ParFiniteElementSpace> output_space_owner_;

   /// Input/trial finite element space; not owned by this operator.
   ParFiniteElementSpace &input_space_;

   /// Output/test finite element space; not owned by this operator.
   ParFiniteElementSpace &output_space_;

   /// Scalar mass weight used in (weight u, v); defaults to one.
   std::shared_ptr<Coefficient> weight_coefficient_;

   /// True when the mixed mass form and operator must be rebuilt.
   mutable bool needs_assembly_ = true;

   /// Mixed bilinear form used to assemble the rectangular mass map.
   mutable std::unique_ptr<ParMixedBilinearForm> form_;

   /// Assembled true-dof operator produced by the mixed form.
   mutable OperatorHandle mass_operator_;
};

/// Solver for a scalar diffusion+mass equation on a user-provided
/// ParFiniteElementSpace:
///     -div(diffusion_coefficient grad u) + mass_coefficient u = rhs.
///
/// The system operator is assembled with partial assembly.  Dirichlet boundary
/// conditions are selected by boundary attribute IDs; if no IDs are added, the
/// solver uses natural Neumann boundary conditions.  Coefficient and boundary
/// changes are lazy: mutators only mark the solver dirty, and the next Solve(),
/// Mult(), or explicit Assemble() rebuilds the operator/preconditioner.
///
/// When ParGridFunction operator coefficients are used with an order > 1 space,
/// the LOR AMG preconditioner transfers their true-dof values to persistent LOR
/// order-1 grid functions and assembles the AMG matrix from those LOR
/// coefficients.  This is intended for compatible tensor-product H1/GLL spaces
/// where the high-order and LOR true-dof layouts match.
class DiffusionMassSolver : public Solver
{
public:
   /// Identifies whether an AttributeCoefficientMap stores domain RHS data or
   /// Dirichlet boundary data.
   enum class MapKind { DomainRHS, Boundary };

   /// Attribute-keyed scalar coefficient container used for RHS and boundary
   /// data.  Boundary maps notify the owning solver that the corresponding
   /// boundary attribute is essential.
   class AttributeCoefficientMap
   {
   public:
      AttributeCoefficientMap() = default;

      /// Attach this map to a solver.  Called by DiffusionMassSolver.
      void SetOwner(DiffusionMassSolver *owner, MapKind kind);

      /// Add a constant coefficient on attribute @a attr.
      void Add(int attr, real_t value);

      /// Add a referenced coefficient on attribute @a attr.  If
      /// @a transfer_ownership is true, the map deletes the coefficient when it
      /// is no longer referenced.
      void Add(int attr, Coefficient &coefficient,
               bool transfer_ownership = false);

      /// Add a shared coefficient on attribute @a attr.
      void Add(int attr, std::shared_ptr<Coefficient> coefficient);

      /// Add a QuadratureFunction coefficient by reference.
      void Add(int attr, QuadratureFunction &qf,
               bool transfer_ownership = false);

      /// Add a QuadratureFunction coefficient held by shared pointer.
      void Add(int attr, std::shared_ptr<QuadratureFunction> qf);

      /// Add a ParGridFunction coefficient by reference.
      void Add(int attr, ParGridFunction &gf,
               bool transfer_ownership = false);

      /// Add a ParGridFunction coefficient held by shared pointer.
      void Add(int attr, std::shared_ptr<ParGridFunction> gf);

      /// Remove all stored coefficients.
      void Clear();

      /// Return true when no attribute coefficients have been added.
      bool Empty() const { return coefficients_.empty(); }

      /// Build a PWCoefficient view.  The view is owned by this map and remains
      /// valid until the next non-const call to this map.
      Coefficient &AsCoefficient() const;

   private:
      /// Wrap a coefficient reference in a shared pointer with optional
      /// ownership transfer.
      std::shared_ptr<Coefficient> ShareCoefficient(Coefficient &coefficient,
                                                    bool transfer_ownership);

      /// Wrap a QuadratureFunction reference in a shared pointer with optional
      /// ownership transfer.
      std::shared_ptr<QuadratureFunction> ShareQuadratureFunction(
         QuadratureFunction &qf, bool transfer_ownership);

      /// Wrap a ParGridFunction reference in a shared pointer with optional
      /// ownership transfer.
      std::shared_ptr<ParGridFunction> ShareParGridFunction(
         ParGridFunction &gf, bool transfer_ownership);

      /// Notify the owning solver that this map changed on @a attr.
      void NotifyChanged(int attr);

      /// Owning solver notified by map mutations; not owned by the map.
      DiffusionMassSolver *owner_ = nullptr;

      /// Semantic role of the map: domain RHS or boundary Dirichlet data.
      MapKind kind_ = MapKind::DomainRHS;

      /// Attribute-to-coefficient table used by the PWCoefficient view.
      std::map<int, std::shared_ptr<Coefficient> > coefficients_;

      /// QuadratureFunction owners, keyed by attribute, for lifetime tracking.
      std::map<int, std::shared_ptr<QuadratureFunction> > qf_owners_;

      /// ParGridFunction owners, keyed by attribute, for lifetime tracking.
      std::map<int, std::shared_ptr<ParGridFunction> > gf_owners_;

      /// Cached piecewise coefficient view rebuilt by AsCoefficient().
      mutable std::unique_ptr<PWCoefficient> piecewise_;
   };

   /// Create a solver on @a fespace.  No essential boundary IDs are installed
   /// by default, so the default boundary condition is Neumann.
   explicit DiffusionMassSolver(ParFiniteElementSpace &fespace);

   /// Create a solver and keep shared ownership of @a fespace.
   explicit DiffusionMassSolver(std::shared_ptr<ParFiniteElementSpace> fespace);

   /// Set diffusion to a constant value and mark the system dirty.
   void SetDiffusionCoefficient(real_t value);

   /// Set diffusion from a coefficient reference and mark the system dirty.
   void SetDiffusionCoefficient(Coefficient &coefficient,
                                bool transfer_ownership = false);

   /// Set diffusion from a shared coefficient and mark the system dirty.
   void SetDiffusionCoefficient(std::shared_ptr<Coefficient> coefficient);

   /// Set diffusion from a QuadratureFunction coefficient by reference.
   void SetDiffusionCoefficient(QuadratureFunction &qf,
                                bool transfer_ownership = false);

   /// Set diffusion from a shared QuadratureFunction coefficient.
   void SetDiffusionCoefficient(std::shared_ptr<QuadratureFunction> qf);

   /// Set diffusion from a ParGridFunction coefficient by reference.
   void SetDiffusionCoefficient(ParGridFunction &gf,
                                bool transfer_ownership = false);

   /// Set diffusion from a shared ParGridFunction coefficient.
   void SetDiffusionCoefficient(std::shared_ptr<ParGridFunction> gf);

   /// Set mass to a constant value and mark the system dirty.
   void SetMassCoefficient(real_t value);

   /// Set mass from a coefficient reference and mark the system dirty.
   void SetMassCoefficient(Coefficient &coefficient,
                           bool transfer_ownership = false);

   /// Set mass from a shared coefficient and mark the system dirty.
   void SetMassCoefficient(std::shared_ptr<Coefficient> coefficient);

   /// Set mass from a QuadratureFunction coefficient by reference.
   void SetMassCoefficient(QuadratureFunction &qf,
                           bool transfer_ownership = false);

   /// Set mass from a shared QuadratureFunction coefficient.
   void SetMassCoefficient(std::shared_ptr<QuadratureFunction> qf);

   /// Set mass from a ParGridFunction coefficient by reference.
   void SetMassCoefficient(ParGridFunction &gf,
                           bool transfer_ownership = false);

   /// Set mass from a shared ParGridFunction coefficient.
   void SetMassCoefficient(std::shared_ptr<ParGridFunction> gf);

   /// Access domain RHS coefficients keyed by element attribute.
   AttributeCoefficientMap &RHS() { return rhs_coefficients_; }

   /// Access Dirichlet boundary coefficients keyed by boundary attribute.
   AttributeCoefficientMap &Boundary() { return boundary_coefficients_; }

   /// Mark a boundary attribute ID as essential with homogeneous data unless a
   /// boundary coefficient for that ID is also provided.
   void AddBoundaryID(int id);

   /// Remove all essential boundary IDs and boundary coefficients.  After this,
   /// the next solve uses natural Neumann boundary conditions.
   void ClearBoundaryConditions();

   /// Return true when Assemble() is required before applying the operator.
   bool NeedsAssembly() const { return needs_assembly_; }

   /// Force or clear the lazy assembly flag.
   void SetNeedsAssembly(bool needs_assembly = true) const;

   /// Set the print level used by the outer CG solver and AMG preconditioners.
   void SetPrintLevel(int print_level)
   {
      print_level_ = print_level;
      needs_assembly_ = true;
   }

   /// Return the current solver print level.
   int GetPrintLevel() const { return print_level_; }

   /// Return the integration order used for domain operators and RHS forms.
   int GetIntegrationOrder() const { return integration_order_; }

   /// Return the domain integration rule used for a geometry type.
   const IntegrationRule &GetIntegrationRule(Geometry::Type geom) const;

   /// Reassemble the partial-assembly operator and AMG preconditioner if dirty.
   void Assemble() const;

   /// Solve the already-eliminated true-dof linear system A x = rhs.  Stored
   /// essential boundary IDs are enforced in the true-dof system.  If boundary
   /// coefficients were added, constrained RHS and solution entries are set to
   /// the projected boundary values; otherwise they are set to zero.  The
   /// Krylov solver and work vectors are reused across calls until the operator
   /// is marked dirty.
   void Mult(const Vector &rhs, Vector &solution) const override;

   /// Same as Mult(), but skips the lazy Assemble() check.  The caller must
   /// first call Assemble() collectively.
   void MultAssembled(const Vector &rhs, Vector &solution) const;

   /// Apply the adjoint solve.  The diffusion+mass operator is symmetric, so
   /// this uses the same system as Mult(), but all constrained Dirichlet true
   /// dofs are treated as homogeneous regardless of stored boundary data.
   void MultTranspose(const Vector &rhs, Vector &solution) const override;

   /// Same as MultTranspose(), but skips the lazy Assemble() check.  The caller
   /// must first call Assemble() collectively.
   void MultTransposeAssembled(const Vector &rhs, Vector &solution) const;

   /// Assemble the RHS from RHS(), apply Dirichlet boundary coefficients from
   /// Boundary(), eliminate essential true dofs, solve, and recover the finite
   /// element solution.  With no boundary IDs, this is a Neumann solve.  The
   /// RHS form, linear-system work vectors, and Krylov solver are cached for
   /// repeated solves with unchanged coefficients and boundary data.
   void Solve(ParGridFunction &solution) const;

   /// IterativeSolver compatibility: validates dimensions and otherwise leaves
   /// the internally owned operator unchanged.
   void SetOperator(const Operator &op) override;

   /// Return the finite element space supplied by the user.
   ParFiniteElementSpace &GetFESpace() { return fespace_; }
   const ParFiniteElementSpace &GetFESpace() const { return fespace_; }

   /// Return the current partial-assembly system operator, assembling first if
   /// needed.
   const Operator *GetOperator() const;

   /// Return the current AMG preconditioner, assembling first if needed.
   const Solver *GetPreconditioner() const;

private:
   friend class AttributeCoefficientMap;

   /// Wrap a coefficient reference in a shared pointer with optional ownership.
   std::shared_ptr<Coefficient> ShareCoefficient(Coefficient &coefficient,
                                                 bool transfer_ownership);

   /// Wrap a QuadratureFunction reference in a shared pointer with optional
   /// ownership.
   std::shared_ptr<QuadratureFunction> ShareQuadratureFunction(
      QuadratureFunction &qf, bool transfer_ownership);

   /// Wrap a ParGridFunction reference in a shared pointer with optional
   /// ownership.
   std::shared_ptr<ParGridFunction> ShareParGridFunction(
      ParGridFunction &gf, bool transfer_ownership);

   /// Convert a QuadratureFunction owner to a scalar coefficient wrapper.
   std::shared_ptr<Coefficient> MakeQuadratureCoefficient(
      std::shared_ptr<QuadratureFunction> qf) const;

   /// Convert a ParGridFunction owner to a scalar coefficient wrapper.
   std::shared_ptr<Coefficient> MakeGridFunctionCoefficient(
      std::shared_ptr<ParGridFunction> gf) const;

   /// Verify that a QuadratureFunction matches this solver's mesh and rule.
   void ValidateQuadratureFunction(const QuadratureFunction &qf) const;

   /// Verify that a ParGridFunction coefficient uses this solver's FE space.
   void ValidateParGridFunction(const ParGridFunction &gf) const;

   /// Solve the true-dof system, optionally imposing constrained dof values.
   void SolveSystem(const Vector &rhs, Vector &solution,
                    bool apply_constraints,
                    bool use_boundary_values) const;

   /// Mark operator/preconditioner data dirty after diffusion or mass changes.
   void MarkCoefficientChanged();

   /// Mark RHS form/vector data dirty after domain RHS changes.
   void MarkRHSChanged();

   /// Mark boundary data dirty and add @a id to the essential boundary set.
   void MarkBoundaryChanged(int id);

   /// Build a boundary attribute marker from boundary_ids_.
   void BuildEssentialMarker(Array<int> &marker) const;

   /// Build the essential true-dof list from boundary_ids_.
   void BuildEssentialTrueDofs() const;

   /// Build the AMG preconditioner selected for the current FE order.
   void BuildPreconditioner() const;

   /// Build BoomerAMG directly on an assembled order-1 operator.
   void BuildAMGPreconditionerOnFESpace() const;

   /// Build BoomerAMG on an assembled LOR operator for high-order spaces.
   void BuildLORAMGPreconditioner() const;

   /// Transfer an HO ParGridFunction coefficient to the LOR space if needed.
   std::shared_ptr<Coefficient> MakeLORCoefficient(
      const std::shared_ptr<Coefficient> &coefficient,
      const std::shared_ptr<ParGridFunction> &ho_gf,
      std::shared_ptr<ParGridFunction> &lor_gf) const;

   /// Project boundary coefficients and extract values on essential true dofs.
   void BuildBoundaryTrueDofs(Vector &boundary_values) const;

   /// User-provided finite element space; not owned by the solver.
   std::shared_ptr<ParFiniteElementSpace> fespace_owner_;

   /// User-provided finite element space.
   ParFiniteElementSpace &fespace_;

   /// Boundary attribute IDs treated as essential Dirichlet boundaries.
   mutable std::set<int> boundary_ids_;

   /// Essential true dofs derived from boundary_ids_.
   mutable Array<int> ess_tdofs_;

   /// Diffusion coefficient used in the domain operator.
   std::shared_ptr<Coefficient> diffusion_coefficient_;

   /// Mass coefficient used in the domain operator.
   std::shared_ptr<Coefficient> mass_coefficient_;

   /// Optional owner for a diffusion QuadratureFunction coefficient.
   std::shared_ptr<QuadratureFunction> diffusion_qf_owner_;

   /// Optional owner for a mass QuadratureFunction coefficient.
   std::shared_ptr<QuadratureFunction> mass_qf_owner_;

   /// Optional owner for a diffusion ParGridFunction coefficient.
   std::shared_ptr<ParGridFunction> diffusion_gf_owner_;

   /// Optional owner for a mass ParGridFunction coefficient.
   std::shared_ptr<ParGridFunction> mass_gf_owner_;

   /// Domain RHS coefficients keyed by element attribute.
   mutable AttributeCoefficientMap rhs_coefficients_;

   /// Dirichlet boundary coefficients keyed by boundary attribute.
   mutable AttributeCoefficientMap boundary_coefficients_;

   /// Domain integration order used for operators and RHS forms.
   int integration_order_;

   /// Print level for the Krylov solver and AMG preconditioners.
   int print_level_ = -1;

   /// True when the system operator/preconditioner must be rebuilt.
   mutable bool needs_assembly_ = true;

   /// Partial-assembly diffusion+mass bilinear form.
   mutable std::unique_ptr<ParBilinearForm> form_;

   /// True-dof partial-assembly system operator.
   mutable OperatorHandle system_operator_;

   /// LOR mesh used for high-order AMG preconditioning.
   mutable std::unique_ptr<ParMesh> lor_mesh_;

   /// Order-one LOR finite element collection.
   mutable std::unique_ptr<FiniteElementCollection> lor_fec_;

   /// LOR finite element space used for AMG preconditioning.
   mutable std::unique_ptr<ParFiniteElementSpace> lor_fespace_;

   /// LOR diffusion grid-function coefficient storage.
   mutable std::shared_ptr<ParGridFunction> lor_diffusion_gf_;

   /// LOR mass grid-function coefficient storage.
   mutable std::shared_ptr<ParGridFunction> lor_mass_gf_;

   /// LOR diffusion coefficient wrapper.
   mutable std::shared_ptr<Coefficient> lor_diffusion_coefficient_;

   /// LOR mass coefficient wrapper.
   mutable std::shared_ptr<Coefficient> lor_mass_coefficient_;

   /// Assembled LOR bilinear form used to create AMG matrix.
   mutable std::unique_ptr<ParBilinearForm> lor_form_;

   /// Assembled LOR true-dof operator for AMG.
   mutable OperatorHandle lor_operator_;

   /// Assembled order-1 bilinear form used to create AMG matrix.
   mutable std::unique_ptr<ParBilinearForm> assembled_form_;

   /// Assembled order-1 true-dof operator for AMG.
   mutable OperatorHandle assembled_operator_;

   /// Cached RHS vector after constrained values are applied.
   mutable Vector constrained_rhs_;

   /// Cached nonzero boundary values on essential true dofs.
   mutable Vector boundary_true_dofs_;

   /// Cached zero values on essential true dofs for adjoint applications.
   mutable Vector homogeneous_boundary_true_dofs_;

   /// Cached all true dofs of projected boundary grid function.
   mutable Vector boundary_all_true_dofs_;

   /// Work vector for transferring HO coefficient values to the LOR space.
   mutable Vector lor_transfer_true_dofs_;

   /// Cached FormLinearSystem solution vector.
   mutable Vector solve_X_;

   /// Cached FormLinearSystem RHS vector.
   mutable Vector solve_B_;

   /// Cached solved true-dof vector.
   mutable Vector solve_Y_;

   /// Cached boundary attribute marker.
   mutable Array<int> boundary_marker_;

   /// Cached operator handle used by FormLinearSystem.
   mutable OperatorHandle solve_operator_;

   /// Grid function used to project nonzero boundary coefficients.
   mutable std::unique_ptr<ParGridFunction> boundary_grid_function_;

   /// Cached domain RHS linear form.
   mutable std::unique_ptr<ParLinearForm> rhs_form_;

   /// True when rhs_form_ must be rebuilt.
   mutable bool rhs_form_dirty_ = true;

   /// True when rhs_form_ must be reassembled.
   mutable bool rhs_vector_dirty_ = true;

   /// True when cached boundary true-dof values must be rebuilt.
   mutable bool boundary_true_dofs_dirty_ = true;

   // Declared last so it is destroyed before operators/forms it may reference.
   /// AMG preconditioner owned by the solver.
   mutable std::unique_ptr<Solver> preconditioner_;

   /// Cached CG solver configured with system_operator_ and preconditioner_.
   mutable std::unique_ptr<CGSolver> cg_solver_;
};

/// PDE topology optimization filter of Lazarov and Sigmund (2011).
///
/// The filter maps an input density true vector rho to a filtered true vector
/// rho_f by first forming the mass RHS in the filtered space and then solving
///
///     -R^2 Delta rho_f + rho_f = rho
///
/// with natural Neumann boundary conditions.  The paper's radius convention is
/// implemented by SetFilterRadius(r_min), which sets R = r_min/(2 sqrt(3)) and
/// the diffusion coefficient to R^2.  MultTranspose() applies the adjoint map:
/// solve the homogeneous-adjoint diffusion-mass problem in the filtered space,
/// then apply the transpose of the mass map to propagate gradients to the input
/// true vector.
class PDEFilter : public Operator
{
public:
   /// Construct a PDE filter from an input space to a filtered/output space.
   PDEFilter(ParFiniteElementSpace &input_space,
             ParFiniteElementSpace &filtered_space);

   /// Construct a PDE filter and keep shared ownership of both spaces.
   PDEFilter(std::shared_ptr<ParFiniteElementSpace> input_space,
             std::shared_ptr<ParFiniteElementSpace> filtered_space);

   /// Set the diffusion coefficient in `-diffusion Delta rho_f + rho_f = rho`.
   void SetDiffusionCoefficient(real_t diffusion);

   /// Set the paper radius r_min, using diffusion = (r_min/(2 sqrt(3)))^2.
   void SetFilterRadius(real_t r_min);

   /// Return the filter radius r_min currently represented by diffusion_.
   real_t GetFilterRadius() const { return filter_radius_; }

   /// Return the diffusion coefficient used by the diffusion-mass solve.
   real_t GetDiffusionCoefficient() const { return diffusion_; }

   /// Return the input finite element space.
   ParFiniteElementSpace &GetInputFESpace() { return input_space_; }

   /// Return the input finite element space.
   const ParFiniteElementSpace &GetInputFESpace() const { return input_space_; }

   /// Return the filtered/output finite element space.
   ParFiniteElementSpace &GetFilteredFESpace() { return filtered_space_; }

   /// Return the filtered/output finite element space.
   const ParFiniteElementSpace &GetFilteredFESpace() const
   { return filtered_space_; }

   /// Return the mass map used to build the diffusion-mass RHS.
   TrueMassMapOperator &GetMassMap() { return mass_map_; }

   /// Return the mass map used to build the diffusion-mass RHS.
   const TrueMassMapOperator &GetMassMap() const { return mass_map_; }

   /// Return the internally owned diffusion-mass solver.
   DiffusionMassSolver &GetSolver() { return solver_; }

   /// Return the internally owned diffusion-mass solver.
   const DiffusionMassSolver &GetSolver() const { return solver_; }

   /// Apply the filter: filtered_true = A^{-1} M input_true.
   void Mult(const Vector &input, Vector &filtered) const override;

   /// Apply the filter without checking assembly state.  The caller must first
   /// call Assemble() collectively.
   void MultAssembled(const Vector &input, Vector &filtered) const;

   /// Assemble the internal mass map and diffusion-mass solver if needed.
   void Assemble() const;

   /// Apply the adjoint filter: input_bar = M^T A^{-T} filtered_bar.
   void MultTranspose(const Vector &filtered_bar, Vector &input_bar) const override;

   /// Apply the adjoint filter without checking assembly state.  The caller
   /// must first call Assemble() collectively.
   void MultTransposeAssembled(const Vector &filtered_bar,
                               Vector &input_bar) const;

private:
   /// Optional shared owner for input_space_.
   std::shared_ptr<ParFiniteElementSpace> input_space_owner_;

   /// Optional shared owner for filtered_space_.
   std::shared_ptr<ParFiniteElementSpace> filtered_space_owner_;

   /// Input density finite element space.
   ParFiniteElementSpace &input_space_;

   /// Filtered density finite element space.
   ParFiniteElementSpace &filtered_space_;

   /// Radius r_min in the Lazarov-Sigmund convention.
   real_t filter_radius_ = 0.0;

   /// Diffusion coefficient R^2 used in the PDE filter.
   real_t diffusion_ = 0.0;

   /// True when the mass map or diffusion-mass solver must be assembled.
   mutable bool needs_assembly_ = true;

   /// Mass map from input true vectors to filtered-space RHS true vectors.
   TrueMassMapOperator mass_map_;

   /// Diffusion-mass solver on the filtered space.
   DiffusionMassSolver solver_;

   /// Cached RHS vector M input.
   mutable Vector rhs_;

   /// Cached adjoint solve A^{-T} filtered_bar.
   mutable Vector adjoint_;
};

/// Mass map from scalar QuadratureFunction values to FE true-dof RHS values.
///
/// Given scalar quadrature values rho_q, this operator computes
///
///     b_i = sum_K sum_q rho_q(x_q) phi_i(x_q) w_q |J_K(x_q)|,
///
/// returning the assembled true-dof vector b in @a output_space.  The transpose
/// maps an output-space true vector lambda to scalar quadrature values
///
///     rho_bar_q = sum_i lambda_i phi_i(x_q) w_q |J_K(x_q)|,
///
/// which is the adjoint needed to propagate gradients from an FE RHS back to a
/// quadrature input field.  The quadrature space and FE space must be defined
/// on the same ParMesh.  The forward map uses MFEM's QuadratureLFIntegrator;
/// the transpose is assembled explicitly at the element level and then reduced
/// through the FE-space prolongation.
class QuadratureFunctionMassMapOperator : public Operator
{
public:
   /// Construct a map from scalar quadrature values to @a output_space.
   QuadratureFunctionMassMapOperator(QuadratureSpace &input_qspace,
                                     ParFiniteElementSpace &output_space);

   /// Construct a map and keep shared ownership of both spaces.
   QuadratureFunctionMassMapOperator(
      std::shared_ptr<QuadratureSpace> input_qspace,
      std::shared_ptr<ParFiniteElementSpace> output_space);

   /// Return the quadrature space defining the input vector layout.
   QuadratureSpace &GetInputQuadratureSpace() { return input_qspace_; }

   /// Return the quadrature space defining the input vector layout.
   const QuadratureSpace &GetInputQuadratureSpace() const
   { return input_qspace_; }

   /// Return the finite element space receiving the RHS true vector.
   ParFiniteElementSpace &GetOutputFESpace() { return output_space_; }

   /// Return the finite element space receiving the RHS true vector.
   const ParFiniteElementSpace &GetOutputFESpace() const
   { return output_space_; }

   /// Return true when cached assembly data must be rebuilt.
   bool NeedsAssembly() const { return needs_assembly_; }

   /// Force or clear the lazy assembly flag.
   void SetNeedsAssembly(bool needs_assembly = true) const;

   /// Assemble cached helper objects and validate space compatibility.
   void Assemble() const;

   /// Apply the quadrature-to-RHS mass map: output_true = M_q input_q.
   void Mult(const Vector &input_q, Vector &output_true) const override;

   /// Apply Mult() without checking assembly state.  The caller must first call
   /// Assemble() collectively.
   void MultAssembled(const Vector &input_q, Vector &output_true) const;

   /// Apply the adjoint map: input_q_bar = M_q^T output_true_bar.
   void MultTranspose(const Vector &output_true_bar,
                      Vector &input_q_bar) const override;

   /// Apply MultTranspose() without checking assembly state.
   void MultTransposeAssembled(const Vector &output_true_bar,
                               Vector &input_q_bar) const;

private:
   /// Verify that a vector has the scalar quadrature input size.
   void ValidateInputSize(const Vector &input_q) const;

   /// Optional shared owner for input_qspace_.
   std::shared_ptr<QuadratureSpace> input_qspace_owner_;

   /// Optional shared owner for output_space_.
   std::shared_ptr<ParFiniteElementSpace> output_space_owner_;

   /// Scalar quadrature input space.
   QuadratureSpace &input_qspace_;

   /// Output finite element space.
   ParFiniteElementSpace &output_space_;

   /// True when cached objects must be rebuilt.
   mutable bool needs_assembly_ = true;

   /// Quadrature function view over the current input vector during Mult().
   mutable std::unique_ptr<QuadratureFunction> input_qf_view_;

   /// Coefficient wrapper around input_qf_view_.
   mutable std::unique_ptr<QuadratureFunctionCoefficient> input_qf_coeff_;

   /// Linear form used for the forward quadrature-to-RHS map.
   mutable std::unique_ptr<ParLinearForm> rhs_form_;

   /// Local FE vector used before true-dof parallel assembly.
   mutable Vector local_output_;

   /// Local FE adjoint vector obtained by prolonging true adjoints.
   mutable Vector local_adjoint_;

   /// Element dof indices reused by the explicit transpose.
   mutable Array<int> vdofs_;

   /// Element shape values reused by the explicit transpose.
   mutable Vector shape_;
};

/// PDE filter whose input is a scalar QuadratureFunction vector.
///
/// This operator has the same PDE solve as PDEFilter but replaces the FE input
/// mass map with QuadratureFunctionMassMapOperator.  Mult() accepts a scalar
/// quadrature vector and returns filtered FE true dofs.  MultTranspose() maps a
/// filtered-space gradient back to scalar quadrature-point gradients, including
/// the quadrature weights and geometric factors from the RHS map.
class QuadraturePDEFilter : public Operator
{
public:
   /// Construct a quadrature-input PDE filter.
   QuadraturePDEFilter(QuadratureSpace &input_qspace,
                       ParFiniteElementSpace &filtered_space);

   /// Construct a quadrature-input PDE filter and keep shared ownership.
   QuadraturePDEFilter(std::shared_ptr<QuadratureSpace> input_qspace,
                       std::shared_ptr<ParFiniteElementSpace> filtered_space);

   /// Set the diffusion coefficient in `-diffusion Delta rho_f + rho_f = rho`.
   void SetDiffusionCoefficient(real_t diffusion);

   /// Set the paper radius r_min, using diffusion = (r_min/(2 sqrt(3)))^2.
   void SetFilterRadius(real_t r_min);

   /// Return the filter radius r_min currently represented by diffusion_.
   real_t GetFilterRadius() const { return filter_radius_; }

   /// Return the diffusion coefficient used by the diffusion-mass solve.
   real_t GetDiffusionCoefficient() const { return diffusion_; }

   /// Return the scalar quadrature input space.
   QuadratureSpace &GetInputQuadratureSpace() { return input_qspace_; }

   /// Return the scalar quadrature input space.
   const QuadratureSpace &GetInputQuadratureSpace() const
   { return input_qspace_; }

   /// Return the filtered/output finite element space.
   ParFiniteElementSpace &GetFilteredFESpace() { return filtered_space_; }

   /// Return the filtered/output finite element space.
   const ParFiniteElementSpace &GetFilteredFESpace() const
   { return filtered_space_; }

   /// Return the quadrature mass map used to build the PDE RHS.
   QuadratureFunctionMassMapOperator &GetMassMap() { return mass_map_; }

   /// Return the quadrature mass map used to build the PDE RHS.
   const QuadratureFunctionMassMapOperator &GetMassMap() const
   { return mass_map_; }

   /// Return the internally owned diffusion-mass solver.
   DiffusionMassSolver &GetSolver() { return solver_; }

   /// Return the internally owned diffusion-mass solver.
   const DiffusionMassSolver &GetSolver() const { return solver_; }

   /// Apply the filter: filtered_true = A^{-1} M_q input_q.
   void Mult(const Vector &input_q, Vector &filtered) const override;

   /// Apply the filter without checking assembly state.
   void MultAssembled(const Vector &input_q, Vector &filtered) const;

   /// Assemble the internal quadrature mass map and diffusion-mass solver.
   void Assemble() const;

   /// Apply the adjoint filter: input_q_bar = M_q^T A^{-T} filtered_bar.
   void MultTranspose(const Vector &filtered_bar,
                      Vector &input_q_bar) const override;

   /// Apply the adjoint filter without checking assembly state.
   void MultTransposeAssembled(const Vector &filtered_bar,
                               Vector &input_q_bar) const;

private:
   /// Optional shared owner for input_qspace_.
   std::shared_ptr<QuadratureSpace> input_qspace_owner_;

   /// Optional shared owner for filtered_space_.
   std::shared_ptr<ParFiniteElementSpace> filtered_space_owner_;

   /// Scalar quadrature input space.
   QuadratureSpace &input_qspace_;

   /// Filtered density finite element space.
   ParFiniteElementSpace &filtered_space_;

   /// Radius r_min in the Lazarov-Sigmund convention.
   real_t filter_radius_ = 0.0;

   /// Diffusion coefficient R^2 used in the PDE filter.
   real_t diffusion_ = 0.0;

   /// True when the mass map or diffusion-mass solver must be assembled.
   mutable bool needs_assembly_ = true;

   /// Mass map from quadrature vectors to filtered-space RHS true vectors.
   QuadratureFunctionMassMapOperator mass_map_;

   /// Diffusion-mass solver on the filtered space.
   DiffusionMassSolver solver_;

   /// Cached RHS vector M_q input_q.
   mutable Vector rhs_;

   /// Cached adjoint solve A^{-T} filtered_bar.
   mutable Vector adjoint_;
};

} // namespace mfem

#endif
