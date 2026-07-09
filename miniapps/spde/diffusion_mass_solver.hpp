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
   /// Identifies whether an AttributeCoefficientMap stores domain RHS data,
   /// surface-load data, or Dirichlet boundary data.
   enum class MapKind { DomainRHS, SurfaceRHS, Boundary };

   /// Attribute-keyed scalar coefficient container used for volume loads,
   /// surface loads, and Dirichlet data.  Only Dirichlet boundary maps notify
   /// the owning solver that the corresponding boundary attribute is essential.
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

      /// Semantic role of the map.
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

   /// Set constant multipliers for the separately stored diffusion and mass
   /// integrators and mark the operator/preconditioner dirty.  The assembled
   /// operator becomes
   ///     diffusion_scale * K(base_diffusion)
   ///   + mass_scale      * M(base_mass).
   /// This is intended for repeated Balakrishnan shifted solves where only the
   /// scalar weights change between steps while the base coefficients remain
   /// fixed.
   void SetScalingConstants(real_t diffusion_scale, real_t mass_scale);

   /// Return the current diffusion scaling constant.
   real_t GetDiffusionScalingConstant() const { return diffusion_scale_; }

   /// Return the current mass scaling constant.
   real_t GetMassScalingConstant() const { return mass_scale_; }

   /// Return the unscaled diffusion coefficient used as the K coefficient.
   const Coefficient &GetBaseDiffusionCoefficient() const
   { return *diffusion_base_coefficient_; }

   /// Return the unscaled mass coefficient used as the M coefficient.
   const Coefficient &GetBaseMassCoefficient() const
   { return *mass_base_coefficient_; }

   /// Access domain RHS coefficients keyed by element attribute.
   AttributeCoefficientMap &RHS() { return rhs_coefficients_; }

   /// Access natural surface-load coefficients keyed by boundary attribute.
   AttributeCoefficientMap &SurfaceLoads() { return surface_load_coefficients_; }

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

   /// Set the relative tolerance used by the outer CG solver.
   void SetRelTol(real_t rel_tol);

   /// Set the absolute tolerance used by the outer CG solver.
   void SetAbsTol(real_t abs_tol);

   /// Set the maximum number of outer CG iterations.
   void SetMaxIter(int max_iter);

   /// Set the print level used by the outer CG solver and AMG preconditioners.
   void SetPrintLevel(int print_level)
   {
      print_level_ = print_level;
      needs_assembly_ = true;
   }

   /// Return the current solver print level.
   int GetPrintLevel() const { return print_level_; }

   /// Return the current outer CG relative tolerance.
   real_t GetRelTol() const { return rel_tol_; }

   /// Return the current outer CG absolute tolerance.
   real_t GetAbsTol() const { return abs_tol_; }

   /// Return the current outer CG maximum iteration count.
   int GetMaxIter() const { return max_iter_; }

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

   /// Verify that a boundary QuadratureFunction matches this solver's mesh and rule.
   void ValidateSurfaceQuadratureFunction(const QuadratureFunction &qf) const;

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
   /// @a coefficient is the (possibly scaled) coefficient returned directly
   /// when no LOR transfer is required; @a base_coefficient is the unscaled
   /// coefficient projected when transferring a QuadratureFunction source, so
   /// that @a scale is applied exactly once to the transferred data.
   std::shared_ptr<Coefficient> MakeLORCoefficient(
      const std::shared_ptr<Coefficient> &coefficient,
      const std::shared_ptr<Coefficient> &base_coefficient,
      const std::shared_ptr<ParGridFunction> &ho_gf,
      const std::shared_ptr<QuadratureFunction> &qf,
      std::shared_ptr<ParGridFunction> &lor_gf,
      real_t scale,
      std::shared_ptr<Coefficient> &lor_base_coefficient) const;

   /// Rebuild active ProductCoefficient wrappers after a base coefficient
   /// pointer changes.
   void RefreshScaledCoefficients();

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

   /// Base diffusion coefficient before applying diffusion_scale_.
   std::shared_ptr<Coefficient> diffusion_base_coefficient_;

   /// Base mass coefficient before applying mass_scale_.
   std::shared_ptr<Coefficient> mass_base_coefficient_;

   /// Scaled diffusion coefficient used in the domain operator.
   std::shared_ptr<Coefficient> diffusion_coefficient_;

   /// Scaled mass coefficient used in the domain operator.
   std::shared_ptr<Coefficient> mass_coefficient_;

   /// Product wrapper representing diffusion_scale_ * diffusion_base_coefficient_.
   std::shared_ptr<ProductCoefficient> scaled_diffusion_coefficient_;

   /// Product wrapper representing mass_scale_ * mass_base_coefficient_.
   std::shared_ptr<ProductCoefficient> scaled_mass_coefficient_;

   /// Constant multiplier for the diffusion integrator.
   real_t diffusion_scale_ = 1.0;

   /// Constant multiplier for the mass integrator.
   real_t mass_scale_ = 1.0;

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

   /// Natural surface-load coefficients keyed by boundary attribute.
   mutable AttributeCoefficientMap surface_load_coefficients_;

   /// Dirichlet boundary coefficients keyed by boundary attribute.
   mutable AttributeCoefficientMap boundary_coefficients_;

   /// Domain integration order used for operators and RHS forms.
   int integration_order_;

   /// Print level for the Krylov solver and AMG preconditioners.
   int print_level_ = -1;

   /// Relative tolerance for the outer CG solver.
   real_t rel_tol_ = 1.0e-12;

   /// Absolute tolerance for the outer CG solver.
   real_t abs_tol_ = 0.0;

   /// Maximum outer CG iterations; <= 0 uses max(200, 2*Height()).
   int max_iter_ = 0;

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

   /// Unscaled LOR diffusion coefficient kept alive when wrapped by a product.
   mutable std::shared_ptr<Coefficient> lor_diffusion_base_coefficient_;

   /// Unscaled LOR mass coefficient kept alive when wrapped by a product.
   mutable std::shared_ptr<Coefficient> lor_mass_base_coefficient_;

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

/// Solver for a scalar diffusion equation on a user-provided
/// ParFiniteElementSpace:
///     -div(diffusion_coefficient grad u) = rhs.
///
/// This class exposes the same coefficient, RHS, and boundary-condition style
/// as the other SPDE solvers, but owns an independent diffusion-only assembly
/// and solve path.  The system operator is assembled with partial assembly.
/// First-order spaces use BoomerAMG directly on an assembled diffusion matrix;
/// higher-order spaces use BoomerAMG on a LOR diffusion matrix as the
/// preconditioner.
class DiffusionSolver : public Solver
{
public:
   /// Identifies whether an AttributeCoefficientMap stores domain RHS data,
   /// surface-load data, or Dirichlet boundary data.
   enum class MapKind { DomainRHS, SurfaceRHS, Boundary };

   /// Attribute-keyed scalar coefficient container used for volume loads,
   /// surface loads, and Dirichlet data.  Only Dirichlet boundary maps notify
   /// the owning solver that the corresponding boundary attribute is essential.
   class AttributeCoefficientMap
   {
   public:
      AttributeCoefficientMap() = default;

      /// Attach this map to a solver.  Called by DiffusionSolver.
      void SetOwner(DiffusionSolver *owner, MapKind kind);

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
      std::shared_ptr<Coefficient> ShareCoefficient(Coefficient &coefficient,
                                                    bool transfer_ownership);
      std::shared_ptr<QuadratureFunction> ShareQuadratureFunction(
         QuadratureFunction &qf, bool transfer_ownership);
      std::shared_ptr<ParGridFunction> ShareParGridFunction(
         ParGridFunction &gf, bool transfer_ownership);
      void NotifyChanged(int attr);

      DiffusionSolver *owner_ = nullptr;
      MapKind kind_ = MapKind::DomainRHS;
      std::map<int, std::shared_ptr<Coefficient> > coefficients_;
      std::map<int, std::shared_ptr<QuadratureFunction> > qf_owners_;
      std::map<int, std::shared_ptr<ParGridFunction> > gf_owners_;
      mutable std::unique_ptr<PWCoefficient> piecewise_;
   };

   /// Create a diffusion solver on @a fespace without taking ownership.
   explicit DiffusionSolver(ParFiniteElementSpace &fespace);

   /// Create a diffusion solver and keep shared ownership of @a fespace.
   explicit DiffusionSolver(std::shared_ptr<ParFiniteElementSpace> fespace);

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

   /// Access domain RHS coefficients keyed by element attribute.
   AttributeCoefficientMap &RHS() { return rhs_coefficients_; }

   /// Access natural surface-load coefficients keyed by boundary attribute.
   AttributeCoefficientMap &SurfaceLoads() { return surface_load_coefficients_; }

   /// Access Dirichlet boundary coefficients keyed by boundary attribute.
   AttributeCoefficientMap &Boundary() { return boundary_coefficients_; }

   /// Mark a boundary attribute ID as essential with homogeneous data unless a
   /// boundary coefficient for that ID is also provided.
   void AddBoundaryID(int id);

   /// Remove all essential boundary IDs and boundary coefficients.
   void ClearBoundaryConditions();

   /// Return true when Assemble() is required before applying the operator.
   bool NeedsAssembly() const { return needs_assembly_; }

   /// Force or clear the lazy assembly flag.
   void SetNeedsAssembly(bool needs_assembly = true) const;

   /// Set the relative tolerance used by the outer CG solver.
   void SetRelTol(real_t rel_tol);

   /// Set the absolute tolerance used by the outer CG solver.
   void SetAbsTol(real_t abs_tol);

   /// Set the maximum number of outer CG iterations.
   void SetMaxIter(int max_iter);

   /// Set the print level used by the outer CG solver and AMG preconditioners.
   void SetPrintLevel(int print_level)
   {
      print_level_ = print_level;
      needs_assembly_ = true;
   }

   /// Return the current solver print level.
   int GetPrintLevel() const { return print_level_; }

   /// Return the current outer CG relative tolerance.
   real_t GetRelTol() const { return rel_tol_; }

   /// Return the current outer CG absolute tolerance.
   real_t GetAbsTol() const { return abs_tol_; }

   /// Return the current outer CG maximum iteration count.
   int GetMaxIter() const { return max_iter_; }

   /// Return the integration order used for domain operators and RHS forms.
   int GetIntegrationOrder() const { return integration_order_; }

   /// Return the domain integration rule used for a geometry type.
   const IntegrationRule &GetIntegrationRule(Geometry::Type geom) const;

   /// Reassemble the partial-assembly operator and AMG preconditioner if dirty.
   void Assemble() const;

   /// Solve the already-eliminated true-dof linear system A x = rhs.
   void Mult(const Vector &rhs, Vector &solution) const override;

   /// Same as Mult(), but skips the lazy Assemble() check.
   void MultAssembled(const Vector &rhs, Vector &solution) const;

   /// Apply the adjoint solve.  The diffusion operator is symmetric.
   void MultTranspose(const Vector &rhs, Vector &solution) const override;

   /// Same as MultTranspose(), but skips the lazy Assemble() check.
   void MultTransposeAssembled(const Vector &rhs, Vector &solution) const;

   /// Assemble RHS(), apply Boundary(), solve, and recover the FE solution.
   void Solve(ParGridFunction &solution) const;

   /// IterativeSolver compatibility.
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

   std::shared_ptr<Coefficient> ShareCoefficient(Coefficient &coefficient,
                                                 bool transfer_ownership);
   std::shared_ptr<QuadratureFunction> ShareQuadratureFunction(
      QuadratureFunction &qf, bool transfer_ownership);
   std::shared_ptr<ParGridFunction> ShareParGridFunction(
      ParGridFunction &gf, bool transfer_ownership);
   std::shared_ptr<Coefficient> MakeQuadratureCoefficient(
      std::shared_ptr<QuadratureFunction> qf) const;
   std::shared_ptr<Coefficient> MakeGridFunctionCoefficient(
      std::shared_ptr<ParGridFunction> gf) const;
   void ValidateQuadratureFunction(const QuadratureFunction &qf) const;
   void ValidateSurfaceQuadratureFunction(const QuadratureFunction &qf) const;
   void ValidateParGridFunction(const ParGridFunction &gf) const;
   void SolveSystem(const Vector &rhs, Vector &solution,
                    bool apply_constraints,
                    bool use_boundary_values) const;
   void MarkCoefficientChanged();
   void MarkRHSChanged();
   void MarkBoundaryChanged(int id);
   void BuildEssentialMarker(Array<int> &marker) const;
   void BuildEssentialTrueDofs() const;
   void BuildPreconditioner() const;
   void BuildAMGPreconditionerOnFESpace() const;
   void BuildLORAMGPreconditioner() const;
   std::shared_ptr<Coefficient> MakeLORCoefficient(
      const std::shared_ptr<Coefficient> &coefficient,
      const std::shared_ptr<ParGridFunction> &ho_gf,
      const std::shared_ptr<QuadratureFunction> &qf,
      std::shared_ptr<ParGridFunction> &lor_gf,
      std::shared_ptr<Coefficient> &lor_base_coefficient) const;
   void BuildBoundaryTrueDofs(Vector &boundary_values) const;

   std::shared_ptr<ParFiniteElementSpace> fespace_owner_;
   ParFiniteElementSpace &fespace_;
   mutable std::set<int> boundary_ids_;
   mutable Array<int> ess_tdofs_;

   std::shared_ptr<Coefficient> diffusion_coefficient_;
   std::shared_ptr<QuadratureFunction> diffusion_qf_owner_;
   std::shared_ptr<ParGridFunction> diffusion_gf_owner_;

   mutable AttributeCoefficientMap rhs_coefficients_;
   mutable AttributeCoefficientMap surface_load_coefficients_;
   mutable AttributeCoefficientMap boundary_coefficients_;

   int integration_order_;
   int print_level_ = -1;
   real_t rel_tol_ = 1.0e-12;
   real_t abs_tol_ = 0.0;
   int max_iter_ = 0;
   mutable bool needs_assembly_ = true;

   mutable std::unique_ptr<ParBilinearForm> form_;
   mutable OperatorHandle system_operator_;

   mutable std::unique_ptr<ParMesh> lor_mesh_;
   mutable std::unique_ptr<FiniteElementCollection> lor_fec_;
   mutable std::unique_ptr<ParFiniteElementSpace> lor_fespace_;
   mutable std::shared_ptr<ParGridFunction> lor_diffusion_gf_;
   mutable std::shared_ptr<Coefficient> lor_diffusion_coefficient_;
   mutable std::shared_ptr<Coefficient> lor_diffusion_base_coefficient_;
   mutable std::unique_ptr<ParBilinearForm> lor_form_;
   mutable OperatorHandle lor_operator_;

   mutable std::unique_ptr<ParBilinearForm> assembled_form_;
   mutable OperatorHandle assembled_operator_;

   mutable Vector constrained_rhs_;
   mutable Vector boundary_true_dofs_;
   mutable Vector homogeneous_boundary_true_dofs_;
   mutable Vector boundary_all_true_dofs_;
   mutable Vector lor_transfer_true_dofs_;
   mutable Vector solve_X_;
   mutable Vector solve_B_;
   mutable Vector solve_Y_;
   mutable Array<int> boundary_marker_;
   mutable OperatorHandle solve_operator_;
   mutable std::unique_ptr<ParGridFunction> boundary_grid_function_;
   mutable std::unique_ptr<ParLinearForm> rhs_form_;
   mutable bool rhs_form_dirty_ = true;
   mutable bool rhs_vector_dirty_ = true;
   mutable bool boundary_true_dofs_dirty_ = true;

   mutable std::unique_ptr<Solver> preconditioner_;
   mutable std::unique_ptr<CGSolver> cg_solver_;
};

/// Parallel Stokes/Brinkman solver for
///     -div(viscosity grad u) + brinkman u + grad p = acceleration,
///                                      div u = 0.
///
/// The velocity and pressure spaces are supplied by the user.  The solver
/// assembles a 2x2 true-dof block operator and accepts/returns BlockVector
/// objects in Mult().  Velocity Dirichlet and pressure Dirichlet data are
/// keyed by boundary attributes.  Boundary attributes without a specified
/// condition are left unconstrained, corresponding to open-flow/natural
/// boundaries.  The velocity block is preconditioned with BoomerAMG on an LOR
/// diffusion-plus-mass operator for high-order discretizations, or directly on
/// the first-order operator.  The pressure block can use a scaled pressure mass
/// matrix, an LSC approximation, or a Cahouet-Chabard preconditioner when
/// Brinkman penalization is set.
class StokesSolver : public Solver
{
public:
   /// Outer Krylov method used for the coupled Stokes block system.
   enum class KrylovSolver { MINRES, GMRES };

   /// Preconditioner strategy used for the velocity block.
   enum class VelocityPreconditioner { AMG, CG };

   /// Preconditioner strategy used for the pressure Schur complement block.
   enum class PressurePreconditioner { DIAGONAL_MASS, CG, AMG, LSC,
                                       CAHOUET_CHABARD };

   /// Krylov method used for the pressure diffusion solve in CC preconditioning.
   enum class CCDiffusionSolver { GMRES, CG };

   /// Velocity operator used in the LSC H block.
   enum class LSCVelocityOperator { PA, ASSEMBLED };

   /// Velocity operator used to extract the LSC velocity diagonal.
   enum class LSCDiagonalOperator { MATCH_VELOCITY, PA, ASSEMBLED };

   /// Preconditioner used for the inner LSC Q solve.
   enum class LSCQPreconditioner { OPERATOR_JACOBI, JACOBI, AMG };

   /// Attribute-keyed vector coefficient map used for acceleration and
   /// velocity Dirichlet data.
   class VectorAttributeCoefficientMap
   {
   public:
      /// Construct an unattached empty coefficient map.
      VectorAttributeCoefficientMap() = default;

      /// Attach the map to @a owner and set whether it stores boundary data.
      void SetOwner(StokesSolver *owner, bool boundary_map);

      /// Add a referenced vector coefficient on attribute @a attr.
      void Add(int attr, VectorCoefficient &coefficient,
               bool transfer_ownership = false);

      /// Add a shared vector coefficient on attribute @a attr.
      void Add(int attr, std::shared_ptr<VectorCoefficient> coefficient);

      /// Remove all stored coefficients and notify the owning solver.
      void Clear();

      /// Return true if no coefficients have been added.
      bool Empty() const { return coefficients_.empty(); }

      /// Return a piecewise vector coefficient view over stored attributes.
      VectorCoefficient &AsCoefficient() const;

   private:
      /// Wrap a coefficient reference in a shared pointer.
      std::shared_ptr<VectorCoefficient> ShareCoefficient(
         VectorCoefficient &coefficient, bool transfer_ownership);

      /// Notify the owning solver that this map changed on @a attr.
      void NotifyChanged(int attr);

      /// Owning Stokes solver; not owned by this map.
      StokesSolver *owner_ = nullptr;

      /// True when this map stores velocity boundary data.
      bool boundary_map_ = false;

      /// Attribute-to-vector-coefficient storage.
      std::map<int, std::shared_ptr<VectorCoefficient> > coefficients_;

      /// Cached piecewise vector coefficient rebuilt by AsCoefficient().
      mutable std::unique_ptr<PWVectorCoefficient> piecewise_;
   };

   /// Attribute-keyed scalar coefficient map used for pressure Dirichlet data.
   class ScalarAttributeCoefficientMap
   {
   public:
      /// Construct an unattached empty coefficient map.
      ScalarAttributeCoefficientMap() = default;

      /// Attach the map to @a owner and set whether it stores boundary data.
      void SetOwner(StokesSolver *owner, bool boundary_map);

      /// Add a constant scalar coefficient on attribute @a attr.
      void Add(int attr, real_t value);

      /// Add a referenced scalar coefficient on attribute @a attr.
      void Add(int attr, Coefficient &coefficient,
               bool transfer_ownership = false);

      /// Add a shared scalar coefficient on attribute @a attr.
      void Add(int attr, std::shared_ptr<Coefficient> coefficient);

      /// Remove all stored coefficients and notify the owning solver.
      void Clear();

      /// Return true if no coefficients have been added.
      bool Empty() const { return coefficients_.empty(); }

      /// Return a piecewise scalar coefficient view over stored attributes.
      Coefficient &AsCoefficient() const;

   private:
      /// Wrap a coefficient reference in a shared pointer.
      std::shared_ptr<Coefficient> ShareCoefficient(Coefficient &coefficient,
                                                    bool transfer_ownership);

      /// Notify the owning solver that this map changed on @a attr.
      void NotifyChanged(int attr);

      /// Owning Stokes solver; not owned by this map.
      StokesSolver *owner_ = nullptr;

      /// True when this map stores pressure boundary data.
      bool boundary_map_ = false;

      /// Attribute-to-scalar-coefficient storage.
      std::map<int, std::shared_ptr<Coefficient> > coefficients_;

      /// Cached piecewise scalar coefficient rebuilt by AsCoefficient().
      mutable std::unique_ptr<PWCoefficient> piecewise_;
   };

   /// Construct a Stokes solver on user-owned velocity and pressure spaces.
   StokesSolver(ParFiniteElementSpace &velocity_space,
                ParFiniteElementSpace &pressure_space);

   /// Construct a Stokes solver and keep shared ownership of both spaces.
   StokesSolver(std::shared_ptr<ParFiniteElementSpace> velocity_space,
                std::shared_ptr<ParFiniteElementSpace> pressure_space);

   /// Set viscosity to a constant value.
   void SetViscosity(real_t value);

   /// Set viscosity from a referenced coefficient.
   void SetViscosity(Coefficient &coefficient,
                     bool transfer_ownership = false);

   /// Set viscosity from a shared coefficient.
   void SetViscosity(std::shared_ptr<Coefficient> coefficient);

   /// Set viscosity from a referenced QuadratureFunction.
   void SetViscosity(QuadratureFunction &qf,
                     bool transfer_ownership = false);

   /// Set viscosity from a shared QuadratureFunction.
   void SetViscosity(std::shared_ptr<QuadratureFunction> qf);

   /// Set viscosity from a referenced ParGridFunction.
   void SetViscosity(ParGridFunction &gf,
                     bool transfer_ownership = false);

   /// Set viscosity from a shared ParGridFunction.
   void SetViscosity(std::shared_ptr<ParGridFunction> gf);

   /// Set Brinkman penalization to a constant value.
   void SetBrinkmanPenalization(real_t value);

   /// Set Brinkman penalization from a referenced coefficient.
   void SetBrinkmanPenalization(Coefficient &coefficient,
                                bool transfer_ownership = false);

   /// Set Brinkman penalization from a shared coefficient.
   void SetBrinkmanPenalization(std::shared_ptr<Coefficient> coefficient);

   /// Set Brinkman penalization from a referenced QuadratureFunction.
   void SetBrinkmanPenalization(QuadratureFunction &qf,
                                bool transfer_ownership = false);

   /// Set Brinkman penalization from a shared QuadratureFunction.
   void SetBrinkmanPenalization(std::shared_ptr<QuadratureFunction> qf);

   /// Set Brinkman penalization from a referenced ParGridFunction.
   void SetBrinkmanPenalization(ParGridFunction &gf,
                                bool transfer_ownership = false);

   /// Set Brinkman penalization from a shared ParGridFunction.
   void SetBrinkmanPenalization(std::shared_ptr<ParGridFunction> gf);

   /// Remove the Brinkman penalization term from the velocity operator.
   void ClearBrinkmanPenalization();

   /// Access acceleration/load coefficients keyed by element attribute.
   VectorAttributeCoefficientMap &Acceleration() { return acceleration_; }

   /// Access velocity Dirichlet data keyed by boundary attribute.
   VectorAttributeCoefficientMap &VelocityBoundary()
   { return velocity_boundary_; }

   /// Access pressure Dirichlet data keyed by boundary attribute.
   ScalarAttributeCoefficientMap &PressureBoundary()
   { return pressure_boundary_; }

   /// Mark a boundary attribute as an essential velocity boundary.
   void AddVelocityBoundaryID(int id);

   /// Mark a boundary attribute as an essential pressure boundary.
   void AddPressureBoundaryID(int id);

   /// Clear all velocity essential boundary IDs and velocity boundary data.
   void ClearVelocityBoundaryConditions();

   /// Clear all pressure essential boundary IDs and pressure boundary data.
   void ClearPressureBoundaryConditions();

   /// Select the outer coupled Krylov solver.
   void SetSolverType(KrylovSolver solver_type);

   /// Return the selected outer coupled Krylov solver.
   KrylovSolver GetSolverType() const { return solver_type_; }

   /// Select the velocity-block preconditioner strategy.
   void SetVelocityPreconditionerType(VelocityPreconditioner prec_type);

   /// Return the selected velocity-block preconditioner strategy.
   VelocityPreconditioner GetVelocityPreconditionerType() const
   { return velocity_prec_type_; }

   /// Enable or disable elasticity near-nullspace vectors in velocity AMG.
   void SetVelocityAMGElasticityNearNullspace(bool use_near_nullspace);

   /// Return whether elasticity near-nullspace vectors are used in velocity AMG.
   bool GetVelocityAMGElasticityNearNullspace() const
   { return velocity_amg_elasticity_near_nullspace_; }

   /// Select the pressure-block preconditioner strategy.
   void SetPressurePreconditionerType(PressurePreconditioner prec_type);

   /// Return the selected pressure-block preconditioner strategy.
   PressurePreconditioner GetPressurePreconditionerType() const
   { return pressure_prec_type_; }

   /// Select the Krylov method for the CC pressure diffusion solve.
   void SetCCDiffusionSolverType(CCDiffusionSolver solver_type);

   /// Return the selected Krylov method for the CC pressure diffusion solve.
   CCDiffusionSolver GetCCDiffusionSolverType() const
   { return cc_diffusion_solver_type_; }

   /// Select the velocity operator used by LSC preconditioning.
   void SetLSCVelocityOperatorType(LSCVelocityOperator operator_type);

   /// Return the selected velocity operator used by LSC preconditioning.
   LSCVelocityOperator GetLSCVelocityOperatorType() const
   { return lsc_velocity_operator_type_; }

   /// Select the velocity operator used to extract the LSC diagonal.
   void SetLSCDiagonalOperatorType(LSCDiagonalOperator operator_type);

   /// Return the selected velocity operator used to extract the LSC diagonal.
   LSCDiagonalOperator GetLSCDiagonalOperatorType() const
   { return lsc_diagonal_operator_type_; }

   /// Select the preconditioner used by the inner LSC Q solve.
   void SetLSCQPreconditionerType(LSCQPreconditioner prec_type);

   /// Return the selected preconditioner used by the inner LSC Q solve.
   LSCQPreconditioner GetLSCQPreconditionerType() const
   { return lsc_q_preconditioner_type_; }

   /// Set the outer Krylov relative tolerance.
   void SetRelTol(real_t rel_tol);

   /// Set the outer Krylov absolute tolerance.
   void SetAbsTol(real_t abs_tol);

   /// Set the outer Krylov maximum iteration count.
   void SetMaxIter(int max_iter);

   /// Set the relative tolerance for both block preconditioner CG solvers.
   void SetPreconditionerCGRelTol(real_t rel_tol);

   /// Set the absolute tolerance for both block preconditioner CG solvers.
   void SetPreconditionerCGAbsTol(real_t abs_tol);

   /// Set the maximum iteration count for both block preconditioner CG solvers.
   void SetPreconditionerCGMaxIter(int max_iter);

   /// Set the velocity-block preconditioner CG relative tolerance.
   void SetVelocityPreconditionerCGRelTol(real_t rel_tol);

   /// Set the velocity-block preconditioner CG absolute tolerance.
   void SetVelocityPreconditionerCGAbsTol(real_t abs_tol);

   /// Set the velocity-block preconditioner CG maximum iteration count.
   void SetVelocityPreconditionerCGMaxIter(int max_iter);

   /// Set the pressure-block preconditioner CG relative tolerance.
   void SetPressurePreconditionerCGRelTol(real_t rel_tol);

   /// Set the pressure-block preconditioner CG absolute tolerance.
   void SetPressurePreconditionerCGAbsTol(real_t abs_tol);

   /// Set the pressure-block preconditioner CG maximum iteration count.
   void SetPressurePreconditionerCGMaxIter(int max_iter);

   /// Set the GMRES restart dimension.
   void SetKDim(int kdim);

   /// Set the print level for Krylov solvers and AMG preconditioners.
   void SetPrintLevel(int print_level);

   /// Return the outer Krylov relative tolerance.
   real_t GetRelTol() const { return rel_tol_; }

   /// Return the outer Krylov absolute tolerance.
   real_t GetAbsTol() const { return abs_tol_; }

   /// Return the outer Krylov maximum iteration count.
   int GetMaxIter() const { return max_iter_; }

   /// Return the velocity-block preconditioner CG relative tolerance.
   real_t GetVelocityPreconditionerCGRelTol() const
   { return velocity_pc_cg_rel_tol_; }

   /// Return the velocity-block preconditioner CG absolute tolerance.
   real_t GetVelocityPreconditionerCGAbsTol() const
   { return velocity_pc_cg_abs_tol_; }

   /// Return the velocity-block preconditioner CG maximum iteration count.
   int GetVelocityPreconditionerCGMaxIter() const
   { return velocity_pc_cg_max_iter_; }

   /// Return the pressure-block preconditioner CG relative tolerance.
   real_t GetPressurePreconditionerCGRelTol() const
   { return pressure_pc_cg_rel_tol_; }

   /// Return the pressure-block preconditioner CG absolute tolerance.
   real_t GetPressurePreconditionerCGAbsTol() const
   { return pressure_pc_cg_abs_tol_; }

   /// Return the pressure-block preconditioner CG maximum iteration count.
   int GetPressurePreconditionerCGMaxIter() const
   { return pressure_pc_cg_max_iter_; }

   /// Return the GMRES restart dimension.
   int GetKDim() const { return kdim_; }

   /// Return the solver print level.
   int GetPrintLevel() const { return print_level_; }

   /// Return true if lazy assembly is required before the next solve.
   bool NeedsAssembly() const { return needs_assembly_; }

   /// Force or clear the lazy assembly flag.
   void SetNeedsAssembly(bool needs_assembly = true) const;

   /// Return the domain integration rule for a geometry type.
   const IntegrationRule &GetIntegrationRule(Geometry::Type geom) const;

   /// Assemble the coupled operator, preconditioner, and Krylov solver if dirty.
   void Assemble() const;

   /// Solve the coupled system for a contiguous true-dof vector [u, p].
   void Mult(const Vector &rhs, Vector &solution) const override;

   /// Solve the coupled system for a block RHS [velocity, pressure].
   void Mult(const BlockVector &rhs, BlockVector &solution) const;

   /// Solve with an already assembled operator and preconditioner.
   void MultAssembled(const BlockVector &rhs, BlockVector &solution) const;

   /// Apply the adjoint solve with homogeneous essential boundary values.
   void MultTranspose(const Vector &rhs, Vector &solution) const override;

   /// Assemble stored acceleration and boundary data, then solve for [u, p].
   void Solve(BlockVector &solution) const;

   /// IterativeSolver compatibility check for an externally supplied operator.
   void SetOperator(const Operator &op) override;

   /// Return the velocity finite element space.
   ParFiniteElementSpace &GetVelocityFESpace() { return velocity_space_; }

   /// Return the velocity finite element space.
   const ParFiniteElementSpace &GetVelocityFESpace() const
   { return velocity_space_; }

   /// Return the pressure finite element space.
   ParFiniteElementSpace &GetPressureFESpace() { return pressure_space_; }

   /// Return the pressure finite element space.
   const ParFiniteElementSpace &GetPressureFESpace() const
   { return pressure_space_; }

   /// Return the true-dof block offsets [0, velocity_size, total_size].
   const Array<int> &GetBlockOffsets() const { return block_offsets_; }

   /// Return the assembled coupled operator, assembling first if needed.
   const Operator *GetOperator() const;

   /// Return the selected block preconditioner, assembling first if needed.
   const Solver *GetPreconditioner() const;

private:
   /// Allow the vector coefficient map to mark solver state dirty.
   friend class VectorAttributeCoefficientMap;

   /// Allow the scalar coefficient map to mark solver state dirty.
   friend class ScalarAttributeCoefficientMap;

   /// Wrap a scalar coefficient reference in a shared pointer.
   std::shared_ptr<Coefficient> ShareCoefficient(Coefficient &coefficient,
                                                 bool transfer_ownership);

   /// Wrap a quadrature-function reference in a shared pointer.
   std::shared_ptr<QuadratureFunction> ShareQuadratureFunction(
      QuadratureFunction &qf, bool transfer_ownership);

   /// Wrap a grid-function reference in a shared pointer.
   std::shared_ptr<ParGridFunction> ShareParGridFunction(
      ParGridFunction &gf, bool transfer_ownership);

   /// Convert a scalar QuadratureFunction to a coefficient wrapper.
   std::shared_ptr<Coefficient> MakeQuadratureCoefficient(
      std::shared_ptr<QuadratureFunction> qf) const;

   /// Convert a scalar ParGridFunction to a coefficient wrapper.
   std::shared_ptr<Coefficient> MakeGridFunctionCoefficient(
      std::shared_ptr<ParGridFunction> gf) const;

   /// Verify that a viscosity QuadratureFunction is scalar and on the mesh.
   void ValidateQuadratureFunction(const QuadratureFunction &qf) const;

   /// Verify that a viscosity ParGridFunction is scalar and on the mesh.
   void ValidateParGridFunction(const ParGridFunction &gf) const;

   /// Mark operator, preconditioner, and Krylov state dirty.
   void MarkOperatorChanged();

   /// Mark stored acceleration RHS data dirty.
   void MarkAccelerationChanged();

   /// Mark velocity boundary data dirty and optionally add an essential ID.
   void MarkVelocityBoundaryChanged(int attr);

   /// Mark pressure boundary data dirty and optionally add an essential ID.
   void MarkPressureBoundaryChanged(int attr);

   /// Build a boundary marker array from one-based boundary attribute IDs.
   void BuildMarker(const std::set<int> &ids, Array<int> &marker,
                    const Array<int> &mesh_bdr_attributes) const;

   /// Build velocity and pressure essential true-dof lists.
   void BuildEssentialTrueDofs() const;

   /// Project stored velocity/pressure boundary coefficients to true dofs.
   void BuildBoundaryValues() const;

   /// Build the partial-assembly Stokes block operator.
   void BuildOperator() const;

   /// Build the selected block preconditioner.
   void BuildPreconditioner() const;

   /// Build the velocity-block preconditioner.
   void BuildVelocityPreconditioner() const;

   /// Build the pressure-block preconditioner.
   void BuildPressurePreconditioner() const;

   /// Build or transfer viscosity data onto the LOR velocity space.
   std::shared_ptr<Coefficient> MakeLORViscosityCoefficient() const;

   /// Build or transfer Brinkman data onto the LOR velocity space.
   std::shared_ptr<Coefficient> MakeLORBrinkmanCoefficient() const;

   /// Return true if any boundary attribute is left open for velocity.
   bool HasOpenVelocityBoundary() const;

   /// Return true if pressure has no essential constraint globally.
   bool HasPressureNullspace() const;

   /// Project the pressure block of @a vector to algebraic mean zero.
   void ProjectPressureMean(Vector &vector) const;

   /// Apply constraints, solve the coupled true-dof system, and recover blocks.
   void SolveSystem(const BlockVector &rhs, BlockVector &solution,
                    bool use_boundary_values) const;

   /// Optional owner for the velocity finite element space.
   std::shared_ptr<ParFiniteElementSpace> velocity_space_owner_;

   /// Optional owner for the pressure finite element space.
   std::shared_ptr<ParFiniteElementSpace> pressure_space_owner_;

   /// Velocity finite element space supplied by the user.
   ParFiniteElementSpace &velocity_space_;

   /// Pressure finite element space supplied by the user.
   ParFiniteElementSpace &pressure_space_;

   /// Coupled true-dof block offsets [0, nv, nv+np].
   mutable Array<int> block_offsets_;

   /// Boundary attributes where velocity Dirichlet conditions are enforced.
   mutable std::set<int> velocity_boundary_ids_;

   /// Boundary attributes where pressure Dirichlet conditions are enforced.
   mutable std::set<int> pressure_boundary_ids_;

   /// Essential true dofs for the velocity block.
   mutable Array<int> velocity_ess_tdofs_;

   /// Essential true dofs for the pressure block.
   mutable Array<int> pressure_ess_tdofs_;

   /// Viscosity coefficient used in the velocity diffusion operator.
   std::shared_ptr<Coefficient> viscosity_coefficient_;

   /// Optional owner for viscosity supplied as a QuadratureFunction.
   std::shared_ptr<QuadratureFunction> viscosity_qf_owner_;

   /// Optional owner for viscosity supplied as a ParGridFunction.
   std::shared_ptr<ParGridFunction> viscosity_gf_owner_;

   /// Brinkman penalization coefficient used in the velocity mass term.
   std::shared_ptr<Coefficient> brinkman_coefficient_;

   /// Optional owner for Brinkman penalization supplied as a QuadratureFunction.
   std::shared_ptr<QuadratureFunction> brinkman_qf_owner_;

   /// Optional owner for Brinkman penalization supplied as a ParGridFunction.
   std::shared_ptr<ParGridFunction> brinkman_gf_owner_;

   /// Attribute-keyed acceleration/load coefficients.
   mutable VectorAttributeCoefficientMap acceleration_;

   /// Attribute-keyed velocity Dirichlet coefficients.
   mutable VectorAttributeCoefficientMap velocity_boundary_;

   /// Attribute-keyed pressure Dirichlet coefficients.
   mutable ScalarAttributeCoefficientMap pressure_boundary_;

   /// Domain quadrature order used by Stokes forms.
   int integration_order_;

   /// Selected outer Krylov solver.
   KrylovSolver solver_type_ = KrylovSolver::MINRES;

   /// Selected velocity-block preconditioner strategy.
   VelocityPreconditioner velocity_prec_type_ = VelocityPreconditioner::AMG;

   /// Whether velocity AMG receives elasticity near-nullspace vectors.
   bool velocity_amg_elasticity_near_nullspace_ = false;

   /// Selected pressure-block preconditioner strategy.
   PressurePreconditioner pressure_prec_type_ =
      PressurePreconditioner::DIAGONAL_MASS;

   /// Krylov method used by the CC pressure diffusion solve.
   CCDiffusionSolver cc_diffusion_solver_type_ = CCDiffusionSolver::GMRES;

   /// Velocity operator used in the LSC H block.
   LSCVelocityOperator lsc_velocity_operator_type_ =
      LSCVelocityOperator::ASSEMBLED;

   /// Velocity operator used to extract the LSC velocity diagonal.
   LSCDiagonalOperator lsc_diagonal_operator_type_ =
      LSCDiagonalOperator::MATCH_VELOCITY;

   /// Preconditioner used by the inner LSC Q solve.
   LSCQPreconditioner lsc_q_preconditioner_type_ =
      LSCQPreconditioner::OPERATOR_JACOBI;

   /// Outer Krylov relative tolerance.
   real_t rel_tol_ = 1.0e-10;

   /// Outer Krylov absolute tolerance.
   real_t abs_tol_ = 0.0;

   /// Outer Krylov maximum iteration count.
   int max_iter_ = 500;

   /// Velocity preconditioner CG relative tolerance.
   real_t velocity_pc_cg_rel_tol_ = 1.0e-8;

   /// Velocity preconditioner CG absolute tolerance.
   real_t velocity_pc_cg_abs_tol_ = 0.0;

   /// Velocity preconditioner CG maximum iteration count.
   int velocity_pc_cg_max_iter_ = 50;

   /// Pressure preconditioner CG relative tolerance.
   real_t pressure_pc_cg_rel_tol_ = 1.0e-8;

   /// Pressure preconditioner CG absolute tolerance.
   real_t pressure_pc_cg_abs_tol_ = 0.0;

   /// Pressure preconditioner CG maximum iteration count.
   int pressure_pc_cg_max_iter_ = 50;

   /// GMRES restart dimension.
   int kdim_ = 50;

   /// Print level for Krylov solvers and AMG.
   int print_level_ = -1;

   /// True when operator/preconditioner/Krylov setup must be rebuilt.
   mutable bool needs_assembly_ = true;

   /// True when the stored acceleration RHS form must be rebuilt.
   mutable bool rhs_dirty_ = true;

   /// True when cached boundary true-dof values must be rebuilt.
   mutable bool boundary_values_dirty_ = true;

   /// Partial-assembly velocity diffusion bilinear form.
   mutable std::unique_ptr<ParBilinearForm> velocity_form_;

   /// True-dof velocity diffusion operator.
   mutable OperatorHandle velocity_operator_;

   /// Partial-assembly mixed divergence bilinear form.
   mutable std::unique_ptr<ParMixedBilinearForm> divergence_form_;

   /// True-dof divergence operator mapping velocity to pressure.
   mutable OperatorHandle divergence_operator_;

   /// Transpose operator representing the pressure gradient block.
   mutable std::unique_ptr<TransposeOperator> gradient_operator_;

   /// Scaled divergence operator used in block triangular GMRES preconditioning.
   mutable std::unique_ptr<ScaledOperator> negative_divergence_operator_;

   /// Coupled 2x2 Stokes block operator.
   mutable std::unique_ptr<BlockOperator> block_operator_;

   /// Partial-assembly pressure mass form used for pressure preconditioning.
   mutable std::unique_ptr<ParBilinearForm> pressure_mass_form_;

   /// True-dof pressure mass operator.
   mutable OperatorHandle pressure_mass_operator_;

   /// Reciprocal-viscosity pressure mass coefficient.
   mutable std::shared_ptr<Coefficient> pressure_mass_coefficient_;

   /// Diagonal smoother/inverse for the pressure mass operator.
   mutable std::unique_ptr<OperatorJacobiSmoother> pressure_mass_jacobi_;

   /// Assembled or LOR velocity preconditioner bilinear form.
   mutable std::unique_ptr<ParBilinearForm> velocity_pc_form_;

   /// True-dof operator used by the velocity preconditioner.
   mutable OperatorHandle velocity_pc_operator_;

   /// Diagonal smoother for optional velocity-block CG preconditioning.
   mutable std::unique_ptr<OperatorJacobiSmoother> velocity_pc_jacobi_;

   /// Low-order refined mesh used for high-order velocity AMG.
   mutable std::unique_ptr<ParMesh> lor_mesh_;

   /// Order-one finite element collection used on the LOR mesh.
   mutable std::unique_ptr<FiniteElementCollection> lor_fec_;

   /// Vector H1 velocity space on the LOR mesh.
   mutable std::unique_ptr<ParFiniteElementSpace> lor_velocity_space_;

   /// Scalar H1 space on the LOR mesh for coefficient transfer.
   mutable std::unique_ptr<ParFiniteElementSpace> lor_scalar_space_;

   /// LOR grid-function storage for transferred viscosity data.
   mutable std::shared_ptr<ParGridFunction> lor_viscosity_gf_;

   /// Coefficient wrapper around LOR viscosity data.
   mutable std::shared_ptr<Coefficient> lor_viscosity_coefficient_;

   /// LOR grid-function storage for transferred Brinkman data.
   mutable std::shared_ptr<ParGridFunction> lor_brinkman_gf_;

   /// Coefficient wrapper around LOR Brinkman data.
   mutable std::shared_ptr<Coefficient> lor_brinkman_coefficient_;

   /// Essential true dofs for the velocity preconditioner space.
   mutable Array<int> velocity_pc_ess_tdofs_;

   /// Solver/preconditioner applied to the velocity block.
   mutable std::unique_ptr<Solver> velocity_preconditioner_;

   /// Solver/preconditioner applied to the pressure block.
   mutable std::unique_ptr<Solver> pressure_preconditioner_;

   /// Reciprocal-Brinkman pressure diffusion coefficient for CC preconditioning.
   mutable std::shared_ptr<Coefficient> cc_diffusion_coefficient_;

   /// Diffusion solve contribution in the Cahouet-Chabard pressure PC.
   mutable std::unique_ptr<DiffusionSolver> cc_diffusion_solver_;

   /// Optional GMRES wrapper for the CC pressure diffusion solve.
   mutable std::unique_ptr<IterativeSolver> cc_diffusion_krylov_solver_;

   /// Weighted mass solve contribution in the Cahouet-Chabard pressure PC.
   mutable std::unique_ptr<CGSolver> cc_mass_solver_;

   /// Sum of pressure mass inverse and diffusion solve for CC preconditioning.
   mutable std::unique_ptr<Solver> cc_pressure_preconditioner_;

   /// Inverse velocity-block diagonal used by algebraic LSC.
   mutable Vector lsc_velocity_diag_inverse_;

   /// Diagonal of the matrix-free LSC Q operator.
   mutable Vector lsc_q_diag_;

   /// Matrix-free LSC Q operator B diag(A)^{-1} B^T.
   mutable std::unique_ptr<Operator> lsc_q_operator_;

   /// LSC H operator B diag(A)^{-1} A diag(A)^{-1} B^T.
   mutable std::unique_ptr<Operator> lsc_h_operator_;

   /// Transposed divergence matrix scaled by diag(A)^{-1} for LSC Q.
   mutable std::unique_ptr<HypreParMatrix> lsc_scaled_divergence_transpose_;

   /// Assembled LSC Q matrix B diag(A)^{-1} B^T.
   mutable std::unique_ptr<HypreParMatrix> lsc_q_matrix_;

   /// Fully assembled divergence form used by LSC pressure operators.
   mutable std::unique_ptr<ParMixedBilinearForm> lsc_divergence_form_;

   /// Fully assembled true-dof divergence matrix used by LSC.
   mutable OperatorHandle lsc_divergence_operator_;

   /// Fully assembled velocity form used by LSC pressure operators.
   mutable std::unique_ptr<ParBilinearForm> lsc_velocity_form_;

   /// Fully assembled true-dof velocity matrix used by LSC.
   mutable OperatorHandle lsc_velocity_operator_;

   /// AMG preconditioner for the inner LSC Q solve.
   mutable std::unique_ptr<HypreBoomerAMG> lsc_q_amg_;

   /// Jacobi preconditioner for the inner LSC Q solve.
   mutable std::unique_ptr<HypreDiagScale> lsc_q_jacobi_;

   /// Operator Jacobi preconditioner for the matrix-free inner LSC Q solve.
   mutable std::unique_ptr<OperatorJacobiSmoother> lsc_q_operator_jacobi_;

   /// Inner solve with the LSC Q operator.
   mutable std::unique_ptr<CGSolver> lsc_q_solver_;

   /// LSC pressure preconditioner Q^{-1} H Q^{-1}.
   mutable std::unique_ptr<Solver> lsc_pressure_preconditioner_;

   /// Sign-corrected pressure preconditioner used by GMRES block triangular PC.
   mutable std::unique_ptr<ScaledOperator> negative_pressure_preconditioner_;

   /// Block diagonal preconditioner used by MINRES.
   mutable std::unique_ptr<BlockDiagonalPreconditioner> diagonal_prec_;

   /// Block lower triangular preconditioner used by GMRES.
   mutable std::unique_ptr<BlockLowerTriangularPreconditioner> triangular_prec_;

   /// Pressure-mean projected view of the coupled operator when needed.
   mutable std::unique_ptr<Operator> projected_operator_;

   /// Pressure-mean projected view of the selected preconditioner when needed.
   mutable std::unique_ptr<Solver> projected_preconditioner_;

   /// Configured outer coupled Krylov solver.
   mutable std::unique_ptr<IterativeSolver> iterative_solver_;

   /// Linear form used to assemble acceleration/load contributions.
   mutable std::unique_ptr<ParLinearForm> acceleration_form_;

   /// True-dof velocity RHS assembled from acceleration.
   mutable Vector rhs_velocity_;

   /// True-dof pressure RHS; currently zero for incompressibility.
   mutable Vector rhs_pressure_;

   /// Cached essential velocity true-dof boundary values.
   mutable Vector velocity_boundary_true_;

   /// Cached essential pressure true-dof boundary values.
   mutable Vector pressure_boundary_true_;

   /// Work vector for transferring high-order viscosity data to LOR space.
   mutable Vector lor_transfer_true_dofs_;

   /// Coupled RHS block assembled by Solve().
   mutable BlockVector rhs_block_;

   /// Coupled RHS block after applying essential constraints.
   mutable BlockVector constrained_rhs_;

   /// Coupled block of cached boundary true-dof values.
   mutable BlockVector boundary_block_;

   /// All velocity true dofs from the projected velocity boundary grid function.
   mutable Vector velocity_boundary_all_;

   /// All pressure true dofs from the projected pressure boundary grid function.
   mutable Vector pressure_boundary_all_;

   /// Boundary marker for velocity essential attributes.
   mutable Array<int> velocity_marker_;

   /// Boundary marker for pressure essential attributes.
   mutable Array<int> pressure_marker_;

   /// Grid function used to project velocity boundary coefficients.
   mutable std::unique_ptr<ParGridFunction> velocity_boundary_gf_;

   /// Grid function used to project pressure boundary coefficients.
   mutable std::unique_ptr<ParGridFunction> pressure_boundary_gf_;
};

/// Named Brinkman-penalized Stokes solver.
///
/// This class uses the StokesSolver implementation with a Brinkman
/// penalization coefficient supplied through SetBrinkmanPenalization().  It is
/// provided as a distinct type for users who want to request the Brinkman
/// formulation explicitly.
class BrinkmanStokesSolver : public StokesSolver
{
public:
   /// Inherit the StokesSolver constructors.
   using StokesSolver::StokesSolver;
};

/// Balakrishnan/sinc-quadrature inverse fractional diffusion solver.
///
/// This solver applies the inverse fractional power of the generalized
/// diffusion+mass operator
///
///     L = M^{-1} (K + beta M),
///
/// where K is the diffusion bilinear form and M is the mass bilinear form on a
/// user-provided ParFiniteElementSpace.  The default beta is zero, giving the
/// diffusion-only generalized operator M^{-1}K.  For 0 < s < 1,
///
///     L^{-s} f = sin(pi s)/pi int_{-infty}^{infty}
///                sigma^(1-s) exp((1-s)y)
///                (L + sigma exp(y) I)^{-1} f dy.
///
/// In weak form each quadrature point solves
///
///     (K + (beta + sigma exp(y)) M) u_y = b,
///
/// where b is the assembled RHS true vector supplied to Mult().
///
/// The integral is approximated by the exponential/sinc trapezoidal rule on the
/// infinite interval with points y_l = l*k, l = -m, ..., n.  The scaling
/// sigma defaults to one and can be selected directly or estimated from
/// sqrt(lambda_min lambda_max).  Mult() accepts an assembled RHS true vector b
/// in the dual/weak space and returns the primal true vector approximating
/// L^{-s} M^{-1} b.  The internally owned DiffusionMassSolver is exposed so
/// users can set diffusion/mass coefficients, boundary conditions, and print
/// options.
class BalakrishnanFractionalSolver : public Solver
{
public:
   /// Construct the fractional solver on @a fespace without taking ownership.
   explicit BalakrishnanFractionalSolver(ParFiniteElementSpace &fespace);

   /// Construct the fractional solver and keep shared ownership of @a fespace.
   explicit BalakrishnanFractionalSolver(
      std::shared_ptr<ParFiniteElementSpace> fespace);

   /// Set the fractional exponent s in L^{-s}; requires 0 < s < 1.
   void SetFractionalPower(real_t s);

   /// Return the fractional exponent s.
   real_t GetFractionalPower() const { return fractional_power_; }

   /// Set sinc quadrature spacing k and truncation l = -m, ..., n.
   void SetQuadrature(real_t spacing, int m, int n);

   /// Enable or disable adaptive sinc quadrature.  When enabled, Mult()
   /// extends the negative and positive tails independently until consecutive
   /// terms satisfy ||term|| <= abs_tol + rel_tol ||sum||.
   void UseAdaptiveQuadrature(bool use_adaptive = true);

   /// Configure adaptive sinc quadrature tolerances and truncation caps.
   void SetAdaptiveQuadrature(real_t rel_tol, real_t abs_tol,
                              int max_negative_points,
                              int max_positive_points,
                              int consecutive_terms = 3);

   /// Set the positive scaling parameter sigma in t = sigma exp(y).
   /// The default is one.  The shifted systems become
   ///     (K + sigma exp(y_l) M) u_l = b
   /// and the quadrature weights include sigma^(1-s).
   void SetQuadratureScaling(real_t scaling);

   /// Set the non-fractional mass shift beta in the generalized operator
   /// L = M^{-1}(K + beta M).  The shifted quadrature systems become
   ///     (K + (beta + sigma exp(y_l)) M) u_l = b.
   /// The default beta is zero, which preserves the original diffusion-only
   /// generalized operator M^{-1}K.
   void SetOperatorMassShift(real_t mass_shift);

   /// Return the non-fractional mass shift beta in L=M^{-1}(K+beta M).
   real_t GetOperatorMassShift() const { return operator_mass_shift_; }

   /// Return the sinc quadrature spacing k.
   real_t GetQuadratureSpacing() const { return quadrature_spacing_; }

   /// Return the exponential quadrature scaling parameter sigma.
   real_t GetQuadratureScaling() const { return quadrature_scaling_; }

   /// Return the number of negative quadrature points m.
   int GetNegativeQuadraturePoints() const { return negative_points_; }

   /// Return the number of positive quadrature points n.
   int GetPositiveQuadraturePoints() const { return positive_points_; }

   /// Return true if Mult() uses adaptive quadrature.
   bool UsesAdaptiveQuadrature() const { return use_adaptive_quadrature_; }

   /// Return the number of negative points used by the last Mult().
   int GetLastNegativeQuadraturePoints() const
   { return last_negative_points_; }

   /// Return the number of positive points used by the last Mult().
   int GetLastPositiveQuadraturePoints() const
   { return last_positive_points_; }

   /// Access the shifted diffusion+mass solver used for every quadrature point.
   DiffusionMassSolver &GetDiffusionMassSolver() { return shifted_solver_; }

   /// Access the shifted diffusion+mass solver used for every quadrature point.
   const DiffusionMassSolver &GetDiffusionMassSolver() const
   { return shifted_solver_; }

   /// Access a weighted mass map users may use to build weak RHS vectors.
   TrueMassMapOperator &GetMassMap() { return mass_map_; }

   /// Access a weighted mass map users may use to build weak RHS vectors.
   const TrueMassMapOperator &GetMassMap() const { return mass_map_; }

   /// Estimate extremal eigenvalues of L = M^{-1}(K + beta M).
   ///
   /// The maximum eigenvalue is estimated by power iteration on L.  The minimum
   /// eigenvalue is estimated by inverse power iteration, applying L^{-1}
   /// through diffusion-only solves.  The method assumes that the configured
   /// diffusion operator is positive definite after the currently configured
   /// boundary conditions.  It also returns the suggested exponential
   /// quadrature scaling sqrt(lambda_min*lambda_max).
   void EstimateEigenvalueBounds(int power_iterations,
                                 int inverse_power_iterations,
                                 real_t &lambda_min,
                                 real_t &lambda_max,
                                 real_t &suggested_scaling) const;

   /// Apply the sinc approximation of L^{-s}.
   void Mult(const Vector &input, Vector &output) const override;

   /// Apply the same symmetric fractional action as Mult().
   void MultTranspose(const Vector &input, Vector &output) const override;

   /// IterativeSolver compatibility: validate dimensions and keep the
   /// internally owned shifted operators unchanged.
   void SetOperator(const Operator &op) override;

private:
   /// Optional shared owner for fespace_.
   std::shared_ptr<ParFiniteElementSpace> fespace_owner_;

   /// Finite element space defining input and output true vectors.
   ParFiniteElementSpace &fespace_;

   /// Fractional exponent s in L^{-s}.
   real_t fractional_power_ = 0.5;

   /// Sinc/trapezoidal spacing k on the y interval.
   real_t quadrature_spacing_ = 0.25;

   /// Number of negative indices in l = -m, ..., n.
   int negative_points_ = 16;

   /// Number of positive indices in l = -m, ..., n.
   int positive_points_ = 16;

   /// Positive scaling parameter sigma in t = sigma exp(y).
   real_t quadrature_scaling_ = 1.0;

   /// Non-fractional mass shift beta in L = M^{-1}(K + beta M).
   real_t operator_mass_shift_ = 0.0;

   /// True when Mult() grows the quadrature interval adaptively.
   bool use_adaptive_quadrature_ = false;

   /// Relative tolerance for adaptive quadrature tails.
   real_t adaptive_rel_tol_ = 1.0e-8;

   /// Absolute tolerance for adaptive quadrature tails.
   real_t adaptive_abs_tol_ = 0.0;

   /// Maximum number of negative tail points in adaptive mode.
   int adaptive_max_negative_points_ = 200;

   /// Maximum number of positive tail points in adaptive mode.
   int adaptive_max_positive_points_ = 200;

   /// Consecutive small tail terms required before stopping a side.
   int adaptive_consecutive_terms_ = 3;

   /// Negative points used by the last Mult().
   mutable int last_negative_points_ = 0;

   /// Positive points used by the last Mult().
   mutable int last_positive_points_ = 0;

   /// Apply the base mass matrix M.
   void ApplyMass(const Vector &input, Vector &output) const;

   /// Apply the base diffusion matrix K.
   void ApplyDiffusion(const Vector &input, Vector &output) const;

   /// Apply L = M^{-1}(K + beta M).
   void ApplyGeneralizedOperator(const Vector &input, Vector &output) const;

   /// Apply L^{-1} = (K + beta M)^{-1}M.
   void ApplyInverseGeneralizedOperator(const Vector &input,
                                        Vector &output) const;

   /// Compute (x,(K+beta M)x)/(x,Mx).
   real_t RayleighQuotient(const Vector &input) const;

   /// Normalize a true vector in parallel Euclidean norm.
   void Normalize(Vector &x) const;

   /// Add one quadrature term at index @a ell and return its Euclidean norm.
   real_t AddQuadratureTerm(int ell, const Vector &mass_rhs,
                            Vector &output) const;

   /// Update mass_map_ only if the internal solver's base mass coefficient
   /// pointer has changed.
   void SyncMassMapCoefficient() const;

   /// Weighted mass map available for building RHS vectors from primal fields.
   mutable TrueMassMapOperator mass_map_;

   /// Base mass coefficient currently installed in mass_map_.
   mutable const Coefficient *mass_map_coefficient_ = nullptr;

   /// Shifted solver for (K + (beta + exp(y)) M) u_y = b.
   mutable DiffusionMassSolver shifted_solver_;

   /// Cached shifted solution u_y.
   mutable Vector shifted_solution_;

   /// Work vector for eigenvalue estimates.
   mutable Vector eigen_work_1_;

   /// Work vector for eigenvalue estimates.
   mutable Vector eigen_work_2_;
};

/// Fractional inverse diffusion solver for the spectral FE diffusion operator.
///
/// This class applies the inverse fractional power of the pure diffusion
/// operator
///
///     L = M_0^{-1} K,
///
/// where K is the diffusion stiffness matrix and M_0 is the standard FE mass
/// matrix used only as the Riesz map/identity for the generalized eigenproblem.
/// This is the usual finite element spectral fractional diffusion operator; the
/// mass/Riesz map is required and is not a reaction term.  Mult() accepts an
/// assembled RHS true vector b and returns the primal true vector approximating
/// L^{-s} M_0^{-1} b.  Diffusion coefficients and boundary conditions are set
/// through GetDiffusionSolver().
class FractionalDiffusionSolver : public Solver
{
public:
   /// Construct the fractional diffusion solver without taking ownership.
   explicit FractionalDiffusionSolver(ParFiniteElementSpace &fespace);

   /// Construct the fractional diffusion solver and keep shared FE ownership.
   explicit FractionalDiffusionSolver(
      std::shared_ptr<ParFiniteElementSpace> fespace);

   /// Set the fractional exponent s in L^{-s}; requires 0 < s < 1.
   void SetFractionalPower(real_t s);

   /// Return the fractional exponent s.
   real_t GetFractionalPower() const
   { return balakrishnan_solver_.GetFractionalPower(); }

   /// Set fixed sinc quadrature spacing and truncation l=-m,...,n.
   void SetQuadrature(real_t spacing, int m, int n);

   /// Enable or disable adaptive sinc quadrature.
   void UseAdaptiveQuadrature(bool use_adaptive = true);

   /// Configure adaptive sinc quadrature tolerances and caps.
   void SetAdaptiveQuadrature(real_t rel_tol, real_t abs_tol,
                              int max_negative_points,
                              int max_positive_points,
                              int consecutive_terms = 3);

   /// Set the exponential scaling sigma in t=sigma exp(y).
   void SetQuadratureScaling(real_t scaling);

   /// Estimate eigenvalue bounds and suggested scaling for L=M_0^{-1}K.
   void EstimateEigenvalueBounds(int power_iterations,
                                 int inverse_power_iterations,
                                 real_t &lambda_min,
                                 real_t &lambda_max,
                                 real_t &suggested_scaling) const;

   /// Access the internal shifted diffusion solver for diffusion coefficients,
   /// Dirichlet boundary IDs/data, and print options.  Do not set its mass
   /// coefficient to zero; this class owns the unit Riesz map mass coefficient.
   DiffusionMassSolver &GetDiffusionSolver()
   { return balakrishnan_solver_.GetDiffusionMassSolver(); }

   /// Access the internal shifted diffusion solver.
   const DiffusionMassSolver &GetDiffusionSolver() const
   { return balakrishnan_solver_.GetDiffusionMassSolver(); }

   /// Access the standard mass map users can use to assemble b=M_0 f.
   TrueMassMapOperator &GetMassMap()
   { return balakrishnan_solver_.GetMassMap(); }

   /// Access the standard mass map users can use to assemble b=M_0 f.
   const TrueMassMapOperator &GetMassMap() const
   { return balakrishnan_solver_.GetMassMap(); }

   /// Return true if adaptive quadrature is enabled.
   bool UsesAdaptiveQuadrature() const
   { return balakrishnan_solver_.UsesAdaptiveQuadrature(); }

   /// Return negative points used by the last Mult().
   int GetLastNegativeQuadraturePoints() const
   { return balakrishnan_solver_.GetLastNegativeQuadraturePoints(); }

   /// Return positive points used by the last Mult().
   int GetLastPositiveQuadraturePoints() const
   { return balakrishnan_solver_.GetLastPositiveQuadraturePoints(); }

   /// Apply the inverse fractional diffusion operator to an assembled RHS.
   void Mult(const Vector &rhs, Vector &solution) const override;

   /// Apply the same symmetric action as Mult().
   void MultTranspose(const Vector &rhs, Vector &solution) const override;

   /// Solver compatibility: validate dimensions and keep internal operators.
   void SetOperator(const Operator &op) override;

private:
   /// Restore the unit mass/Riesz coefficient only if a user replaced it.
   void EnsureUnitMassCoefficient() const;

   /// Shared unit coefficient used for the FE mass/Riesz map in M_0^{-1}K.
   std::shared_ptr<Coefficient> unit_mass_coefficient_;

   /// Internal Balakrishnan solver with unit mass/Riesz coefficient.
   mutable BalakrishnanFractionalSolver balakrishnan_solver_;
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
