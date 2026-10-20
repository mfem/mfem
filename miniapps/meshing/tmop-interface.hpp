// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TMOP_INTERFACE_HPP
#define MFEM_TMOP_INTERFACE_HPP

#include "mfem.hpp"
#include <algorithm>
#include <map>
#include <memory>
#include <vector>

namespace mfem
{

/// Remapping policy for fields that follow the mesh as it moves.
enum class TMOPFieldTransfer
{
   Advection,    ///< AdvectorCG; no extra third-party dependency.
   Interpolation ///< InterpolatorFP; requires MFEM_USE_GSLIB.
};

/// Linear solver used inside each Newton iteration.
enum class TMOPLinearSolver { MINRES };

/// Preconditioner for the linear solver.
enum class TMOPPreconditioner
{
   None,
   Jacobi,  ///< Jacobi smoother; supports partial assembly.
   L1Jacobi ///< l1-Jacobi smoother; no partial assembly.
};

/// Outer nonlinear solver.
enum class TMOPNonlinearSolver { Newton, LBFGS };

/// Eisenstat-Walker adaptive linear tolerance selection.
enum class TMOPAdaptiveTol { None, EisenstatWalker1, EisenstatWalker2 };

/** Scaling of the objective by its value on the initial mesh. Each
    integrator's limiting term is scaled by 1 / (its total target volume),
    or by 1 / (the number of elements) when the targets carry no volume or
    the integrator does not integrate over the target.

    CombinedInitialMetric scales every metric term by 1 / (the sum of all
    integrators' initial metric energies).
    PerIntegratorInitialMetric scales each metric term by 1 / (its own
    initial metric energy). With one integrator the two are the same. */
enum class TMOPNormalization
{
   None,
   CombinedInitialMetric,
   PerIntegratorInitialMetric
};

/// Which kind of boundary constraint to apply.
enum class TMOPBdrConstraint
{
   Free,                       ///< No essential degrees of freedom.
   FixAll,                     ///< Every boundary node is frozen.
   FixXYZComponentsByAttribute ///< Read TMOPBoundarySpec::boundary_overrides.
};

/** Components to fix, as the values of TMOPBoundarySpec::boundary_overrides.
    They combine with bitwise or. */
enum
{
   TMOP_BDR_FREE    = 0,
   TMOP_BDR_FIX_X   = 1,
   TMOP_BDR_FIX_Y   = 2,
   TMOP_BDR_FIX_Z   = 4,
   TMOP_BDR_FIX_ALL = 7
};

/// How the boundary nodes may move.
struct TMOPBoundarySpec
{
   TMOPBdrConstraint constraint =
      TMOPBdrConstraint::FixXYZComponentsByAttribute;

   /** Boundary attribute -> components to fix. Read only when @a constraint
       is FixXYZComponentsByAttribute; attributes without an entry are free.
       The default is : 1/2/3 fix x/y/z and dim+1 fixes everything.*/
   std::map<int, int> boundary_overrides =
   {
      {1, TMOP_BDR_FIX_X}, {2, TMOP_BDR_FIX_Y},
      {3, TMOP_BDR_FIX_Z}, {4, TMOP_BDR_FIX_ALL}
   };
};

/** How to build one TMOP_QualityMetric.

    Negative real-valued members request the library default, documented per
    member, so that "unset" needs no extra flag. */
struct TMOPMetricSpec
{
   using BarrierType = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType;
   using WorstCaseType = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType;

   /// Metric ID
   int id = 2;
   /// Combination weight; negative uses 0.9 for 49/126 and 0.5 otherwise.
   real_t gamma = -1.0;
   /// Replace a combination metric's built-in weights by balanced weights.
   bool balance_components = false;

   /// TMOP_WorstCaseUntangleOptimizer_Metric options
   BarrierType barrier = BarrierType::None;
   WorstCaseType worst_case = WorstCaseType::None;
   /// Exponent p of the PMean worst-case form.
   int barrier_exponent = 2;
   /// Scaling alpha of min(det(J)) in the shifted barrier.
   real_t barrier_alpha = 1.5;
   /// Shift eps subtracted from alpha*min(det(J)); also the pseudo-barrier eps.
   real_t barrier_epsilon = 0.001;
   /// Offset added to max(mu_hat) to form beta in the Beta worst-case form.
   real_t worst_case_epsilon = 0.001;

   /// User-provided external metric. When set, all parameters above are
   /// ignored.
   TMOP_QualityMetric *external = nullptr;

   /// Shape-only barrier metric: 002 in 2D, 303 in 3D.
   static TMOPMetricSpec Shape(int dim);
   /// Size-only metric: 077 in 2D, 315 in 3D.
   static TMOPMetricSpec Size(int dim);
   /// Shape+size barrier metric: 007 in 2D, 321 in 3D.
   static TMOPMetricSpec ShapeAndSize(int dim);
   /// Untangling shifted-barrier metric: 022 in 2D, 352 in 3D.
   static TMOPMetricSpec Untangler(int dim);
   /// Any metric, by its ID.
   static TMOPMetricSpec FromId(int id);
   /// Borrow a caller-constructed metric.
   static TMOPMetricSpec External(TMOP_QualityMetric &metric);
};

/// Family of target-matrix construction.
enum class TMOPTargetKind
{
   IdealShape, ///< TargetConstructor with an ideal element shape.
   Analytic, ///< AnalyticAdaptTC with a full given target matrix.
   Discrete, ///< DiscreteAdaptTC driven by nodal parameter fields.
   External, ///< A caller-owned TargetConstructor.
   SameAs    ///< The target already built for an earlier integrator.
};

/// Ideal element size convention for TMOPTargetKind::IdealShape.
enum class TMOPIdealShapeTargetSize
{
   Unit, ///< IDEAL_SHAPE_UNIT_SIZE; the only type that can untangle.
   Equal,///< IDEAL_SHAPE_EQUAL_SIZE; the global average element volume.
   Given ///< IDEAL_SHAPE_GIVEN_SIZE; the volume of the reference element.
};

/** How to build one TargetConstructor. Grid functions are borrowed and
    read when the optimizer is built.

    A plain aggregate: fill the members directly, or start from one of the
    static presets below and adjust. Members a kind does not use are
    ignored. */
struct TMOPTargetSpec
{
   TMOPTargetKind kind = TMOPTargetKind::IdealShape;

   /// Ideal-shape targets.
   TMOPIdealShapeTargetSize size_kind = TMOPIdealShapeTargetSize::Unit;
   /// Scales the average element volume of equal-size targets.
   real_t volume_scale = 1.0;
   /** Reference coordinates for equal- and given-size targets; borrowed and
       read through the pointer, so later changes to it are seen. Null uses
       the optimizer's initial nodes: those seen at construction, or at the
       latest TMOPOptimizer::Reset(). */
   const GridFunction *reference_nodes = nullptr;

   /// Analytic targets; borrowed, and must be dim x dim.
   TMOPMatrixCoefficient *analytic = nullptr;

   /** Discrete targets; at least one field is required. All fields share
       one H1 space, ordered byNODES: scalar in 2D, and in 3D a vector of
       that space for aspect ratio, skew and orientation. On a ParMesh they
       must be ParGridFunctions. Their values are copied when the optimizer
       is built. */
   const GridFunction *size = nullptr;
   const GridFunction *aspect_ratio = nullptr;
   const GridFunction *skew = nullptr;
   const GridFunction *orientation = nullptr;
   TMOPFieldTransfer transfer = TMOPFieldTransfer::Advection;
   /// Lower bound on the target size; negative leaves it unbounded.
   real_t min_size = -1.0;

   /// Caller-owned target, used as it is. When set, every member above is
   /// ignored.
   TargetConstructor *external = nullptr;

   /** Index of an earlier integrator whose target to reuse; see SameAs().
       Negative when this specification builds its own. */
   int shared_index = -1;

   /// Ideal shape, unit size: IDEAL_SHAPE_UNIT_SIZE. The only target that
   /// can untangle an inverted mesh.
   static TMOPTargetSpec IdealShapeUnitSize();
   /// Ideal shape, the average element volume times @a volume_scale:
   /// IDEAL_SHAPE_EQUAL_SIZE.
   static TMOPTargetSpec IdealShapeEqualSize(real_t volume_scale = 1.0);
   /// Ideal shape, each element's own volume: IDEAL_SHAPE_GIVEN_SIZE.
   static TMOPTargetSpec IdealShapeGivenSize();
   static TMOPTargetSpec Analytic(TMOPMatrixCoefficient &matrix);
   /// Discrete targets; then set at least one of the field pointers.
   static TMOPTargetSpec Discrete();
   static TMOPTargetSpec External(TargetConstructor &target);
   /** Reuse the target built for integrator @a index, which must be an
       earlier integrator that builds its own target. Saves a second remap of
       a discrete target's fields. */
   static TMOPTargetSpec SameAs(int index);
};

/// Quadrature and assembly settings shared by every integrator and the
/// solver. Per-integrator overrides live in TMOPIntegratorSpec.
struct TMOPQuadratureSpec
{
   /// Default quadrature order; also the solver's determinant checks.
   int quadrature_order = 8;
   /// A Quadrature1D value.
   int quadrature_type = Quadrature1D::GaussLobatto;
   /// Assembly level of the nonlinear form, so necessarily problem-wide.
   bool partial_assembly = false;
};

/// Displacement limiting: a penalty holding nodes near given positions.
struct TMOPLimitingSpec
{
   bool enabled = false;
   /// Constant weight, used when @a weight_coeff is null.
   real_t weight = 0.0;
   /// Spatially varying weight; borrowed, and overrides @a weight.
   Coefficient *weight_coeff = nullptr;
   /// Allowed displacement per node; borrowed. Null uses @a distance.
   const GridFunction *distance_field = nullptr;
   real_t distance = 1.0;
   /// Use TMOP_ExponentialLimiter instead of the default quadratic limiter.
   bool exponential = false;
   /** Positions to limit against; borrowed. Null uses the optimizer's
       initial nodes: those seen at construction, or at the latest
       TMOPOptimizer::Reset(). */
   const GridFunction *reference_nodes = nullptr;
};

/** One TMOP_Integrator: a metric measured against a target, plus its
    limiting term. With several integrators the problem uses a
    TMOPComboIntegrator; @a weight scales only the quality integral, never
    the limiting term. */
struct TMOPIntegratorSpec
{
   TMOPMetricSpec metric;
   TMOPTargetSpec target;

   /// Scales this integrator's quality integral only, as in TMOP_Integrator.
   /// Borrowed; null means unit weight.
   Coefficient *weight = nullptr;

   /// Per-integrator integration rules; borrowed. Null uses the problem's.
   IntegrationRules *irules = nullptr;
   /// Order used with @a irules; negative uses the problem's order.
   int quadrature_order = -1;

   /// Integrate over the target element rather than the reference element.
   bool integrate_over_target = true;
   /// Exact action of TMOP_Integrator, instead of the assembled operator.
   bool exact_action = false;

   TMOPLimitingSpec limiting;
};

/// Outer and inner solver settings.
struct TMOPSolverSpec
{
   TMOPNonlinearSolver type = TMOPNonlinearSolver::Newton;
   int max_iterations = 100;
#ifdef MFEM_USE_SINGLE
   real_t relative_tolerance = 1e-4;
#else
   real_t relative_tolerance = 1e-10;
#endif
   TMOPLinearSolver linear_solver = TMOPLinearSolver::MINRES;
   TMOPPreconditioner preconditioner = TMOPPreconditioner::None;
   int max_linear_iterations = 100;
   TMOPAdaptiveTol adaptive_tolerance = TMOPAdaptiveTol::None;

   /// Bound det(J) from below during the line search using PLBound.
   bool determinant_bound = false;
   int determinant_bound_ref_factor = 4;
   int determinant_bound_depth = 4;
   /// Reject a line-search step whose min det(J) falls to or below this.
   real_t min_determinant_limit = 0.0;

   /// Order of the determinant-check rules; negative uses the problem order.
   int quadrature_order = -1;
   /** 0 silent. 1 prints the min det(J) before and after on rank 0 and the
       Newton iterations; 2 adds the linear solver's first and last
       iterations; 3 adds every linear iteration. Energies are never
       printed; they are in the Result. */
   int verbosity = 2;
};

/** A complete, copyable description of a TMOP optimization problem.

    Members may be filled in any order and inspected at any time. No member
    is validated here; validation happens once, in the TMOPOptimizer
    constructor, where the mesh is known. */
struct TMOPProblem
{
   TMOPProblem() = default;

   /** Start from one integrator, which is the common case. More can still
       be added, and the first is Integrator(0). */
   TMOPProblem(const TMOPMetricSpec &metric, const TMOPTargetSpec &target,
               Coefficient *weight = nullptr)
   { AddIntegrator(metric, target, weight); }

   /// Start from one integrator that already carries its extra terms.
   explicit TMOPProblem(const TMOPIntegratorSpec &integrator)
   { AddIntegrator(integrator); }

   /** The integrators to sum. One entry builds a bare TMOP_Integrator;
       several build a TMOPComboIntegrator over them. */
   std::vector<TMOPIntegratorSpec> integrators;

   /// Settings TMOP applies once, to the form or to the solver.
   TMOPQuadratureSpec quadrature;
   TMOPSolverSpec solver;
   TMOPNormalization normalization = TMOPNormalization::None;

   TMOPBoundarySpec boundary;

   /// Smallest element size over the whole mesh, used only for the
   /// untangling barrier margin. Negative uses the current mesh.
   real_t initial_min_size = -1.0;

   /** Append an integrator and return its index. An index, rather than a
       reference, because a later AddIntegrator() can move the vector's
       elements; an index stays correct, and is what Integrator() and
       TMOPTargetSpec::SameAs() take. */
   int AddIntegrator(const TMOPIntegratorSpec &integrator);
   int AddIntegrator(const TMOPMetricSpec &metric,
                     const TMOPTargetSpec &target,
                     Coefficient *weight = nullptr);

   /** The integrator at @a index, for filling in its extra terms or editing
       one added earlier.

       Edits are free in any order and in any combination until the problem
       is handed to a TMOPOptimizer, which copies it. Editing the problem
       afterwards has no effect on that optimizer. */
   TMOPIntegratorSpec &Integrator(int index)
   {
      MFEM_VERIFY(index >= 0 && index < GetNumIntegrators(),
                  "TMOP integrator index " << index << " is out of range.");
      return integrators[index];
   }

   const TMOPIntegratorSpec &Integrator(int index) const
   {
      MFEM_VERIFY(index >= 0 && index < GetNumIntegrators(),
                  "TMOP integrator index " << index << " is out of range.");
      return integrators[index];
   }

   int GetNumIntegrators() const
   { return static_cast<int>(integrators.size()); }
};

/** Builds native TMOP objects from a TMOPProblem and optimizes a mesh.

    A ParMesh is optimized in parallel and any other Mesh in serial; the
    two differ only in which native objects are built. In parallel every
    GridFunction the problem borrows must be a ParGridFunction.

    The constructor copies the problem and borrows the mesh, whose nodes
    Optimize() moves. Anything the problem gives by
    pointer is borrowed, as in TMOP, must outlive the optimizer, and stays
    live; anything given by value is owned and fixed at construction.
    Discrete target fields are the exception: DiscreteAdaptTC copies their
    values.

    The "initial nodes" are a copy of the mesh nodes taken at construction
    and again by every Reset(). They are what normalization, ideal-shape
    target volumes, limiting and the periodic displacement are measured
    against when the problem names no reference nodes of its own. So a
    mesh optimized once and Reset() is optimized the second time against
    its optimized positions. With partial assembly, changes to borrowed
    objects also take effect after Reset(). Changing a metric, a target or
    whether limiting is on needs a new optimizer, as does a change of mesh
    topology or nodal space. */
class TMOPOptimizer
{
public:
   /// Outcome of the latest completed Optimize() call.
   struct Result
   {
      real_t initial_energy = 0.0;
      real_t final_energy = 0.0;
      /// Physical det(J) sampled at the solver's check points, reduced over
      /// all ranks in parallel. Not a bound over whole curved elements.
      real_t initial_minimum_determinant = 0.0;
      real_t final_minimum_determinant = 0.0;
      bool converged = false;
      int iterations = 0;
      real_t final_norm = 0.0;
   };

   /// Record @a problem by value, borrow @a mesh, validate both, and build
   /// every TMOP object the problem describes.
   TMOPOptimizer(Mesh &mesh, const TMOPProblem &problem);

   TMOPOptimizer(const TMOPOptimizer &) = delete;
   TMOPOptimizer &operator=(const TMOPOptimizer &) = delete;

   /** Take the current mesh nodes as the initial nodes, then recompute the
       target reference volumes, the partial-assembly data, normalization,
       the barrier shift and the untangler quantiles from them and from the
       borrowed inputs. Rebuilds nothing; requires unchanged mesh topology
       and nodal space. */
   void Reset();

   /// Optimize the mesh nodes in place.
   Result Optimize();

   /// Latest completed Optimize() result.
   const Result &GetResult() const;

   /// Total objective energy at the current mesh positions.
   real_t GetEnergy() const;
   /// Sampled minimum det(J) at the current mesh positions.
   real_t GetMinimumDeterminant() const;
   /// Per-integrator metric normalization factor; 1 where it is off.
   void GetNormalizationFactors(Array<real_t> &factors) const;

   int GetNumIntegrators() const;
   /// The recorded description, as given to the constructor.
   const TMOPProblem &GetProblem() const;

   /** The metric and target of integrator @a index, a position in
       TMOPProblem::integrators, e.g. for vis_tmop_metric_p(). The metric is
       the one before any untangling wrapper; it is non-const only because
       TMOP's evaluation interface is. Both stay owned by the optimizer. */
   ///@{
   TMOP_QualityMetric &GetMetric(int index);
   const TargetConstructor &GetTarget(int index) const;
   ///@}

private:
   // Member declaration order is load bearing: owners are declared
   // before the objects that borrow from them.

   /// Everything owned on behalf of one TMOPIntegratorSpec.
   struct Term
   {
      std::unique_ptr<TMOP_QualityMetric> base_metric;
      // Declared after base_metric so the borrowing wrapper dies first.
      std::unique_ptr<TMOP_QualityMetric> wrapper_metric;
      std::unique_ptr<TargetConstructor> owned_target;
      /// The metric handed to the integrator; may be the wrapper.
      TMOP_QualityMetric *integrator_metric = nullptr;
      /// Owned here or borrowed from the spec.
      TargetConstructor *target = nullptr;
      /// Owned by the form, or by the combination.
      TMOP_Integrator *integrator = nullptr;

      /** The limiting distance and weight, owned only when the spec gives
          constants; destroyed with the Term, which the form outlives. */
      std::unique_ptr<GridFunction> limit_distance;
      std::unique_ptr<ConstantCoefficient> limit_weight;
   };

   Mesh &mesh;
#ifdef MFEM_USE_MPI
   /// The mesh again when it is a ParMesh; null in serial.
   ParMesh *pmesh = nullptr;
#endif
   TMOPProblem problem;
   GridFunction *nodes = nullptr;
   FiniteElementSpace *nodal_space = nullptr;
   /// The space the nonlinear form is built on: nodal, or H1 displacement.
   FiniteElementSpace *solve_space = nullptr;
   /// H1 displacement on periodic meshes; owns its space and collection.
   std::unique_ptr<GridFunction> displacement;
   std::unique_ptr<GridFunction> initial_nodes;
   // Declared before everything that borrows the rules.
   std::unique_ptr<IntegrationRules> irules;
   // Shifted-barrier metrics hold a reference to this.
   real_t barrier_shift = -0.1;
   // Minimum det(J) of the current nodes, kept by InitializeBarrierShift()
   // and Optimize(); nodes moved any other way require Reset().
   real_t current_min_detJ = 0.0;

   std::vector<Term> terms;

   // Owns the integrators, and dies before the metrics and targets they use.
   std::unique_ptr<NonlinearForm> form;
   /// Owned by the form; null with a single integrator.
   TMOPComboIntegrator *combo = nullptr;
   // Solvers die before the form and the coefficients they borrow.
   std::unique_ptr<Solver> linear_preconditioner;
   std::unique_ptr<Solver> linear_solver;
   std::unique_ptr<TMOPNewtonSolver> nonlinear_solver;

   bool optimized = false;
   Result result;

   bool Periodic() const { return displacement != nullptr; }

   bool Parallel() const
   {
#ifdef MFEM_USE_MPI
      return pmesh != nullptr;
#else
      return false;
#endif
   }

   /// True in serial and on rank 0 in parallel; gates the summary lines.
   bool Root() const
   {
#ifdef MFEM_USE_MPI
      return !pmesh || pmesh->GetMyRank() == 0;
#else
      return true;
#endif
   }

   /// Global minimum of a per-rank value; a no-op in serial.
   void ReduceMin(real_t &value) const
   {
#ifdef MFEM_USE_MPI
      if (pmesh)
      {
         MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_MIN, pmesh->GetComm());
      }
#endif
   }

   /// A space on the mesh of the kind the mesh needs; the caller owns it.
   FiniteElementSpace *NewSpace(const FiniteElementCollection *fec, int vdim,
                                int ordering) const
   {
#ifdef MFEM_USE_MPI
      if (pmesh)
      {
         return new ParFiniteElementSpace(pmesh, fec, vdim, ordering);
      }
#endif
      return new FiniteElementSpace(&mesh, fec, vdim, ordering);
   }

   /// A function on a space made by NewSpace(); does not take the space.
   std::unique_ptr<GridFunction> NewGridFunction(FiniteElementSpace *fes) const
   {
#ifdef MFEM_USE_MPI
      if (pmesh)
      {
         return std::make_unique<ParGridFunction>(
                   static_cast<ParFiniteElementSpace *>(fes));
      }
#endif
      return std::make_unique<GridFunction>(fes);
   }

   /// A copy of the mesh nodes, of the same kind as the nodes.
   std::unique_ptr<GridFunction> CopyNodes() const
   {
#ifdef MFEM_USE_MPI
      if (pmesh)
      {
         return std::make_unique<ParGridFunction>(
                   static_cast<const ParGridFunction &>(*nodes));
      }
#endif
      return std::make_unique<GridFunction>(*nodes);
   }

   int TermOrder(const TMOPIntegratorSpec &term) const
   {
      return term.quadrature_order >= 0 ? term.quadrature_order
             : problem.quadrature.quadrature_order;
   }

   IntegrationRules &TermRules(const TMOPIntegratorSpec &term) const
   {
      return term.irules ? *term.irules : *irules;
   }

   int SolverOrder() const
   {
      return problem.solver.quadrature_order >= 0
             ? problem.solver.quadrature_order
             : problem.quadrature.quadrature_order;
   }

   const IntegrationRule &SolverRule(Geometry::Type geometry) const
   {
      return irules->Get(geometry, SolverOrder());
   }

   /// Nodes an owned target measures against: the spec's, or the initial.
   const GridFunction &TargetNodes(const TMOPTargetSpec &spec) const
   {
      return spec.kind == TMOPTargetKind::IdealShape && spec.reference_nodes
             ? *spec.reference_nodes : *initial_nodes;
   }

   const Term &CheckedTerm(int index) const
   {
      MFEM_VERIFY(index >= 0 && index < static_cast<int>(terms.size()),
                  "TMOP integrator index " << index << " is out of range.");
      return terms[index];
   }

   /// True when the spec asks for a TMOP_WorstCaseUntangleOptimizer_Metric.
   static bool NeedsUntanglerWrapper(const TMOPMetricSpec &metric)
   {
      return metric.barrier != TMOPMetricSpec::BarrierType::None ||
             metric.worst_case != TMOPMetricSpec::WorstCaseType::None;
   }

   void Validate() const;
   std::unique_ptr<TMOP_QualityMetric> ConstructMetric(
      const TMOPMetricSpec &spec);
   void BuildIdealShapeTarget(Term &owner, const TMOPTargetSpec &spec);
   void BuildAnalyticTarget(Term &owner, const TMOPTargetSpec &spec);
   void BuildDiscreteTarget(Term &owner, const TMOPTargetSpec &spec);
   void BuildTarget(Term &owner, const TMOPIntegratorSpec &integrator);
   void BalanceMetric(Term &owner, const TMOPIntegratorSpec &spec);
   void BuildIntegrators();
   void AttachLimiting(Term &owner, const TMOPLimitingSpec &lim);
   void EnableNormalization();
   void InitializeBarrierShift();
   void ApplyBoundaryConditions();
   void SetupSolver();
   /// Recompute the global quantiles the untangling wrapper depends on.
   void RefreshUntanglerQuantiles();
   /// Called once, by the constructor.
   void Build();
};

inline TMOPMetricSpec TMOPMetricSpec::Shape(int dim)
{
   TMOPMetricSpec spec;
   spec.id = dim == 2 ? 2 : 303;
   return spec;
}

inline TMOPMetricSpec TMOPMetricSpec::Size(int dim)
{
   TMOPMetricSpec spec;
   spec.id = dim == 2 ? 77 : 315;
   return spec;
}

inline TMOPMetricSpec TMOPMetricSpec::ShapeAndSize(int dim)
{
   TMOPMetricSpec spec;
   spec.id = dim == 2 ? 7 : 321;
   return spec;
}

inline TMOPMetricSpec TMOPMetricSpec::Untangler(int dim)
{
   TMOPMetricSpec spec;
   spec.id = dim == 2 ? 22 : 352;
   return spec;
}

inline TMOPMetricSpec TMOPMetricSpec::FromId(int id)
{
   TMOPMetricSpec spec;
   spec.id = id;
   return spec;
}

inline TMOPMetricSpec TMOPMetricSpec::External(TMOP_QualityMetric &metric)
{
   TMOPMetricSpec spec;
   spec.external = &metric;
   return spec;
}

inline TMOPTargetSpec TMOPTargetSpec::IdealShapeUnitSize()
{
   TMOPTargetSpec spec;
   spec.kind = TMOPTargetKind::IdealShape;
   spec.size_kind = TMOPIdealShapeTargetSize::Unit;
   return spec;
}

inline TMOPTargetSpec TMOPTargetSpec::IdealShapeEqualSize(real_t volume_scale)
{
   TMOPTargetSpec spec;
   spec.kind = TMOPTargetKind::IdealShape;
   spec.size_kind = TMOPIdealShapeTargetSize::Equal;
   spec.volume_scale = volume_scale;
   return spec;
}

inline TMOPTargetSpec TMOPTargetSpec::IdealShapeGivenSize()
{
   TMOPTargetSpec spec;
   spec.kind = TMOPTargetKind::IdealShape;
   spec.size_kind = TMOPIdealShapeTargetSize::Given;
   return spec;
}

inline TMOPTargetSpec TMOPTargetSpec::Analytic(TMOPMatrixCoefficient &matrix)
{
   TMOPTargetSpec spec;
   spec.kind = TMOPTargetKind::Analytic;
   spec.analytic = &matrix;
   return spec;
}

inline TMOPTargetSpec TMOPTargetSpec::Discrete()
{
   TMOPTargetSpec spec;
   spec.kind = TMOPTargetKind::Discrete;
   return spec;
}

inline TMOPTargetSpec TMOPTargetSpec::External(TargetConstructor &target)
{
   TMOPTargetSpec spec;
   spec.kind = TMOPTargetKind::External;
   spec.external = &target;
   return spec;
}

inline TMOPTargetSpec TMOPTargetSpec::SameAs(int index)
{
   TMOPTargetSpec spec;
   spec.kind = TMOPTargetKind::SameAs;
   spec.shared_index = index;
   return spec;
}

inline int TMOPProblem::AddIntegrator(const TMOPIntegratorSpec &integrator)
{
   integrators.push_back(integrator);
   return GetNumIntegrators() - 1;
}

inline int TMOPProblem::AddIntegrator(const TMOPMetricSpec &metric,
                                      const TMOPTargetSpec &target,
                                      Coefficient *weight)
{
   TMOPIntegratorSpec integrator;
   integrator.metric = metric;
   integrator.target = target;
   integrator.weight = weight;
   return AddIntegrator(integrator);
}

inline TMOPOptimizer::TMOPOptimizer(Mesh &mesh_,
                                    const TMOPProblem &problem_)
   : mesh(mesh_), problem(problem_)
{
#ifdef MFEM_USE_MPI
   pmesh = dynamic_cast<ParMesh *>(&mesh);
#endif
   if (!mesh.GetNodes()) { mesh.EnsureNodes(); }
   nodes = mesh.GetNodes();
#ifdef MFEM_USE_MPI
   MFEM_VERIFY(!pmesh || dynamic_cast<ParGridFunction *>(nodes),
               "A ParMesh must have parallel nodal coordinates.");
#endif
   MFEM_VERIFY(mesh.Dimension() == 2 || mesh.Dimension() == 3,
               "TMOP requires a 2D or 3D mesh.");
   nodal_space = nodes->FESpace();
   MFEM_VERIFY(nodal_space->GetVDim() == mesh.Dimension(),
               "Mesh nodal vector dimension must equal mesh dimension.");

   // Periodic meshes carry L2 nodes; TMOP then solves for an H1
   // displacement and the Newton solver reconstructs the positions.
   solve_space = nodal_space;
   if (nodal_space->IsDGSpace())
   {
      auto *fec = new H1_FECollection(nodal_space->GetMaxElementOrder(),
                                      mesh.Dimension());
      solve_space = NewSpace(fec, mesh.Dimension(),
                             nodal_space->GetOrdering());
      displacement = NewGridFunction(solve_space);
      displacement->MakeOwner(fec);
      *displacement = 0.0;
   }
   initial_nodes = CopyNodes();
   irules = std::make_unique<IntegrationRules>(
               0, problem.quadrature.quadrature_type);
   Build();
}

inline void TMOPOptimizer::Validate() const
{
   const bool pa = problem.quadrature.partial_assembly;
   MFEM_VERIFY(!pa || nodal_space->GetOrdering() == Ordering::byNODES,
               "Partial assembly requires Ordering::byNODES.");
   MFEM_VERIFY(!pa ||
               problem.solver.preconditioner != TMOPPreconditioner::L1Jacobi,
               "PA l1-Jacobi is not implemented.");
   MFEM_VERIFY(!problem.integrators.empty(),
               "A TMOP problem needs at least one integrator.");
   const int dim = mesh.Dimension();
   for (std::size_t i = 0; i < problem.integrators.size(); i++)
   {
      const TMOPIntegratorSpec &spec = problem.integrators[i];
      const TMOPMetricSpec &metric = spec.metric;
      if (!metric.external)
      {
         MFEM_VERIFY((metric.id < 300 && dim == 2) ||
                     (metric.id >= 300 && dim == 3),
                     "Metric " << metric.id << " is incompatible with "
                     "dimension " << dim << '.');
         MFEM_VERIFY(!NeedsUntanglerWrapper(metric) || !Periodic(),
                     "The untangler quantiles min(det(J)) and max(mu_hat) "
                     "cannot be computed on a periodic mesh.");
      }
      else
      {
         // BalanceMetric() works on the metric this class constructs.
         MFEM_VERIFY(!metric.balance_components,
                     "Integrator " << i << ": balance_components cannot be "
                     "applied to an external metric; balance it before "
                     "handing it over.");
      }
      const TMOPTargetSpec &target = spec.target;
      switch (target.kind)
      {
         case TMOPTargetKind::IdealShape: break;
         case TMOPTargetKind::Analytic:
            MFEM_VERIFY(target.analytic,
                        "Integrator " << i << ": an analytic target needs a "
                        "TMOPMatrixCoefficient; see TMOPTargetSpec::"
                        "Analytic().");
            break;
         case TMOPTargetKind::Discrete:
            MFEM_VERIFY(target.size || target.aspect_ratio || target.skew ||
                        target.orientation,
                        "Integrator " << i << ": a discrete target needs at "
                        "least one field: size, aspect_ratio, skew or "
                        "orientation.");
            break;
         case TMOPTargetKind::External:
            MFEM_VERIFY(target.external,
                        "Integrator " << i << ": an external target needs a "
                        "TargetConstructor; see TMOPTargetSpec::External().");
            break;
         case TMOPTargetKind::SameAs: break;
         default: MFEM_ABORT("Unknown TMOPTargetKind.");
      }
      if (target.kind == TMOPTargetKind::SameAs)
      {
         const int source = spec.target.shared_index;
         MFEM_VERIFY(source >= 0 && static_cast<std::size_t>(source) < i,
                     "TMOPTargetSpec::SameAs on integrator " << i
                     << " names " << source << "; it must name an earlier "
                     "integrator, so that the shared target already exists.");
         MFEM_VERIFY(problem.integrators[source].target.kind !=
                     TMOPTargetKind::SameAs,
                     "TMOPTargetSpec::SameAs cannot name another SameAs; "
                     "point it at the integrator that builds the target.");
      }
      MFEM_VERIFY(!pa || !spec.exact_action,
                  "Partial assembly does not support exact action.");
      MFEM_VERIFY(!spec.limiting.enabled || !pa || !Periodic(),
                  "Native partial-assembly limiting does not support "
                  "periodic reference nodes.");
   }
}

inline std::unique_ptr<TMOP_QualityMetric> TMOPOptimizer::ConstructMetric(
   const TMOPMetricSpec &spec)
{
   const real_t gamma = spec.gamma >= 0.0 ? spec.gamma
                        : (spec.id == 49 || spec.id == 126 ? 0.9 : 0.5);
   switch (spec.id)
   {
      case 1: return std::make_unique<TMOP_Metric_001>();
      case 2: return std::make_unique<TMOP_Metric_002>();
      case 4: return std::make_unique<TMOP_Metric_004>();
      case 7: return std::make_unique<TMOP_Metric_007>();
      case 9: return std::make_unique<TMOP_Metric_009>();
      case 14: return std::make_unique<TMOP_Metric_014>();
      case 22: return std::make_unique<TMOP_Metric_022>(barrier_shift);
      case 50: return std::make_unique<TMOP_Metric_050>();
      case 55: return std::make_unique<TMOP_Metric_055>();
      case 56: return std::make_unique<TMOP_Metric_056>();
      case 58: return std::make_unique<TMOP_Metric_058>();
      case 66: return std::make_unique<TMOP_Metric_066>(gamma);
      case 77: return std::make_unique<TMOP_Metric_077>();
      case 80: return std::make_unique<TMOP_Metric_080>(gamma);
      case 85: return std::make_unique<TMOP_Metric_085>();
      case 90: return std::make_unique<TMOP_Metric_090>();
      case 94: return std::make_unique<TMOP_Metric_094>();
      case 98: return std::make_unique<TMOP_Metric_098>();
      case 252: return std::make_unique<TMOP_Metric_252>(barrier_shift);
      case 301: return std::make_unique<TMOP_Metric_301>();
      case 302: return std::make_unique<TMOP_Metric_302>();
      case 303: return std::make_unique<TMOP_Metric_303>();
      case 304: return std::make_unique<TMOP_Metric_304>();
      case 315: return std::make_unique<TMOP_Metric_315>();
      case 316: return std::make_unique<TMOP_Metric_316>();
      case 318: return std::make_unique<TMOP_Metric_318>();
      case 321: return std::make_unique<TMOP_Metric_321>();
      case 322: return std::make_unique<TMOP_Metric_322>();
      case 323: return std::make_unique<TMOP_Metric_323>();
      case 328: return std::make_unique<TMOP_Metric_328>();
      case 332: return std::make_unique<TMOP_Metric_332>(gamma);
      case 333: return std::make_unique<TMOP_Metric_333>(gamma);
      case 334: return std::make_unique<TMOP_Metric_334>(gamma);
      case 338: return std::make_unique<TMOP_Metric_338>();
      case 347: return std::make_unique<TMOP_Metric_347>(gamma);
      case 352: return std::make_unique<TMOP_Metric_352>(barrier_shift);
      case 360: return std::make_unique<TMOP_Metric_360>();
      case 11: return std::make_unique<TMOP_AMetric_011>();
      case 36: return std::make_unique<TMOP_AMetric_036>();
      case 49: return std::make_unique<TMOP_AMetric_049>(gamma);
      case 51: return std::make_unique<TMOP_AMetric_051>();
      case 107: return std::make_unique<TMOP_AMetric_107>();
      case 126: return std::make_unique<TMOP_AMetric_126>(gamma);
      default: MFEM_ABORT("Unknown metric ID: " << spec.id);
   }
   return nullptr;
}

inline void TMOPOptimizer::BuildIdealShapeTarget(Term &owner,
                                                 const TMOPTargetSpec &spec)
{
   TargetConstructor::TargetType type =
      TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   switch (spec.size_kind)
   {
      case TMOPIdealShapeTargetSize::Unit: break;
      case TMOPIdealShapeTargetSize::Equal:
         type = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case TMOPIdealShapeTargetSize::Given:
         type = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      default: MFEM_ABORT("Unknown TMOPIdealShapeTargetSize.");
   }
   std::unique_ptr<TargetConstructor> target;
#ifdef MFEM_USE_MPI
   if (pmesh)
   {
      target = std::make_unique<TargetConstructor>(type, pmesh->GetComm());
   }
#endif
   if (!target) { target = std::make_unique<TargetConstructor>(type); }
   target->SetNodes(TargetNodes(spec));
   target->SetVolumeScale(spec.volume_scale);
   owner.target = target.get();
   owner.owned_target = std::move(target);
}

inline void TMOPOptimizer::BuildAnalyticTarget(Term &owner,
                                               const TMOPTargetSpec &spec)
{
   MFEM_VERIFY(spec.analytic->GetHeight() == mesh.Dimension() &&
               spec.analytic->GetWidth() == mesh.Dimension(),
               "The analytic target matrix dimension must match the mesh.");
   auto target = std::make_unique<AnalyticAdaptTC>(
                    TargetConstructor::GIVEN_FULL);
   target->SetAnalyticTargetSpec(nullptr, nullptr, spec.analytic);
   target->SetNodes(TargetNodes(spec));
   owner.target = target.get();
   owner.owned_target = std::move(target);
}

inline void TMOPOptimizer::BuildDiscreteTarget(Term &owner,
                                               const TMOPTargetSpec &spec)
{
   MFEM_VERIFY(mesh.GetNE() > 0,
               "Discrete targets need elements on every rank in the "
               "current backend.");
   MFEM_VERIFY(!(mesh.Dimension() == 3 && spec.orientation),
               "The current DiscreteAdaptTC backend does not support "
               "valid 3D orientation targets.");
   const auto type = spec.aspect_ratio || spec.skew || spec.orientation
                     ? TargetConstructor::GIVEN_SHAPE_AND_SIZE
                     : TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
   auto target = std::make_unique<DiscreteAdaptTC>(type);
   switch (spec.transfer)
   {
      case TMOPFieldTransfer::Advection:
      {
         const AssemblyLevel assembly = problem.quadrature.partial_assembly
                                        ? AssemblyLevel::PARTIAL
                                        : AssemblyLevel::LEGACY;
         target->SetAdaptivityEvaluator(new AdvectorCG(assembly));
         break;
      }
      case TMOPFieldTransfer::Interpolation:
#ifdef MFEM_USE_GSLIB
         target->SetAdaptivityEvaluator(new InterpolatorFP);
#else
         MFEM_ABORT("Discrete target interpolation requires "
                    "MFEM_USE_GSLIB.");
#endif
         break;
      default: MFEM_ABORT("Unknown TMOPFieldTransfer.");
   }
   // The native setters copy the values; the fields' FE collection is
   // borrowed, which the spec's lifetime rule already covers.
#ifdef MFEM_USE_MPI
   if (pmesh)
   {
      auto par = [](const GridFunction *field)
      {
         auto *pfield = dynamic_cast<const ParGridFunction *>(field);
         MFEM_VERIFY(pfield, "Discrete target fields on a ParMesh must be "
                     "ParGridFunctions.");
         return pfield;
      };
      if (spec.size) { target->SetParDiscreteTargetSize(*par(spec.size)); }
      if (spec.aspect_ratio)
      {
         target->SetParDiscreteTargetAspectRatio(*par(spec.aspect_ratio));
      }
      if (spec.skew) { target->SetParDiscreteTargetSkew(*par(spec.skew)); }
      if (spec.orientation)
      {
         target->SetParDiscreteTargetOrientation(*par(spec.orientation));
      }
   }
   else
#endif
   {
      if (spec.size) { target->SetSerialDiscreteTargetSize(*spec.size); }
      if (spec.aspect_ratio)
      {
         target->SetSerialDiscreteTargetAspectRatio(*spec.aspect_ratio);
      }
      if (spec.skew) { target->SetSerialDiscreteTargetSkew(*spec.skew); }
      if (spec.orientation)
      {
         target->SetSerialDiscreteTargetOrientation(*spec.orientation);
      }
   }
   if (spec.min_size >= 0.0) { target->SetMinSizeForTargets(spec.min_size); }
   target->SetNodes(TargetNodes(spec));
   owner.target = target.get();
   owner.owned_target = std::move(target);
}

inline void TMOPOptimizer::BuildTarget(Term &owner,
                                       const TMOPIntegratorSpec &integrator)
{
   const TMOPTargetSpec &spec = integrator.target;
   switch (spec.kind)
   {
      case TMOPTargetKind::IdealShape:
         BuildIdealShapeTarget(owner, spec); break;
      case TMOPTargetKind::Analytic: BuildAnalyticTarget(owner, spec); break;
      case TMOPTargetKind::Discrete: BuildDiscreteTarget(owner, spec); break;
      case TMOPTargetKind::External: owner.target = spec.external; break;
      case TMOPTargetKind::SameAs:
         // Validated already; the source was built on an earlier pass, so
         // this borrows it rather than owning a second copy.
         owner.target = terms[spec.shared_index].target;
         break;
      default: MFEM_ABORT("Unknown TMOPTargetKind.");
   }
}

inline void TMOPOptimizer::BalanceMetric(Term &owner,
                                         const TMOPIntegratorSpec &spec)
{
   auto *combination =
      dynamic_cast<TMOP_Combo_QualityMetric *>(owner.base_metric.get());
   MFEM_VERIFY(combination, "Expected a combination metric.");
   Vector weights;
   const IntegrationRule &ir = TermRules(spec).Get(
                                  mesh.GetTypicalElementGeometry(),
                                  TermOrder(spec));
   combination->ComputeBalancedWeights(
      *nodes, *owner.target, weights,
      problem.quadrature.partial_assembly, &ir);
   combination->SetWeights(weights);
}

inline void TMOPOptimizer::BuildIntegrators()
{
   terms.resize(problem.integrators.size());
   if (terms.size() > 1)
   {
      combo = new TMOPComboIntegrator;
      form->AddDomainIntegrator(combo);
   }
   for (std::size_t i = 0; i < problem.integrators.size(); i++)
   {
      const TMOPIntegratorSpec &spec = problem.integrators[i];
      Term &owner = terms[i];
      BuildTarget(owner, spec);

      if (spec.metric.external)
      {
         owner.integrator_metric = spec.metric.external;
      }
      else
      {
         owner.base_metric = ConstructMetric(spec.metric);
         if (NeedsUntanglerWrapper(spec.metric))
         {
            owner.wrapper_metric =
               std::make_unique<TMOP_WorstCaseUntangleOptimizer_Metric>(
                  *owner.base_metric, spec.metric.barrier_exponent,
                  spec.metric.barrier_alpha, spec.metric.barrier_epsilon,
                  spec.metric.worst_case_epsilon, spec.metric.barrier,
                  spec.metric.worst_case);
         }
         owner.integrator_metric = owner.wrapper_metric
                                   ? owner.wrapper_metric.get()
                                   : owner.base_metric.get();
      }

      auto *integrator = new TMOP_Integrator(owner.integrator_metric,
                                             owner.target);
      if (combo) { combo->AddTMOPIntegrator(integrator); }
      else { form->AddDomainIntegrator(integrator); }
      owner.integrator = integrator;
      integrator->IntegrateOverTarget(spec.integrate_over_target);
      integrator->SetExactActionFlag(spec.exact_action);
      integrator->SetIntegrationRules(TermRules(spec), TermOrder(spec));
      if (owner.wrapper_metric)
      {
         // Seeds min(det(J)) for the shifted barrier and max(mu_hat) for
         // the Beta worst-case form; both are global over the mesh. Done
         // after SetIntegrationRules() so that this and Reset() sample the
         // same rule.
         integrator->ComputeUntangleMetricQuantiles(*nodes, *nodal_space);
      }
      if (spec.metric.balance_components) { BalanceMetric(owner, spec); }
      if (spec.weight) { integrator->SetCoefficient(*spec.weight); }

      // Limiting hangs off this same integrator, so it is attached here
      // rather than hoisted to the problem.
      AttachLimiting(owner, spec.limiting);
   }
}

inline void TMOPOptimizer::AttachLimiting(Term &owner,
                                          const TMOPLimitingSpec &lim)
{
   if (!lim.enabled) { return; }
   const GridFunction *distance = lim.distance_field;
   if (!distance)
   {
      const FiniteElementCollection *nodal_fec = solve_space->FEColl();
      auto *fec = nodal_fec->Clone(nodal_fec->GetOrder());
      auto *fes = NewSpace(fec, 1, solve_space->GetOrdering());
      owner.limit_distance = NewGridFunction(fes);
      owner.limit_distance->MakeOwner(fec);
      *owner.limit_distance = lim.distance;
      distance = owner.limit_distance.get();
   }
   Coefficient *weight = lim.weight_coeff;
   if (!weight)
   {
      owner.limit_weight = std::make_unique<ConstantCoefficient>(lim.weight);
      weight = owner.limit_weight.get();
   }
   const GridFunction &reference = lim.reference_nodes
                                   ? *lim.reference_nodes
                                   : *initial_nodes;
   auto *limiter = lim.exponential ? new TMOP_ExponentialLimiter : nullptr;
   owner.integrator->EnableLimiting(reference, *distance, *weight, limiter);
}

inline void TMOPOptimizer::EnableNormalization()
{
   // The parallel variants reduce the initial energies over all ranks.
   auto normalize = [&](auto &object)
   {
#ifdef MFEM_USE_MPI
      if (pmesh)
      {
         object.ParEnableNormalization(
            static_cast<const ParGridFunction &>(*initial_nodes));
         return;
      }
#endif
      object.EnableNormalization(*initial_nodes);
   };
   switch (problem.normalization)
   {
      case TMOPNormalization::None: return;
      case TMOPNormalization::CombinedInitialMetric:
         // The combination normalizes each of its integrators itself.
         if (combo) { normalize(*combo); }
         else { normalize(*terms[0].integrator); }
         break;
      case TMOPNormalization::PerIntegratorInitialMetric:
         for (auto &term : terms) { normalize(*term.integrator); }
         break;
      default: MFEM_ABORT("Unknown TMOPNormalization.");
   }
}

inline real_t TMOPOptimizer::GetMinimumDeterminant() const
{
   real_t min_detJ = infinity();
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      const FiniteElement *fe = nodal_space->GetFE(i);
      const IntegrationRule &ir = SolverRule(fe->GetGeomType());
      ElementTransformation *transf = mesh.GetElementTransformation(i);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         min_detJ = std::min(min_detJ, transf->Jacobian().Det());
      }
   }
   ReduceMin(min_detJ);
   return min_detJ;
}

inline void TMOPOptimizer::InitializeBarrierShift()
{
   current_min_detJ = GetMinimumDeterminant();
   barrier_shift = current_min_detJ;
   if (current_min_detJ < 0.0)
   {
      for (std::size_t i = 0; i < problem.integrators.size(); i++)
      {
         const TMOPMetricSpec &metric = problem.integrators[i].metric;
         MFEM_VERIFY(metric.external ||
                     metric.barrier != TMOPMetricSpec::BarrierType::None ||
                     metric.id == 22 || metric.id == 252 || metric.id == 352,
                     "An inverted mesh requires untangling support in "
                     "every integrator; metric " << metric.id
                     << " cannot untangle.");
         MFEM_VERIFY(terms[i].target->GetTargetType() ==
                     TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                     "Untangling is supported only for ideal unit-size "
                     "targets.");
      }
      const DenseMatrix &ideal = Geometries.GetGeomToPerfGeomJac(
                                    mesh.GetTypicalElementGeometry());
      barrier_shift /= ideal.Det();
      real_t h0_min = problem.initial_min_size;
      if (h0_min < 0.0)
      {
         h0_min = infinity();
         for (int i = 0; i < mesh.GetNE(); i++)
         {
            h0_min = std::min(h0_min, mesh.GetElementSize(i));
         }
         ReduceMin(h0_min);
      }
      barrier_shift -= 0.01 * h0_min;
   }
}

inline void TMOPOptimizer::ApplyBoundaryConditions()
{
   const int dim = mesh.Dimension();
   const int all = (1 << dim) - 1;
   const TMOPBoundarySpec &bdr = problem.boundary;

   switch (bdr.constraint)
   {
      case TMOPBdrConstraint::Free:
         form->SetEssentialVDofs(Array<int>());
         return;
      case TMOPBdrConstraint::FixAll:
      {
         // Every attribute fully fixed: attribute marking is enough, and no
         // boundary element loop is needed.
         const int max_attribute = mesh.bdr_attributes.Size()
                                   ? mesh.bdr_attributes.Max() : 0;
         Array<int> essential_boundary(max_attribute);
         essential_boundary = 1;
         form->SetEssentialBC(essential_boundary);
         return;
      }
      case TMOPBdrConstraint::FixXYZComponentsByAttribute: break;
      default: MFEM_ABORT("Unknown TMOPBdrConstraint.");
   }

   Array<int> vdofs, essential_vdofs;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      const int attr = mesh.GetBdrElement(i)->GetAttribute();
      const auto entry = bdr.boundary_overrides.find(attr);
      if (entry == bdr.boundary_overrides.end()) { continue; }
      int flags = entry->second;
      // z has no meaning in 2D, so asking for it fixes everything.
      if (dim == 2 && (flags & TMOP_BDR_FIX_Z)) { flags = TMOP_BDR_FIX_ALL; }
      flags &= all;
      if (flags == TMOP_BDR_FREE) { continue; }
      solve_space->GetBdrElementVDofs(i, vdofs);
      const int nd = solve_space->GetBE(i)->GetDof();
      for (int c = 0; c < dim; c++)
      {
         if (!(flags & (1 << c))) { continue; }
         for (int j = 0; j < nd; j++)
         {
            essential_vdofs.Append(vdofs[j + c * nd]);
         }
      }
   }
   form->SetEssentialVDofs(essential_vdofs);
}

inline void TMOPOptimizer::SetupSolver()
{
   const auto &sol = problem.solver;
#ifdef MFEM_USE_SINGLE
   const real_t linear_relative_tolerance = 1e-5;
#else
   const real_t linear_relative_tolerance = 1e-12;
#endif
   IterativeSolver::PrintLevel linear_print;
   if (sol.verbosity == 2)
   { linear_print.Errors().Warnings().FirstAndLast(); }
   if (sol.verbosity > 2)
   { linear_print.Errors().Warnings().Iterations(); }

   std::unique_ptr<IterativeSolver> krylov;
   switch (sol.linear_solver)
   {
      case TMOPLinearSolver::MINRES:
#ifdef MFEM_USE_MPI
         if (pmesh)
         {
            krylov = std::make_unique<MINRESSolver>(pmesh->GetComm());
            break;
         }
#endif
         krylov = std::make_unique<MINRESSolver>();
         break;
      default: MFEM_ABORT("Unknown TMOPLinearSolver.");
   }
   krylov->SetMaxIter(sol.max_linear_iterations);
   krylov->SetRelTol(linear_relative_tolerance);
   krylov->SetAbsTol(0.0);
   krylov->SetPrintLevel(linear_print);

   // Matrix-based diagonal smoothers: Hypre's in parallel, DSmoother in
   // serial. Both take the absolute value of the diagonal.
   auto diagonal_smoother = [&](bool l1) -> std::unique_ptr<Solver>
   {
#ifdef MFEM_USE_MPI
      if (pmesh)
      {
         auto smoother = std::make_unique<HypreSmoother>();
         smoother->SetType(l1 ? HypreSmoother::l1Jacobi : HypreSmoother::Jacobi,
                           1);
         smoother->SetPositiveDiagonal(true);
         return smoother;
      }
#endif
      auto smoother = std::make_unique<DSmoother>(
                         l1 ? DSmoother::L1_JACOBI : DSmoother::JACOBI, 1.0, 1);
      smoother->SetPositiveDiagonal(true);
      return smoother;
   };
   switch (sol.preconditioner)
   {
      case TMOPPreconditioner::None: break;
      case TMOPPreconditioner::Jacobi:
         if (problem.quadrature.partial_assembly)
         {
            auto smoother = std::make_unique<OperatorJacobiSmoother>();
            smoother->SetPositiveDiagonal(true);
            linear_preconditioner = std::move(smoother);
         }
         else
         {
            linear_preconditioner = diagonal_smoother(false);
         }
         break;
      case TMOPPreconditioner::L1Jacobi:
         linear_preconditioner = diagonal_smoother(true);
         break;
      default: MFEM_ABORT("Unknown TMOPPreconditioner.");
   }
   if (linear_preconditioner)
   {
      krylov->SetPreconditioner(*linear_preconditioner);
   }
   linear_solver = std::move(krylov);

   int type = 0;
   switch (sol.type)
   {
      case TMOPNonlinearSolver::Newton: type = 0; break;
      case TMOPNonlinearSolver::LBFGS: type = 1; break;
      default: MFEM_ABORT("Unknown TMOPNonlinearSolver.");
   }
   const IntegrationRule &ir = SolverRule(mesh.GetTypicalElementGeometry());
#ifdef MFEM_USE_MPI
   if (pmesh)
   {
      nonlinear_solver = std::make_unique<TMOPNewtonSolver>(pmesh->GetComm(),
                                                            ir, type);
   }
#endif
   if (!nonlinear_solver)
   {
      nonlinear_solver = std::make_unique<TMOPNewtonSolver>(ir, type);
   }
   TMOPNewtonSolver &newton = *nonlinear_solver;
   newton.SetIntegrationRules(*irules, SolverOrder());
   if (type == 0) { newton.SetPreconditioner(*linear_solver); }
   newton.SetMinDetPtr(&barrier_shift);
   newton.SetMaxIter(sol.max_iterations);
   newton.SetRelTol(sol.relative_tolerance);
   newton.SetAbsTol(0.0);
   switch (sol.adaptive_tolerance)
   {
      case TMOPAdaptiveTol::None: break;
      case TMOPAdaptiveTol::EisenstatWalker1:
         newton.SetAdaptiveLinRtol(1, 0.5, 0.9); break;
      case TMOPAdaptiveTol::EisenstatWalker2:
         newton.SetAdaptiveLinRtol(2, 0.5, 0.9); break;
      default: MFEM_ABORT("Unknown TMOPAdaptiveTol.");
   }
   if (sol.min_determinant_limit != 0.0)
   {
      newton.SetMinimumDeterminantThreshold(sol.min_determinant_limit);
   }
   IterativeSolver::PrintLevel newton_print;
   if (sol.verbosity > 0)
   { newton_print.Errors().Warnings().Iterations(); }
   else { newton_print.Errors().Warnings(); }
   newton.SetPrintLevel(newton_print);
}

inline void TMOPOptimizer::RefreshUntanglerQuantiles()
{
   for (auto &term : terms)
   {
      if (!term.wrapper_metric) { continue; }
      term.integrator->ComputeUntangleMetricQuantiles(*nodes, *nodal_space);
   }
}

inline void TMOPOptimizer::Build()
{
   Validate();

#ifdef MFEM_USE_MPI
   if (pmesh)
   {
      form = std::make_unique<ParNonlinearForm>(
                static_cast<ParFiniteElementSpace *>(solve_space));
   }
#endif
   if (!form) { form = std::make_unique<NonlinearForm>(solve_space); }
   if (problem.quadrature.partial_assembly)
   {
      form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   // Adds the integrators to the form and attaches their limiting.
   BuildIntegrators();
   // Partial assembly must follow the full integrator setup.
   if (problem.quadrature.partial_assembly) { form->Setup(); }

   EnableNormalization();
   InitializeBarrierShift();
   if (problem.solver.verbosity > 0 && Root())
   {
      mfem::out << "Minimum det(J) of the original mesh is "
                << current_min_detJ << std::endl;
   }
   ApplyBoundaryConditions();
   SetupSolver();
}

inline void TMOPOptimizer::Reset()
{
   // The current positions become the initial nodes. Everything that
   // borrowed *initial_nodes (targets, limiting, the periodic displacement
   // origin) sees the new values through the same object; owned targets
   // are told again so that they drop their cached average volume.
   *initial_nodes = *nodes;
   for (std::size_t i = 0; i < terms.size(); i++)
   {
      if (terms[i].owned_target)
      {
         terms[i].owned_target->SetNodes(
            TargetNodes(problem.integrators[i].target));
      }
   }
   // Partial assembly copies the limiting reference nodes, distance and
   // constant weights, so reassemble before normalization reads that data.
   // The essential DOFs depend only on the topology and are kept.
   if (problem.quadrature.partial_assembly) { form->Setup(); }
   EnableNormalization();
   result = Result();
   InitializeBarrierShift();
   RefreshUntanglerQuantiles();
   optimized = false;
}

inline real_t TMOPOptimizer::GetEnergy() const
{
   real_t energy;
   if (!Periodic())
   {
      energy = form->GetGridFunctionEnergy(*nodes);
   }
   else
   {
      // On a periodic mesh the form lives on the H1 displacement space, so
      // the integrators must be told where the displacement starts from.
      // Elementwise projection, so a serial copy serves in parallel too.
      GridFunction displacement_l2(*nodes);
      displacement_l2 -= *initial_nodes;
      displacement->ProjectGridFunction(displacement_l2);
      for (const auto &term : terms)
      {
         term.integrator->SetInitialMeshPos(initial_nodes.get());
      }
      energy = form->GetGridFunctionEnergy(*displacement);
      for (const auto &term : terms)
      {
         term.integrator->SetInitialMeshPos(nullptr);
      }
   }
   return energy;
}

inline void TMOPOptimizer::GetNormalizationFactors(Array<real_t> &factors) const
{
   factors.SetSize(static_cast<int>(terms.size()));
   for (std::size_t i = 0; i < terms.size(); i++)
   {
      real_t metric_normal = 1.0, lim_normal = 1.0, fit_normal = 1.0;
      terms[i].integrator->GetNormalizationFactors(metric_normal,
                                                   lim_normal, fit_normal);
      factors[static_cast<int>(i)] = metric_normal;
   }
}

inline TMOPOptimizer::Result TMOPOptimizer::Optimize()
{
   optimized = false;
   Result outcome;
   outcome.initial_energy = GetEnergy();
   outcome.initial_minimum_determinant = current_min_detJ;
   if (problem.solver.determinant_bound)
   {
      // The bound field follows the mesh, so refresh it before the solve.
      nonlinear_solver->EnsurePositiveDeterminantBound(
         mesh, problem.solver.determinant_bound_ref_factor,
         problem.solver.determinant_bound_depth);
   }
   // A caller may have moved nodes without refreshing the true DOFs.
   nodes->SetTrueVector();

   // TMOPNewtonSolver::Mult takes the mesh-node true vector and handles
   // the periodic displacement conversion internally.
   TMOPNewtonSolver &newton = *nonlinear_solver;
   Vector rhs(0);
   newton.SetOperator(*form);
   newton.Mult(rhs, nodes->GetTrueVector());
   nodes->SetFromTrueVector();
   mesh.NodesUpdated();

   outcome.final_energy = GetEnergy();
   outcome.final_minimum_determinant = GetMinimumDeterminant();
   current_min_detJ = outcome.final_minimum_determinant;
   outcome.converged = newton.GetConverged();
   outcome.iterations = newton.GetNumIterations();
   outcome.final_norm = newton.GetFinalNorm();
   result = outcome;
   optimized = true;
   if (problem.solver.verbosity > 0 && Root())
   {
      mfem::out << "Minimum det(J) of the optimized mesh is "
                << outcome.final_minimum_determinant << std::endl;
   }
   return result;
}

inline const TMOPOptimizer::Result &TMOPOptimizer::GetResult() const
{
   MFEM_VERIFY(optimized,
               "No completed Optimize() call to report a result for.");
   return result;
}

inline int TMOPOptimizer::GetNumIntegrators() const
{ return static_cast<int>(terms.size()); }

inline const TMOPProblem &TMOPOptimizer::GetProblem() const
{ return problem; }

inline TMOP_QualityMetric &TMOPOptimizer::GetMetric(int index)
{
   const Term &term = CheckedTerm(index);
   return term.base_metric ? *term.base_metric : *term.integrator_metric;
}

inline const TargetConstructor &TMOPOptimizer::GetTarget(int index) const
{ return *CheckedTerm(index).target; }

} // namespace mfem

#endif // MFEM_TMOP_INTERFACE_HPP
