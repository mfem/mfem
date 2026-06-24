// par_diffusion_solver.hpp
//
// A documented MFEM parallel scalar diffusion solver with optional essential
// Dirichlet boundary conditions.  The public solve interface matches the
// mean-free Neumann solver pattern: construct from a ParFiniteElementSpace and
// diffusion coefficient, assemble after changing boundary conditions, and call
// Mult(rhs, x) on true-DOF vectors.

#ifndef PAR_DIFFUSION_SOLVER_HPP
#define PAR_DIFFUSION_SOLVER_HPP

#include "mfem.hpp"

#include <memory>
#include <utility>
#include <vector>

/**
 * @brief Parallel scalar diffusion solver with optional Dirichlet boundary data.
 *
 * This class assembles the true-DOF diffusion operator associated with
 *
 *     a(u,v) = integral_Omega kappa grad(u) . grad(v) dx,
 *
 * where @c kappa is the scalar coefficient supplied to the constructor.  The
 * primary solve interface is @c Mult(rhs, x), where both vectors are true-DOF
 * vectors.  This mirrors the pure-Neumann solver interface: the caller owns the
 * finite element space and RHS assembly, while this object owns the diffusion
 * matrix, boundary-condition projection data, Krylov solver, and preconditioner.
 *
 * Essential Dirichlet boundary conditions are optional.  Users add them by MFEM
 * boundary attribute id with @c AddBoundaryCondition(), then call @c Assemble()
 * once after all boundary-condition edits.  The boundary attribute ids are
 * MFEM's one-based mesh boundary attributes, not zero-based array indices.
 *
 * If at least one essential boundary true DOF is present, @c Mult() solves the
 * eliminated Dirichlet system and returns a vector satisfying the prescribed
 * essential true-DOF values.  If no essential boundary true DOFs are present,
 * the class falls back to pure-Neumann mode: the RHS is projected to the
 * compatible subspace and the returned solution is shifted to zero integral
 * mean.  This makes @c ClearBoundaryConditions(); Assemble(); usable without
 * producing an unconstrained singular solve.
 *
 * The class assumes a scalar continuous H1-type finite element space.  It is
 * not intended for L2/DG spaces or vector-valued systems.
 */
class ParDiffusionSolver : public mfem::Solver
{
public:
   /**
    * @brief Construct, assemble the diffusion matrix, and initialize the solver.
    *
    * The constructor assembles the uneliminated true-DOF diffusion matrix,
    * builds the mean-free projection data used in pure-Neumann mode, and calls
    * @c Assemble() with no Dirichlet boundary conditions.  After construction,
    * the object is immediately usable as a mean-free Neumann solver.  To impose
    * Dirichlet data, call one or more @c AddBoundaryCondition() overloads and
    * then call @c Assemble() once before @c Mult().
    *
    * @param fes Parallel finite element space defining the scalar H1 trial and
    *            test space.  The space and its mesh must outlive this solver.
    * @param diffusion_coefficient Scalar coefficient multiplying grad(u) in the
    *                              diffusion bilinear form.  The coefficient must
    *                              remain valid whenever @c ReassembleOperator()
    *                              is called.
    * @param rel_tol Relative tolerance for the internal CG solver.
    * @param max_iter Maximum number of CG iterations.
    * @param print_level MFEM print level for the internal CG solver.
    */
   ParDiffusionSolver(mfem::ParFiniteElementSpace &fes,
                      mfem::Coefficient &diffusion_coefficient,
                      mfem::real_t rel_tol = 1e-12,
                      int max_iter = 500,
                      int print_level = 0)
      : mfem::Solver(fes.GetTrueVSize(), false),
        fes_(fes),
        diffusion_coefficient_(diffusion_coefficient),
        comm_(fes.GetComm()),
        cg_(comm_),
        rel_tol_(rel_tol),
        max_iter_(max_iter),
        print_level_(print_level),
        rhs_work_(fes.GetTrueVSize()),
        z_(fes.GetTrueVSize()),
        m_(fes.GetTrueVSize()),
        x_bc_(fes.GetTrueVSize())
   {
      MFEM_VERIFY(fes_.GetVDim() == 1,
                  "ParDiffusionSolver expects a scalar finite element space.");
      MFEM_VERIFY(!fes_.IsDGSpace(),
                  "ParDiffusionSolver expects a continuous H1-type space, not DG/L2.");

      ReassembleOperator();
      BuildConstantModeAndMassVector();
      Assemble();
   }

   /**
    * @brief Destroy the owned matrices, preconditioner, and work data.
    *
    * MFEM and hypre resources are held through RAII objects, so no explicit
    * cleanup is needed here.  The referenced finite element space and any
    * borrowed boundary coefficients remain owned by the caller.
    */
   ~ParDiffusionSolver() override = default;

   /**
    * @brief Disable copy construction.
    *
    * The solver owns hypre matrices, a preconditioner, and work vectors tied to
    * a specific parallel finite element space.  Copying would be ambiguous and
    * expensive, so it is explicitly disabled.
    */
   ParDiffusionSolver(const ParDiffusionSolver &) = delete;

   /**
    * @brief Disable copy assignment.
    *
    * The object stores references and owns non-copyable matrix/preconditioner
    * resources, so assignment is intentionally not supported.
    */
   ParDiffusionSolver &operator=(const ParDiffusionSolver &) = delete;

   /**
    * @brief Reject replacement through the generic MFEM Solver interface.
    *
    * This solver owns an operator assembled from the constructor finite element
    * space and diffusion coefficient.  Replacing the operator externally would
    * leave boundary-condition and mean-free projection data inconsistent.
    *
    * @param op Ignored operator argument required by @c mfem::Solver.
    */
   void SetOperator(const mfem::Operator &op) override
   {
      (void) op;
      MFEM_ABORT("ParDiffusionSolver owns its operator; use ReassembleOperator() "
                 "after mesh/coefficient changes or rebuild the solver.");
   }

   /**
    * @brief Add or replace a Dirichlet boundary condition from a coefficient.
    *
    * The supplied coefficient is projected onto boundary DOFs associated with
    * the given MFEM boundary attribute id.  Attribute ids are one-based values
    * stored in @c Mesh::bdr_attributes.  If the same attribute id was already
    * added, the old entry is replaced.  If multiple different boundary
    * attributes meet at a shared DOF and prescribe different values, the entry
    * added later determines the value at that shared DOF.
    *
    * This method only records the setting and marks the solver as needing
    * reassembly.  Call @c Assemble() once after all boundary-condition edits and
    * before the next @c Mult().
    *
    * @param boundary_attribute One-based MFEM boundary attribute id.
    * @param coefficient Dirichlet value coefficient.  The caller must keep this
    *                    object alive until after @c Assemble() has projected it.
    */
   void AddBoundaryCondition(int boundary_attribute, mfem::Coefficient &coefficient)
   {
      ValidateBoundaryAttribute(boundary_attribute);
      RemoveBoundaryCondition(boundary_attribute);

      BoundaryConditionEntry entry;
      entry.boundary_attribute = boundary_attribute;
      entry.coefficient = &coefficient;
      boundary_conditions_.push_back(std::move(entry));

      needs_assembly_ = true;
   }

   /**
    * @brief Add or replace a constant Dirichlet boundary condition.
    *
    * This overload creates and stores an owned @c mfem::ConstantCoefficient with
    * the requested value, then applies it to the specified boundary attribute at
    * the next @c Assemble().  The owned coefficient remains valid until the
    * boundary condition is replaced, cleared, or the solver is destroyed.
    *
    * This method only records the setting and marks the solver as needing
    * reassembly.  Call @c Assemble() once after all boundary-condition edits and
    * before the next @c Mult().
    *
    * @param boundary_attribute One-based MFEM boundary attribute id.
    * @param value Constant Dirichlet value on that boundary attribute.
    */
   void AddBoundaryCondition(int boundary_attribute, mfem::real_t value)
   {
      ValidateBoundaryAttribute(boundary_attribute);
      RemoveBoundaryCondition(boundary_attribute);

      BoundaryConditionEntry entry;
      entry.boundary_attribute = boundary_attribute;
      entry.owned_coefficient.reset(new mfem::ConstantCoefficient(value));
      entry.coefficient = entry.owned_coefficient.get();
      boundary_conditions_.push_back(std::move(entry));

      needs_assembly_ = true;
   }

   /**
    * @brief Remove every recorded Dirichlet boundary-condition setting.
    *
    * After this call, the next @c Assemble() produces a solver with no essential
    * boundary true DOFs.  In that state @c Mult() uses the mean-free pure-Neumann
    * fallback: it projects the RHS to the compatible subspace and returns the
    * zero-mean solution representative.
    *
    * This method only clears the settings and marks the solver as needing
    * reassembly.  Call @c Assemble() before the next @c Mult().
    */
   void ClearBoundaryConditions()
   {
      boundary_conditions_.clear();
      needs_assembly_ = true;
   }

   /**
    * @brief Rebuild boundary-condition data, eliminated matrix, and AMG setup.
    *
    * Call this method once after one or more calls to @c AddBoundaryCondition()
    * or @c ClearBoundaryConditions().  The method projects all recorded boundary
    * coefficients onto a boundary grid function, converts those values to a
    * true-DOF vector, computes the essential true-DOF list, creates the system
    * matrix, and reinitializes the BoomerAMG-preconditioned CG solver.
    *
    * This method does not reassemble element diffusion integrals.  Use
    * @c ReassembleOperator() before @c Assemble() if the mesh, finite element
    * space, or diffusion coefficient has changed.
    */
   void Assemble()
   {
      BuildBoundaryValuesAndMarkers();

      A_system_.reset(new mfem::HypreParMatrix(*A_neumann_));
      A_eliminated_.reset();

      if (boundary_conditions_.size() > 0)
      {
         A_eliminated_.reset(A_system_->EliminateRowsCols(ess_tdof_list_));
         use_mean_free_mode_ = false;
      }
      else
      {
         use_mean_free_mode_ = true;
      }

      ConfigureLinearSolver();
      needs_assembly_ = false;
   }

   /**
    * @brief Reassemble the uneliminated diffusion matrix from the FE space.
    *
    * This method rebuilds the true-DOF matrix for the bilinear form
    * @c (diffusion_coefficient grad u, grad v).  It is useful after the values
    * represented by the diffusion coefficient change while the finite element
    * space remains the same.  It marks the boundary-condition-dependent system
    * matrix as stale; call @c Assemble() afterward to rebuild the eliminated
    * matrix and preconditioner.  The finite element space size is assumed to be
    * unchanged; rebuild the solver if the mesh or space has been updated.
    */
   void ReassembleOperator()
   {
      mfem::ParBilinearForm a(&fes_);
      a.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffusion_coefficient_));
      a.Assemble();
      a.Finalize();

      A_neumann_.reset(a.ParallelAssemble());
      needs_assembly_ = true;
   }

   /**
    * @brief Solve the diffusion problem for a true-DOF RHS vector.
    *
    * If Dirichlet boundary conditions have been assembled, this method forms the
    * eliminated RHS corresponding to the prescribed essential values and solves
    * the eliminated SPD system.  The returned vector is then overwritten on the
    * essential true DOFs with the prescribed boundary values.
    *
    * If no Dirichlet boundary conditions are assembled, this method treats the
    * problem as pure Neumann: it projects @a rhs to the compatible subspace,
    * solves the singular system from a zero initial guess, and shifts the result
    * to zero integral mean.
    *
    * @param rhs Input true-DOF RHS vector.
    * @param x Output true-DOF solution vector.
    */
   void Mult(const mfem::Vector &rhs, mfem::Vector &x) const override
   {
      MFEM_VERIFY(!needs_assembly_,
                  "Boundary conditions or operator changed; call Assemble() before Mult().");
      MFEM_VERIFY(rhs.Size() == A_system_->Height(),
                  "ParDiffusionSolver::Mult expects a true-DOF RHS vector.");

      x.SetSize(A_system_->Width());
      FormSystemRHS(rhs, rhs_work_);

      x = 0.0;
      cg_.Mult(rhs_work_, x);

      if (use_mean_free_mode_)
      {
         SetZeroMean(x);
      }
      else
      {
         CopyEssentialValues(x);
      }
   }

   /**
    * @brief Form the RHS actually used by the current assembled linear system.
    *
    * In Dirichlet mode this applies the nonzero essential-value elimination to
    * the input true-DOF RHS.  In pure-Neumann mode this returns the compatible
    * mean-free projection of the input RHS.  The method is useful for testing,
    * residual checks, and diagnostics because @c GetSystemMatrix() times the
    * solution should match this vector.
    *
    * @param rhs Input true-DOF RHS vector.
    * @param system_rhs Output true-DOF RHS for the assembled system matrix.
    */
   void FormSystemRHS(const mfem::Vector &rhs, mfem::Vector &system_rhs) const
   {
      MFEM_VERIFY(!needs_assembly_,
                  "Boundary conditions or operator changed; call Assemble() before FormSystemRHS().");
      MFEM_VERIFY(rhs.Size() == A_system_->Height(),
                  "FormSystemRHS expects a true-DOF RHS vector.");

      system_rhs.SetSize(A_system_->Height());

      if (use_mean_free_mode_)
      {
         ProjectRHS(rhs, system_rhs);
      }
      else
      {
         system_rhs = rhs;
         A_system_->EliminateBC(*A_eliminated_, ess_tdof_list_, x_bc_, system_rhs);
      }
   }

   /**
    * @brief Project a RHS vector to the compatible pure-Neumann range.
    *
    * The pure-Neumann diffusion operator requires the total load against the
    * constant function to vanish.  This method computes
    *
    *     projected_rhs = rhs - (z^T rhs / |Omega|) m,
    *
    * where @c z is the true-DOF vector for the constant function 1 and @c m is
    * the true-DOF load vector representing @c v -> integral_Omega v dx.
    * Dirichlet-mode solves do not use this projection, but the method remains
    * available for diagnostics and compatibility with the Neumann solver API.
    *
    * @param rhs Input true-DOF RHS vector.
    * @param projected_rhs Output compatible true-DOF RHS vector.
    */
   void ProjectRHS(const mfem::Vector &rhs, mfem::Vector &projected_rhs) const
   {
      projected_rhs.SetSize(rhs.Size());
      projected_rhs = rhs;
      MakeCompatible(projected_rhs);
   }

   /**
    * @brief Return whether the solver is in pure-Neumann mean-free mode.
    *
    * The return value reflects the most recent @c Assemble() call.  It is true
    * when the assembled essential true-DOF list is empty and false when at least
    * one Dirichlet true DOF is constrained.
    *
    * @return True for mean-free Neumann mode; false for Dirichlet-eliminated mode.
    */
   bool UsesMeanFreeMode() const
   {
      MFEM_VERIFY(!needs_assembly_,
                  "Call Assemble() before querying the assembled solver mode.");
      return use_mean_free_mode_;
   }

   /**
    * @brief Return whether assembled Dirichlet true DOFs are present.
    *
    * This is the logical complement of @c UsesMeanFreeMode() for a currently
    * assembled solver.
    *
    * @return True if the latest assembly constrained at least one true DOF.
    */
   bool HasEssentialBoundaryConditions() const
   {
      MFEM_VERIFY(!needs_assembly_,
                  "Call Assemble() before querying assembled boundary conditions.");
      return boundary_conditions_.size() > 0;
   }

   /**
    * @brief Return whether recorded settings need to be assembled.
    *
    * The flag becomes true after adding, replacing, or clearing boundary
    * conditions, and after @c ReassembleOperator().  It becomes false after a
    * successful @c Assemble().
    *
    * @return True if @c Assemble() must be called before @c Mult().
    */
   bool NeedsAssembly() const { return needs_assembly_; }

   /**
    * @brief Return the number of recorded boundary-condition settings.
    *
    * This counts user-specified boundary attributes, not true DOFs.  The value
    * may be nonzero even before @c Assemble() has converted the settings to an
    * essential true-DOF list.
    *
    * @return Number of recorded boundary-condition entries.
    */
   int GetNumBoundaryConditions() const
   {
      return static_cast<int>(boundary_conditions_.size());
   }

   /**
    * @brief Access the currently assembled system matrix.
    *
    * In Dirichlet mode this matrix has essential rows and columns eliminated
    * with unit diagonal entries.  In mean-free Neumann mode this matrix is the
    * uneliminated diffusion matrix.
    *
    * @return Reference to the owned true-DOF system matrix.
    */
   const mfem::HypreParMatrix &GetSystemMatrix() const
   {
      MFEM_VERIFY(!needs_assembly_,
                  "Call Assemble() before accessing the system matrix.");
      return *A_system_;
   }

   /**
    * @brief Access the uneliminated true-DOF diffusion matrix.
    *
    * This matrix corresponds to the pure diffusion bilinear form before any
    * Dirichlet elimination.  It is useful for diagnostics and for forming custom
    * eliminated systems outside this class.
    *
    * @return Reference to the owned uneliminated true-DOF matrix.
    */
   const mfem::HypreParMatrix &GetFullMatrix() const { return *A_neumann_; }

   /**
    * @brief Access the assembled essential true-DOF list.
    *
    * The indices are local true-DOF indices on each MPI rank, matching MFEM's
    * @c ParFiniteElementSpace::GetEssentialTrueDofs() convention.
    *
    * @return Reference to the current essential true-DOF list.
    */
   const mfem::Array<int> &GetEssentialTrueDofs() const
   {
      MFEM_VERIFY(!needs_assembly_,
                  "Call Assemble() before accessing essential true DOFs.");
      return ess_tdof_list_;
   }

   /**
    * @brief Access the assembled boundary-attribute marker.
    *
    * The marker has length equal to the maximum boundary attribute id.  Entry
    * @c attr-1 is one when attribute @c attr is part of the assembled essential
    * boundary and zero otherwise.
    *
    * @return Reference to the current boundary-attribute marker.
    */
   const mfem::Array<int> &GetBoundaryAttributeMarker() const
   {
      MFEM_VERIFY(!needs_assembly_,
                  "Call Assemble() before accessing the boundary marker.");
      return bdr_attr_marker_;
   }

   /**
    * @brief Access the assembled true-DOF vector of Dirichlet values.
    *
    * In Dirichlet mode, entries on essential true DOFs contain prescribed
    * values and all other entries are zero.  In mean-free Neumann mode the
    * vector is identically zero.
    *
    * @return Reference to the true-DOF boundary-value vector.
    */
   const mfem::Vector &GetEssentialTrueDofValues() const
   {
      MFEM_VERIFY(!needs_assembly_,
                  "Call Assemble() before accessing essential true-DOF values.");
      return x_bc_;
   }

   /**
    * @brief Access the true-DOF vector for the constant function one.
    *
    * This vector is used internally for pure-Neumann compatibility projection
    * and zero-mean gauge fixing.
    *
    * @return Reference to the constant-mode true-DOF vector.
    */
   const mfem::Vector &GetConstantMode() const { return z_; }

   /**
    * @brief Access the true-DOF integration/load vector.
    *
    * The vector represents the linear functional
    * @c v -> integral_Omega v dx.  For a true-DOF coefficient vector @c x, the
    * integral is @c InnerProduct(comm, GetMassVector(), x).
    *
    * @return Reference to the true-DOF integration vector.
    */
   const mfem::Vector &GetMassVector() const { return m_; }

   /**
    * @brief Return the global measure of the domain.
    *
    * The value is computed as @c z^T m, where @c z is the constant one vector
    * and @c m is the integration/load vector.
    *
    * @return Domain volume or area.
    */
   mfem::real_t GetVolume() const { return volume_; }

   /**
    * @brief Compute the integral mean of a true-DOF vector.
    *
    * The mean is @c integral_Omega x dx / |Omega|, evaluated using the stored
    * true-DOF integration vector and a parallel inner product.
    *
    * @param x True-DOF vector representing a scalar finite element function.
    * @return Integral mean over the global domain.
    */
   mfem::real_t Mean(const mfem::Vector &x) const
   {
      MFEM_VERIFY(x.Size() == m_.Size(), "Mean expects a true-DOF vector.");
      return mfem::InnerProduct(comm_, m_, x)/volume_;
   }

   /**
    * @brief Compute the total load of a true-DOF RHS vector.
    *
    * This returns @c z^T rhs, where @c z represents the constant function one.
    * A compatible pure-Neumann RHS has total load zero up to discretization and
    * roundoff errors.
    *
    * @param rhs True-DOF RHS vector.
    * @return Global total load against the constant function.
    */
   mfem::real_t TotalLoad(const mfem::Vector &rhs) const
   {
      MFEM_VERIFY(rhs.Size() == z_.Size(), "TotalLoad expects a true-DOF RHS.");
      return mfem::InnerProduct(comm_, z_, rhs);
   }

private:
   /**
    * @brief Stored description of one user-specified Dirichlet boundary entry.
    *
    * The coefficient pointer is either borrowed from the caller or points to the
    * owned constant coefficient.  Borrowed coefficients only need to remain
    * alive through @c Assemble(), because assembly immediately projects their
    * values into @c x_bc_;
    */
   struct BoundaryConditionEntry
   {
      /// One-based MFEM boundary attribute id.
      int boundary_attribute = 0;

      /// Non-owning pointer to the coefficient used during projection.
      mfem::Coefficient *coefficient = nullptr;

      /// Optional owned coefficient used by the constant-value overload.
      std::unique_ptr<mfem::Coefficient> owned_coefficient;
   };

   /// Parallel finite element space used for assembly and true-DOF maps.
   mfem::ParFiniteElementSpace &fes_;

   /// Diffusion coefficient used when assembling the uneliminated operator.
   mfem::Coefficient &diffusion_coefficient_;

   /// MPI communicator associated with the finite element space.
   MPI_Comm comm_;

   /// User-recorded boundary-condition entries, one per boundary attribute.
   std::vector<BoundaryConditionEntry> boundary_conditions_;

   /// Marker for assembled essential boundary attributes.
   mfem::Array<int> bdr_attr_marker_;

   /// Assembled local true-DOF indices constrained by Dirichlet conditions.
   mfem::Array<int> ess_tdof_list_;

   /// Uneliminated true-DOF diffusion matrix.
   std::unique_ptr<mfem::HypreParMatrix> A_neumann_;

   /// Current system matrix: eliminated in Dirichlet mode, full in Neumann mode.
   std::unique_ptr<mfem::HypreParMatrix> A_system_;

   /// Matrix entries removed during Dirichlet row/column elimination.
   std::unique_ptr<mfem::HypreParMatrix> A_eliminated_;

   /// BoomerAMG preconditioner for the current system matrix.
   std::unique_ptr<mfem::HypreBoomerAMG> amg_;

   /// CG solver configured for the current system matrix.
   mutable mfem::CGSolver cg_;

   /// CG relative tolerance stored for reconfiguration after assembly.
   mfem::real_t rel_tol_;

   /// CG maximum iteration count stored for reconfiguration after assembly.
   int max_iter_;

   /// CG print level stored for reconfiguration after assembly.
   int print_level_;

   /// Work vector for projected or eliminated right-hand sides.
   mutable mfem::Vector rhs_work_;

   /// True-DOF coefficient vector representing the constant function one.
   mfem::Vector z_;

   /// True-DOF integration vector m_i = integral_Omega phi_i dx.
   mfem::Vector m_;

   /// Global domain measure, computed as z^T m.
   mfem::real_t volume_ = 0.0;

   /// True-DOF vector containing prescribed Dirichlet values on essential DOFs.
   mfem::Vector x_bc_;

   /// True after settings change and before Assemble() refreshes the system.
   bool needs_assembly_ = true;

   /// True when no essential true DOFs are assembled and Neumann mode is active.
   bool use_mean_free_mode_ = true;

   /**
    * @brief Return the maximum boundary attribute id on the parallel mesh.
    *
    * @return Maximum one-based boundary attribute id, or zero if no boundary
    *         attributes are present.
    */
   int MaxBoundaryAttribute() const
   {
      const mfem::ParMesh *pmesh = fes_.GetParMesh();
      if (pmesh->bdr_attributes.Size() == 0)
      {
         return 0;
      }
      return pmesh->bdr_attributes.Max();
   }

   /**
    * @brief Verify that a boundary attribute id exists on the mesh.
    *
    * The method checks both that the id lies in the marker range and that it is
    * present in the mesh's boundary-attribute table.  MFEM boundary attributes
    * are one-based values.
    *
    * @param boundary_attribute Attribute id to validate.
    */
   void ValidateBoundaryAttribute(int boundary_attribute) const
   {
      const mfem::ParMesh *pmesh = fes_.GetParMesh();
      const int max_attr = MaxBoundaryAttribute();

      MFEM_VERIFY(max_attr > 0,
                  "Cannot add Dirichlet data: the mesh has no boundary attributes.");
      MFEM_VERIFY(boundary_attribute >= 1 && boundary_attribute <= max_attr,
                  "Boundary attribute " << boundary_attribute
                  << " is outside the valid one-based range [1, " << max_attr << "].");

      bool found = false;
      for (int i = 0; i < pmesh->bdr_attributes.Size(); i++)
      {
         if (pmesh->bdr_attributes[i] == boundary_attribute)
         {
            found = true;
            break;
         }
      }
      MFEM_VERIFY(found,
                  "Boundary attribute " << boundary_attribute
                  << " is not present in mesh.bdr_attributes.");
   }

   /**
    * @brief Remove any recorded boundary condition for one attribute id.
    *
    * @param boundary_attribute One-based MFEM boundary attribute id to remove.
    */
   void RemoveBoundaryCondition(int boundary_attribute)
   {
      for (std::size_t i = 0; i < boundary_conditions_.size(); )
      {
         if (boundary_conditions_[i].boundary_attribute == boundary_attribute)
         {
            boundary_conditions_.erase(boundary_conditions_.begin() + i);
         }
         else
         {
            i++;
         }
      }
   }

   /**
    * @brief Build Dirichlet markers, essential true DOFs, and boundary values.
    *
    * The method projects each recorded boundary coefficient onto a temporary
    * @c ParGridFunction using @c ProjectBdrCoefficient(), converts the resulting
    * local DOF data to true DOFs, and asks the finite element space for the
    * essential true-DOF list associated with the combined boundary marker.
    */
   void BuildBoundaryValuesAndMarkers()
   {
      const int max_attr = MaxBoundaryAttribute();
      bdr_attr_marker_.SetSize(max_attr);
      bdr_attr_marker_ = 0;

      x_bc_.SetSize(fes_.GetTrueVSize());
      x_bc_ = 0.0;
      ess_tdof_list_.DeleteAll();

      if (boundary_conditions_.empty())
      {
         return;
      }

      mfem::ParGridFunction bc_gf(&fes_);
      bc_gf = 0.0;

      mfem::Array<int> one_attr_marker(max_attr);
      for (std::size_t i = 0; i < boundary_conditions_.size(); i++)
      {
         const BoundaryConditionEntry &entry = boundary_conditions_[i];
         MFEM_VERIFY(entry.coefficient != nullptr,
                     "Internal error: null boundary coefficient.");

         one_attr_marker = 0;
         one_attr_marker[entry.boundary_attribute - 1] = 1;
         bc_gf.ProjectBdrCoefficient(*entry.coefficient, one_attr_marker);
         bdr_attr_marker_[entry.boundary_attribute - 1] = 1;
      }

      fes_.GetEssentialTrueDofs(bdr_attr_marker_, ess_tdof_list_);
      bc_gf.ParallelProject(x_bc_);
   }

   /**
    * @brief Configure CG and BoomerAMG for the current system matrix.
    *
    * The method recreates the AMG object, attaches the current system matrix to
    * it, and configures CG with the tolerances supplied to the constructor.
    */
   void ConfigureLinearSolver()
   {
      amg_.reset(new mfem::HypreBoomerAMG);
      amg_->SetPrintLevel(print_level_);
      amg_->SetOperator(*A_system_);

      cg_.SetRelTol(rel_tol_);
      cg_.SetAbsTol(0.0);
      cg_.SetMaxIter(max_iter_);
      cg_.SetPrintLevel(print_level_);
      cg_.SetPreconditioner(*amg_);
      cg_.SetOperator(*A_system_);
   }

   /**
    * @brief Build the constant mode, integration vector, and domain volume.
    *
    * The constant mode is obtained by projecting the scalar coefficient 1 into
    * the finite element space and restricting it to true DOFs.  The integration
    * vector is assembled from the linear form @c integral_Omega v dx.
    */
   void BuildConstantModeAndMassVector()
   {
      mfem::ConstantCoefficient one(1.0);

      mfem::ParGridFunction one_gf(&fes_);
      one_gf.ProjectCoefficient(one);
      one_gf.ParallelProject(z_);

      mfem::ParLinearForm lf_one(&fes_);
      lf_one.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
      lf_one.Assemble();
      lf_one.ParallelAssemble(m_);

      volume_ = mfem::InnerProduct(comm_, z_, m_);
      MFEM_VERIFY(volume_ > 0.0, "Non-positive domain volume.");
   }

   /**
    * @brief Modify a RHS vector in place so that z^T rhs = 0.
    *
    * The correction subtracts the constant load component using the integration
    * vector @c m_:
    *
    *     rhs <- rhs - (z^T rhs / |Omega|) m.
    *
    * @param rhs True-DOF RHS vector to modify in place.
    */
   void MakeCompatible(mfem::Vector &rhs) const
   {
      const mfem::real_t total_load = mfem::InnerProduct(comm_, z_, rhs);
      rhs.Add(-total_load/volume_, m_);
   }

   /**
    * @brief Shift a true-DOF solution vector to have zero integral mean.
    *
    * This method is used only in pure-Neumann mode, where the solution is
    * determined up to an additive constant.
    *
    * @param x True-DOF solution vector modified in place.
    */
   void SetZeroMean(mfem::Vector &x) const
   {
      const mfem::real_t mean = Mean(x);
      x.Add(-mean, z_);
   }

   /**
    * @brief Overwrite essential true DOFs with their prescribed values.
    *
    * The eliminated system should already enforce these values through unit
    * diagonal rows.  This final copy removes Krylov roundoff from the boundary
    * entries returned to the caller.
    *
    * @param x True-DOF solution vector modified in place.
    */
   void CopyEssentialValues(mfem::Vector &x) const
   {
      for (int i = 0; i < ess_tdof_list_.Size(); i++)
      {
         const int tdof = ess_tdof_list_[i];
         x(tdof) = x_bc_(tdof);
      }
   }
};

#endif // PAR_DIFFUSION_SOLVER_HPP
