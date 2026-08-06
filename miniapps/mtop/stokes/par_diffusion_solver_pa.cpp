#include "par_diffusion_solver_pa.hpp"

#include <utility>

namespace
{

/**
 * @brief Prevent an iterative solver from replacing a preconditioner's operator.
 *
 * MFEM's IterativeSolver::SetOperator() forwards the high-order PA operator to
 * its preconditioner. A LOR preconditioner must remain bound to its assembled
 * low-order Hypre matrix, so this adapter deliberately ignores SetOperator().
 */
class FixedOperatorPreconditioner : public mfem::Solver
{
public:
   /**
    * @brief Wrap a preconfigured solver whose operator must remain unchanged.
    * @param solver Borrowed solver that must outlive this adapter.
    */
   explicit FixedOperatorPreconditioner(mfem::Solver &solver)
      : mfem::Solver(solver.Height(), false), solver_(solver) { }

   /**
    * @brief Ignore operator replacement requested by the outer Krylov solver.
    * @param op Ignored high-order operator.
    */
   void SetOperator(const mfem::Operator &op) override { (void) op; }

   /**
    * @brief Apply the wrapped fixed-operator preconditioner.
    * @param x Input residual vector.
    * @param y Output preconditioned vector.
    */
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override
   {
      solver_.Mult(x, y);
   }

   /// @brief Return the wrapped solver's preferred memory class.
   mfem::MemoryClass GetMemoryClass() const override
   {
      return solver_.GetMemoryClass();
   }

private:
   mfem::Solver &solver_;
};

}

/**
 * @brief Construct a PA diffusion solver with an owned constant coefficient.
 * @param fes Parallel scalar H1 space.
 * @param value Constant diffusion coefficient.
 * @param rel_tol Relative CG tolerance.
 * @param max_iter Maximum CG iterations.
 * @param print_level CG and LOR-AMG verbosity.
 */
ParDiffusionSolverPA::ParDiffusionSolverPA(
   mfem::ParFiniteElementSpace &fes,
   mfem::real_t value,
   mfem::real_t rel_tol,
   int max_iter,
   int print_level)
   : mfem::Solver(fes.GetTrueVSize(), false),
     fes_(fes),
     comm_(fes.GetComm()),
     owned_diffusion_coefficient_(
        std::make_shared<mfem::ConstantCoefficient>(value)),
     diffusion_coefficient_(owned_diffusion_coefficient_.get()),
     rel_tol_(rel_tol),
     max_iter_(max_iter),
     print_level_(print_level),
     cg_(comm_),
     rhs_work_(fes.GetTrueVSize()),
     z_(fes.GetTrueVSize()),
     m_(fes.GetTrueVSize()),
     x_bc_(fes.GetTrueVSize())
{
   mfem::real_t min_value = 0.0, max_value = 0.0;
   MPI_Allreduce(&value, &min_value, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN, comm_);
   MPI_Allreduce(&value, &max_value, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, comm_);
   MFEM_VERIFY(min_value == max_value,
               "Diffusion coefficient value must agree on all ranks.");
   Initialize(0);
}

/**
 * @brief Construct a PA diffusion solver with a borrowed coefficient.
 * @param fes Parallel scalar H1 space.
 * @param coefficient Borrowed coefficient that must outlive the solver.
 * @param rel_tol Relative CG tolerance.
 * @param max_iter Maximum CG iterations.
 * @param print_level CG and LOR-AMG verbosity.
 */
ParDiffusionSolverPA::ParDiffusionSolverPA(
   mfem::ParFiniteElementSpace &fes,
   mfem::Coefficient &coefficient,
   mfem::real_t rel_tol,
   int max_iter,
   int print_level)
   : mfem::Solver(fes.GetTrueVSize(), false),
     fes_(fes),
     comm_(fes.GetComm()),
     diffusion_coefficient_(&coefficient),
     rel_tol_(rel_tol),
     max_iter_(max_iter),
     print_level_(print_level),
     cg_(comm_),
     rhs_work_(fes.GetTrueVSize()),
     z_(fes.GetTrueVSize()),
     m_(fes.GetTrueVSize()),
     x_bc_(fes.GetTrueVSize())
{
   Initialize(1);
}

/**
 * @brief Construct a PA diffusion solver with shared coefficient ownership.
 * @param fes Parallel scalar H1 space.
 * @param coefficient Non-null shared coefficient.
 * @param rel_tol Relative CG tolerance.
 * @param max_iter Maximum CG iterations.
 * @param print_level CG and LOR-AMG verbosity.
 */
ParDiffusionSolverPA::ParDiffusionSolverPA(
   mfem::ParFiniteElementSpace &fes,
   std::shared_ptr<mfem::Coefficient> coefficient,
   mfem::real_t rel_tol,
   int max_iter,
   int print_level)
   : mfem::Solver(fes.GetTrueVSize(), false),
     fes_(fes),
     comm_(fes.GetComm()),
     owned_diffusion_coefficient_(std::move(coefficient)),
     diffusion_coefficient_(owned_diffusion_coefficient_.get()),
     rel_tol_(rel_tol),
     max_iter_(max_iter),
     print_level_(print_level),
     cg_(comm_),
     rhs_work_(fes.GetTrueVSize()),
     z_(fes.GetTrueVSize()),
     m_(fes.GetTrueVSize()),
     x_bc_(fes.GetTrueVSize())
{
   Initialize(2);
}

/**
 * @brief Validate collective state and initialize the operator and solver.
 * @param constructor_kind Identifier for the selected constructor overload.
 */
void ParDiffusionSolverPA::Initialize(int constructor_kind)
{
   VerifyConstructorKind(constructor_kind);
   CollectiveVerify(fes_.GetVDim() == 1 && !fes_.IsDGSpace(),
                    "ParDiffusionSolverPA requires a scalar continuous space.");
   CollectiveVerify(diffusion_coefficient_ != nullptr,
                    "ParDiffusionSolverPA requires a coefficient on every rank.");
   VerifyCollectiveConfiguration();
   ReassembleOperator();
   BuildConstantModeAndMassVector();
   Assemble();
}

/**
 * @brief Verify that all ranks selected the same constructor overload.
 * @param kind Constructor identifier local to this rank.
 */
void ParDiffusionSolverPA::VerifyConstructorKind(int kind) const
{
   int min_kind = 0, max_kind = 0;
   MPI_Allreduce(&kind, &min_kind, 1, MPI_INT, MPI_MIN, comm_);
   MPI_Allreduce(&kind, &max_kind, 1, MPI_INT, MPI_MAX, comm_);
   MFEM_VERIFY(min_kind == max_kind,
               "All ranks must use the same ParDiffusionSolverPA constructor.");
}

/// @brief Verify rank-invariant Krylov configuration and local validity.
void ParDiffusionSolverPA::VerifyCollectiveConfiguration() const
{
   const int local[2] = {max_iter_, print_level_};
   int minimum[2], maximum[2];
   MPI_Allreduce(local, minimum, 2, MPI_INT, MPI_MIN, comm_);
   MPI_Allreduce(local, maximum, 2, MPI_INT, MPI_MAX, comm_);
   mfem::real_t min_tol = 0.0, max_tol = 0.0;
   MPI_Allreduce(&rel_tol_, &min_tol, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN, comm_);
   MPI_Allreduce(&rel_tol_, &max_tol, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, comm_);
   MFEM_VERIFY(minimum[0] == maximum[0] && minimum[1] == maximum[1] &&
               min_tol == max_tol,
               "Solver parameters must agree on all ranks.");
   MFEM_VERIFY(max_iter_ > 0 && rel_tol_ >= 0.0,
               "Invalid CG convergence parameters.");
}

/**
 * @brief Assert a condition collectively over the FE-space communicator.
 * @param condition Rank-local condition.
 * @param message Failure message if any rank reports false.
 */
void ParDiffusionSolverPA::CollectiveVerify(bool condition,
                                            const char *message) const
{
   const int local = condition ? 1 : 0;
   int global = 0;
   MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm_);
   MFEM_VERIFY(global == 1, message);
}

/**
 * @brief Reject external replacement of the internally assembled PA operator.
 * @param op Ignored operator required by mfem::Solver.
 */
void ParDiffusionSolverPA::SetOperator(const mfem::Operator &op)
{
   (void) op;
   MFEM_ABORT("ParDiffusionSolverPA owns its operator; use "
              "ReassembleOperator() followed by Assemble().");
}

/// @brief Return the memory class of the current system operator.
mfem::MemoryClass ParDiffusionSolverPA::GetMemoryClass() const
{
   MFEM_VERIFY(system_operator_.Ptr(), "Call Assemble() first.");
   return system_operator_->GetMemoryClass();
}

/**
 * @brief Record borrowed coefficient-valued Dirichlet data.
 * @param attribute One-based boundary attribute.
 * @param coefficient Borrowed boundary value coefficient.
 */
void ParDiffusionSolverPA::AddBoundaryCondition(
   int attribute, mfem::Coefficient &coefficient)
{
   ValidateBoundaryAttribute(attribute);
   RemoveBoundaryCondition(attribute);
   BoundaryConditionEntry entry;
   entry.boundary_attribute = attribute;
   entry.coefficient = &coefficient;
   boundary_conditions_.push_back(std::move(entry));
   needs_assembly_ = true;
}

/**
 * @brief Record an owned constant Dirichlet value.
 * @param attribute One-based boundary attribute.
 * @param value Constant boundary value.
 */
void ParDiffusionSolverPA::AddBoundaryCondition(int attribute,
                                                mfem::real_t value)
{
   ValidateBoundaryAttribute(attribute);
   RemoveBoundaryCondition(attribute);
   BoundaryConditionEntry entry;
   entry.boundary_attribute = attribute;
   entry.owned_coefficient =
      std::make_shared<mfem::ConstantCoefficient>(value);
   entry.coefficient = entry.owned_coefficient.get();
   boundary_conditions_.push_back(std::move(entry));
   needs_assembly_ = true;
}

/// @brief Remove all recorded boundary conditions and mark assembly stale.
void ParDiffusionSolverPA::ClearBoundaryConditions()
{
   boundary_conditions_.clear();
   needs_assembly_ = true;
}

/// @brief Rebuild the unconstrained high-order partial-assembly operator.
void ParDiffusionSolverPA::ReassembleOperator()
{
   preconditioner_.reset();
   lor_amg_.reset();
   lor_discretization_.reset();
   system_operator_.Clear();
   full_operator_.Clear();
   diffusion_form_.reset(new mfem::ParBilinearForm(&fes_));
   diffusion_form_->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
   diffusion_form_->AddDomainIntegrator(
      new mfem::DiffusionIntegrator(*diffusion_coefficient_));
   diffusion_form_->Assemble();
   diffusion_form_->FormSystemMatrix(empty_tdof_list_, full_operator_);
   needs_assembly_ = true;
}

/**
 * @brief Rebuild boundary data, constrained PA operator, and LOR-AMG solver.
 *
 * The boundary-condition count and attribute set are verified collectively
 * before any rank enters the operator construction sequence.
 */
void ParDiffusionSolverPA::Assemble()
{
   const int local_count = static_cast<int>(boundary_conditions_.size());
   int minimum_count = 0, maximum_count = 0;
   MPI_Allreduce(&local_count, &minimum_count, 1, MPI_INT, MPI_MIN, comm_);
   MPI_Allreduce(&local_count, &maximum_count, 1, MPI_INT, MPI_MAX, comm_);
   MFEM_VERIFY(minimum_count == maximum_count,
               "Boundary-condition count must agree on all ranks.");

   int local_attribute_sum = 0;
   for (const BoundaryConditionEntry &entry : boundary_conditions_)
   {
      local_attribute_sum += entry.boundary_attribute;
   }
   int minimum_attribute_sum = 0, maximum_attribute_sum = 0;
   MPI_Allreduce(&local_attribute_sum, &minimum_attribute_sum, 1, MPI_INT,
                 MPI_MIN, comm_);
   MPI_Allreduce(&local_attribute_sum, &maximum_attribute_sum, 1, MPI_INT,
                 MPI_MAX, comm_);
   MFEM_VERIFY(minimum_attribute_sum == maximum_attribute_sum,
               "Boundary attributes must agree on all ranks.");

   BuildBoundaryValuesAndMarkers();
   system_operator_.Clear();
   diffusion_form_->FormSystemMatrix(ess_tdof_list_, system_operator_);
   use_mean_free_mode_ = boundary_conditions_.empty();
   ConfigureLinearSolver();
   needs_assembly_ = false;
}

/**
 * @brief Construct the typed LOR matrix, BoomerAMG, adapter, and CG bindings.
 *
 * The fixed-operator adapter prevents CG from replacing AMG's LOR matrix with
 * the high-order partial-assembly operator.
 */
void ParDiffusionSolverPA::ConfigureLinearSolver()
{
   preconditioner_.reset();
   lor_amg_.reset();
   lor_discretization_.reset(new mfem::ParLORDiscretization(
      *diffusion_form_, ess_tdof_list_));
   mfem::HypreParMatrix &lor_matrix =
      lor_discretization_->GetAssembledMatrix();
   lor_amg_.reset(new mfem::HypreBoomerAMG(lor_matrix));
   lor_amg_->SetPrintLevel(print_level_);
   preconditioner_.reset(new FixedOperatorPreconditioner(*lor_amg_));
   cg_.SetRelTol(rel_tol_);
   cg_.SetAbsTol(0.0);
   cg_.SetMaxIter(max_iter_);
   cg_.SetPrintLevel(print_level_);
   cg_.SetPreconditioner(*preconditioner_);
   cg_.SetOperator(*system_operator_);
}

/**
 * @brief Solve the current Dirichlet or mean-free Neumann system.
 * @param rhs Input true-DOF right-hand side.
 * @param x Output true-DOF solution.
 */
void ParDiffusionSolverPA::Mult(const mfem::Vector &rhs,
                                mfem::Vector &x) const
{
   CollectiveVerify(!needs_assembly_, "Call Assemble() before Mult().");
   CollectiveVerify(rhs.Size() == Height(),
                    "Mult expects a true-DOF RHS on every rank.");
   FormSystemRHS(rhs, rhs_work_);
   x.SetSize(Width());
   x.UseDevice(true);
   x = 0.0;
   cg_.Mult(rhs_work_, x);
   if (use_mean_free_mode_) { SetZeroMean(x); }
   else { CopyEssentialValues(x); }
   last_num_iterations_ = cg_.GetNumIterations();
   last_final_residual_ = cg_.GetFinalNorm();
   last_converged_ = cg_.GetConverged();
}

/**
 * @brief Form the right-hand side used by the current PA system operator.
 * @param rhs Input true-DOF load.
 * @param system_rhs Output eliminated or compatibility-projected load.
 */
void ParDiffusionSolverPA::FormSystemRHS(
   const mfem::Vector &rhs, mfem::Vector &system_rhs) const
{
   MFEM_VERIFY(!needs_assembly_, "Call Assemble() first.");
   system_rhs.SetSize(rhs.Size());
   system_rhs.UseDevice(true);
   if (use_mean_free_mode_)
   {
      ProjectRHS(rhs, system_rhs);
   }
   else
   {
      system_rhs = rhs;
      const mfem::ConstrainedOperator *constrained =
         dynamic_cast<const mfem::ConstrainedOperator*>(system_operator_.Ptr());
      MFEM_VERIFY(constrained, "Expected a constrained PA system operator.");
      constrained->EliminateRHS(x_bc_, system_rhs);
   }
}

/**
 * @brief Project a load into the range of the pure-Neumann operator.
 * @param rhs Input true-DOF load.
 * @param projected_rhs Output compatible load.
 */
void ParDiffusionSolverPA::ProjectRHS(
   const mfem::Vector &rhs, mfem::Vector &projected_rhs) const
{
   projected_rhs.SetSize(rhs.Size());
   projected_rhs.UseDevice(true);
   projected_rhs = rhs;
   MakeCompatible(projected_rhs);
}

/// @brief Return whether the assembled solver uses mean-free Neumann mode.
bool ParDiffusionSolverPA::UsesMeanFreeMode() const
{
   MFEM_VERIFY(!needs_assembly_, "Call Assemble() first.");
   return use_mean_free_mode_;
}

/// @brief Return whether the assembled solver has Dirichlet conditions.
bool ParDiffusionSolverPA::HasEssentialBoundaryConditions() const
{
   MFEM_VERIFY(!needs_assembly_, "Call Assemble() first.");
   return !use_mean_free_mode_;
}

/// @brief Return the number of recorded boundary-attribute conditions.
int ParDiffusionSolverPA::GetNumBoundaryConditions() const
{
   return static_cast<int>(boundary_conditions_.size());
}

/// @brief Return the current constrained or unconstrained PA system operator.
const mfem::Operator &ParDiffusionSolverPA::GetSystemOperator() const
{
   MFEM_VERIFY(!needs_assembly_, "Call Assemble() first.");
   return *system_operator_;
}

/// @brief Return the unconstrained true-DOF PA diffusion operator.
const mfem::Operator &ParDiffusionSolverPA::GetFullOperator() const
{
   return *full_operator_;
}

/// @brief Return the current local essential true-DOF list.
const mfem::Array<int> &ParDiffusionSolverPA::GetEssentialTrueDofs() const
{
   MFEM_VERIFY(!needs_assembly_, "Call Assemble() first.");
   return ess_tdof_list_;
}

/// @brief Return the current essential boundary-attribute marker.
const mfem::Array<int> &
ParDiffusionSolverPA::GetBoundaryAttributeMarker() const
{
   MFEM_VERIFY(!needs_assembly_, "Call Assemble() first.");
   return bdr_attr_marker_;
}

/// @brief Return prescribed boundary values in true-DOF layout.
const mfem::Vector &
ParDiffusionSolverPA::GetEssentialTrueDofValues() const
{
   MFEM_VERIFY(!needs_assembly_, "Call Assemble() first.");
   return x_bc_;
}

/**
 * @brief Compute the integral mean of a distributed true-DOF vector.
 * @param x True-DOF finite-element coefficient vector.
 * @return Global integral mean.
 */
mfem::real_t ParDiffusionSolverPA::Mean(const mfem::Vector &x) const
{
   return mfem::InnerProduct(comm_, m_, x)/volume_;
}

/**
 * @brief Compute the total load against the constant function.
 * @param rhs Distributed true-DOF load.
 * @return Global constant-mode load.
 */
mfem::real_t ParDiffusionSolverPA::TotalLoad(const mfem::Vector &rhs) const
{
   return mfem::InnerProduct(comm_, z_, rhs);
}

/// @brief Return the maximum one-based mesh boundary attribute, or zero.
int ParDiffusionSolverPA::MaxBoundaryAttribute() const
{
   const mfem::ParMesh *mesh = fes_.GetParMesh();
   return mesh->bdr_attributes.Size() ? mesh->bdr_attributes.Max() : 0;
}

/**
 * @brief Validate a requested one-based mesh boundary attribute.
 * @param attribute Boundary attribute to validate.
 */
void ParDiffusionSolverPA::ValidateBoundaryAttribute(int attribute) const
{
   const mfem::ParMesh *mesh = fes_.GetParMesh();
   bool found = false;
   for (int i = 0; i < mesh->bdr_attributes.Size(); i++)
   {
      found = found || mesh->bdr_attributes[i] == attribute;
   }
   MFEM_VERIFY(attribute > 0 && attribute <= MaxBoundaryAttribute() && found,
               "Invalid boundary attribute.");
}

/**
 * @brief Remove any recorded condition for a boundary attribute.
 * @param attribute One-based boundary attribute.
 */
void ParDiffusionSolverPA::RemoveBoundaryCondition(int attribute)
{
   for (std::size_t i = 0; i < boundary_conditions_.size();)
   {
      if (boundary_conditions_[i].boundary_attribute == attribute)
      {
         boundary_conditions_.erase(boundary_conditions_.begin() + i);
      }
      else { i++; }
   }
}

/// @brief Project recorded boundary data and build essential true-DOF metadata.
void ParDiffusionSolverPA::BuildBoundaryValuesAndMarkers()
{
   const int max_attribute = MaxBoundaryAttribute();
   bdr_attr_marker_.SetSize(max_attribute);
   bdr_attr_marker_ = 0;
   ess_tdof_list_.DeleteAll();
   x_bc_.SetSize(fes_.GetTrueVSize());
   x_bc_.UseDevice(true);
   x_bc_ = 0.0;
   if (boundary_conditions_.empty()) { return; }

   mfem::ParGridFunction boundary_values(&fes_);
   boundary_values = 0.0;
   mfem::Array<int> marker(max_attribute);
   for (const BoundaryConditionEntry &entry : boundary_conditions_)
   {
      marker = 0;
      marker[entry.boundary_attribute - 1] = 1;
      boundary_values.ProjectBdrCoefficient(*entry.coefficient, marker);
      bdr_attr_marker_[entry.boundary_attribute - 1] = 1;
   }
   fes_.GetEssentialTrueDofs(bdr_attr_marker_, ess_tdof_list_);
   boundary_values.ParallelProject(x_bc_);
}

/// @brief Build the constant mode, integration vector, and global domain volume.
void ParDiffusionSolverPA::BuildConstantModeAndMassVector()
{
   mfem::ConstantCoefficient one(1.0);
   mfem::ParGridFunction one_grid_function(&fes_);
   one_grid_function.ProjectCoefficient(one);
   one_grid_function.ParallelProject(z_);
   z_.UseDevice(true);
   mfem::ParLinearForm integral(&fes_);
   integral.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
   integral.Assemble();
   integral.ParallelAssemble(m_);
   m_.UseDevice(true);
   volume_ = mfem::InnerProduct(comm_, z_, m_);
   MFEM_VERIFY(volume_ > 0.0, "Non-positive domain volume.");
}

/**
 * @brief Remove the incompatible constant component of a load in place.
 * @param rhs True-DOF load to modify.
 */
void ParDiffusionSolverPA::MakeCompatible(mfem::Vector &rhs) const
{
   rhs.Add(-TotalLoad(rhs)/volume_, m_);
}

/**
 * @brief Shift a Neumann solution to zero integral mean.
 * @param x True-DOF solution to modify.
 */
void ParDiffusionSolverPA::SetZeroMean(mfem::Vector &x) const
{
   x.Add(-Mean(x), z_);
}

/**
 * @brief Copy prescribed values to essential true DOFs using an MFEM kernel.
 * @param x True-DOF solution to modify.
 */
void ParDiffusionSolverPA::CopyEssentialValues(mfem::Vector &x) const
{
   auto indices = ess_tdof_list_.Read();
   auto values = x_bc_.Read();
   auto output = x.ReadWrite();
   mfem::forall(ess_tdof_list_.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      output[indices[i]] = values[indices[i]];
   });
}
