#include "ParMassMatrixSolver.hpp"

#include <utility>

ParMassMatrixSolver::ParMassMatrixSolver(
   mfem::ParFiniteElementSpace &fes,
   mfem::Coefficient &mass_coefficient,
   mfem::real_t rel_tol,
   int max_iter,
   int print_level)
   : mfem::Solver(fes.GetTrueVSize(), false),
     fes_(fes),
     comm_(fes.GetComm()),
     mass_coefficient_(&mass_coefficient),
     rel_tol_(rel_tol),
     max_iter_(max_iter),
     print_level_(print_level),
     cg_(comm_)
{
   VerifyConstructorKind(0);
   Initialize();
}

ParMassMatrixSolver::ParMassMatrixSolver(
   mfem::ParFiniteElementSpace &fes,
   mfem::real_t coefficient_value,
   mfem::real_t rel_tol,
   int max_iter,
   int print_level)
   : mfem::Solver(fes.GetTrueVSize(), false),
     fes_(fes),
     comm_(fes.GetComm()),
     owned_coefficient_(
        std::make_shared<mfem::ConstantCoefficient>(coefficient_value)),
     mass_coefficient_(owned_coefficient_.get()),
     rel_tol_(rel_tol),
     max_iter_(max_iter),
     print_level_(print_level),
     cg_(comm_)
{
   VerifyConstructorKind(1);

   mfem::real_t min_value = 0.0;
   mfem::real_t max_value = 0.0;
   MPI_Allreduce(&coefficient_value, &min_value, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN, comm_);
   MPI_Allreduce(&coefficient_value, &max_value, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, comm_);
   MFEM_VERIFY(min_value == max_value,
               "Constant mass coefficient must be identical on all ranks.");

   Initialize();
}

ParMassMatrixSolver::ParMassMatrixSolver(
   mfem::ParFiniteElementSpace &fes,
   std::shared_ptr<mfem::Coefficient> mass_coefficient,
   mfem::real_t rel_tol,
   int max_iter,
   int print_level)
   : mfem::Solver(fes.GetTrueVSize(), false),
     fes_(fes),
     comm_(fes.GetComm()),
     owned_coefficient_(std::move(mass_coefficient)),
     mass_coefficient_(owned_coefficient_.get()),
     rel_tol_(rel_tol),
     max_iter_(max_iter),
     print_level_(print_level),
     cg_(comm_)
{
   VerifyConstructorKind(2);
   Initialize();
}

ParMassMatrixSolver::~ParMassMatrixSolver() = default;

void ParMassMatrixSolver::Initialize()
{
   CollectiveVerify(
      fes_.GetVDim() == 1,
      "ParMassMatrixSolver expects a scalar finite element space on every rank.");
   CollectiveVerify(
      mass_coefficient_ != nullptr,
      "ParMassMatrixSolver requires a valid coefficient on every rank.");
   VerifyCollectiveConfiguration();
   ReassembleOperator();
}

void ParMassMatrixSolver::VerifyConstructorKind(int constructor_kind) const
{
   int minimum_kind = 0;
   int maximum_kind = 0;
   MPI_Allreduce(&constructor_kind, &minimum_kind, 1, MPI_INT, MPI_MIN, comm_);
   MPI_Allreduce(&constructor_kind, &maximum_kind, 1, MPI_INT, MPI_MAX, comm_);
   MFEM_VERIFY(minimum_kind == maximum_kind,
               "All ranks must use the same ParMassMatrixSolver constructor.");
}

void ParMassMatrixSolver::VerifyCollectiveConfiguration() const
{
   const int local_values[2] = {max_iter_, print_level_};
   int minimum_values[2];
   int maximum_values[2];
   MPI_Allreduce(local_values, minimum_values, 2, MPI_INT, MPI_MIN, comm_);
   MPI_Allreduce(local_values, maximum_values, 2, MPI_INT, MPI_MAX, comm_);

   mfem::real_t minimum_tolerance = 0.0;
   mfem::real_t maximum_tolerance = 0.0;
   MPI_Allreduce(&rel_tol_, &minimum_tolerance, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN, comm_);
   MPI_Allreduce(&rel_tol_, &maximum_tolerance, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, comm_);

   MFEM_VERIFY(minimum_values[0] == maximum_values[0],
               "CG maximum iterations must be identical on all ranks.");
   MFEM_VERIFY(minimum_values[1] == maximum_values[1],
               "CG print level must be identical on all ranks.");
   MFEM_VERIFY(minimum_tolerance == maximum_tolerance,
               "CG relative tolerance must be identical on all ranks.");
   MFEM_VERIFY(rel_tol_ >= 0.0, "CG relative tolerance must be non-negative.");
   MFEM_VERIFY(max_iter_ > 0, "CG maximum iterations must be positive.");
}

void ParMassMatrixSolver::CollectiveVerify(bool local_condition,
                                           const char *message) const
{
   const int local_ok = local_condition ? 1 : 0;
   int global_ok = 0;
   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm_);
   MFEM_VERIFY(global_ok == 1, message);
}

void ParMassMatrixSolver::ReassembleOperator()
{
   // All ranks execute the same assembly sequence.
   preconditioner_.reset();
   mass_operator_.Clear();

   mass_form_.reset(new mfem::ParBilinearForm(&fes_));
   mass_form_->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
   mass_form_->AddDomainIntegrator(
      new mfem::MassIntegrator(*mass_coefficient_));
   mass_form_->Assemble();

   mass_form_->FormSystemMatrix(empty_ess_tdofs_, mass_operator_);
   preconditioner_.reset(
      new mfem::OperatorJacobiSmoother(*mass_form_, empty_ess_tdofs_));

   cg_.SetRelTol(rel_tol_);
   cg_.SetAbsTol(0.0);
   cg_.SetMaxIter(max_iter_);
   cg_.SetPrintLevel(print_level_);
   cg_.SetPreconditioner(*preconditioner_);
   cg_.SetOperator(*mass_operator_);
}

void ParMassMatrixSolver::Mult(const mfem::Vector &rhs,
                               mfem::Vector &x) const
{
   const int local_iterative_mode = iterative_mode ? 1 : 0;
   int minimum_iterative_mode = 0;
   int maximum_iterative_mode = 0;
   MPI_Allreduce(&local_iterative_mode, &minimum_iterative_mode, 1, MPI_INT,
                 MPI_MIN, comm_);
   MPI_Allreduce(&local_iterative_mode, &maximum_iterative_mode, 1, MPI_INT,
                 MPI_MAX, comm_);
   MFEM_VERIFY(minimum_iterative_mode == maximum_iterative_mode,
               "iterative_mode must be identical on all ranks.");

   CollectiveVerify(rhs.Size() == Height(),
                    "Mult expects a true-DOF RHS vector on every rank.");

   const bool valid_initial_guess =
      !iterative_mode || x.Size() == Width();
   CollectiveVerify(valid_initial_guess,
                    "With iterative_mode enabled, x must have true-DOF size "
                    "on every rank.");

   if (!iterative_mode)
   {
      x.SetSize(Width());
      x = 0.0;
   }

   cg_.iterative_mode = iterative_mode;
   cg_.Mult(rhs, x);

   last_num_iterations_ = cg_.GetNumIterations();
   last_final_residual_ = cg_.GetFinalNorm();
   last_converged_ = cg_.GetConverged();
}

mfem::MemoryClass ParMassMatrixSolver::GetMemoryClass() const
{
   MFEM_VERIFY(mass_operator_.Ptr() != nullptr,
               "Mass operator is not assembled.");
   return mass_operator_->GetMemoryClass();
}

void ParMassMatrixSolver::SetOperator(const mfem::Operator &op)
{
   (void) op;
   MFEM_ABORT("ParMassMatrixSolver owns its operator; use "
              "ReassembleOperator() or rebuild the solver.");
}
