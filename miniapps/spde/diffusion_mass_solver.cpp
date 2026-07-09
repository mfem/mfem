#include "diffusion_mass_solver.hpp"

#include <algorithm>
#include <cmath>

namespace mfem
{

namespace
{

bool GlobalBooleanOr(MPI_Comm comm, bool value)
{
   int local = value ? 1 : 0;
   int global = 0;
   MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm);
   return global != 0;
}

ParFiniteElementSpace &CheckedFESpace(
   const std::shared_ptr<ParFiniteElementSpace> &fespace,
   const char *name)
{
   (void)name;
   MFEM_VERIFY(fespace != nullptr, "FE space pointer is null.");
   return *fespace;
}

QuadratureSpace &CheckedQuadratureSpace(
   const std::shared_ptr<QuadratureSpace> &qspace,
   const char *name)
{
   (void)name;
   MFEM_VERIFY(qspace != nullptr, "QuadratureSpace pointer is null.");
   return *qspace;
}

class ReciprocalCoefficient : public Coefficient
{
public:
   explicit ReciprocalCoefficient(Coefficient &coefficient)
      : coefficient_(coefficient)
   {
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      const real_t value = coefficient_.Eval(T, ip);
      MFEM_VERIFY(value != 0.0,
                  "Cannot build reciprocal coefficient from zero value.");
      return 1.0/value;
   }

private:
   Coefficient &coefficient_;
};

void ProjectPressureMeanBlock(MPI_Comm comm, const Array<int> &block_offsets,
                              Vector &vector)
{
   BlockVector block(vector, block_offsets);
   Vector &pressure = block.GetBlock(1);
   real_t local_sum = pressure.Sum();
   real_t global_sum = 0.0;
   MPI_Allreduce(&local_sum, &global_sum, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
   real_t local_size = static_cast<real_t>(pressure.Size());
   real_t global_size = 0.0;
   MPI_Allreduce(&local_size, &global_size, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
   if (global_size == 0.0) { return; }

   const real_t mean = global_sum/global_size;
   real_t *p = pressure.HostReadWrite();
   for (int i = 0; i < pressure.Size(); i++) { p[i] -= mean; }
}

void ProjectMean(MPI_Comm comm, Vector &vector)
{
   real_t local_sum = vector.Sum();
   real_t global_sum = 0.0;
   MPI_Allreduce(&local_sum, &global_sum, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
   real_t local_size = static_cast<real_t>(vector.Size());
   real_t global_size = 0.0;
   MPI_Allreduce(&local_size, &global_size, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
   if (global_size == 0.0) { return; }

   const real_t mean = global_sum/global_size;
   real_t *data = vector.HostReadWrite();
   for (int i = 0; i < vector.Size(); i++) { data[i] -= mean; }
}

class PressureMeanProjectedOperator : public Operator
{
public:
   PressureMeanProjectedOperator(const Operator &op,
                                 const Array<int> &block_offsets,
                                 MPI_Comm comm)
      : Operator(op.Height(), op.Width()),
        op_(op),
        block_offsets_(block_offsets),
        comm_(comm)
   {
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      x_projected_ = x;
      ProjectPressureMeanBlock(comm_, block_offsets_, x_projected_);
      op_.Mult(x_projected_, y);
      ProjectPressureMeanBlock(comm_, block_offsets_, y);
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      x_projected_ = x;
      ProjectPressureMeanBlock(comm_, block_offsets_, x_projected_);
      op_.MultTranspose(x_projected_, y);
      ProjectPressureMeanBlock(comm_, block_offsets_, y);
   }

private:
   const Operator &op_;
   Array<int> block_offsets_;
   MPI_Comm comm_;
   mutable Vector x_projected_;
};

class PressureMeanProjectedSolver : public Solver
{
public:
   PressureMeanProjectedSolver(Solver &solver,
                               const Array<int> &block_offsets,
                               MPI_Comm comm)
      : Solver(solver.Height(), solver.Width()),
        solver_(solver),
        block_offsets_(block_offsets),
        comm_(comm)
   {
   }

   void SetOperator(const Operator &op) override
   {
      solver_.SetOperator(op);
      height = solver_.Height();
      width = solver_.Width();
   }

   void Mult(const Vector &b, Vector &x) const override
   {
      b_projected_ = b;
      ProjectPressureMeanBlock(comm_, block_offsets_, b_projected_);
      solver_.iterative_mode = iterative_mode;
      solver_.Mult(b_projected_, x);
      ProjectPressureMeanBlock(comm_, block_offsets_, x);
   }

private:
   Solver &solver_;
   Array<int> block_offsets_;
   MPI_Comm comm_;
   mutable Vector b_projected_;
};

class CahouetChabardPreconditioner : public Solver
{
public:
   CahouetChabardPreconditioner(Solver &mass_inverse,
                                Solver &diffusion_inverse,
                                MPI_Comm comm,
                                bool project_diffusion_nullspace)
      : Solver(mass_inverse.Height(), mass_inverse.Width()),
        mass_inverse_(mass_inverse),
        diffusion_inverse_(diffusion_inverse),
        comm_(comm),
        project_diffusion_nullspace_(project_diffusion_nullspace)
   {
      MFEM_VERIFY(mass_inverse.Height() == diffusion_inverse.Height() &&
                  mass_inverse.Width() == diffusion_inverse.Width(),
                  "Cahouet-Chabard preconditioner blocks have incompatible sizes.");
   }

   void SetOperator(const Operator &) override { }

   void Mult(const Vector &x, Vector &y) const override
   {
      rhs_ = x;
      if (project_diffusion_nullspace_) { ProjectMean(comm_, rhs_); }
      y.SetSize(Height());
      diffusion_work_.SetSize(Height());
      mass_inverse_.iterative_mode = iterative_mode;
      mass_inverse_.Mult(rhs_, y);
      diffusion_inverse_.iterative_mode = false;
      diffusion_inverse_.Mult(rhs_, diffusion_work_);
      y += diffusion_work_;
      if (project_diffusion_nullspace_) { ProjectMean(comm_, y); }
   }

private:
   Solver &mass_inverse_;
   Solver &diffusion_inverse_;
   MPI_Comm comm_;
   bool project_diffusion_nullspace_;
   mutable Vector rhs_;
   mutable Vector diffusion_work_;
};

class LSCPreconditioner : public Solver
{
public:
   LSCPreconditioner(Solver &q_inverse,
                     Operator &h_operator,
                     MPI_Comm comm,
                     bool project_pressure_nullspace)
      : Solver(q_inverse.Height(), q_inverse.Width()),
        q_inverse_(q_inverse),
        h_operator_(h_operator),
        comm_(comm),
        project_pressure_nullspace_(project_pressure_nullspace)
   {
      MFEM_VERIFY(q_inverse.Height() == h_operator.Height() &&
                  q_inverse.Width() == h_operator.Width(),
                  "LSC preconditioner blocks have incompatible sizes.");
   }

   void SetOperator(const Operator &) override { }

   void Mult(const Vector &x, Vector &y) const override
   {
      rhs_ = x;
      if (project_pressure_nullspace_) { ProjectMean(comm_, rhs_); }

      first_solve_.SetSize(Height());
      h_work_.SetSize(Height());
      y.SetSize(Height());

      q_inverse_.iterative_mode = false;
      q_inverse_.Mult(rhs_, first_solve_);

      h_operator_.Mult(first_solve_, h_work_);
      if (project_pressure_nullspace_) { ProjectMean(comm_, h_work_); }

      q_inverse_.Mult(h_work_, y);
   }

private:
   Solver &q_inverse_;
   Operator &h_operator_;
   MPI_Comm comm_;
   bool project_pressure_nullspace_;
   mutable Vector rhs_;
   mutable Vector first_solve_;
   mutable Vector h_work_;
};

void MultElementwise(const Vector &diag, const Vector &x, Vector &y)
{
   y.SetSize(x.Size());
   const real_t *d = diag.HostRead();
   const real_t *xp = x.HostRead();
   real_t *yp = y.HostWrite();
   for (int i = 0; i < x.Size(); i++) { yp[i] = d[i]*xp[i]; }
}

void CopyEssentialValues(const Array<int> &ess_tdofs,
                         const Vector &x,
                         Vector &y)
{
   const int *ess = ess_tdofs.HostRead();
   const real_t *xp = x.HostRead();
   real_t *yp = y.HostReadWrite();
   for (int i = 0; i < ess_tdofs.Size(); i++) { yp[ess[i]] = xp[ess[i]]; }
}

void ZeroEssentialValues(const Array<int> &ess_tdofs, Vector &x)
{
   const int *ess = ess_tdofs.HostRead();
   real_t *xp = x.HostReadWrite();
   for (int i = 0; i < ess_tdofs.Size(); i++) { xp[ess[i]] = 0.0; }
}

class LSCQOperator : public Operator
{
public:
   LSCQOperator(const Operator &divergence_operator,
                const Vector &velocity_diag_inverse,
                const Array<int> &velocity_ess_tdofs,
                const Array<int> &pressure_ess_tdofs,
                bool eliminate_pressure_ess_tdofs,
                MPI_Comm comm)
      : Operator(divergence_operator.Height(), divergence_operator.Height()),
        divergence_operator_(divergence_operator),
        velocity_diag_inverse_(velocity_diag_inverse),
        velocity_ess_tdofs_(velocity_ess_tdofs),
        pressure_ess_tdofs_(pressure_ess_tdofs),
        eliminate_pressure_ess_tdofs_(eliminate_pressure_ess_tdofs),
        comm_(comm)
   {
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      pressure_work_ = x;
      if (eliminate_pressure_ess_tdofs_)
      {
         ZeroEssentialValues(pressure_ess_tdofs_, pressure_work_);
      }
      velocity_work_.SetSize(divergence_operator_.Width());
      scaled_velocity_work_.SetSize(divergence_operator_.Width());
      y.SetSize(Height());
      divergence_operator_.MultTranspose(pressure_work_, velocity_work_);
      ZeroEssentialValues(velocity_ess_tdofs_, velocity_work_);
      MultElementwise(velocity_diag_inverse_, velocity_work_,
                      scaled_velocity_work_);
      divergence_operator_.Mult(scaled_velocity_work_, y);
      CopyEssentialValues(pressure_ess_tdofs_, x, y);
   }

   void AssembleDiagonal(Vector &diag) const override
   {
      diag.SetSize(Height());
      Vector basis(Width());
      Vector column(Height());
      real_t *dp = diag.HostWrite();
      int global_width = 0;
      const int local_width = Width();
      MPI_Allreduce(&local_width, &global_width, 1, MPI_INT, MPI_MAX, comm_);
      for (int i = 0; i < global_width; i++)
      {
         basis = 0.0;
         if (i < Width()) { basis(i) = 1.0; }
         Mult(basis, column);
         if (i < Height()) { dp[i] = column(i); }
      }
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      Mult(x, y);
   }

private:
   const Operator &divergence_operator_;
   const Vector &velocity_diag_inverse_;
   Array<int> velocity_ess_tdofs_;
   Array<int> pressure_ess_tdofs_;
   bool eliminate_pressure_ess_tdofs_;
   MPI_Comm comm_;
   mutable Vector pressure_work_;
   mutable Vector velocity_work_;
   mutable Vector scaled_velocity_work_;
};

class LSCHOperator : public Operator
{
public:
   LSCHOperator(const Operator &divergence_operator,
                const Operator &velocity_operator,
                const Vector &velocity_diag_inverse,
                const Array<int> &velocity_ess_tdofs,
                const Array<int> &pressure_ess_tdofs,
                bool eliminate_pressure_ess_tdofs)
      : Operator(divergence_operator.Height(), divergence_operator.Height()),
        divergence_operator_(divergence_operator),
        velocity_operator_(velocity_operator),
        velocity_diag_inverse_(velocity_diag_inverse),
        velocity_ess_tdofs_(velocity_ess_tdofs),
        pressure_ess_tdofs_(pressure_ess_tdofs),
        eliminate_pressure_ess_tdofs_(eliminate_pressure_ess_tdofs)
   {
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      pressure_work_ = x;
      if (eliminate_pressure_ess_tdofs_)
      {
         ZeroEssentialValues(pressure_ess_tdofs_, pressure_work_);
      }
      velocity_work_.SetSize(divergence_operator_.Width());
      scaled_velocity_work_.SetSize(divergence_operator_.Width());
      velocity_operator_work_.SetSize(velocity_operator_.Height());
      scaled_velocity_operator_work_.SetSize(velocity_operator_.Height());
      y.SetSize(Height());
      divergence_operator_.MultTranspose(pressure_work_, velocity_work_);
      ZeroEssentialValues(velocity_ess_tdofs_, velocity_work_);
      MultElementwise(velocity_diag_inverse_, velocity_work_,
                      scaled_velocity_work_);
      velocity_operator_.Mult(scaled_velocity_work_, velocity_operator_work_);
      ZeroEssentialValues(velocity_ess_tdofs_, velocity_operator_work_);
      MultElementwise(velocity_diag_inverse_, velocity_operator_work_,
                      scaled_velocity_operator_work_);
      divergence_operator_.Mult(scaled_velocity_operator_work_, y);
      CopyEssentialValues(pressure_ess_tdofs_, x, y);
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      Mult(x, y);
   }

private:
   const Operator &divergence_operator_;
   const Operator &velocity_operator_;
   const Vector &velocity_diag_inverse_;
   Array<int> velocity_ess_tdofs_;
   Array<int> pressure_ess_tdofs_;
   bool eliminate_pressure_ess_tdofs_;
   mutable Vector pressure_work_;
   mutable Vector velocity_work_;
   mutable Vector scaled_velocity_work_;
   mutable Vector velocity_operator_work_;
   mutable Vector scaled_velocity_operator_work_;
};

}

// RieszMapOperator implementation notes:
// - The map is the L2 mass matrix on true dofs, assembled with partial
//   assembly through ParBilinearForm::FormSystemMatrix.
// - No essential boundary conditions are eliminated; the map acts on the full
//   true-vector space of the supplied ParFiniteElementSpace.
RieszMapOperator::RieszMapOperator(ParFiniteElementSpace &fespace)
   : Operator(fespace.GetTrueVSize()),
     fespace_(fespace)
{
}

RieszMapOperator::RieszMapOperator(
   std::shared_ptr<ParFiniteElementSpace> fespace)
   : Operator(CheckedFESpace(fespace, "Riesz").GetTrueVSize()),
     fespace_owner_(fespace),
     fespace_(CheckedFESpace(fespace, "Riesz"))
{
}

void RieszMapOperator::SetNeedsAssembly(bool needs_assembly) const
{
   needs_assembly_ = needs_assembly;
}

void RieszMapOperator::Assemble() const
{
   if (!GlobalBooleanOr(fespace_.GetComm(), needs_assembly_)) { return; }

   mass_operator_.Clear();
   form_.reset(new ParBilinearForm(&fespace_));
   form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form_->AddDomainIntegrator(new MassIntegrator);
   form_->Assemble();

   Array<int> empty_tdofs;
   mass_operator_.SetType(Operator::ANY_TYPE);
   form_->FormSystemMatrix(empty_tdofs, mass_operator_);

   RieszMapOperator *self = const_cast<RieszMapOperator *>(this);
   self->height = mass_operator_->Height();
   self->width = mass_operator_->Width();
   needs_assembly_ = false;
}

void RieszMapOperator::Mult(const Vector &primal, Vector &dual) const
{
   Assemble();
   MultAssembled(primal, dual);
}

void RieszMapOperator::MultAssembled(const Vector &primal, Vector &dual) const
{
   MFEM_VERIFY(mass_operator_.Ptr() != nullptr,
               "RieszMapOperator has not been assembled.");
   MFEM_VERIFY(primal.Size() == Width(),
               "Primal vector has incompatible size.");
   dual.SetSize(Height());
   mass_operator_->Mult(primal, dual);
}

void RieszMapOperator::MultTranspose(const Vector &dual,
                                     Vector &primal) const
{
   Assemble();
   MultTransposeAssembled(dual, primal);
}

void RieszMapOperator::MultTransposeAssembled(const Vector &dual,
                                              Vector &primal) const
{
   MFEM_VERIFY(mass_operator_.Ptr() != nullptr,
               "RieszMapOperator has not been assembled.");
   MFEM_VERIFY(dual.Size() == Height(), "Dual vector has incompatible size.");
   primal.SetSize(Width());
   mass_operator_->MultTranspose(dual, primal);
}

const Operator *RieszMapOperator::GetOperator() const
{
   Assemble();
   return mass_operator_.Ptr();
}

// InverseRieszMapOperator implementation notes:
// - The inverse is applied iteratively instead of explicitly assembling an
//   inverse matrix.
// - OperatorJacobiSmoother extracts the diagonal from the PA mass operator via
//   AssembleDiagonal and is reused as the CG preconditioner.
InverseRieszMapOperator::InverseRieszMapOperator(
   ParFiniteElementSpace &fespace)
   : Operator(fespace.GetTrueVSize()),
     riesz_(fespace)
{
}

InverseRieszMapOperator::InverseRieszMapOperator(
   std::shared_ptr<ParFiniteElementSpace> fespace)
   : Operator(CheckedFESpace(fespace, "Inverse Riesz").GetTrueVSize()),
     riesz_(fespace)
{
}

void InverseRieszMapOperator::SetRelTol(real_t rel_tol)
{
   MFEM_VERIFY(rel_tol >= 0.0, "Relative tolerance must be nonnegative.");
   rel_tol_ = rel_tol;
   needs_assembly_ = true;
}

void InverseRieszMapOperator::SetAbsTol(real_t abs_tol)
{
   MFEM_VERIFY(abs_tol >= 0.0, "Absolute tolerance must be nonnegative.");
   abs_tol_ = abs_tol;
   needs_assembly_ = true;
}

void InverseRieszMapOperator::SetMaxIter(int max_iter)
{
   MFEM_VERIFY(max_iter > 0, "Maximum iteration count must be positive.");
   max_iter_ = max_iter;
   needs_assembly_ = true;
}

void InverseRieszMapOperator::SetPrintLevel(int print_level)
{
   print_level_ = print_level;
   needs_assembly_ = true;
}

void InverseRieszMapOperator::SetNeedsAssembly(bool needs_assembly) const
{
   needs_assembly_ = needs_assembly;
   riesz_.SetNeedsAssembly(needs_assembly);
}

void InverseRieszMapOperator::Assemble() const
{
   const bool local_needs_assembly = needs_assembly_ || riesz_.NeedsAssembly();
   if (!GlobalBooleanOr(riesz_.GetFESpace().GetComm(), local_needs_assembly))
   {
      return;
   }

   riesz_.Assemble();
   const Operator *mass = riesz_.GetOperator();
   MFEM_VERIFY(mass != nullptr, "Riesz map operator is null.");

   jacobi_.reset(new OperatorJacobiSmoother);
   jacobi_->SetPositiveDiagonal();
   jacobi_->SetOperator(*mass);

   cg_solver_.reset(new CGSolver(riesz_.GetFESpace().GetComm()));
   cg_solver_->SetRelTol(rel_tol_);
   cg_solver_->SetAbsTol(abs_tol_);
   cg_solver_->SetMaxIter(max_iter_);
   cg_solver_->SetPrintLevel(print_level_);
   cg_solver_->SetOperator(*mass);
   cg_solver_->SetPreconditioner(*jacobi_);

   InverseRieszMapOperator *self =
      const_cast<InverseRieszMapOperator *>(this);
   self->height = mass->Height();
   self->width = mass->Width();
   needs_assembly_ = false;
}

void InverseRieszMapOperator::Mult(const Vector &dual,
                                   Vector &primal) const
{
   Assemble();
   MultAssembled(dual, primal);
}

void InverseRieszMapOperator::MultAssembled(const Vector &dual,
                                            Vector &primal) const
{
   MFEM_VERIFY(cg_solver_ != nullptr,
               "InverseRieszMapOperator has not been assembled.");
   MFEM_VERIFY(dual.Size() == Width(), "Dual vector has incompatible size.");
   primal.SetSize(Height());
   primal = 0.0;
   cg_solver_->Mult(dual, primal);
}

void InverseRieszMapOperator::MultTranspose(const Vector &dual,
                                            Vector &primal) const
{
   Assemble();
   MultTransposeAssembled(dual, primal);
}

void InverseRieszMapOperator::MultTransposeAssembled(const Vector &dual,
                                                     Vector &primal) const
{
   MultAssembled(dual, primal);
}

// TrueMassMapOperator implementation notes:
// - The input space is the ParMixedBilinearForm trial space and the output
//   space is the test space, so Mult() maps input true dofs to output true dofs.
// - MixedScalarMassIntegrator does not implement partial assembly in this MFEM
//   tree. The rectangular true-dof operator is therefore assembled as a
//   HypreParMatrix and cached until the weight is changed or the user marks the
//   map dirty.
// - No essential boundary elimination is applied; callers pass and receive full
//   true vectors in their respective spaces.
TrueMassMapOperator::TrueMassMapOperator(
   ParFiniteElementSpace &input_space,
   ParFiniteElementSpace &output_space)
   : Operator(output_space.GetTrueVSize(), input_space.GetTrueVSize()),
     input_space_(input_space),
     output_space_(output_space),
     weight_coefficient_(std::make_shared<ConstantCoefficient>(1.0))
{
   MFEM_VERIFY(input_space_.GetParMesh() == output_space_.GetParMesh(),
               "TrueMassMapOperator requires input and output spaces on the "
               "same ParMesh.");
}

TrueMassMapOperator::TrueMassMapOperator(
   std::shared_ptr<ParFiniteElementSpace> input_space,
   std::shared_ptr<ParFiniteElementSpace> output_space)
   : Operator(CheckedFESpace(output_space, "Output").GetTrueVSize(),
              CheckedFESpace(input_space, "Input").GetTrueVSize()),
     input_space_owner_(input_space),
     output_space_owner_(output_space),
     input_space_(CheckedFESpace(input_space, "Input")),
     output_space_(CheckedFESpace(output_space, "Output")),
     weight_coefficient_(std::make_shared<ConstantCoefficient>(1.0))
{
   MFEM_VERIFY(input_space_.GetParMesh() == output_space_.GetParMesh(),
               "TrueMassMapOperator requires input and output spaces on the "
               "same ParMesh.");
}

void TrueMassMapOperator::SetWeightCoefficient(real_t value)
{
   SetWeightCoefficient(std::make_shared<ConstantCoefficient>(value));
}

void TrueMassMapOperator::SetWeightCoefficient(
   Coefficient &coefficient, bool transfer_ownership)
{
   SetWeightCoefficient(ShareCoefficient(coefficient, transfer_ownership));
}

void TrueMassMapOperator::SetWeightCoefficient(
   std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(coefficient != nullptr, "Mass-map weight coefficient is null.");
   weight_coefficient_ = coefficient;
   SetNeedsAssembly();
}

void TrueMassMapOperator::SetNeedsAssembly(bool needs_assembly) const
{
   needs_assembly_ = needs_assembly;
}

void TrueMassMapOperator::Assemble() const
{
   if (!GlobalBooleanOr(input_space_.GetComm(), needs_assembly_)) { return; }

   mass_operator_.Clear();
   form_.reset(new ParMixedBilinearForm(&input_space_, &output_space_));
   form_->AddDomainIntegrator(new MixedScalarMassIntegrator(
                                 *weight_coefficient_));
   form_->Assemble();

   Array<int> empty_trial_tdofs;
   Array<int> empty_test_tdofs;
   mass_operator_.SetType(Operator::Hypre_ParCSR);
   form_->FormRectangularSystemMatrix(empty_trial_tdofs, empty_test_tdofs,
                                      mass_operator_);

   TrueMassMapOperator *self = const_cast<TrueMassMapOperator *>(this);
   self->height = mass_operator_->Height();
   self->width = mass_operator_->Width();
   needs_assembly_ = false;
}

void TrueMassMapOperator::Mult(const Vector &input, Vector &output) const
{
   Assemble();
   MultAssembled(input, output);
}

void TrueMassMapOperator::MultAssembled(const Vector &input,
                                        Vector &output) const
{
   MFEM_VERIFY(mass_operator_.Ptr() != nullptr,
               "TrueMassMapOperator has not been assembled.");
   MFEM_VERIFY(input.Size() == Width(), "Input vector has incompatible size.");
   output.SetSize(Height());
   mass_operator_->Mult(input, output);
}

void TrueMassMapOperator::MultTranspose(const Vector &output,
                                        Vector &input) const
{
   Assemble();
   MultTransposeAssembled(output, input);
}

void TrueMassMapOperator::MultTransposeAssembled(const Vector &output,
                                                 Vector &input) const
{
   MFEM_VERIFY(mass_operator_.Ptr() != nullptr,
               "TrueMassMapOperator has not been assembled.");
   MFEM_VERIFY(output.Size() == Height(),
               "Output adjoint vector has incompatible size.");
   input.SetSize(Width());
   mass_operator_->MultTranspose(output, input);
}

const Operator *TrueMassMapOperator::GetOperator() const
{
   Assemble();
   return mass_operator_.Ptr();
}

std::shared_ptr<Coefficient> TrueMassMapOperator::ShareCoefficient(
   Coefficient &coefficient, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<Coefficient>(&coefficient);
   }
   return std::shared_ptr<Coefficient>(&coefficient, [](Coefficient *) { });
}

DiffusionMassSolver::DiffusionMassSolver(
   std::shared_ptr<ParFiniteElementSpace> fespace)
   : DiffusionMassSolver(CheckedFESpace(fespace, "Solver"))
{
   fespace_owner_ = fespace;
}

PDEFilter::PDEFilter(ParFiniteElementSpace &input_space,
                     ParFiniteElementSpace &filtered_space)
   : Operator(filtered_space.GetTrueVSize(), input_space.GetTrueVSize()),
     input_space_(input_space),
     filtered_space_(filtered_space),
     mass_map_(input_space, filtered_space),
     solver_(filtered_space)
{
   MFEM_VERIFY(input_space_.GetParMesh() == filtered_space_.GetParMesh(),
               "PDEFilter requires input and filtered spaces on the same "
               "ParMesh.");
   solver_.SetMassCoefficient(1.0);
   SetFilterRadius(0.0);
}

PDEFilter::PDEFilter(std::shared_ptr<ParFiniteElementSpace> input_space,
                     std::shared_ptr<ParFiniteElementSpace> filtered_space)
   : Operator(CheckedFESpace(filtered_space, "Filtered").GetTrueVSize(),
              CheckedFESpace(input_space, "Input").GetTrueVSize()),
     input_space_owner_(input_space),
     filtered_space_owner_(filtered_space),
     input_space_(CheckedFESpace(input_space, "Input")),
     filtered_space_(CheckedFESpace(filtered_space, "Filtered")),
     mass_map_(input_space, filtered_space),
     solver_(filtered_space)
{
   MFEM_VERIFY(input_space_.GetParMesh() == filtered_space_.GetParMesh(),
               "PDEFilter requires input and filtered spaces on the same "
               "ParMesh.");
   solver_.SetMassCoefficient(1.0);
   SetFilterRadius(0.0);
}

void PDEFilter::SetDiffusionCoefficient(real_t diffusion)
{
   MFEM_VERIFY(diffusion >= 0.0,
               "PDE filter diffusion coefficient must be nonnegative.");
   diffusion_ = diffusion;
   filter_radius_ = 2.0*std::sqrt(3.0*diffusion_);
   solver_.SetDiffusionCoefficient(diffusion_);
   needs_assembly_ = true;
}

void PDEFilter::SetFilterRadius(real_t r_min)
{
   MFEM_VERIFY(r_min >= 0.0, "PDE filter radius must be nonnegative.");
   filter_radius_ = r_min;
   const real_t pde_radius = r_min/(2.0*std::sqrt(3.0));
   diffusion_ = pde_radius*pde_radius;
   solver_.SetDiffusionCoefficient(diffusion_);
   needs_assembly_ = true;
}

void PDEFilter::Mult(const Vector &input, Vector &filtered) const
{
   Assemble();
   MultAssembled(input, filtered);
}

void PDEFilter::MultAssembled(const Vector &input, Vector &filtered) const
{
   MFEM_VERIFY(input.Size() == Width(), "Input vector has incompatible size.");
   mass_map_.MultAssembled(input, rhs_);
   solver_.MultAssembled(rhs_, filtered);
}

void PDEFilter::MultTranspose(const Vector &filtered_bar,
                              Vector &input_bar) const
{
   Assemble();
   MultTransposeAssembled(filtered_bar, input_bar);
}

void PDEFilter::MultTransposeAssembled(const Vector &filtered_bar,
                                       Vector &input_bar) const
{
   MFEM_VERIFY(filtered_bar.Size() == Height(),
               "Filtered adjoint vector has incompatible size.");
   solver_.MultTransposeAssembled(filtered_bar, adjoint_);
   mass_map_.MultTransposeAssembled(adjoint_, input_bar);
}

void PDEFilter::Assemble() const
{
   const bool local_needs_assembly =
      needs_assembly_ || mass_map_.NeedsAssembly() || solver_.NeedsAssembly();
   if (!GlobalBooleanOr(filtered_space_.GetComm(), local_needs_assembly))
   {
      return;
   }

   mass_map_.Assemble();
   solver_.Assemble();
   needs_assembly_ = false;
}

QuadratureFunctionMassMapOperator::QuadratureFunctionMassMapOperator(
   QuadratureSpace &input_qspace, ParFiniteElementSpace &output_space)
   : Operator(output_space.GetTrueVSize(), input_qspace.GetSize()),
     input_qspace_(input_qspace),
     output_space_(output_space)
{
   MFEM_VERIFY(input_qspace_.GetMesh() == output_space_.GetParMesh(),
               "QuadratureFunctionMassMapOperator requires quadrature and "
               "FE spaces on the same ParMesh.");
}

QuadratureFunctionMassMapOperator::QuadratureFunctionMassMapOperator(
   std::shared_ptr<QuadratureSpace> input_qspace,
   std::shared_ptr<ParFiniteElementSpace> output_space)
   : Operator(CheckedFESpace(output_space, "Output").GetTrueVSize(),
              CheckedQuadratureSpace(input_qspace, "Input quadrature").GetSize()),
     input_qspace_owner_(input_qspace),
     output_space_owner_(output_space),
     input_qspace_(CheckedQuadratureSpace(input_qspace, "Input quadrature")),
     output_space_(CheckedFESpace(output_space, "Output"))
{
   MFEM_VERIFY(input_qspace_.GetMesh() == output_space_.GetParMesh(),
               "QuadratureFunctionMassMapOperator requires quadrature and "
               "FE spaces on the same ParMesh.");
}

void QuadratureFunctionMassMapOperator::SetNeedsAssembly(
   bool needs_assembly) const
{
   needs_assembly_ = needs_assembly;
}

void QuadratureFunctionMassMapOperator::Assemble() const
{
   if (!GlobalBooleanOr(output_space_.GetComm(), needs_assembly_)) { return; }

   MFEM_VERIFY(input_qspace_.GetNE() == output_space_.GetNE(),
               "Quadrature space and FE space have incompatible elements.");

   input_qf_view_.reset(new QuadratureFunction(&input_qspace_, 1));
   *input_qf_view_ = 0.0;
   input_qf_coeff_.reset(
      new QuadratureFunctionCoefficient(*input_qf_view_));

   rhs_form_.reset(new ParLinearForm(&output_space_));
   rhs_form_->AddDomainIntegrator(
      new QuadratureLFIntegrator(*input_qf_coeff_));

   QuadratureFunctionMassMapOperator *self =
      const_cast<QuadratureFunctionMassMapOperator *>(this);
   self->height = output_space_.GetTrueVSize();
   self->width = input_qspace_.GetSize();
   needs_assembly_ = false;
}

void QuadratureFunctionMassMapOperator::ValidateInputSize(
   const Vector &input_q) const
{
   MFEM_VERIFY(input_q.Size() == Width(),
               "Quadrature input vector has incompatible size.");
}

void QuadratureFunctionMassMapOperator::Mult(const Vector &input_q,
                                             Vector &output_true) const
{
   Assemble();
   MultAssembled(input_q, output_true);
}

void QuadratureFunctionMassMapOperator::MultAssembled(
   const Vector &input_q, Vector &output_true) const
{
   MFEM_VERIFY(rhs_form_ != nullptr,
               "QuadratureFunctionMassMapOperator has not been assembled.");
   ValidateInputSize(input_q);

   *input_qf_view_ = input_q;
   rhs_form_->Assemble();
   output_true.SetSize(Height());
   rhs_form_->ParallelAssemble(output_true);
}

void QuadratureFunctionMassMapOperator::MultTranspose(
   const Vector &output_true_bar, Vector &input_q_bar) const
{
   Assemble();
   MultTransposeAssembled(output_true_bar, input_q_bar);
}

void QuadratureFunctionMassMapOperator::MultTransposeAssembled(
   const Vector &output_true_bar, Vector &input_q_bar) const
{
   MFEM_VERIFY(output_true_bar.Size() == Height(),
               "Output adjoint true vector has incompatible size.");

   const Operator *P = output_space_.GetProlongationMatrix();
   MFEM_VERIFY(P != nullptr, "ParFiniteElementSpace prolongation is null.");
   local_adjoint_.SetSize(output_space_.GetVSize());
   P->Mult(output_true_bar, local_adjoint_);

   input_q_bar.SetSize(Width());
   input_q_bar = 0.0;
   real_t *q_bar = input_q_bar.HostWrite();

   Vector element_adjoint;
   for (int e = 0; e < output_space_.GetNE(); e++)
   {
      const FiniteElement *fe = output_space_.GetFE(e);
      ElementTransformation *Tr = output_space_.GetElementTransformation(e);
      const IntegrationRule &ir = input_qspace_.GetIntRule(e);
      const int q_offset = input_qspace_.Offset(e);

      output_space_.GetElementVDofs(e, vdofs_);
      local_adjoint_.GetSubVector(vdofs_, element_adjoint);

      shape_.SetSize(fe->GetDof());
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);
         fe->CalcShape(ip, shape_);
         q_bar[q_offset + q] = ip.weight*Tr->Weight()*
                               (shape_*element_adjoint);
      }
   }
}

QuadraturePDEFilter::QuadraturePDEFilter(
   QuadratureSpace &input_qspace, ParFiniteElementSpace &filtered_space)
   : Operator(filtered_space.GetTrueVSize(), input_qspace.GetSize()),
     input_qspace_(input_qspace),
     filtered_space_(filtered_space),
     mass_map_(input_qspace, filtered_space),
     solver_(filtered_space)
{
   MFEM_VERIFY(input_qspace_.GetMesh() == filtered_space_.GetParMesh(),
               "QuadraturePDEFilter requires quadrature and filtered spaces "
               "on the same ParMesh.");
   solver_.SetMassCoefficient(1.0);
   SetFilterRadius(0.0);
}

QuadraturePDEFilter::QuadraturePDEFilter(
   std::shared_ptr<QuadratureSpace> input_qspace,
   std::shared_ptr<ParFiniteElementSpace> filtered_space)
   : Operator(CheckedFESpace(filtered_space, "Filtered").GetTrueVSize(),
              CheckedQuadratureSpace(input_qspace, "Input quadrature").GetSize()),
     input_qspace_owner_(input_qspace),
     filtered_space_owner_(filtered_space),
     input_qspace_(CheckedQuadratureSpace(input_qspace, "Input quadrature")),
     filtered_space_(CheckedFESpace(filtered_space, "Filtered")),
     mass_map_(input_qspace, filtered_space),
     solver_(filtered_space)
{
   MFEM_VERIFY(input_qspace_.GetMesh() == filtered_space_.GetParMesh(),
               "QuadraturePDEFilter requires quadrature and filtered spaces "
               "on the same ParMesh.");
   solver_.SetMassCoefficient(1.0);
   SetFilterRadius(0.0);
}

void QuadraturePDEFilter::SetDiffusionCoefficient(real_t diffusion)
{
   MFEM_VERIFY(diffusion >= 0.0,
               "PDE filter diffusion coefficient must be nonnegative.");
   diffusion_ = diffusion;
   filter_radius_ = 2.0*std::sqrt(3.0*diffusion_);
   solver_.SetDiffusionCoefficient(diffusion_);
   needs_assembly_ = true;
}

void QuadraturePDEFilter::SetFilterRadius(real_t r_min)
{
   MFEM_VERIFY(r_min >= 0.0, "PDE filter radius must be nonnegative.");
   filter_radius_ = r_min;
   const real_t pde_radius = r_min/(2.0*std::sqrt(3.0));
   diffusion_ = pde_radius*pde_radius;
   solver_.SetDiffusionCoefficient(diffusion_);
   needs_assembly_ = true;
}

void QuadraturePDEFilter::Mult(const Vector &input_q, Vector &filtered) const
{
   Assemble();
   MultAssembled(input_q, filtered);
}

void QuadraturePDEFilter::MultAssembled(const Vector &input_q,
                                        Vector &filtered) const
{
   MFEM_VERIFY(input_q.Size() == Width(),
               "Quadrature input vector has incompatible size.");
   mass_map_.MultAssembled(input_q, rhs_);
   solver_.MultAssembled(rhs_, filtered);
}

void QuadraturePDEFilter::Assemble() const
{
   const bool local_needs_assembly =
      needs_assembly_ || mass_map_.NeedsAssembly() || solver_.NeedsAssembly();
   if (!GlobalBooleanOr(filtered_space_.GetComm(), local_needs_assembly))
   {
      return;
   }

   mass_map_.Assemble();
   solver_.Assemble();
   needs_assembly_ = false;
}

void QuadraturePDEFilter::MultTranspose(const Vector &filtered_bar,
                                        Vector &input_q_bar) const
{
   Assemble();
   MultTransposeAssembled(filtered_bar, input_q_bar);
}

void QuadraturePDEFilter::MultTransposeAssembled(
   const Vector &filtered_bar, Vector &input_q_bar) const
{
   MFEM_VERIFY(filtered_bar.Size() == Height(),
               "Filtered adjoint vector has incompatible size.");
   solver_.MultTransposeAssembled(filtered_bar, adjoint_);
   mass_map_.MultTransposeAssembled(adjoint_, input_q_bar);
}

// Implementation notes:
// - Public mutators only set needs_assembly_; Assemble() is called lazily by
//   Solve(), Mult(), and accessor methods.
// - Boundary conditions are stored as boundary attribute IDs.  Markers are
//   constructed internally only when MFEM APIs require them.
// - Solve() applies boundary coefficients to the GridFunction, then lets
//   ParBilinearForm::FormLinearSystem eliminate essential true dofs from the RHS.
// - Mult() solves an already-formed true-dof system and enforces stored
//   essential boundary IDs.  If boundary coefficients are present, constrained
//   true dofs are set to their projected boundary values; otherwise they are
//   treated as homogeneous constraints.
// - The system operator is partial assembly.  AMG preconditioning uses an
//   assembled operator on the original order-1 space or on an LOR space for
//   higher-order tensor-product spaces.
// - For ParGridFunction operator coefficients on order > 1 spaces, the LOR
//   preconditioner copies the coefficient true-dof vector into a persistent LOR
//   grid function and assembles AMG from that transferred coefficient.
// - Repeated Mult() and Solve() calls reuse the CG solver, RHS form, linear
//   system vectors, boundary projection storage, and coefficient transfer vector
//   until a public mutator marks the corresponding cache dirty.

void DiffusionMassSolver::AttributeCoefficientMap::SetOwner(
   DiffusionMassSolver *owner, MapKind kind)
{
   owner_ = owner;
   kind_ = kind;
}

void DiffusionMassSolver::AttributeCoefficientMap::Add(int attr, real_t value)
{
   Add(attr, std::make_shared<ConstantCoefficient>(value));
}

void DiffusionMassSolver::AttributeCoefficientMap::Add(
   int attr, Coefficient &coefficient, bool transfer_ownership)
{
   Add(attr, ShareCoefficient(coefficient, transfer_ownership));
}

void DiffusionMassSolver::AttributeCoefficientMap::Add(
   int attr, std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(attr > 0, "Attribute ids are one-based and must be positive.");
   MFEM_VERIFY(coefficient != nullptr, "Coefficient pointer is null.");
   coefficients_[attr] = coefficient;
   qf_owners_.erase(attr);
   gf_owners_.erase(attr);
   piecewise_.reset();
   NotifyChanged(attr);
}

void DiffusionMassSolver::AttributeCoefficientMap::Add(
   int attr, QuadratureFunction &qf, bool transfer_ownership)
{
   Add(attr, ShareQuadratureFunction(qf, transfer_ownership));
}

void DiffusionMassSolver::AttributeCoefficientMap::Add(
   int attr, std::shared_ptr<QuadratureFunction> qf)
{
   MFEM_VERIFY(owner_ != nullptr, "Coefficient map is not attached to a solver.");
   MFEM_VERIFY(attr > 0, "Attribute ids are one-based and must be positive.");
   MFEM_VERIFY(qf != nullptr, "QuadratureFunction pointer is null.");
   if (kind_ == MapKind::SurfaceRHS)
   {
      owner_->ValidateSurfaceQuadratureFunction(*qf);
   }
   else
   {
      owner_->ValidateQuadratureFunction(*qf);
   }
   qf_owners_[attr] = qf;
   gf_owners_.erase(attr);
   coefficients_[attr] =
      std::make_shared<QuadratureFunctionCoefficient>(*qf);
   piecewise_.reset();
   NotifyChanged(attr);
}

void DiffusionMassSolver::AttributeCoefficientMap::Add(
   int attr, ParGridFunction &gf, bool transfer_ownership)
{
   Add(attr, ShareParGridFunction(gf, transfer_ownership));
}

void DiffusionMassSolver::AttributeCoefficientMap::Add(
   int attr, std::shared_ptr<ParGridFunction> gf)
{
   MFEM_VERIFY(owner_ != nullptr, "Coefficient map is not attached to a solver.");
   MFEM_VERIFY(attr > 0, "Attribute ids are one-based and must be positive.");
   MFEM_VERIFY(gf != nullptr, "ParGridFunction pointer is null.");
   owner_->ValidateParGridFunction(*gf);
   gf_owners_[attr] = gf;
   qf_owners_.erase(attr);
   coefficients_[attr] = owner_->MakeGridFunctionCoefficient(gf);
   piecewise_.reset();
   NotifyChanged(attr);
}

void DiffusionMassSolver::AttributeCoefficientMap::Clear()
{
   coefficients_.clear();
   qf_owners_.clear();
   gf_owners_.clear();
   piecewise_.reset();
   NotifyChanged(0);
}

Coefficient &DiffusionMassSolver::AttributeCoefficientMap::AsCoefficient() const
{
   piecewise_.reset(new PWCoefficient);
   for (auto &entry : coefficients_)
   {
      piecewise_->UpdateCoefficient(entry.first, *entry.second);
   }
   return *piecewise_;
}

std::shared_ptr<Coefficient>
DiffusionMassSolver::AttributeCoefficientMap::ShareCoefficient(
   Coefficient &coefficient, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<Coefficient>(&coefficient);
   }
   return std::shared_ptr<Coefficient>(&coefficient, [](Coefficient *) { });
}

std::shared_ptr<QuadratureFunction>
DiffusionMassSolver::AttributeCoefficientMap::ShareQuadratureFunction(
   QuadratureFunction &qf, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<QuadratureFunction>(&qf);
   }
   return std::shared_ptr<QuadratureFunction>(&qf,
                                              [](QuadratureFunction *) { });
}

std::shared_ptr<ParGridFunction>
DiffusionMassSolver::AttributeCoefficientMap::ShareParGridFunction(
   ParGridFunction &gf, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<ParGridFunction>(&gf);
   }
   return std::shared_ptr<ParGridFunction>(&gf, [](ParGridFunction *) { });
}

void DiffusionMassSolver::AttributeCoefficientMap::NotifyChanged(int attr)
{
   if (!owner_) { return; }
   if (kind_ == MapKind::Boundary)
   {
      owner_->MarkBoundaryChanged(attr);
   }
   else
   {
      owner_->MarkRHSChanged();
   }
}

DiffusionMassSolver::DiffusionMassSolver(ParFiniteElementSpace &fespace)
   : Solver(fespace.GetTrueVSize()),
     fespace_(fespace),
     diffusion_base_coefficient_(std::make_shared<ConstantCoefficient>(1.0)),
     mass_base_coefficient_(std::make_shared<ConstantCoefficient>(1.0)),
     integration_order_(2*fespace.GetMaxElementOrder())
{
   rhs_coefficients_.SetOwner(this, MapKind::DomainRHS);
   surface_load_coefficients_.SetOwner(this, MapKind::SurfaceRHS);
   boundary_coefficients_.SetOwner(this, MapKind::Boundary);
   RefreshScaledCoefficients();
}

void DiffusionMassSolver::SetDiffusionCoefficient(real_t value)
{
   SetDiffusionCoefficient(std::make_shared<ConstantCoefficient>(value));
}

void DiffusionMassSolver::SetDiffusionCoefficient(
   Coefficient &coefficient, bool transfer_ownership)
{
   diffusion_qf_owner_.reset();
   diffusion_gf_owner_.reset();
   SetDiffusionCoefficient(ShareCoefficient(coefficient, transfer_ownership));
}

void DiffusionMassSolver::SetDiffusionCoefficient(
   std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(coefficient != nullptr, "Diffusion coefficient is null.");
   diffusion_base_coefficient_ = coefficient;
   RefreshScaledCoefficients();
   MarkCoefficientChanged();
}

void DiffusionMassSolver::SetDiffusionCoefficient(
   QuadratureFunction &qf, bool transfer_ownership)
{
   SetDiffusionCoefficient(ShareQuadratureFunction(qf, transfer_ownership));
}

void DiffusionMassSolver::SetDiffusionCoefficient(
   std::shared_ptr<QuadratureFunction> qf)
{
   MFEM_VERIFY(qf != nullptr, "Diffusion QuadratureFunction is null.");
   ValidateQuadratureFunction(*qf);
   diffusion_qf_owner_ = qf;
   diffusion_gf_owner_.reset();
   diffusion_base_coefficient_ = MakeQuadratureCoefficient(qf);
   RefreshScaledCoefficients();
   MarkCoefficientChanged();
}

void DiffusionMassSolver::SetDiffusionCoefficient(
   ParGridFunction &gf, bool transfer_ownership)
{
   SetDiffusionCoefficient(ShareParGridFunction(gf, transfer_ownership));
}

void DiffusionMassSolver::SetDiffusionCoefficient(
   std::shared_ptr<ParGridFunction> gf)
{
   MFEM_VERIFY(gf != nullptr, "Diffusion ParGridFunction is null.");
   ValidateParGridFunction(*gf);
   diffusion_qf_owner_.reset();
   diffusion_gf_owner_ = gf;
   diffusion_base_coefficient_ = MakeGridFunctionCoefficient(gf);
   RefreshScaledCoefficients();
   MarkCoefficientChanged();
}

void DiffusionMassSolver::SetMassCoefficient(real_t value)
{
   SetMassCoefficient(std::make_shared<ConstantCoefficient>(value));
}

void DiffusionMassSolver::SetMassCoefficient(
   Coefficient &coefficient, bool transfer_ownership)
{
   mass_qf_owner_.reset();
   mass_gf_owner_.reset();
   SetMassCoefficient(ShareCoefficient(coefficient, transfer_ownership));
}

void DiffusionMassSolver::SetMassCoefficient(
   std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(coefficient != nullptr, "Mass coefficient is null.");
   mass_base_coefficient_ = coefficient;
   RefreshScaledCoefficients();
   MarkCoefficientChanged();
}

void DiffusionMassSolver::SetMassCoefficient(
   ParGridFunction &gf, bool transfer_ownership)
{
   SetMassCoefficient(ShareParGridFunction(gf, transfer_ownership));
}

void DiffusionMassSolver::SetMassCoefficient(
   std::shared_ptr<ParGridFunction> gf)
{
   MFEM_VERIFY(gf != nullptr, "Mass ParGridFunction is null.");
   ValidateParGridFunction(*gf);
   mass_qf_owner_.reset();
   mass_gf_owner_ = gf;
   mass_base_coefficient_ = MakeGridFunctionCoefficient(gf);
   RefreshScaledCoefficients();
   MarkCoefficientChanged();
}

void DiffusionMassSolver::SetMassCoefficient(
   QuadratureFunction &qf, bool transfer_ownership)
{
   SetMassCoefficient(ShareQuadratureFunction(qf, transfer_ownership));
}

void DiffusionMassSolver::SetMassCoefficient(
   std::shared_ptr<QuadratureFunction> qf)
{
   MFEM_VERIFY(qf != nullptr, "Mass QuadratureFunction is null.");
   ValidateQuadratureFunction(*qf);
   mass_qf_owner_ = qf;
   mass_gf_owner_.reset();
   mass_base_coefficient_ = MakeQuadratureCoefficient(qf);
   RefreshScaledCoefficients();
   MarkCoefficientChanged();
}

void DiffusionMassSolver::SetScalingConstants(real_t diffusion_scale,
                                              real_t mass_scale)
{
   MFEM_VERIFY(diffusion_scale >= 0.0,
               "Diffusion scaling constant must be nonnegative.");
   MFEM_VERIFY(mass_scale >= 0.0,
               "Mass scaling constant must be nonnegative.");
   if (diffusion_scale_ == diffusion_scale && mass_scale_ == mass_scale)
   {
      return;
   }

   diffusion_scale_ = diffusion_scale;
   mass_scale_ = mass_scale;
   if (scaled_diffusion_coefficient_)
   {
      scaled_diffusion_coefficient_->SetAConst(diffusion_scale_);
   }
   if (scaled_mass_coefficient_)
   {
      scaled_mass_coefficient_->SetAConst(mass_scale_);
   }
   MarkCoefficientChanged();
}

void DiffusionMassSolver::AddBoundaryID(int id)
{
   MFEM_VERIFY(id > 0, "Boundary ids are one-based and must be positive.");
   MarkBoundaryChanged(id);
}

void DiffusionMassSolver::ClearBoundaryConditions()
{
   boundary_ids_.clear();
   boundary_coefficients_.Clear();
   MarkCoefficientChanged();
}

void DiffusionMassSolver::SetNeedsAssembly(bool needs_assembly) const
{
   needs_assembly_ = needs_assembly;
   if (needs_assembly)
   {
      rhs_form_dirty_ = true;
      rhs_vector_dirty_ = true;
      boundary_true_dofs_dirty_ = true;
   }
}

void DiffusionMassSolver::SetRelTol(real_t rel_tol)
{
   MFEM_VERIFY(rel_tol >= 0.0, "Relative tolerance must be nonnegative.");
   rel_tol_ = rel_tol;
   needs_assembly_ = true;
}

void DiffusionMassSolver::SetAbsTol(real_t abs_tol)
{
   MFEM_VERIFY(abs_tol >= 0.0, "Absolute tolerance must be nonnegative.");
   abs_tol_ = abs_tol;
   needs_assembly_ = true;
}

void DiffusionMassSolver::SetMaxIter(int max_iter)
{
   MFEM_VERIFY(max_iter > 0, "Maximum iteration count must be positive.");
   max_iter_ = max_iter;
   needs_assembly_ = true;
}

const IntegrationRule &DiffusionMassSolver::GetIntegrationRule(
   Geometry::Type geom) const
{
   return IntRules.Get(geom, integration_order_);
}

void DiffusionMassSolver::Assemble() const
{
   if (!GlobalBooleanOr(fespace_.GetComm(), needs_assembly_)) { return; }

   cg_solver_.reset();
   solve_operator_.Clear();
   system_operator_.Clear();
   BuildEssentialTrueDofs();

   form_.reset(new ParBilinearForm(&fespace_));
   form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form_->AddDomainIntegrator(new DiffusionIntegrator(
                                 *diffusion_coefficient_,
                                 &GetIntegrationRule(
                                    fespace_.GetParMesh()->GetElementGeometry(0))));
   form_->AddDomainIntegrator(new MassIntegrator(
                                 *mass_coefficient_,
                                 &GetIntegrationRule(
                                    fespace_.GetParMesh()->GetElementGeometry(0))));
   form_->Assemble();

   system_operator_.SetType(Operator::ANY_TYPE);
   form_->FormSystemMatrix(ess_tdofs_, system_operator_);

   DiffusionMassSolver *self = const_cast<DiffusionMassSolver *>(this);
   self->height = system_operator_->Height();
   self->width = system_operator_->Width();
   BuildPreconditioner();

   cg_solver_.reset(new CGSolver(fespace_.GetComm()));
   cg_solver_->SetRelTol(rel_tol_);
   cg_solver_->SetAbsTol(abs_tol_);
   cg_solver_->SetMaxIter(max_iter_ > 0 ? max_iter_ :
                          std::max(200, 2*Height()));
   cg_solver_->SetPrintLevel(print_level_);
   cg_solver_->SetOperator(*system_operator_);
   if (preconditioner_)
   {
      cg_solver_->SetPreconditioner(*preconditioner_);
   }

   needs_assembly_ = false;
}

void DiffusionMassSolver::Mult(const Vector &rhs, Vector &solution) const
{
   Assemble();
   MultAssembled(rhs, solution);
}

void DiffusionMassSolver::MultAssembled(const Vector &rhs,
                                        Vector &solution) const
{
   MFEM_VERIFY(system_operator_.Ptr() != nullptr,
               "DiffusionMassSolver has not been assembled.");
   MFEM_VERIFY(rhs.Size() == Width(), "RHS has incompatible size.");

   SolveSystem(rhs, solution, true, true);
}

void DiffusionMassSolver::MultTranspose(const Vector &rhs,
                                        Vector &solution) const
{
   Assemble();
   MultTransposeAssembled(rhs, solution);
}

void DiffusionMassSolver::MultTransposeAssembled(const Vector &rhs,
                                                 Vector &solution) const
{
   MFEM_VERIFY(system_operator_.Ptr() != nullptr,
               "DiffusionMassSolver has not been assembled.");
   MFEM_VERIFY(rhs.Size() == Height(), "RHS has incompatible size.");

   SolveSystem(rhs, solution, true, false);
}

void DiffusionMassSolver::Solve(ParGridFunction &solution) const
{
   Assemble();
   MFEM_VERIFY(solution.ParFESpace() == &fespace_,
               "Solution grid function must use the solver finite element space.");

   solution = 0.0;
   const bool has_boundary_data =
      GlobalBooleanOr(fespace_.GetComm(),
                      !boundary_coefficients_.Empty() &&
                      !boundary_ids_.empty());
   if (has_boundary_data)
   {
      BuildEssentialMarker(boundary_marker_);
      solution.ProjectBdrCoefficient(boundary_coefficients_.AsCoefficient(),
                                     boundary_marker_);
   }

   if (!rhs_form_ || rhs_form_dirty_)
   {
      rhs_form_.reset(new ParLinearForm(&fespace_));
      if (!rhs_coefficients_.Empty())
      {
         rhs_form_->AddDomainIntegrator(new DomainLFIntegrator(
                                           rhs_coefficients_.AsCoefficient(),
                                           &GetIntegrationRule(
                                              fespace_.GetParMesh()
                                              ->GetElementGeometry(0))));
      }
      if (!surface_load_coefficients_.Empty())
      {
         rhs_form_->AddBoundaryIntegrator(new BoundaryLFIntegrator(
                                             surface_load_coefficients_
                                             .AsCoefficient(), 2, 0));
      }
      rhs_form_dirty_ = false;
      rhs_vector_dirty_ = true;
   }
   if (rhs_vector_dirty_)
   {
      *rhs_form_ = 0.0;
      rhs_form_->Assemble();
      rhs_vector_dirty_ = false;
   }

   form_->FormLinearSystem(ess_tdofs_, solution, *rhs_form_, solve_operator_,
                           solve_X_, solve_B_);

   SolveSystem(solve_B_, solve_Y_, false, false);
   solve_X_ = solve_Y_;

   form_->RecoverFEMSolution(solve_X_, *rhs_form_, solution);
}

void DiffusionMassSolver::SetOperator(const Operator &op)
{
   Assemble();
   MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
               "External operator dimensions do not match DiffusionMassSolver.");
}

const Operator *DiffusionMassSolver::GetOperator() const
{
   Assemble();
   return system_operator_.Ptr();
}

const Solver *DiffusionMassSolver::GetPreconditioner() const
{
   Assemble();
   return preconditioner_.get();
}

std::shared_ptr<Coefficient> DiffusionMassSolver::ShareCoefficient(
   Coefficient &coefficient, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<Coefficient>(&coefficient);
   }
   return std::shared_ptr<Coefficient>(&coefficient, [](Coefficient *) { });
}

std::shared_ptr<QuadratureFunction> DiffusionMassSolver::ShareQuadratureFunction(
   QuadratureFunction &qf, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<QuadratureFunction>(&qf);
   }
   return std::shared_ptr<QuadratureFunction>(&qf,
                                              [](QuadratureFunction *) { });
}

std::shared_ptr<ParGridFunction> DiffusionMassSolver::ShareParGridFunction(
   ParGridFunction &gf, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<ParGridFunction>(&gf);
   }
   return std::shared_ptr<ParGridFunction>(&gf, [](ParGridFunction *) { });
}

std::shared_ptr<Coefficient> DiffusionMassSolver::MakeQuadratureCoefficient(
   std::shared_ptr<QuadratureFunction> qf) const
{
   ValidateQuadratureFunction(*qf);
   return std::make_shared<QuadratureFunctionCoefficient>(*qf);
}

std::shared_ptr<Coefficient> DiffusionMassSolver::MakeGridFunctionCoefficient(
   std::shared_ptr<ParGridFunction> gf) const
{
   ValidateParGridFunction(*gf);
   return std::make_shared<GridFunctionCoefficient>(gf.get());
}

void DiffusionMassSolver::ValidateQuadratureFunction(
   const QuadratureFunction &qf) const
{
   MFEM_VERIFY(qf.GetVDim() == 1, "Expected scalar QuadratureFunction.");
   MFEM_VERIFY(qf.GetSpace() != nullptr, "QuadratureFunction has no space.");
   MFEM_VERIFY(qf.GetSpace()->GetMesh() == fespace_.GetParMesh(),
               "QuadratureFunction must be defined on the solver mesh.");
   MFEM_VERIFY(qf.GetSpace()->GetOrder() == integration_order_,
               "QuadratureFunction integration order does not match solver.");
   MFEM_VERIFY(qf.Size() == qf.GetVDim()*qf.GetSpace()->GetSize(),
               "QuadratureFunction size is incompatible with its space.");

   for (int e = 0; e < fespace_.GetParMesh()->GetNE(); e++)
   {
      const IntegrationRule &solver_ir =
         GetIntegrationRule(fespace_.GetParMesh()->GetElementGeometry(e));
      const IntegrationRule &qf_ir = qf.GetIntRule(e);
      MFEM_VERIFY(qf_ir.Size() == solver_ir.Size() &&
                  qf_ir.GetOrder() == solver_ir.GetOrder(),
                  "QuadratureFunction rule does not match solver rule.");
   }
}

void DiffusionMassSolver::ValidateSurfaceQuadratureFunction(
   const QuadratureFunction &qf) const
{
   MFEM_VERIFY(qf.GetVDim() == 1, "Expected scalar QuadratureFunction.");
   MFEM_VERIFY(qf.GetSpace() != nullptr, "QuadratureFunction has no space.");
   MFEM_VERIFY(qf.GetSpace()->GetMesh() == fespace_.GetParMesh(),
               "QuadratureFunction must be defined on the solver mesh.");
   const auto *face_space =
      dynamic_cast<const FaceQuadratureSpace *>(qf.GetSpace());
   MFEM_VERIFY(face_space != nullptr,
               "Surface-load QuadratureFunction must use a "
               "FaceQuadratureSpace.");
   MFEM_VERIFY(face_space->GetFaceType() == FaceType::Boundary,
               "Surface-load QuadratureFunction must use boundary faces.");
   MFEM_VERIFY(qf.GetSpace()->GetOrder() == integration_order_,
               "Surface-load QuadratureFunction integration order does not "
               "match solver boundary integration order.");
   MFEM_VERIFY(qf.Size() == qf.GetVDim()*qf.GetSpace()->GetSize(),
               "QuadratureFunction size is incompatible with its space.");
}

void DiffusionMassSolver::ValidateParGridFunction(const ParGridFunction &gf) const
{
   MFEM_VERIFY(gf.ParFESpace() == &fespace_,
               "ParGridFunction coefficient must use the solver FE space.");
   MFEM_VERIFY(gf.Size() == fespace_.GetVSize(),
               "ParGridFunction coefficient has incompatible local size.");
}

void DiffusionMassSolver::SolveSystem(const Vector &rhs, Vector &solution,
                                      bool apply_constraints,
                                      bool use_boundary_values) const
{
   constrained_rhs_.SetSize(rhs.Size());
   constrained_rhs_ = rhs;
   const Vector *constraint_values = nullptr;
   if (apply_constraints)
   {
      if (use_boundary_values)
      {
         BuildBoundaryTrueDofs(boundary_true_dofs_);
         constraint_values = &boundary_true_dofs_;
      }
      else
      {
         homogeneous_boundary_true_dofs_.SetSize(ess_tdofs_.Size());
         homogeneous_boundary_true_dofs_ = 0.0;
         constraint_values = &homogeneous_boundary_true_dofs_;
      }
      constrained_rhs_.SetSubVector(ess_tdofs_, *constraint_values);
   }

   solution.SetSize(Height());
   solution = 0.0;

   MFEM_VERIFY(cg_solver_ != nullptr,
               "DiffusionMassSolver linear solver has not been configured.");
   cg_solver_->Mult(constrained_rhs_, solution);

   if (apply_constraints)
   {
      solution.SetSubVector(ess_tdofs_, *constraint_values);
   }
}

void DiffusionMassSolver::BuildBoundaryTrueDofs(Vector &boundary_values) const
{
   const bool rebuild_boundary_values =
      GlobalBooleanOr(fespace_.GetComm(),
                      boundary_true_dofs_dirty_ ||
                      boundary_values.Size() != ess_tdofs_.Size());
   if (!rebuild_boundary_values)
   {
      return;
   }

   boundary_values.SetSize(ess_tdofs_.Size());
   boundary_values = 0.0;
   const bool has_boundary_data =
      GlobalBooleanOr(fespace_.GetComm(),
                      !boundary_coefficients_.Empty() &&
                      !boundary_ids_.empty());
   if (!has_boundary_data)
   {
      boundary_true_dofs_dirty_ = false;
      return;
   }

   if (!boundary_grid_function_ ||
       boundary_grid_function_->ParFESpace() != &fespace_)
   {
      boundary_grid_function_.reset(new ParGridFunction(&fespace_));
   }

   *boundary_grid_function_ = 0.0;
   BuildEssentialMarker(boundary_marker_);
   boundary_grid_function_->ProjectBdrCoefficient(
      boundary_coefficients_.AsCoefficient(), boundary_marker_);

   boundary_grid_function_->GetTrueDofs(boundary_all_true_dofs_);
   boundary_all_true_dofs_.GetSubVector(ess_tdofs_, boundary_values);
   boundary_true_dofs_dirty_ = false;
}

void DiffusionMassSolver::MarkCoefficientChanged()
{
   needs_assembly_ = true;
}

void DiffusionMassSolver::MarkRHSChanged()
{
   needs_assembly_ = true;
   rhs_form_dirty_ = true;
   rhs_vector_dirty_ = true;
}

void DiffusionMassSolver::MarkBoundaryChanged(int id)
{
   if (id > 0) { boundary_ids_.insert(id); }
   needs_assembly_ = true;
   boundary_true_dofs_dirty_ = true;
}

void DiffusionMassSolver::BuildEssentialMarker(Array<int> &marker) const
{
   int max_attr = fespace_.GetParMesh()->bdr_attributes.Size()
                ? fespace_.GetParMesh()->bdr_attributes.Max() : 0;
   for (int id : boundary_ids_)
   {
      max_attr = std::max(max_attr, id);
   }
   marker.SetSize(max_attr);
   marker = 0;
   for (int id : boundary_ids_)
   {
      marker[id - 1] = 1;
   }
}

void DiffusionMassSolver::BuildEssentialTrueDofs() const
{
   ess_tdofs_.SetSize(0);
   if (GlobalBooleanOr(fespace_.GetComm(), !boundary_ids_.empty()))
   {
      BuildEssentialMarker(boundary_marker_);
      fespace_.GetEssentialTrueDofs(boundary_marker_, ess_tdofs_);
   }
   boundary_true_dofs_dirty_ = true;
}

void DiffusionMassSolver::BuildPreconditioner() const
{
   preconditioner_.reset();
   lor_operator_.Clear();
   lor_form_.reset();
   lor_diffusion_coefficient_.reset();
   lor_mass_coefficient_.reset();
   lor_diffusion_base_coefficient_.reset();
   lor_mass_base_coefficient_.reset();
   lor_diffusion_gf_.reset();
   lor_mass_gf_.reset();
   lor_fespace_.reset();
   lor_fec_.reset();
   lor_mesh_.reset();
   assembled_operator_.Clear();
   assembled_form_.reset();

   if (fespace_.GetMaxElementOrder() > 1)
   {
      BuildLORAMGPreconditioner();
   }
   else
   {
      BuildAMGPreconditionerOnFESpace();
   }
}

void DiffusionMassSolver::BuildAMGPreconditionerOnFESpace() const
{
   assembled_form_.reset(new ParBilinearForm(&fespace_));
   assembled_form_->AddDomainIntegrator(new DiffusionIntegrator(
                                           *diffusion_coefficient_,
                                           &GetIntegrationRule(
                                              fespace_.GetParMesh()
                                              ->GetElementGeometry(0))));
   assembled_form_->AddDomainIntegrator(new MassIntegrator(
                                           *mass_coefficient_,
                                           &GetIntegrationRule(
                                              fespace_.GetParMesh()
                                              ->GetElementGeometry(0))));
   assembled_form_->Assemble();

   assembled_operator_.SetType(Operator::Hypre_ParCSR);
   assembled_form_->FormSystemMatrix(ess_tdofs_, assembled_operator_);
   HypreParMatrix *matrix = assembled_operator_.Is<HypreParMatrix>();
   MFEM_VERIFY(matrix != nullptr, "Assembled operator is not a HypreParMatrix.");

   HypreBoomerAMG *amg = new HypreBoomerAMG(*matrix);
   amg->SetPrintLevel(print_level_);
   preconditioner_.reset(amg);
}

void DiffusionMassSolver::BuildLORAMGPreconditioner() const
{
   const int order = fespace_.GetMaxElementOrder();
   ParMesh *pmesh = fespace_.GetParMesh();

   MFEM_VERIFY((pmesh->MeshGenerator() & 1) == 0,
               "LOR+AMG preconditioning requires tensor-product elements.");

   lor_mesh_.reset(new ParMesh);
   *lor_mesh_ = ParMesh::MakeRefined(*pmesh, order, BasisType::GaussLobatto);
   lor_fec_.reset(new H1_FECollection(1, pmesh->Dimension()));
   lor_fespace_.reset(new ParFiniteElementSpace(lor_mesh_.get(),
                                                lor_fec_.get()));

   MFEM_VERIFY(lor_fespace_->GetTrueVSize() == fespace_.GetTrueVSize(),
               "LOR true-vector size does not match the high-order space. "
               "Use a compatible tensor H1 space, e.g. Gauss-Lobatto basis.");

   lor_diffusion_coefficient_ = MakeLORCoefficient(diffusion_coefficient_,
                                                   diffusion_base_coefficient_,
                                                   diffusion_gf_owner_,
                                                   diffusion_qf_owner_,
                                                   lor_diffusion_gf_,
                                                   diffusion_scale_,
                                                   lor_diffusion_base_coefficient_);
   lor_mass_coefficient_ = MakeLORCoefficient(mass_coefficient_,
                                              mass_base_coefficient_,
                                              mass_gf_owner_,
                                              mass_qf_owner_,
                                              lor_mass_gf_,
                                              mass_scale_,
                                              lor_mass_base_coefficient_);

   lor_form_.reset(new ParBilinearForm(lor_fespace_.get()));
   lor_form_->AddDomainIntegrator(new DiffusionIntegrator(
                                    *lor_diffusion_coefficient_,
                                    &GetIntegrationRule(
                                       pmesh->GetElementGeometry(0))));
   lor_form_->AddDomainIntegrator(new MassIntegrator(
                                    *lor_mass_coefficient_,
                                    &GetIntegrationRule(
                                       pmesh->GetElementGeometry(0))));
   lor_form_->Assemble();

   lor_operator_.SetType(Operator::Hypre_ParCSR);
   lor_form_->FormSystemMatrix(ess_tdofs_, lor_operator_);
   HypreParMatrix *lor_matrix = lor_operator_.Is<HypreParMatrix>();
   MFEM_VERIFY(lor_matrix != nullptr, "LOR operator is not a HypreParMatrix.");

   HypreBoomerAMG *amg = new HypreBoomerAMG(*lor_matrix);
   amg->SetPrintLevel(print_level_);
   preconditioner_.reset(amg);
}

std::shared_ptr<Coefficient> DiffusionMassSolver::MakeLORCoefficient(
   const std::shared_ptr<Coefficient> &coefficient,
   const std::shared_ptr<Coefficient> &base_coefficient,
   const std::shared_ptr<ParGridFunction> &ho_gf,
   const std::shared_ptr<QuadratureFunction> &qf,
   std::shared_ptr<ParGridFunction> &lor_gf,
   real_t scale,
   std::shared_ptr<Coefficient> &lor_base_coefficient) const
{
   if (!ho_gf && !qf)
   {
      return coefficient;
   }

   MFEM_VERIFY(lor_fespace_,
               "LOR finite element space must be constructed before transfer.");

   if (ho_gf)
   {
      MFEM_VERIFY(lor_fespace_->GetTrueVSize() ==
                  ho_gf->ParFESpace()->GetTrueVSize(),
                  "Cannot transfer ParGridFunction coefficient to LOR space: "
                  "true-vector sizes differ.");
      ho_gf->GetTrueDofs(lor_transfer_true_dofs_);
   }
   else
   {
      // TODO: ProjectCoefficient() evaluates base_coefficient at the FE's
      // nodal integration points, but QuadratureFunctionCoefficient::Eval
      // indexes QuadF by the passed IntegrationPoint's index within the
      // QuadratureFunction's own quadrature rule. The two indexings only
      // coincide by accident, so this silently misindexes (or, if the FE has
      // more nodes per element than the QuadratureSpace has points, reads out
      // of bounds of) a spatially-varying QuadratureFunction. It only works
      // today because callers use spatially-constant QuadratureFunctions.
      ParGridFunction projected(&fespace_);
      projected.ProjectCoefficient(*base_coefficient);
      MFEM_VERIFY(lor_fespace_->GetTrueVSize() == fespace_.GetTrueVSize(),
                  "Cannot transfer QuadratureFunction coefficient to LOR "
                  "space: true-vector sizes differ.");
      projected.GetTrueDofs(lor_transfer_true_dofs_);
   }

   lor_gf = std::make_shared<ParGridFunction>(lor_fespace_.get());
   lor_gf->SetFromTrueDofs(lor_transfer_true_dofs_);
   lor_base_coefficient =
      std::make_shared<GridFunctionCoefficient>(lor_gf.get());
   return std::make_shared<ProductCoefficient>(scale,
                                               *lor_base_coefficient);
}

void DiffusionMassSolver::RefreshScaledCoefficients()
{
   MFEM_VERIFY(diffusion_base_coefficient_ != nullptr,
               "Diffusion base coefficient is null.");
   MFEM_VERIFY(mass_base_coefficient_ != nullptr,
               "Mass base coefficient is null.");

   scaled_diffusion_coefficient_ =
      std::make_shared<ProductCoefficient>(diffusion_scale_,
                                           *diffusion_base_coefficient_);
   scaled_mass_coefficient_ =
      std::make_shared<ProductCoefficient>(mass_scale_,
                                           *mass_base_coefficient_);
   diffusion_coefficient_ = scaled_diffusion_coefficient_;
   mass_coefficient_ = scaled_mass_coefficient_;
}

void DiffusionSolver::AttributeCoefficientMap::SetOwner(
   DiffusionSolver *owner, MapKind kind)
{
   owner_ = owner;
   kind_ = kind;
}

void DiffusionSolver::AttributeCoefficientMap::Add(int attr, real_t value)
{
   Add(attr, std::make_shared<ConstantCoefficient>(value));
}

void DiffusionSolver::AttributeCoefficientMap::Add(
   int attr, Coefficient &coefficient, bool transfer_ownership)
{
   Add(attr, ShareCoefficient(coefficient, transfer_ownership));
}

void DiffusionSolver::AttributeCoefficientMap::Add(
   int attr, std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(attr > 0, "Attribute ids are one-based and must be positive.");
   MFEM_VERIFY(coefficient != nullptr, "Coefficient pointer is null.");
   coefficients_[attr] = coefficient;
   qf_owners_.erase(attr);
   gf_owners_.erase(attr);
   piecewise_.reset();
   NotifyChanged(attr);
}

void DiffusionSolver::AttributeCoefficientMap::Add(
   int attr, QuadratureFunction &qf, bool transfer_ownership)
{
   Add(attr, ShareQuadratureFunction(qf, transfer_ownership));
}

void DiffusionSolver::AttributeCoefficientMap::Add(
   int attr, std::shared_ptr<QuadratureFunction> qf)
{
   MFEM_VERIFY(owner_ != nullptr, "Coefficient map is not attached to a solver.");
   MFEM_VERIFY(attr > 0, "Attribute ids are one-based and must be positive.");
   MFEM_VERIFY(qf != nullptr, "QuadratureFunction pointer is null.");
   if (kind_ == MapKind::SurfaceRHS)
   {
      owner_->ValidateSurfaceQuadratureFunction(*qf);
   }
   else
   {
      owner_->ValidateQuadratureFunction(*qf);
   }
   qf_owners_[attr] = qf;
   gf_owners_.erase(attr);
   coefficients_[attr] =
      std::make_shared<QuadratureFunctionCoefficient>(*qf);
   piecewise_.reset();
   NotifyChanged(attr);
}

void DiffusionSolver::AttributeCoefficientMap::Add(
   int attr, ParGridFunction &gf, bool transfer_ownership)
{
   Add(attr, ShareParGridFunction(gf, transfer_ownership));
}

void DiffusionSolver::AttributeCoefficientMap::Add(
   int attr, std::shared_ptr<ParGridFunction> gf)
{
   MFEM_VERIFY(owner_ != nullptr, "Coefficient map is not attached to a solver.");
   MFEM_VERIFY(attr > 0, "Attribute ids are one-based and must be positive.");
   MFEM_VERIFY(gf != nullptr, "ParGridFunction pointer is null.");
   owner_->ValidateParGridFunction(*gf);
   gf_owners_[attr] = gf;
   qf_owners_.erase(attr);
   coefficients_[attr] = owner_->MakeGridFunctionCoefficient(gf);
   piecewise_.reset();
   NotifyChanged(attr);
}

void DiffusionSolver::AttributeCoefficientMap::Clear()
{
   coefficients_.clear();
   qf_owners_.clear();
   gf_owners_.clear();
   piecewise_.reset();
   NotifyChanged(0);
}

Coefficient &DiffusionSolver::AttributeCoefficientMap::AsCoefficient() const
{
   piecewise_.reset(new PWCoefficient);
   for (auto &entry : coefficients_)
   {
      piecewise_->UpdateCoefficient(entry.first, *entry.second);
   }
   return *piecewise_;
}

std::shared_ptr<Coefficient>
DiffusionSolver::AttributeCoefficientMap::ShareCoefficient(
   Coefficient &coefficient, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<Coefficient>(&coefficient);
   }
   return std::shared_ptr<Coefficient>(&coefficient, [](Coefficient *) { });
}

std::shared_ptr<QuadratureFunction>
DiffusionSolver::AttributeCoefficientMap::ShareQuadratureFunction(
   QuadratureFunction &qf, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<QuadratureFunction>(&qf);
   }
   return std::shared_ptr<QuadratureFunction>(&qf,
                                              [](QuadratureFunction *) { });
}

std::shared_ptr<ParGridFunction>
DiffusionSolver::AttributeCoefficientMap::ShareParGridFunction(
   ParGridFunction &gf, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<ParGridFunction>(&gf);
   }
   return std::shared_ptr<ParGridFunction>(&gf, [](ParGridFunction *) { });
}

void DiffusionSolver::AttributeCoefficientMap::NotifyChanged(int attr)
{
   if (!owner_) { return; }
   if (kind_ == MapKind::Boundary)
   {
      owner_->MarkBoundaryChanged(attr);
   }
   else
   {
      owner_->MarkRHSChanged();
   }
}

DiffusionSolver::DiffusionSolver(ParFiniteElementSpace &fespace)
   : Solver(fespace.GetTrueVSize()),
     fespace_(fespace),
     diffusion_coefficient_(std::make_shared<ConstantCoefficient>(1.0)),
     integration_order_(2*fespace.GetMaxElementOrder())
{
   rhs_coefficients_.SetOwner(this, MapKind::DomainRHS);
   surface_load_coefficients_.SetOwner(this, MapKind::SurfaceRHS);
   boundary_coefficients_.SetOwner(this, MapKind::Boundary);
}

DiffusionSolver::DiffusionSolver(
   std::shared_ptr<ParFiniteElementSpace> fespace)
   : Solver(CheckedFESpace(fespace, "Diffusion").GetTrueVSize()),
     fespace_owner_(fespace),
     fespace_(CheckedFESpace(fespace, "Diffusion")),
     diffusion_coefficient_(std::make_shared<ConstantCoefficient>(1.0)),
     integration_order_(2*fespace_.GetMaxElementOrder())
{
   rhs_coefficients_.SetOwner(this, MapKind::DomainRHS);
   surface_load_coefficients_.SetOwner(this, MapKind::SurfaceRHS);
   boundary_coefficients_.SetOwner(this, MapKind::Boundary);
}

void DiffusionSolver::SetDiffusionCoefficient(real_t value)
{
   SetDiffusionCoefficient(std::make_shared<ConstantCoefficient>(value));
}

void DiffusionSolver::SetDiffusionCoefficient(
   Coefficient &coefficient, bool transfer_ownership)
{
   diffusion_qf_owner_.reset();
   diffusion_gf_owner_.reset();
   SetDiffusionCoefficient(ShareCoefficient(coefficient, transfer_ownership));
}

void DiffusionSolver::SetDiffusionCoefficient(
   std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(coefficient != nullptr, "Diffusion coefficient is null.");
   diffusion_coefficient_ = coefficient;
   MarkCoefficientChanged();
}

void DiffusionSolver::SetDiffusionCoefficient(
   QuadratureFunction &qf, bool transfer_ownership)
{
   SetDiffusionCoefficient(ShareQuadratureFunction(qf, transfer_ownership));
}

void DiffusionSolver::SetDiffusionCoefficient(
   std::shared_ptr<QuadratureFunction> qf)
{
   MFEM_VERIFY(qf != nullptr, "Diffusion QuadratureFunction is null.");
   ValidateQuadratureFunction(*qf);
   diffusion_qf_owner_ = qf;
   diffusion_gf_owner_.reset();
   diffusion_coefficient_ = MakeQuadratureCoefficient(qf);
   MarkCoefficientChanged();
}

void DiffusionSolver::SetDiffusionCoefficient(
   ParGridFunction &gf, bool transfer_ownership)
{
   SetDiffusionCoefficient(ShareParGridFunction(gf, transfer_ownership));
}

void DiffusionSolver::SetDiffusionCoefficient(
   std::shared_ptr<ParGridFunction> gf)
{
   MFEM_VERIFY(gf != nullptr, "Diffusion ParGridFunction is null.");
   ValidateParGridFunction(*gf);
   diffusion_qf_owner_.reset();
   diffusion_gf_owner_ = gf;
   diffusion_coefficient_ = MakeGridFunctionCoefficient(gf);
   MarkCoefficientChanged();
}

void DiffusionSolver::AddBoundaryID(int id)
{
   MFEM_VERIFY(id > 0, "Boundary ids are one-based and must be positive.");
   MarkBoundaryChanged(id);
}

void DiffusionSolver::ClearBoundaryConditions()
{
   boundary_ids_.clear();
   boundary_coefficients_.Clear();
   MarkCoefficientChanged();
}

void DiffusionSolver::SetNeedsAssembly(bool needs_assembly) const
{
   needs_assembly_ = needs_assembly;
   if (needs_assembly)
   {
      rhs_form_dirty_ = true;
      rhs_vector_dirty_ = true;
      boundary_true_dofs_dirty_ = true;
   }
}

void DiffusionSolver::SetRelTol(real_t rel_tol)
{
   MFEM_VERIFY(rel_tol >= 0.0, "Relative tolerance must be nonnegative.");
   rel_tol_ = rel_tol;
   needs_assembly_ = true;
}

void DiffusionSolver::SetAbsTol(real_t abs_tol)
{
   MFEM_VERIFY(abs_tol >= 0.0, "Absolute tolerance must be nonnegative.");
   abs_tol_ = abs_tol;
   needs_assembly_ = true;
}

void DiffusionSolver::SetMaxIter(int max_iter)
{
   MFEM_VERIFY(max_iter > 0, "Maximum iteration count must be positive.");
   max_iter_ = max_iter;
   needs_assembly_ = true;
}

const IntegrationRule &DiffusionSolver::GetIntegrationRule(
   Geometry::Type geom) const
{
   return IntRules.Get(geom, integration_order_);
}

void DiffusionSolver::Assemble() const
{
   if (!GlobalBooleanOr(fespace_.GetComm(), needs_assembly_)) { return; }

   cg_solver_.reset();
   solve_operator_.Clear();
   system_operator_.Clear();
   BuildEssentialTrueDofs();

   form_.reset(new ParBilinearForm(&fespace_));
   form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form_->AddDomainIntegrator(new DiffusionIntegrator(
                                 *diffusion_coefficient_,
                                 &GetIntegrationRule(
                                    fespace_.GetParMesh()->GetElementGeometry(0))));
   form_->Assemble();

   system_operator_.SetType(Operator::ANY_TYPE);
   form_->FormSystemMatrix(ess_tdofs_, system_operator_);

   DiffusionSolver *self = const_cast<DiffusionSolver *>(this);
   self->height = system_operator_->Height();
   self->width = system_operator_->Width();
   BuildPreconditioner();

   cg_solver_.reset(new CGSolver(fespace_.GetComm()));
   cg_solver_->SetRelTol(rel_tol_);
   cg_solver_->SetAbsTol(abs_tol_);
   cg_solver_->SetMaxIter(max_iter_ > 0 ? max_iter_ :
                          std::max(200, 2*Height()));
   cg_solver_->SetPrintLevel(print_level_);
   cg_solver_->SetOperator(*system_operator_);
   if (preconditioner_)
   {
      cg_solver_->SetPreconditioner(*preconditioner_);
   }

   needs_assembly_ = false;
}

void DiffusionSolver::Mult(const Vector &rhs, Vector &solution) const
{
   Assemble();
   MultAssembled(rhs, solution);
}

void DiffusionSolver::MultAssembled(const Vector &rhs,
                                    Vector &solution) const
{
   MFEM_VERIFY(system_operator_.Ptr() != nullptr,
               "DiffusionSolver has not been assembled.");
   MFEM_VERIFY(rhs.Size() == Width(), "RHS has incompatible size.");

   SolveSystem(rhs, solution, true, true);
}

void DiffusionSolver::MultTranspose(const Vector &rhs,
                                    Vector &solution) const
{
   Assemble();
   MultTransposeAssembled(rhs, solution);
}

void DiffusionSolver::MultTransposeAssembled(const Vector &rhs,
                                             Vector &solution) const
{
   MFEM_VERIFY(system_operator_.Ptr() != nullptr,
               "DiffusionSolver has not been assembled.");
   MFEM_VERIFY(rhs.Size() == Height(), "RHS has incompatible size.");

   SolveSystem(rhs, solution, true, false);
}

void DiffusionSolver::Solve(ParGridFunction &solution) const
{
   Assemble();
   MFEM_VERIFY(solution.ParFESpace() == &fespace_,
               "Solution grid function must use the solver finite element space.");

   solution = 0.0;
   const bool has_boundary_data =
      GlobalBooleanOr(fespace_.GetComm(),
                      !boundary_coefficients_.Empty() &&
                      !boundary_ids_.empty());
   if (has_boundary_data)
   {
      BuildEssentialMarker(boundary_marker_);
      solution.ProjectBdrCoefficient(boundary_coefficients_.AsCoefficient(),
                                     boundary_marker_);
   }

   if (!rhs_form_ || rhs_form_dirty_)
   {
      rhs_form_.reset(new ParLinearForm(&fespace_));
      if (!rhs_coefficients_.Empty())
      {
         rhs_form_->AddDomainIntegrator(new DomainLFIntegrator(
                                           rhs_coefficients_.AsCoefficient(),
                                           &GetIntegrationRule(
                                              fespace_.GetParMesh()
                                              ->GetElementGeometry(0))));
      }
      if (!surface_load_coefficients_.Empty())
      {
         rhs_form_->AddBoundaryIntegrator(new BoundaryLFIntegrator(
                                             surface_load_coefficients_
                                             .AsCoefficient(), 2, 0));
      }
      rhs_form_dirty_ = false;
      rhs_vector_dirty_ = true;
   }
   if (rhs_vector_dirty_)
   {
      *rhs_form_ = 0.0;
      rhs_form_->Assemble();
      rhs_vector_dirty_ = false;
   }

   form_->FormLinearSystem(ess_tdofs_, solution, *rhs_form_, solve_operator_,
                           solve_X_, solve_B_);

   SolveSystem(solve_B_, solve_Y_, false, false);
   solve_X_ = solve_Y_;

   form_->RecoverFEMSolution(solve_X_, *rhs_form_, solution);
}

void DiffusionSolver::SetOperator(const Operator &op)
{
   Assemble();
   MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
               "External operator dimensions do not match DiffusionSolver.");
}

const Operator *DiffusionSolver::GetOperator() const
{
   Assemble();
   return system_operator_.Ptr();
}

const Solver *DiffusionSolver::GetPreconditioner() const
{
   Assemble();
   return preconditioner_.get();
}

std::shared_ptr<Coefficient> DiffusionSolver::ShareCoefficient(
   Coefficient &coefficient, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<Coefficient>(&coefficient);
   }
   return std::shared_ptr<Coefficient>(&coefficient, [](Coefficient *) { });
}

std::shared_ptr<QuadratureFunction> DiffusionSolver::ShareQuadratureFunction(
   QuadratureFunction &qf, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<QuadratureFunction>(&qf);
   }
   return std::shared_ptr<QuadratureFunction>(&qf,
                                              [](QuadratureFunction *) { });
}

std::shared_ptr<ParGridFunction> DiffusionSolver::ShareParGridFunction(
   ParGridFunction &gf, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<ParGridFunction>(&gf);
   }
   return std::shared_ptr<ParGridFunction>(&gf, [](ParGridFunction *) { });
}

std::shared_ptr<Coefficient> DiffusionSolver::MakeQuadratureCoefficient(
   std::shared_ptr<QuadratureFunction> qf) const
{
   ValidateQuadratureFunction(*qf);
   return std::make_shared<QuadratureFunctionCoefficient>(*qf);
}

std::shared_ptr<Coefficient> DiffusionSolver::MakeGridFunctionCoefficient(
   std::shared_ptr<ParGridFunction> gf) const
{
   ValidateParGridFunction(*gf);
   return std::make_shared<GridFunctionCoefficient>(gf.get());
}

void DiffusionSolver::ValidateQuadratureFunction(
   const QuadratureFunction &qf) const
{
   MFEM_VERIFY(qf.GetVDim() == 1, "Expected scalar QuadratureFunction.");
   MFEM_VERIFY(qf.GetSpace() != nullptr, "QuadratureFunction has no space.");
   MFEM_VERIFY(qf.GetSpace()->GetMesh() == fespace_.GetParMesh(),
               "QuadratureFunction must be defined on the solver mesh.");
   MFEM_VERIFY(qf.GetSpace()->GetOrder() == integration_order_,
               "QuadratureFunction integration order does not match solver.");
   MFEM_VERIFY(qf.Size() == qf.GetVDim()*qf.GetSpace()->GetSize(),
               "QuadratureFunction size is incompatible with its space.");

   for (int e = 0; e < fespace_.GetParMesh()->GetNE(); e++)
   {
      const IntegrationRule &solver_ir =
         GetIntegrationRule(fespace_.GetParMesh()->GetElementGeometry(e));
      const IntegrationRule &qf_ir = qf.GetIntRule(e);
      MFEM_VERIFY(qf_ir.Size() == solver_ir.Size() &&
                  qf_ir.GetOrder() == solver_ir.GetOrder(),
                  "QuadratureFunction rule does not match solver rule.");
   }
}

void DiffusionSolver::ValidateSurfaceQuadratureFunction(
   const QuadratureFunction &qf) const
{
   MFEM_VERIFY(qf.GetVDim() == 1, "Expected scalar QuadratureFunction.");
   MFEM_VERIFY(qf.GetSpace() != nullptr, "QuadratureFunction has no space.");
   MFEM_VERIFY(qf.GetSpace()->GetMesh() == fespace_.GetParMesh(),
               "QuadratureFunction must be defined on the solver mesh.");
   const auto *face_space =
      dynamic_cast<const FaceQuadratureSpace *>(qf.GetSpace());
   MFEM_VERIFY(face_space != nullptr,
               "Surface-load QuadratureFunction must use a "
               "FaceQuadratureSpace.");
   MFEM_VERIFY(face_space->GetFaceType() == FaceType::Boundary,
               "Surface-load QuadratureFunction must use boundary faces.");
   MFEM_VERIFY(qf.GetSpace()->GetOrder() == integration_order_,
               "Surface-load QuadratureFunction integration order does not "
               "match solver boundary integration order.");
   MFEM_VERIFY(qf.Size() == qf.GetVDim()*qf.GetSpace()->GetSize(),
               "QuadratureFunction size is incompatible with its space.");
}

void DiffusionSolver::ValidateParGridFunction(const ParGridFunction &gf) const
{
   MFEM_VERIFY(gf.ParFESpace() == &fespace_,
               "ParGridFunction coefficient must use the solver FE space.");
   MFEM_VERIFY(gf.Size() == fespace_.GetVSize(),
               "ParGridFunction coefficient has incompatible local size.");
}

void DiffusionSolver::SolveSystem(const Vector &rhs, Vector &solution,
                                  bool apply_constraints,
                                  bool use_boundary_values) const
{
   constrained_rhs_.SetSize(rhs.Size());
   constrained_rhs_ = rhs;
   const Vector *constraint_values = nullptr;
   if (apply_constraints)
   {
      if (use_boundary_values)
      {
         BuildBoundaryTrueDofs(boundary_true_dofs_);
         constraint_values = &boundary_true_dofs_;
      }
      else
      {
         homogeneous_boundary_true_dofs_.SetSize(ess_tdofs_.Size());
         homogeneous_boundary_true_dofs_ = 0.0;
         constraint_values = &homogeneous_boundary_true_dofs_;
      }
      constrained_rhs_.SetSubVector(ess_tdofs_, *constraint_values);
   }

   solution.SetSize(Height());
   solution = 0.0;

   MFEM_VERIFY(cg_solver_ != nullptr,
               "DiffusionSolver linear solver has not been configured.");
   cg_solver_->Mult(constrained_rhs_, solution);

   if (apply_constraints)
   {
      solution.SetSubVector(ess_tdofs_, *constraint_values);
   }
}

void DiffusionSolver::BuildBoundaryTrueDofs(Vector &boundary_values) const
{
   const bool rebuild_boundary_values =
      GlobalBooleanOr(fespace_.GetComm(),
                      boundary_true_dofs_dirty_ ||
                      boundary_values.Size() != ess_tdofs_.Size());
   if (!rebuild_boundary_values)
   {
      return;
   }

   boundary_values.SetSize(ess_tdofs_.Size());
   boundary_values = 0.0;
   const bool has_boundary_data =
      GlobalBooleanOr(fespace_.GetComm(),
                      !boundary_coefficients_.Empty() &&
                      !boundary_ids_.empty());
   if (!has_boundary_data)
   {
      boundary_true_dofs_dirty_ = false;
      return;
   }

   if (!boundary_grid_function_ ||
       boundary_grid_function_->ParFESpace() != &fespace_)
   {
      boundary_grid_function_.reset(new ParGridFunction(&fespace_));
   }

   *boundary_grid_function_ = 0.0;
   BuildEssentialMarker(boundary_marker_);
   boundary_grid_function_->ProjectBdrCoefficient(
      boundary_coefficients_.AsCoefficient(), boundary_marker_);

   boundary_grid_function_->GetTrueDofs(boundary_all_true_dofs_);
   boundary_all_true_dofs_.GetSubVector(ess_tdofs_, boundary_values);
   boundary_true_dofs_dirty_ = false;
}

void DiffusionSolver::MarkCoefficientChanged()
{
   needs_assembly_ = true;
}

void DiffusionSolver::MarkRHSChanged()
{
   needs_assembly_ = true;
   rhs_form_dirty_ = true;
   rhs_vector_dirty_ = true;
}

void DiffusionSolver::MarkBoundaryChanged(int id)
{
   if (id > 0) { boundary_ids_.insert(id); }
   needs_assembly_ = true;
   boundary_true_dofs_dirty_ = true;
}

void DiffusionSolver::BuildEssentialMarker(Array<int> &marker) const
{
   int max_attr = fespace_.GetParMesh()->bdr_attributes.Size()
                ? fespace_.GetParMesh()->bdr_attributes.Max() : 0;
   for (int id : boundary_ids_)
   {
      max_attr = std::max(max_attr, id);
   }
   marker.SetSize(max_attr);
   marker = 0;
   for (int id : boundary_ids_)
   {
      marker[id - 1] = 1;
   }
}

void DiffusionSolver::BuildEssentialTrueDofs() const
{
   ess_tdofs_.SetSize(0);
   if (GlobalBooleanOr(fespace_.GetComm(), !boundary_ids_.empty()))
   {
      BuildEssentialMarker(boundary_marker_);
      fespace_.GetEssentialTrueDofs(boundary_marker_, ess_tdofs_);
   }
   boundary_true_dofs_dirty_ = true;
}

void DiffusionSolver::BuildPreconditioner() const
{
   preconditioner_.reset();
   lor_operator_.Clear();
   lor_form_.reset();
   lor_diffusion_coefficient_.reset();
   lor_diffusion_base_coefficient_.reset();
   lor_diffusion_gf_.reset();
   lor_fespace_.reset();
   lor_fec_.reset();
   lor_mesh_.reset();
   assembled_operator_.Clear();
   assembled_form_.reset();

   if (fespace_.GetMaxElementOrder() > 1)
   {
      BuildLORAMGPreconditioner();
   }
   else
   {
      BuildAMGPreconditionerOnFESpace();
   }
}

void DiffusionSolver::BuildAMGPreconditionerOnFESpace() const
{
   assembled_form_.reset(new ParBilinearForm(&fespace_));
   assembled_form_->AddDomainIntegrator(new DiffusionIntegrator(
                                           *diffusion_coefficient_,
                                           &GetIntegrationRule(
                                              fespace_.GetParMesh()
                                              ->GetElementGeometry(0))));
   assembled_form_->Assemble();

   assembled_operator_.SetType(Operator::Hypre_ParCSR);
   assembled_form_->FormSystemMatrix(ess_tdofs_, assembled_operator_);
   HypreParMatrix *matrix = assembled_operator_.Is<HypreParMatrix>();
   MFEM_VERIFY(matrix != nullptr, "Assembled operator is not a HypreParMatrix.");

   HypreBoomerAMG *amg = new HypreBoomerAMG(*matrix);
   amg->SetPrintLevel(print_level_);
   preconditioner_.reset(amg);
}

void DiffusionSolver::BuildLORAMGPreconditioner() const
{
   const int order = fespace_.GetMaxElementOrder();
   ParMesh *pmesh = fespace_.GetParMesh();

   MFEM_VERIFY((pmesh->MeshGenerator() & 1) == 0,
               "LOR+AMG preconditioning requires tensor-product elements.");

   lor_mesh_.reset(new ParMesh);
   *lor_mesh_ = ParMesh::MakeRefined(*pmesh, order, BasisType::GaussLobatto);
   lor_fec_.reset(new H1_FECollection(1, pmesh->Dimension()));
   lor_fespace_.reset(new ParFiniteElementSpace(lor_mesh_.get(),
                                                lor_fec_.get()));

   MFEM_VERIFY(lor_fespace_->GetTrueVSize() == fespace_.GetTrueVSize(),
               "LOR true-vector size does not match the high-order space. "
               "Use a compatible tensor H1 space, e.g. Gauss-Lobatto basis.");

   lor_diffusion_coefficient_ = MakeLORCoefficient(diffusion_coefficient_,
                                                   diffusion_gf_owner_,
                                                   diffusion_qf_owner_,
                                                   lor_diffusion_gf_,
                                                   lor_diffusion_base_coefficient_);

   lor_form_.reset(new ParBilinearForm(lor_fespace_.get()));
   lor_form_->AddDomainIntegrator(new DiffusionIntegrator(
                                    *lor_diffusion_coefficient_,
                                    &GetIntegrationRule(
                                       pmesh->GetElementGeometry(0))));
   lor_form_->Assemble();

   lor_operator_.SetType(Operator::Hypre_ParCSR);
   lor_form_->FormSystemMatrix(ess_tdofs_, lor_operator_);
   HypreParMatrix *lor_matrix = lor_operator_.Is<HypreParMatrix>();
   MFEM_VERIFY(lor_matrix != nullptr, "LOR operator is not a HypreParMatrix.");

   HypreBoomerAMG *amg = new HypreBoomerAMG(*lor_matrix);
   amg->SetPrintLevel(print_level_);
   preconditioner_.reset(amg);
}

std::shared_ptr<Coefficient> DiffusionSolver::MakeLORCoefficient(
   const std::shared_ptr<Coefficient> &coefficient,
   const std::shared_ptr<ParGridFunction> &ho_gf,
   const std::shared_ptr<QuadratureFunction> &qf,
   std::shared_ptr<ParGridFunction> &lor_gf,
   std::shared_ptr<Coefficient> &lor_base_coefficient) const
{
   if (!ho_gf && !qf)
   {
      return coefficient;
   }

   MFEM_VERIFY(lor_fespace_,
               "LOR finite element space must be constructed before transfer.");

   if (ho_gf)
   {
      MFEM_VERIFY(lor_fespace_->GetTrueVSize() ==
                  ho_gf->ParFESpace()->GetTrueVSize(),
                  "Cannot transfer ParGridFunction coefficient to LOR space: "
                  "true-vector sizes differ.");
      ho_gf->GetTrueDofs(lor_transfer_true_dofs_);
   }
   else
   {
      ParGridFunction projected(&fespace_);
      projected.ProjectCoefficient(*coefficient);
      MFEM_VERIFY(lor_fespace_->GetTrueVSize() == fespace_.GetTrueVSize(),
                  "Cannot transfer QuadratureFunction coefficient to LOR "
                  "space: true-vector sizes differ.");
      projected.GetTrueDofs(lor_transfer_true_dofs_);
   }

   lor_gf = std::make_shared<ParGridFunction>(lor_fespace_.get());
   lor_gf->SetFromTrueDofs(lor_transfer_true_dofs_);
   lor_base_coefficient =
      std::make_shared<GridFunctionCoefficient>(lor_gf.get());
   return lor_base_coefficient;
}

void StokesSolver::VectorAttributeCoefficientMap::SetOwner(
   StokesSolver *owner, bool boundary_map)
{
   owner_ = owner;
   boundary_map_ = boundary_map;
}

void StokesSolver::VectorAttributeCoefficientMap::Add(
   int attr, VectorCoefficient &coefficient, bool transfer_ownership)
{
   Add(attr, ShareCoefficient(coefficient, transfer_ownership));
}

void StokesSolver::VectorAttributeCoefficientMap::Add(
   int attr, std::shared_ptr<VectorCoefficient> coefficient)
{
   MFEM_VERIFY(attr > 0, "Attribute ids are one-based and must be positive.");
   MFEM_VERIFY(coefficient != nullptr, "VectorCoefficient pointer is null.");
   coefficients_[attr] = coefficient;
   piecewise_.reset();
   NotifyChanged(attr);
}

void StokesSolver::VectorAttributeCoefficientMap::Clear()
{
   coefficients_.clear();
   piecewise_.reset();
   NotifyChanged(0);
}

VectorCoefficient &
StokesSolver::VectorAttributeCoefficientMap::AsCoefficient() const
{
   MFEM_VERIFY(owner_ != nullptr, "Coefficient map is not attached to a solver.");
   piecewise_.reset(new PWVectorCoefficient(
                       owner_->velocity_space_.GetVDim()));
   for (auto &entry : coefficients_)
   {
      piecewise_->UpdateCoefficient(entry.first, *entry.second);
   }
   return *piecewise_;
}

std::shared_ptr<VectorCoefficient>
StokesSolver::VectorAttributeCoefficientMap::ShareCoefficient(
   VectorCoefficient &coefficient, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<VectorCoefficient>(&coefficient);
   }
   return std::shared_ptr<VectorCoefficient>(&coefficient,
                                             [](VectorCoefficient *) { });
}

void StokesSolver::VectorAttributeCoefficientMap::NotifyChanged(int attr)
{
   if (!owner_) { return; }
   if (boundary_map_)
   {
      owner_->MarkVelocityBoundaryChanged(attr);
   }
   else
   {
      owner_->MarkAccelerationChanged();
   }
}

void StokesSolver::ScalarAttributeCoefficientMap::SetOwner(
   StokesSolver *owner, bool boundary_map)
{
   owner_ = owner;
   boundary_map_ = boundary_map;
}

void StokesSolver::ScalarAttributeCoefficientMap::Add(int attr, real_t value)
{
   Add(attr, std::make_shared<ConstantCoefficient>(value));
}

void StokesSolver::ScalarAttributeCoefficientMap::Add(
   int attr, Coefficient &coefficient, bool transfer_ownership)
{
   Add(attr, ShareCoefficient(coefficient, transfer_ownership));
}

void StokesSolver::ScalarAttributeCoefficientMap::Add(
   int attr, std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(attr > 0, "Attribute ids are one-based and must be positive.");
   MFEM_VERIFY(coefficient != nullptr, "Coefficient pointer is null.");
   coefficients_[attr] = coefficient;
   piecewise_.reset();
   NotifyChanged(attr);
}

void StokesSolver::ScalarAttributeCoefficientMap::Clear()
{
   coefficients_.clear();
   piecewise_.reset();
   NotifyChanged(0);
}

Coefficient &StokesSolver::ScalarAttributeCoefficientMap::AsCoefficient() const
{
   piecewise_.reset(new PWCoefficient);
   for (auto &entry : coefficients_)
   {
      piecewise_->UpdateCoefficient(entry.first, *entry.second);
   }
   return *piecewise_;
}

std::shared_ptr<Coefficient>
StokesSolver::ScalarAttributeCoefficientMap::ShareCoefficient(
   Coefficient &coefficient, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<Coefficient>(&coefficient);
   }
   return std::shared_ptr<Coefficient>(&coefficient, [](Coefficient *) { });
}

void StokesSolver::ScalarAttributeCoefficientMap::NotifyChanged(int attr)
{
   if (!owner_) { return; }
   if (boundary_map_)
   {
      owner_->MarkPressureBoundaryChanged(attr);
   }
}

StokesSolver::StokesSolver(ParFiniteElementSpace &velocity_space,
                           ParFiniteElementSpace &pressure_space)
   : Solver(velocity_space.GetTrueVSize() + pressure_space.GetTrueVSize()),
     velocity_space_(velocity_space),
     pressure_space_(pressure_space),
     viscosity_coefficient_(std::make_shared<ConstantCoefficient>(1.0)),
     integration_order_(2*std::max(velocity_space.GetMaxElementOrder(),
                                   pressure_space.GetMaxElementOrder()))
{
   MFEM_VERIFY(velocity_space_.GetParMesh() == pressure_space_.GetParMesh(),
               "Stokes spaces must be defined on the same ParMesh.");
   MFEM_VERIFY(velocity_space_.GetVDim() ==
               velocity_space_.GetParMesh()->Dimension(),
               "Velocity space must have vector dimension equal to mesh dimension.");
   MFEM_VERIFY(velocity_space_.GetOrdering() == Ordering::byNODES,
               "StokesSolver partial assembly requires velocity Ordering::byNODES.");
   block_offsets_.SetSize(3);
   block_offsets_[0] = 0;
   block_offsets_[1] = velocity_space_.GetTrueVSize();
   block_offsets_[2] = pressure_space_.GetTrueVSize();
   block_offsets_.PartialSum();
   acceleration_.SetOwner(this, false);
   velocity_boundary_.SetOwner(this, true);
   pressure_boundary_.SetOwner(this, true);
}

StokesSolver::StokesSolver(
   std::shared_ptr<ParFiniteElementSpace> velocity_space,
   std::shared_ptr<ParFiniteElementSpace> pressure_space)
   : StokesSolver(CheckedFESpace(velocity_space, "Velocity"),
                  CheckedFESpace(pressure_space, "Pressure"))
{
   velocity_space_owner_ = velocity_space;
   pressure_space_owner_ = pressure_space;
}

void StokesSolver::SetViscosity(real_t value)
{
   SetViscosity(std::make_shared<ConstantCoefficient>(value));
}

void StokesSolver::SetViscosity(Coefficient &coefficient,
                                bool transfer_ownership)
{
   viscosity_qf_owner_.reset();
   viscosity_gf_owner_.reset();
   SetViscosity(ShareCoefficient(coefficient, transfer_ownership));
}

void StokesSolver::SetViscosity(std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(coefficient != nullptr, "Viscosity coefficient is null.");
   viscosity_coefficient_ = coefficient;
   MarkOperatorChanged();
}

void StokesSolver::SetViscosity(QuadratureFunction &qf,
                                bool transfer_ownership)
{
   SetViscosity(ShareQuadratureFunction(qf, transfer_ownership));
}

void StokesSolver::SetViscosity(std::shared_ptr<QuadratureFunction> qf)
{
   MFEM_VERIFY(qf != nullptr, "Viscosity QuadratureFunction is null.");
   ValidateQuadratureFunction(*qf);
   viscosity_qf_owner_ = qf;
   viscosity_gf_owner_.reset();
   viscosity_coefficient_ = MakeQuadratureCoefficient(qf);
   MarkOperatorChanged();
}

void StokesSolver::SetViscosity(ParGridFunction &gf,
                                bool transfer_ownership)
{
   SetViscosity(ShareParGridFunction(gf, transfer_ownership));
}

void StokesSolver::SetViscosity(std::shared_ptr<ParGridFunction> gf)
{
   MFEM_VERIFY(gf != nullptr, "Viscosity ParGridFunction is null.");
   ValidateParGridFunction(*gf);
   viscosity_qf_owner_.reset();
   viscosity_gf_owner_ = gf;
   viscosity_coefficient_ = MakeGridFunctionCoefficient(gf);
   MarkOperatorChanged();
}

void StokesSolver::SetBrinkmanPenalization(real_t value)
{
   SetBrinkmanPenalization(std::make_shared<ConstantCoefficient>(value));
}

void StokesSolver::SetBrinkmanPenalization(Coefficient &coefficient,
                                           bool transfer_ownership)
{
   brinkman_qf_owner_.reset();
   brinkman_gf_owner_.reset();
   SetBrinkmanPenalization(ShareCoefficient(coefficient, transfer_ownership));
}

void StokesSolver::SetBrinkmanPenalization(
   std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(coefficient != nullptr,
               "Brinkman penalization coefficient is null.");
   brinkman_coefficient_ = coefficient;
   MarkOperatorChanged();
}

void StokesSolver::SetBrinkmanPenalization(
   QuadratureFunction &qf, bool transfer_ownership)
{
   SetBrinkmanPenalization(ShareQuadratureFunction(qf, transfer_ownership));
}

void StokesSolver::SetBrinkmanPenalization(
   std::shared_ptr<QuadratureFunction> qf)
{
   MFEM_VERIFY(qf != nullptr,
               "Brinkman penalization QuadratureFunction is null.");
   ValidateQuadratureFunction(*qf);
   brinkman_qf_owner_ = qf;
   brinkman_gf_owner_.reset();
   brinkman_coefficient_ = MakeQuadratureCoefficient(qf);
   MarkOperatorChanged();
}

void StokesSolver::SetBrinkmanPenalization(
   ParGridFunction &gf, bool transfer_ownership)
{
   SetBrinkmanPenalization(ShareParGridFunction(gf, transfer_ownership));
}

void StokesSolver::SetBrinkmanPenalization(
   std::shared_ptr<ParGridFunction> gf)
{
   MFEM_VERIFY(gf != nullptr,
               "Brinkman penalization ParGridFunction is null.");
   ValidateParGridFunction(*gf);
   brinkman_qf_owner_.reset();
   brinkman_gf_owner_ = gf;
   brinkman_coefficient_ = MakeGridFunctionCoefficient(gf);
   MarkOperatorChanged();
}

void StokesSolver::ClearBrinkmanPenalization()
{
   brinkman_coefficient_.reset();
   brinkman_qf_owner_.reset();
   brinkman_gf_owner_.reset();
   MarkOperatorChanged();
}

void StokesSolver::AddVelocityBoundaryID(int id)
{
   MFEM_VERIFY(id > 0, "Boundary ids are one-based and must be positive.");
   MarkVelocityBoundaryChanged(id);
}

void StokesSolver::AddPressureBoundaryID(int id)
{
   MFEM_VERIFY(id > 0, "Boundary ids are one-based and must be positive.");
   MarkPressureBoundaryChanged(id);
}

void StokesSolver::ClearVelocityBoundaryConditions()
{
   velocity_boundary_ids_.clear();
   velocity_boundary_.Clear();
   MarkOperatorChanged();
}

void StokesSolver::ClearPressureBoundaryConditions()
{
   pressure_boundary_ids_.clear();
   pressure_boundary_.Clear();
   MarkOperatorChanged();
}

void StokesSolver::SetSolverType(KrylovSolver solver_type)
{
   solver_type_ = solver_type;
   needs_assembly_ = true;
}

void StokesSolver::SetVelocityPreconditionerType(
   VelocityPreconditioner prec_type)
{
   velocity_prec_type_ = prec_type;
   needs_assembly_ = true;
}

void StokesSolver::SetVelocityAMGElasticityNearNullspace(
   bool use_near_nullspace)
{
   velocity_amg_elasticity_near_nullspace_ = use_near_nullspace;
   needs_assembly_ = true;
}

void StokesSolver::SetPressurePreconditionerType(
   PressurePreconditioner prec_type)
{
   pressure_prec_type_ = prec_type;
   needs_assembly_ = true;
}

void StokesSolver::SetCCDiffusionSolverType(CCDiffusionSolver solver_type)
{
   cc_diffusion_solver_type_ = solver_type;
   needs_assembly_ = true;
}

void StokesSolver::SetLSCVelocityOperatorType(
   LSCVelocityOperator operator_type)
{
   lsc_velocity_operator_type_ = operator_type;
   needs_assembly_ = true;
}

void StokesSolver::SetLSCDiagonalOperatorType(
   LSCDiagonalOperator operator_type)
{
   lsc_diagonal_operator_type_ = operator_type;
   needs_assembly_ = true;
}

void StokesSolver::SetLSCQPreconditionerType(
   LSCQPreconditioner prec_type)
{
   lsc_q_preconditioner_type_ = prec_type;
   needs_assembly_ = true;
}

void StokesSolver::SetRelTol(real_t rel_tol)
{
   MFEM_VERIFY(rel_tol >= 0.0, "Relative tolerance must be nonnegative.");
   rel_tol_ = rel_tol;
   needs_assembly_ = true;
}

void StokesSolver::SetAbsTol(real_t abs_tol)
{
   MFEM_VERIFY(abs_tol >= 0.0, "Absolute tolerance must be nonnegative.");
   abs_tol_ = abs_tol;
   needs_assembly_ = true;
}

void StokesSolver::SetMaxIter(int max_iter)
{
   MFEM_VERIFY(max_iter > 0, "Maximum iteration count must be positive.");
   max_iter_ = max_iter;
   needs_assembly_ = true;
}

void StokesSolver::SetPreconditionerCGRelTol(real_t rel_tol)
{
   SetVelocityPreconditionerCGRelTol(rel_tol);
   SetPressurePreconditionerCGRelTol(rel_tol);
}

void StokesSolver::SetPreconditionerCGAbsTol(real_t abs_tol)
{
   SetVelocityPreconditionerCGAbsTol(abs_tol);
   SetPressurePreconditionerCGAbsTol(abs_tol);
}

void StokesSolver::SetPreconditionerCGMaxIter(int max_iter)
{
   SetVelocityPreconditionerCGMaxIter(max_iter);
   SetPressurePreconditionerCGMaxIter(max_iter);
}

void StokesSolver::SetVelocityPreconditionerCGRelTol(real_t rel_tol)
{
   MFEM_VERIFY(rel_tol >= 0.0,
               "Velocity preconditioner CG relative tolerance must be nonnegative.");
   velocity_pc_cg_rel_tol_ = rel_tol;
   needs_assembly_ = true;
}

void StokesSolver::SetVelocityPreconditionerCGAbsTol(real_t abs_tol)
{
   MFEM_VERIFY(abs_tol >= 0.0,
               "Velocity preconditioner CG absolute tolerance must be nonnegative.");
   velocity_pc_cg_abs_tol_ = abs_tol;
   needs_assembly_ = true;
}

void StokesSolver::SetVelocityPreconditionerCGMaxIter(int max_iter)
{
   MFEM_VERIFY(max_iter > 0,
               "Velocity preconditioner CG maximum iteration count must be positive.");
   velocity_pc_cg_max_iter_ = max_iter;
   needs_assembly_ = true;
}

void StokesSolver::SetPressurePreconditionerCGRelTol(real_t rel_tol)
{
   MFEM_VERIFY(rel_tol >= 0.0,
               "Pressure preconditioner CG relative tolerance must be nonnegative.");
   pressure_pc_cg_rel_tol_ = rel_tol;
   needs_assembly_ = true;
}

void StokesSolver::SetPressurePreconditionerCGAbsTol(real_t abs_tol)
{
   MFEM_VERIFY(abs_tol >= 0.0,
               "Pressure preconditioner CG absolute tolerance must be nonnegative.");
   pressure_pc_cg_abs_tol_ = abs_tol;
   needs_assembly_ = true;
}

void StokesSolver::SetPressurePreconditionerCGMaxIter(int max_iter)
{
   MFEM_VERIFY(max_iter > 0,
               "Pressure preconditioner CG maximum iteration count must be positive.");
   pressure_pc_cg_max_iter_ = max_iter;
   needs_assembly_ = true;
}

void StokesSolver::SetKDim(int kdim)
{
   MFEM_VERIFY(kdim > 0, "GMRES Krylov dimension must be positive.");
   kdim_ = kdim;
   needs_assembly_ = true;
}

void StokesSolver::SetPrintLevel(int print_level)
{
   print_level_ = print_level;
   needs_assembly_ = true;
}

void StokesSolver::SetNeedsAssembly(bool needs_assembly) const
{
   needs_assembly_ = needs_assembly;
   if (needs_assembly)
   {
      rhs_dirty_ = true;
      boundary_values_dirty_ = true;
   }
}

const IntegrationRule &StokesSolver::GetIntegrationRule(
   Geometry::Type geom) const
{
   return IntRules.Get(geom, integration_order_);
}

void StokesSolver::Assemble() const
{
   if (!GlobalBooleanOr(velocity_space_.GetComm(), needs_assembly_)) { return; }
   BuildEssentialTrueDofs();
   BuildOperator();
   BuildPreconditioner();

   projected_operator_.reset();
   projected_preconditioner_.reset();
   Operator *linear_operator = block_operator_.get();
   if (HasPressureNullspace())
   {
      projected_operator_.reset(new PressureMeanProjectedOperator(
                                   *block_operator_, block_offsets_,
                                   pressure_space_.GetComm()));
      linear_operator = projected_operator_.get();
   }

   if (solver_type_ == KrylovSolver::MINRES)
   {
      auto *minres = new MINRESSolver(velocity_space_.GetComm());
      Solver *preconditioner = diagonal_prec_.get();
      if (HasPressureNullspace())
      {
         projected_preconditioner_.reset(new PressureMeanProjectedSolver(
                                            *diagonal_prec_, block_offsets_,
                                            pressure_space_.GetComm()));
         preconditioner = projected_preconditioner_.get();
      }
      minres->SetPreconditioner(*preconditioner);
      iterative_solver_.reset(minres);
   }
   else
   {
      auto *gmres = new GMRESSolver(velocity_space_.GetComm());
      gmres->SetKDim(kdim_);
      Solver *preconditioner = triangular_prec_.get();
      if (HasPressureNullspace())
      {
         projected_preconditioner_.reset(new PressureMeanProjectedSolver(
                                            *triangular_prec_, block_offsets_,
                                            pressure_space_.GetComm()));
         preconditioner = projected_preconditioner_.get();
      }
      gmres->SetPreconditioner(*preconditioner);
      iterative_solver_.reset(gmres);
   }
   iterative_solver_->SetRelTol(rel_tol_);
   iterative_solver_->SetAbsTol(abs_tol_);
   iterative_solver_->SetMaxIter(max_iter_);
   iterative_solver_->SetPrintLevel(print_level_);
   iterative_solver_->SetOperator(*linear_operator);

   StokesSolver *self = const_cast<StokesSolver *>(this);
   self->height = block_operator_->Height();
   self->width = block_operator_->Width();
   needs_assembly_ = false;
}

void StokesSolver::Mult(const Vector &rhs, Vector &solution) const
{
   BlockVector rhs_block(const_cast<Vector &>(rhs), block_offsets_);
   solution.SetSize(Height());
   BlockVector sol_block(solution, block_offsets_);
   Mult(rhs_block, sol_block);
}

void StokesSolver::Mult(const BlockVector &rhs, BlockVector &solution) const
{
   Assemble();
   MultAssembled(rhs, solution);
}

void StokesSolver::MultAssembled(const BlockVector &rhs,
                                 BlockVector &solution) const
{
   MFEM_VERIFY(rhs.Size() == Width(), "RHS has incompatible size.");
   MFEM_VERIFY(solution.Size() == Height(), "Solution has incompatible size.");
   SolveSystem(rhs, solution, true);
}

void StokesSolver::MultTranspose(const Vector &rhs, Vector &solution) const
{
   Assemble();
   MFEM_VERIFY(rhs.Size() == Height(), "RHS has incompatible size.");
   solution.SetSize(Width());
   BlockVector rhs_block(const_cast<Vector &>(rhs), block_offsets_);
   BlockVector sol_block(solution, block_offsets_);
   SolveSystem(rhs_block, sol_block, false);
}

void StokesSolver::Solve(BlockVector &solution) const
{
   Assemble();
   BuildBoundaryValues();
   if (!acceleration_form_ || rhs_dirty_)
   {
      acceleration_form_.reset(new ParLinearForm(&velocity_space_));
      if (!acceleration_.Empty())
      {
         acceleration_form_->AddDomainIntegrator(
            new VectorDomainLFIntegrator(acceleration_.AsCoefficient(),
                                         &GetIntegrationRule(
                                            velocity_space_.GetParMesh()
                                            ->GetElementGeometry(0))));
      }
      rhs_dirty_ = false;
   }
   *acceleration_form_ = 0.0;
   acceleration_form_->Assemble();
   rhs_velocity_.SetSize(velocity_space_.GetTrueVSize());
   acceleration_form_->ParallelAssemble(rhs_velocity_);
   rhs_pressure_.SetSize(pressure_space_.GetTrueVSize());
   rhs_pressure_ = 0.0;
   rhs_block_.Update(block_offsets_);
   rhs_block_.GetBlock(0) = rhs_velocity_;
   rhs_block_.GetBlock(1) = rhs_pressure_;
   SolveSystem(rhs_block_, solution, true);
}

void StokesSolver::SetOperator(const Operator &op)
{
   Assemble();
   MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
               "External operator dimensions do not match StokesSolver.");
}

const Operator *StokesSolver::GetOperator() const
{
   Assemble();
   return projected_operator_ ? projected_operator_.get() : block_operator_.get();
}

const Solver *StokesSolver::GetPreconditioner() const
{
   Assemble();
   if (projected_preconditioner_) { return projected_preconditioner_.get(); }
   return solver_type_ == KrylovSolver::GMRES
          ? static_cast<const Solver *>(triangular_prec_.get())
          : static_cast<const Solver *>(diagonal_prec_.get());
}

std::shared_ptr<Coefficient> StokesSolver::ShareCoefficient(
   Coefficient &coefficient, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<Coefficient>(&coefficient);
   }
   return std::shared_ptr<Coefficient>(&coefficient, [](Coefficient *) { });
}

std::shared_ptr<QuadratureFunction> StokesSolver::ShareQuadratureFunction(
   QuadratureFunction &qf, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<QuadratureFunction>(&qf);
   }
   return std::shared_ptr<QuadratureFunction>(&qf,
                                              [](QuadratureFunction *) { });
}

std::shared_ptr<ParGridFunction> StokesSolver::ShareParGridFunction(
   ParGridFunction &gf, bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<ParGridFunction>(&gf);
   }
   return std::shared_ptr<ParGridFunction>(&gf, [](ParGridFunction *) { });
}

std::shared_ptr<Coefficient> StokesSolver::MakeQuadratureCoefficient(
   std::shared_ptr<QuadratureFunction> qf) const
{
   ValidateQuadratureFunction(*qf);
   return std::make_shared<QuadratureFunctionCoefficient>(*qf);
}

std::shared_ptr<Coefficient> StokesSolver::MakeGridFunctionCoefficient(
   std::shared_ptr<ParGridFunction> gf) const
{
   ValidateParGridFunction(*gf);
   return std::make_shared<GridFunctionCoefficient>(gf.get());
}

void StokesSolver::ValidateQuadratureFunction(
   const QuadratureFunction &qf) const
{
   MFEM_VERIFY(qf.GetVDim() == 1, "Expected scalar QuadratureFunction.");
   MFEM_VERIFY(qf.GetSpace() != nullptr, "QuadratureFunction has no space.");
   MFEM_VERIFY(qf.GetSpace()->GetMesh() == velocity_space_.GetParMesh(),
               "QuadratureFunction must be defined on the solver mesh.");
}

void StokesSolver::ValidateParGridFunction(const ParGridFunction &gf) const
{
   MFEM_VERIFY(gf.ParFESpace()->GetParMesh() == velocity_space_.GetParMesh(),
               "ParGridFunction coefficient must use the solver mesh.");
   MFEM_VERIFY(gf.ParFESpace()->GetVDim() == 1,
               "Scalar Stokes coefficient ParGridFunction must be scalar.");
}

void StokesSolver::MarkOperatorChanged()
{
   needs_assembly_ = true;
}

void StokesSolver::MarkAccelerationChanged()
{
   rhs_dirty_ = true;
}

void StokesSolver::MarkVelocityBoundaryChanged(int attr)
{
   if (attr > 0) { velocity_boundary_ids_.insert(attr); }
   needs_assembly_ = true;
   boundary_values_dirty_ = true;
}

void StokesSolver::MarkPressureBoundaryChanged(int attr)
{
   if (attr > 0) { pressure_boundary_ids_.insert(attr); }
   needs_assembly_ = true;
   boundary_values_dirty_ = true;
}

void StokesSolver::BuildMarker(const std::set<int> &ids, Array<int> &marker,
                               const Array<int> &mesh_bdr_attributes) const
{
   int max_attr = mesh_bdr_attributes.Size() ? mesh_bdr_attributes.Max() : 0;
   for (int id : ids) { max_attr = std::max(max_attr, id); }
   marker.SetSize(max_attr);
   marker = 0;
   for (int id : ids) { marker[id - 1] = 1; }
}

void StokesSolver::BuildEssentialTrueDofs() const
{
   velocity_ess_tdofs_.SetSize(0);
   pressure_ess_tdofs_.SetSize(0);
   if (GlobalBooleanOr(velocity_space_.GetComm(),
                       !velocity_boundary_ids_.empty()))
   {
      BuildMarker(velocity_boundary_ids_, velocity_marker_,
                  velocity_space_.GetParMesh()->bdr_attributes);
      velocity_space_.GetEssentialTrueDofs(velocity_marker_,
                                           velocity_ess_tdofs_);
   }
   if (GlobalBooleanOr(pressure_space_.GetComm(),
                       !pressure_boundary_ids_.empty()))
   {
      BuildMarker(pressure_boundary_ids_, pressure_marker_,
                  pressure_space_.GetParMesh()->bdr_attributes);
      pressure_space_.GetEssentialTrueDofs(pressure_marker_,
                                           pressure_ess_tdofs_);
   }
   boundary_values_dirty_ = true;
}

void StokesSolver::BuildBoundaryValues() const
{
   if (!GlobalBooleanOr(velocity_space_.GetComm(), boundary_values_dirty_))
   {
      return;
   }
   boundary_block_.Update(block_offsets_);
   boundary_block_ = 0.0;
   velocity_boundary_true_.SetSize(velocity_ess_tdofs_.Size());
   velocity_boundary_true_ = 0.0;
   pressure_boundary_true_.SetSize(pressure_ess_tdofs_.Size());
   pressure_boundary_true_ = 0.0;
   if (!velocity_boundary_.Empty() && !velocity_boundary_ids_.empty())
   {
      if (!velocity_boundary_gf_)
      {
         velocity_boundary_gf_.reset(new ParGridFunction(&velocity_space_));
      }
      *velocity_boundary_gf_ = 0.0;
      BuildMarker(velocity_boundary_ids_, velocity_marker_,
                  velocity_space_.GetParMesh()->bdr_attributes);
      velocity_boundary_gf_->ProjectBdrCoefficient(
         velocity_boundary_.AsCoefficient(), velocity_marker_);
      velocity_boundary_gf_->GetTrueDofs(velocity_boundary_all_);
      velocity_boundary_all_.GetSubVector(velocity_ess_tdofs_,
                                          velocity_boundary_true_);
      boundary_block_.GetBlock(0).SetSubVector(velocity_ess_tdofs_,
                                               velocity_boundary_true_);
   }
   if (!pressure_boundary_.Empty() && !pressure_boundary_ids_.empty())
   {
      if (!pressure_boundary_gf_)
      {
         pressure_boundary_gf_.reset(new ParGridFunction(&pressure_space_));
      }
      *pressure_boundary_gf_ = 0.0;
      BuildMarker(pressure_boundary_ids_, pressure_marker_,
                  pressure_space_.GetParMesh()->bdr_attributes);
      pressure_boundary_gf_->ProjectBdrCoefficient(
         pressure_boundary_.AsCoefficient(), pressure_marker_);
      pressure_boundary_gf_->GetTrueDofs(pressure_boundary_all_);
      pressure_boundary_all_.GetSubVector(pressure_ess_tdofs_,
                                          pressure_boundary_true_);
      boundary_block_.GetBlock(1).SetSubVector(pressure_ess_tdofs_,
                                               pressure_boundary_true_);
   }
   boundary_values_dirty_ = false;
}

void StokesSolver::BuildOperator() const
{
   velocity_operator_.Clear();
   divergence_operator_.Clear();
   gradient_operator_.reset();
   negative_divergence_operator_.reset();
   block_operator_.reset();

   velocity_form_.reset(new ParBilinearForm(&velocity_space_));
   velocity_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   velocity_form_->AddDomainIntegrator(new VectorDiffusionIntegrator(
                                          *viscosity_coefficient_,
                                          &GetIntegrationRule(
                                             velocity_space_.GetParMesh()
                                             ->GetElementGeometry(0))));
   if (brinkman_coefficient_)
   {
      velocity_form_->AddDomainIntegrator(new VectorMassIntegrator(
                                             *brinkman_coefficient_));
   }
   velocity_form_->Assemble();
   velocity_operator_.SetType(Operator::ANY_TYPE);
   velocity_form_->FormSystemMatrix(velocity_ess_tdofs_, velocity_operator_);

   divergence_form_.reset(new ParMixedBilinearForm(&velocity_space_,
                                                   &pressure_space_));
   divergence_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   divergence_form_->AddDomainIntegrator(new VectorDivergenceIntegrator);
   divergence_form_->Assemble();
   divergence_operator_.SetType(Operator::ANY_TYPE);
   divergence_form_->FormRectangularSystemMatrix(velocity_ess_tdofs_,
                                                 pressure_ess_tdofs_,
                                                 divergence_operator_);
   gradient_operator_.reset(new TransposeOperator(divergence_operator_.Ptr()));
   negative_divergence_operator_.reset(
      new ScaledOperator(divergence_operator_.Ptr(), -1.0));

   block_operator_.reset(new BlockOperator(block_offsets_));
   block_operator_->SetBlock(0, 0, velocity_operator_.Ptr());
   block_operator_->SetBlock(0, 1, gradient_operator_.get(), -1.0);
   block_operator_->SetBlock(1, 0, divergence_operator_.Ptr(), -1.0);
}

void StokesSolver::BuildPreconditioner() const
{
   BuildVelocityPreconditioner();
   BuildPressurePreconditioner();
   negative_pressure_preconditioner_.reset(
      new ScaledOperator(pressure_preconditioner_.get(), -1.0));

   diagonal_prec_.reset(new BlockDiagonalPreconditioner(block_offsets_));
   diagonal_prec_->SetDiagonalBlock(0, velocity_preconditioner_.get());
   diagonal_prec_->SetDiagonalBlock(1, pressure_preconditioner_.get());

   triangular_prec_.reset(new BlockLowerTriangularPreconditioner(block_offsets_));
   triangular_prec_->SetDiagonalBlock(0, velocity_preconditioner_.get());
   triangular_prec_->SetDiagonalBlock(1, negative_pressure_preconditioner_.get());
   triangular_prec_->SetBlock(1, 0, negative_divergence_operator_.get());
}

void StokesSolver::BuildVelocityPreconditioner() const
{
   velocity_preconditioner_.reset();
   velocity_pc_jacobi_.reset();
   velocity_pc_operator_.Clear();
   velocity_pc_form_.reset();
   lor_mesh_.reset();
   lor_fec_.reset();
   lor_velocity_space_.reset();
   lor_scalar_space_.reset();
   lor_viscosity_gf_.reset();
   lor_viscosity_coefficient_.reset();
   lor_brinkman_gf_.reset();
   lor_brinkman_coefficient_.reset();
   velocity_pc_ess_tdofs_.SetSize(0);

   ParFiniteElementSpace *pc_space = &velocity_space_;
   if (velocity_space_.GetMaxElementOrder() > 1)
   {
      ParMesh *pmesh = velocity_space_.GetParMesh();
      MFEM_VERIFY((pmesh->MeshGenerator() & 1) == 0,
                  "LOR velocity preconditioning requires tensor-product elements.");
      lor_mesh_.reset(new ParMesh);
      *lor_mesh_ = ParMesh::MakeRefined(*pmesh,
                                        velocity_space_.GetMaxElementOrder(),
                                        BasisType::GaussLobatto);
      lor_fec_.reset(new H1_FECollection(1, pmesh->Dimension()));
      lor_velocity_space_.reset(new ParFiniteElementSpace(
                                   lor_mesh_.get(), lor_fec_.get(),
                                   velocity_space_.GetVDim(),
                                   velocity_space_.GetOrdering()));
      MFEM_VERIFY(lor_velocity_space_->GetTrueVSize() ==
                  velocity_space_.GetTrueVSize(),
                  "LOR velocity true-vector size does not match.");
      pc_space = lor_velocity_space_.get();
      if (!velocity_boundary_ids_.empty())
      {
         lor_velocity_space_->GetEssentialTrueDofs(velocity_marker_,
                                                  velocity_pc_ess_tdofs_);
      }
      lor_viscosity_coefficient_ = MakeLORViscosityCoefficient();
      if (brinkman_coefficient_)
      {
         lor_brinkman_coefficient_ = MakeLORBrinkmanCoefficient();
      }
   }
   else
   {
      lor_viscosity_coefficient_ = viscosity_coefficient_;
      lor_brinkman_coefficient_ = brinkman_coefficient_;
      velocity_pc_ess_tdofs_ = velocity_ess_tdofs_;
   }

   velocity_pc_form_.reset(new ParBilinearForm(pc_space));
   velocity_pc_form_->AddDomainIntegrator(new VectorDiffusionIntegrator(
                                             *lor_viscosity_coefficient_,
                                             &GetIntegrationRule(
                                                velocity_space_.GetParMesh()
                                                ->GetElementGeometry(0))));
   if (lor_brinkman_coefficient_)
   {
      velocity_pc_form_->AddDomainIntegrator(new VectorMassIntegrator(
                                                *lor_brinkman_coefficient_));
   }
   velocity_pc_form_->Assemble();
   velocity_pc_operator_.SetType(Operator::Hypre_ParCSR);
   velocity_pc_form_->FormSystemMatrix(velocity_pc_ess_tdofs_,
                                       velocity_pc_operator_);
   HypreParMatrix *matrix = velocity_pc_operator_.Is<HypreParMatrix>();
   MFEM_VERIFY(matrix != nullptr,
               "Velocity preconditioner matrix is not a HypreParMatrix.");
   if (velocity_prec_type_ == VelocityPreconditioner::AMG)
   {
      auto *amg = new HypreBoomerAMG(*matrix);
      if (velocity_amg_elasticity_near_nullspace_)
      {
         amg->SetElasticityOptions(pc_space);
      }
      amg->SetPrintLevel(print_level_);
      velocity_preconditioner_.reset(amg);
   }
   else
   {
      velocity_pc_jacobi_.reset(new OperatorJacobiSmoother(
                                   *velocity_pc_form_, velocity_pc_ess_tdofs_));
      auto *cg = new CGSolver(velocity_space_.GetComm());
      cg->SetOperator(*velocity_pc_operator_);
      cg->SetPreconditioner(*velocity_pc_jacobi_);
      cg->SetRelTol(velocity_pc_cg_rel_tol_);
      cg->SetAbsTol(velocity_pc_cg_abs_tol_);
      cg->SetMaxIter(velocity_pc_cg_max_iter_);
      cg->SetPrintLevel(print_level_);
      velocity_preconditioner_.reset(cg);
   }
}

void StokesSolver::BuildPressurePreconditioner() const
{
   pressure_preconditioner_.reset();
   pressure_mass_jacobi_.reset();
   pressure_mass_operator_.Clear();
   cc_diffusion_coefficient_.reset();
   cc_diffusion_solver_.reset();
   cc_diffusion_krylov_solver_.reset();
   cc_mass_solver_.reset();
   cc_pressure_preconditioner_.reset();
   lsc_q_operator_.reset();
   lsc_h_operator_.reset();
   lsc_scaled_divergence_transpose_.reset();
   lsc_q_matrix_.reset();
   lsc_divergence_form_.reset();
   lsc_divergence_operator_.Clear();
   lsc_velocity_form_.reset();
   lsc_velocity_operator_.Clear();
   lsc_q_amg_.reset();
   lsc_q_jacobi_.reset();
   lsc_q_operator_jacobi_.reset();
   lsc_q_solver_.reset();
   lsc_pressure_preconditioner_.reset();
   pressure_mass_coefficient_.reset(
      new ReciprocalCoefficient(*viscosity_coefficient_));
   pressure_mass_form_.reset(new ParBilinearForm(&pressure_space_));
   if (pressure_prec_type_ != PressurePreconditioner::AMG)
   {
      pressure_mass_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   pressure_mass_form_->AddDomainIntegrator(new MassIntegrator(
                                              *pressure_mass_coefficient_));
   pressure_mass_form_->Assemble();

   if (pressure_prec_type_ == PressurePreconditioner::DIAGONAL_MASS)
   {
      pressure_preconditioner_.reset(
         new OperatorJacobiSmoother(*pressure_mass_form_, pressure_ess_tdofs_));
   }
   else if (pressure_prec_type_ == PressurePreconditioner::LSC)
   {
      MFEM_VERIFY(velocity_operator_.Ptr() != nullptr &&
                  divergence_operator_.Ptr() != nullptr,
                  "Stokes operator must be assembled before LSC setup.");

      // The PA velocity block returned by FormSystemMatrix is a true-dof
      // ConstrainedOperator with DIAG_ONE essential dof semantics.
      MFEM_VERIFY(dynamic_cast<ConstrainedOperator *>(velocity_operator_.Ptr()) !=
                  nullptr,
                  "The PA LSC velocity operator must be constrained.");
      Operator *lsc_velocity_operator = velocity_operator_.Ptr();
      if (lsc_velocity_operator_type_ == LSCVelocityOperator::ASSEMBLED)
      {
         lsc_velocity_form_.reset(new ParBilinearForm(&velocity_space_));
         lsc_velocity_form_->AddDomainIntegrator(new VectorDiffusionIntegrator(
                                                   *viscosity_coefficient_,
                                                   &GetIntegrationRule(
                                                      velocity_space_.GetParMesh()
                                                      ->GetElementGeometry(0))));
         if (brinkman_coefficient_)
         {
            lsc_velocity_form_->AddDomainIntegrator(new VectorMassIntegrator(
                                                      *brinkman_coefficient_));
         }
         lsc_velocity_form_->Assemble();
         lsc_velocity_operator_.SetType(Operator::Hypre_ParCSR);
         lsc_velocity_form_->FormSystemMatrix(velocity_ess_tdofs_,
                                              lsc_velocity_operator_);
         MFEM_VERIFY(lsc_velocity_operator_.Ptr() != nullptr,
                     "LSC velocity operator is not assembled.");
         lsc_velocity_operator = lsc_velocity_operator_.Ptr();
      }

      Operator *lsc_diagonal_operator = lsc_velocity_operator;
      if (lsc_diagonal_operator_type_ == LSCDiagonalOperator::PA)
      {
         lsc_diagonal_operator = velocity_operator_.Ptr();
      }
      else if (lsc_diagonal_operator_type_ == LSCDiagonalOperator::ASSEMBLED)
      {
         if (lsc_velocity_operator_type_ != LSCVelocityOperator::ASSEMBLED)
         {
            lsc_velocity_form_.reset(new ParBilinearForm(&velocity_space_));
            lsc_velocity_form_->AddDomainIntegrator(
               new VectorDiffusionIntegrator(
                  *viscosity_coefficient_,
                  &GetIntegrationRule(velocity_space_.GetParMesh()
                                      ->GetElementGeometry(0))));
            if (brinkman_coefficient_)
            {
               lsc_velocity_form_->AddDomainIntegrator(new VectorMassIntegrator(
                                                          *brinkman_coefficient_));
            }
            lsc_velocity_form_->Assemble();
            lsc_velocity_operator_.SetType(Operator::Hypre_ParCSR);
            lsc_velocity_form_->FormSystemMatrix(velocity_ess_tdofs_,
                                                 lsc_velocity_operator_);
            MFEM_VERIFY(lsc_velocity_operator_.Ptr() != nullptr,
                        "LSC velocity operator is not assembled.");
         }
         lsc_diagonal_operator = lsc_velocity_operator_.Ptr();
      }

      if (lsc_diagonal_operator == velocity_operator_.Ptr())
      {
         if (lsc_velocity_operator_.Ptr() == nullptr)
         {
            lsc_velocity_form_.reset(new ParBilinearForm(&velocity_space_));
            lsc_velocity_form_->AddDomainIntegrator(
               new VectorDiffusionIntegrator(
                  *viscosity_coefficient_,
                  &GetIntegrationRule(velocity_space_.GetParMesh()
                                      ->GetElementGeometry(0))));
            if (brinkman_coefficient_)
            {
               lsc_velocity_form_->AddDomainIntegrator(new VectorMassIntegrator(
                                                          *brinkman_coefficient_));
            }
            lsc_velocity_form_->Assemble();
            lsc_velocity_operator_.SetType(Operator::Hypre_ParCSR);
            lsc_velocity_form_->FormSystemMatrix(velocity_ess_tdofs_,
                                                 lsc_velocity_operator_);
            MFEM_VERIFY(lsc_velocity_operator_.Ptr() != nullptr,
                        "LSC velocity operator is not assembled.");
         }
         lsc_diagonal_operator = lsc_velocity_operator_.Ptr();
      }

      lsc_velocity_diag_inverse_.SetSize(velocity_space_.GetTrueVSize());
      lsc_diagonal_operator->AssembleDiagonal(lsc_velocity_diag_inverse_);
      {
         real_t *diag = lsc_velocity_diag_inverse_.HostReadWrite();
         const int *ess = velocity_ess_tdofs_.HostRead();
         // Match DIAG_ONE elimination before inverting the true-dof diagonal.
         for (int i = 0; i < velocity_ess_tdofs_.Size(); i++)
         {
            diag[ess[i]] = 1.0;
         }
         for (int i = 0; i < lsc_velocity_diag_inverse_.Size(); i++)
         {
            MFEM_VERIFY(diag[i] != 0.0,
                        "Zero velocity diagonal entry in LSC setup.");
            diag[i] = 1.0/diag[i];
         }
      }

      lsc_divergence_form_.reset(new ParMixedBilinearForm(&velocity_space_,
                                                          &pressure_space_));
      lsc_divergence_form_->AddDomainIntegrator(
         new VectorDivergenceIntegrator);
      lsc_divergence_form_->Assemble();
      lsc_divergence_operator_.SetType(Operator::Hypre_ParCSR);
      lsc_divergence_form_->FormRectangularSystemMatrix(
         velocity_ess_tdofs_, pressure_ess_tdofs_, lsc_divergence_operator_);
      MFEM_VERIFY(lsc_divergence_operator_.Ptr() != nullptr,
                  "LSC divergence operator is not assembled.");
      HypreParMatrix *lsc_divergence_matrix =
         lsc_divergence_operator_.Is<HypreParMatrix>();
      MFEM_VERIFY(lsc_divergence_matrix != nullptr,
                  "LSC divergence operator is not a HypreParMatrix.");

      lsc_q_solver_.reset(new CGSolver(pressure_space_.GetComm()));
      if (lsc_q_preconditioner_type_ == LSCQPreconditioner::OPERATOR_JACOBI)
      {
         lsc_h_operator_.reset(new LSCHOperator(*lsc_divergence_matrix,
                                                *lsc_velocity_operator,
                                                lsc_velocity_diag_inverse_,
                                                velocity_ess_tdofs_,
                                                pressure_ess_tdofs_,
                                                false));
         lsc_q_operator_.reset(new LSCQOperator(*lsc_divergence_matrix,
                                                lsc_velocity_diag_inverse_,
                                                velocity_ess_tdofs_,
                                                pressure_ess_tdofs_,
                                                false,
                                                pressure_space_.GetComm()));
         lsc_q_diag_.SetSize(pressure_space_.GetTrueVSize());
         lsc_q_operator_->AssembleDiagonal(lsc_q_diag_);
         {
            real_t *diag = lsc_q_diag_.HostReadWrite();
            for (int i = 0; i < lsc_q_diag_.Size(); i++)
            {
               diag[i] = std::abs(diag[i]);
               if (diag[i] == 0.0) { diag[i] = 1.0; }
            }
            const int *ess = pressure_ess_tdofs_.HostRead();
            for (int i = 0; i < pressure_ess_tdofs_.Size(); i++)
            {
               diag[ess[i]] = 1.0;
            }
         }
         lsc_q_operator_jacobi_.reset(new OperatorJacobiSmoother(
                                         lsc_q_diag_, pressure_ess_tdofs_));
         lsc_q_solver_->SetOperator(*lsc_q_operator_);
         lsc_q_solver_->SetPreconditioner(*lsc_q_operator_jacobi_);
      }
      else
      {
         lsc_h_operator_.reset(new LSCHOperator(*lsc_divergence_matrix,
                                                *lsc_velocity_operator,
                                                lsc_velocity_diag_inverse_,
                                                velocity_ess_tdofs_,
                                                pressure_ess_tdofs_,
                                                true));
         lsc_scaled_divergence_transpose_.reset(
            lsc_divergence_matrix->Transpose());
         lsc_scaled_divergence_transpose_->ScaleRows(
            lsc_velocity_diag_inverse_);
         lsc_q_matrix_.reset(ParMult(lsc_divergence_matrix,
                                     lsc_scaled_divergence_transpose_.get(),
                                     true));
         lsc_q_matrix_->EliminateBC(pressure_ess_tdofs_, Operator::DIAG_ONE);
         lsc_q_solver_->SetOperator(*lsc_q_matrix_);
         if (lsc_q_preconditioner_type_ == LSCQPreconditioner::AMG)
         {
            lsc_q_amg_.reset(new HypreBoomerAMG(*lsc_q_matrix_));
            lsc_q_amg_->SetPrintLevel(print_level_);
            lsc_q_solver_->SetPreconditioner(*lsc_q_amg_);
         }
         else
         {
            lsc_q_jacobi_.reset(new HypreDiagScale(*lsc_q_matrix_));
            lsc_q_solver_->SetPreconditioner(*lsc_q_jacobi_);
         }
      }
      lsc_q_solver_->SetRelTol(pressure_pc_cg_rel_tol_);
      lsc_q_solver_->SetAbsTol(pressure_pc_cg_abs_tol_);
      lsc_q_solver_->SetMaxIter(pressure_pc_cg_max_iter_);
      lsc_q_solver_->SetPrintLevel(print_level_);
      lsc_pressure_preconditioner_.reset(
         new LSCPreconditioner(*lsc_q_solver_,
                               *lsc_h_operator_,
                               pressure_space_.GetComm(),
                               !HasOpenVelocityBoundary()));
      pressure_preconditioner_ = std::move(lsc_pressure_preconditioner_);
   }
   else if (pressure_prec_type_ == PressurePreconditioner::CAHOUET_CHABARD)
   {
      MFEM_VERIFY(brinkman_coefficient_ != nullptr,
                  "Cahouet-Chabard preconditioning requires a Brinkman "
                  "penalization coefficient.");
      pressure_mass_operator_.SetType(Operator::ANY_TYPE);
      pressure_mass_form_->FormSystemMatrix(pressure_ess_tdofs_,
                                            pressure_mass_operator_);
      pressure_mass_jacobi_.reset(new OperatorJacobiSmoother(
                                     *pressure_mass_form_, pressure_ess_tdofs_));
      cc_mass_solver_.reset(new CGSolver(pressure_space_.GetComm()));
      cc_mass_solver_->SetOperator(*pressure_mass_operator_);
      cc_mass_solver_->SetPreconditioner(*pressure_mass_jacobi_);
      cc_mass_solver_->SetRelTol(pressure_pc_cg_rel_tol_);
      cc_mass_solver_->SetAbsTol(pressure_pc_cg_abs_tol_);
      cc_mass_solver_->SetMaxIter(pressure_pc_cg_max_iter_);
      cc_mass_solver_->SetPrintLevel(print_level_);

      cc_diffusion_coefficient_.reset(
         new ReciprocalCoefficient(*brinkman_coefficient_));
      cc_diffusion_solver_.reset(new DiffusionSolver(pressure_space_));
      cc_diffusion_solver_->SetDiffusionCoefficient(cc_diffusion_coefficient_);
      cc_diffusion_solver_->SetRelTol(pressure_pc_cg_rel_tol_);
      cc_diffusion_solver_->SetAbsTol(pressure_pc_cg_abs_tol_);
      cc_diffusion_solver_->SetMaxIter(pressure_pc_cg_max_iter_);
      cc_diffusion_solver_->SetPrintLevel(print_level_);

      int local_max_bdr_attr =
         pressure_space_.GetParMesh()->bdr_attributes.Size()
         ? pressure_space_.GetParMesh()->bdr_attributes.Max() : 0;
      int max_bdr_attr = 0;
      MPI_Allreduce(&local_max_bdr_attr, &max_bdr_attr, 1, MPI_INT, MPI_MAX,
                    pressure_space_.GetComm());
      for (int attr = 1; attr <= max_bdr_attr; attr++)
      {
         if (velocity_boundary_ids_.find(attr) == velocity_boundary_ids_.end())
         {
            cc_diffusion_solver_->AddBoundaryID(attr);
         }
      }
      cc_diffusion_solver_->Assemble();
      Solver *diffusion_inverse = cc_diffusion_solver_.get();
      if (cc_diffusion_solver_type_ == CCDiffusionSolver::GMRES)
      {
         auto *gmres = new GMRESSolver(pressure_space_.GetComm());
         gmres->SetOperator(*cc_diffusion_solver_->GetOperator());
         if (cc_diffusion_solver_->GetPreconditioner())
         {
            gmres->SetPreconditioner(
               *const_cast<Solver *>(cc_diffusion_solver_->GetPreconditioner()));
         }
         gmres->SetRelTol(pressure_pc_cg_rel_tol_);
         gmres->SetAbsTol(pressure_pc_cg_abs_tol_);
         gmres->SetMaxIter(pressure_pc_cg_max_iter_);
         gmres->SetKDim(kdim_);
         gmres->SetPrintLevel(print_level_);
         cc_diffusion_krylov_solver_.reset(gmres);
         diffusion_inverse = cc_diffusion_krylov_solver_.get();
      }
      cc_pressure_preconditioner_.reset(
         new CahouetChabardPreconditioner(*cc_mass_solver_,
                                          *diffusion_inverse,
                                          pressure_space_.GetComm(),
                                          !HasOpenVelocityBoundary()));
      pressure_preconditioner_ = std::move(cc_pressure_preconditioner_);
   }
   else if (pressure_prec_type_ == PressurePreconditioner::CG)
   {
      pressure_mass_operator_.SetType(Operator::ANY_TYPE);
      pressure_mass_form_->FormSystemMatrix(pressure_ess_tdofs_,
                                            pressure_mass_operator_);
      pressure_mass_jacobi_.reset(new OperatorJacobiSmoother(
                                     *pressure_mass_form_, pressure_ess_tdofs_));
      auto *cg = new CGSolver(pressure_space_.GetComm());
      cg->SetOperator(*pressure_mass_operator_);
      cg->SetPreconditioner(*pressure_mass_jacobi_);
      cg->SetRelTol(pressure_pc_cg_rel_tol_);
      cg->SetAbsTol(pressure_pc_cg_abs_tol_);
      cg->SetMaxIter(pressure_pc_cg_max_iter_);
      cg->SetPrintLevel(print_level_);
      pressure_preconditioner_.reset(cg);
   }
   else
   {
      pressure_mass_operator_.SetType(Operator::Hypre_ParCSR);
      pressure_mass_form_->FormSystemMatrix(pressure_ess_tdofs_,
                                            pressure_mass_operator_);
      HypreParMatrix *matrix = pressure_mass_operator_.Is<HypreParMatrix>();
      MFEM_VERIFY(matrix != nullptr,
                  "Pressure mass matrix is not a HypreParMatrix.");
      auto *amg = new HypreBoomerAMG(*matrix);
      amg->SetPrintLevel(print_level_);
      pressure_preconditioner_.reset(amg);
   }
}

std::shared_ptr<Coefficient> StokesSolver::MakeLORViscosityCoefficient() const
{
   if (!viscosity_gf_owner_ && !viscosity_qf_owner_)
   {
      return viscosity_coefficient_;
   }
   MFEM_VERIFY(lor_velocity_space_,
               "LOR velocity space must be available before coefficient transfer.");
   if (!lor_scalar_space_)
   {
      lor_scalar_space_.reset(new ParFiniteElementSpace(lor_mesh_.get(),
                                                        lor_fec_.get()));
   }
   lor_viscosity_gf_.reset(new ParGridFunction(lor_scalar_space_.get()));
   if (viscosity_gf_owner_)
   {
      viscosity_gf_owner_->GetTrueDofs(lor_transfer_true_dofs_);
      MFEM_VERIFY(lor_transfer_true_dofs_.Size() ==
                  lor_scalar_space_->GetTrueVSize(),
                  "Cannot transfer viscosity grid function to LOR space.");
      lor_viscosity_gf_->SetFromTrueDofs(lor_transfer_true_dofs_);
   }
   else
   {
      ParFiniteElementSpace scalar_velocity_space(velocity_space_.GetParMesh(),
                                                  velocity_space_.FEColl());
      ParGridFunction projected(&scalar_velocity_space);
      projected.ProjectCoefficient(*viscosity_coefficient_);
      projected.GetTrueDofs(lor_transfer_true_dofs_);
      MFEM_VERIFY(lor_transfer_true_dofs_.Size() ==
                  lor_scalar_space_->GetTrueVSize(),
                  "Cannot transfer viscosity quadrature function to LOR space.");
      lor_viscosity_gf_->SetFromTrueDofs(lor_transfer_true_dofs_);
   }
   return std::make_shared<GridFunctionCoefficient>(lor_viscosity_gf_.get());
}

std::shared_ptr<Coefficient> StokesSolver::MakeLORBrinkmanCoefficient() const
{
   if (!brinkman_coefficient_)
   {
      return nullptr;
   }
   if (!brinkman_gf_owner_ && !brinkman_qf_owner_)
   {
      return brinkman_coefficient_;
   }
   MFEM_VERIFY(lor_velocity_space_,
               "LOR velocity space must be available before coefficient transfer.");
   if (!lor_scalar_space_)
   {
      lor_scalar_space_.reset(new ParFiniteElementSpace(lor_mesh_.get(),
                                                        lor_fec_.get()));
   }
   lor_brinkman_gf_.reset(new ParGridFunction(lor_scalar_space_.get()));
   if (brinkman_gf_owner_)
   {
      brinkman_gf_owner_->GetTrueDofs(lor_transfer_true_dofs_);
      MFEM_VERIFY(lor_transfer_true_dofs_.Size() ==
                  lor_scalar_space_->GetTrueVSize(),
                  "Cannot transfer Brinkman grid function to LOR space.");
      lor_brinkman_gf_->SetFromTrueDofs(lor_transfer_true_dofs_);
   }
   else
   {
      ParFiniteElementSpace scalar_velocity_space(velocity_space_.GetParMesh(),
                                                  velocity_space_.FEColl());
      ParGridFunction projected(&scalar_velocity_space);
      projected.ProjectCoefficient(*brinkman_coefficient_);
      projected.GetTrueDofs(lor_transfer_true_dofs_);
      MFEM_VERIFY(lor_transfer_true_dofs_.Size() ==
                  lor_scalar_space_->GetTrueVSize(),
                  "Cannot transfer Brinkman quadrature function to LOR space.");
      lor_brinkman_gf_->SetFromTrueDofs(lor_transfer_true_dofs_);
   }
   return std::make_shared<GridFunctionCoefficient>(lor_brinkman_gf_.get());
}

bool StokesSolver::HasOpenVelocityBoundary() const
{
   int local_max_attr = velocity_space_.GetParMesh()->bdr_attributes.Size()
                        ? velocity_space_.GetParMesh()->bdr_attributes.Max()
                        : 0;
   int max_attr = 0;
   MPI_Allreduce(&local_max_attr, &max_attr, 1, MPI_INT, MPI_MAX,
                 velocity_space_.GetComm());
   bool local_open = false;
   for (int attr = 1; attr <= max_attr; attr++)
   {
      if (velocity_boundary_ids_.find(attr) == velocity_boundary_ids_.end())
      {
         local_open = true;
         break;
      }
   }
   return GlobalBooleanOr(velocity_space_.GetComm(), local_open);
}

bool StokesSolver::HasPressureNullspace() const
{
   return !GlobalBooleanOr(pressure_space_.GetComm(),
                           pressure_ess_tdofs_.Size() > 0);
}

void StokesSolver::ProjectPressureMean(Vector &vector) const
{
   if (HasPressureNullspace())
   {
      ProjectPressureMeanBlock(pressure_space_.GetComm(), block_offsets_,
                               vector);
   }
}

void StokesSolver::SolveSystem(const BlockVector &rhs, BlockVector &solution,
                               bool use_boundary_values) const
{
   BuildBoundaryValues();
   constrained_rhs_.Update(block_offsets_);
   constrained_rhs_ = rhs;
   if (use_boundary_values)
   {
      constrained_rhs_.GetBlock(0).SetSubVector(
         velocity_ess_tdofs_, velocity_boundary_true_);
      constrained_rhs_.GetBlock(1).SetSubVector(
         pressure_ess_tdofs_, pressure_boundary_true_);
   }
   else
   {
      constrained_rhs_.GetBlock(0).SetSubVector(velocity_ess_tdofs_, 0.0);
      constrained_rhs_.GetBlock(1).SetSubVector(pressure_ess_tdofs_, 0.0);
   }
   ProjectPressureMean(constrained_rhs_);
   MFEM_VERIFY(iterative_solver_ != nullptr, "StokesSolver is not assembled.");
   solution = 0.0;
   iterative_solver_->Mult(constrained_rhs_, solution);
   ProjectPressureMean(solution);
   if (use_boundary_values)
   {
      solution.GetBlock(0).SetSubVector(velocity_ess_tdofs_,
                                        velocity_boundary_true_);
      solution.GetBlock(1).SetSubVector(pressure_ess_tdofs_,
                                        pressure_boundary_true_);
   }
}

BalakrishnanFractionalSolver::BalakrishnanFractionalSolver(
   ParFiniteElementSpace &fespace)
   : Solver(fespace.GetTrueVSize()),
     fespace_(fespace),
     mass_map_(fespace, fespace),
     shifted_solver_(fespace)
{
}

BalakrishnanFractionalSolver::BalakrishnanFractionalSolver(
   std::shared_ptr<ParFiniteElementSpace> fespace)
   : Solver(CheckedFESpace(fespace, "Balakrishnan").GetTrueVSize()),
     fespace_owner_(fespace),
     fespace_(CheckedFESpace(fespace, "Balakrishnan")),
     mass_map_(fespace, fespace),
     shifted_solver_(fespace)
{
}

void BalakrishnanFractionalSolver::SetFractionalPower(real_t s)
{
   MFEM_VERIFY(s > 0.0 && s < 1.0,
               "Balakrishnan fractional power must satisfy 0 < s < 1.");
   fractional_power_ = s;
}

void BalakrishnanFractionalSolver::SetQuadrature(real_t spacing,
                                                 int m, int n)
{
   MFEM_VERIFY(spacing > 0.0, "Quadrature spacing must be positive.");
   MFEM_VERIFY(m >= 0 && n >= 0,
               "Quadrature truncation counts must be nonnegative.");
   quadrature_spacing_ = spacing;
   negative_points_ = m;
   positive_points_ = n;
}

void BalakrishnanFractionalSolver::UseAdaptiveQuadrature(bool use_adaptive)
{
   use_adaptive_quadrature_ = use_adaptive;
}

void BalakrishnanFractionalSolver::SetAdaptiveQuadrature(
   real_t rel_tol, real_t abs_tol, int max_negative_points,
   int max_positive_points, int consecutive_terms)
{
   MFEM_VERIFY(rel_tol >= 0.0,
               "Adaptive quadrature relative tolerance must be nonnegative.");
   MFEM_VERIFY(abs_tol >= 0.0,
               "Adaptive quadrature absolute tolerance must be nonnegative.");
   MFEM_VERIFY(rel_tol > 0.0 || abs_tol > 0.0,
               "At least one adaptive quadrature tolerance must be positive.");
   MFEM_VERIFY(max_negative_points >= 0 && max_positive_points >= 0,
               "Adaptive quadrature caps must be nonnegative.");
   MFEM_VERIFY(consecutive_terms > 0,
               "Adaptive quadrature consecutive count must be positive.");
   adaptive_rel_tol_ = rel_tol;
   adaptive_abs_tol_ = abs_tol;
   adaptive_max_negative_points_ = max_negative_points;
   adaptive_max_positive_points_ = max_positive_points;
   adaptive_consecutive_terms_ = consecutive_terms;
}

void BalakrishnanFractionalSolver::SetQuadratureScaling(real_t scaling)
{
   MFEM_VERIFY(scaling > 0.0, "Quadrature scaling must be positive.");
   quadrature_scaling_ = scaling;
}

void BalakrishnanFractionalSolver::SetOperatorMassShift(real_t mass_shift)
{
   MFEM_VERIFY(mass_shift >= 0.0, "Operator mass shift must be nonnegative.");
   operator_mass_shift_ = mass_shift;
}

void BalakrishnanFractionalSolver::ApplyMass(const Vector &input,
                                             Vector &output) const
{
   SyncMassMapCoefficient();
   mass_map_.Mult(input, output);
}

void BalakrishnanFractionalSolver::SyncMassMapCoefficient() const
{
   const Coefficient &mass = shifted_solver_.GetBaseMassCoefficient();
   if (mass_map_coefficient_ == &mass) { return; }

   mass_map_.SetWeightCoefficient(const_cast<Coefficient &>(mass));
   mass_map_coefficient_ = &mass;
}

void BalakrishnanFractionalSolver::ApplyDiffusion(const Vector &input,
                                                  Vector &output) const
{
   const real_t old_diffusion_scale =
      shifted_solver_.GetDiffusionScalingConstant();
   const real_t old_mass_scale = shifted_solver_.GetMassScalingConstant();

   shifted_solver_.SetScalingConstants(1.0, 0.0);
   const Operator *diffusion = shifted_solver_.GetOperator();
   MFEM_VERIFY(diffusion != nullptr, "Diffusion operator is null.");
   output.SetSize(diffusion->Height());
   diffusion->Mult(input, output);

   shifted_solver_.SetScalingConstants(old_diffusion_scale, old_mass_scale);
}

void BalakrishnanFractionalSolver::ApplyGeneralizedOperator(
   const Vector &input, Vector &output) const
{
   const real_t old_diffusion_scale =
      shifted_solver_.GetDiffusionScalingConstant();
   const real_t old_mass_scale = shifted_solver_.GetMassScalingConstant();

   ApplyDiffusion(input, eigen_work_1_);
   shifted_solver_.SetScalingConstants(0.0, 1.0);
   shifted_solver_.Mult(eigen_work_1_, output);
   if (operator_mass_shift_ != 0.0)
   {
      output.Add(operator_mass_shift_, input);
   }

   shifted_solver_.SetScalingConstants(old_diffusion_scale, old_mass_scale);
}

void BalakrishnanFractionalSolver::ApplyInverseGeneralizedOperator(
   const Vector &input, Vector &output) const
{
   const real_t old_diffusion_scale =
      shifted_solver_.GetDiffusionScalingConstant();
   const real_t old_mass_scale = shifted_solver_.GetMassScalingConstant();

   ApplyMass(input, eigen_work_1_);
   shifted_solver_.SetScalingConstants(1.0, operator_mass_shift_);
   shifted_solver_.Mult(eigen_work_1_, output);

   shifted_solver_.SetScalingConstants(old_diffusion_scale, old_mass_scale);
}

real_t BalakrishnanFractionalSolver::RayleighQuotient(
   const Vector &input) const
{
   ApplyDiffusion(input, eigen_work_1_);
   ApplyMass(input, eigen_work_2_);
   const real_t numerator =
      InnerProduct(fespace_.GetComm(), input, eigen_work_1_) +
      operator_mass_shift_*InnerProduct(fespace_.GetComm(), input,
                                        eigen_work_2_);
   const real_t denominator =
      InnerProduct(fespace_.GetComm(), input, eigen_work_2_);
   MFEM_VERIFY(denominator > 0.0,
               "Mass Rayleigh denominator must be positive.");
   return numerator/denominator;
}

void BalakrishnanFractionalSolver::Normalize(Vector &x) const
{
   const real_t norm = std::sqrt(InnerProduct(fespace_.GetComm(), x, x));
   MFEM_VERIFY(norm > 0.0, "Cannot normalize a zero vector.");
   x /= norm;
}

void BalakrishnanFractionalSolver::EstimateEigenvalueBounds(
   int power_iterations, int inverse_power_iterations,
   real_t &lambda_min, real_t &lambda_max,
   real_t &suggested_scaling) const
{
   MFEM_VERIFY(power_iterations > 0,
               "Power iteration count must be positive.");
   MFEM_VERIFY(inverse_power_iterations > 0,
               "Inverse power iteration count must be positive.");

   Vector x(Width());
   for (int i = 0; i < x.Size(); i++)
   {
      x(i) = 1.0 + real_t((i % 17) + 1)/17.0;
   }
   Normalize(x);

   Vector y;
   for (int it = 0; it < power_iterations; it++)
   {
      ApplyGeneralizedOperator(x, y);
      Normalize(y);
      x = y;
   }
   lambda_max = RayleighQuotient(x);

   x.SetSize(Width());
   for (int i = 0; i < x.Size(); i++)
   {
      x(i) = 1.0 + real_t((i % 19) + 1)/19.0;
   }
   Normalize(x);

   for (int it = 0; it < inverse_power_iterations; it++)
   {
      ApplyInverseGeneralizedOperator(x, y);
      Normalize(y);
      x = y;
   }
   lambda_min = RayleighQuotient(x);
   MFEM_VERIFY(lambda_min > 0.0 && lambda_max > 0.0,
               "Estimated eigenvalues must be positive.");
   suggested_scaling = std::sqrt(lambda_min*lambda_max);
}

real_t BalakrishnanFractionalSolver::AddQuadratureTerm(
   int ell, const Vector &mass_rhs, Vector &output) const
{
   const real_t pi = 4.0*std::atan(1.0);
   const real_t y = quadrature_spacing_*ell;
   const real_t shift = quadrature_scaling_*std::exp(y);
   const real_t weight =
      quadrature_spacing_*std::sin(pi*fractional_power_)/pi*
      std::pow(quadrature_scaling_, 1.0 - fractional_power_)*
      std::exp((1.0 - fractional_power_)*y);

   shifted_solver_.SetScalingConstants(1.0, operator_mass_shift_ + shift);
   shifted_solver_.Mult(mass_rhs, shifted_solution_);
   output.Add(weight, shifted_solution_);

   const real_t shifted_norm =
      std::sqrt(InnerProduct(fespace_.GetComm(), shifted_solution_,
                             shifted_solution_));
   return std::abs(weight)*shifted_norm;
}

void BalakrishnanFractionalSolver::Mult(const Vector &input,
                                        Vector &output) const
{
   MFEM_VERIFY(input.Size() == Width(),
               "Assembled RHS vector has incompatible size.");

   output.SetSize(Height());
   output = 0.0;

   last_negative_points_ = 0;
   last_positive_points_ = 0;
   if (!use_adaptive_quadrature_)
   {
      for (int ell = -negative_points_; ell <= positive_points_; ell++)
      {
         AddQuadratureTerm(ell, input, output);
      }
      last_negative_points_ = negative_points_;
      last_positive_points_ = positive_points_;
      return;
   }

   AddQuadratureTerm(0, input, output);
   int small_negative = 0;
   int small_positive = 0;
   bool done_negative = adaptive_max_negative_points_ == 0;
   bool done_positive = adaptive_max_positive_points_ == 0;

   for (int j = 1; !(done_negative && done_positive); j++)
   {
      const real_t output_norm =
         std::sqrt(InnerProduct(fespace_.GetComm(), output, output));
      const real_t threshold =
         adaptive_abs_tol_ + adaptive_rel_tol_*output_norm;

      if (!done_negative)
      {
         const real_t term_norm = AddQuadratureTerm(-j, input, output);
         last_negative_points_ = j;
         small_negative = (term_norm <= threshold) ? small_negative + 1 : 0;
         done_negative =
            (j >= adaptive_max_negative_points_) ||
            (small_negative >= adaptive_consecutive_terms_);
      }

      if (!done_positive)
      {
         const real_t term_norm = AddQuadratureTerm(j, input, output);
         last_positive_points_ = j;
         small_positive = (term_norm <= threshold) ? small_positive + 1 : 0;
         done_positive =
            (j >= adaptive_max_positive_points_) ||
            (small_positive >= adaptive_consecutive_terms_);
      }
   }
}

void BalakrishnanFractionalSolver::MultTranspose(const Vector &input,
                                                 Vector &output) const
{
   Mult(input, output);
}

void BalakrishnanFractionalSolver::SetOperator(const Operator &op)
{
   MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
               "External operator dimensions do not match "
               "BalakrishnanFractionalSolver.");
}

FractionalDiffusionSolver::FractionalDiffusionSolver(
   ParFiniteElementSpace &fespace)
   : Solver(fespace.GetTrueVSize()),
     unit_mass_coefficient_(std::make_shared<ConstantCoefficient>(1.0)),
     balakrishnan_solver_(fespace)
{
   EnsureUnitMassCoefficient();
}

FractionalDiffusionSolver::FractionalDiffusionSolver(
   std::shared_ptr<ParFiniteElementSpace> fespace)
   : Solver(CheckedFESpace(fespace, "Fractional diffusion").GetTrueVSize()),
     unit_mass_coefficient_(std::make_shared<ConstantCoefficient>(1.0)),
     balakrishnan_solver_(fespace)
{
   EnsureUnitMassCoefficient();
}

void FractionalDiffusionSolver::SetFractionalPower(real_t s)
{
   balakrishnan_solver_.SetFractionalPower(s);
}

void FractionalDiffusionSolver::SetQuadrature(real_t spacing, int m, int n)
{
   balakrishnan_solver_.SetQuadrature(spacing, m, n);
}

void FractionalDiffusionSolver::UseAdaptiveQuadrature(bool use_adaptive)
{
   balakrishnan_solver_.UseAdaptiveQuadrature(use_adaptive);
}

void FractionalDiffusionSolver::SetAdaptiveQuadrature(
   real_t rel_tol, real_t abs_tol, int max_negative_points,
   int max_positive_points, int consecutive_terms)
{
   balakrishnan_solver_.SetAdaptiveQuadrature(rel_tol, abs_tol,
                                              max_negative_points,
                                              max_positive_points,
                                              consecutive_terms);
}

void FractionalDiffusionSolver::SetQuadratureScaling(real_t scaling)
{
   balakrishnan_solver_.SetQuadratureScaling(scaling);
}

void FractionalDiffusionSolver::EstimateEigenvalueBounds(
   int power_iterations, int inverse_power_iterations,
   real_t &lambda_min, real_t &lambda_max,
   real_t &suggested_scaling) const
{
   EnsureUnitMassCoefficient();
   balakrishnan_solver_.EstimateEigenvalueBounds(power_iterations,
                                                 inverse_power_iterations,
                                                 lambda_min,
                                                 lambda_max,
                                                 suggested_scaling);
}

void FractionalDiffusionSolver::Mult(const Vector &rhs,
                                     Vector &solution) const
{
   EnsureUnitMassCoefficient();
   balakrishnan_solver_.Mult(rhs, solution);
}

void FractionalDiffusionSolver::MultTranspose(const Vector &rhs,
                                              Vector &solution) const
{
   Mult(rhs, solution);
}

void FractionalDiffusionSolver::SetOperator(const Operator &op)
{
   MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
               "External operator dimensions do not match "
               "FractionalDiffusionSolver.");
}

void FractionalDiffusionSolver::EnsureUnitMassCoefficient() const
{
   DiffusionMassSolver &solver =
      balakrishnan_solver_.GetDiffusionMassSolver();
   if (&solver.GetBaseMassCoefficient() != unit_mass_coefficient_.get())
   {
      solver.SetMassCoefficient(unit_mass_coefficient_);
   }
}

} // namespace mfem
