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
   MFEM_VERIFY(qf != nullptr, "QuadratureFunction pointer is null.");
   owner_->ValidateQuadratureFunction(*qf);
   qf_owners_[attr] = qf;
   Add(attr, owner_->MakeQuadratureCoefficient(qf));
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
   MFEM_VERIFY(gf != nullptr, "ParGridFunction pointer is null.");
   owner_->ValidateParGridFunction(*gf);
   gf_owners_[attr] = gf;
   Add(attr, owner_->MakeGridFunctionCoefficient(gf));
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
     diffusion_coefficient_(std::make_shared<ConstantCoefficient>(1.0)),
     mass_coefficient_(std::make_shared<ConstantCoefficient>(1.0)),
     integration_order_(2*fespace.GetMaxElementOrder())
{
   rhs_coefficients_.SetOwner(this, MapKind::DomainRHS);
   boundary_coefficients_.SetOwner(this, MapKind::Boundary);
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
   diffusion_coefficient_ = coefficient;
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
   diffusion_coefficient_ = MakeQuadratureCoefficient(qf);
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
   diffusion_coefficient_ = MakeGridFunctionCoefficient(gf);
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
   mass_coefficient_ = coefficient;
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
   mass_coefficient_ = MakeGridFunctionCoefficient(gf);
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
   mass_coefficient_ = MakeQuadratureCoefficient(qf);
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
   cg_solver_->SetRelTol(1e-12);
   cg_solver_->SetAbsTol(0.0);
   cg_solver_->SetMaxIter(std::max(200, 2*Height()));
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

   MFEM_VERIFY(!diffusion_qf_owner_ && !mass_qf_owner_,
               "QuadratureFunction coefficients are tied to the high-order "
               "mesh quadrature space and cannot be used to assemble the "
               "separate LOR AMG preconditioner.");
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
                                                   lor_diffusion_gf_);
   lor_mass_coefficient_ = MakeLORCoefficient(mass_coefficient_,
                                              mass_gf_owner_,
                                              lor_mass_gf_);

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
   const std::shared_ptr<ParGridFunction> &ho_gf,
   std::shared_ptr<ParGridFunction> &lor_gf) const
{
   if (!ho_gf)
   {
      return coefficient;
   }

   MFEM_VERIFY(lor_fespace_,
               "LOR finite element space must be constructed before transfer.");
   MFEM_VERIFY(lor_fespace_->GetTrueVSize() == ho_gf->ParFESpace()->GetTrueVSize(),
               "Cannot transfer ParGridFunction coefficient to LOR space: "
               "true-vector sizes differ.");

   ho_gf->GetTrueDofs(lor_transfer_true_dofs_);

   lor_gf = std::make_shared<ParGridFunction>(lor_fespace_.get());
   lor_gf->SetFromTrueDofs(lor_transfer_true_dofs_);
   return std::make_shared<GridFunctionCoefficient>(lor_gf.get());
}

} // namespace mfem
