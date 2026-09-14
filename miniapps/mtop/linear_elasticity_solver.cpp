// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "linear_elasticity_solver.hpp"

#include <algorithm>

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

/// Apply a fixed auxiliary solver through a vector-ordering permutation.
class ReorderedFixedPreconditioner : public Solver
{
public:
   /// Take ownership of @a solver and define its input/output orderings.
   ReorderedFixedPreconditioner(std::unique_ptr<Solver> solver, int vdim,
                                Ordering::Type outer_ordering,
                                Ordering::Type inner_ordering)
      : Solver(solver->Height(), solver->Width()), solver_(std::move(solver)),
        vdim_(vdim), outer_ordering_(outer_ordering),
        inner_ordering_(inner_ordering)
   {
   }

   /// Ignore changes to the outer high-order operator.
   void SetOperator(const Operator &) override { }

   /// Reorder, apply the auxiliary solver, and reorder the result back.
   void Mult(const Vector &x, Vector &y) const override
   {
      inner_x_ = x;
      Ordering::Reorder(inner_x_, vdim_, outer_ordering_, inner_ordering_);
      inner_y_.SetSize(Height());
      solver_->iterative_mode = iterative_mode;
      solver_->Mult(inner_x_, inner_y_);
      y = inner_y_;
      Ordering::Reorder(y, vdim_, inner_ordering_, outer_ordering_);
   }

   /// Apply the symmetric auxiliary preconditioner transpose.
   void MultTranspose(const Vector &x, Vector &y) const override
   {
      Mult(x, y);
   }

private:
   std::unique_ptr<Solver> solver_;
   int vdim_;
   Ordering::Type outer_ordering_;
   Ordering::Type inner_ordering_;
   mutable Vector inner_x_;
   mutable Vector inner_y_;
};
}

ParFiniteElementSpace &LinearElasticitySolver::CheckedFESpace(
   const std::shared_ptr<ParFiniteElementSpace> &fespace)
{
   MFEM_VERIFY(fespace != nullptr, "Finite element space pointer is null.");
   return *fespace;
}

LinearElasticitySolver::LinearElasticitySolver(
   ParFiniteElementSpace &fespace)
   : Solver(fespace.GetTrueVSize()), fespace_(fespace),
     lambda_(std::make_shared<ConstantCoefficient>(1.0)),
     mu_(std::make_shared<ConstantCoefficient>(1.0))
{
   ValidateSpace();
}

LinearElasticitySolver::LinearElasticitySolver(
   std::shared_ptr<ParFiniteElementSpace> fespace)
   : Solver(CheckedFESpace(fespace).GetTrueVSize()),
     fespace_owner_(fespace), fespace_(CheckedFESpace(fespace)),
     lambda_(std::make_shared<ConstantCoefficient>(1.0)),
     mu_(std::make_shared<ConstantCoefficient>(1.0))
{
   ValidateSpace();
}

void LinearElasticitySolver::ValidateSpace() const
{
   MFEM_VERIFY(fespace_.GetParMesh() != nullptr,
               "LinearElasticitySolver requires a parallel FE space.");
   MFEM_VERIFY(fespace_.GetVDim() == fespace_.GetParMesh()->Dimension(),
               "Elasticity space vector dimension must equal mesh dimension.");
}

std::shared_ptr<Coefficient> LinearElasticitySolver::ShareCoefficient(
   Coefficient &coefficient, bool transfer_ownership)
{
   return transfer_ownership ? std::shared_ptr<Coefficient>(&coefficient) :
   std::shared_ptr<Coefficient>(&coefficient, [](Coefficient *) { });
}

std::shared_ptr<VectorCoefficient>
LinearElasticitySolver::ShareVectorCoefficient(
   VectorCoefficient &coefficient, bool transfer_ownership)
{
   return transfer_ownership ?
          std::shared_ptr<VectorCoefficient>(&coefficient) :
          std::shared_ptr<VectorCoefficient>(&coefficient,
   [](VectorCoefficient *) { });
}

void LinearElasticitySolver::ValidateVectorCoefficient(
   int id, VectorCoefficient &coefficient) const
{
   MFEM_VERIFY(id > 0, "Attribute ids are one-based and must be positive.");
   MFEM_VERIFY(coefficient.GetVDim() == fespace_.GetVDim(),
               "Vector coefficient dimension must match the FE vector dimension.");
}

void LinearElasticitySolver::SetLambda(real_t value)
{
   SetLambda(std::make_shared<ConstantCoefficient>(value));
}

void LinearElasticitySolver::SetLambda(Coefficient &coefficient,
                                       bool transfer_ownership)
{
   SetLambda(ShareCoefficient(coefficient, transfer_ownership));
}

void LinearElasticitySolver::SetLambda(
   std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(coefficient != nullptr, "Lambda coefficient is null.");
   lambda_ = coefficient;
   SetNeedsAssembly();
}

void LinearElasticitySolver::SetMu(real_t value)
{
   SetMu(std::make_shared<ConstantCoefficient>(value));
}

void LinearElasticitySolver::SetMu(Coefficient &coefficient,
                                   bool transfer_ownership)
{
   SetMu(ShareCoefficient(coefficient, transfer_ownership));
}

void LinearElasticitySolver::SetMu(std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(coefficient != nullptr, "Mu coefficient is null.");
   mu_ = coefficient;
   SetNeedsAssembly();
}

void LinearElasticitySolver::AddBoundaryID(int id)
{
   MFEM_VERIFY(id > 0, "Boundary ids are one-based and must be positive.");
   boundary_ids_.insert(id);
   SetNeedsAssembly();
}

void LinearElasticitySolver::AddDisplacementBC(int id, int component,
                                               real_t value)
{
   AddDisplacementBC(id, component,
                     std::make_shared<ConstantCoefficient>(value));
}

void LinearElasticitySolver::AddDisplacementBC(
   int id, int component, Coefficient &coefficient, bool transfer_ownership)
{
   AddDisplacementBC(id, component,
                     ShareCoefficient(coefficient, transfer_ownership));
}

void LinearElasticitySolver::AddDisplacementBC(
   int id, int component, std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(id > 0, "Boundary ids are one-based and must be positive.");
   MFEM_VERIFY(component >= 0 && component < fespace_.GetVDim(),
               "Displacement component is outside the FE vector dimension.");
   MFEM_VERIFY(coefficient != nullptr,
               "Displacement boundary coefficient is null.");
   displacement_bcs_[std::make_pair(id, component)] = coefficient;
   SetNeedsAssembly();
}

void LinearElasticitySolver::AddDisplacementBC(
   int id, VectorCoefficient &coefficient, bool transfer_ownership)
{
   AddDisplacementBC(id,
                     ShareVectorCoefficient(coefficient, transfer_ownership));
}

void LinearElasticitySolver::AddDisplacementBC(
   int id, std::shared_ptr<VectorCoefficient> coefficient)
{
   MFEM_VERIFY(coefficient != nullptr,
               "Displacement vector coefficient is null.");
   ValidateVectorCoefficient(id, *coefficient);
   vector_displacement_bcs_[id] = coefficient;
   SetNeedsAssembly();
}

void LinearElasticitySolver::AddVolumeLoad(
   int id, VectorCoefficient &coefficient, bool transfer_ownership)
{
   AddVolumeLoad(id, ShareVectorCoefficient(coefficient,
                                            transfer_ownership));
}

void LinearElasticitySolver::AddVolumeLoad(
   int id, std::shared_ptr<VectorCoefficient> coefficient)
{
   MFEM_VERIFY(coefficient != nullptr, "Volume load coefficient is null.");
   ValidateVectorCoefficient(id, *coefficient);
   volume_loads_[id] = coefficient;
}

void LinearElasticitySolver::AddBoundaryLoad(
   int id, VectorCoefficient &coefficient, bool transfer_ownership)
{
   AddBoundaryLoad(id, ShareVectorCoefficient(coefficient,
                                              transfer_ownership));
}

void LinearElasticitySolver::AddBoundaryLoad(
   int id, std::shared_ptr<VectorCoefficient> coefficient)
{
   MFEM_VERIFY(coefficient != nullptr, "Boundary load coefficient is null.");
   ValidateVectorCoefficient(id, *coefficient);
   boundary_loads_[id] = coefficient;
}

void LinearElasticitySolver::ClearLoads()
{
   volume_loads_.clear();
   boundary_loads_.clear();
}

void LinearElasticitySolver::ClearBoundaryConditions()
{
   boundary_ids_.clear();
   displacement_bcs_.clear();
   vector_displacement_bcs_.clear();
   SetNeedsAssembly();
}

void LinearElasticitySolver::SetNeedsAssembly(bool needs_assembly) const
{
   needs_assembly_ = needs_assembly;
}

void LinearElasticitySolver::SetRelTol(real_t value)
{
   MFEM_VERIFY(value >= 0.0, "Relative tolerance must be nonnegative.");
   rel_tol_ = value;
   SetNeedsAssembly();
}

void LinearElasticitySolver::SetAbsTol(real_t value)
{
   MFEM_VERIFY(value >= 0.0, "Absolute tolerance must be nonnegative.");
   abs_tol_ = value;
   SetNeedsAssembly();
}

void LinearElasticitySolver::SetMaxIter(int value)
{
   MFEM_VERIFY(value > 0, "Maximum iteration count must be positive.");
   max_iter_ = value;
   SetNeedsAssembly();
}

void LinearElasticitySolver::SetPrintLevel(int value)
{
   print_level_ = value;
   SetNeedsAssembly();
}

void LinearElasticitySolver::SetPreconditionerType(PreconditionerType type)
{
   preconditioner_type_ = type;
   SetNeedsAssembly();
}

void LinearElasticitySolver::SetMonolithicLOROrdering(Ordering::Type ordering)
{
   MFEM_VERIFY(ordering == Ordering::byNODES ||
               ordering == Ordering::byVDIM,
               "AMG ordering must be byNODES or byVDIM.");
   monolithic_lor_ordering_ = ordering;
   SetNeedsAssembly();
}

int LinearElasticitySolver::GetNumIterations() const
{
   Assemble();
   MFEM_VERIFY(cg_ != nullptr, "LinearElasticitySolver is not assembled.");
   return cg_->GetNumIterations();
}

void LinearElasticitySolver::BuildEssentialTrueDofs() const
{
   ParMesh &mesh = *fespace_.GetParMesh();
   Array<int> marker(mesh.bdr_attributes.Size() ?
                     mesh.bdr_attributes.Max() : 0);
   marker = 0;
   for (const int id : boundary_ids_)
   {
      MFEM_VERIFY(id <= marker.Size(), "Boundary id is not present in mesh.");
      marker[id - 1] = 1;
   }
   fespace_.GetEssentialTrueDofs(marker, ess_tdofs_);

   for (const auto &entry : displacement_bcs_)
   {
      const int id = entry.first.first;
      const int component = entry.first.second;
      MFEM_VERIFY(id <= marker.Size(), "Boundary id is not present in mesh.");
      marker = 0;
      marker[id - 1] = 1;
      Array<int> component_tdofs;
      fespace_.GetEssentialTrueDofs(marker, component_tdofs, component);
      ess_tdofs_.Append(component_tdofs);
   }
   for (const auto &entry : vector_displacement_bcs_)
   {
      MFEM_VERIFY(entry.first <= marker.Size(),
                  "Boundary id is not present in mesh.");
      marker = 0;
      marker[entry.first - 1] = 1;
      Array<int> vector_tdofs;
      fespace_.GetEssentialTrueDofs(marker, vector_tdofs);
      ess_tdofs_.Append(vector_tdofs);
   }
   ess_tdofs_.Sort();
   ess_tdofs_.Unique();
}

void LinearElasticitySolver::BuildComponentBoundaryMarker(
   int component, Array<int> &marker) const
{
   ParMesh &mesh = *fespace_.GetParMesh();
   marker.SetSize(mesh.bdr_attributes.Size() ?
                  mesh.bdr_attributes.Max() : 0);
   marker = 0;
   for (const int id : boundary_ids_) { marker[id - 1] = 1; }
   for (const auto &entry : vector_displacement_bcs_)
   {
      marker[entry.first - 1] = 1;
   }
   for (const auto &entry : displacement_bcs_)
   {
      if (entry.first.second == component)
      {
         marker[entry.first.first - 1] = 1;
      }
   }
}

void LinearElasticitySolver::BuildLORDiagonalAMG() const
{
   MFEM_VERIFY(fespace_.GetOrdering() == Ordering::byNODES,
               "Diagonal LOR/AMG requires Ordering::byNODES.");
   const int dim = fespace_.GetVDim();
   lor_disc_.reset(new ParLORDiscretization(fespace_));
   ParFiniteElementSpace &lor_space = lor_disc_->GetParFESpace();
   ParMesh &lor_mesh = *lor_space.GetParMesh();
   lor_scalar_fespace_.reset(new ParFiniteElementSpace(
                                &lor_mesh, lor_space.FEColl(), 1,
                                Ordering::byNODES));
   lor_integrator_.reset(new ElasticityIntegrator(*lambda_, *mu_));
   lor_integrator_->AssemblePA(lor_space);

   lor_block_offsets_.SetSize(dim + 1);
   lor_block_offsets_[0] = 0;
   for (int component = 0; component < dim; ++component)
   {
      lor_forms_.emplace_back(new ParBilinearForm(
                                 lor_scalar_fespace_.get()));
      lor_forms_.back()->SetAssemblyLevel(AssemblyLevel::FULL);
      lor_forms_.back()->EnableSparseMatrixSorting(Device::IsEnabled());
      lor_forms_.back()->AddDomainIntegrator(
         new ElasticityComponentIntegrator(*lor_integrator_, component,
                                           component));
      lor_forms_.back()->Assemble();

      Array<int> marker, block_ess_tdofs;
      BuildComponentBoundaryMarker(component, marker);
      lor_scalar_fespace_->GetEssentialTrueDofs(marker, block_ess_tdofs);
      lor_blocks_.emplace_back(lor_forms_.back()->ParallelAssemble());
      lor_blocks_.back()->EliminateBC(block_ess_tdofs,
                                      Operator::DiagonalPolicy::DIAG_ONE);

      lor_amg_blocks_.emplace_back(new HypreBoomerAMG);
      lor_amg_blocks_.back()->SetStrengthThresh(0.25);
      lor_amg_blocks_.back()->SetRelaxType(16);
      lor_amg_blocks_.back()->SetPrintLevel(print_level_ > 1 ? 1 : 0);
      lor_amg_blocks_.back()->SetOperator(*lor_blocks_.back());
      lor_block_offsets_[component + 1] =
         lor_amg_blocks_.back()->Height();
   }
   lor_block_offsets_.PartialSum();
   MFEM_VERIFY(lor_block_offsets_.Last() == system_operator_->Height(),
               "LOR block sizes do not match the PA elasticity system.");

   std::unique_ptr<BlockDiagonalPreconditioner> block_prec(
      new BlockDiagonalPreconditioner(lor_block_offsets_));
   for (int component = 0; component < dim; ++component)
   {
      block_prec->SetDiagonalBlock(component,
                                   lor_amg_blocks_[component].get());
   }
   preconditioner_ = std::move(block_prec);
}

void LinearElasticitySolver::BuildAuxiliaryEssentialTrueDofs(
   ParFiniteElementSpace &space, Array<int> &ess_tdofs) const
{
   ParMesh &mesh = *space.GetParMesh();
   Array<int> marker(mesh.bdr_attributes.Size() ?
                     mesh.bdr_attributes.Max() : 0);
   marker = 0;
   for (const int id : boundary_ids_) { marker[id - 1] = 1; }
   for (const auto &entry : vector_displacement_bcs_)
   {
      marker[entry.first - 1] = 1;
   }
   space.GetEssentialTrueDofs(marker, ess_tdofs);
   for (const auto &entry : displacement_bcs_)
   {
      marker = 0;
      marker[entry.first.first - 1] = 1;
      Array<int> component_tdofs;
      space.GetEssentialTrueDofs(marker, component_tdofs,
                                 entry.first.second);
      ess_tdofs.Append(component_tdofs);
   }
   ess_tdofs.Sort();
   ess_tdofs.Unique();
}

void LinearElasticitySolver::BuildAMG() const
{
   amg_fespace_.reset(new ParFiniteElementSpace(
                         fespace_.GetParMesh(), fespace_.FEColl(), fespace_.GetVDim(),
                         monolithic_lor_ordering_));
   Array<int> auxiliary_ess_tdofs;
   BuildAuxiliaryEssentialTrueDofs(*amg_fespace_, auxiliary_ess_tdofs);

   amg_form_.reset(new ParBilinearForm(amg_fespace_.get()));
   amg_form_->AddDomainIntegrator(new ElasticityIntegrator(*lambda_, *mu_));
   amg_form_->Assemble();
   amg_form_->Finalize();
   amg_matrix_.reset(amg_form_->ParallelAssemble());
   amg_matrix_->EliminateBC(auxiliary_ess_tdofs,
                            Operator::DiagonalPolicy::DIAG_ONE);

   std::unique_ptr<HypreBoomerAMG> amg(new HypreBoomerAMG(*amg_matrix_));
   if (monolithic_lor_ordering_ == Ordering::byVDIM)
   {
      amg->SetElasticityOptions(amg_fespace_.get());
   }
   else
   {
      amg->SetSystemsOptions(fespace_.GetVDim(), true);
   }
   amg->SetPrintLevel(print_level_ > 1 ? 1 : 0);
   preconditioner_.reset(new ReorderedFixedPreconditioner(
                            std::move(amg), fespace_.GetVDim(), fespace_.GetOrdering(),
                            monolithic_lor_ordering_));
}

void LinearElasticitySolver::BuildLORMonolithicAMG() const
{
   lor_disc_.reset(new ParLORDiscretization(fespace_));
   ParFiniteElementSpace &base_lor_space = lor_disc_->GetParFESpace();
   ParMesh &lor_mesh = *base_lor_space.GetParMesh();
   lor_monolithic_fespace_.reset(new ParFiniteElementSpace(
                                    &lor_mesh, base_lor_space.FEColl(), fespace_.GetVDim(),
                                    monolithic_lor_ordering_));

   Array<int> lor_ess_tdofs;
   BuildAuxiliaryEssentialTrueDofs(*lor_monolithic_fespace_, lor_ess_tdofs);

   lor_monolithic_form_.reset(new ParBilinearForm(
                                 lor_monolithic_fespace_.get()));
   lor_monolithic_form_->AddDomainIntegrator(
      new ElasticityIntegrator(*lambda_, *mu_));
   lor_monolithic_form_->Assemble();
   lor_monolithic_form_->Finalize();
   lor_monolithic_matrix_.reset(lor_monolithic_form_->ParallelAssemble());
   lor_monolithic_matrix_->EliminateBC(
      lor_ess_tdofs, Operator::DiagonalPolicy::DIAG_ONE);

   std::unique_ptr<HypreBoomerAMG> amg(
      new HypreBoomerAMG(*lor_monolithic_matrix_));
   if (monolithic_lor_ordering_ == Ordering::byVDIM)
   {
      amg->SetElasticityOptions(lor_monolithic_fespace_.get());
   }
   else
   {
      amg->SetSystemsOptions(fespace_.GetVDim(), true);
   }
   amg->SetPrintLevel(print_level_ > 1 ? 1 : 0);
   preconditioner_.reset(new ReorderedFixedPreconditioner(
                            std::move(amg), fespace_.GetVDim(), fespace_.GetOrdering(),
                            monolithic_lor_ordering_));
}

void LinearElasticitySolver::BuildPreconditioner() const
{
   preconditioner_.reset();
   amg_matrix_.reset();
   amg_form_.reset();
   amg_fespace_.reset();
   lor_amg_blocks_.clear();
   lor_monolithic_matrix_.reset();
   lor_monolithic_form_.reset();
   lor_monolithic_fespace_.reset();
   lor_blocks_.clear();
   lor_forms_.clear();
   lor_integrator_.reset();
   lor_scalar_fespace_.reset();
   lor_disc_.reset();

   if (preconditioner_type_ == PreconditionerType::Jacobi)
   {
      preconditioner_.reset(
         new OperatorJacobiSmoother(*form_, ess_tdofs_));
   }
   else if (preconditioner_type_ == PreconditionerType::AMG)
   {
      BuildAMG();
   }
   else if (preconditioner_type_ == PreconditionerType::LORDiagonalAMG)
   {
      BuildLORDiagonalAMG();
   }
   else
   {
      BuildLORMonolithicAMG();
   }
}

void LinearElasticitySolver::ProjectBoundaryValues(
   ParGridFunction &solution) const
{
   ParMesh &mesh = *fespace_.GetParMesh();
   Array<int> marker(mesh.bdr_attributes.Size() ?
                     mesh.bdr_attributes.Max() : 0);
   for (const auto &entry : vector_displacement_bcs_)
   {
      marker = 0;
      marker[entry.first - 1] = 1;
      solution.ProjectBdrCoefficient(*entry.second, marker);
   }
   std::set<int> prescribed_attributes;
   for (const auto &entry : displacement_bcs_)
   {
      prescribed_attributes.insert(entry.first.first);
   }

   for (const int id : prescribed_attributes)
   {
      marker = 0;
      marker[id - 1] = 1;
      Array<Coefficient *> values(fespace_.GetVDim());
      values = nullptr;
      for (int component = 0; component < fespace_.GetVDim(); ++component)
      {
         const auto entry = displacement_bcs_.find(
                               std::make_pair(id, component));
         if (entry != displacement_bcs_.end())
         {
            values[component] = entry->second.get();
         }
      }
      solution.ProjectBdrCoefficient(values.GetData(), marker);
   }
}

void LinearElasticitySolver::BuildBoundaryTrueVector(Vector &values) const
{
   ParGridFunction boundary_values(&fespace_);
   boundary_values = 0.0;
   ProjectBoundaryValues(boundary_values);
   boundary_values.GetTrueDofs(values);
}

void LinearElasticitySolver::ZeroConstrainedDofs(Vector &vector) const
{
   const int *indices = ess_tdofs_.HostRead();
   real_t *data = vector.HostReadWrite();
   for (int i = 0; i < ess_tdofs_.Size(); ++i)
   {
      data[indices[i]] = 0.0;
   }
}

void LinearElasticitySolver::Assemble() const
{
   if (!GlobalBooleanOr(fespace_.GetComm(), needs_assembly_)) { return; }

   BuildEssentialTrueDofs();
   cg_.reset();
   preconditioner_.reset();
   system_operator_.Clear();

   StopWatch assembly_timer;
   assembly_timer.Start();
   form_.reset(new ParBilinearForm(&fespace_));
   form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form_->AddDomainIntegrator(new ElasticityIntegrator(*lambda_, *mu_));
   form_->Assemble();
   system_operator_.SetType(Operator::ANY_TYPE);
   form_->FormSystemMatrix(ess_tdofs_, system_operator_);
   assembly_timer.Stop();
   assembly_time_ = assembly_timer.RealTime();

   StopWatch prec_timer;
   prec_timer.Start();
   BuildPreconditioner();
   prec_timer.Stop();
   prec_assembly_time_ = prec_timer.RealTime();
   cg_.reset(new CGSolver(fespace_.GetComm()));
   cg_->SetRelTol(rel_tol_);
   cg_->SetAbsTol(abs_tol_);
   cg_->SetMaxIter(max_iter_ > 0 ? max_iter_ :
                   std::max(200, 2*system_operator_->Height()));
   cg_->SetPrintLevel(print_level_);
   cg_->SetOperator(*system_operator_);
   cg_->SetPreconditioner(*preconditioner_);

   MFEM_ASSERT(system_operator_->Height() == Height() &&
               system_operator_->Width() == Width(),
               "Assembled elasticity operator size does not match the solver.");
   needs_assembly_ = false;
}

void LinearElasticitySolver::Mult(const Vector &rhs, Vector &solution) const
{
   Assemble();
   MultAssembled(rhs, solution);
}

void LinearElasticitySolver::MultAssembled(const Vector &rhs,
                                           Vector &solution) const
{
   SolveForward(rhs, solution, iterative_mode);
}

void LinearElasticitySolver::SolveForward(const Vector &rhs,
                                          Vector &solution,
                                          bool use_initial_guess) const
{
   MFEM_VERIFY(cg_ != nullptr, "LinearElasticitySolver is not assembled.");
   MFEM_VERIFY(rhs.Size() == Width(), "RHS has incompatible size.");
   BuildBoundaryTrueVector(boundary_true_values_);
   solve_rhs_ = rhs;
   ConstrainedOperator *constrained =
      dynamic_cast<ConstrainedOperator *>(system_operator_.Ptr());
   MFEM_VERIFY(constrained != nullptr,
               "Elasticity system is not a constrained operator.");
   constrained->EliminateRHS(boundary_true_values_, solve_rhs_);

   solution.SetSize(Height());
   if (!use_initial_guess) { solution = 0.0; }
   const int *indices = ess_tdofs_.HostRead();
   const real_t *boundary = boundary_true_values_.HostRead();
   real_t *data = solution.HostReadWrite();
   for (int i = 0; i < ess_tdofs_.Size(); ++i)
   {
      data[indices[i]] = boundary[indices[i]];
   }
   cg_->iterative_mode = use_initial_guess;
   cg_->Mult(solve_rhs_, solution);
   data = solution.HostReadWrite();
   for (int i = 0; i < ess_tdofs_.Size(); ++i)
   {
      data[indices[i]] = boundary[indices[i]];
   }
}

void LinearElasticitySolver::MultTranspose(const Vector &rhs,
                                           Vector &solution) const
{
   Assemble();
   MultTransposeAssembled(rhs, solution);
}

void LinearElasticitySolver::MultTransposeAssembled(
   const Vector &rhs, Vector &solution) const
{
   MFEM_VERIFY(cg_ != nullptr, "LinearElasticitySolver is not assembled.");
   MFEM_VERIFY(rhs.Size() == Height(), "Adjoint RHS has incompatible size.");
   solve_rhs_ = rhs;
   ZeroConstrainedDofs(solve_rhs_);
   solution.SetSize(Width());
   if (!iterative_mode) { solution = 0.0; }
   cg_->iterative_mode = iterative_mode;
   cg_->Mult(solve_rhs_, solution);
   ZeroConstrainedDofs(solution);
}

void LinearElasticitySolver::Solve(ParGridFunction &solution) const
{
   Assemble();
   MFEM_VERIFY(solution.ParFESpace() == &fespace_,
               "Solution must use the solver finite element space.");

   ParLinearForm loads(&fespace_);
   std::unique_ptr<PWVectorCoefficient> volume;
   std::unique_ptr<PWVectorCoefficient> boundary;
   if (!volume_loads_.empty())
   {
      volume.reset(new PWVectorCoefficient(fespace_.GetVDim()));
      for (const auto &entry : volume_loads_)
      {
         volume->UpdateCoefficient(entry.first, *entry.second);
      }
      loads.AddDomainIntegrator(new VectorDomainLFIntegrator(*volume));
   }
   if (!boundary_loads_.empty())
   {
      boundary.reset(new PWVectorCoefficient(fespace_.GetVDim()));
      for (const auto &entry : boundary_loads_)
      {
         boundary->UpdateCoefficient(entry.first, *entry.second);
      }
      loads.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*boundary));
   }
   loads.Assemble();
   Vector true_rhs(fespace_.GetTrueVSize());
   loads.ParallelAssemble(true_rhs);
   if (!has_previous_solution_)
   {
      previous_solution_.SetSize(fespace_.GetTrueVSize());
      previous_solution_ = 0.0;
   }
   SolveForward(true_rhs, previous_solution_, has_previous_solution_);
   has_previous_solution_ = true;
   solution.SetFromTrueDofs(previous_solution_);
}

void LinearElasticitySolver::SetOperator(const Operator &)
{
   MFEM_ABORT("LinearElasticitySolver always uses its internally assembled "
              "PA elasticity operator; it is not a drop-in Solver for an "
              "external operator.");
}

const Array<int> &LinearElasticitySolver::GetEssentialTrueDofs() const
{
   Assemble();
   return ess_tdofs_;
}

const Operator *LinearElasticitySolver::GetOperator() const
{
   Assemble();
   return system_operator_.Ptr();
}

const Solver *LinearElasticitySolver::GetPreconditioner() const
{
   Assemble();
   return preconditioner_.get();
}

} // namespace mfem
