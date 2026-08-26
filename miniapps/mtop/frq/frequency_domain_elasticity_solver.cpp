// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.

#include "frequency_domain_elasticity_solver.hpp"

#include <utility>

namespace mfem
{

namespace
{

/// Return the collective logical OR of a rank-local flag.
bool GlobalBooleanOr(const MPI_Comm communicator, const bool value)
{
   int local = value ? 1 : 0;
   int global = 0;
   MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, communicator);
   return global != 0;
}

/// Return true on rank zero of the finite element space communicator.
bool IsRoot(const MPI_Comm communicator)
{
   int rank = 0;
   MPI_Comm_rank(communicator, &rank);
   return rank == 0;
}

/// Permute vector-valued true dofs without staging data through host memory.
void ReorderVector(const Vector &input, Vector &output, const int vdim,
                   const Ordering::Type input_ordering,
                   const Ordering::Type output_ordering)
{
   MFEM_VERIFY(input.Size()%vdim == 0,
               "Vector size is incompatible with its vector dimension.");
   output.SetSize(input.Size());
   output.UseDevice(true);
   if (input_ordering == output_ordering)
   {
      output = input;
      return;
   }

   const int scalar_size = input.Size()/vdim;
   const real_t *source = input.Read();
   real_t *destination = output.Write();
   if (output_ordering == Ordering::byVDIM)
   {
      mfem::forall(input.Size(), [=] MFEM_HOST_DEVICE(int index)
      {
         const int dof = index/vdim;
         const int component = index - dof*vdim;
         destination[index] = source[dof + scalar_size*component];
      });
   }
   else
   {
      mfem::forall(input.Size(), [=] MFEM_HOST_DEVICE(int index)
      {
         const int component = index/scalar_size;
         const int dof = index - component*scalar_size;
         destination[index] = source[component + vdim*dof];
      });
   }
}

/// Convert shared engineering coefficients to the first Lame coefficient.
class SolverLameLambdaCoefficient : public Coefficient
{
public:
   /// Retain the engineering coefficients needed during evaluation.
   SolverLameLambdaCoefficient(std::shared_ptr<Coefficient> young_modulus,
                               std::shared_ptr<Coefficient> poisson_ratio)
      : young_modulus_(std::move(young_modulus)),
        poisson_ratio_(std::move(poisson_ratio)) { }

   /// Evaluate lambda=E*nu/((1+nu)(1-2nu)).
   real_t Eval(ElementTransformation &transformation,
               const IntegrationPoint &point) override
   {
      const real_t E = young_modulus_->Eval(transformation, point);
      const real_t nu = poisson_ratio_->Eval(transformation, point);
      return E*nu/((1.0 + nu)*(1.0 - 2.0*nu));
   }

private:
   std::shared_ptr<Coefficient> young_modulus_;
   std::shared_ptr<Coefficient> poisson_ratio_;
};

/// Convert shared engineering coefficients to the shear modulus.
class SolverLameMuCoefficient : public Coefficient
{
public:
   /// Retain the engineering coefficients needed during evaluation.
   SolverLameMuCoefficient(std::shared_ptr<Coefficient> young_modulus,
                           std::shared_ptr<Coefficient> poisson_ratio)
      : young_modulus_(std::move(young_modulus)),
        poisson_ratio_(std::move(poisson_ratio)) { }

   /// Evaluate mu=E/(2(1+nu)).
   real_t Eval(ElementTransformation &transformation,
               const IntegrationPoint &point) override
   {
      const real_t E = young_modulus_->Eval(transformation, point);
      const real_t nu = poisson_ratio_->Eval(transformation, point);
      return E/(2.0*(1.0 + nu));
   }

private:
   std::shared_ptr<Coefficient> young_modulus_;
   std::shared_ptr<Coefficient> poisson_ratio_;
};

/// Apply an owned auxiliary solver through an optional ordering permutation.
class ReorderedFrequencyDomainSolver : public Solver
{
public:
   /// Take ownership of @a solver and record its inner and outer orderings.
   ReorderedFrequencyDomainSolver(std::unique_ptr<Solver> solver, int vdim,
                                  Ordering::Type outer_ordering,
                                  Ordering::Type inner_ordering)
      : Solver(solver->Height(), solver->Width()), solver_(std::move(solver)),
        vdim_(vdim), outer_ordering_(outer_ordering),
        inner_ordering_(inner_ordering)
   {
      inner_x_.UseDevice(true);
      inner_y_.UseDevice(true);
   }

   /// Validate an outer operator while keeping the fixed auxiliary solver.
   void SetOperator(const Operator &op) override
   {
      MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
                  "Auxiliary solver operator has incompatible dimensions.");
   }

   /// Reorder, apply the auxiliary solver, and restore the outer ordering.
   void Mult(const Vector &x, Vector &y) const override
   {
      ReorderVector(x, inner_x_, vdim_, outer_ordering_, inner_ordering_);
      inner_y_.SetSize(Height());
      solver_->iterative_mode = false;
      solver_->Mult(inner_x_, inner_y_);
      ReorderVector(inner_y_, y, vdim_, inner_ordering_, outer_ordering_);
   }

   /// Apply the symmetric auxiliary inverse transpose.
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

} // namespace

/// Validate and dereference a shared finite element space.
ParFiniteElementSpace &FrequencyDomainLinearElasticitySolver::CheckedFESpace(
   const std::shared_ptr<ParFiniteElementSpace> &fespace)
{
   MFEM_VERIFY(fespace, "Finite element space pointer is null.");
   return *fespace;
}

/// Construct a solver borrowing its finite element space.
FrequencyDomainLinearElasticitySolver::
FrequencyDomainLinearElasticitySolver(ParFiniteElementSpace &fespace)
   : Solver(2*fespace.GetTrueVSize()), fespace_(fespace), operator_(fespace),
     lambda_(std::make_shared<ConstantCoefficient>(1.0)),
     mu_(std::make_shared<ConstantCoefficient>(1.0)),
     density_(std::make_shared<ConstantCoefficient>(1.0))
{
   ValidateSpace();
   UpdateMaterial();
   operator_.SetRayleighDamping(0.0, 0.0);
   boundary_true_values_.UseDevice(true);
   solve_rhs_.UseDevice(true);
   previous_solution_.UseDevice(true);
}

/// Construct a solver retaining its finite element space.
FrequencyDomainLinearElasticitySolver::
FrequencyDomainLinearElasticitySolver(
   std::shared_ptr<ParFiniteElementSpace> fespace)
   : Solver(2*CheckedFESpace(fespace).GetTrueVSize()),
     fespace_owner_(fespace), fespace_(CheckedFESpace(fespace)),
     operator_(fespace_),
     lambda_(std::make_shared<ConstantCoefficient>(1.0)),
     mu_(std::make_shared<ConstantCoefficient>(1.0)),
     density_(std::make_shared<ConstantCoefficient>(1.0))
{
   ValidateSpace();
   UpdateMaterial();
   operator_.SetRayleighDamping(0.0, 0.0);
   boundary_true_values_.UseDevice(true);
   solve_rhs_.UseDevice(true);
   previous_solution_.UseDevice(true);
}

/// Destroy solver objects before the operators and matrices they reference.
FrequencyDomainLinearElasticitySolver::~FrequencyDomainLinearElasticitySolver()
{
   transpose_solver_.reset();
   iterative_solver_.reset();
   transpose_preconditioner_.reset();
   preconditioner_.reset();
   h_inverse_.reset();
   h_auxiliary_preconditioner_.reset();
   direct_solver_.reset();
   lor_h_matrix_.reset();
   lor_h_form_.reset();
   lor_h_coefficients_.clear();
   lor_fespace_.reset();
   lor_discretization_.reset();
}

/// Wrap a scalar coefficient reference with optional ownership transfer.
std::shared_ptr<Coefficient>
FrequencyDomainLinearElasticitySolver::ShareCoefficient(
   Coefficient &coefficient, const bool transfer_ownership)
{
   return transfer_ownership ? std::shared_ptr<Coefficient>(&coefficient) :
          std::shared_ptr<Coefficient>(&coefficient, [](Coefficient *) { });
}

/// Wrap a vector coefficient reference with optional ownership transfer.
std::shared_ptr<VectorCoefficient>
FrequencyDomainLinearElasticitySolver::ShareVectorCoefficient(
   VectorCoefficient &coefficient, const bool transfer_ownership)
{
   return transfer_ownership ?
          std::shared_ptr<VectorCoefficient>(&coefficient) :
          std::shared_ptr<VectorCoefficient>(
             &coefficient, [](VectorCoefficient *) { });
}

/// Verify the H1 vector space required by isotropic elasticity.
void FrequencyDomainLinearElasticitySolver::ValidateSpace() const
{
   MFEM_VERIFY(fespace_.GetParMesh(),
               "A parallel finite element space is required.");
   MFEM_VERIFY(fespace_.GetVDim() ==
               fespace_.GetParMesh()->SpaceDimension(),
               "Elasticity vector dimension must equal the space dimension.");
}

/// Verify that a one-based attribute ID is present in the mesh.
void FrequencyDomainLinearElasticitySolver::ValidateAttribute(
   const int id, const bool boundary) const
{
   MFEM_VERIFY(id > 0, "Attribute IDs are one-based and must be positive.");
   const Array<int> &attributes = boundary ?
                                  fespace_.GetParMesh()->bdr_attributes :
                                  fespace_.GetParMesh()->attributes;
   MFEM_VERIFY(attributes.Find(id) >= 0,
               (boundary ? "Boundary" : "Domain")
               << " attribute ID is not present in the mesh.");
}

/// Verify an attribute ID and a vector coefficient dimension.
void FrequencyDomainLinearElasticitySolver::ValidateVectorCoefficient(
   const int id, VectorCoefficient &coefficient, const bool boundary) const
{
   ValidateAttribute(id, boundary);
   MFEM_VERIFY(coefficient.GetVDim() == fespace_.GetVDim(),
               "Vector coefficient dimension does not match the FE space.");
}

/// Push the stored Lame material into the component operator.
void FrequencyDomainLinearElasticitySolver::UpdateMaterial()
{
   operator_.SetLameMaterial(lambda_, mu_, density_);
   SetNeedsAssembly();
}

/// Set a constant first Lame coefficient.
void FrequencyDomainLinearElasticitySolver::SetLambda(const real_t value)
{
   SetLambda(std::make_shared<ConstantCoefficient>(value));
}

/// Set the first Lame coefficient from a reference.
void FrequencyDomainLinearElasticitySolver::SetLambda(
   Coefficient &coefficient, const bool transfer_ownership)
{
   SetLambda(ShareCoefficient(coefficient, transfer_ownership));
}

/// Set the first Lame coefficient from a shared pointer.
void FrequencyDomainLinearElasticitySolver::SetLambda(
   std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(coefficient, "Lambda coefficient is null.");
   lambda_ = std::move(coefficient);
   UpdateMaterial();
}

/// Set a constant shear modulus.
void FrequencyDomainLinearElasticitySolver::SetMu(const real_t value)
{
   SetMu(std::make_shared<ConstantCoefficient>(value));
}

/// Set the shear modulus from a reference.
void FrequencyDomainLinearElasticitySolver::SetMu(
   Coefficient &coefficient, const bool transfer_ownership)
{
   SetMu(ShareCoefficient(coefficient, transfer_ownership));
}

/// Set the shear modulus from a shared pointer.
void FrequencyDomainLinearElasticitySolver::SetMu(
   std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(coefficient, "Mu coefficient is null.");
   mu_ = std::move(coefficient);
   UpdateMaterial();
}

/// Set a constant mass density.
void FrequencyDomainLinearElasticitySolver::SetDensity(const real_t value)
{
   SetDensity(std::make_shared<ConstantCoefficient>(value));
}

/// Set the mass density from a reference.
void FrequencyDomainLinearElasticitySolver::SetDensity(
   Coefficient &coefficient, const bool transfer_ownership)
{
   SetDensity(ShareCoefficient(coefficient, transfer_ownership));
}

/// Set the mass density from a shared pointer.
void FrequencyDomainLinearElasticitySolver::SetDensity(
   std::shared_ptr<Coefficient> coefficient)
{
   MFEM_VERIFY(coefficient, "Density coefficient is null.");
   density_ = std::move(coefficient);
   UpdateMaterial();
}

/// Set all Lame and density coefficients from optionally owned references.
void FrequencyDomainLinearElasticitySolver::SetLameMaterial(
   Coefficient &lambda, Coefficient &mu, Coefficient &density,
   const bool transfer_ownership)
{
   std::shared_ptr<Coefficient> shared_lambda =
      ShareCoefficient(lambda, transfer_ownership);
   std::shared_ptr<Coefficient> shared_mu =
      &mu == &lambda ? shared_lambda :
      ShareCoefficient(mu, transfer_ownership);
   std::shared_ptr<Coefficient> shared_density =
      &density == &lambda ? shared_lambda :
      (&density == &mu ? shared_mu :
       ShareCoefficient(density, transfer_ownership));
   SetLameMaterial(std::move(shared_lambda), std::move(shared_mu),
                   std::move(shared_density));
}

/// Set all shared Lame and density coefficients in one update.
void FrequencyDomainLinearElasticitySolver::SetLameMaterial(
   std::shared_ptr<Coefficient> lambda,
   std::shared_ptr<Coefficient> mu,
   std::shared_ptr<Coefficient> density)
{
   MFEM_VERIFY(lambda && mu && density,
               "All isotropic material coefficients are required.");
   lambda_ = std::move(lambda);
   mu_ = std::move(mu);
   density_ = std::move(density);
   UpdateMaterial();
}

/// Set engineering material from optionally owned coefficient references.
void FrequencyDomainLinearElasticitySolver::SetEngineeringMaterial(
   Coefficient &young_modulus, Coefficient &poisson_ratio,
   Coefficient &density, const bool transfer_ownership)
{
   std::shared_ptr<Coefficient> shared_young_modulus =
      ShareCoefficient(young_modulus, transfer_ownership);
   std::shared_ptr<Coefficient> shared_poisson_ratio =
      &poisson_ratio == &young_modulus ? shared_young_modulus :
      ShareCoefficient(poisson_ratio, transfer_ownership);
   std::shared_ptr<Coefficient> shared_density =
      &density == &young_modulus ? shared_young_modulus :
      (&density == &poisson_ratio ? shared_poisson_ratio :
       ShareCoefficient(density, transfer_ownership));
   SetEngineeringMaterial(std::move(shared_young_modulus),
                          std::move(shared_poisson_ratio),
                          std::move(shared_density));
}

/// Convert shared engineering coefficients to the stored Lame material.
void FrequencyDomainLinearElasticitySolver::SetEngineeringMaterial(
   std::shared_ptr<Coefficient> young_modulus,
   std::shared_ptr<Coefficient> poisson_ratio,
   std::shared_ptr<Coefficient> density)
{
   MFEM_VERIFY(young_modulus && poisson_ratio && density,
               "All engineering material coefficients are required.");
   lambda_ = std::make_shared<SolverLameLambdaCoefficient>(
                young_modulus, poisson_ratio);
   mu_ = std::make_shared<SolverLameMuCoefficient>(
            std::move(young_modulus), std::move(poisson_ratio));
   density_ = std::move(density);
   UpdateMaterial();
}

/// Wrap distributed lambda, mu, and density fields as material coefficients.
void FrequencyDomainLinearElasticitySolver::SetLameMaterialFields(
   ParGridFunction &lambda, ParGridFunction &mu, ParGridFunction &density)
{
   lambda_ = std::make_shared<GridFunctionCoefficient>(&lambda);
   mu_ = std::make_shared<GridFunctionCoefficient>(&mu);
   density_ = std::make_shared<GridFunctionCoefficient>(&density);
   UpdateMaterial();
}

/// Wrap distributed E, nu, and density fields as material coefficients.
void FrequencyDomainLinearElasticitySolver::SetEngineeringMaterialFields(
   ParGridFunction &young_modulus, ParGridFunction &poisson_ratio,
   ParGridFunction &density)
{
   SetEngineeringMaterial(
      std::make_shared<GridFunctionCoefficient>(&young_modulus),
      std::make_shared<GridFunctionCoefficient>(&poisson_ratio),
      std::make_shared<GridFunctionCoefficient>(&density));
}

/// Set the excitation frequency and invalidate frequency-dependent setup.
void FrequencyDomainLinearElasticitySolver::SetFrequency(const real_t omega)
{
   operator_.SetFrequency(omega);
   SetNeedsAssembly();
}

/// Select Rayleigh damping and invalidate the auxiliary H setup.
void FrequencyDomainLinearElasticitySolver::SetRayleighDamping(
   const real_t alpha, const real_t beta)
{
   operator_.SetRayleighDamping(alpha, beta);
   SetNeedsAssembly();
}

/// Select independent damping through coefficient references.
void FrequencyDomainLinearElasticitySolver::SetDampingCoefficients(
   Coefficient &mass_damping, Coefficient &damping_lambda,
   Coefficient &damping_mu, const bool transfer_ownership)
{
   std::shared_ptr<Coefficient> shared_mass_damping =
      ShareCoefficient(mass_damping, transfer_ownership);
   std::shared_ptr<Coefficient> shared_damping_lambda =
      &damping_lambda == &mass_damping ? shared_mass_damping :
      ShareCoefficient(damping_lambda, transfer_ownership);
   std::shared_ptr<Coefficient> shared_damping_mu =
      &damping_mu == &mass_damping ? shared_mass_damping :
      (&damping_mu == &damping_lambda ? shared_damping_lambda :
       ShareCoefficient(damping_mu, transfer_ownership));
   operator_.SetDampingCoefficients(
      std::move(shared_mass_damping), std::move(shared_damping_lambda),
      std::move(shared_damping_mu));
   SetNeedsAssembly();
}

/// Wrap distributed c_M, lambda_C, and mu_C fields as damping coefficients.
void FrequencyDomainLinearElasticitySolver::SetDampingCoefficientFields(
   ParGridFunction &mass_damping, ParGridFunction &damping_lambda,
   ParGridFunction &damping_mu)
{
   operator_.SetDampingCoefficientFields(
      mass_damping, damping_lambda, damping_mu);
   SetNeedsAssembly();
}

/// Select independent damping through shared coefficients.
void FrequencyDomainLinearElasticitySolver::SetDampingCoefficients(
   std::shared_ptr<Coefficient> mass_damping,
   std::shared_ptr<Coefficient> damping_lambda,
   std::shared_ptr<Coefficient> damping_mu)
{
   operator_.SetDampingCoefficients(
      std::move(mass_damping), std::move(damping_lambda),
      std::move(damping_mu));
   SetNeedsAssembly();
}

/// Invalidate material-dependent data after in-place coefficient changes.
void FrequencyDomainLinearElasticitySolver::MaterialChanged()
{
   operator_.MaterialChanged();
   SetNeedsAssembly();
}

/// Add a homogeneous displacement boundary attribute.
void FrequencyDomainLinearElasticitySolver::AddBoundaryID(const int id)
{
   ValidateAttribute(id, true);
   boundary_ids_.insert(id);
   boundary_values_stale_ = true;
   SetNeedsAssembly();
}

/// Add a constant complex component displacement.
void FrequencyDomainLinearElasticitySolver::AddDisplacementBC(
   const int id, const int component, const real_t real_value,
   const real_t imaginary_value)
{
   AddDisplacementBC(
      id, component,
      std::make_shared<ConstantCoefficient>(real_value),
      std::make_shared<ConstantCoefficient>(imaginary_value));
}

/// Add a complex component displacement through coefficient references.
void FrequencyDomainLinearElasticitySolver::AddDisplacementBC(
   const int id, const int component, Coefficient &real_coefficient,
   Coefficient &imaginary_coefficient, const bool transfer_ownership)
{
   std::shared_ptr<Coefficient> shared_real =
      ShareCoefficient(real_coefficient, transfer_ownership);
   std::shared_ptr<Coefficient> shared_imaginary =
      &imaginary_coefficient == &real_coefficient ? shared_real :
      ShareCoefficient(imaginary_coefficient, transfer_ownership);
   AddDisplacementBC(
      id, component, std::move(shared_real), std::move(shared_imaginary));
}

/// Add a complex component displacement through shared coefficients.
void FrequencyDomainLinearElasticitySolver::AddDisplacementBC(
   const int id, const int component,
   std::shared_ptr<Coefficient> real_coefficient,
   std::shared_ptr<Coefficient> imaginary_coefficient)
{
   ValidateAttribute(id, true);
   MFEM_VERIFY(component >= 0 && component < fespace_.GetVDim(),
               "Displacement component is outside the FE vector dimension.");
   MFEM_VERIFY(real_coefficient && imaginary_coefficient,
               "Both complex displacement coefficients are required.");
   displacement_bcs_[std::make_pair(id, component)] =
      {std::move(real_coefficient), std::move(imaginary_coefficient)};
   boundary_values_stale_ = true;
   SetNeedsAssembly();
}

/// Add a real vector displacement through a coefficient reference.
void FrequencyDomainLinearElasticitySolver::AddDisplacementBC(
   const int id, VectorCoefficient &real_coefficient,
   const bool transfer_ownership)
{
   AddDisplacementBC(id,
                     ShareVectorCoefficient(real_coefficient,
                                            transfer_ownership),
                     nullptr);
}

/// Add a complex vector displacement through coefficient references.
void FrequencyDomainLinearElasticitySolver::AddDisplacementBC(
   const int id, VectorCoefficient &real_coefficient,
   VectorCoefficient &imaginary_coefficient,
   const bool transfer_ownership)
{
   std::shared_ptr<VectorCoefficient> shared_real =
      ShareVectorCoefficient(real_coefficient, transfer_ownership);
   std::shared_ptr<VectorCoefficient> shared_imaginary =
      &imaginary_coefficient == &real_coefficient ? shared_real :
      ShareVectorCoefficient(imaginary_coefficient, transfer_ownership);
   AddDisplacementBC(
      id, std::move(shared_real), std::move(shared_imaginary));
}

/// Add a complex vector displacement through shared coefficients.
void FrequencyDomainLinearElasticitySolver::AddDisplacementBC(
   const int id, std::shared_ptr<VectorCoefficient> real_coefficient,
   std::shared_ptr<VectorCoefficient> imaginary_coefficient)
{
   MFEM_VERIFY(real_coefficient || imaginary_coefficient,
               "At least one displacement coefficient is required.");
   if (real_coefficient)
   {
      ValidateVectorCoefficient(id, *real_coefficient, true);
   }
   if (imaginary_coefficient)
   {
      ValidateVectorCoefficient(id, *imaginary_coefficient, true);
   }
   vector_displacement_bcs_[id] =
      {std::move(real_coefficient), std::move(imaginary_coefficient)};
   boundary_values_stale_ = true;
   SetNeedsAssembly();
}

/// Add a real volume load through a coefficient reference.
void FrequencyDomainLinearElasticitySolver::AddVolumeLoad(
   const int id, VectorCoefficient &real_coefficient,
   const bool transfer_ownership)
{
   AddVolumeLoad(id,
                 ShareVectorCoefficient(real_coefficient,
                                        transfer_ownership), nullptr);
}

/// Add a complex volume load through coefficient references.
void FrequencyDomainLinearElasticitySolver::AddVolumeLoad(
   const int id, VectorCoefficient &real_coefficient,
   VectorCoefficient &imaginary_coefficient,
   const bool transfer_ownership)
{
   std::shared_ptr<VectorCoefficient> shared_real =
      ShareVectorCoefficient(real_coefficient, transfer_ownership);
   std::shared_ptr<VectorCoefficient> shared_imaginary =
      &imaginary_coefficient == &real_coefficient ? shared_real :
      ShareVectorCoefficient(imaginary_coefficient, transfer_ownership);
   AddVolumeLoad(
      id, std::move(shared_real), std::move(shared_imaginary));
}

/// Add a complex volume load through shared coefficients.
void FrequencyDomainLinearElasticitySolver::AddVolumeLoad(
   const int id, std::shared_ptr<VectorCoefficient> real_coefficient,
   std::shared_ptr<VectorCoefficient> imaginary_coefficient)
{
   MFEM_VERIFY(real_coefficient || imaginary_coefficient,
               "At least one volume-load coefficient is required.");
   if (real_coefficient)
   {
      ValidateVectorCoefficient(id, *real_coefficient, false);
   }
   if (imaginary_coefficient)
   {
      ValidateVectorCoefficient(id, *imaginary_coefficient, false);
   }
   volume_loads_[id] =
      {std::move(real_coefficient), std::move(imaginary_coefficient)};
}

/// Add a real boundary traction through a coefficient reference.
void FrequencyDomainLinearElasticitySolver::AddBoundaryLoad(
   const int id, VectorCoefficient &real_coefficient,
   const bool transfer_ownership)
{
   AddBoundaryLoad(id,
                   ShareVectorCoefficient(real_coefficient,
                                          transfer_ownership), nullptr);
}

/// Add a complex boundary traction through coefficient references.
void FrequencyDomainLinearElasticitySolver::AddBoundaryLoad(
   const int id, VectorCoefficient &real_coefficient,
   VectorCoefficient &imaginary_coefficient,
   const bool transfer_ownership)
{
   std::shared_ptr<VectorCoefficient> shared_real =
      ShareVectorCoefficient(real_coefficient, transfer_ownership);
   std::shared_ptr<VectorCoefficient> shared_imaginary =
      &imaginary_coefficient == &real_coefficient ? shared_real :
      ShareVectorCoefficient(imaginary_coefficient, transfer_ownership);
   AddBoundaryLoad(
      id, std::move(shared_real), std::move(shared_imaginary));
}

/// Add a complex boundary traction through shared coefficients.
void FrequencyDomainLinearElasticitySolver::AddBoundaryLoad(
   const int id, std::shared_ptr<VectorCoefficient> real_coefficient,
   std::shared_ptr<VectorCoefficient> imaginary_coefficient)
{
   MFEM_VERIFY(real_coefficient || imaginary_coefficient,
               "At least one boundary-load coefficient is required.");
   if (real_coefficient)
   {
      ValidateVectorCoefficient(id, *real_coefficient, true);
   }
   if (imaginary_coefficient)
   {
      ValidateVectorCoefficient(id, *imaginary_coefficient, true);
   }
   boundary_loads_[id] =
      {std::move(real_coefficient), std::move(imaginary_coefficient)};
}

/// Remove all configured loads without rebuilding the operator.
void FrequencyDomainLinearElasticitySolver::ClearLoads()
{
   volume_loads_.clear();
   boundary_loads_.clear();
}

/// Remove all displacement conditions and invalidate constrained setup.
void FrequencyDomainLinearElasticitySolver::ClearBoundaryConditions()
{
   boundary_ids_.clear();
   displacement_bcs_.clear();
   vector_displacement_bcs_.clear();
   boundary_values_stale_ = true;
   SetNeedsAssembly();
}

/// Select the requested outer linear solver.
void FrequencyDomainLinearElasticitySolver::SetLinearSolverType(
   const LinearSolverType type)
{
   linear_solver_type_ = type;
   SetNeedsAssembly();
}

/// Select the requested block preconditioner.
void FrequencyDomainLinearElasticitySolver::SetPreconditionerType(
   const PreconditionerType type)
{
   preconditioner_type_ = type;
   SetNeedsAssembly();
}

/// Select the requested inverse backend for H.
void FrequencyDomainLinearElasticitySolver::SetHInverseType(
   const HInverseType type)
{
   h_inverse_type_ = type;
   SetNeedsAssembly();
}

/// Select the monolithic LOR vector ordering.
void FrequencyDomainLinearElasticitySolver::SetLOROrdering(
   const Ordering::Type ordering)
{
   MFEM_VERIFY(ordering == Ordering::byNODES || ordering == Ordering::byVDIM,
               "LOR ordering must be byNODES or byVDIM.");
   lor_ordering_ = ordering;
   SetNeedsAssembly();
}

/// Set the outer relative tolerance.
void FrequencyDomainLinearElasticitySolver::SetRelTol(const real_t value)
{
   MFEM_VERIFY(value >= 0.0, "Relative tolerance must be nonnegative.");
   rel_tol_ = value;
   SetNeedsAssembly();
}

/// Set the outer absolute tolerance.
void FrequencyDomainLinearElasticitySolver::SetAbsTol(const real_t value)
{
   MFEM_VERIFY(value >= 0.0, "Absolute tolerance must be nonnegative.");
   abs_tol_ = value;
   SetNeedsAssembly();
}

/// Set the outer maximum iteration count.
void FrequencyDomainLinearElasticitySolver::SetMaxIter(const int value)
{
   MFEM_VERIFY(value > 0, "Maximum iterations must be positive.");
   max_iter_ = value;
   SetNeedsAssembly();
}

/// Set the GMRES restart dimension.
void FrequencyDomainLinearElasticitySolver::SetKDim(const int value)
{
   MFEM_VERIFY(value > 0, "Krylov restart dimension must be positive.");
   kdim_ = value;
   SetNeedsAssembly();
}

/// Set the outer print level.
void FrequencyDomainLinearElasticitySolver::SetPrintLevel(const int value)
{
   print_level_ = value;
   SetNeedsAssembly();
}

/// Set the nested H-solve relative tolerance.
void FrequencyDomainLinearElasticitySolver::SetPreconditionerRelTol(
   const real_t value)
{
   MFEM_VERIFY(value >= 0.0,
               "Preconditioner relative tolerance must be nonnegative.");
   preconditioner_rel_tol_ = value;
   SetNeedsAssembly();
}

/// Set the nested H-solve absolute tolerance.
void FrequencyDomainLinearElasticitySolver::SetPreconditionerAbsTol(
   const real_t value)
{
   MFEM_VERIFY(value >= 0.0,
               "Preconditioner absolute tolerance must be nonnegative.");
   preconditioner_abs_tol_ = value;
   SetNeedsAssembly();
}

/// Set the nested H-solve maximum iteration count.
void FrequencyDomainLinearElasticitySolver::SetPreconditionerMaxIter(
   const int value)
{
   MFEM_VERIFY(value > 0,
               "Preconditioner maximum iterations must be positive.");
   preconditioner_max_iter_ = value;
   SetNeedsAssembly();
}

/// Set the nested H-solve and AMG print level.
void FrequencyDomainLinearElasticitySolver::SetPreconditionerPrintLevel(
   const int value)
{
   preconditioner_print_level_ = value;
   SetNeedsAssembly();
}

/// Set or clear the collective lazy-assembly flag.
void FrequencyDomainLinearElasticitySolver::SetNeedsAssembly(
   const bool value) const
{
   if (value)
   {
      // Drop every object that retains non-owning references to operator or
      // auxiliary-matrix state. This also makes setter calls safe before the
      // next collective lazy assembly.
      transpose_solver_.reset();
      iterative_solver_.reset();
      transpose_preconditioner_.reset();
      preconditioner_.reset();
      h_inverse_.reset();
      h_auxiliary_preconditioner_.reset();
      transpose_operator_.reset();
      system_operator_.reset();
      direct_solver_.reset();
      direct_matrix_.reset();
      assembled_complex_.reset();
      h_matrix_.reset();
      lor_h_matrix_.reset();
      lor_h_form_.reset();
      lor_h_coefficients_.clear();
      lor_fespace_.reset();
      lor_discretization_.reset();
      has_previous_solution_ = false;
      num_iterations_ = 0;
      // Note: boundary_values_stale_ is NOT reset here because boundary
      // conditions are independent of operator assembly. It's managed
      // separately through AddBoundaryID(), AddDisplacementBC(), etc.
   }
   needs_assembly_ = value;
}

/// Build the current component-aware essential true-dof list.
void FrequencyDomainLinearElasticitySolver::BuildEssentialTrueDofs() const
{
   ParMesh &mesh = *fespace_.GetParMesh();
   Array<int> marker(mesh.bdr_attributes.Size() ?
                     mesh.bdr_attributes.Max() : 0);
   marker = 0;
   for (const int id : boundary_ids_)
   {
      MFEM_VERIFY(id <= marker.Size(), "Boundary ID is not present in mesh.");
      marker[id - 1] = 1;
   }
   fespace_.GetEssentialTrueDofs(marker, ess_tdofs_);

   for (const auto &entry : vector_displacement_bcs_)
   {
      MFEM_VERIFY(entry.first <= marker.Size(),
                  "Boundary ID is not present in mesh.");
      marker = 0;
      marker[entry.first - 1] = 1;
      Array<int> dofs;
      fespace_.GetEssentialTrueDofs(marker, dofs);
      ess_tdofs_.Append(dofs);
   }
   for (const auto &entry : displacement_bcs_)
   {
      MFEM_VERIFY(entry.first.first <= marker.Size(),
                  "Boundary ID is not present in mesh.");
      marker = 0;
      marker[entry.first.first - 1] = 1;
      Array<int> dofs;
      fespace_.GetEssentialTrueDofs(marker, dofs, entry.first.second);
      ess_tdofs_.Append(dofs);
   }
   ess_tdofs_.Sort();
   ess_tdofs_.Unique();
}

/// Build essential true dofs on a monolithic auxiliary vector space.
void FrequencyDomainLinearElasticitySolver::
BuildAuxiliaryEssentialTrueDofs(ParFiniteElementSpace &space,
                                Array<int> &ess_tdofs) const
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
      Array<int> component_dofs;
      space.GetEssentialTrueDofs(marker, component_dofs,
                                 entry.first.second);
      ess_tdofs.Append(component_dofs);
   }
   ess_tdofs.Sort();
   ess_tdofs.Unique();
}

/// Project complex prescribed displacement values into true-dof blocks.
void FrequencyDomainLinearElasticitySolver::BuildBoundaryTrueVector(
   Vector &values) const
{
   ParGridFunction real_values(&fespace_);
   ParGridFunction imaginary_values(&fespace_);
   real_values = 0.0;
   imaginary_values = 0.0;
   ParMesh &mesh = *fespace_.GetParMesh();
   Array<int> marker(mesh.bdr_attributes.Size() ?
                     mesh.bdr_attributes.Max() : 0);

   for (const auto &entry : vector_displacement_bcs_)
   {
      marker = 0;
      marker[entry.first - 1] = 1;
      if (entry.second.real)
      {
         real_values.ProjectBdrCoefficient(*entry.second.real, marker);
      }
      if (entry.second.imaginary)
      {
         imaginary_values.ProjectBdrCoefficient(
            *entry.second.imaginary, marker);
      }
   }

   std::set<int> attributes;
   for (const auto &entry : displacement_bcs_)
   {
      attributes.insert(entry.first.first);
   }
   for (const int id : attributes)
   {
      marker = 0;
      marker[id - 1] = 1;
      Array<Coefficient *> real_components(fespace_.GetVDim());
      Array<Coefficient *> imaginary_components(fespace_.GetVDim());
      real_components = nullptr;
      imaginary_components = nullptr;
      for (int component = 0; component < fespace_.GetVDim(); ++component)
      {
         const auto entry = displacement_bcs_.find(
                               std::make_pair(id, component));
         if (entry != displacement_bcs_.end())
         {
            real_components[component] = entry->second.real.get();
            imaginary_components[component] = entry->second.imaginary.get();
         }
      }
      real_values.ProjectBdrCoefficient(real_components.GetData(), marker);
      imaginary_values.ProjectBdrCoefficient(
         imaginary_components.GetData(), marker);
   }

   Vector real_true, imaginary_true;
   real_values.GetTrueDofs(real_true);
   imaginary_values.GetTrueDofs(imaginary_true);
   values.SetSize(2*fespace_.GetTrueVSize());
   values.UseDevice(true);
   values.Write();
   Vector real_block;
   Vector imaginary_block;
   real_block.MakeRef(values, 0, fespace_.GetTrueVSize());
   imaginary_block.MakeRef(values, fespace_.GetTrueVSize(),
                           fespace_.GetTrueVSize());
   real_block = real_true;
   imaginary_block = imaginary_true;
   real_block.SyncAliasMemory(values);
   imaginary_block.SyncAliasMemory(values);
}

/// Assemble configured complex volume loads and boundary tractions.
void FrequencyDomainLinearElasticitySolver::BuildLoadTrueVector(
   Vector &values) const
{
   ParLinearForm real_form(&fespace_);
   ParLinearForm imaginary_form(&fespace_);
   std::unique_ptr<PWVectorCoefficient> real_volume;
   std::unique_ptr<PWVectorCoefficient> imaginary_volume;
   std::unique_ptr<PWVectorCoefficient> real_boundary;
   std::unique_ptr<PWVectorCoefficient> imaginary_boundary;

   if (!volume_loads_.empty())
   {
      real_volume.reset(new PWVectorCoefficient(fespace_.GetVDim()));
      imaginary_volume.reset(new PWVectorCoefficient(fespace_.GetVDim()));
      bool has_real = false;
      bool has_imaginary = false;
      for (const auto &entry : volume_loads_)
      {
         if (entry.second.real)
         {
            real_volume->UpdateCoefficient(entry.first, *entry.second.real);
            has_real = true;
         }
         if (entry.second.imaginary)
         {
            imaginary_volume->UpdateCoefficient(
               entry.first, *entry.second.imaginary);
            has_imaginary = true;
         }
      }
      if (has_real)
      {
         real_form.AddDomainIntegrator(
            new VectorDomainLFIntegrator(*real_volume));
      }
      if (has_imaginary)
      {
         imaginary_form.AddDomainIntegrator(
            new VectorDomainLFIntegrator(*imaginary_volume));
      }
   }
   if (!boundary_loads_.empty())
   {
      real_boundary.reset(new PWVectorCoefficient(fespace_.GetVDim()));
      imaginary_boundary.reset(new PWVectorCoefficient(fespace_.GetVDim()));
      bool has_real = false;
      bool has_imaginary = false;
      for (const auto &entry : boundary_loads_)
      {
         if (entry.second.real)
         {
            real_boundary->UpdateCoefficient(entry.first, *entry.second.real);
            has_real = true;
         }
         if (entry.second.imaginary)
         {
            imaginary_boundary->UpdateCoefficient(
               entry.first, *entry.second.imaginary);
            has_imaginary = true;
         }
      }
      if (has_real)
      {
         real_form.AddBoundaryIntegrator(
            new VectorBoundaryLFIntegrator(*real_boundary));
      }
      if (has_imaginary)
      {
         imaginary_form.AddBoundaryIntegrator(
            new VectorBoundaryLFIntegrator(*imaginary_boundary));
      }
   }

   real_form.Assemble();
   imaginary_form.Assemble();
   Vector real_true(fespace_.GetTrueVSize());
   Vector imaginary_true(fespace_.GetTrueVSize());
   real_form.ParallelAssemble(real_true);
   imaginary_form.ParallelAssemble(imaginary_true);
   values.SetSize(2*fespace_.GetTrueVSize());
   values.UseDevice(true);
   values.Write();
   Vector real_block;
   Vector imaginary_block;
   real_block.MakeRef(values, 0, fespace_.GetTrueVSize());
   imaginary_block.MakeRef(values, fespace_.GetTrueVSize(),
                           fespace_.GetTrueVSize());
   real_block = real_true;
   imaginary_block = imaginary_true;
   real_block.SyncAliasMemory(values);
   imaginary_block.SyncAliasMemory(values);
}

/// Assemble a monolithic LOR approximation of H=W+T.
void FrequencyDomainLinearElasticitySolver::BuildLORHMatrix() const
{
   lor_discretization_.reset(new ParLORDiscretization(fespace_));
   ParFiniteElementSpace &base_space =
      lor_discretization_->GetParFESpace();
   ParMesh &lor_mesh = *base_space.GetParMesh();
   lor_fespace_.reset(new ParFiniteElementSpace(
                         &lor_mesh, base_space.FEColl(), fespace_.GetVDim(),
                         lor_ordering_));

   Coefficient &lambda = const_cast<Coefficient &>(
                            operator_.GetLambdaCoefficient());
   Coefficient &mu = const_cast<Coefficient &>(
                        operator_.GetMuCoefficient());
   Coefficient &density = const_cast<Coefficient &>(
                             operator_.GetDensityCoefficient());
   const real_t omega = operator_.GetFrequency();
   lor_h_coefficients_.clear();
   if (operator_.HasIndependentDamping())
   {
      Coefficient &damping_lambda = const_cast<Coefficient &>(
         operator_.GetDampingLambdaCoefficient());
      Coefficient &damping_mu = const_cast<Coefficient &>(
         operator_.GetDampingMuCoefficient());
      Coefficient &mass_damping = const_cast<Coefficient &>(
         operator_.GetMassDampingCoefficient());
      lor_h_coefficients_.emplace_back(
         new SumCoefficient(lambda, damping_lambda, 1.0, omega));
      lor_h_coefficients_.emplace_back(
         new SumCoefficient(mu, damping_mu, 1.0, omega));
      lor_h_coefficients_.emplace_back(
         new SumCoefficient(density, mass_damping,
                            -omega*omega, omega));
   }
   else
   {
      lor_h_coefficients_.emplace_back(
         new ProductCoefficient(1.0 + omega*operator_.GetRayleighBeta(),
                                lambda));
      lor_h_coefficients_.emplace_back(
         new ProductCoefficient(1.0 + omega*operator_.GetRayleighBeta(), mu));
      lor_h_coefficients_.emplace_back(
         new ProductCoefficient(
            omega*operator_.GetRayleighAlpha() - omega*omega, density));
   }

   lor_h_form_.reset(new ParBilinearForm(lor_fespace_.get()));
   lor_h_form_->EnableSparseMatrixSorting(Device::IsEnabled());
   lor_h_form_->AddDomainIntegrator(
      new ElasticityIntegrator(*lor_h_coefficients_[0],
                               *lor_h_coefficients_[1]));
   lor_h_form_->AddDomainIntegrator(
      new VectorMassIntegrator(*lor_h_coefficients_[2]));
   lor_h_form_->Assemble();
   lor_h_form_->Finalize();
   lor_h_matrix_.reset(lor_h_form_->ParallelAssemble());
   Array<int> lor_ess_tdofs;
   BuildAuxiliaryEssentialTrueDofs(*lor_fespace_, lor_ess_tdofs);
   lor_h_matrix_->EliminateBC(lor_ess_tdofs,
                              Operator::DiagonalPolicy::DIAG_ONE);
   MFEM_VERIFY(lor_h_matrix_->Height() == fespace_.GetTrueVSize(),
               "LOR and high-order true-dof sizes do not match.");
}

/// Construct fixed AMG, nested CG/AMG, or exact MUMPS for H.
void FrequencyDomainLinearElasticitySolver::BuildHInverse() const
{
   h_inverse_.reset();
   h_auxiliary_preconditioner_.reset();
   h_matrix_.reset();
   lor_h_matrix_.reset();
   lor_h_form_.reset();
   lor_fespace_.reset();
   lor_discretization_.reset();
   lor_h_coefficients_.clear();

   if (h_inverse_type_ == HInverseType::MUMPS)
   {
#ifdef MFEM_USE_MUMPS
      h_matrix_ = operator_.FormHMatrix();
      std::unique_ptr<MUMPSSolver> mumps(
         new MUMPSSolver(h_matrix_->GetComm()));
      mumps->SetPrintLevel(preconditioner_print_level_ > 0 ?
                           preconditioner_print_level_ : 0);
      mumps->SetMatrixSymType(
         MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
      mumps->SetOperator(*h_matrix_);
      h_inverse_ = std::move(mumps);
      if (Device::IsEnabled() && preconditioner_print_level_ >= 0 &&
          IsRoot(fespace_.GetComm()))
      {
         mfem::out << "Frequency-domain H MUMPS inverse uses a host "
                   << "fallback.\n";
      }
#else
      MFEM_ABORT("MFEM was not built with MUMPS support.");
#endif
      return;
   }

   BuildLORHMatrix();
   std::unique_ptr<HypreBoomerAMG> amg(
      new HypreBoomerAMG(*lor_h_matrix_));
   amg->SetSystemsOptions(fespace_.GetVDim(),
                          lor_ordering_ == Ordering::byNODES);
   // A symmetric point relaxation makes one Galerkin V-cycle suitable as the
   // SPD diagonal preconditioner required by MINRES. l1-Jacobi is supported
   // by both CPU and GPU hypre builds.
   amg->SetRelaxType(18);
   amg->SetTol(0.0);
   amg->SetMaxIter(1);
   amg->SetPrintLevel(preconditioner_print_level_ >= 0 ?
                      preconditioner_print_level_ : 0);
   std::unique_ptr<Solver> reordered(
      new ReorderedFrequencyDomainSolver(
         std::move(amg), fespace_.GetVDim(), fespace_.GetOrdering(),
         lor_ordering_));

   if (h_inverse_type_ == HInverseType::LORMonolithicAMG)
   {
      h_inverse_ = std::move(reordered);
      return;
   }

   h_auxiliary_preconditioner_ = std::move(reordered);
   std::unique_ptr<CGSolver> cg(new CGSolver(fespace_.GetComm()));

   cg->SetRelTol(preconditioner_rel_tol_);
   cg->SetAbsTol(preconditioner_abs_tol_);
   cg->SetMaxIter(preconditioner_max_iter_);
   cg->SetPrintLevel(preconditioner_print_level_);
   cg->SetOperator(operator_.GetHOperator());
   cg->SetPreconditioner(*h_auxiliary_preconditioner_);
   cg->iterative_mode = false;

   h_inverse_ = std::move(cg);
}

/// Construct the selected block preconditioner and report its traits.
FrequencyDomainLinearElasticitySolver::PreconditionerTraits
FrequencyDomainLinearElasticitySolver::BuildBlockPreconditioner() const
{
   MFEM_VERIFY(h_inverse_, "H inverse must be constructed first.");
   PreconditionerTraits traits;
   traits.variable =
      h_inverse_type_ == HInverseType::LORMonolithicCGAMG;
   if (preconditioner_type_ == PreconditionerType::PRESB)
   {
      traits.convention = ComplexOperator::HERMITIAN;
      traits.symmetric_positive_definite = false;
      preconditioner_.reset(new PRESBPreconditioner(
                               operator_.GetTOperator(), *h_inverse_, 1));
   }
   else
   {
      traits.convention = ComplexOperator::BLOCK_SYMMETRIC;
      traits.symmetric_positive_definite = !traits.variable;
      preconditioner_.reset(
         new RealBlockDiagonalPreconditioner(*h_inverse_));
   }
   return traits;
}

/// Resolve Automatic and reject incompatible explicit Krylov choices.
FrequencyDomainLinearElasticitySolver::LinearSolverType
FrequencyDomainLinearElasticitySolver::ResolveLinearSolver(
   const PreconditionerTraits &traits) const
{
   LinearSolverType type = linear_solver_type_;
   if (type == LinearSolverType::Automatic)
   {
      if (traits.variable) { return LinearSolverType::FGMRES; }
      return traits.convention == ComplexOperator::BLOCK_SYMMETRIC ?
             LinearSolverType::MINRES : LinearSolverType::GMRES;
   }
   MFEM_VERIFY(type != LinearSolverType::MUMPS,
               "MUMPS does not use an iterative preconditioner.");
   MFEM_VERIFY(type != LinearSolverType::GMRES || !traits.variable,
               "Variable preconditioning requires FGMRES.");
   if (type == LinearSolverType::MINRES)
   {
      MFEM_VERIFY(traits.convention == ComplexOperator::BLOCK_SYMMETRIC &&
                  traits.symmetric_positive_definite && !traits.variable,
                  "MINRES requires a fixed SPD preconditioner and a symmetric "
                  "real formulation.");
   }
   return type;
}

/// Construct and configure a requested MFEM iterative solver.
std::unique_ptr<IterativeSolver>
FrequencyDomainLinearElasticitySolver::BuildIterativeSolver(
   const LinearSolverType type, const Operator &system,
   Solver &preconditioner) const
{
   std::unique_ptr<IterativeSolver> solver;
   if (type == LinearSolverType::GMRES)
   {
      std::unique_ptr<GMRESSolver> gmres(new GMRESSolver(fespace_.GetComm()));
      gmres->SetKDim(kdim_);
      solver = std::move(gmres);
   }
   else if (type == LinearSolverType::FGMRES)
   {
      std::unique_ptr<FGMRESSolver> fgmres(
         new FGMRESSolver(fespace_.GetComm()));
      fgmres->SetKDim(kdim_);
      solver = std::move(fgmres);
   }
   else
   {
      MFEM_VERIFY(type == LinearSolverType::MINRES,
                  "Unsupported iterative solver type.");
      solver.reset(new MINRESSolver(fespace_.GetComm()));
   }
   solver->SetRelTol(rel_tol_);
   solver->SetAbsTol(abs_tol_);
   solver->SetMaxIter(max_iter_);
   solver->SetPrintLevel(print_level_);
   preconditioner.SetOperator(system);
   solver->SetOperator(system);
   solver->SetPreconditioner(preconditioner);
   return solver;
}

/// Construct the assembled block-symmetric MUMPS reference solver.
void FrequencyDomainLinearElasticitySolver::BuildDirectSolver() const
{
#ifdef MFEM_USE_MUMPS
   active_convention_ = ComplexOperator::BLOCK_SYMMETRIC;
   assembled_complex_ = operator_.FormAssembledComplexOperator(
                           active_convention_);
   direct_matrix_.reset(assembled_complex_->GetSystemMatrix());
   std::unique_ptr<MUMPSSolver> mumps(
      new MUMPSSolver(direct_matrix_->GetComm()));
   mumps->SetPrintLevel(print_level_ > 0 ? print_level_ : 0);
   mumps->SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_INDEFINITE);
   mumps->SetOperator(*direct_matrix_);
   direct_solver_ = std::move(mumps);
   active_solver_type_ = LinearSolverType::MUMPS;
   if (Device::IsEnabled() && print_level_ >= 0 &&
       IsRoot(fespace_.GetComm()))
   {
      mfem::out << "Frequency-domain MUMPS solve uses a host fallback.\n";
   }
#else
   MFEM_ABORT("MFEM was not built with MUMPS support.");
#endif
}

/// Assemble all frequency-domain solver state collectively.
void FrequencyDomainLinearElasticitySolver::Assemble() const
{
   if (!GlobalBooleanOr(fespace_.GetComm(), needs_assembly_)) { return; }

   // Release objects in dependency order before refreshing their operators.
   transpose_solver_.reset();
   iterative_solver_.reset();
   transpose_preconditioner_.reset();
   preconditioner_.reset();
   transpose_operator_.reset();
   system_operator_.reset();
   direct_solver_.reset();
   direct_matrix_.reset();
   assembled_complex_.reset();
   h_inverse_.reset();
   h_auxiliary_preconditioner_.reset();
   h_matrix_.reset();
   lor_h_matrix_.reset();
   lor_h_form_.reset();
   lor_h_coefficients_.clear();
   lor_fespace_.reset();
   lor_discretization_.reset();

   BuildEssentialTrueDofs();
   operator_.SetEssentialTrueDofs(ess_tdofs_);
   StopWatch assembly_timer;
   assembly_timer.Start();
   operator_.Assemble();
   assembly_timer.Stop();
   assembly_time_ = assembly_timer.RealTime();

   if (linear_solver_type_ == LinearSolverType::MUMPS)
   {
      StopWatch setup_timer;
      setup_timer.Start();
      BuildDirectSolver();
      setup_timer.Stop();
      solver_setup_time_ = setup_timer.RealTime();
      preconditioner_assembly_time_ = 0.0;
   }
   else
   {
      StopWatch preconditioner_timer;
      preconditioner_timer.Start();
      BuildHInverse();
      const PreconditionerTraits traits = BuildBlockPreconditioner();
      preconditioner_timer.Stop();
      preconditioner_assembly_time_ = preconditioner_timer.RealTime();

      active_convention_ = traits.convention;
      active_solver_type_ = ResolveLinearSolver(traits);
      system_operator_ = operator_.FormBlockOperator(active_convention_);
      StopWatch solver_timer;
      solver_timer.Start();
      iterative_solver_ = BuildIterativeSolver(
                             active_solver_type_, *system_operator_,
                             *preconditioner_);
      if (preconditioner_type_ == PreconditionerType::PRESB)
      {
         transpose_operator_.reset(new TransposeOperator(*system_operator_));
         transpose_preconditioner_.reset(new PRESBPreconditioner(
            operator_.GetTOperator(), *h_inverse_, -1));
         transpose_solver_ = BuildIterativeSolver(
                                active_solver_type_, *transpose_operator_,
                                *transpose_preconditioner_);
      }
      solver_timer.Stop();
      solver_setup_time_ = solver_timer.RealTime();
   }

   FrequencyDomainLinearElasticitySolver *self =
      const_cast<FrequencyDomainLinearElasticitySolver *>(this);
   self->height = 2*fespace_.GetTrueVSize();
   self->width = self->height;
   needs_assembly_ = false;
   has_previous_solution_ = false;
   num_iterations_ = 0;
}

/// Negate the imaginary block without forcing host synchronization.
void FrequencyDomainLinearElasticitySolver::NegateImaginaryBlock(
   Vector &vector)
{
   MFEM_VERIFY(vector.Size()%2 == 0,
               "A complex block vector must have even size.");
   const int block_size = vector.Size()/2;
   real_t *values = vector.ReadWrite();
   mfem::forall(block_size, [=] MFEM_HOST_DEVICE(int i)
   {
      values[block_size + i] = -values[block_size + i];
   });
}

/// Insert standard complex prescribed values on constrained dofs.
void FrequencyDomainLinearElasticitySolver::InsertBoundaryValues(
   const Vector &boundary_values, Vector &solution) const
{
   const int block_size = fespace_.GetTrueVSize();
   const int count = ess_tdofs_.Size();
   const int *indices = ess_tdofs_.Read();
   const real_t *boundary = boundary_values.Read();
   real_t *values = solution.ReadWrite();
   mfem::forall(count, [=] MFEM_HOST_DEVICE(int i)
   {
      const int index = indices[i];
      values[index] = boundary[index];
      values[block_size + index] = boundary[block_size + index];
   });
}

/// Apply the configured forward direct or iterative solve.
void FrequencyDomainLinearElasticitySolver::SolveForward(
   const Vector &rhs, Vector &solution, const bool use_initial_guess) const
{
   MFEM_VERIFY(rhs.Size() == Width(), "RHS has incompatible size.");

   // Lazy boundary assembly: only rebuild when boundary conditions changed
   if (boundary_values_stale_)
   {
      BuildBoundaryTrueVector(boundary_true_values_);
      boundary_values_stale_ = false;
   }

   solve_rhs_ = rhs;
   if (active_convention_ == ComplexOperator::BLOCK_SYMMETRIC)
   {
      NegateImaginaryBlock(solve_rhs_);
   }
   operator_.EliminateRHS(boundary_true_values_, solve_rhs_,
                          active_convention_);

   solution.SetSize(Height());
   solution.UseDevice(true);
   if (!use_initial_guess) { solution = 0.0; }
   InsertBoundaryValues(boundary_true_values_, solution);
   if (active_solver_type_ == LinearSolverType::MUMPS)
   {
      MFEM_VERIFY(direct_solver_, "MUMPS solver is not assembled.");
      direct_solver_->iterative_mode = false;
      direct_solver_->Mult(solve_rhs_, solution);
      num_iterations_ = 0;
   }
   else
   {
      MFEM_VERIFY(iterative_solver_, "Iterative solver is not assembled.");
      iterative_solver_->iterative_mode = use_initial_guess;
      iterative_solver_->Mult(solve_rhs_, solution);
      num_iterations_ = iterative_solver_->GetNumIterations();
   }
   InsertBoundaryValues(boundary_true_values_, solution);
}

/// Perform lazy assembly and solve a supplied complex true-dof vector.
void FrequencyDomainLinearElasticitySolver::Mult(
   const Vector &rhs, Vector &solution) const
{
   Assemble();
   MultAssembled(rhs, solution);
}

/// Solve a supplied complex vector without checking lazy assembly.
void FrequencyDomainLinearElasticitySolver::MultAssembled(
   const Vector &rhs, Vector &solution) const
{
   SolveForward(rhs, solution, iterative_mode);
}

/// Perform lazy assembly and solve the homogeneous transpose problem.
void FrequencyDomainLinearElasticitySolver::MultTranspose(
   const Vector &rhs, Vector &solution) const
{
   Assemble();
   MultTransposeAssembled(rhs, solution);
}

/// Solve the standard complex transpose problem without lazy assembly.
void FrequencyDomainLinearElasticitySolver::MultTransposeAssembled(
   const Vector &rhs, Vector &solution) const
{
   MFEM_VERIFY(rhs.Size() == Height(), "Transpose RHS has incompatible size.");
   solve_rhs_ = rhs;
   Vector homogeneous_boundary(Height());
   homogeneous_boundary.UseDevice(true);
   homogeneous_boundary = 0.0;
   operator_.EliminateRHS(homogeneous_boundary, solve_rhs_,
                          active_convention_);
   solution.SetSize(Width());
   solution.UseDevice(true);
   if (!iterative_mode) { solution = 0.0; }

   if (active_solver_type_ == LinearSolverType::MUMPS)
   {
      direct_solver_->iterative_mode = false;
      direct_solver_->Mult(solve_rhs_, solution);
      NegateImaginaryBlock(solution);
      num_iterations_ = 0;
   }
   else if (preconditioner_type_ == PreconditionerType::PRESB)
   {
      MFEM_VERIFY(transpose_solver_, "Transpose solver is not assembled.");
      transpose_solver_->iterative_mode = iterative_mode;
      transpose_solver_->Mult(solve_rhs_, solution);
      num_iterations_ = transpose_solver_->GetNumIterations();
   }
   else
   {
      if (iterative_mode) { NegateImaginaryBlock(solution); }
      iterative_solver_->iterative_mode = iterative_mode;
      iterative_solver_->Mult(solve_rhs_, solution);
      NegateImaginaryBlock(solution);
      num_iterations_ = iterative_solver_->GetNumIterations();
   }
   InsertBoundaryValues(homogeneous_boundary, solution);
}

/// Assemble configured loads, solve, and distribute a complex grid function.
void FrequencyDomainLinearElasticitySolver::Solve(
   ParComplexGridFunction &solution) const
{
   Assemble();
   MFEM_VERIFY(solution.ParFESpace() == &fespace_,
               "Solution must use the solver finite element space.");
   Vector rhs;
   BuildLoadTrueVector(rhs);
   if (!has_previous_solution_)
   {
      previous_solution_.SetSize(Height());
      previous_solution_ = 0.0;
   }
   SolveForward(rhs, previous_solution_, has_previous_solution_);
   has_previous_solution_ = true;
   solution.Distribute(previous_solution_);
}

/// Validate dimensions of an external standard complex operator.
void FrequencyDomainLinearElasticitySolver::SetOperator(const Operator &op)
{
   Assemble();
   MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
               "External operator dimensions do not match the solver.");
}

/// Return the effective outer solver selected during assembly.
FrequencyDomainLinearElasticitySolver::LinearSolverType
FrequencyDomainLinearElasticitySolver::GetActiveLinearSolverType() const
{
   Assemble();
   return active_solver_type_;
}

/// Return the outer iteration count, or zero for MUMPS.
int FrequencyDomainLinearElasticitySolver::GetNumIterations() const
{
   Assemble();
   return num_iterations_;
}

/// Return the current constrained true-dof list after lazy assembly.
const Array<int> &
FrequencyDomainLinearElasticitySolver::GetEssentialTrueDofs() const
{
   Assemble();
   return ess_tdofs_;
}

/// Return the active matrix-free real-block operator after lazy assembly.
const Operator *FrequencyDomainLinearElasticitySolver::GetOperator() const
{
   Assemble();
   return system_operator_.get();
}

/// Return the constrained component operator after lazy assembly.
const FrequencyDomainElasticityOperator &
FrequencyDomainLinearElasticitySolver::GetFrequencyDomainOperator() const
{
   Assemble();
   return operator_;
}

/// Return the active block preconditioner after lazy assembly.
const Solver *FrequencyDomainLinearElasticitySolver::GetPreconditioner() const
{
   Assemble();
   return preconditioner_.get();
}

} // namespace mfem
