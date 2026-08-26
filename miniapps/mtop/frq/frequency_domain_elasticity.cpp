// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.

#include "frequency_domain_elasticity.hpp"

#include <utility>

namespace mfem
{

namespace
{

/// Coefficient that converts E and nu to the first Lame parameter.
class LameLambdaCoefficient : public Coefficient
{
public:
   /// Retain shared ownership of the engineering coefficients.
   LameLambdaCoefficient(std::shared_ptr<Coefficient> young_modulus,
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

/// Coefficient that converts E and nu to the shear modulus.
class LameMuCoefficient : public Coefficient
{
public:
   /// Retain shared ownership of the engineering coefficients.
   LameMuCoefficient(std::shared_ptr<Coefficient> young_modulus,
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

} // namespace

/// Constrained matrix-free action a*K+b*M+c*C.
///
/// The unconstrained true-dof operators are combined before essential rows and
/// columns are imposed. This avoids combining separately inserted unit
/// diagonals. The input copy also makes in-place application safe.
class FrequencyDomainElasticityOperator::WeightedOperator : public Operator
{
public:
   /// Construct a weighted operator with fixed component and constraint views.
   WeightedOperator(const Operator &stiffness, const Operator &mass,
                    const Operator *damping,
                    const Array<int> &ess_tdofs,
                    const Operator::DiagonalPolicy diagonal_policy)
      : Operator(stiffness.Height()),
        stiffness_(&stiffness),
        mass_(&mass),
        damping_(damping),
        ess_tdofs_(&ess_tdofs),
        diagonal_policy_(diagonal_policy),
        original_(stiffness.Height()),
        input_(stiffness.Height()),
        work_(stiffness.Height())
   {
      MFEM_VERIFY(stiffness.Height() == stiffness.Width(),
                  "The stiffness operator must be square.");
      MFEM_VERIFY(mass.Height() == mass.Width() &&
                  mass.Height() == stiffness.Height(),
                  "The mass and stiffness sizes must agree.");
      MFEM_VERIFY(!damping ||
                  (damping->Height() == damping->Width() &&
                   damping->Height() == stiffness.Height()),
                  "The damping, mass, and stiffness sizes must agree.");
      original_.UseDevice(true);
      input_.UseDevice(true);
      work_.UseDevice(true);
   }

   /// Change scalar weights without rebuilding the component operators.
   void SetWeights(const real_t stiffness_weight,
                   const real_t mass_weight,
                   const real_t damping_weight)
   {
      stiffness_weight_ = stiffness_weight;
      mass_weight_ = mass_weight;
      damping_weight_ = damping_weight;
   }

   /// Apply the constrained combination a*K+b*M+c*C.
   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == Width(), "Weighted-operator size mismatch.");
      original_ = x;
      input_ = original_;
      ZeroEssential(input_);

      y.SetSize(Height());
      y = 0.0;
      if (stiffness_weight_ != 0.0)
      {
         stiffness_->Mult(input_, y);
         y *= stiffness_weight_;
      }
      if (mass_weight_ != 0.0)
      {
         mass_->Mult(input_, work_);
         y.Add(mass_weight_, work_);
      }
      if (damping_weight_ != 0.0)
      {
         MFEM_VERIFY(damping_, "A damping operator is required.");
         damping_->Mult(input_, work_);
         y.Add(damping_weight_, work_);
      }
      SetEssentialDiagonal(original_, y);
   }

   /// Apply the transpose using symmetry of K, M, C, and the constraints.
   ///
   /// The elasticity, vector-mass, and coefficient-based damping forms are
   /// self-adjoint. Calling their PA MultTranspose() implementations is both
   /// unnecessary and unsupported by some MFEM integrators, so the transpose
   /// action is exactly the forward action.
   void MultTranspose(const Vector &x, Vector &y) const override
   {
      Mult(x, y);
   }

private:
   /// Set constrained entries of @a vector to zero on the active device.
   void ZeroEssential(Vector &vector) const
   {
      const int count = ess_tdofs_->Size();
      const int *indices = ess_tdofs_->Read();
      real_t *values = vector.ReadWrite();
      mfem::forall(count, [=] MFEM_HOST_DEVICE(int i)
      {
         values[indices[i]] = 0.0;
      });
   }

   /// Insert either zero or the identity action on constrained entries.
   void SetEssentialDiagonal(const Vector &x, Vector &y) const
   {
      const real_t diagonal =
         diagonal_policy_ == Operator::DiagonalPolicy::DIAG_ONE ? 1.0 : 0.0;
      const int count = ess_tdofs_->Size();
      const int *indices = ess_tdofs_->Read();
      const real_t *input = x.Read();
      real_t *output = y.ReadWrite();
      mfem::forall(count, [=] MFEM_HOST_DEVICE(int i)
      {
         const int index = indices[i];
         output[index] = diagonal*input[index];
      });
   }

   const Operator *stiffness_;
   const Operator *mass_;
   const Operator *damping_;
   const Array<int> *ess_tdofs_;
   Operator::DiagonalPolicy diagonal_policy_;
   real_t stiffness_weight_ = 0.0;
   real_t mass_weight_ = 0.0;
   real_t damping_weight_ = 0.0;
   mutable Vector original_;
   mutable Vector input_;
   mutable Vector work_;
};

/// Construct an unconfigured real-block elasticity operator.
FrequencyDomainElasticityOperator::FrequencyDomainElasticityOperator(
   ParFiniteElementSpace &fespace)
   : Operator(2*fespace.GetTrueVSize()),
     fespace_(fespace)
{
   MFEM_VERIFY(fespace.GetParMesh() != nullptr,
               "A parallel finite element space is required.");
   MFEM_VERIFY(fespace.GetVDim() == fespace.GetParMesh()->SpaceDimension(),
               "Elasticity requires vector dimension equal to space dimension.");
}

/// Release PA forms, operator handles, and optional assembled matrices.
FrequencyDomainElasticityOperator::~FrequencyDomainElasticityOperator() =
   default;

/// Wrap a coefficient reference with optional ownership transfer.
std::shared_ptr<Coefficient>
FrequencyDomainElasticityOperator::ShareCoefficient(
   Coefficient &coefficient, const bool transfer_ownership)
{
   if (transfer_ownership)
   {
      return std::shared_ptr<Coefficient>(&coefficient);
   }
   return std::shared_ptr<Coefficient>(&coefficient,
                                       [](Coefficient *) { });
}

/// Replace the material with Lame and density coefficient references.
void FrequencyDomainElasticityOperator::SetLameMaterial(
   Coefficient &lambda, Coefficient &mu, Coefficient &density,
   const bool transfer_ownership)
{
   SetLameMaterial(ShareCoefficient(lambda, transfer_ownership),
                   ShareCoefficient(mu, transfer_ownership),
                   ShareCoefficient(density, transfer_ownership));
}

/// Replace the material with shared Lame and density coefficients.
void FrequencyDomainElasticityOperator::SetLameMaterial(
   std::shared_ptr<Coefficient> lambda,
   std::shared_ptr<Coefficient> mu,
   std::shared_ptr<Coefficient> density)
{
   MFEM_VERIFY(lambda && mu && density,
               "All isotropic material coefficients are required.");
   lambda_ = std::move(lambda);
   mu_ = std::move(mu);
   density_ = std::move(density);
   InvalidateMaterialData();
}

/// Wrap distributed Lame and density grid functions as coefficients.
void FrequencyDomainElasticityOperator::SetLameMaterialFields(
   ParGridFunction &lambda, ParGridFunction &mu, ParGridFunction &density)
{
   SetLameMaterial(std::make_shared<GridFunctionCoefficient>(&lambda),
                   std::make_shared<GridFunctionCoefficient>(&mu),
                   std::make_shared<GridFunctionCoefficient>(&density));
}

/// Replace the material with engineering coefficient references.
void FrequencyDomainElasticityOperator::SetEngineeringMaterial(
   Coefficient &young_modulus, Coefficient &poisson_ratio,
   Coefficient &density, const bool transfer_ownership)
{
   SetEngineeringMaterial(ShareCoefficient(young_modulus, transfer_ownership),
                          ShareCoefficient(poisson_ratio, transfer_ownership),
                          ShareCoefficient(density, transfer_ownership));
}

/// Convert shared E and nu coefficients into shared Lame coefficients.
void FrequencyDomainElasticityOperator::SetEngineeringMaterial(
   std::shared_ptr<Coefficient> young_modulus,
   std::shared_ptr<Coefficient> poisson_ratio,
   std::shared_ptr<Coefficient> density)
{
   MFEM_VERIFY(young_modulus && poisson_ratio && density,
               "All isotropic material coefficients are required.");
   std::shared_ptr<Coefficient> lambda =
      std::make_shared<LameLambdaCoefficient>(young_modulus, poisson_ratio);
   std::shared_ptr<Coefficient> mu =
      std::make_shared<LameMuCoefficient>(std::move(young_modulus),
                                          std::move(poisson_ratio));
   SetLameMaterial(std::move(lambda), std::move(mu), std::move(density));
}

/// Wrap distributed E, nu, and density grid functions as coefficients.
void FrequencyDomainElasticityOperator::SetEngineeringMaterialFields(
   ParGridFunction &young_modulus, ParGridFunction &poisson_ratio,
   ParGridFunction &density)
{
   SetEngineeringMaterial(
      std::make_shared<GridFunctionCoefficient>(&young_modulus),
      std::make_shared<GridFunctionCoefficient>(&poisson_ratio),
      std::make_shared<GridFunctionCoefficient>(&density));
}

/// Invalidate PA and assembled data after an in-place coefficient update.
void FrequencyDomainElasticityOperator::MaterialChanged()
{
   InvalidateMaterialData();
}

/// Clear all state derived from material or damping coefficients.
void FrequencyDomainElasticityOperator::InvalidateMaterialData()
{
   pa_dirty_ = true;
   hypre_dirty_ = true;
   W1_.reset();
   W2_.reset();
   T_.reset();
   W_.reset();
   H_.reset();
   stiffness_operator_.Clear();
   mass_operator_.Clear();
   damping_operator_.Clear();
   stiffness_form_.reset();
   mass_form_.reset();
   damping_form_.reset();
   stiffness_matrix_.reset();
   mass_matrix_.reset();
   damping_matrix_.reset();
}

/// Change omega without rebuilding frequency-independent K and M data.
void FrequencyDomainElasticityOperator::SetFrequency(const real_t omega)
{
   MFEM_VERIFY(omega >= 0.0, "Angular frequency must be nonnegative.");
   omega_ = omega;
   UpdateDerivedOperators();
}

/// Select scalar Rayleigh damping and discard independent damping fields.
void FrequencyDomainElasticityOperator::SetRayleighDamping(
   const real_t alpha, const real_t beta)
{
   MFEM_VERIFY(alpha >= 0.0 && beta >= 0.0,
               "Rayleigh damping coefficients must be nonnegative.");
   const bool switching_models = static_cast<bool>(mass_damping_);
   rayleigh_alpha_ = alpha;
   rayleigh_beta_ = beta;
   mass_damping_.reset();
   damping_lambda_.reset();
   damping_mu_.reset();
   if (switching_models)
   {
      InvalidateMaterialData();
   }
   else
   {
      UpdateDerivedOperators();
   }
}

/// Select independent damping coefficient references with optional ownership.
void FrequencyDomainElasticityOperator::SetDampingCoefficients(
   Coefficient &mass_damping, Coefficient &damping_lambda,
   Coefficient &damping_mu, const bool transfer_ownership)
{
   SetDampingCoefficients(
      ShareCoefficient(mass_damping, transfer_ownership),
      ShareCoefficient(damping_lambda, transfer_ownership),
      ShareCoefficient(damping_mu, transfer_ownership));
}

/// Select independent shared damping coefficients and disable Rayleigh mode.
void FrequencyDomainElasticityOperator::SetDampingCoefficients(
   std::shared_ptr<Coefficient> mass_damping,
   std::shared_ptr<Coefficient> damping_lambda,
   std::shared_ptr<Coefficient> damping_mu)
{
   MFEM_VERIFY(mass_damping && damping_lambda && damping_mu,
               "All independent damping coefficients are required.");
   mass_damping_ = std::move(mass_damping);
   damping_lambda_ = std::move(damping_lambda);
   damping_mu_ = std::move(damping_mu);
   rayleigh_alpha_ = 0.0;
   rayleigh_beta_ = 0.0;
   InvalidateMaterialData();
}

/// Wrap distributed mass and elasticity damping fields as coefficients.
void FrequencyDomainElasticityOperator::SetDampingCoefficientFields(
   ParGridFunction &mass_damping, ParGridFunction &damping_lambda,
   ParGridFunction &damping_mu)
{
   SetDampingCoefficients(
      std::make_shared<GridFunctionCoefficient>(&mass_damping),
      std::make_shared<GridFunctionCoefficient>(&damping_lambda),
      std::make_shared<GridFunctionCoefficient>(&damping_mu));
}

/// Replace the homogeneous essential true-dof set.
void FrequencyDomainElasticityOperator::SetEssentialTrueDofs(
   const Array<int> &ess_tdofs)
{
   Array<int> sorted_tdofs(ess_tdofs);
   sorted_tdofs.Sort();
   sorted_tdofs.Unique();
   bool changed = sorted_tdofs.Size() != ess_tdofs_.Size();
   for (int i = 0; i < sorted_tdofs.Size() && !changed; ++i)
   {
      changed = sorted_tdofs[i] != ess_tdofs_[i];
   }
   if (!changed) { return; }
   ess_tdofs_.Swap(sorted_tdofs);
   InvalidateMaterialData();
}

/// Assemble unconstrained true-dof K, M, and optional C with partial assembly.
void FrequencyDomainElasticityOperator::Assemble() const
{
   if (!pa_dirty_) { return; }
   MFEM_VERIFY(lambda_ && mu_ && density_,
               "Set an isotropic material before assembly.");

   stiffness_form_.reset(new ParBilinearForm(&fespace_));
   stiffness_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   stiffness_form_->AddDomainIntegrator(
      new ElasticityIntegrator(*lambda_, *mu_));
   stiffness_form_->Assemble();

   mass_form_.reset(new ParBilinearForm(&fespace_));
   mass_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass_form_->AddDomainIntegrator(new VectorMassIntegrator(*density_));
   mass_form_->Assemble();

   if (mass_damping_)
   {
      MFEM_VERIFY(damping_lambda_ && damping_mu_,
                  "All independent damping coefficients are required.");
      damping_form_.reset(new ParBilinearForm(&fespace_));
      damping_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      damping_form_->AddDomainIntegrator(
         new ElasticityIntegrator(*damping_lambda_, *damping_mu_));
      damping_form_->AddDomainIntegrator(
         new VectorMassIntegrator(*mass_damping_));
      damping_form_->Assemble();
   }

   Array<int> no_essential_dofs;
   stiffness_operator_.Clear();
   mass_operator_.Clear();
   damping_operator_.Clear();
   stiffness_operator_.SetType(Operator::ANY_TYPE);
   mass_operator_.SetType(Operator::ANY_TYPE);
   stiffness_form_->FormSystemMatrix(no_essential_dofs,
                                     stiffness_operator_);
   mass_form_->FormSystemMatrix(no_essential_dofs, mass_operator_);
   if (damping_form_)
   {
      damping_operator_.SetType(Operator::ANY_TYPE);
      damping_form_->FormSystemMatrix(no_essential_dofs,
                                      damping_operator_);
   }

   const Operator *damping = damping_operator_.Ptr();

   W1_.reset(new WeightedOperator(*stiffness_operator_.Ptr(),
                                  *mass_operator_.Ptr(), damping, ess_tdofs_,
                                  Operator::DiagonalPolicy::DIAG_ONE));
   W2_.reset(new WeightedOperator(*stiffness_operator_.Ptr(),
                                  *mass_operator_.Ptr(), damping, ess_tdofs_,
                                  Operator::DiagonalPolicy::DIAG_ONE));
   T_.reset(new WeightedOperator(*stiffness_operator_.Ptr(),
                                 *mass_operator_.Ptr(), damping, ess_tdofs_,
                                 Operator::DiagonalPolicy::DIAG_ZERO));
   W_.reset(new WeightedOperator(*stiffness_operator_.Ptr(),
                                 *mass_operator_.Ptr(), damping, ess_tdofs_,
                                 Operator::DiagonalPolicy::DIAG_ONE));
   H_.reset(new WeightedOperator(*stiffness_operator_.Ptr(),
                                 *mass_operator_.Ptr(), damping, ess_tdofs_,
                                 Operator::DiagonalPolicy::DIAG_ONE));
   pa_dirty_ = false;
   UpdateDerivedOperators();
}

/// Update the frequency and damping weights defining W1, W2, T, W, and H.
void FrequencyDomainElasticityOperator::UpdateDerivedOperators() const
{
   if (!W1_) { return; }
   const real_t omega_squared = omega_*omega_;
   W1_->SetWeights(1.0, 0.0, 0.0);
   W2_->SetWeights(0.0, omega_squared, 0.0);
   W_->SetWeights(1.0, -omega_squared, 0.0);
   if (mass_damping_)
   {
      T_->SetWeights(0.0, 0.0, omega_);
      H_->SetWeights(1.0, -omega_squared, omega_);
   }
   else
   {
      T_->SetWeights(omega_*rayleigh_beta_, omega_*rayleigh_alpha_, 0.0);
      H_->SetWeights(1.0 + omega_*rayleigh_beta_,
                     omega_*rayleigh_alpha_ - omega_squared, 0.0);
   }
}

/// Assemble unconstrained Hypre matrices for K, M, and optional C.
void FrequencyDomainElasticityOperator::AssembleHypreMatrices() const
{
   if (!hypre_dirty_) { return; }
   MFEM_VERIFY(lambda_ && mu_ && density_,
               "Set an isotropic material before assembly.");

   ParBilinearForm stiffness(&fespace_);
   stiffness.AddDomainIntegrator(new ElasticityIntegrator(*lambda_, *mu_));
   stiffness.Assemble();
   stiffness.Finalize();
   stiffness_matrix_.reset(stiffness.ParallelAssemble());

   ParBilinearForm mass(&fespace_);
   mass.AddDomainIntegrator(new VectorMassIntegrator(*density_));
   mass.Assemble();
   mass.Finalize();
   mass_matrix_.reset(mass.ParallelAssemble());

   if (mass_damping_)
   {
      MFEM_VERIFY(damping_lambda_ && damping_mu_,
                  "All independent damping coefficients are required.");
      ParBilinearForm damping(&fespace_);
      damping.AddDomainIntegrator(
         new ElasticityIntegrator(*damping_lambda_, *damping_mu_));
      damping.AddDomainIntegrator(
         new VectorMassIntegrator(*mass_damping_));
      damping.Assemble();
      damping.Finalize();
      damping_matrix_.reset(damping.ParallelAssemble());
   }
   hypre_dirty_ = false;
}

/// Form and constrain aK+bM+cC after completing the linear combination.
std::unique_ptr<HypreParMatrix>
FrequencyDomainElasticityOperator::FormMatrix(
   const real_t stiffness_weight, const real_t mass_weight,
   const real_t damping_weight,
   const Operator::DiagonalPolicy diagonal_policy) const
{
   AssembleHypreMatrices();
   std::unique_ptr<HypreParMatrix> matrix(
      Add(stiffness_weight, *stiffness_matrix_,
          mass_weight, *mass_matrix_));
   if (damping_weight != 0.0)
   {
      MFEM_VERIFY(damping_matrix_, "A damping matrix is required.");
      matrix.reset(Add(1.0, *matrix, damping_weight, *damping_matrix_));
   }
   matrix->EliminateBC(ess_tdofs_, diagonal_policy);
   return matrix;
}

/// Return the current constrained PA stiffness operator W1.
const Operator &FrequencyDomainElasticityOperator::GetW1Operator() const
{
   Assemble();
   return *W1_;
}

/// Return the current constrained PA inertia operator W2.
const Operator &FrequencyDomainElasticityOperator::GetW2Operator() const
{
   Assemble();
   return *W2_;
}

/// Return the current constrained PA damping operator T.
const Operator &FrequencyDomainElasticityOperator::GetTOperator() const
{
   Assemble();
   return *T_;
}

/// Return the current constrained PA real operator W.
const Operator &FrequencyDomainElasticityOperator::GetWOperator() const
{
   Assemble();
   return *W_;
}

/// Return the current constrained PA shifted operator H=W+T.
const Operator &FrequencyDomainElasticityOperator::GetHOperator() const
{
   Assemble();
   return *H_;
}

/// Form W1=K with unit diagonal treatment on essential dofs.
std::unique_ptr<HypreParMatrix>
FrequencyDomainElasticityOperator::FormW1Matrix() const
{
   return FormMatrix(1.0, 0.0, 0.0,
                     Operator::DiagonalPolicy::DIAG_ONE);
}

/// Form W2=omega^2 M with unit diagonal treatment on essential dofs.
std::unique_ptr<HypreParMatrix>
FrequencyDomainElasticityOperator::FormW2Matrix() const
{
   return FormMatrix(0.0, omega_*omega_, 0.0,
                     Operator::DiagonalPolicy::DIAG_ONE);
}

/// Form T=omega*C with zero essential rows and columns.
std::unique_ptr<HypreParMatrix>
FrequencyDomainElasticityOperator::FormTMatrix() const
{
   if (mass_damping_)
   {
      return FormMatrix(0.0, 0.0, omega_,
                        Operator::DiagonalPolicy::DIAG_ZERO);
   }
   return FormMatrix(omega_*rayleigh_beta_, omega_*rayleigh_alpha_, 0.0,
                     Operator::DiagonalPolicy::DIAG_ZERO);
}

/// Form W=K-omega^2 M with unit essential diagonal treatment.
std::unique_ptr<HypreParMatrix>
FrequencyDomainElasticityOperator::FormWMatrix() const
{
   return FormMatrix(1.0, -omega_*omega_, 0.0,
                     Operator::DiagonalPolicy::DIAG_ONE);
}

/// Form H=W+T with unit essential diagonal treatment.
std::unique_ptr<HypreParMatrix>
FrequencyDomainElasticityOperator::FormHMatrix() const
{
   if (mass_damping_)
   {
      return FormMatrix(1.0, -omega_*omega_, omega_,
                        Operator::DiagonalPolicy::DIAG_ONE);
   }
   return FormMatrix(1.0 + omega_*rayleigh_beta_,
                     omega_*rayleigh_alpha_ - omega_*omega_, 0.0,
                     Operator::DiagonalPolicy::DIAG_ONE);
}

/// Form a non-owning matrix-free ComplexOperator view of W+iT.
std::unique_ptr<ComplexOperator>
FrequencyDomainElasticityOperator::FormBlockOperator(
   const ComplexOperator::Convention convention) const
{
   Assemble();
   return std::unique_ptr<ComplexOperator>(
             new ComplexOperator(W_.get(), T_.get(), false, false,
                                 convention));
}

/// Form an owning assembled ComplexHypreParMatrix representation of W+iT.
std::unique_ptr<ComplexHypreParMatrix>
FrequencyDomainElasticityOperator::FormAssembledComplexOperator(
   const ComplexOperator::Convention convention) const
{
   std::unique_ptr<HypreParMatrix> W = FormWMatrix();
   std::unique_ptr<HypreParMatrix> T = FormTMatrix();
   return std::unique_ptr<ComplexHypreParMatrix>(
             new ComplexHypreParMatrix(W.release(), T.release(), true, true,
                                       convention));
}

/// Apply [W,-T;T,W] to a contiguous [real;imaginary] vector.
void FrequencyDomainElasticityOperator::Mult(const Vector &x,
                                             Vector &y) const
{
   MFEM_VERIFY(x.Size() == Width(), "Frequency-domain input size mismatch.");
   Assemble();
   ComplexOperator complex_operator(W_.get(), T_.get(), false, false,
                                    ComplexOperator::HERMITIAN);
   complex_operator.Mult(x, y);
}

/// Apply [W^T,T^T;-T^T,W^T] to a real block vector.
void FrequencyDomainElasticityOperator::MultTranspose(
   const Vector &x, Vector &y) const
{
   MFEM_VERIFY(x.Size() == Height(), "Frequency-domain input size mismatch.");
   Assemble();
   ComplexOperator complex_operator(W_.get(), T_.get(), false, false,
                                    ComplexOperator::HERMITIAN);
   complex_operator.MultTranspose(x, y);
}

/// Apply the unconstrained real block form for the selected convention.
void FrequencyDomainElasticityOperator::MultUnconstrained(
   const Vector &x, Vector &y,
   const ComplexOperator::Convention convention) const
{
   MFEM_VERIFY(x.Size() == Width(), "Frequency-domain input size mismatch.");
   Assemble();
   Array<int> no_essential_dofs;
   const Operator *damping = damping_operator_.Ptr();
   WeightedOperator W_unconstrained(
      *stiffness_operator_.Ptr(), *mass_operator_.Ptr(), damping,
      no_essential_dofs, Operator::DiagonalPolicy::DIAG_ZERO);
   WeightedOperator T_unconstrained(
      *stiffness_operator_.Ptr(), *mass_operator_.Ptr(), damping,
      no_essential_dofs, Operator::DiagonalPolicy::DIAG_ZERO);

   const real_t omega_squared = omega_*omega_;
   W_unconstrained.SetWeights(1.0, -omega_squared, 0.0);
   if (mass_damping_)
   {
      T_unconstrained.SetWeights(0.0, 0.0, omega_);
   }
   else
   {
      T_unconstrained.SetWeights(
         omega_*rayleigh_beta_, omega_*rayleigh_alpha_, 0.0);
   }

   ComplexOperator complex_operator(&W_unconstrained, &T_unconstrained,
                                    false, false, convention);
   complex_operator.Mult(x, y);
}

/// Lift prescribed complex values and set constrained right-hand-side entries.
void FrequencyDomainElasticityOperator::EliminateRHS(
   const Vector &boundary_values, Vector &rhs,
   const ComplexOperator::Convention convention) const
{
   MFEM_VERIFY(boundary_values.Size() == Width() && rhs.Size() == Height(),
               "Frequency-domain boundary or right-hand-side size mismatch.");
   MFEM_VERIFY(convention == ComplexOperator::HERMITIAN ||
               convention == ComplexOperator::BLOCK_SYMMETRIC,
               "Unsupported complex-operator convention.");

   Vector prescribed(boundary_values);
   prescribed.UseDevice(true);
   Vector action(Height());
   action.UseDevice(true);
   MultUnconstrained(prescribed, action, convention);
   rhs -= action;

   const int block_size = Width()/2;
   const int count = ess_tdofs_.Size();
   const int *indices = ess_tdofs_.Read();
   const real_t *values = prescribed.Read();
   real_t *right_hand_side = rhs.ReadWrite();
   const real_t imaginary_diagonal =
      convention == ComplexOperator::BLOCK_SYMMETRIC ? -1.0 : 1.0;
   mfem::forall(count, [=] MFEM_HOST_DEVICE(int i)
   {
      const int index = indices[i];
      right_hand_side[index] = values[index];
      right_hand_side[block_size + index] =
         imaginary_diagonal*values[block_size + index];
   });
}

/// Return the configured first Lame material coefficient.
const Coefficient &
FrequencyDomainElasticityOperator::GetLambdaCoefficient() const
{
   MFEM_VERIFY(lambda_, "The first Lame coefficient is not configured.");
   return *lambda_;
}

/// Return the configured shear-modulus material coefficient.
const Coefficient &FrequencyDomainElasticityOperator::GetMuCoefficient() const
{
   MFEM_VERIFY(mu_, "The shear-modulus coefficient is not configured.");
   return *mu_;
}

/// Return the configured mass-density coefficient.
const Coefficient &
FrequencyDomainElasticityOperator::GetDensityCoefficient() const
{
   MFEM_VERIFY(density_, "The mass-density coefficient is not configured.");
   return *density_;
}

/// Return the independent mass-form damping coefficient.
const Coefficient &
FrequencyDomainElasticityOperator::GetMassDampingCoefficient() const
{
   MFEM_VERIFY(mass_damping_,
               "Independent-coefficient damping is not configured.");
   return *mass_damping_;
}

/// Return the independent first Lame-like damping coefficient.
const Coefficient &
FrequencyDomainElasticityOperator::GetDampingLambdaCoefficient() const
{
   MFEM_VERIFY(damping_lambda_,
               "Independent-coefficient damping is not configured.");
   return *damping_lambda_;
}

/// Return the independent shear-like damping coefficient.
const Coefficient &
FrequencyDomainElasticityOperator::GetDampingMuCoefficient() const
{
   MFEM_VERIFY(damping_mu_,
               "Independent-coefficient damping is not configured.");
   return *damping_mu_;
}

} // namespace mfem
