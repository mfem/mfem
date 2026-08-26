// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.

#ifndef MFEM_MTOP_FREQUENCY_DOMAIN_ELASTICITY_HPP
#define MFEM_MTOP_FREQUENCY_DOMAIN_ELASTICITY_HPP

#include "mfem.hpp"

#include <memory>

namespace mfem
{

/// Matrix-free frequency-domain operator for isotropic linear elasticity.
///
/// The operator represents
///
///     (K - omega^2 M + i omega C) u = f,
///
/// using the real block form [W,-T;T,W], where W=K-omega^2 M and
/// T=omega C. Damping can use either the Rayleigh model C=alpha M+beta K or
/// independent isotropic coefficients in a mass form and an elasticity form.
/// All component operators use MFEM partial assembly; optional HypreParMatrix
/// snapshots are available for verification and direct-solver experiments.
///
/// Material coefficients and the finite element space are not owned unless
/// supplied through shared pointers. Configure mfem::Device before creating
/// this object. One instance is not safe for concurrent Mult() calls because
/// it reuses mutable work vectors.
class FrequencyDomainElasticityOperator : public Operator
{
public:
   /// Construct an unconfigured operator on an H1 vector finite element space.
   explicit FrequencyDomainElasticityOperator(
      ParFiniteElementSpace &fespace);

   /// Destroy all partial-assembly and assembled auxiliary data.
   ~FrequencyDomainElasticityOperator() override;

   /// Set isotropic material through Lame lambda, shear modulus mu, and mass
   /// density. References are non-owning unless @a transfer_ownership is true.
   void SetLameMaterial(Coefficient &lambda, Coefficient &mu,
                        Coefficient &density,
                        bool transfer_ownership = false);

   /// Set isotropic material through shared Lame and density coefficients.
   void SetLameMaterial(std::shared_ptr<Coefficient> lambda,
                        std::shared_ptr<Coefficient> mu,
                        std::shared_ptr<Coefficient> density);

   /// Use three distributed grid functions as Lame and density fields.
   ///
   /// The grid functions remain owned by the caller and must outlive this
   /// operator.
   void SetLameMaterialFields(ParGridFunction &lambda,
                              ParGridFunction &mu,
                              ParGridFunction &density);

   /// Set isotropic material through Young's modulus, Poisson's ratio, and
   /// mass density. References are non-owning unless ownership is transferred.
   void SetEngineeringMaterial(Coefficient &young_modulus,
                               Coefficient &poisson_ratio,
                               Coefficient &density,
                               bool transfer_ownership = false);

   /// Set isotropic material through shared engineering coefficients.
   ///
   /// Lame coefficients are evaluated from E and nu during assembly using
   /// lambda=E*nu/((1+nu)(1-2nu)) and mu=E/(2(1+nu)).
   void SetEngineeringMaterial(
      std::shared_ptr<Coefficient> young_modulus,
      std::shared_ptr<Coefficient> poisson_ratio,
      std::shared_ptr<Coefficient> density);

   /// Use distributed grid functions for E, nu, and mass density.
   ///
   /// The grid functions remain owned by the caller and must outlive this
   /// operator.
   void SetEngineeringMaterialFields(ParGridFunction &young_modulus,
                                     ParGridFunction &poisson_ratio,
                                     ParGridFunction &density);

   /// Mark PA data and assembled matrices stale after material or damping
   /// coefficient values have changed in place.
   void MaterialChanged();

   /// Set the angular frequency. This only updates scalar operator weights.
   void SetFrequency(real_t omega);

   /// Set nonnegative Rayleigh coefficients in C=alpha M+beta K.
   ///
   /// This selects Rayleigh damping and discards any independent damping
   /// coefficients previously supplied to SetDampingCoefficients().
   void SetRayleighDamping(real_t alpha, real_t beta);

   /// Set independent isotropic damping coefficients.
   ///
   /// The damping bilinear form is
   ///
   ///     c(u,v) = integral(lambda_C div(u) div(v)
   ///                + 2 mu_C epsilon(u):epsilon(v)
   ///                + c_M u.v) dx.
   ///
   /// Here @a mass_damping is c_M, while @a damping_lambda and
   /// @a damping_mu are the two stiffness-like damping coefficients.
   /// References are non-owning unless @a transfer_ownership is true. This
   /// setup replaces Rayleigh damping.
   void SetDampingCoefficients(Coefficient &mass_damping,
                               Coefficient &damping_lambda,
                               Coefficient &damping_mu,
                               bool transfer_ownership = false);

   /// Set independent isotropic damping through shared coefficients.
   ///
   /// The arguments define c_M, lambda_C, and mu_C, respectively, in the
   /// damping bilinear form documented by the reference overload. This setup
   /// replaces Rayleigh damping.
   void SetDampingCoefficients(
      std::shared_ptr<Coefficient> mass_damping,
      std::shared_ptr<Coefficient> damping_lambda,
      std::shared_ptr<Coefficient> damping_mu);

   /// Use distributed grid functions for c_M, lambda_C, and mu_C damping.
   ///
   /// The grid functions remain owned by the caller and must outlive this
   /// operator. This setup replaces Rayleigh damping.
   void SetDampingCoefficientFields(ParGridFunction &mass_damping,
                                    ParGridFunction &damping_lambda,
                                    ParGridFunction &damping_mu);

   /// Set essential true dofs for homogeneous displacement conditions.
   ///
   /// Nonhomogeneous data must be lifted into the right-hand side externally.
   void SetEssentialTrueDofs(const Array<int> &ess_tdofs);

   /// Assemble or refresh the matrix-free K, M, and optional C operators.
   void Assemble() const;

   /// Assemble unconstrained true-dof Hypre matrices for K, M, and optional C.
   ///
   /// Derived matrices apply essential conditions after their complete linear
   /// combination has been formed.
   void AssembleHypreMatrices() const;

   /// Apply the nonsymmetric real form [W,-T;T,W].
   void Mult(const Vector &x, Vector &y) const override;

   /// Apply its transpose [W,T;-T,W], using symmetry of W and T.
   void MultTranspose(const Vector &x, Vector &y) const override;

   /// Eliminate prescribed complex displacement values from a right-hand side.
   ///
   /// Both vectors use contiguous [real;imaginary] blocks. The right-hand side
   /// must already use @a convention: its imaginary block is the physical
   /// imaginary load for HERMITIAN and its negative for BLOCK_SYMMETRIC.
   /// Essential entries are overwritten with the corresponding constrained
   /// diagonal action. This method is safe when the input vectors use device
   /// memory.
   void EliminateRHS(
      const Vector &boundary_values, Vector &rhs,
      ComplexOperator::Convention convention =
         ComplexOperator::HERMITIAN) const;

   /// Return the constrained matrix-free W1=K split operator.
   const Operator &GetW1Operator() const;

   /// Return the constrained matrix-free W2=omega^2 M split operator.
   const Operator &GetW2Operator() const;

   /// Return the constrained matrix-free coupling T=omega C.
   ///
   /// Its essential rows and columns use zero diagonal treatment.
   const Operator &GetTOperator() const;

   /// Return the constrained matrix-free W=K-omega^2 M operator.
   const Operator &GetWOperator() const;

   /// Return the constrained matrix-free H=W+T operator.
   const Operator &GetHOperator() const;

   /// Form an independently owned constrained Hypre matrix for W1=K.
   std::unique_ptr<HypreParMatrix> FormW1Matrix() const;

   /// Form an independently owned constrained Hypre matrix for W2=omega^2 M.
   std::unique_ptr<HypreParMatrix> FormW2Matrix() const;

   /// Form an independently owned Hypre matrix for T=omega C with zero
   /// essential rows and columns.
   std::unique_ptr<HypreParMatrix> FormTMatrix() const;

   /// Form an independently owned constrained Hypre matrix for
   /// W=K-omega^2 M.
   std::unique_ptr<HypreParMatrix> FormWMatrix() const;

   /// Form an independently owned constrained Hypre matrix for H=W+T.
   std::unique_ptr<HypreParMatrix> FormHMatrix() const;

   /// Form a non-owning matrix-free real-block view of W+iT.
   ///
   /// With ComplexOperator::HERMITIAN (the default), its real representation
   /// is [W,-T;T,W]. With ComplexOperator::BLOCK_SYMMETRIC it is
   /// [W,-T;-T,-W], i.e. the standard form left-multiplied by
   /// diag(I,-I). The returned view must not outlive this object and becomes
   /// invalid after changing material, damping, or essential true dofs.
   /// @see mfem::ComplexOperator
   std::unique_ptr<ComplexOperator> FormBlockOperator(
      ComplexOperator::Convention convention =
         ComplexOperator::HERMITIAN) const;

   /// Assemble W and T and return an owning complex Hypre operator.
   ///
   /// The convention selects [W,-T;T,W] (HERMITIAN) or [W,-T;-T,-W]
   /// (BLOCK_SYMMETRIC). The latter system must use the right-hand side
   /// [b_r;-b_i]. The returned object owns both Hypre blocks and is independent
   /// of later material, frequency, and damping changes. This is the preferred
   /// assembled representation: keep the two Hypre blocks as primary storage
   /// and call GetSystemMatrix() only when a solver requires one real
   /// monolithic HypreParMatrix, for example MUMPS.
   /// @see mfem::ComplexHypreParMatrix
   std::unique_ptr<ComplexHypreParMatrix> FormAssembledComplexOperator(
      ComplexOperator::Convention convention =
         ComplexOperator::HERMITIAN) const;

   /// Return the current angular frequency.
   real_t GetFrequency() const { return omega_; }

   /// Return the configured first Lame material coefficient.
   const Coefficient &GetLambdaCoefficient() const;

   /// Return the configured shear-modulus material coefficient.
   const Coefficient &GetMuCoefficient() const;

   /// Return the configured mass-density coefficient.
   const Coefficient &GetDensityCoefficient() const;

   /// Return true when the independent-coefficient damping model is active.
   bool HasIndependentDamping() const
   { return static_cast<bool>(mass_damping_); }

   /// Return the independent mass-form damping coefficient.
   const Coefficient &GetMassDampingCoefficient() const;

   /// Return the independent first Lame-like damping coefficient.
   const Coefficient &GetDampingLambdaCoefficient() const;

   /// Return the independent shear-like damping coefficient.
   const Coefficient &GetDampingMuCoefficient() const;

   /// Return the mass-proportional Rayleigh coefficient, or zero when the
   /// independent-coefficient damping model is active.
   real_t GetRayleighAlpha() const { return rayleigh_alpha_; }

   /// Return the stiffness-proportional Rayleigh coefficient, or zero when the
   /// independent-coefficient damping model is active.
   real_t GetRayleighBeta() const { return rayleigh_beta_; }

   /// Return the underlying finite element space.
   ParFiniteElementSpace &GetFESpace() { return fespace_; }

   /// Return the underlying finite element space.
   const ParFiniteElementSpace &GetFESpace() const { return fespace_; }

   /// Return the homogeneous essential true-dof list.
   const Array<int> &GetEssentialTrueDofs() const { return ess_tdofs_; }

private:
   class WeightedOperator;

   /// Wrap a coefficient reference in an owning or non-owning shared pointer.
   static std::shared_ptr<Coefficient> ShareCoefficient(
      Coefficient &coefficient, bool transfer_ownership);

   /// Invalidate all data that depends on material or damping values.
   void InvalidateMaterialData();

   /// Update frequency and damping weights in all derived PA operators.
   void UpdateDerivedOperators() const;

   /// Apply the unconstrained complex operator using @a convention.
   void MultUnconstrained(const Vector &x, Vector &y,
                          ComplexOperator::Convention convention) const;

   /// Form a constrained combination aK+bM+cC from assembled base matrices.
   std::unique_ptr<HypreParMatrix>
   FormMatrix(real_t stiffness_weight, real_t mass_weight,
              real_t damping_weight,
              Operator::DiagonalPolicy diagonal_policy) const;

   ParFiniteElementSpace &fespace_;
   std::shared_ptr<Coefficient> lambda_;
   std::shared_ptr<Coefficient> mu_;
   std::shared_ptr<Coefficient> density_;
   std::shared_ptr<Coefficient> mass_damping_;
   std::shared_ptr<Coefficient> damping_lambda_;
   std::shared_ptr<Coefficient> damping_mu_;

   real_t omega_ = 0.0;
   real_t rayleigh_alpha_ = 0.0;
   real_t rayleigh_beta_ = 0.0;
   Array<int> ess_tdofs_;
   mutable bool pa_dirty_ = true;
   mutable bool hypre_dirty_ = true;
   mutable std::unique_ptr<ParBilinearForm> stiffness_form_;
   mutable std::unique_ptr<ParBilinearForm> mass_form_;
   mutable std::unique_ptr<ParBilinearForm> damping_form_;
   mutable OperatorHandle stiffness_operator_;
   mutable OperatorHandle mass_operator_;
   mutable OperatorHandle damping_operator_;

   mutable std::unique_ptr<WeightedOperator> W1_;
   mutable std::unique_ptr<WeightedOperator> W2_;
   mutable std::unique_ptr<WeightedOperator> T_;
   mutable std::unique_ptr<WeightedOperator> W_;
   mutable std::unique_ptr<WeightedOperator> H_;

   mutable std::unique_ptr<HypreParMatrix> stiffness_matrix_;
   mutable std::unique_ptr<HypreParMatrix> mass_matrix_;
   mutable std::unique_ptr<HypreParMatrix> damping_matrix_;
};

} // namespace mfem

#endif
