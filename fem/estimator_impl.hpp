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

#ifndef MFEM_ESTIMATOR_IMPL
#define MFEM_ESTIMATOR_IMPL

#include "estimators.hpp"

namespace mfem
{

class ComplexGridFunction;

/** @brief Shared discontinuous reconstructions used by Maxwell residual terms.

    The object owns the L2 spaces and fields for
    \f$H=\mu^{-1}\curl E\f$ and \f$D=\epsilon E\f$. It is updated at most
    once during a GeneralErrorEstimator sweep. */
class MaxwellResidualFields : public ErrorEstimatorData
{
private:
   GridFunction &solution;
   Coefficient *epsilon;
   MatrixCoefficient *epsilon_matrix;
   Coefficient &mu_inv;
   int order;
   long mesh_sequence = -1;
   std::unique_ptr<L2_FECollection> h_fec, d_fec;
   std::unique_ptr<FiniteElementSpace> h_fes, d_fes;
   std::unique_ptr<GridFunction> h, d;

   void BuildSpaces();

public:
   MaxwellResidualFields(GridFunction &solution_, Coefficient &epsilon_,
                         Coefficient &mu_inv_, int order_);
   MaxwellResidualFields(GridFunction &solution_, MatrixCoefficient &epsilon_,
                         Coefficient &mu_inv_, int order_);
   ~MaxwellResidualFields();
   void Update() override;
   GridFunction &H() { return *h; }
   GridFunction &D() { return *d; }
};

/** @brief Complex counterpart of MaxwellResidualFields. */
class ComplexMaxwellResidualFields : public ErrorEstimatorData
{
private:
   ComplexGridFunction &solution;
   MatrixCoefficient *epsilon_real, *epsilon_imag;
   Coefficient *epsilon_real_scalar, *epsilon_imag_scalar;
   Coefficient &mu_inv;
   int order;
   long mesh_sequence = -1;
   std::unique_ptr<L2_FECollection> h_fec, d_fec;
   std::unique_ptr<FiniteElementSpace> h_fes, d_fes;
   std::unique_ptr<ComplexGridFunction> h, d;

   void BuildSpaces();

public:
   ComplexMaxwellResidualFields(ComplexGridFunction &solution_,
                                MatrixCoefficient &epsilon_real_,
                                MatrixCoefficient &epsilon_imag_,
                                Coefficient &mu_inv_, int order_);
   ComplexMaxwellResidualFields(ComplexGridFunction &solution_,
                                Coefficient &epsilon_real_,
                                Coefficient &epsilon_imag_,
                                Coefficient &mu_inv_, int order_);
   ~ComplexMaxwellResidualFields();
   void Update() override;
   ComplexGridFunction &H() { return *h; }
   ComplexGridFunction &D() { return *d; }
};

/** @brief Residual estimator for the time-harmonic Maxwell equation.

    This implements the real-valued indicator of
    Chaumont-Frelet and Vega, SIAM J. Numer. Anal. 60 (2022), (3.3)--(3.4),
    for
    \f$ \curl(\mu^{-1}\curl E)-\omega^2\epsilon E=f \f$.

    The source must be a GridFunction so that its divergence can be evaluated.
    The estimator assumes elementwise constant positive @a mu_inv and either
    an elementwise constant positive scalar or symmetric positive-definite
    matrix @a epsilon. For complex fields and material coefficients, use
    ComplexMaxwellResidualEstimator below.

    The implementation is serial. It includes the volume residuals and both
    interior-face jumps; homogeneous tangential boundary conditions are assumed.
 */
class MaxwellResidualEstimator final : public ErrorEstimator
{
private:
   long current_sequence = -1;
   Vector error_estimates;
   real_t total_error = 0.0;
   GridFunction &solution;
   GridFunction &source;
   Coefficient *epsilon;
   MatrixCoefficient *epsilon_matrix;
   Coefficient &mu_inv;
   real_t omega;
   int order;

   bool MeshIsModified()
   {
      const long sequence = solution.FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(sequence >= current_sequence, "improper mesh update sequence");
      return sequence > current_sequence;
   }
   real_t EpsilonMin(ElementTransformation &trans,
                     const IntegrationPoint &ip) const;
   void ComputeEstimates();

public:
   MaxwellResidualEstimator(GridFunction &solution_, GridFunction &source_,
                            Coefficient &epsilon_, Coefficient &mu_inv_,
                            real_t omega_, int order_)
      : solution(solution_), source(source_), epsilon(&epsilon_),
        epsilon_matrix(nullptr), mu_inv(mu_inv_),
        omega(omega_), order(order_)
   { MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive"); }

   /// Construct an estimator with a symmetric positive-definite permittivity.
   MaxwellResidualEstimator(GridFunction &solution_, GridFunction &source_,
                            MatrixCoefficient &epsilon_, Coefficient &mu_inv_,
                            real_t omega_, int order_)
      : solution(solution_), source(source_), epsilon(nullptr),
        epsilon_matrix(&epsilon_), mu_inv(mu_inv_),
        omega(omega_), order(order_)
   { MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive"); }

   const Vector &GetLocalErrors() override
   { if (MeshIsModified()) { ComputeEstimates(); } return error_estimates; }
   real_t GetTotalError() const override;
   void Reset() override { current_sequence = -1; }
};

/** @brief Complex extension of MaxwellResidualEstimator.

    Uses \f$\epsilon=\epsilon_r+i\epsilon_i\f$ and complex solution/source
    GridFunctions. Matrix coefficients must be symmetric; @a epsilon_real must
    be positive definite. The indicator combines the coupled real and imaginary
    residuals before taking its elementwise norm.
 */
class ComplexMaxwellResidualEstimator final : public ErrorEstimator
{
private:
   long current_sequence = -1;
   Vector error_estimates;
   real_t total_error = 0.0;
   ComplexGridFunction &solution;
   ComplexGridFunction &source;
   MatrixCoefficient *epsilon_real;
   MatrixCoefficient *epsilon_imag;
   Coefficient *epsilon_real_scalar;
   Coefficient *epsilon_imag_scalar;
   Coefficient &mu_inv;
   real_t omega;
   int order;

   bool MeshIsModified();
   real_t EpsilonMin(ElementTransformation &trans,
                     const IntegrationPoint &ip) const;
   void ComputeEstimates();

public:
   ComplexMaxwellResidualEstimator(ComplexGridFunction &solution_,
                                   ComplexGridFunction &source_,
                                   MatrixCoefficient &epsilon_real_,
                                   MatrixCoefficient &epsilon_imag_,
                                   Coefficient &mu_inv_, real_t omega_,
                                   int order_)
      : solution(solution_), source(source_), epsilon_real(&epsilon_real_),
        epsilon_imag(&epsilon_imag_), epsilon_real_scalar(nullptr),
        epsilon_imag_scalar(nullptr), mu_inv(mu_inv_), omega(omega_), order(order_)
   { MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive"); }

   /// Construct an estimator with scalar complex permittivity coefficients.
   ComplexMaxwellResidualEstimator(ComplexGridFunction &solution_,
                                   ComplexGridFunction &source_,
                                   Coefficient &epsilon_real_,
                                   Coefficient &epsilon_imag_,
                                   Coefficient &mu_inv_, real_t omega_,
                                   int order_)
      : solution(solution_), source(source_), epsilon_real(nullptr),
        epsilon_imag(nullptr), epsilon_real_scalar(&epsilon_real_),
        epsilon_imag_scalar(&epsilon_imag_), mu_inv(mu_inv_), omega(omega_),
        order(order_)
   { MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive"); }

   const Vector &GetLocalErrors() override
   { if (MeshIsModified()) { ComputeEstimates(); } return error_estimates; }
   real_t GetTotalError() const override;
   void Reset() override { current_sequence = -1; }
};


/** @brief Volume terms of the residual estimator for real Maxwell problems.

    The supplied @a h and @a d fields must be discontinuous L2 projections of
    \f$\mu^{-1}\curl E\f$ and \f$\epsilon E\f$, respectively. This is the same
    reconstruction used by MaxwellResidualEstimator. The returned value is the
    squared local indicator contribution. */
class MaxwellResidualDomainEstimator final : public DomainErrorEstimator
{
private:
   GridFunction &solution, &source;
   GridFunction *h, *d;
   std::shared_ptr<MaxwellResidualFields> fields;
   Coefficient *epsilon;
   MatrixCoefficient *epsilon_matrix;
   Coefficient &mu_inv;
   real_t omega;
   int order;
   real_t EpsilonMin(ElementTransformation &trans,
                     const IntegrationPoint &ip) const;

public:
   MaxwellResidualDomainEstimator(GridFunction &solution_, GridFunction &source_,
                                  GridFunction &h_, GridFunction &d_,
                                  Coefficient &epsilon_, Coefficient &mu_inv_,
                                  real_t omega_, int order_);
   MaxwellResidualDomainEstimator(GridFunction &solution_, GridFunction &source_,
                                  GridFunction &h_, GridFunction &d_,
                                  MatrixCoefficient &epsilon_, Coefficient &mu_inv_,
                                  real_t omega_, int order_);
   MaxwellResidualDomainEstimator(GridFunction &solution_, GridFunction &source_,
                                  std::shared_ptr<MaxwellResidualFields> fields_,
                                  Coefficient &epsilon_, Coefficient &mu_inv_,
                                  real_t omega_, int order_);
   MaxwellResidualDomainEstimator(GridFunction &solution_, GridFunction &source_,
                                  std::shared_ptr<MaxwellResidualFields> fields_,
                                  MatrixCoefficient &epsilon_, Coefficient &mu_inv_,
                                  real_t omega_, int order_);
   void Prepare(ErrorEstimatorContext &context) override;
   real_t GetElementError(const FiniteElement &el,
                          ElementTransformation &Tr) override;
};

/** Build the discontinuous residual fields required by the Maxwell estimators. */
void BuildMaxwellResidualFields(GridFunction &solution, Coefficient &mu_inv,
                                Coefficient &epsilon, GridFunction &h,
                                GridFunction &d);
void BuildMaxwellResidualFields(GridFunction &solution, Coefficient &mu_inv,
                                MatrixCoefficient &epsilon, GridFunction &h,
                                GridFunction &d);

/** @brief Face-jump terms of the residual estimator for real Maxwell problems.

    @a h and @a d have the same meaning as in MaxwellResidualDomainEstimator.
    Interior contributions are returned separately for the two neighboring
    elements, matching MaxwellResidualEstimator's elementwise indicators. */
class MaxwellResidualFaceEstimator final : public FaceErrorEstimator
{
private:
   GridFunction *h, *d;
   std::shared_ptr<MaxwellResidualFields> fields;
   Coefficient *epsilon;
   MatrixCoefficient *epsilon_matrix;
   Coefficient &mu_inv;
   real_t omega;
   int order;
   real_t EpsilonMin(ElementTransformation &trans,
                     const IntegrationPoint &ip) const;

public:
   MaxwellResidualFaceEstimator(GridFunction &h_, GridFunction &d_,
                                Coefficient &epsilon_, Coefficient &mu_inv_,
                                real_t omega_, int order_);
   MaxwellResidualFaceEstimator(GridFunction &h_, GridFunction &d_,
                                MatrixCoefficient &epsilon_, Coefficient &mu_inv_,
                                real_t omega_, int order_);
   MaxwellResidualFaceEstimator(std::shared_ptr<MaxwellResidualFields> fields_,
                                Coefficient &epsilon_, Coefficient &mu_inv_,
                                real_t omega_, int order_);
   MaxwellResidualFaceEstimator(std::shared_ptr<MaxwellResidualFields> fields_,
                                MatrixCoefficient &epsilon_, Coefficient &mu_inv_,
                                real_t omega_, int order_);
   void Prepare(ErrorEstimatorContext &context) override;
   void GetFaceError(const FiniteElement &el1, const FiniteElement &el2,
                     FaceElementTransformations &Tr, real_t &error1,
                     real_t &error2) override;
   void ExchangeFaceNbrData() override;
};

/** @brief Volume terms of ComplexMaxwellResidualEstimator.

    @a h and @a d are discontinuous complex fields representing
    \f$\mu^{-1}\curl E\f$ and \f$\epsilon E\f$. Returned values are squared
    local indicator contributions. */
class ComplexMaxwellResidualDomainEstimator final : public DomainErrorEstimator
{
private:
   ComplexGridFunction &solution, &source;
   ComplexGridFunction *h, *d;
   std::shared_ptr<ComplexMaxwellResidualFields> fields;
   MatrixCoefficient *epsilon_real, *epsilon_imag;
   Coefficient *epsilon_real_scalar, *epsilon_imag_scalar;
   Coefficient &mu_inv;
   real_t omega;
   int order;
   real_t EpsilonMin(ElementTransformation &trans,
                     const IntegrationPoint &ip) const;

public:
   ComplexMaxwellResidualDomainEstimator(ComplexGridFunction &solution_,
                                         ComplexGridFunction &source_,
                                         ComplexGridFunction &h_, ComplexGridFunction &d_,
                                         MatrixCoefficient &epsilon_real_,
                                         MatrixCoefficient &epsilon_imag_,
                                         Coefficient &mu_inv_, real_t omega_, int order_);
   ComplexMaxwellResidualDomainEstimator(
      ComplexGridFunction &solution_, ComplexGridFunction &source_,
      std::shared_ptr<ComplexMaxwellResidualFields> fields_,
      MatrixCoefficient &epsilon_real_, MatrixCoefficient &epsilon_imag_,
      Coefficient &mu_inv_, real_t omega_, int order_);
   ComplexMaxwellResidualDomainEstimator(
      ComplexGridFunction &solution_, ComplexGridFunction &source_,
      std::shared_ptr<ComplexMaxwellResidualFields> fields_,
      Coefficient &epsilon_real_, Coefficient &epsilon_imag_,
      Coefficient &mu_inv_, real_t omega_, int order_);
   void Prepare(ErrorEstimatorContext &context) override;
   ComplexMaxwellResidualDomainEstimator(ComplexGridFunction &solution_,
                                         ComplexGridFunction &source_,
                                         ComplexGridFunction &h_, ComplexGridFunction &d_,
                                         Coefficient &epsilon_real_,
                                         Coefficient &epsilon_imag_,
                                         Coefficient &mu_inv_, real_t omega_, int order_);
   real_t GetElementError(const FiniteElement &el,
                          ElementTransformation &Tr) override;
};

/** @brief Face-jump terms of ComplexMaxwellResidualEstimator. */
class ComplexMaxwellResidualFaceEstimator final : public FaceErrorEstimator
{
private:
   ComplexGridFunction *h, *d;
   std::shared_ptr<ComplexMaxwellResidualFields> fields;
   MatrixCoefficient *epsilon_real, *epsilon_imag;
   Coefficient *epsilon_real_scalar, *epsilon_imag_scalar;
   Coefficient &mu_inv;
   real_t omega;
   int order;
   real_t EpsilonMin(ElementTransformation &trans,
                     const IntegrationPoint &ip) const;

public:
   ComplexMaxwellResidualFaceEstimator(ComplexGridFunction &h_,
                                       ComplexGridFunction &d_,
                                       MatrixCoefficient &epsilon_real_,
                                       MatrixCoefficient &epsilon_imag_,
                                       Coefficient &mu_inv_, real_t omega_, int order_);
   ComplexMaxwellResidualFaceEstimator(
      std::shared_ptr<ComplexMaxwellResidualFields> fields_,
      MatrixCoefficient &epsilon_real_, MatrixCoefficient &epsilon_imag_,
      Coefficient &mu_inv_, real_t omega_, int order_);
   ComplexMaxwellResidualFaceEstimator(
      std::shared_ptr<ComplexMaxwellResidualFields> fields_,
      Coefficient &epsilon_real_, Coefficient &epsilon_imag_,
      Coefficient &mu_inv_, real_t omega_, int order_);
   void Prepare(ErrorEstimatorContext &context) override;
   ComplexMaxwellResidualFaceEstimator(ComplexGridFunction &h_,
                                       ComplexGridFunction &d_, Coefficient &epsilon_real_,
                                       Coefficient &epsilon_imag_, Coefficient &mu_inv_,
                                       real_t omega_, int order_);
   void GetFaceError(const FiniteElement &el1, const FiniteElement &el2,
                     FaceElementTransformations &Tr, real_t &error1,
                     real_t &error2) override;
   void ExchangeFaceNbrData() override;
};

/** @brief Check a complex tangential-electric (Dirichlet) boundary trace.

    The estimator returns \f$\|E\times n-g_D\|^2_{L^2(F)}\f$. Passing no
    boundary data checks homogeneous Dirichlet conditions. */
class ComplexMaxwellDirichletBCErrorEstimator final : public FaceErrorEstimator
{
private:
   ComplexGridFunction &electric;
   VectorCoefficient *data_real;
   VectorCoefficient *data_imag;

public:
   explicit ComplexMaxwellDirichletBCErrorEstimator(ComplexGridFunction &electric_)
      : electric(electric_), data_real(nullptr), data_imag(nullptr) { }
   ComplexMaxwellDirichletBCErrorEstimator(ComplexGridFunction &electric_,
                                           VectorCoefficient &data_real_,
                                           VectorCoefficient &data_imag_)
      : electric(electric_), data_real(&data_real_), data_imag(&data_imag_) { }
   real_t GetFaceError(const FiniteElement &el,
                       FaceElementTransformations &Tr) override;
};

/** @brief Check a complex tangential magnetic-flux (Neumann) boundary trace.

    @a magnetic_flux must be the reconstructed field
    \f$H=\mu^{-1}\curl E\f$. The estimator returns
    \f$\|H\times n-g_N\|^2_{L^2(F)}\f$. */
class ComplexMaxwellNeumannBCErrorEstimator final : public FaceErrorEstimator
{
private:
   ComplexGridFunction &magnetic_flux;
   VectorCoefficient *data_real;
   VectorCoefficient *data_imag;

public:
   explicit ComplexMaxwellNeumannBCErrorEstimator(ComplexGridFunction
                                                  &magnetic_flux_)
      : magnetic_flux(magnetic_flux_), data_real(nullptr), data_imag(nullptr) { }
   ComplexMaxwellNeumannBCErrorEstimator(ComplexGridFunction &magnetic_flux_,
                                         VectorCoefficient &data_real_,
                                         VectorCoefficient &data_imag_)
      : magnetic_flux(magnetic_flux_), data_real(&data_real_),
        data_imag(&data_imag_) { }
   real_t GetFaceError(const FiniteElement &el,
                       FaceElementTransformations &Tr) override;
};

/** Build the discontinuous complex residual fields used by the complex estimators. */
void BuildComplexMaxwellResidualFields(ComplexGridFunction &solution,
                                       Coefficient &mu_inv,
                                       MatrixCoefficient &epsilon_real,
                                       MatrixCoefficient &epsilon_imag,
                                       ComplexGridFunction &h,
                                       ComplexGridFunction &d);

/** Add all real Maxwell residual terms using one shared reconstruction. */
void AddMaxwellResidualEstimators(GeneralErrorEstimator &estimator,
                                  GridFunction &solution, GridFunction &source,
                                  Coefficient &epsilon, Coefficient &mu_inv,
                                  real_t omega, int order);
void AddMaxwellResidualEstimators(GeneralErrorEstimator &estimator,
                                  GridFunction &solution, GridFunction &source,
                                  MatrixCoefficient &epsilon, Coefficient &mu_inv,
                                  real_t omega, int order);

/** Add all complex Maxwell residual terms using one shared reconstruction. */
void AddComplexMaxwellResidualEstimators(
   GeneralErrorEstimator &estimator, ComplexGridFunction &solution,
   ComplexGridFunction &source, MatrixCoefficient &epsilon_real,
   MatrixCoefficient &epsilon_imag, Coefficient &mu_inv, real_t omega, int order);
void AddComplexMaxwellResidualEstimators(
   GeneralErrorEstimator &estimator, ComplexGridFunction &solution,
   ComplexGridFunction &source, Coefficient &epsilon_real,
   Coefficient &epsilon_imag, Coefficient &mu_inv, real_t omega, int order);
void BuildComplexMaxwellResidualFields(ComplexGridFunction &solution,
                                       Coefficient &mu_inv, Coefficient &epsilon_real,
                                       Coefficient &epsilon_imag,
                                       ComplexGridFunction &h,
                                       ComplexGridFunction &d);

} // namespace mfem

#endif // MFEM_ESTIMATOR_IMPL
