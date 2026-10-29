// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and j_src code
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

/** @brief Common operations for weighted interior face-jump estimators. */
class FaceJumpEstimatorBase
{
protected:
   FaceJumpEstimatorBase() = default;
   ~FaceJumpEstimatorBase() = default;

   static real_t SmallestEigenvalue(const DenseMatrix &a, int vector_dim,
                                    const char *name);
   static real_t CoefficientMinimum(Coefficient *scalar,
                                    MatrixCoefficient *matrix,
                                    ElementTransformation &tr,
                                    const IntegrationPoint &ip, int vector_dim,
                                    const char *name);
   static void GetFaceNormal(FaceElementTransformations &tr, int vector_dim,
                             Vector &normal);
   static real_t TangentialComponentSquared(const Vector &normal,
                                            const Vector &value);
};

/** @brief Common Maxwell-specific operations for residual estimators.

    This non-instantiable base class centralizes the dimensional conventions
    shared by the real and complex estimators. In particular, it distinguishes
    ordinary 2D fields from the embedded three-component R1D/R2D fields. */
class MaxwellResidualEstimatorBase : protected FaceJumpEstimatorBase
{
protected:
   struct FieldLayout
   {
      int mesh_dim;
      int vector_dim;
      int curl_dim;
   };

   MaxwellResidualEstimatorBase() = default;
   ~MaxwellResidualEstimatorBase() = default;

   static FieldLayout GetFieldLayout(const GridFunction &field);
   static FieldLayout GetFieldLayout(const FiniteElementSpace &fes,
                                     int vector_dim);
   static void GetCurl(const GridFunction &field, ElementTransformation &tr,
                       const FieldLayout &layout, Vector &curl);
   static real_t TangentialJump(const GridFunction &field,
                                FaceElementTransformations &tr,
                                const FieldLayout &layout,
                                const Vector &normal, Vector &first,
                                Vector &second, Vector &cross);
   static void BuildCurlFlux(GridFunction &e, Coefficient &mu_inv,
                             GridFunction &h);
   static void BuildElectricDisplacement(GridFunction &e,
                                         Coefficient &epsilon, GridFunction &d);
   static void BuildElectricDisplacement(GridFunction &e,
                                         MatrixCoefficient &epsilon, GridFunction &d);
   static void BuildComplexCurlFlux(ComplexGridFunction &e,
                                    Coefficient &mu_inv,
                                    ComplexGridFunction &h);
   static void BuildComplexElectricDisplacement(
      ComplexGridFunction &e, MatrixCoefficient &epsilon_real,
      MatrixCoefficient &epsilon_imag, ComplexGridFunction &d);
   static void BuildComplexElectricDisplacement(
      ComplexGridFunction &e, Coefficient &epsilon_real,
      Coefficient &epsilon_imag, ComplexGridFunction &d);
};

/** @brief Shared discontinuous reconstructions used by Maxwell residual terms.

    The object owns the L2 spaces and fields for the curl flux
    $\mathcal{H}=\mu^{-1}\curl E$ and $D=\epsilon E$. The curl flux is the
    auxiliary quantity in the second-order residual; it is not the physical
    time-harmonic magnetic field, which differs by a phasor-dependent factor
    involving $i\omega$. The fields are updated at most once during a
    GeneralErrorEstimator sweep. */
class MaxwellResidualFields : public ErrorEstimatorData,
   protected MaxwellResidualEstimatorBase
{
private:
   GridFunction &e;
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
   MaxwellResidualFields(GridFunction &e_, Coefficient &epsilon_,
                         Coefficient &mu_inv_, int order_);
   MaxwellResidualFields(GridFunction &e_, MatrixCoefficient &epsilon_,
                         Coefficient &mu_inv_, int order_);
   ~MaxwellResidualFields();
   void Update() override;
   GridFunction &CurlFlux() { return *h; }
   GridFunction &D() { return *d; }
};

/** @brief Complex counterpart of MaxwellResidualFields.

    Its curl-flux reconstruction is $\mathcal{H}=\mu^{-1}\curl E$, not the
    physical time-harmonic magnetic field. */
class ComplexMaxwellResidualFields : public ErrorEstimatorData,
   protected MaxwellResidualEstimatorBase
{
private:
   ComplexGridFunction &e;
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
   ComplexMaxwellResidualFields(ComplexGridFunction &e_,
                                MatrixCoefficient &epsilon_real_,
                                MatrixCoefficient &epsilon_imag_,
                                Coefficient &mu_inv_, int order_);
   ComplexMaxwellResidualFields(ComplexGridFunction &e_,
                                Coefficient &epsilon_real_,
                                Coefficient &epsilon_imag_,
                                Coefficient &mu_inv_, int order_);
   ~ComplexMaxwellResidualFields();
   void Update() override;
   ComplexGridFunction &CurlFlux() { return *h; }
   ComplexGridFunction &D() { return *d; }
};

/** @brief Residual estimator for the time-harmonic Maxwell equation.

    This implements the real-valued indicator of
    Chaumont-Frelet and Vega, SIAM J. Numer. Anal. 60 (2022), (3.3)--(3.4),
    for
    $ \curl(\mu^{-1}\curl E)-\omega^2\epsilon E=f $.

    The j_src must be a GridFunction so that its divergence can be evaluated.
    The estimator assumes elementwise constant positive @a mu_inv and either
    an elementwise constant positive scalar or symmetric positive-definite
    matrix @a epsilon. It supports ordinary two- and three-dimensional fields,
    as well as three-component R1D and R2D fields such as ND_R1D, RT_R1D,
    ND_R2D, and RT_R2D. For complex fields and material coefficients, use
    ComplexMaxwellResidualEstimator below.

    The implementation is serial. It includes the volume residuals and both
    interior-face jumps; homogeneous tangential boundary conditions are assumed.
 */
class MaxwellResidualEstimator final : public ErrorEstimator,
   protected MaxwellResidualEstimatorBase
{
private:
   long current_sequence = -1;
   Vector error_estimates;
   real_t total_error = 0.0;
   GridFunction &e;
   GridFunction &j_src;
   Coefficient *epsilon;
   MatrixCoefficient *epsilon_matrix;
   Coefficient &mu_inv;
   real_t omega;
   int order;

   bool MeshIsModified()
   {
      const long sequence = e.FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(sequence >= current_sequence, "improper mesh update sequence");
      return sequence > current_sequence;
   }
   real_t EpsilonMin(ElementTransformation &trans,
                     const IntegrationPoint &ip) const;
   void ComputeEstimates();

public:
   MaxwellResidualEstimator(GridFunction &e_, GridFunction &j_src_,
                            Coefficient &epsilon_, Coefficient &mu_inv_,
                            real_t omega_, int order_)
      : e(e_), j_src(j_src_), epsilon(&epsilon_),
        epsilon_matrix(nullptr), mu_inv(mu_inv_),
        omega(omega_), order(order_)
   { MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive"); }

   /// Construct an estimator with a symmetric positive-definite permittivity.
   MaxwellResidualEstimator(GridFunction &e_, GridFunction &j_src_,
                            MatrixCoefficient &epsilon_, Coefficient &mu_inv_,
                            real_t omega_, int order_)
      : e(e_), j_src(j_src_), epsilon(nullptr),
        epsilon_matrix(&epsilon_), mu_inv(mu_inv_),
        omega(omega_), order(order_)
   { MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive"); }

   const Vector &GetLocalErrors() override
   { if (MeshIsModified()) { ComputeEstimates(); } return error_estimates; }
   real_t GetTotalError() const override;
   void Reset() override { current_sequence = -1; }
};

/** @brief Complex extension of MaxwellResidualEstimator.

    Uses $\epsilon=\epsilon_r+i\epsilon_i$ and complex e/j_src
    GridFunctions. Matrix coefficients must be symmetric; @a epsilon_real must
    be positive definite. The indicator combines the coupled real and imaginary
    residuals before taking its elementwise norm.
 */
class ComplexMaxwellResidualEstimator final : public ErrorEstimator,
   protected MaxwellResidualEstimatorBase
{
private:
   long current_sequence = -1;
   Vector error_estimates;
   real_t total_error = 0.0;
   ComplexGridFunction &e;
   ComplexGridFunction &j_src;
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
   ComplexMaxwellResidualEstimator(ComplexGridFunction &e_,
                                   ComplexGridFunction &j_src_,
                                   MatrixCoefficient &epsilon_real_,
                                   MatrixCoefficient &epsilon_imag_,
                                   Coefficient &mu_inv_, real_t omega_,
                                   int order_)
      : e(e_), j_src(j_src_), epsilon_real(&epsilon_real_),
        epsilon_imag(&epsilon_imag_), epsilon_real_scalar(nullptr),
        epsilon_imag_scalar(nullptr), mu_inv(mu_inv_), omega(omega_), order(order_)
   { MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive"); }

   /// Construct an estimator with scalar complex permittivity coefficients.
   ComplexMaxwellResidualEstimator(ComplexGridFunction &e_,
                                   ComplexGridFunction &j_src_,
                                   Coefficient &epsilon_real_,
                                   Coefficient &epsilon_imag_,
                                   Coefficient &mu_inv_, real_t omega_,
                                   int order_)
      : e(e_), j_src(j_src_), epsilon_real(nullptr),
        epsilon_imag(nullptr), epsilon_real_scalar(&epsilon_real_),
        epsilon_imag_scalar(&epsilon_imag_), mu_inv(mu_inv_), omega(omega_),
        order(order_)
   { MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive"); }

   const Vector &GetLocalErrors() override
   { if (MeshIsModified()) { ComputeEstimates(); } return error_estimates; }
   real_t GetTotalError() const override;
   void Reset() override { current_sequence = -1; }
};


/** @brief Common state for real Maxwell residual volume terms. */
class MaxwellResidualDomainEstimatorBase : public DomainErrorEstimator,
   protected MaxwellResidualEstimatorBase
{
protected:
   GridFunction &e, &j_src;
   std::shared_ptr<MaxwellResidualFields> fields;

   MaxwellResidualDomainEstimatorBase(
      GridFunction &e_, GridFunction &j_src_,
      std::shared_ptr<MaxwellResidualFields> fields_ = nullptr)
      : e(e_), j_src(j_src_), fields(std::move(fields_)) { }
   void Prepare(ErrorEstimatorContext &context) override;
};

/** @brief Curl-residual volume term for real Maxwell problems. */
class MaxwellResidualCurlDomainEstimator final
   : public MaxwellResidualDomainEstimatorBase
{
private:
   GridFunction *h;
   Coefficient *epsilon;
   MatrixCoefficient *epsilon_matrix;
   Coefficient &mu_inv;
   real_t omega;
   int order;
   real_t EpsilonMin(ElementTransformation &trans,
                     const IntegrationPoint &ip) const;

public:
   MaxwellResidualCurlDomainEstimator(GridFunction &e_, GridFunction &j_src_,
                                      GridFunction &h_,
                                      Coefficient &epsilon_, Coefficient &mu_inv_,
                                      real_t omega_, int order_);
   MaxwellResidualCurlDomainEstimator(GridFunction &e_, GridFunction &j_src_,
                                      GridFunction &h_,
                                      MatrixCoefficient &epsilon_, Coefficient &mu_inv_,
                                      real_t omega_, int order_);
   MaxwellResidualCurlDomainEstimator(
      GridFunction &e_, GridFunction &j_src_,
      std::shared_ptr<MaxwellResidualFields> fields_, Coefficient &epsilon_,
      Coefficient &mu_inv_, real_t omega_, int order_);
   MaxwellResidualCurlDomainEstimator(
      GridFunction &e_, GridFunction &j_src_,
      std::shared_ptr<MaxwellResidualFields> fields_, MatrixCoefficient &epsilon_,
      Coefficient &mu_inv_, real_t omega_, int order_);
   real_t GetElementError(ElementTransformation &Tr) override;
};

/** @brief Divergence-residual volume term for real Maxwell problems. */
class MaxwellResidualDivergenceDomainEstimator final
   : public MaxwellResidualDomainEstimatorBase
{
private:
   GridFunction *d;
   Coefficient *epsilon;
   MatrixCoefficient *epsilon_matrix;
   real_t omega;
   int order;
   real_t EpsilonMin(ElementTransformation &trans,
                     const IntegrationPoint &ip) const;

public:
   MaxwellResidualDivergenceDomainEstimator(GridFunction &e_,
                                            GridFunction &j_src_, GridFunction &d_,
                                            Coefficient &epsilon_, real_t omega_,
                                            int order_);
   MaxwellResidualDivergenceDomainEstimator(GridFunction &e_,
                                            GridFunction &j_src_, GridFunction &d_,
                                            MatrixCoefficient &epsilon_, real_t omega_,
                                            int order_);
   MaxwellResidualDivergenceDomainEstimator(
      GridFunction &e_, GridFunction &j_src_,
      std::shared_ptr<MaxwellResidualFields> fields_, Coefficient &epsilon_,
      real_t omega_, int order_);
   MaxwellResidualDivergenceDomainEstimator(
      GridFunction &e_, GridFunction &j_src_,
      std::shared_ptr<MaxwellResidualFields> fields_, MatrixCoefficient &epsilon_,
      real_t omega_, int order_);
   real_t GetElementError(ElementTransformation &Tr) override;
};

/** @brief Common state for real Maxwell residual face terms. */
class MaxwellResidualFaceEstimatorBase : public FaceErrorEstimator,
   protected MaxwellResidualEstimatorBase
{
protected:
   std::shared_ptr<MaxwellResidualFields> fields;

   explicit MaxwellResidualFaceEstimatorBase(
      std::shared_ptr<MaxwellResidualFields> fields_ = nullptr)
      : fields(std::move(fields_)) { }
   void Prepare(ErrorEstimatorContext &context) override;
};

/** @brief Scaling policy for general weighted face-jump estimators. */
enum class FaceJumpScaling
{
   NONE, /**< Use only the supplied constant scale factor. */
   H_OVER_P, /**< Scale each element contribution by $h_K/p_K$. */
   H_OVER_P_OVER_COEFFICIENT /**< Scale each contribution by
      $h_K/(p_K a_{\min,K})$. Requires a positive scalar coefficient or
      a symmetric positive-definite matrix coefficient. */
};

/** @brief Common state and scaling for weighted face-jump estimators. */
class WeightedFaceJumpErrorEstimatorBase : public FaceErrorEstimator,
   protected FaceJumpEstimatorBase
{
protected:
   GridFunction &x;
   Coefficient *a;
   MatrixCoefficient *a_matrix;
   real_t alpha;
   FaceJumpScaling scaling;

   WeightedFaceJumpErrorEstimatorBase(GridFunction &x_, Coefficient *a_,
                                      MatrixCoefficient *a_matrix_, real_t alpha_,
                                      FaceJumpScaling scaling_);
   real_t FaceScale(int element) const;
   void ExchangeFaceNbrData() override;
};

/** @brief Estimate the jump in the normal trace of a weighted Nedelec field.

    On each interior face this estimator evaluates the coefficient separately
    on its two adjacent elements. Thus, @a a may be discontinuous. The field
    @a x must use an H(curl), H(curl)-R2D, or H(curl)-R1D finite-element map.
    Omitting @a a applies the identity map. The optional @a alpha scales the
    squared contribution and defaults to one.

    The @a scaling policy additionally selects no mesh scaling, $h_K/p_K$
    scaling, or $h_K/(p_K a_{\min,K})$ scaling for each element. The last
    option requires a positive scalar @a a or a symmetric positive-definite
    matrix @a a. Its weights can match a Maxwell normal-jump term when
    @a a is permittivity and @a alpha is $\omega^2$, but this estimator
    evaluates $a x$ directly. It does not reproduce a specialized Maxwell
    indicator that instead jumps a discontinuous reconstructed field. */
class NedelecNormalJumpErrorEstimator final
   : public WeightedFaceJumpErrorEstimatorBase
{
public:
   NedelecNormalJumpErrorEstimator(GridFunction &x_, real_t alpha_ = 1.0,
                                   FaceJumpScaling scaling_ = FaceJumpScaling::NONE);
   NedelecNormalJumpErrorEstimator(GridFunction &x_, Coefficient &a_,
                                   real_t alpha_ = 1.0,
                                   FaceJumpScaling scaling_ = FaceJumpScaling::NONE);
   NedelecNormalJumpErrorEstimator(GridFunction &x_, MatrixCoefficient &a_,
                                   real_t alpha_ = 1.0,
                                   FaceJumpScaling scaling_ = FaceJumpScaling::NONE);
   void GetFaceError(FaceElementTransformations &Tr, real_t &error1,
                     real_t &error2) override;
};

/** @brief Estimate the jump in the tangential trace of a weighted RT field.

    On each interior face this estimator evaluates the coefficient separately
    on its two adjacent elements and returns the same contribution for each:
    $\alpha\|n\times((a x)_1-(a x)_2)\|^2_{L^2(F)}$.
    In two dimensions, the cross product denotes its scalar out-of-plane
    component. Thus, @a a may be discontinuous. The field @a x must use an
    H(div), H(div)-R2D, or H(div)-R1D finite-element map. Omitting @a a
    applies the identity map. The optional @a alpha scales the squared
    contribution and defaults to one.

    The @a scaling policy additionally selects no mesh scaling, $h_K/p_K$
    scaling, or $h_K/(p_K a_{\min,K})$ scaling for each element. The last
    option requires a positive scalar @a a or a symmetric positive-definite
    matrix @a a. Its weights can match a Maxwell tangential-jump term when
    @a a is inverse permeability, but this estimator evaluates $a x$
    directly. It does not reproduce a specialized Maxwell indicator that
    jumps a discontinuous reconstructed curl-flux field. */
class RTTangentialJumpErrorEstimator final
   : public WeightedFaceJumpErrorEstimatorBase
{
public:
   RTTangentialJumpErrorEstimator(GridFunction &x_, real_t alpha_ = 1.0,
                                  FaceJumpScaling scaling_ = FaceJumpScaling::NONE);
   RTTangentialJumpErrorEstimator(GridFunction &x_, Coefficient &a_,
                                  real_t alpha_ = 1.0,
                                  FaceJumpScaling scaling_ = FaceJumpScaling::NONE);
   RTTangentialJumpErrorEstimator(GridFunction &x_, MatrixCoefficient &a_,
                                  real_t alpha_ = 1.0,
                                  FaceJumpScaling scaling_ = FaceJumpScaling::NONE);
   void GetFaceError(FaceElementTransformations &Tr, real_t &error1,
                     real_t &error2) override;
};

/** @brief Tangential curl-flux jump term for real Maxwell problems.

    This specialized term jumps the shared discontinuous reconstruction
    $\mathcal{H}_h\approx\mu^{-1}\curl E$, rather than evaluating the flux directly on
    the face. This preserves equivalence with MaxwellResidualEstimator and
    shares the reconstruction with its related Maxwell residual terms. */
class MaxwellResidualTangentialFaceEstimator final
   : public MaxwellResidualFaceEstimatorBase
{
private:
   GridFunction *h;
   Coefficient &mu_inv;
   int order;

public:
   MaxwellResidualTangentialFaceEstimator(GridFunction &h_, Coefficient &mu_inv_,
                                          int order_);
   MaxwellResidualTangentialFaceEstimator(
      std::shared_ptr<MaxwellResidualFields> fields_, Coefficient &mu_inv_,
      int order_);
   void GetFaceError(FaceElementTransformations &Tr, real_t &error1,
                     real_t &error2) override;
   void ExchangeFaceNbrData() override;
};

/** @brief Normal electric-displacement jump term for real Maxwell problems.

    This specialized term jumps the shared discontinuous reconstruction
    $D_h\approx\epsilon E$, rather than evaluating $\epsilon E$ directly on
    the face. This preserves equivalence with MaxwellResidualEstimator and
    shares the reconstruction with its related Maxwell residual terms. */
class MaxwellResidualNormalFaceEstimator final
   : public MaxwellResidualFaceEstimatorBase
{
private:
   GridFunction *d;
   Coefficient *epsilon;
   MatrixCoefficient *epsilon_matrix;
   real_t omega;
   int order;
   real_t EpsilonMin(ElementTransformation &trans,
                     const IntegrationPoint &ip) const;

public:
   MaxwellResidualNormalFaceEstimator(GridFunction &d_, Coefficient &epsilon_,
                                      real_t omega_, int order_);
   MaxwellResidualNormalFaceEstimator(GridFunction &d_,
                                      MatrixCoefficient &epsilon_,
                                      real_t omega_, int order_);
   MaxwellResidualNormalFaceEstimator(std::shared_ptr<MaxwellResidualFields>
                                      fields_,
                                      Coefficient &epsilon_, real_t omega_, int order_);
   MaxwellResidualNormalFaceEstimator(std::shared_ptr<MaxwellResidualFields>
                                      fields_,
                                      MatrixCoefficient &epsilon_, real_t omega_, int order_);
   void GetFaceError(FaceElementTransformations &Tr, real_t &error1,
                     real_t &error2) override;
   void ExchangeFaceNbrData() override;
};

/** @brief Shared state for complex Maxwell residual volume terms. */
class ComplexMaxwellResidualDomainEstimatorBase : public DomainErrorEstimator,
   protected MaxwellResidualEstimatorBase
{
protected:
   ComplexGridFunction &e, &j_src;
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
   ComplexMaxwellResidualDomainEstimatorBase(ComplexGridFunction &e_,
                                             ComplexGridFunction &j_src_,
                                             ComplexGridFunction &h_,
                                             ComplexGridFunction &d_,
                                             MatrixCoefficient &epsilon_real_,
                                             MatrixCoefficient &epsilon_imag_,
                                             Coefficient &mu_inv_, real_t omega_, int order_);
   ComplexMaxwellResidualDomainEstimatorBase(
      ComplexGridFunction &e_, ComplexGridFunction &j_src_,
      std::shared_ptr<ComplexMaxwellResidualFields> fields_,
      MatrixCoefficient &epsilon_real_, MatrixCoefficient &epsilon_imag_,
      Coefficient &mu_inv_, real_t omega_, int order_);
   ComplexMaxwellResidualDomainEstimatorBase(
      ComplexGridFunction &e_, ComplexGridFunction &j_src_,
      std::shared_ptr<ComplexMaxwellResidualFields> fields_,
      Coefficient &epsilon_real_, Coefficient &epsilon_imag_,
      Coefficient &mu_inv_, real_t omega_, int order_);
   void Prepare(ErrorEstimatorContext &context) override;
   ComplexMaxwellResidualDomainEstimatorBase(ComplexGridFunction &e_,
                                             ComplexGridFunction &j_src_,
                                             ComplexGridFunction &h_,
                                             ComplexGridFunction &d_,
                                             Coefficient &epsilon_real_,
                                             Coefficient &epsilon_imag_,
                                             Coefficient &mu_inv_, real_t omega_, int order_);
};

/** @brief Curl-residual volume term of ComplexMaxwellResidualEstimator.

    @a h and @a d are discontinuous complex fields representing
    $\mu^{-1}\curl E$ and $\epsilon E$. Returned values are squared
    local indicator contributions. */
class ComplexMaxwellResidualCurlDomainEstimator
   : public ComplexMaxwellResidualDomainEstimatorBase
{
public:
   using ComplexMaxwellResidualDomainEstimatorBase::ComplexMaxwellResidualDomainEstimatorBase;
   real_t GetElementError(ElementTransformation &Tr) override;
};

/** @brief Divergence-residual volume term of ComplexMaxwellResidualEstimator. */
class ComplexMaxwellResidualDivergenceDomainEstimator final
   : public ComplexMaxwellResidualDomainEstimatorBase
{
public:
   using ComplexMaxwellResidualDomainEstimatorBase::ComplexMaxwellResidualDomainEstimatorBase;
   real_t GetElementError(ElementTransformation &Tr) override;
};

/** @brief Shared reconstruction state for complex Maxwell face terms. */
class ComplexMaxwellResidualFaceEstimatorBase : public FaceErrorEstimator,
   protected MaxwellResidualEstimatorBase
{
protected:
   ComplexGridFunction *h, *d;
   std::shared_ptr<ComplexMaxwellResidualFields> fields;

   ComplexMaxwellResidualFaceEstimatorBase(ComplexGridFunction *h_,
                                           ComplexGridFunction *d_)
      : h(h_), d(d_) { }
   explicit ComplexMaxwellResidualFaceEstimatorBase(
      std::shared_ptr<ComplexMaxwellResidualFields> fields_)
      : h(nullptr), d(nullptr), fields(std::move(fields_))
   { MFEM_VERIFY(fields, "Complex Maxwell residual fields must be provided."); }

   ComplexGridFunction &CurlFlux()
   { return fields ? fields->CurlFlux() : *h; }
   ComplexGridFunction &D()
   { return fields ? fields->D() : *d; }
   const ComplexGridFunction &D() const
   { return fields ? fields->D() : *d; }
   void Prepare(ErrorEstimatorContext &context) override;
   static void ExchangeFieldFaceNbrData(ComplexGridFunction &field);
};

/** @brief Tangential curl-flux jump term of ComplexMaxwellResidualEstimator.

    This specialized term jumps the shared complex discontinuous reconstruction
    $\mathcal{H}_h\approx\mu^{-1}\curl E$, preserving equivalence with
    ComplexMaxwellResidualEstimator and sharing it with the related complex
    residual terms. */
class ComplexMaxwellResidualTangentialFaceEstimator final
   : public ComplexMaxwellResidualFaceEstimatorBase
{
private:
   Coefficient &mu_inv;
   int order;

public:
   ComplexMaxwellResidualTangentialFaceEstimator(ComplexGridFunction &h_,
                                                 Coefficient &mu_inv_, int order_);
   ComplexMaxwellResidualTangentialFaceEstimator(
      std::shared_ptr<ComplexMaxwellResidualFields> fields_, Coefficient &mu_inv_,
      int order_);
   void GetFaceError(FaceElementTransformations &Tr, real_t &error1,
                     real_t &error2) override;
   void ExchangeFaceNbrData() override;
};

/** @brief Normal electric-displacement jump term of ComplexMaxwellResidualEstimator.

    This specialized term jumps the shared complex discontinuous reconstruction
    $D_h\approx\epsilon E$, rather than evaluating $\epsilon E$ directly on
    the face. This preserves equivalence with ComplexMaxwellResidualEstimator
    and shares the reconstruction with the related complex residual terms. */
class ComplexMaxwellResidualNormalFaceEstimator final
   : public ComplexMaxwellResidualFaceEstimatorBase
{
private:
   MatrixCoefficient *epsilon_real;
   Coefficient *epsilon_real_scalar;
   real_t omega;
   int order;
   real_t EpsilonMin(ElementTransformation &trans,
                     const IntegrationPoint &ip) const;

public:
   ComplexMaxwellResidualNormalFaceEstimator(ComplexGridFunction &d_,
                                             MatrixCoefficient &epsilon_real_,
                                             real_t omega_, int order_);
   ComplexMaxwellResidualNormalFaceEstimator(ComplexGridFunction &d_,
                                             Coefficient &epsilon_real_,
                                             real_t omega_, int order_);
   ComplexMaxwellResidualNormalFaceEstimator(
      std::shared_ptr<ComplexMaxwellResidualFields> fields_,
      MatrixCoefficient &epsilon_real_, real_t omega_, int order_);
   ComplexMaxwellResidualNormalFaceEstimator(
      std::shared_ptr<ComplexMaxwellResidualFields> fields_,
      Coefficient &epsilon_real_, real_t omega_, int order_);
   void GetFaceError(FaceElementTransformations &Tr, real_t &error1,
                     real_t &error2) override;
   void ExchangeFaceNbrData() override;
};

/** @brief Check a complex tangential-electric (Dirichlet) boundary trace.

    The estimator returns $\|E\times n-g_D\|^2_{L^2(F)}$. Passing no
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
   real_t GetFaceError(FaceElementTransformations &Tr) override;
};

/** @brief Check a complex tangential magnetic-flux (Neumann) boundary trace.

    @a magnetic_flux must be the reconstructed field
    $H=\mu^{-1}\curl E$. The estimator returns
    $\|H\times n-g_N\|^2_{L^2(F)}$. */
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
   real_t GetFaceError(FaceElementTransformations &Tr) override;
};

/** Add all real Maxwell residual terms using one shared reconstruction. */
void AddMaxwellResidualEstimators(GeneralErrorEstimator &estimator,
                                  GridFunction &e, GridFunction &j_src,
                                  Coefficient &epsilon, Coefficient &mu_inv,
                                  real_t omega, int order);
void AddMaxwellResidualEstimators(GeneralErrorEstimator &estimator,
                                  GridFunction &e, GridFunction &j_src,
                                  MatrixCoefficient &epsilon, Coefficient &mu_inv,
                                  real_t omega, int order);

/** Add all complex Maxwell residual terms using one shared reconstruction. */
void AddComplexMaxwellResidualEstimators(
   GeneralErrorEstimator &estimator, ComplexGridFunction &e,
   ComplexGridFunction &j_src, MatrixCoefficient &epsilon_real,
   MatrixCoefficient &epsilon_imag, Coefficient &mu_inv, real_t omega, int order);
void AddComplexMaxwellResidualEstimators(
   GeneralErrorEstimator &estimator, ComplexGridFunction &e,
   ComplexGridFunction &j_src, Coefficient &epsilon_real,
   Coefficient &epsilon_imag, Coefficient &mu_inv, real_t omega, int order);
} // namespace mfem

#endif // MFEM_ESTIMATOR_IMPL
