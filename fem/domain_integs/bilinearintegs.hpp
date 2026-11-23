#ifndef BILINEARINTEGS_HPP
#define BILINEARINTEGS_HPP

/// @file
/// Public volume bilinear-form operations and element/partial-assembly adapters.
/// See bilinearintegs_components.md for examples.

#include <mfem.hpp>

#include "parallel_wrappers.hpp"
#include "bilininteg_operator.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <limits>
#include <memory>
#include <type_traits>

namespace mfem {

/// Selection policy for tensor-product element/partial assembly. Auto detects
/// scalar (including Cartesian component) and native H(curl)/H(div) tensor bases
/// with compatible quadrature within any supplied runtime optimization limits.
/// Force selects the tensor path regardless of those limits, but rejects
/// unsupported inputs; PA still uses registered kernels and its small-basis GPU
/// optimization. Disable always uses the multidimensional basis path.
enum class BlfTensorProductMode { Auto, Force, Disable };

/// Runtime limits for selecting tensor-product kernels in Auto mode. Zero
/// means unbounded. These limits are checked only during assembly setup and do
/// not affect the generated contraction kernels.
struct BlfTensorProductLimits {
  /// Maximum trial closed-basis DOFs per axis; zero disables this limit.
  int max_trial_dof_1d = 0;
  /// Maximum test closed-basis DOFs per axis; zero disables this limit.
  int max_test_dof_1d = 0;
  /// Maximum quadrature points per axis; zero disables this limit.
  int max_quad_1d = 0;

  /// Reject negative limits via MFEM_VERIFY.
  void Verify() const {
    MFEM_VERIFY(max_trial_dof_1d >= 0 && max_test_dof_1d >= 0 &&
                    max_quad_1d >= 0,
                "tensor-product optimization maxima must be nonnegative");
  }

  /// @param trial_dof_1d Trial closed-basis DOFs per axis.
  /// @param test_dof_1d Test closed-basis DOFs per axis.
  /// @param quad_1d Quadrature points per axis.
  /// @return Whether all enabled limits admit this layout.
  bool Contains(int trial_dof_1d, int test_dof_1d, int quad_1d) const {
    return (max_trial_dof_1d == 0 || trial_dof_1d <= max_trial_dof_1d) &&
           (max_test_dof_1d == 0 || test_dof_1d <= max_test_dof_1d) &&
           (max_quad_1d == 0 || quad_1d <= max_quad_1d);
  }
};

namespace internal {

/// Detect an optional compact representation contract; custom operations retain
/// the original dense PrepareQData interface.
/// @tparam T Coefficient/geometry operation.
template <class T> constexpr BlfPAStorage PreferredPAStorage = [] {
  if constexpr (requires { T::PAStorage; }) { return T::PAStorage; }
  else { return BlfPAStorage::Dense; }
}();

/// A compact symmetric tensor is an explicit operation contract. Custom dense
/// assemblers cannot infer symmetry merely from matching field descriptors.
template <class T> constexpr bool HasSymmetricEA = [] {
  using Traits = BlfIntegratorTraits<T>;
  constexpr auto storage = PreferredPAStorage<T>;
  return Traits::TrialBasisOperator == Traits::TestBasisOperator &&
      Traits::trial_type == Traits::test_type &&
      Traits::TrialVDim == Traits::TestVDim &&
      Traits::TrialOutOfPlane == Traits::TestOutOfPlane &&
      (storage == BlfPAStorage::Symmetric || storage == BlfPAStorage::Diagonal ||
       storage == BlfPAStorage::UniformDiagonal ||
       storage == BlfPAStorage::SymmetricKelvin ||
       storage == BlfPAStorage::IsotropicSymmetric);
}();

/// Host setup result type passed by value into CPU/GPU quadrature kernels.
/// @tparam T Operation implementing PrepareQData(trial, test, rule, use_device).
template <class T>
using PreparedBlfQData = decltype(std::declval<T &>().PrepareQData(
    std::declval<const mfem::FiniteElementSpace &>(),
    std::declval<const mfem::FiniteElementSpace &>(),
    std::declval<const mfem::IntegrationRule &>(), bool{}));

/// Device assembly contract: setup returns a trivially copyable view whose
/// AssembleQData(element, point, qdata) fills the whole reference tensor,
/// including geometry measure but excluding the integration point's weight.
/// @tparam T Operation with PrepareQData() and a device-callable setup result.
template <class T>
concept BlfQDataAssembler =
    requires(T &integrator, const mfem::FiniteElementSpace &trial_space,
             const mfem::FiniteElementSpace &test_space,
             const mfem::IntegrationRule &integration_rule, bool use_device) {
      integrator.PrepareQData(trial_space, test_space, integration_rule,
                              use_device);
    } &&
    std::is_trivially_copyable_v<PreparedBlfQData<T>> &&
    requires(const PreparedBlfQData<T> &prepared, int element,
             int quadrature_point,
             BlfQDataMatrix<BlfIntegratorTraits<T>::TestComponents,
                              BlfIntegratorTraits<T>::TrialComponents>
                 qdata) {
      {
        prepared.AssembleQData(element, quadrature_point, qdata)
      } -> std::same_as<void>;
    };

/// Host single-element contract; caller sets the transformation's current point
/// and provides a pre-sized reference tensor. Exclude the point's weight.
/// @tparam T Operation implementing AssembleQDataHost(transformation, ip, qdata).
template <class T>
concept BlfHostQDataAssembler =
    requires(T &integrator, mfem::ElementTransformation &transformation,
             const mfem::IntegrationPoint &ip, mfem::DenseMatrix &qdata) {
      {
        integrator.AssembleQDataHost(transformation, ip, qdata)
      } -> std::same_as<void>;
    };

template <class Traits>
constexpr bool BlfShareTensorBasis = !Traits::Mixed;

template <class Traits>
constexpr bool BlfShareFullBasis =
    !Traits::Mixed && Traits::trial_operator == Traits::test_operator &&
    Traits::TrialComponents == Traits::TestComponents;

template <BlfPAStorage Storage, class Traits, int SpaceDimensions>
struct BlfPAAction;
template <class Traits> struct TensorEASizes;
template <class Traits, class Prepared, class Entry,
          int FixedD1D = 0, int FixedQ1D = 0,
          int FixedTestD1D = FixedD1D> struct TensorEAWorkFunctor;
template <class Traits> struct OperatorTensorEASizes;
template <class Traits, class Prepared, class Entry>
struct OperatorTensorEAWorkFunctor;
template <class Traits, class Output,
          int FixedTrialDofs = 0, int FixedTestDofs = 0, int FixedNQ = 0>
struct FullEAEntryFunctor;
template <class Traits, class Prepared, class Output,
          int FixedTrialDofs = 0, int FixedTestDofs = 0, int FixedNQ = 0>
struct FullEAWorkFunctor;
struct FullApplySingleFunctor;
template <class Traits, bool Transpose, bool Scalar, int FixedNQ>
struct FullApplySizes;
template <class Traits, bool Transpose> struct TensorOperatorSizes;
template <class Traits> struct TensorApplySizes;
template <class Traits, bool Transpose, int FixedInputDofs,
          int FixedOutputDofs, int FixedNQ>
struct FullScalarApplyFunctor;
template <class Traits, bool Transpose, BlfPAStorage Storage,
          int SpaceDimensions, int FixedInputDofs, int FixedOutputDofs, int FixedNQ>
struct FullApplyFunctor;
template <class Traits, bool Transpose, bool Direct, int FixedD1DIn,
          int FixedD1DOut, int FixedQ1D, BlfPAStorage Storage, int SpaceDimensions>
struct TensorOperatorApplyFunctor;
template <class Traits, bool Direct, int FixedD1DIn, int FixedD1DOut, int FixedQ1D,
          int QDataVDim>
struct TensorApplyFunctor;

}

/// Device-accessible, full multidimensional basis table.  This is analogous to
/// DofToQuad::FULL, but its component count follows BlfIntegratorTraits and so
/// also covers MFEM's three-component R1D/R2D finite elements.  Values use the
/// q + nq * (component + components * dof) layout.
/// Component-space DOFs are component-major (scalar DOF + scalar_dofs * field).
/// @tparam Operator Reference basis operation; use Gradient for component-space
/// physical gradient/curl/divergence, as selected by ReferenceFieldOperator().
/// @tparam Map MFEM finite-element map determining scalar versus vector shapes.
/// @tparam Components Total reference output components, including VDim copies.
/// @tparam VDim Scalar-space component count; must be 1 for native vector FEs.
template <BlfFieldOperator Operator, mfem::FiniteElement::MapType Map,
          int Components, int VDim = 1>
class BlfReferenceBasis {
  int dofs_ = 0;
  int quadrature_points_ = 0;
  mfem::Vector values_;

public:
  /// Construct an empty table; call Reset() before reading entries.
  BlfReferenceBasis() = default;

  /// @param fe Finite element providing reference shapes.
  /// @param integration_rule Points at which to tabulate shapes.
  /// @param ordering Local DOF ordering, including signed lexicographic maps.
  BlfReferenceBasis(
      const mfem::FiniteElement &fe,
      const mfem::IntegrationRule &integration_rule,
      mfem::ElementDofOrdering ordering = mfem::ElementDofOrdering::NATIVE) {
    Reset(fe, integration_rule, ordering);
  }

  /// Replace the table, evaluating reference values/derivatives without geometry.
  /// @param fe Finite element providing reference shapes.
  /// @param integration_rule Points at which to tabulate shapes.
  /// @param ordering Local DOF ordering applied within each field component.
  void
  Reset(const mfem::FiniteElement &fe,
        const mfem::IntegrationRule &integration_rule,
        mfem::ElementDofOrdering ordering = mfem::ElementDofOrdering::NATIVE);

  /// @return Element DOFs including scalar-space VDim copies.
  int Dofs() const { return dofs_; }
  /// @return Number of tabulated quadrature points.
  int QuadraturePoints() const { return quadrature_points_; }
  /// @return Total reference operator-output components.
  constexpr static int NumComponents() { return Components; }

  /// @param on_device Select a device-accessible pointer (true) or host pointer.
  /// @return Borrowed table pointer, invalidated by Reset() or destruction.
  const mfem::real_t *Read(bool on_device = true) const {
    return values_.Read(on_device);
  }

  /// @param q Zero-based quadrature index.
  /// @param component Zero-based reference-output component.
  /// @param dof Zero-based ordered local DOF, including field copies.
  /// @param quadrature_points Number of points in the table.
  /// @return Flat index; arguments are not bounds-checked.
  MFEM_HOST_DEVICE static int Index(int q, int component, int dof,
                                    int quadrature_points) {
    return q + quadrature_points * (component + Components * dof);
  }

  /// @tparam Pointer Pointer or indexable buffer of scalar basis values.
  /// @param values Table in Index() layout.
  /// @param q Zero-based quadrature index.
  /// @param component Zero-based reference-output component.
  /// @param dof Zero-based ordered local DOF.
  /// @param quadrature_points Number of points in the table.
  /// @return Basis entry without bounds checking.
  template <class Pointer>
  MFEM_HOST_DEVICE static mfem::real_t Eval(Pointer values, int q,
                                            int component, int dof,
                                            int quadrature_points) {
    return values[Index(q, component, dof, quadrature_points)];
  }
};

/// Element- and partial-assembly integrator for a compile-time bilinear-form
/// descriptor, implementing MFEM's BilinearFormIntegrator API. T must implement
///
///   PreparedQData PrepareQData(const FiniteElementSpace &trial_space,
///                              const FiniteElementSpace &test_space,
///                              const IntegrationRule &, bool use_device);
///
/// PreparedQData must be trivially copyable and provide
///
///   MFEM_HOST_DEVICE void AssembleQData(int element, int quadrature_point,
///                                       BlfQDataMatrix<...> qdata) const;
///
/// PrepareQData is the host-side setup stage for coefficient projection and
/// requested GeometricFactors.  AssembleQData completely fills the
/// TestComponents by TrialComponents view with the coefficient and geometry
/// pullback, but not IntegrationPoint::weight.  It is invoked inside the EA
/// CPU/GPU kernel.  Integrators should request only the geometric factors they
/// use, and should use CoefficientVector with COMPRESSED storage when possible.
/// @tparam T Operation supplying BlfIntegratorTraits metadata and the device
/// quadrature setup contract above. Single-element host methods additionally
/// require AssembleQDataHost(); no ownership of T is transferred.
/// All basis operators are evaluated in reference space, including for
/// embedded elements with rectangular Jacobians.
///
/// Tensor-product scalar value/gradient (including Cartesian component fields)
/// and vector H(curl)/H(div) value/differential elements are detected
/// automatically for batched EA/PA and assembled with
/// dimension-by-dimension sum factorization when their 1D sizes do not exceed
/// the optional runtime BlfTensorProductLimits. A zero maximum is unbounded and
/// therefore still selects sum factorization. The runtime workspace size
/// determines whether that kernel uses shared or global device memory.
/// BlfTensorProductMode::Force selects tensor kernels even above explicit maxima;
/// PA uses registered fixed-size kernels when available and runtime sizes
/// otherwise. Non-tensor and over-limit Auto cases use the generic full-basis
/// path. Single-element host assembly uses full reference shapes regardless of
/// the tensor selection mode.
///
/// The optional legacy single-element entry points additionally require
/// AssembleQDataHost(ElementTransformation &, const IntegrationPoint &,
/// DenseMatrix &).
/// Partial assembly for bilinear forms described by
/// BlfIntegratorTraits. Scalar value/gradient and H(curl)/H(div) value or
/// differential operators, including mixed scalar/vector forms, use sum
/// factorization on complete tensor-product spaces; all other forms use full
/// multidimensional reference-basis tables. On GPU tensor-product spaces, the
/// order-one H(curl) curl-curl and H(div) div-div self forms use a cached full
/// basis when it fits the small-basis limit, since this avoids synchronization
/// between their separable derivative terms. Both paths store only quadrature
/// data.
/// Component H1/L2 vector fields also use sum factorization on compatible tensor
/// spaces. Component curl/divergence use reference-gradient channels combined
/// through quadrature data. L2 differential operations are broken elementwise
/// operations, without face contributions.
/// Mixed operations always stage separate trial/test basis tables, including
/// when given the same space. Non-mixed operations require the same space and
/// share tensor maps; evaluated full bases also need matching field operations.
/// Registered kernels derive workspace and batching extents at compile time;
/// runtime-sized kernels use the same constexpr extent helpers.
/// PA always selects the smaller preferred representation at compile time for
/// each Cartesian space dimension; no runtime representation policy exists.
/// T and the rule must outlive this integrator; reassemble PA after geometry
/// or coefficient changes. EA calls do not invalidate the assembled PA cache.
template <class T>
class DomainBilinearFormIntegrator : public mfem::BilinearFormIntegrator {
public:
  using blf_traits = internal::BlfIntegratorTraits<T>;
  static_assert(internal::BlfQDataAssembler<T>,
                "T must provide PrepareQData(...) returning a trivially "
                "copyable object with MFEM_HOST_DEVICE AssembleQData(...)");

  /// Uniform scale used only by AddMultPA and AddMultTransposePA.
  mfem::real_t multiplier = 1.0;

private:
  using TrialBasis =
      BlfReferenceBasis<blf_traits::TrialBasisOperator, blf_traits::trial_type,
                        blf_traits::TrialComponents, blf_traits::TrialVDim>;
  using TestBasis =
      BlfReferenceBasis<blf_traits::TestBasisOperator, blf_traits::test_type,
                        blf_traits::TestComponents, blf_traits::TestVDim>;

  T &integrator_;
  const mfem::IntegrationRule &integration_rule_;
  BlfTensorProductMode tensor_product_mode_;
  BlfTensorProductLimits tensor_product_limits_;

  /// @param fe Finite element whose reference dimension and map are checked.
  /// @param expected_map Required MFEM map on this side of the form.
  /// @param which Trial/test label included in MFEM_VERIFY diagnostics.
  static void VerifyElement(const mfem::FiniteElement &fe,
                            mfem::FiniteElement::MapType expected_map,
                            const char *which) {
    MFEM_VERIFY(fe.GetDim() == blf_traits::Dims,
                which << " finite element has the wrong reference dimension");
    MFEM_VERIFY(fe.GetMapType() == expected_map,
                which << " finite element has the wrong MapType");
  }

  /// @param trial_fe Trial finite element.
  /// @param test_fe Test finite element.
  /// @param rule Common quadrature rule.
  /// @return Tensor compatibility for both reference field layouts.
  static bool SupportsTensorProduct(const mfem::FiniteElement &trial_fe,
                                    const mfem::FiniteElement &test_fe,
                                    const mfem::IntegrationRule &rule) {
    return internal::SupportsTensorProduct<
        blf_traits::Dims,
        internal::ReferenceFieldOperator(blf_traits::trial_operator,
            blf_traits::trial_type, blf_traits::TrialVDim,
            blf_traits::TrialOutOfPlane), blf_traits::trial_type,
        internal::ReferenceFieldOperator(blf_traits::test_operator,
            blf_traits::test_type, blf_traits::TestVDim,
            blf_traits::TestOutOfPlane), blf_traits::test_type>(trial_fe, test_fe,
                                                          rule);
  }

  /// @param trial_fe Trial tensor finite element.
  /// @param test_fe Test tensor finite element.
  /// @param integration_rule Compatible tensor quadrature rule.
  /// @param limits Auto-mode bounds to check against closed-basis sizes.
  static bool
  WithinTensorOptimizationLimits(const mfem::FiniteElement &trial_fe,
                                 const mfem::FiniteElement &test_fe,
                                 const mfem::IntegrationRule &integration_rule,
                                 const BlfTensorProductLimits &limits) {
    const auto &trial_map =
        trial_fe.GetDofToQuad(integration_rule, mfem::DofToQuad::TENSOR);
    const auto &test_map =
        test_fe.GetDofToQuad(integration_rule, mfem::DofToQuad::TENSOR);
    return limits.Contains(trial_map.ndof, test_map.ndof, trial_map.nqpt);
  }

  static constexpr bool UsesOperatorTensorEA = blf_traits::HasComponentFields ||
      internal::IsVectorTensorMap(blf_traits::trial_type) ||
      internal::IsVectorTensorMap(blf_traits::test_type);
  static constexpr bool UsesScalarMassTensorEA = !UsesOperatorTensorEA &&
      blf_traits::TrialComponents == 1 && blf_traits::TestComponents == 1 &&
      blf_traits::trial_operator == BlfFieldOperator::Value &&
      blf_traits::test_operator == BlfFieldOperator::Value;

  using TensorEASizes = internal::TensorEASizes<blf_traits>;

  /// Launch scalar tensor contractions using the caller's output entry policy.
  template <class Entry>
  static void LaunchScalarTensorEA(T &integrator,
      const mfem::FiniteElementSpace &trial_space,
      const mfem::FiniteElementSpace &test_space,
      const mfem::IntegrationRule &integration_rule, Entry add_entry);

  using OperatorTensorEASizes = internal::OperatorTensorEASizes<blf_traits>;

  template <class Prepared, class Entry>
  using OperatorTensorEAWorkFunctor =
      internal::OperatorTensorEAWorkFunctor<blf_traits, Prepared, Entry>;

  /// Launch vector/component tensor contractions with the same output policies.
  template <class Entry>
  static void LaunchOperatorTensorEA(T &integrator,
      const mfem::FiniteElementSpace &trial_space,
      const mfem::FiniteElementSpace &test_space,
      const mfem::IntegrationRule &integration_rule, Entry add_entry);

  template <class Entry>
  static void LaunchTensorEA(T &integrator,
      const mfem::FiniteElementSpace &trial_space,
      const mfem::FiniteElementSpace &test_space,
      const mfem::IntegrationRule &rule, Entry entry) {
    if constexpr (UsesOperatorTensorEA) {
      LaunchOperatorTensorEA(integrator, trial_space, test_space, rule, entry);
    } else {
      LaunchScalarTensorEA(integrator, trial_space, test_space, rule, entry);
    }
  }

public:
  /// Construct an element/partial-assembly integrator. The operation object and
  /// integration rule must outlive this object.
  /// @param integrator Borrowed operation supplying coefficient/geometry data.
  /// @param integration_rule Borrowed common element quadrature rule.
  /// @param tensor_product_mode Tensor selection policy; Force rejects
  /// unsupported inputs.
  /// @param tensor_product_limits Auto-mode bounds; ignored by Force/Disable.
  /// @param multiplier Uniform scale used only by AddMultPA and AddMultTransposePA.
  DomainBilinearFormIntegrator(
      T &integrator, const mfem::IntegrationRule &integration_rule,
      BlfTensorProductMode tensor_product_mode = BlfTensorProductMode::Auto,
      BlfTensorProductLimits tensor_product_limits = {},
      mfem::real_t multiplier = 1.0)
      : multiplier(multiplier), integrator_(integrator),
        integration_rule_(integration_rule),
        tensor_product_mode_(tensor_product_mode),
        tensor_product_limits_(tensor_product_limits) {
    tensor_product_limits_.Verify();
  }

  /// Assemble a self-space matrix; requires T::Mixed == false and host qdata.
  /// @param fe Finite element used on both sides.
  /// @param transformation Geometry of the element being integrated.
  /// @param[out] element_matrix Resized/overwritten local matrix in native,
  /// component-major DOF order (test rows, trial columns).
  void AssembleElementMatrix(const mfem::FiniteElement &fe,
                             mfem::ElementTransformation &transformation,
                             mfem::DenseMatrix &element_matrix) override {
    MFEM_VERIFY(!blf_traits::Mixed,
                "the single-space overload cannot assemble a mixed form");
    AssembleElementMatrix2(fe, fe, transformation, element_matrix);
  }

  /// Assemble a rectangular local matrix using T's host qdata interface.
  /// @param trial_fe Trial finite element.
  /// @param test_fe Test finite element on the same element geometry.
  /// @param transformation Geometry of the element being integrated.
  /// @param[out] element_matrix Resized/overwritten matrix in native,
  /// component-major DOF order (test rows, trial columns).
  void AssembleElementMatrix2(const mfem::FiniteElement &trial_fe,
                              const mfem::FiniteElement &test_fe,
                              mfem::ElementTransformation &transformation,
                              mfem::DenseMatrix &element_matrix) override {
    if constexpr (requires {
                    integrator_.AssembleQDataHost(
                        transformation, integration_rule_.IntPoint(0),
                        element_matrix);
                  }) {
      AssembleElementMatrix2Impl(trial_fe, test_fe, transformation, element_matrix);
    } else {
      MFEM_ABORT("single-element assembly requires AssembleQDataHost; "
                 "use AssembleEA or AssemblePA for device-only operations");
    }
  }

  /// Assemble all element matrices in MFEM's element-assembly layout.  MFEM
  /// stores these matrices transposed (trial dof, test dof, element) so that
  /// its action kernels read contiguous trial entries for each test dof.
  /// Elements must have uniform polynomial order and geometry; this batch API
  /// assumes homogeneity and does not validate individual element layouts.
  /// @param space Trial and test space; its vdim must match both traits.
  /// @param[in,out] emat Flattened MFEM EA matrices, including vdim copies.
  /// @param add Accumulate into correctly sized emat when true; otherwise resize
  /// and overwrite.
  void AssembleEA(const mfem::FiniteElementSpace &space, mfem::Vector &emat,
                  const bool add = true) override {
    AssembleEAImpl(space, space, emat, add, true);
  }

  /// Rectangular element assembly used by MixedBilinearForm.
  /// Each space must have uniform element order/geometry; no per-element
  /// homogeneity checks are performed.
  /// @param trial_space Trial space on the common homogeneous mesh.
  /// @param test_space Test space on the same mesh with matching element count.
  /// @param[in,out] emat Flattened (trial DOF, test DOF, element) EA matrices.
  /// @param add Accumulate into correctly sized emat, or resize/overwrite when
  /// false. Accumulation requires an already correctly sized output.
  void AssembleEA(const mfem::FiniteElementSpace &trial_space,
                  const mfem::FiniteElementSpace &test_space,
                  mfem::Vector &emat, const bool add = true) override {
    AssembleEAImpl(trial_space, test_space, emat, add, false);
  }

private:
  void AssembleElementMatrix2Impl(const mfem::FiniteElement &trial_fe,
      const mfem::FiniteElement &test_fe,
      mfem::ElementTransformation &transformation, mfem::DenseMatrix &element_matrix);

  template <int TrialD1D, int TestD1D, int Q1D>
  static void AssembleFusedTensorEA(
      T &integrator, const mfem::FiniteElementSpace &trial_space,
      const mfem::FiniteElementSpace &test_space,
      const mfem::IntegrationRule &rule, mfem::Vector &emat,
      bool add, bool same_space_interface);

  template <int TrialDofs, int TestDofs, int NQ>
  static void AssembleFusedFullEA(
      T &integrator, const mfem::FiniteElementSpace &trial_space,
      const mfem::FiniteElementSpace &test_space,
      const mfem::IntegrationRule &rule, mfem::Vector &emat,
      bool add, bool same_space_interface);

  void AssembleEAImpl(const mfem::FiniteElementSpace &trial_space,
                      const mfem::FiniteElementSpace &test_space,
                      mfem::Vector &emat, bool add, bool same_space_interface);

public:
  /// Scalar value/value tensor EA: weighted qdata and matrix contraction in
  /// one kernel writing directly to the MFEM EA vector. Keys are trial/test
  /// DOFs per axis and quadrature points per axis. Unregistered sizes use the
  /// runtime kernel; e.g. AssembleEATensorKernels::Specialization<3,3,4>::Add().
  using AssembleEATensorKernelType = void (*)(
      T &, const mfem::FiniteElementSpace &, const mfem::FiniteElementSpace &,
      const mfem::IntegrationRule &, mfem::Vector &, bool, bool);
  MFEM_REGISTER_KERNELS(AssembleEATensorKernels, AssembleEATensorKernelType,
                        (int, int, int));

  /// Full-basis EA keyed by total trial/test DOFs (including vdim) and total
  /// quadrature points per element. Unregistered sizes use the fused runtime
  /// kernel; e.g. AssembleEAFullKernels::Specialization<10,10,14>::Add().
  using AssembleEAFullKernelType = AssembleEATensorKernelType;
  MFEM_REGISTER_KERNELS(AssembleEAFullKernels, AssembleEAFullKernelType,
                        (int, int, int));


public:
  /// Registering a specialization instantiates a tensor kernel with fixed
  /// one-dimensional sizes and adds it to the dispatch table for this exact
  /// operator type. Registration must happen before AssemblePA/AddMultPA.
  /// Unregistered layouts continue to use the runtime-sized fallback.
  ///
  /// Forward tensor kernels use
  ///
  ///   ApplyTensorKernels::Specialization<D1DIn,D1DOut,Q1D>::Add();
  ///
  /// and transpose kernels swap the input/output sizes:
  ///
  ///   ApplyTensorTransposeKernels::Specialization<D1DOut,D1DIn,Q1D>::Add();
  ///
  /// The arguments are:
  ///
  /// - D1DIn: input basis degrees of freedom in one dimension. The input is
  ///   the trial space for ApplyTensorKernels and the test space for the
  ///   transpose registry.
  /// - D1DOut: output basis degrees of freedom in one dimension. The output is
  ///   the test space for ApplyTensorKernels and the trial space for the
  ///   transpose registry.
  /// - Q1D: quadrature points in one dimension.
  ///
  /// The same registries and arguments cover scalar and vector tensor maps.
  /// For vector maps, D1DIn and D1DOut are the closed-basis sizes. Their
  /// open-basis sizes are inferred from the input and output map types in the
  /// operator traits: H(curl) and H(div) use one fewer open dof, while a scalar
  /// side uses the same size. The reference dimension and the choice between the
  /// scalar and vector implementations are also obtained from the traits. For
  /// example, a non-mixed vector form with six closed dofs and seven
  /// quadrature points registers as
  ///
  ///   ApplyTensorKernels::Specialization<6,6,7>::Add();
  ///
  /// Kernel() is defined out of line so an application may explicitly
  /// specialize it and return a different member-function kernel.
  using ApplyTensorKernelType =
      void (DomainBilinearFormIntegrator::*)(const mfem::Vector &,
                                            mfem::Vector &, mfem::real_t) const;
  /// Forward tensor registry keyed by input/output closed-basis DOFs per axis
  /// and quadrature points per axis; see ApplyTensorKernelType for registration.
  MFEM_REGISTER_KERNELS(ApplyTensorKernels, ApplyTensorKernelType,
                        (int, int, int));
  /// Transpose tensor registry with the same keys and trial/test roles swapped.
  MFEM_REGISTER_KERNELS(ApplyTensorTransposeKernels, ApplyTensorKernelType,
                        (int, int, int));

  /// Full-basis kernels may be specialized by their element input and output
  /// dof counts and quadrature-point count. Unregistered layouts use the
  /// runtime-sized full-basis implementation.
  ///
  ///   ApplyFullKernels::Specialization<InputDofs,OutputDofs,NQ>::Add();
  ///
  /// The transpose registry swaps InputDofs and OutputDofs.
  /// InputDofs and OutputDofs include scalar-space vdim copies; NQ is the total
  /// number of element quadrature points, not the number per axis.
  using ApplyFullKernelType = ApplyTensorKernelType;
  /// Forward full-basis registry keyed by input DOFs, output DOFs and total
  /// quadrature count per element; see ApplyFullKernelType for registration.
  MFEM_REGISTER_KERNELS(ApplyFullKernels, ApplyFullKernelType, (int, int, int));
  /// Transpose full-basis registry with the same keys and swapped spaces.
  MFEM_REGISTER_KERNELS(ApplyFullTransposeKernels, ApplyFullKernelType,
                        (int, int, int));

  using DiagonalKernelType =
      void (DomainBilinearFormIntegrator::*)(const mfem::Vector &,
                                            mfem::Vector &) const;
  /// Tensor diagonal registries use closed trial/test axis sizes and Q1D.
  /// ADAt contracts complete columns inside a block before weighted squaring.
  MFEM_REGISTER_KERNELS(DiagonalTensorKernels, DiagonalKernelType,
                        (int, int, int));
  MFEM_REGISTER_KERNELS(DiagonalADAtTensorKernels, DiagonalKernelType,
                        (int, int, int));
  MFEM_REGISTER_KERNELS(DiagonalFullKernels, DiagonalKernelType,
                        (int, int, int));
  MFEM_REGISTER_KERNELS(DiagonalADAtFullKernels, DiagonalKernelType,
                        (int, int, int));

private:

  enum class Kernel { Unassembled, Tensor, Full };

  template <bool ADAt, int FixedTrialDofs, int FixedTestDofs,
            int FixedNQ>
  void DiagonalFull(const mfem::Vector &diagonal_weights,
                    mfem::Vector &diag) const;

  template <bool ADAt, int FixedD1D = 0, int FixedTestD1D = 0, int FixedQ1D = 0>
  void DiagonalTensor(const mfem::Vector &diagonal_weights, mfem::Vector &diag) const;

  template <bool ADAt, class TrialBasisType, class TestBasisType, class Projection>
  void DiagonalContract(const mfem::Vector &diagonal_weights, mfem::Vector &diag,
                        TrialBasisType trial, TestBasisType test, Projection projection) const;

  constexpr static bool is_hcurl_curl_curl =
      blf_traits::Dims >= 2 &&
      blf_traits::trial_type == mfem::FiniteElement::H_CURL &&
      blf_traits::test_type == mfem::FiniteElement::H_CURL &&
      blf_traits::trial_operator == BlfFieldOperator::Curl &&
      blf_traits::test_operator == BlfFieldOperator::Curl;

  constexpr static bool is_hdiv_div_div =
      blf_traits::Dims >= 2 &&
      blf_traits::trial_type == mfem::FiniteElement::H_DIV &&
      blf_traits::test_type == mfem::FiniteElement::H_DIV &&
      blf_traits::trial_operator == BlfFieldOperator::Divergence &&
      blf_traits::test_operator == BlfFieldOperator::Divergence;

  int space_dimensions_ = blf_traits::Dims;
  Kernel kernel_ = Kernel::Unassembled;
  const mfem::DofToQuad *trial_maps_ = nullptr;
  const mfem::DofToQuad *test_maps_ = nullptr;
  const mfem::DofToQuad *trial_open_maps_ = nullptr;
  const mfem::DofToQuad *test_open_maps_ = nullptr;
  std::unique_ptr<TrialBasis> trial_basis_;
  std::unique_ptr<TestBasis> test_basis_;
public:
  /// Weighted partial-assembly data. Scalar mass and uniformly weighted
  /// Cartesian mass use q + nq * element, matching MassIntegrator::pa_data.
  /// Other forms use the representation reported by PAStorage() and
  /// PAStoredComponents(); their buffers are not scalar mass weights.
  mfem::Vector qdata_;

private:
  mutable mfem::Vector workspace_;
  int elements_ = 0;
  int trial_dofs_ = 0;
  int test_dofs_ = 0;
  int quadrature_points_ = 0;
  bool same_space_ = false;
  bool tensor_uses_full_basis_ = false;

  /// Smaller preferred payload, or dense when compression does not save space.
  /// @tparam SpaceDimensions Cartesian space dimension.
  template <int SpaceDimensions> static constexpr BlfPAStorage SelectedPAStorage = [] {
    constexpr auto preferred = internal::PreferredPAStorage<T>;
    constexpr int components =
        internal::BlfPAAction<preferred, blf_traits, SpaceDimensions>::Components;
    return components < blf_traits::MaxQDataComponents ? preferred
                                                       : BlfPAStorage::Dense;
  }();

  /// Dispatch only the space dimension when the payload depends on it.
  /// @tparam F Callable templated on selected storage and space dimension.
  /// @param action Host assembly/apply/query action; all dimension branches
  /// must return the same type. No runtime PA policy or layout is stored.
  template <class F> decltype(auto) WithPALayout(F &&action) const;

  /// Non-mixed tensor kernels share interpolation and derivative tables.
  static constexpr bool ShareTensorBasis =
      internal::BlfShareTensorBasis<blf_traits>;

  template <bool Transpose, bool Scalar, int FixedNQ>
  using FullApplySizes =
      internal::FullApplySizes<blf_traits, Transpose, Scalar, FixedNQ>;

  template <bool Transpose>
  using TensorOperatorSizes =
      internal::TensorOperatorSizes<blf_traits, Transpose>;

  using TensorApplySizes = internal::TensorApplySizes<blf_traits>;
  /// @return Whether cached qdata uses quadrature-fastest, component-major
  /// storage. Non-dense payloads always do; dense payloads do only for tensor
  /// selection without component fields, including the small-basis GPU path.
  /// Dense component-field and other full-basis payloads are point-major.
  bool UsesComponentMajorQData() const {
    if constexpr (blf_traits::HasComponentFields) {
      return PAStorage() != BlfPAStorage::Dense;
    }
    return kernel_ == Kernel::Tensor || PAStorage() != BlfPAStorage::Dense;
  }

  /// @param trial_map Trial closed-basis tensor map.
  /// @param test_map Test closed-basis tensor map.
  /// @return Whether the configured Auto-mode limits admit these sizes.
  bool WithinTensorOptimizationLimits(const mfem::DofToQuad &trial_map,
                                      const mfem::DofToQuad &test_map) const {
    return tensor_product_limits_.Contains(trial_map.ndof, test_map.ndof,
                                           trial_map.nqpt);
  }

  /// Launch a PA block kernel with shared or global per-block scratch storage.
  /// @tparam F Device callable with (begin, end, workspace) signature.
  /// @param kernel Kernel holding non-owning input/output views.
  /// @param threads Requested threads per block.
  /// @param workspace_bytes Scratch byte count per block.
  /// @param block_limit Maximum concurrently assigned element blocks.
  template <class F>
  void Launch(F &kernel, int threads, std::size_t workspace_bytes,
              int block_limit, int elements = 0) const;

  using FullApplySingleFunctor = internal::FullApplySingleFunctor;

  template <bool Transpose, int FixedInputDofs = 0,
            int FixedOutputDofs = 0, int FixedNQ = 0>
  using FullScalarApplyFunctor = internal::FullScalarApplyFunctor<
      blf_traits, Transpose, FixedInputDofs, FixedOutputDofs, FixedNQ>;

  template <bool Transpose, BlfPAStorage Storage = BlfPAStorage::Dense,
            int SpaceDimensions = blf_traits::Dims, int FixedInputDofs = 0,
            int FixedOutputDofs = 0, int FixedNQ = 0>
  using FullApplyFunctor = internal::FullApplyFunctor<
      blf_traits, Transpose, Storage, SpaceDimensions, FixedInputDofs,
      FixedOutputDofs, FixedNQ>;

  /// Dispatch full-basis PA application, adding to an element output vector.
  /// @tparam Transpose Swap trial/test and transpose the quadrature action.
  /// @tparam FixedInputDofs Input element DOFs, or zero for runtime sizing.
  /// @tparam FixedOutputDofs Output element DOFs, or zero for runtime sizing.
  /// @tparam FixedNQ Element quadrature count, or zero for runtime sizing.
  /// @tparam Storage Inlined coefficient action layout.
  /// @tparam SpaceDimensions Cartesian space dimension for factored physical actions.
  /// @param x Input E-vector.
  /// @param[in,out] y Output E-vector to accumulate into.
  template <bool Transpose, int FixedInputDofs = 0, int FixedOutputDofs = 0,
            int FixedNQ = 0, BlfPAStorage Storage = BlfPAStorage::Dense,
            int SpaceDimensions = blf_traits::Dims>
  void ApplyFullImpl(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const;

  /// Dispatch the coefficient layout once, outside all element/point loops.
  /// @tparam Transpose Swap trial/test basis contractions.
  /// @tparam FixedInputDofs Registered input size, or zero.
  /// @tparam FixedOutputDofs Registered output size, or zero.
  /// @tparam FixedNQ Registered quadrature size, or zero.
  /// @param x Input E-vector.
  /// @param[in,out] y Output E-vector.
  template <bool Transpose, int FixedInputDofs = 0, int FixedOutputDofs = 0,
            int FixedNQ = 0>
  void ApplyFull(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const {
    WithPALayout([this, &x, &y, apply_scale]<BlfPAStorage Storage, int SpaceDimensions>() {
      this->template ApplyFullImpl<Transpose, FixedInputDofs, FixedOutputDofs, FixedNQ,
                    Storage, SpaceDimensions>(x, y, apply_scale);
    });
  }

  template <bool Transpose, bool Direct, int FixedD1DIn = 0,
            int FixedD1DOut = 0, int FixedQ1D = 0,
            BlfPAStorage Storage = BlfPAStorage::Dense,
            int SpaceDimensions = blf_traits::Dims>
  using TensorOperatorApplyFunctor = internal::TensorOperatorApplyFunctor<
      blf_traits, Transpose, Direct, FixedD1DIn, FixedD1DOut, FixedQ1D, Storage,
      SpaceDimensions>;

  /// Apply separable scalar/vector differential operators with open/closed maps.
  /// @tparam Transpose Transpose the action and swap trial/test roles.
  /// @tparam FixedD1DIn Input closed-basis size, or zero for runtime sizing.
  /// @tparam FixedD1DOut Output closed-basis size, or zero for runtime sizing.
  /// @tparam FixedQ1D Quadrature points per axis, or zero for runtime sizing.
  /// @tparam Storage Inlined dense, packed, or factored coefficient action.
  /// @param x Input E-vector.
  /// @param[in,out] y Output E-vector to accumulate into.
  template <bool Transpose, int FixedD1DIn = 0, int FixedD1DOut = 0,
            int FixedQ1D = 0, BlfPAStorage Storage = BlfPAStorage::Dense,
            int SpaceDimensions = blf_traits::Dims>
  void ApplyVectorTensorImpl(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const;

  /// Select the coefficient action while sharing the separable basis kernel.
  /// @tparam Transpose Swap trial/test roles.
  /// @tparam FixedD1DIn Registered input size, or zero.
  /// @tparam FixedD1DOut Registered output size, or zero.
  /// @tparam FixedQ1D Registered quadrature size, or zero.
  /// @param x Input E-vector.
  /// @param[in,out] y Output E-vector.
  template <bool Transpose, int FixedD1DIn = 0, int FixedD1DOut = 0,
            int FixedQ1D = 0>
  void ApplyVectorTensorGeneric(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const {
    WithPALayout([this, &x, &y, apply_scale]<BlfPAStorage Storage, int SpaceDimensions>() {
      this->template ApplyVectorTensorImpl<Transpose, FixedD1DIn, FixedD1DOut, FixedQ1D,
                            Storage, SpaceDimensions>(x, y, apply_scale);
    });
  }

  template <bool Direct, int FixedD1DIn = 0, int FixedD1DOut = 0,
            int FixedQ1D = 0>
  using TensorApplyFunctor = internal::TensorApplyFunctor<
      blf_traits, Direct, FixedD1DIn, FixedD1DOut, FixedQ1D,
      SelectedPAStorage<blf_traits::Dims> == BlfPAStorage::UniformDiagonal
          ? blf_traits::TrialVDim : 1>;

  /// Apply scalar value/value tensor PA, adding to the output E-vector.
  /// @tparam Transpose Transpose the action and swap trial/test roles.
  /// @tparam FixedD1DIn Input basis DOFs per axis, or zero for runtime sizing.
  /// @tparam FixedD1DOut Output basis DOFs per axis, or zero for runtime sizing.
  /// @tparam FixedQ1D Quadrature points per axis, or zero for runtime sizing.
  /// @param x Input E-vector.
  /// @param[in,out] y Output E-vector to accumulate into.
  template <bool Transpose, int FixedD1DIn = 0, int FixedD1DOut = 0,
            int FixedQ1D = 0>
  void ApplyScalarTensorGeneric(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const;

  /// Select the scalar value/value or general scalar/vector tensor PA kernel.
  /// @tparam Transpose Transpose the action and swap trial/test roles.
  /// @tparam FixedD1DIn Input closed-basis size, or zero for runtime sizing.
  /// @tparam FixedD1DOut Output closed-basis size, or zero for runtime sizing.
  /// @tparam FixedQ1D Quadrature points per axis, or zero for runtime sizing.
  /// @param x Input E-vector.
  /// @param[in,out] y Output E-vector to accumulate into.
  template <bool Transpose, int FixedD1DIn, int FixedD1DOut,
            int FixedQ1D>
  void ApplyTensorGeneric(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const;

  /// Assemble the preferred packed/factored payload using a single layout-
  /// independent write loop. Quadrature weights are included exactly once.
  /// @tparam Storage Selected non-dense representation.
  /// @tparam SpaceDimensions Cartesian space dimension determining payload size.
  /// @param trial_space Trial space.
  /// @param test_space Test space on the same mesh.
  /// @param use_device Project and assemble on the configured device.
  template <BlfPAStorage Storage, int SpaceDimensions>
  void AssembleCompactPA(const mfem::FiniteElementSpace &trial_space,
                          const mfem::FiniteElementSpace &test_space,
                          bool use_device);

  /// Dispatch registered tensor PA specializations or the runtime fallback.
  /// @tparam Transpose Use the transpose registry and swap input/output spaces.
  /// @param x Input E-vector.
  /// @param[in,out] y Output E-vector to accumulate into.
  template <bool Transpose>
  void ApplyTensor(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const {
    const auto &input_map = Transpose ? *test_maps_ : *trial_maps_;
    const auto &output_map = Transpose ? *trial_maps_ : *test_maps_;
    if constexpr (Transpose) {
      ApplyTensorTransposeKernels::Run(input_map.ndof, output_map.ndof,
                                       input_map.nqpt, *this, x, y, apply_scale);
    } else {
      ApplyTensorKernels::Run(input_map.ndof, output_map.ndof, input_map.nqpt,
                              *this, x, y, apply_scale);
    }
  }

  /// Validate assembled state/vector sizes and dispatch the cached PA kernel.
  /// @tparam Transpose Apply A^T rather than A.
  /// @param x Input E-vector.
  /// @param[in,out] y Output E-vector to accumulate into.
  template <bool Transpose>
  void Apply(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const;

  /// Rebuild the weighted quadrature cache assuming homogeneous element layouts.
  /// Empty local spaces retain typical-FE metadata and clear all cached data;
  /// applying their assembled operator to empty E-vectors is a no-op.
  /// @param trial_space Trial space on the common mesh.
  /// @param test_space Test space on the common mesh.
  /// @param same_space_interface Whether invoked through MFEM's single-space
  /// overload; selects the corresponding element-restriction ordering.
  void Assemble(const mfem::FiniteElementSpace &trial_space,
                const mfem::FiniteElementSpace &test_space,
                bool same_space_interface);

public:
  /// @return Stored PA entries per quadrature point after assembly.
  int PAStoredComponents() const {
    return WithPALayout([=]<BlfPAStorage Storage, int SpaceDimensions>() {
      return internal::BlfPAAction<Storage, blf_traits, SpaceDimensions>::Components;
    });
  }
  /// @return Total number of stored PA scalar entries.
  int PADataSize() const { return qdata_.Size(); }
  /// @return Selected representation for the operation and space dimension.
  /// Assemble before querying embedded-space layouts.
  BlfPAStorage PAStorage() const {
    return WithPALayout([=]<BlfPAStorage Storage, int SpaceDimensions>() {
      return Storage;
    });
  }

  /// Prepare cached weighted quadrature data for a self-space operator.
  /// @param space Trial/test space; requires T::Mixed == false and matching vdim.
  /// Elements must share one FE layout and quadrature rule; homogeneity is
  /// assumed, with no per-element layout checks.
  void AssemblePA(const mfem::FiniteElementSpace &space) override {
    MFEM_VERIFY(!blf_traits::Mixed,
                "the single-space partial-assembly overload cannot assemble "
                "an operation declared Mixed");
    Assemble(space, space, true);
  }

  /// Prepare cached weighted quadrature data for a rectangular operator.
  /// @param trial_space Homogeneous trial space with matching trial vdim.
  /// @param test_space Homogeneous test space with matching test vdim, on the
  /// same mesh and with the same element count as trial_space. Non-mixed
  /// operations require this to be the same space object as trial_space.
  /// Uniform element order/geometry is assumed in each space, without a scan.
  void AssemblePA(const mfem::FiniteElementSpace &trial_space,
                  const mfem::FiniteElementSpace &test_space) override {
    if constexpr (!blf_traits::Mixed) {
      MFEM_VERIFY(&trial_space == &test_space,
                  "non-mixed partial assembly requires the same trial/test "
                  "space; declare Mixed=true to use separate spaces");
    }
    Assemble(trial_space, test_space, false);
  }

  /// Add A x to y after AssemblePA().
  /// @param x Trial E-vector in the assembled element-restriction ordering.
  /// @param[in,out] y Test E-vector; existing entries are preserved and added to.
  /// Component layouts are component-major locally; global MFEM ordering is
  /// handled by the element restriction, not by this method.
  void AddMultPA(const mfem::Vector &x, mfem::Vector &y) const override {
    Apply<false>(x, y, multiplier);
  }

  /// Add A^T x to y after AssemblePA().
  /// @param x Test E-vector in the assembled element-restriction ordering.
  /// @param[in,out] y Trial E-vector; existing entries are preserved and added to.
  void AddMultTransposePA(const mfem::Vector &x,
                          mfem::Vector &y) const override {
    Apply<true>(x, y, multiplier);
  }

  void AssembleDiagonalPA(mfem::Vector &diag) override;

  void AssembleDiagonalPA_ADAt(const mfem::Vector &diagonal_weights,
                               mfem::Vector &diag) override {
    MFEM_VERIFY(kernel_ != Kernel::Unassembled,
                "AssemblePA must be called before assembling the ADAt diagonal");
    MFEM_VERIFY(diagonal_weights.Size() == elements_ * trial_dofs_ &&
                    diag.Size() == elements_ * test_dofs_,
                "ADAt diagonal assembly requires matching trial/test E-vectors");
    if (!elements_) { return; }
    if (kernel_ == Kernel::Full || tensor_uses_full_basis_) {
      DiagonalADAtFullKernels::Run(trial_dofs_, test_dofs_, quadrature_points_,
                                  *this, diagonal_weights, diag);
    } else {
      DiagonalADAtTensorKernels::Run(trial_maps_->ndof, test_maps_->ndof,
          trial_maps_->nqpt, *this, diagonal_weights, diag);
    }
  }
};
} // namespace mfem

#include "bilinearintegs_impl.hpp"

#endif
