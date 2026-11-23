#ifndef BILININTEG_OPERATOR_HPP
#define BILININTEG_OPERATOR_HPP

/// @file
/// Physical-space bilinear-form operations and quadrature-data views.

#include <mfem.hpp>

#include "formintegs_impl.hpp"
#include "parallel_wrappers.hpp"
#include "bilinearintegs_tensor_coefficients.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <limits>
#include <memory>
#include <type_traits>

namespace mfem {

/// Operation applied to a finite-element field before the pointwise part of a
/// bilinear form.  Value deliberately means both scalar interpolation and
/// vector interpolation; the finite element's MapType distinguishes the two.
enum class BlfFieldOperator { Value, Gradient, Divergence, Curl };

/// Structure of the coefficient in the physical-space bilinear form.
///
/// Vector denotes either multiplication of a scalar by a vector coefficient or
/// contraction of a vector with a vector coefficient.  Which one is meant is
/// determined by the ranks of the trial and test field operators.  CrossProduct
/// is separate because it maps a vector to a vector.  Geometry factors may turn
/// even a Scalar coefficient into a dense rectangular quadrature tensor.
/// For component-gradient/vector pairs, Vector contracts the Cartesian
/// derivative direction, preserving the field component (vector advection).
enum class BlfCoefficient {
  Identity,
  Scalar,
  Vector,
  CrossProduct,
  DiagonalMatrix,
  SymmetricMatrix,
  Matrix,
  /// General fourth-order coefficient on flattened component gradients.
  Tensor,
  /// Fourth-order coefficient with A_ia,jb = A_jb,ia (major symmetry).
  MajorSymmetricTensor,
  /// Explicit field/direction product A_ia,jb = Q_ij K_ab.
  KroneckerTensor,
  /// Coefficient acting on symmetric gradients, without major symmetry.
  MinorSymmetricTensor,
  /// Coefficient on symmetric gradients with both minor and major symmetries.
  MinorMajorSymmetricTensor,
  /// Two-parameter isotropic tensor acting on symmetric gradients.
  IsotropicSymmetricTensor
};

/// Pointwise PA payload layout, selected at compile time for each space
/// dimension. Compact data is used only when it stores fewer entries than dense.
enum class BlfPAStorage {
  Dense, Symmetric, Diagonal, Kronecker, Kelvin, SymmetricKelvin,
  IsotropicSymmetric, UniformDiagonal
};

/// Writable packed reference tensor used by symmetry-aware setup.
/// @tparam Components Number of coincident trial/test reference components.
/// @tparam Diagonal Store only diagonal entries instead of the upper triangle.
template <int Components, bool Diagonal> class BlfPackedQDataMatrix {
  mfem::real_t *data_;

public:
  /// @param data Borrowed writable packed buffer.
  MFEM_HOST_DEVICE explicit BlfPackedQDataMatrix(mfem::real_t *data)
      : data_(data) {}
  /// @param test Reference row.
  /// @param trial Reference column.
  /// @return Whether this pair belongs to the stored triangle/diagonal.
  MFEM_HOST_DEVICE static bool StoresEntry(int test, int trial) {
    return Diagonal ? test == trial : test <= trial;
  }
  /// @param test Stored reference row.
  /// @param trial Stored reference column.
  /// @return Writable entry, valid only when StoresEntry() is true.
  MFEM_HOST_DEVICE mfem::real_t &operator()(int test, int trial) const {
    return data_[Diagonal ? test : test + trial * (trial + 1) / 2];
  }
};

/// Compile-time description of
///
///   (test operator v)^T coefficient-action (trial operator u).
///
/// Together with the trial/test MapTypes this spans the supported non-trace
/// forms. The coefficient and geometric pullback are assembled by the
/// integrator T into a (possibly rectangular) quadrature tensor; this
/// descriptor tells the common assembly kernels which basis operations
/// surround that tensor.
/// @tparam TrialOperator Physical operation applied to the trial field.
/// @tparam Coefficient Structure of the physical pointwise coefficient action.
/// @tparam TestOperator Physical operation applied to the test field.
template <BlfFieldOperator TrialOperator, BlfCoefficient Coefficient,
          BlfFieldOperator TestOperator>
struct BlfOperation {
  constexpr static auto trial_operator = TrialOperator;
  constexpr static auto coefficient = Coefficient;
  constexpr static auto test_operator = TestOperator;
};

/// A non-owning, device-accessible view of one quadrature-point tensor.
/// Entries are row-major: trial + TrialComponents * test.
/// @tparam TestComponents Number of reference test-output components (rows).
/// @tparam TrialComponents Number of reference trial-output components (columns).
template <int TestComponents, int TrialComponents> class BlfQDataMatrix {
  mfem::real_t *data_;

public:
  /// @param data Writable buffer of TestComponents * TrialComponents entries;
  /// the view does not extend its lifetime.
  MFEM_HOST_DEVICE explicit BlfQDataMatrix(mfem::real_t *data) : data_(data) {}

  /// @return True for every pair in this dense representation.
  MFEM_HOST_DEVICE static bool StoresEntry(int, int) { return true; }

  /// @param test Zero-based reference test-output index.
  /// @param trial Zero-based reference trial-output index.
  /// @return Writable tensor entry; indices are not bounds-checked.
  MFEM_HOST_DEVICE mfem::real_t &operator()(int test, int trial) const {
    return data_[trial + TrialComponents * test];
  }
};

/// Owns coefficient values projected to a homogeneous element quadrature
/// space. Constant and symmetric coefficients use MFEM's compressed storage.
class BlfCoefficientStorage {
public:
  /// Trivially copyable borrowed view; invalid after storage destruction or
  /// another Prepare(). Components are contiguous within each quadrature point.
  struct DeviceView {
    const mfem::real_t *values;
    int quadrature_points;
    int components;
    bool constant;

    /// @param component Zero-based stored coefficient component (matrix entries
    /// follow the selected MFEM CoefficientStorage convention).
    /// @param element Zero-based mesh element index.
    /// @param quadrature_point Zero-based rule index; ignored for constants.
    /// @return Projected coefficient entry; indices are not bounds-checked.
    MFEM_HOST_DEVICE mfem::real_t operator()(int component, int element,
                                             int quadrature_point) const {
      const int point =
          constant ? 0 : quadrature_point + quadrature_points * element;
      return values[component + components * point];
    }
  };

private:
  std::unique_ptr<mfem::QuadratureSpace> quadrature_space_;
  std::unique_ptr<mfem::CoefficientVector> values_;

public:
  /// Replace owned quadrature storage by a projection of the supplied coefficient.
  /// @tparam Coefficient MFEM scalar, vector, or matrix coefficient type.
  /// @param coefficient Coefficient evaluated during projection.
  /// @param space Space whose mesh defines the quadrature space.
  /// @param integration_rule Common rule for the homogeneous element batch.
  /// @param use_device Select device-accessible versus host read pointers.
  /// @param storage MFEM coefficient compression policy.
  /// @return Non-owning view into the new storage.
  template <class Coefficient>
  DeviceView Prepare(
      Coefficient &coefficient, const mfem::FiniteElementSpace &space,
      const mfem::IntegrationRule &integration_rule, bool use_device,
      mfem::CoefficientStorage storage = mfem::CoefficientStorage::COMPRESSED) {
    values_.reset();
    quadrature_space_ = std::make_unique<mfem::QuadratureSpace>(
        *space.GetMesh(), integration_rule);
    values_ = std::make_unique<mfem::CoefficientVector>(
        coefficient, *quadrature_space_, storage);
    const int components = values_->GetVDim();
    return {values_->Read(use_device), integration_rule.GetNPoints(),
            components, values_->Size() == components};
  }

  /// @return Stored scalar-entry count, or zero before Prepare().
  int Size() const { return values_ ? values_->Size() : 0; }
};

/// Lazily requests exactly the geometric factors selected by Flags and exposes
/// their MFEM layouts through a trivially copyable device view. For embedded
/// meshes, Determinant() is sqrt(det(J^T J)), i.e. MFEM's detJ/Weight().
/// @tparam Flags Bitwise MFEM GeometricFactors flags for coordinates, Jacobians,
/// and/or determinants; unrequested view pointers are null.
/// @tparam Dims Mesh topological dimension, between 1 and 3.
template <int Flags, int Dims> class BlfGeometryFactors {
public:
  /// Borrowed mesh-owned geometry arrays in MFEM's quadrature-fastest layout.
  /// The mesh and its geometric-factor cache must remain valid while in use.
  struct DeviceView {
    const mfem::real_t *coordinates;
    const mfem::real_t *jacobians;
    const mfem::real_t *determinants;
    int quadrature_points;
    int space_dimensions;

    /// @param element Zero-based mesh element index.
    /// @param quadrature_point Zero-based prepared rule index.
    /// @param component Cartesian coordinate index; requires COORDINATES in Flags.
    MFEM_HOST_DEVICE mfem::real_t Coordinate(int element, int quadrature_point,
                                             int component) const {
      static_assert(Flags & mfem::GeometricFactors::COORDINATES,
                    "coordinates were not requested");
      return coordinates[quadrature_point +
                         quadrature_points *
                             (component + space_dimensions * element)];
    }

    /// @param element Zero-based mesh element index.
    /// @param quadrature_point Zero-based prepared rule index.
    /// @param row Cartesian space index in [0, space_dimensions).
    /// @param column Reference direction in [0, Dims); requires JACOBIANS.
    MFEM_HOST_DEVICE mfem::real_t Jacobian(int element, int quadrature_point,
                                           int row, int column) const {
      static_assert(Flags & mfem::GeometricFactors::JACOBIANS,
                    "Jacobians were not requested");
      return jacobians[quadrature_point +
                       quadrature_points *
                           (row +
                            space_dimensions * (column + Dims * element))];
    }

    /// @param element Zero-based mesh element index.
    /// @param quadrature_point Zero-based prepared rule index.
    /// @return MFEM geometric measure; requires DETERMINANTS in Flags.
    MFEM_HOST_DEVICE mfem::real_t Determinant(int element,
                                              int quadrature_point) const {
      static_assert(Flags & mfem::GeometricFactors::DETERMINANTS,
                    "determinants were not requested");
      return determinants[quadrature_point + quadrature_points * element];
    }

    /// Overwrite the column-major Dims by Dims metric J^T J.
    MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
    Metric(int element, int quadrature_point, mfem::real_t *metric) const {
      for (int column = 0; column < Dims; ++column) {
        for (int row = 0; row < Dims; ++row) {
          metric[row + Dims * column] = 0.0;
          for (int physical = 0; physical < space_dimensions; ++physical) {
            metric[row + Dims * column] +=
                Jacobian(element, quadrature_point, physical, row) *
                Jacobian(element, quadrature_point, physical, column);
          }
        }
      }
    }

    /// Overwrite the first Dims * Dims entries with (J^T J)^(-1).
    /// Requires a nonsingular metric, including on embedded meshes.
    MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
    InverseMetric(int element, int quadrature_point, mfem::real_t *inverse) const {
      mfem::real_t metric[9] = {};
      Metric(element, quadrature_point, metric);
      if constexpr (Dims == 1) {
        inverse[0] = 1.0 / metric[0];
      } else if constexpr (Dims == 2) {
        const mfem::real_t determinant =
            metric[0] * metric[3] - metric[2] * metric[1];
        inverse[0] = metric[3] / determinant;
        inverse[1] = -metric[1] / determinant;
        inverse[2] = -metric[2] / determinant;
        inverse[3] = metric[0] / determinant;
      } else {
        const mfem::real_t determinant =
            metric[0] * (metric[4] * metric[8] - metric[7] * metric[5]) -
            metric[3] * (metric[1] * metric[8] - metric[7] * metric[2]) +
            metric[6] * (metric[1] * metric[5] - metric[4] * metric[2]);
        inverse[0] =
            (metric[4] * metric[8] - metric[7] * metric[5]) / determinant;
        inverse[1] =
            (metric[7] * metric[2] - metric[1] * metric[8]) / determinant;
        inverse[2] =
            (metric[1] * metric[5] - metric[4] * metric[2]) / determinant;
        inverse[3] =
            (metric[6] * metric[5] - metric[3] * metric[8]) / determinant;
        inverse[4] =
            (metric[0] * metric[8] - metric[6] * metric[2]) / determinant;
        inverse[5] =
            (metric[3] * metric[2] - metric[0] * metric[5]) / determinant;
        inverse[6] =
            (metric[3] * metric[7] - metric[6] * metric[4]) / determinant;
        inverse[7] =
            (metric[6] * metric[1] - metric[0] * metric[7]) / determinant;
        inverse[8] =
            (metric[0] * metric[4] - metric[3] * metric[1]) / determinant;
      }
    }
  };

private:
  const mfem::GeometricFactors *factors_ = nullptr;

public:
  /// Request only Flags from the mesh; validate reference/space dimensions.
  /// @param space Space supplying the mesh (space dimension Dims through 3).
  /// @param integration_rule Common element quadrature rule.
  /// @param use_device Select device-accessible versus host read pointers.
  /// @return Borrowed geometric-factor view, not an independent geometry copy.
  DeviceView Prepare(const mfem::FiniteElementSpace &space,
                     const mfem::IntegrationRule &integration_rule,
                     bool use_device) {
    return Prepare(*space.GetMesh(), integration_rule, use_device);
  }

  /// Prepare coordinate/geometry data without a solution finite-element space.
  /// @param mesh Mesh with reference dimension Dims and space dimension up to 3.
  /// @param integration_rule Common element quadrature rule.
  /// @param use_device Select device-accessible versus host read pointers.
  /// @return Borrowed geometric-factor view owned by mesh.
  DeviceView Prepare(mfem::Mesh &mesh,
                     const mfem::IntegrationRule &integration_rule,
                     bool use_device);

  /// @return True before Prepare(), or when no geometric factors were requested.
  bool Empty() const { return factors_ == nullptr; }
};

namespace internal {
/// Check the compile-time field/coefficient members of a BlfOperation descriptor.
/// @tparam T Descriptor with trial_operator, coefficient, and test_operator.
template <class T>
concept BlfOperationDescriptor = requires {
  typename RequireConstexpr<BlfFieldOperator, T::trial_operator>;
  typename RequireConstexpr<BlfCoefficient, T::coefficient>;
  typename RequireConstexpr<BlfFieldOperator, T::test_operator>;
};

/// @param map MFEM map to classify; VALUE and INTEGRAL are scalar maps.
MFEM_HOST_DEVICE constexpr bool IsScalarMap(mfem::FiniteElement::MapType map) {
  return map == mfem::FiniteElement::VALUE ||
         map == mfem::FiniteElement::INTEGRAL;
}

/// @param map MFEM map to classify, including reduced H(div) variants.
constexpr bool IsHDivMap(mfem::FiniteElement::MapType map) {
  return map == mfem::FiniteElement::H_DIV ||
         map == mfem::FiniteElement::H_DIV_R2D ||
         map == mfem::FiniteElement::H_DIV_R1D;
}

/// @param map MFEM map to classify, including reduced H(curl) variants.
constexpr bool IsHCurlMap(mfem::FiniteElement::MapType map) {
  return map == mfem::FiniteElement::H_CURL ||
         map == mfem::FiniteElement::H_CURL_R2D ||
         map == mfem::FiniteElement::H_CURL_R1D;
}

/// @param map MFEM map to classify as a reduced R1D/R2D vector map.
MFEM_HOST_DEVICE constexpr bool
IsReducedVectorMap(mfem::FiniteElement::MapType map) {
  return map == mfem::FiniteElement::H_DIV_R2D ||
         map == mfem::FiniteElement::H_CURL_R2D ||
         map == mfem::FiniteElement::H_DIV_R1D ||
         map == mfem::FiniteElement::H_CURL_R1D;
}

/// @param map MFEM map to classify; only ordinary H_DIV/H_CURL use these kernels.
constexpr bool IsVectorTensorMap(mfem::FiniteElement::MapType map) {
  return map == mfem::FiniteElement::H_DIV ||
         map == mfem::FiniteElement::H_CURL;
}

/// One separable term in a scalar or vector finite-element operator. The term
/// applies either a value map in every direction (derivative_dimension == -1),
/// or a derivative map in one direction, to one field component.
struct BlfTensorOperatorTerm {
  /// Input vector component, or zero for a scalar field.
  int field_component;
  /// Reference derivative axis; -1 means interpolation without differentiation.
  int derivative_dimension;
  /// Signed contribution to the operator-output component.
  mfem::real_t sign;
};

/// @param op Reference field operation.
/// @param dims Reference dimension (2 or 3 for vector differential operators).
/// @return Separable term count per output component, or zero for an invalid op.
MFEM_HOST_DEVICE constexpr int TensorOperatorTermCount(BlfFieldOperator op,
                                                       int dims) {
  switch (op) {
  case BlfFieldOperator::Value:
  case BlfFieldOperator::Gradient:
    return 1;
  case BlfFieldOperator::Divergence:
    return dims;
  case BlfFieldOperator::Curl:
    return 2;
  default:
    return 0;
  }
}

/// Describe a separable contribution to a reference operator-output component.
/// @param op Reference value, gradient, divergence, or curl operation.
/// @param dims Reference dimension; curl uses its 2D or 3D convention.
/// @param output_component Zero-based reference operator-output index.
/// @param term Zero-based contribution index below TensorOperatorTermCount().
MFEM_HOST_DEVICE constexpr BlfTensorOperatorTerm
TensorOperatorTerm(BlfFieldOperator op, int dims, int output_component,
                   int term) {
  if (op == BlfFieldOperator::Value) {
    return {output_component, -1, 1.0};
  }
  if (op == BlfFieldOperator::Gradient) {
    return {output_component / dims, output_component % dims, 1.0};
  }
  if (op == BlfFieldOperator::Divergence) {
    return {term, term, 1.0};
  }
  if (dims == 2) {
    // curl(u) = d_x u_y - d_y u_x.
    return term == 0 ? BlfTensorOperatorTerm{1, 0, 1.0}
                     : BlfTensorOperatorTerm{0, 1, -1.0};
  }
  // curl(u) = (d_y u_z - d_z u_y,
  //            d_z u_x - d_x u_z,
  //            d_x u_y - d_y u_x).
  constexpr int fields[3][2] = {{2, 1}, {0, 2}, {1, 0}};
  constexpr int derivatives[3][2] = {{1, 2}, {2, 0}, {0, 1}};
  return {fields[output_component][term], derivatives[output_component][term],
          term == 0 ? 1.0 : -1.0};
}

/// Choose the closed versus open one-dimensional tensor basis.
/// @param map Scalar, ordinary H_CURL, or ordinary H_DIV map.
/// @param field_component Input vector component (zero for scalar fields).
/// @param dimension Reference axis being evaluated.
/// @return True for the closed map, false for the open map.
MFEM_HOST_DEVICE constexpr bool
TensorFieldUsesClosedMap(mfem::FiniteElement::MapType map, int field_component,
                         int dimension) {
  if (IsScalarMap(map)) {
    return true;
  }
  // H(curl) is open along the field component and closed transversely;
  // H(div) has the complementary layout.
  return map == mfem::FiniteElement::H_CURL ? dimension != field_component
                                            : dimension == field_component;
}

/// @param base Integer base.
/// @param exponent Nonnegative exponent.
/// @return Integer power without overflow checking.
MFEM_HOST_DEVICE constexpr int TensorIntPow(int base, int exponent) {
  int result = 1;
  for (int i = 0; i < exponent; ++i) {
    result *= base;
  }
  return result;
}

/// Return whether one side of an operator can use the common separable
/// scalar/vector tensor kernel.
/// @tparam Dims Reference dimension.
/// @tparam Operator Reference field operation on this side.
/// @tparam Map MFEM map on this side (not a component-space layout).
/// @param fe Finite element to check for a complete tensor basis.
/// @param rule Rule to check for matching tensor quadrature.
template <int Dims, BlfFieldOperator Operator, mfem::FiniteElement::MapType Map>
bool SupportsTensorField(const mfem::FiniteElement &fe,
                         const mfem::IntegrationRule &rule);

/// Check that both sides support separable kernels with common quadrature.
/// @tparam Dims Reference dimension.
/// @tparam TrialOperator Reference trial operation.
/// @tparam TrialMap Trial MFEM map.
/// @tparam TestOperator Reference test operation.
/// @tparam TestMap Test MFEM map.
/// @param trial_fe Trial finite element.
/// @param test_fe Test finite element.
/// @param rule Common quadrature rule.
template <int Dims, BlfFieldOperator TrialOperator,
          mfem::FiniteElement::MapType TrialMap, BlfFieldOperator TestOperator,
          mfem::FiniteElement::MapType TestMap>
bool SupportsTensorProduct(const mfem::FiniteElement &trial_fe,
                           const mfem::FiniteElement &test_fe,
                           const mfem::IntegrationRule &rule);

/// @param map MFEM map; only scalar maps can represent component fields.
/// @param vdim Space GetVDim(); values greater than one select Cartesian copies.
/// @param out_of_plane Whether the scalar represents a fixed z component.
MFEM_HOST_DEVICE constexpr bool IsComponentField(mfem::FiniteElement::MapType map, int vdim,
                                bool out_of_plane) {
  return IsScalarMap(map) && (vdim > 1 || out_of_plane);
}

/// Select scalar gradients for component-field differential operations; the
/// quadrature pullback subsequently combines them into physical curl/divergence.
/// @param op Desired physical field operation.
/// @param map MFEM finite-element map.
/// @param vdim Space component count.
/// @param out_of_plane Whether the scalar is a fixed Cartesian z component.
constexpr BlfFieldOperator
ReferenceFieldOperator(BlfFieldOperator op, mfem::FiniteElement::MapType map,
                       int vdim, bool out_of_plane) {
  return IsComponentField(map, vdim, out_of_plane) &&
                 op != BlfFieldOperator::Value
             ? BlfFieldOperator::Gradient
             : op;
}

/// Validate operator/map/component compatibility without runtime geometry.
/// @param op Desired physical operation.
/// @param map MFEM finite-element map.
/// @param vdim Positive component count; native vector maps require 1.
/// @param out_of_plane Scalar z-component interpretation; requires vdim == 1.
/// Component derivatives on INTEGRAL maps are rejected to avoid differentiating
/// geometric measure scaling. Runtime dimension checks are performed separately.
constexpr bool IsValidFieldOperator(BlfFieldOperator op,
                                    mfem::FiniteElement::MapType map,
                                    int vdim = 1, bool out_of_plane = false) {
  if (vdim < 1 || (out_of_plane && (!IsScalarMap(map) || vdim != 1)) ||
      (!IsScalarMap(map) && vdim != 1)) {
    return false;
  }
  if (IsComponentField(map, vdim, out_of_plane)) {
    if (op != BlfFieldOperator::Value && map == mfem::FiniteElement::INTEGRAL) {
      return false;
    }
    return op == BlfFieldOperator::Value || op == BlfFieldOperator::Gradient ||
           ((op == BlfFieldOperator::Divergence ||
             op == BlfFieldOperator::Curl) &&
            vdim <= 3);
  }
  switch (op) {
  case BlfFieldOperator::Value:
    return IsScalarMap(map) || IsHDivMap(map) || IsHCurlMap(map);
  case BlfFieldOperator::Gradient:
    return IsScalarMap(map);
  case BlfFieldOperator::Divergence:
    return IsHDivMap(map);
  case BlfFieldOperator::Curl:
    return IsHCurlMap(map);
  }
  return false;
}

/// Number of reference components produced by an operator.  Quadrature data
/// maps these trial components to the test components, so a conservative dense
/// PA representation uses TestComponents * TrialComponents values per point.
/// Component differential operations retain vdim * dims reference gradients,
/// even when the physical output is a scalar divergence or three-component curl.
/// @param op Physical field operation.
/// @param map MFEM finite-element map.
/// @param dims Reference dimension.
/// @param vdim Space component count.
/// @param out_of_plane Fixed scalar z-component interpretation.
constexpr int FieldOperatorComponents(BlfFieldOperator op,
                                      mfem::FiniteElement::MapType map,
                                      int dims, int vdim = 1,
                                      bool out_of_plane = false) {
  if (IsComponentField(map, vdim, out_of_plane)) {
    return vdim * (op == BlfFieldOperator::Value ? 1 : dims);
  }
  switch (op) {
  case BlfFieldOperator::Value:
    return IsScalarMap(map) ? 1 : (IsReducedVectorMap(map) ? 3 : dims);
  case BlfFieldOperator::Gradient:
    return dims;
  case BlfFieldOperator::Divergence:
    return 1;
  case BlfFieldOperator::Curl:
    return IsReducedVectorMap(map) ? 3 : (dims == 2 ? 1 : dims);
  }
  return 0;
}

/// Classify physical outputs: 0 scalar, 1 vector, 2 component gradient.
/// @param op Physical field operation.
/// @param map MFEM finite-element map.
/// @param dims Reference dimension; ordinary 2D H(curl) curl is a pseudoscalar.
/// @param vdim Space component count.
/// @param out_of_plane Fixed scalar z-component interpretation.
/// @return Output rank, or -1 for an unrecognized operation.
MFEM_HOST_DEVICE constexpr int FieldOperatorRank(BlfFieldOperator op,
                                mfem::FiniteElement::MapType map, int dims,
                                int vdim = 1, bool out_of_plane = false) {
  if (IsComponentField(map, vdim, out_of_plane)) {
    return op == BlfFieldOperator::Divergence
               ? 0
               : op == BlfFieldOperator::Gradient ? 2 : 1;
  }
  switch (op) {
  case BlfFieldOperator::Value:
    return IsScalarMap(map) ? 0 : 1;
  case BlfFieldOperator::Gradient:
    return 1;
  case BlfFieldOperator::Divergence:
    return 0;
  case BlfFieldOperator::Curl:
    // The ordinary 2D curl is a pseudoscalar.  Reduced R1D/R2D elements retain
    // all three components and are therefore vector-valued.
    return !IsReducedVectorMap(map) && dims == 2 ? 0 : 1;
  }
  return -1;
}

/// Validate rank compatibility; actual coefficient dimensions are checked later.
/// @param coefficient Physical coefficient-action kind.
/// @param trial_rank Trial output rank, from FieldOperatorRank().
/// @param test_rank Test output rank, from FieldOperatorRank().
constexpr bool IsValidCoefficientAction(BlfCoefficient coefficient,
                                        int trial_rank, int test_rank) {
  switch (coefficient) {
  case BlfCoefficient::Identity:
  case BlfCoefficient::Scalar:
    return trial_rank == test_rank;
  case BlfCoefficient::Vector:
    return (trial_rank == 0 && test_rank == 1) ||
           (trial_rank == 1 && test_rank == 0) ||
           (trial_rank == 1 && test_rank == 2) ||
           (trial_rank == 2 && test_rank == 1);
  case BlfCoefficient::CrossProduct:
    return trial_rank == 1 && test_rank == 1;
  case BlfCoefficient::DiagonalMatrix:
  case BlfCoefficient::SymmetricMatrix:
  case BlfCoefficient::Matrix:
    return true;
  case BlfCoefficient::Tensor:
  case BlfCoefficient::MajorSymmetricTensor:
  case BlfCoefficient::KroneckerTensor:
  case BlfCoefficient::MinorSymmetricTensor:
  case BlfCoefficient::MinorMajorSymmetricTensor:
  case BlfCoefficient::IsotropicSymmetricTensor:
    return (trial_rank == 1 || trial_rank == 2) &&
           (test_rank == 1 || test_rank == 2);
  }
  return false;
}

/// @param coefficient Coefficient kind to classify as a fourth-order tensor.
constexpr bool IsTensorCoefficient(BlfCoefficient coefficient) {
  return coefficient == BlfCoefficient::Tensor ||
         coefficient == BlfCoefficient::MajorSymmetricTensor ||
         coefficient == BlfCoefficient::KroneckerTensor ||
         coefficient == BlfCoefficient::MinorSymmetricTensor ||
         coefficient == BlfCoefficient::MinorMajorSymmetricTensor ||
         coefficient == BlfCoefficient::IsotropicSymmetricTensor;
}

/// Count physical outputs after the geometry pullback, not reference channels.
/// @param op Physical field operation.
/// @param map MFEM finite-element map.
/// @param dims Reference dimension.
/// @param space_dimensions Cartesian space dimension.
/// @param vdim Scalar-space component count (1 for native vector FEs).
/// @param out_of_plane Whether the scalar is the z component of a 3-vector.
/// @return Flattened output size; component curl always has three entries.
MFEM_HOST_DEVICE constexpr int PhysicalFieldComponents(BlfFieldOperator op,
                                      mfem::FiniteElement::MapType map,
                                      int dims, int space_dimensions, int vdim,
                                      bool out_of_plane) {
  if (IsComponentField(map, vdim, out_of_plane)) {
    const int vector_components = out_of_plane ? 3 : vdim;
    switch (op) {
    case BlfFieldOperator::Value:
      return vector_components;
    case BlfFieldOperator::Gradient:
      return vector_components * space_dimensions;
    case BlfFieldOperator::Divergence:
      return 1;
    case BlfFieldOperator::Curl:
      return 3;
    }
  }
  return FieldOperatorRank(op, map, dims) == 0 ? 1 : space_dimensions;
}

/// Reference/physical metadata and pullback for one side of a form.
/// Contains no runtime state; linear and bilinear operations share this policy.
template <int Dims, BlfFieldOperator FieldOperator,
          mfem::FiniteElement::MapType Map, int VDim = 1, bool OutOfPlane = false>
struct BlfFieldDescriptor {
  constexpr static int Components =
      FieldOperatorComponents(FieldOperator, Map, Dims, VDim, OutOfPlane);
  constexpr static int Rank =
      FieldOperatorRank(FieldOperator, Map, Dims, VDim, OutOfPlane);
  constexpr static auto BasisOperator =
      ReferenceFieldOperator(FieldOperator, Map, VDim, OutOfPlane);
  constexpr static bool NeedsInverseMetric =
      FieldOperator == BlfFieldOperator::Gradient ||
      (IsComponentField(Map, VDim, OutOfPlane) &&
       FieldOperator != BlfFieldOperator::Value) ||
      (FieldOperator == BlfFieldOperator::Value && IsHCurlMap(Map));
  constexpr static bool NeedsJacobian = Rank == 1 || NeedsInverseMetric;
  constexpr static int MeasurePower =
      ((!IsComponentField(Map, VDim, OutOfPlane) &&
        (FieldOperator == BlfFieldOperator::Divergence ||
         FieldOperator == BlfFieldOperator::Curl)) ||
       (FieldOperator == BlfFieldOperator::Value &&
        (Map == mfem::FiniteElement::INTEGRAL || IsHDivMap(Map)))) ? -1 : 0;

  MFEM_HOST_DEVICE static constexpr int PhysicalComponents(int space_dimensions) {
    return PhysicalFieldComponents(FieldOperator, Map, Dims, space_dimensions,
                                   VDim, OutOfPlane);
  }

  /// Entry mapping a reference basis channel into a physical output channel.
  /// Geometry is a borrowed view; inverse is the precomputed inverse metric.
  template <class Geometry>
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE static mfem::real_t
  Pullback(const Geometry &geometry, int element, int quadrature_point,
           int physical_component, int reference_component,
           const mfem::real_t *inverse, mfem::real_t measure);
};

/// Payload metadata shared by PA preparation, allocation, and application.
/// Runtime space dimensions become constant when used by a specialized kernel.
template <BlfPAStorage Storage, class Traits> struct BlfPADataLayout {
  constexpr static bool Pointwise =
      Storage == BlfPAStorage::Kronecker || Storage == BlfPAStorage::Kelvin ||
      Storage == BlfPAStorage::SymmetricKelvin ||
      Storage == BlfPAStorage::IsotropicSymmetric;

  static constexpr int Components(int space_dimensions) {
    const int kelvin = space_dimensions * (space_dimensions + 1) / 2;
    if constexpr (Storage == BlfPAStorage::UniformDiagonal) { return 1; }
    if constexpr (Storage == BlfPAStorage::Diagonal) { return Traits::TrialComponents; }
    if constexpr (Storage == BlfPAStorage::Symmetric) {
      return Traits::TrialComponents * (Traits::TrialComponents + 1) / 2;
    }
    if constexpr (Storage == BlfPAStorage::Kronecker) {
      return Traits::TrialVDim * Traits::TestVDim + Traits::Dims * Traits::Dims;
    }
    if constexpr (Storage == BlfPAStorage::Kelvin) {
      return space_dimensions * Traits::Dims + kelvin * kelvin;
    }
    if constexpr (Storage == BlfPAStorage::SymmetricKelvin) {
      return space_dimensions * Traits::Dims + kelvin * (kelvin + 1) / 2;
    }
    if constexpr (Storage == BlfPAStorage::IsotropicSymmetric) {
      return space_dimensions * Traits::Dims + 2;
    }
    return Traits::TrialComponents * Traits::TestComponents;
  }
};

/// Borrow the finite element's lexicographic ordering, including signed maps.
inline const mfem::Array<int> *LexicographicDofMap(const mfem::FiniteElement &element) {
  if (const auto *nodal = dynamic_cast<const mfem::NodalFiniteElement *>(&element)) {
    return &nodal->GetLexicographicOrdering();
  }
  if (const auto *tensor = dynamic_cast<const mfem::TensorBasisElement *>(&element)) {
    return &tensor->GetDofMap();
  }
  return nullptr;
}

/// An absent or empty ordering map denotes the identity ordering.
inline const int *ReadLexicographicDofMap(const mfem::FiniteElement &element,
                                         bool use_device = true) {
  const auto *map = LexicographicDofMap(element);
  return map && map->Size() ? map->Read(use_device) : nullptr;
}

/// Traits for implementing y = [d]v^T M [d]u, where M is potentially a matrix.
/// T should provide a mechanism for assembling M at a quadrature point. This
/// should bake in any geometric factors required. M is stored as a flattened
/// vector.
///
/// T also provides a mapping y += [d]v^T M [d]u where [d] represents optional
/// partial derivatives with respect to the local coordinates.
///
/// PA/MF is implemented by staging Q = M [d]u into a temporary buffer evaluated
/// at quadrature points before performing the contraction [d]v^T Q.
/// @tparam T Operation with mandatory Operation, Dims, trial_type, and test_type
/// compile-time members. Optional Mixed defaults to false, trial_vdim/test_vdim
/// to 1, and trial_out_of_plane/test_out_of_plane to false. Reference component
/// counts describe basis channels, not necessarily physical coefficient sizes.
template <class T> struct BlfIntegratorTraits {
  /// T::Operation is mandatory: silently defaulting to value/value would make
  /// gradient, divergence and curl integrators instantiate the wrong kernels.
  using Operation = typename T::Operation;
  static_assert(BlfOperationDescriptor<Operation>,
                "T::Operation must be a BlfOperation descriptor");

  /// Whether trial/test spaces may differ. Mixed PA kernels stage separate
  /// basis tables even when the spaces coincide; non-mixed kernels share them.
  constexpr static auto Mixed = ExtractMixed<T>::value;
  /// trial space MapType
  constexpr static mfem::FiniteElement::MapType trial_type = T::trial_type;
  /// test space MapType
  constexpr static mfem::FiniteElement::MapType test_type = T::test_type;

  /// reference space number of dimensions
  constexpr static int Dims = T::Dims;

  constexpr static auto trial_operator = Operation::trial_operator;
  constexpr static auto coefficient = Operation::coefficient;
  constexpr static auto test_operator = Operation::test_operator;
  constexpr static int TrialVDim = ExtractTrialVDim<T>();
  constexpr static int TestVDim = ExtractTestVDim<T>();
  constexpr static bool TrialOutOfPlane = ExtractTrialOutOfPlane<T>();
  constexpr static bool TestOutOfPlane = ExtractTestOutOfPlane<T>();
  using TrialField = internal::BlfFieldDescriptor<Dims, trial_operator, trial_type,
                                                   TrialVDim, TrialOutOfPlane>;
  using TestField = internal::BlfFieldDescriptor<Dims, test_operator, test_type,
                                                  TestVDim, TestOutOfPlane>;

  constexpr static bool HasComponentFields =
      IsComponentField(trial_type, TrialVDim, TrialOutOfPlane) ||
      IsComponentField(test_type, TestVDim, TestOutOfPlane);
  constexpr static auto TrialBasisOperator = TrialField::BasisOperator;
  constexpr static auto TestBasisOperator = TestField::BasisOperator;

  static_assert(Dims >= 1 && Dims <= 3,
                "reference-space dimension must be in [1, 3]");
  static_assert(IsValidFieldOperator(trial_operator, trial_type, TrialVDim,
                                     TrialOutOfPlane),
                "trial operator is incompatible with the trial MapType");
  static_assert(IsValidFieldOperator(test_operator, test_type, TestVDim,
                                     TestOutOfPlane),
                "test operator is incompatible with the test MapType");

  constexpr static int TrialComponents = TrialField::Components;
  constexpr static int TestComponents = TestField::Components;
  constexpr static int TrialRank = TrialField::Rank;
  constexpr static int TestRank = TestField::Rank;

  static_assert(IsValidCoefficientAction(coefficient, TrialRank, TestRank),
                "coefficient cannot map the trial operator result to the test "
                "operator result");
  static_assert(!IsTensorCoefficient(coefficient) ||
                    (trial_operator == BlfFieldOperator::Gradient &&
                     test_operator == BlfFieldOperator::Gradient),
                "fourth-order tensors require gradient operators on both sides");

  /// Conservative storage for the geometry-transformed pointwise operator.
  /// Integrators may opt into a smaller representation when symmetry or a
  /// special geometry makes that possible.
  constexpr static int MaxQDataComponents = TrialComponents * TestComponents;
};


}

/// Generic physical-space bilinear form operation. The field operators and
/// finite-element maps determine the pullbacks on each side; CoefficientKind
/// determines the pointwise action between them.
///
/// Assembles (test operator v)^T C (trial operator u), including the geometric
/// measure but excluding the integration rule's point weight. Coefficients are
/// borrowed and must outlive this operation and any adapters using it.
///
/// Scalar VALUE/INTEGRAL maps with VDim > 1 represent fixed Cartesian vector
/// components. Component gradients are flattened as
/// direction + space_dimensions * field_component. Curl/divergence use
/// Cartesian first derivatives
/// using J (J^T J)^(-1), including on embedded meshes; no Hessians are used.
/// Component derivatives require VALUE maps (H1 or value-mapped L2); L2
/// derivatives are elementwise and do not add face terms. Component layouts use
/// tensor-product EA/PA kernels when both bases and quadrature are compatible,
/// with full-basis fallback otherwise. Reduced R1D/R2D vector maps are not
/// supported by this operation.
///
/// @tparam ReferenceDimensions Mesh topological dimension, between 1 and 3.
/// @tparam TrialOperator Operation on the trial field before coefficient action.
/// @tparam TrialMap MFEM map of the trial finite element.
/// @tparam CoefficientKind Identity, scalar, vector, cross product, diagonal,
/// symmetric matrix, general matrix, or explicit fourth-order tensor action.
/// Matrix/Tensor height is the full physical test output size and width the full
/// physical trial output size. KroneckerTensor represents Q_ij K_ab explicitly;
/// Other tensor kinds specify major/minor symmetry or isotropic symmetric action.
/// Tensor kinds require gradients on both sides; a scalar gradient is the
/// one-field-component case of the tensor contraction. No coefficient
/// interpretation is inferred from dimensions.
/// Vector action multiplies/contracts scalar/vector outputs, or contracts the
/// derivative direction of a component gradient against a vector coefficient.
/// CrossProduct applies coefficient cross trial and requires a 3-vector.
/// @tparam TestOperator Operation on the test field before contraction.
/// @tparam TestMap MFEM map of the test finite element.
/// @tparam IsMixed Whether trial/test spaces may differ. Mixed PA kernels always
/// stage separate trial/test basis maps, even when given the same space. False
/// requires the same trial/test space and enables compile-time basis sharing;
/// full-basis tables additionally require matching field operations. Set false
/// for MFEM's single-space PA and single-element assembly overloads.
/// @tparam TrialVDim Trial space GetVDim(); positive, and 1 for native vector FEs.
/// @tparam TestVDim Test space GetVDim(); positive, and 1 for native vector FEs.
/// @tparam TrialOutOfPlane Interpret a scalar trial field as (0,0,s); requires a
/// scalar map and TrialVDim == 1. This is a fixed Cartesian z component, not a
/// varying surface-normal field.
/// @tparam TestOutOfPlane Same z-component interpretation for the scalar test
/// field; requires a scalar map and TestVDim == 1.
template <int ReferenceDimensions, BlfFieldOperator TrialOperator,
          mfem::FiniteElement::MapType TrialMap, BlfCoefficient CoefficientKind,
          BlfFieldOperator TestOperator, mfem::FiniteElement::MapType TestMap,
          bool IsMixed = true, int TrialVDim = 1, int TestVDim = 1,
          bool TrialOutOfPlane = false, bool TestOutOfPlane = false>
struct BilinearFormOperator {
  using Operation = BlfOperation<TrialOperator, CoefficientKind, TestOperator>;
  constexpr static auto trial_type = TrialMap;
  constexpr static auto test_type = TestMap;
  constexpr static int Dims = ReferenceDimensions;
  constexpr static bool Mixed = IsMixed;
  constexpr static int trial_vdim = TrialVDim;
  constexpr static int test_vdim = TestVDim;
  constexpr static bool trial_out_of_plane = TrialOutOfPlane;
  constexpr static bool test_out_of_plane = TestOutOfPlane;

  using TrialField = internal::BlfFieldDescriptor<Dims, TrialOperator, TrialMap,
                                                   TrialVDim, TrialOutOfPlane>;
  using TestField = internal::BlfFieldDescriptor<Dims, TestOperator, TestMap,
                                                  TestVDim, TestOutOfPlane>;

  static_assert(internal::IsValidFieldOperator(TrialOperator, TrialMap,
                                               TrialVDim, TrialOutOfPlane));
  static_assert(internal::IsValidFieldOperator(TestOperator, TestMap, TestVDim,
                                               TestOutOfPlane));

  constexpr static int TrialRank = TrialField::Rank;
  constexpr static int TestRank = TestField::Rank;
  constexpr static int TrialComponents = TrialField::Components;
  constexpr static int TestComponents = TestField::Components;

  /// Whether reference pullbacks coincide, irrespective of polynomial order.
  constexpr static bool MatchingPullbacks =
      TrialOperator == TestOperator && TrialMap == TestMap &&
      TrialVDim == TestVDim && TrialOutOfPlane == TestOutOfPlane;
  /// Preferred PA representation. Setup falls back to dense if not smaller.
  constexpr static BlfPAStorage PAStorage = [] {
    if constexpr (TrialVDim > 1 && TrialVDim == TestVDim &&
                  TrialOutOfPlane == TestOutOfPlane &&
                  internal::IsScalarMap(TrialMap) &&
                  internal::IsScalarMap(TestMap) &&
                  (((CoefficientKind == BlfCoefficient::Identity ||
                     CoefficientKind == BlfCoefficient::Scalar) &&
                    ((TrialOperator == BlfFieldOperator::Value &&
                      TestOperator == BlfFieldOperator::Value) ||
                     (Dims == 1 &&
                      TrialOperator == BlfFieldOperator::Gradient &&
                      TestOperator == BlfFieldOperator::Gradient))) ||
                   (Dims == 1 && CoefficientKind == BlfCoefficient::Vector &&
                    ((TrialOperator == BlfFieldOperator::Gradient &&
                      TestOperator == BlfFieldOperator::Value) ||
                     (TrialOperator == BlfFieldOperator::Value &&
                      TestOperator == BlfFieldOperator::Gradient))))) {
      return BlfPAStorage::UniformDiagonal;
    } else if constexpr (CoefficientKind == BlfCoefficient::KroneckerTensor) {
      return BlfPAStorage::Kronecker;
    } else if constexpr (CoefficientKind == BlfCoefficient::MinorSymmetricTensor) {
      return BlfPAStorage::Kelvin;
    } else if constexpr (CoefficientKind == BlfCoefficient::MinorMajorSymmetricTensor) {
      return BlfPAStorage::SymmetricKelvin;
    } else if constexpr (CoefficientKind == BlfCoefficient::IsotropicSymmetricTensor) {
      return BlfPAStorage::IsotropicSymmetric;
    } else if constexpr (MatchingPullbacks &&
                         (CoefficientKind == BlfCoefficient::Identity ||
                          CoefficientKind == BlfCoefficient::Scalar ||
                          CoefficientKind == BlfCoefficient::DiagonalMatrix) &&
                         TrialOperator == BlfFieldOperator::Value &&
                         internal::IsScalarMap(TrialMap)) {
      return BlfPAStorage::Diagonal;
    } else if constexpr (MatchingPullbacks &&
                         (CoefficientKind == BlfCoefficient::Identity ||
                          CoefficientKind == BlfCoefficient::Scalar ||
                          CoefficientKind == BlfCoefficient::DiagonalMatrix ||
                          CoefficientKind == BlfCoefficient::SymmetricMatrix ||
                          CoefficientKind == BlfCoefficient::MajorSymmetricTensor)) {
      return BlfPAStorage::Symmetric;
    } else {
      return BlfPAStorage::Dense;
    }
  }();

  /// Whether coefficient projection uses a full MFEM matrix coefficient.
  constexpr static bool UsesMatrixCoefficient =
      CoefficientKind == BlfCoefficient::Matrix ||
      CoefficientKind == BlfCoefficient::SymmetricMatrix ||
      CoefficientKind == BlfCoefficient::Tensor ||
      CoefficientKind == BlfCoefficient::MajorSymmetricTensor ||
      CoefficientKind == BlfCoefficient::KroneckerTensor ||
      CoefficientKind == BlfCoefficient::MinorSymmetricTensor ||
      CoefficientKind == BlfCoefficient::MinorMajorSymmetricTensor ||
      CoefficientKind == BlfCoefficient::IsotropicSymmetricTensor;

  static_assert(internal::IsValidCoefficientAction(CoefficientKind, TrialRank,
                                                   TestRank));
  static_assert(!internal::IsTensorCoefficient(CoefficientKind) ||
                    (TrialOperator == BlfFieldOperator::Gradient &&
                     TestOperator == BlfFieldOperator::Gradient),
                "fourth-order tensors require gradient operators on both sides");

  constexpr static bool NeedsInverseMetric =
      TrialField::NeedsInverseMetric || TestField::NeedsInverseMetric;
  constexpr static bool NeedsJacobian =
      TrialField::NeedsJacobian || TestField::NeedsJacobian;
  constexpr static int MeasurePower =
      1 + TrialField::MeasurePower + TestField::MeasurePower;
  constexpr static bool NeedsDeterminant = MeasurePower != 0;
  constexpr static int GeometryFlags =
      (NeedsJacobian ? mfem::GeometricFactors::JACOBIANS : 0) |
      (NeedsDeterminant ? mfem::GeometricFactors::DETERMINANTS : 0);

  static_assert(!NeedsInverseMetric || NeedsJacobian);

  static_assert(!internal::IsReducedVectorMap(TrialMap) &&
                    !internal::IsReducedVectorMap(TestMap),
                "the generic physical-space integrator does not yet support "
                "reduced R1D/R2D vector maps");

  /// Construct an operation with identity coefficient action.
  BilinearFormOperator()
    requires(CoefficientKind == BlfCoefficient::Identity)
  = default;

  /// @param coefficient Borrowed scalar multiplier.
  explicit BilinearFormOperator(mfem::Coefficient &coefficient)
    requires(CoefficientKind == BlfCoefficient::Scalar)
      : scalar_coefficient(&coefficient) {}

  /// @param coefficient Borrowed vector for multiplication/contraction, cross
  /// product, or diagonal entries, as selected by CoefficientKind.
  explicit BilinearFormOperator(mfem::VectorCoefficient &coefficient)
    requires(CoefficientKind == BlfCoefficient::Vector ||
             CoefficientKind == BlfCoefficient::CrossProduct ||
             CoefficientKind == BlfCoefficient::DiagonalMatrix)
      : vector_coefficient(&coefficient) {}

  /// @param coefficient Borrowed physical-space matrix; dimensions are checked
  /// at setup. SymmetricMatrix additionally promises pointwise symmetry and
  /// requires equal output sizes; it may use packed PA reference storage.
  explicit BilinearFormOperator(mfem::MatrixCoefficient &coefficient)
    requires(CoefficientKind == BlfCoefficient::SymmetricMatrix ||
             CoefficientKind == BlfCoefficient::Matrix ||
             CoefficientKind == BlfCoefficient::Tensor)
      : matrix_coefficient(&coefficient) {}

  /// @param coefficient Symmetric matrix on full flattened gradient entries;
  /// major symmetry is guaranteed by MFEM's symmetric coefficient interface.
  explicit BilinearFormOperator(mfem::SymmetricMatrixCoefficient &coefficient)
    requires(CoefficientKind == BlfCoefficient::MajorSymmetricTensor)
      : matrix_coefficient(&coefficient) {}

  /// @param coefficient Explicit field/direction product Q tensor K. Use an
  /// identity direction factor to replace the old component-only convention.
  explicit BilinearFormOperator(BlfKroneckerTensorCoefficient &coefficient)
    requires(CoefficientKind == BlfCoefficient::KroneckerTensor)
      : matrix_coefficient(&coefficient),
        tensor_dimensions(coefficient.Dimension()) {}

  /// @param coefficient Kelvin constitutive matrix enforcing both minor
  /// symmetries, without requiring major symmetry.
  explicit BilinearFormOperator(BlfMinorSymmetricTensorCoefficient &coefficient)
    requires(CoefficientKind == BlfCoefficient::MinorSymmetricTensor)
      : matrix_coefficient(&coefficient),
        tensor_dimensions(coefficient.Dimension()) {}

  /// @param coefficient Symmetric Kelvin constitutive matrix enforcing minor
  /// and major symmetries.
  explicit BilinearFormOperator(BlfMinorMajorSymmetricTensorCoefficient &coefficient)
    requires(CoefficientKind == BlfCoefficient::MinorMajorSymmetricTensor)
      : matrix_coefficient(&coefficient),
        tensor_dimensions(coefficient.Dimension()) {}

  /// @param coefficient Isotropic symmetric tensor built from lambda and mu.
  explicit BilinearFormOperator(BlfIsotropicSymmetricTensorCoefficient &coefficient)
    requires(CoefficientKind == BlfCoefficient::IsotropicSymmetricTensor)
      : matrix_coefficient(&coefficient),
        tensor_dimensions(coefficient.Dimension()) {}

  mfem::Coefficient *scalar_coefficient = nullptr;
  mfem::VectorCoefficient *vector_coefficient = nullptr;
  mfem::MatrixCoefficient *matrix_coefficient = nullptr;
  /// Cartesian dimension declared by structured tensor adapters; zero for
  /// unstructured coefficients. Must match the mesh space dimension.
  int tensor_dimensions = 0;
  BlfCoefficientStorage coefficient_storage;
  /// Auxiliary factor projection owned by compact structured PA setup.
  BlfCoefficientStorage pa_factor_storage;
  BlfGeometryFactors<GeometryFlags, Dims> geometry_storage;

  /// Trivially copyable, non-owning coefficient/geometry views for CPU/GPU
  /// quadrature assembly. Views become invalid when their backing storage is
  /// replaced or destroyed; do not retain them across PrepareQData() calls.
  struct PreparedQData {
    BlfCoefficientStorage::DeviceView coefficient;
    typename BlfGeometryFactors<GeometryFlags, Dims>::DeviceView geometry;

    /// Read one physical coefficient-action entry, including structural zeros.
    /// @param row Flattened physical test-output index.
    /// @param column Flattened physical trial-output index.
    /// @param rows Full physical test-output size.
    /// @param element Zero-based mesh element index.
    /// @param quadrature_point Zero-based prepared quadrature index.
    MFEM_HOST_DEVICE mfem::real_t CoefficientEntry(int row, int column,
                                                   int rows, int element,
                                                   int quadrature_point) const;

    /// Fill the reference-space tensor P_test^T C P_trial times the measure.
    /// The quadrature point's integration weight is applied by the adapter.
    /// @param element Zero-based mesh element index.
    /// @param quadrature_point Zero-based prepared quadrature index.
    /// @param[out] qdata Writable TestComponents by TrialComponents view.
    /// @tparam QData Writable dense or packed reference tensor view.
    template <class QData>
    MFEM_HOST_DEVICE void
    AssembleQData(int element, int quadrature_point, QData qdata) const;
  };

  /// Check coefficient presence, dimensions, and cross-product compatibility.
  /// @param space_dimensions Mesh Cartesian space dimension.
  /// @note A failed check invokes MFEM_VERIFY; no coefficient is resized.
  void VerifyCoefficientDimensions(int space_dimensions) const;

  /// Project the coefficient and obtain the requested mesh geometric factors.
  /// @param trial_space Trial space, with GetVDim() == TrialVDim.
  /// @param test_space Test space on the same mesh, with GetVDim() == TestVDim.
  /// @param integration_rule Common element rule used for projection/geometry.
  /// @param use_device Whether returned pointers refer to device-accessible
  /// storage or host storage.
  /// @return Borrowed views owned by this operation and the mesh. A subsequent
  /// setup may invalidate the views; consume them before replacing the storage.
  PreparedQData PrepareQData(const mfem::FiniteElementSpace &trial_space,
                             const mfem::FiniteElementSpace &test_space,
                             const mfem::IntegrationRule &integration_rule,
                             bool use_device);

  /// @param space_dimensions Cartesian space dimension.
  /// @return Number of entries per quadrature point in the preferred layout.
  static constexpr int PAComponents(int space_dimensions);

  /// Borrowed projection/geometry views for structured PA, without expansion
  /// of the physical fourth-order tensor. Valid until the next preparation.
  struct PreparedPAData {
    PreparedQData base;
    BlfCoefficientStorage::DeviceView factor;
    int field_rows;
    bool diagonal_field;
    bool identity_directions;

    /// @param element Element index.
    /// @param point Quadrature index.
    /// @param[out] data Compact contiguous payload of PAComponents() entries.
    /// @param weight Integration-rule weight, included in the material action.
    MFEM_HOST_DEVICE void AssemblePAData(int element, int point,
                                         mfem::real_t *data,
                                         mfem::real_t weight) const;
  };

  /// Project only the factors needed by structured PA. Dense EA is unchanged.
  /// @param trial_space Trial space with the declared vdim.
  /// @param test_space Test space on the same mesh.
  /// @param rule Common element quadrature rule.
  /// @param use_device Make projection views device accessible.
  /// @return Trivially copyable compact setup view.
  PreparedPAData PreparePAData(const mfem::FiniteElementSpace &trial_space,
                               const mfem::FiniteElementSpace &test_space,
                               const mfem::IntegrationRule &rule,
                               bool use_device)
    requires(PAStorage == BlfPAStorage::Kronecker ||
             PAStorage == BlfPAStorage::Kelvin ||
             PAStorage == BlfPAStorage::SymmetricKelvin ||
             PAStorage == BlfPAStorage::IsotropicSymmetric);

  /// Host single-point counterpart of PreparedQData::AssembleQData().
  /// @param transformation Element geometry with its current integration point
  /// already set to ip by the caller.
  /// @param ip Point at which to evaluate the coefficient.
  /// @param[out] qdata Pre-sized TestComponents by TrialComponents matrix;
  /// overwritten with geometry-transformed entries, excluding ip.weight.
  void AssembleQDataHost(mfem::ElementTransformation &transformation,
                         const mfem::IntegrationPoint &ip,
                         mfem::DenseMatrix &qdata);
};
} // namespace mfem

#include "bilininteg_operator_impl.hpp"

#endif
