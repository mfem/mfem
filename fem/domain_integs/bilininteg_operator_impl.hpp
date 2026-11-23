#ifndef BILININTEG_OPERATOR_IMPL_HPP
#define BILININTEG_OPERATOR_IMPL_HPP

/// @file
/// Coefficient projection, geometric pullbacks, and quadrature assembly for
/// BilinearFormOperator.

#include "bilininteg_operator.hpp"

namespace mfem {

namespace internal {

template <int Dims, BlfFieldOperator Operator, mfem::FiniteElement::MapType Map>
bool SupportsTensorField(const mfem::FiniteElement &fe,
                         const mfem::IntegrationRule &rule) {
  if constexpr (IsScalarMap(Map)) {
    if constexpr (Operator != BlfFieldOperator::Value &&
                  Operator != BlfFieldOperator::Gradient) {
      return false;
    }
    if (!dynamic_cast<const mfem::TensorBasisElement *>(&fe)) {
      return false;
    }
    const auto &map = fe.GetDofToQuad(rule, mfem::DofToQuad::TENSOR);
    return TensorIntPow(map.ndof, Dims) == fe.GetDof() &&
           TensorIntPow(map.nqpt, Dims) == rule.GetNPoints();
  } else if constexpr (IsVectorTensorMap(Map)) {
    if constexpr (Operator != BlfFieldOperator::Value &&
                  Operator != BlfFieldOperator::Curl &&
                  Operator != BlfFieldOperator::Divergence) {
      return false;
    }
    const auto *vector =
        dynamic_cast<const mfem::VectorTensorFiniteElement *>(&fe);
    if (!vector || (Dims != 2 && Dims != 3)) {
      return false;
    }
    const auto &closed = vector->GetDofToQuad(rule, mfem::DofToQuad::TENSOR);
    const auto &open = vector->GetDofToQuadOpen(rule, mfem::DofToQuad::TENSOR);
    const int component_dofs =
        Map == mfem::FiniteElement::H_CURL
            ? open.ndof * TensorIntPow(closed.ndof, Dims - 1)
            : closed.ndof * TensorIntPow(open.ndof, Dims - 1);
    return closed.ndof == open.ndof + 1 && closed.nqpt == open.nqpt &&
           Dims * component_dofs == fe.GetDof() &&
           TensorIntPow(closed.nqpt, Dims) == rule.GetNPoints();
  } else {
    return false;
  }
}

template <int Dims, BlfFieldOperator TrialOperator,
          mfem::FiniteElement::MapType TrialMap, BlfFieldOperator TestOperator,
          mfem::FiniteElement::MapType TestMap>
bool SupportsTensorProduct(const mfem::FiniteElement &trial_fe,
                           const mfem::FiniteElement &test_fe,
                           const mfem::IntegrationRule &rule) {
  if (!SupportsTensorField<Dims, TrialOperator, TrialMap>(trial_fe, rule) ||
      !SupportsTensorField<Dims, TestOperator, TestMap>(test_fe, rule)) {
    return false;
  }
  const auto &trial_map = trial_fe.GetDofToQuad(rule, mfem::DofToQuad::TENSOR);
  const auto &test_map = test_fe.GetDofToQuad(rule, mfem::DofToQuad::TENSOR);
  return trial_map.nqpt == test_map.nqpt;
}

template <int Dims, BlfFieldOperator FieldOperator,
          mfem::FiniteElement::MapType Map, int VDim, bool OutOfPlane>
template <class Geometry>
MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE mfem::real_t
BlfFieldDescriptor<Dims, FieldOperator, Map, VDim, OutOfPlane>::Pullback(
    const Geometry &geometry, int element, int quadrature_point,
    int physical_component, int reference_component,
    const mfem::real_t *inverse, mfem::real_t measure) {
  constexpr int rank = Rank;
  if constexpr (internal::IsComponentField(Map, VDim, OutOfPlane)) {
    const int field = FieldOperator == BlfFieldOperator::Value
                          ? reference_component
                          : reference_component / Dims;
    const int cartesian_field = OutOfPlane ? 2 : field;
    if constexpr (FieldOperator == BlfFieldOperator::Value) {
      return physical_component == cartesian_field
                 ? (Map == mfem::FiniteElement::INTEGRAL ? 1.0 / measure
                                                         : 1.0)
                 : 0.0;
    } else {
      const int reference_derivative = reference_component % Dims;
      mfem::real_t value = 0.0;
      for (int cartesian_derivative = 0;
           cartesian_derivative < geometry.space_dimensions;
           ++cartesian_derivative) {
        mfem::real_t factor = 0.0;
        if constexpr (FieldOperator == BlfFieldOperator::Gradient) {
          factor = physical_component ==
                           cartesian_derivative +
                               geometry.space_dimensions * cartesian_field
                       ? 1.0
                       : 0.0;
        } else if constexpr (FieldOperator ==
                             BlfFieldOperator::Divergence) {
          factor = cartesian_derivative == cartesian_field ? 1.0 : 0.0;
        } else {
          static_assert(FieldOperator == BlfFieldOperator::Curl);
          if (cartesian_derivative == (physical_component + 1) % 3 &&
              cartesian_field == (physical_component + 2) % 3) {
            factor = 1.0;
          } else if (cartesian_derivative == (physical_component + 2) % 3 &&
                     cartesian_field == (physical_component + 1) % 3) {
            factor = -1.0;
          }
        }
        for (int reference = 0; reference < Dims; ++reference) {
          value += factor *
                   geometry.Jacobian(element, quadrature_point,
                                     cartesian_derivative, reference) *
                   inverse[reference + Dims * reference_derivative];
        }
      }
      return value;
    }
  } else if constexpr (rank == 0) {
    static_assert(FieldOperator == BlfFieldOperator::Value ||
                  FieldOperator == BlfFieldOperator::Divergence ||
                  FieldOperator == BlfFieldOperator::Curl);
    if constexpr (FieldOperator == BlfFieldOperator::Value) {
      return Map == mfem::FiniteElement::INTEGRAL ? 1.0 / measure : 1.0;
    } else {
      return 1.0 / measure;
    }
  } else if constexpr (FieldOperator == BlfFieldOperator::Gradient ||
                       (FieldOperator == BlfFieldOperator::Value &&
                        internal::IsHCurlMap(Map))) {
    mfem::real_t value = 0.0;
    for (int k = 0; k < Dims; ++k) {
      value += geometry.Jacobian(element, quadrature_point,
                                 physical_component, k) *
               inverse[k + Dims * reference_component];
    }
    return value;
  } else {
    static_assert((FieldOperator == BlfFieldOperator::Value &&
                   internal::IsHDivMap(Map)) ||
                  FieldOperator == BlfFieldOperator::Curl);
    return geometry.Jacobian(element, quadrature_point, physical_component,
                             reference_component) /
           measure;
  }
}

} // namespace internal

template <int Flags, int Dims>
typename BlfGeometryFactors<Flags, Dims>::DeviceView BlfGeometryFactors<Flags, Dims>::Prepare(mfem::Mesh &mesh,
                     const mfem::IntegrationRule &integration_rule,
                     bool use_device) {
  MFEM_VERIFY(mesh.Dimension() == Dims,
              "mesh has the wrong reference dimension");
  MFEM_VERIFY(mesh.SpaceDimension() >= Dims &&
                  mesh.SpaceDimension() <= 3,
              "mesh has an invalid physical dimension");
  if constexpr (Flags != 0) {
    factors_ = mesh.GetGeometricFactors(
        integration_rule, Flags, mfem::Device::GetDeviceMemoryType());
  } else {
    factors_ = nullptr;
  }
  return {Flags & mfem::GeometricFactors::COORDINATES
              ? factors_->X.Read(use_device)
              : nullptr,
          Flags & mfem::GeometricFactors::JACOBIANS
              ? factors_->J.Read(use_device)
              : nullptr,
          Flags & mfem::GeometricFactors::DETERMINANTS
              ? factors_->detJ.Read(use_device)
              : nullptr,
          integration_rule.GetNPoints(), mesh.SpaceDimension()};
}
template <int ReferenceDimensions, BlfFieldOperator TrialOperator,
          mfem::FiniteElement::MapType TrialMap, BlfCoefficient CoefficientKind,
          BlfFieldOperator TestOperator, mfem::FiniteElement::MapType TestMap,
          bool IsMixed, int TrialVDim, int TestVDim,
          bool TrialOutOfPlane, bool TestOutOfPlane>
MFEM_HOST_DEVICE mfem::real_t BilinearFormOperator<ReferenceDimensions, TrialOperator, TrialMap,
                     CoefficientKind, TestOperator, TestMap, IsMixed,
                     TrialVDim, TestVDim, TrialOutOfPlane, TestOutOfPlane>::PreparedQData::CoefficientEntry(int row, int column,
                                                   int rows, int element,
                                                   int quadrature_point) const {
  if constexpr (CoefficientKind == BlfCoefficient::Identity) {
    return row == column ? 1.0 : 0.0;
  } else if constexpr (CoefficientKind == BlfCoefficient::Scalar) {
    return (row == column ? 1.0 : 0.0) *
           coefficient(0, element, quadrature_point);
  } else if constexpr (CoefficientKind == BlfCoefficient::Vector) {
    if constexpr (TrialRank == 2) {
      return row == column / geometry.space_dimensions
                 ? coefficient(column % geometry.space_dimensions,
                               element, quadrature_point)
                 : 0.0;
    } else if constexpr (TestRank == 2) {
      return row / geometry.space_dimensions == column
                 ? coefficient(row % geometry.space_dimensions, element,
                               quadrature_point)
                 : 0.0;
    } else {
      return coefficient(TestRank == 0 ? column : row, element,
                         quadrature_point);
    }
  } else if constexpr (CoefficientKind == BlfCoefficient::DiagonalMatrix) {
    return row == column ? coefficient(row, element, quadrature_point)
                         : 0.0;
  } else if constexpr (CoefficientKind == BlfCoefficient::CrossProduct) {
    if (row == column) {
      return 0.0;
    }
    if (row == 0) {
      return column == 1 ? -coefficient(2, element, quadrature_point)
                         : coefficient(1, element, quadrature_point);
    }
    if (row == 1) {
      return column == 0 ? coefficient(2, element, quadrature_point)
                         : -coefficient(0, element, quadrature_point);
    }
    return column == 0 ? -coefficient(1, element, quadrature_point)
                       : coefficient(0, element, quadrature_point);
  } else {
    return coefficient(row + rows * column, element,
                       quadrature_point);
  }
}

template <int ReferenceDimensions, BlfFieldOperator TrialOperator,
          mfem::FiniteElement::MapType TrialMap, BlfCoefficient CoefficientKind,
          BlfFieldOperator TestOperator, mfem::FiniteElement::MapType TestMap,
          bool IsMixed, int TrialVDim, int TestVDim,
          bool TrialOutOfPlane, bool TestOutOfPlane>
template <class QData>
MFEM_HOST_DEVICE void
    BilinearFormOperator<ReferenceDimensions, TrialOperator, TrialMap,
                     CoefficientKind, TestOperator, TestMap, IsMixed,
                     TrialVDim, TestVDim, TrialOutOfPlane, TestOutOfPlane>::PreparedQData::AssembleQData(int element, int quadrature_point, QData qdata) const {
  mfem::real_t measure = 1.0;
  if constexpr (NeedsDeterminant) {
    measure = geometry.Determinant(element, quadrature_point);
  }
  mfem::real_t inverse[9] = {};
  if constexpr (NeedsInverseMetric) {
    geometry.InverseMetric(element, quadrature_point, inverse);
  }

  const int test_physical_components = internal::PhysicalFieldComponents(
      TestOperator, TestMap, Dims, geometry.space_dimensions, TestVDim,
      TestOutOfPlane);
  const int trial_physical_components = internal::PhysicalFieldComponents(
      TrialOperator, TrialMap, Dims, geometry.space_dimensions, TrialVDim,
      TrialOutOfPlane);

  for (int test = 0; test < TestComponents; ++test) {
    for (int trial = 0; trial < TrialComponents; ++trial) {
      if (!QData::StoresEntry(test, trial)) { continue; }
      mfem::real_t value = 0.0;
      for (int test_physical = 0; test_physical < test_physical_components;
           ++test_physical) {
        for (int trial_physical = 0;
             trial_physical < trial_physical_components; ++trial_physical) {
          value += TestField::Pullback(geometry, element, quadrature_point,
                                                 test_physical, test,
                                                 inverse, measure) *
                   CoefficientEntry(test_physical, trial_physical,
                                    test_physical_components, element,
                                    quadrature_point) *
                   TrialField::Pullback(geometry, element, quadrature_point,
                                                  trial_physical, trial,
                                                  inverse, measure);
        }
      }
      qdata(test, trial) = measure * value;
    }
  }
}

template <int ReferenceDimensions, BlfFieldOperator TrialOperator,
          mfem::FiniteElement::MapType TrialMap, BlfCoefficient CoefficientKind,
          BlfFieldOperator TestOperator, mfem::FiniteElement::MapType TestMap,
          bool IsMixed, int TrialVDim, int TestVDim,
          bool TrialOutOfPlane, bool TestOutOfPlane>
void BilinearFormOperator<ReferenceDimensions, TrialOperator, TrialMap,
                     CoefficientKind, TestOperator, TestMap, IsMixed,
                     TrialVDim, TestVDim, TrialOutOfPlane, TestOutOfPlane>::VerifyCoefficientDimensions(int space_dimensions) const {
  int rows = internal::PhysicalFieldComponents(
      TestOperator, TestMap, Dims, space_dimensions, TestVDim,
      TestOutOfPlane);
  int columns = internal::PhysicalFieldComponents(
      TrialOperator, TrialMap, Dims, space_dimensions, TrialVDim,
      TrialOutOfPlane);
  if constexpr (CoefficientKind == BlfCoefficient::KroneckerTensor ||
                CoefficientKind == BlfCoefficient::MinorSymmetricTensor ||
                CoefficientKind == BlfCoefficient::MinorMajorSymmetricTensor ||
                CoefficientKind == BlfCoefficient::IsotropicSymmetricTensor) {
    MFEM_VERIFY(tensor_dimensions == space_dimensions,
                "tensor coefficient Cartesian dimension does not match mesh");
  }
  if constexpr (CoefficientKind == BlfCoefficient::MinorSymmetricTensor ||
                CoefficientKind == BlfCoefficient::MinorMajorSymmetricTensor ||
                CoefficientKind == BlfCoefficient::IsotropicSymmetricTensor) {
    MFEM_VERIFY(rows == space_dimensions * space_dimensions &&
                    columns == space_dimensions * space_dimensions,
                "elasticity requires one field component per Cartesian axis");
  }
  if constexpr (CoefficientKind == BlfCoefficient::SymmetricMatrix ||
                CoefficientKind == BlfCoefficient::MajorSymmetricTensor) {
    MFEM_VERIFY(rows == columns,
                "coefficient action requires equal physical field sizes");
  }
  if constexpr (CoefficientKind == BlfCoefficient::Scalar) {
    MFEM_VERIFY(scalar_coefficient, "missing scalar coefficient");
  } else if constexpr (CoefficientKind == BlfCoefficient::Vector ||
                       CoefficientKind == BlfCoefficient::CrossProduct ||
                       CoefficientKind == BlfCoefficient::DiagonalMatrix) {
    MFEM_VERIFY(vector_coefficient, "missing vector coefficient");
    if constexpr (CoefficientKind == BlfCoefficient::CrossProduct) {
      MFEM_VERIFY(rows <= 3 && columns <= 3,
                  "cross-product action requires Cartesian vector fields");
    }
    const int vector_dimensions =
        CoefficientKind == BlfCoefficient::Vector &&
                (TrialRank == 2 || TestRank == 2)
            ? space_dimensions
            : CoefficientKind == BlfCoefficient::CrossProduct
                  ? 3
                  : std::max(rows, columns);
    MFEM_VERIFY(vector_coefficient->GetVDim() == vector_dimensions,
                "vector coefficient has the wrong dimension");
  } else if constexpr (UsesMatrixCoefficient) {
    MFEM_VERIFY(matrix_coefficient, "missing matrix coefficient");
    MFEM_VERIFY(matrix_coefficient->GetHeight() == rows &&
                    matrix_coefficient->GetWidth() == columns,
                "matrix coefficient has the wrong dimensions");
  }
}

template <int ReferenceDimensions, BlfFieldOperator TrialOperator,
          mfem::FiniteElement::MapType TrialMap, BlfCoefficient CoefficientKind,
          BlfFieldOperator TestOperator, mfem::FiniteElement::MapType TestMap,
          bool IsMixed, int TrialVDim, int TestVDim,
          bool TrialOutOfPlane, bool TestOutOfPlane>
typename BilinearFormOperator<ReferenceDimensions, TrialOperator, TrialMap,
                     CoefficientKind, TestOperator, TestMap, IsMixed,
                     TrialVDim, TestVDim, TrialOutOfPlane, TestOutOfPlane>::PreparedQData BilinearFormOperator<ReferenceDimensions, TrialOperator, TrialMap,
                     CoefficientKind, TestOperator, TestMap, IsMixed,
                     TrialVDim, TestVDim, TrialOutOfPlane, TestOutOfPlane>::PrepareQData(const mfem::FiniteElementSpace &trial_space,
                             const mfem::FiniteElementSpace &test_space,
                             const mfem::IntegrationRule &integration_rule,
                             bool use_device) {
  MFEM_VERIFY(trial_space.GetMesh() == test_space.GetMesh(),
              "trial and test spaces must use the same mesh");
  MFEM_VERIFY(trial_space.GetVDim() == TrialVDim &&
                  test_space.GetVDim() == TestVDim,
              "finite-element space vdim does not match the operation");
  const int space_dimensions = trial_space.GetMesh()->SpaceDimension();
  VerifyCoefficientDimensions(space_dimensions);

  BlfCoefficientStorage::DeviceView coefficient_view{nullptr, 0, 0, true};
  if constexpr (CoefficientKind == BlfCoefficient::Scalar) {
    coefficient_view = coefficient_storage.Prepare(
        *scalar_coefficient, trial_space, integration_rule, use_device,
        mfem::CoefficientStorage::CONSTANTS);
  } else if constexpr (CoefficientKind == BlfCoefficient::Vector ||
                       CoefficientKind == BlfCoefficient::CrossProduct ||
                       CoefficientKind == BlfCoefficient::DiagonalMatrix) {
    coefficient_view = coefficient_storage.Prepare(
        *vector_coefficient, trial_space, integration_rule, use_device,
        mfem::CoefficientStorage::CONSTANTS);
  } else if constexpr (UsesMatrixCoefficient) {
    // Store symmetric matrices densely so all matrix coefficient kinds have
    // the same device-side indexing.
    coefficient_view = coefficient_storage.Prepare(
        *matrix_coefficient, trial_space, integration_rule, use_device,
        mfem::CoefficientStorage::CONSTANTS);
  }
  return {coefficient_view,
          geometry_storage.Prepare(trial_space, integration_rule, use_device)};
}

template <int ReferenceDimensions, BlfFieldOperator TrialOperator,
          mfem::FiniteElement::MapType TrialMap, BlfCoefficient CoefficientKind,
          BlfFieldOperator TestOperator, mfem::FiniteElement::MapType TestMap,
          bool IsMixed, int TrialVDim, int TestVDim,
          bool TrialOutOfPlane, bool TestOutOfPlane>
constexpr int BilinearFormOperator<ReferenceDimensions, TrialOperator, TrialMap,
                     CoefficientKind, TestOperator, TestMap, IsMixed,
                     TrialVDim, TestVDim, TrialOutOfPlane, TestOutOfPlane>::PAComponents(int space_dimensions) {
  return internal::BlfPADataLayout<PAStorage,
      internal::BlfIntegratorTraits<BilinearFormOperator>>::Components(space_dimensions);
}

template <int ReferenceDimensions, BlfFieldOperator TrialOperator,
          mfem::FiniteElement::MapType TrialMap, BlfCoefficient CoefficientKind,
          BlfFieldOperator TestOperator, mfem::FiniteElement::MapType TestMap,
          bool IsMixed, int TrialVDim, int TestVDim,
          bool TrialOutOfPlane, bool TestOutOfPlane>
MFEM_HOST_DEVICE void BilinearFormOperator<ReferenceDimensions, TrialOperator, TrialMap,
                     CoefficientKind, TestOperator, TestMap, IsMixed,
                     TrialVDim, TestVDim, TrialOutOfPlane, TestOutOfPlane>::PreparedPAData::AssemblePAData(int element, int point,
                                         mfem::real_t *data,
                                         mfem::real_t weight) const {
  const int space_dimensions = base.geometry.space_dimensions;
  const mfem::real_t scale = weight * base.geometry.Determinant(element, point);
  mfem::real_t inverse[9] = {};
  base.geometry.InverseMetric(element, point, inverse);
  mfem::real_t pullback[9] = {};
  for (int reference = 0; reference < Dims; ++reference) {
    for (int physical = 0; physical < space_dimensions; ++physical) {
      for (int direction = 0; direction < Dims; ++direction) {
        pullback[physical + space_dimensions * reference] +=
            base.geometry.Jacobian(element, point, physical, direction) *
            inverse[direction + Dims * reference];
      }
    }
  }
  if constexpr (PAStorage == BlfPAStorage::Kronecker) {
    for (int test = 0; test < TestVDim; ++test) {
      const int row = TestOutOfPlane ? 2 : test;
      for (int trial = 0; trial < TrialVDim; ++trial) {
        const int column = TrialOutOfPlane ? 2 : trial;
        data[trial + TrialVDim * test] = diagonal_field
            ? (row == column ? base.coefficient(row, element, point) : 0.0)
            : base.coefficient(row + field_rows * column, element, point);
      }
    }
    for (int test = 0; test < Dims; ++test) {
      for (int trial = 0; trial < Dims; ++trial) {
        mfem::real_t value = 0.0;
        for (int row = 0; row < space_dimensions; ++row) {
          for (int column = 0; column < space_dimensions; ++column) {
            const mfem::real_t direction = identity_directions
                ? (row == column ? 1.0 : 0.0)
                : factor(row + space_dimensions * column, element, point);
            value += pullback[row + space_dimensions * test] * direction *
                     pullback[column + space_dimensions * trial];
          }
        }
        data[TrialVDim * TestVDim + trial + Dims * test] = scale * value;
      }
    }
  } else {
    for (int entry = 0; entry < space_dimensions * Dims; ++entry) {
      data[entry] = pullback[entry];
    }
    auto *material = data + space_dimensions * Dims;
    if constexpr (PAStorage == BlfPAStorage::IsotropicSymmetric) {
      material[0] = scale * base.coefficient(0, element, point);
      material[1] = scale * factor(0, element, point);
    } else {
      const int kelvin = space_dimensions * (space_dimensions + 1) / 2;
      for (int column = 0; column < kelvin; ++column) {
        for (int row = 0; row < kelvin; ++row) {
          if constexpr (PAStorage == BlfPAStorage::SymmetricKelvin) {
            if (row > column) { continue; }
            material[row + column * (column + 1) / 2] =
                scale * base.coefficient(row + kelvin * column, element, point);
          } else {
            material[column + kelvin * row] =
                scale * base.coefficient(row + kelvin * column, element, point);
          }
        }
      }
    }
  }
}

template <int ReferenceDimensions, BlfFieldOperator TrialOperator,
          mfem::FiniteElement::MapType TrialMap, BlfCoefficient CoefficientKind,
          BlfFieldOperator TestOperator, mfem::FiniteElement::MapType TestMap,
          bool IsMixed, int TrialVDim, int TestVDim,
          bool TrialOutOfPlane, bool TestOutOfPlane>
typename BilinearFormOperator<ReferenceDimensions, TrialOperator, TrialMap,
                     CoefficientKind, TestOperator, TestMap, IsMixed,
                     TrialVDim, TestVDim, TrialOutOfPlane, TestOutOfPlane>::PreparedPAData BilinearFormOperator<ReferenceDimensions, TrialOperator, TrialMap,
                     CoefficientKind, TestOperator, TestMap, IsMixed,
                     TrialVDim, TestVDim, TrialOutOfPlane, TestOutOfPlane>::PreparePAData(const mfem::FiniteElementSpace &trial_space,
                               const mfem::FiniteElementSpace &test_space,
                               const mfem::IntegrationRule &rule,
                               bool use_device)
    requires(PAStorage == BlfPAStorage::Kronecker ||
             PAStorage == BlfPAStorage::Kelvin ||
             PAStorage == BlfPAStorage::SymmetricKelvin ||
             PAStorage == BlfPAStorage::IsotropicSymmetric) {
  MFEM_VERIFY(trial_space.GetMesh() == test_space.GetMesh() &&
                  trial_space.GetVDim() == TrialVDim &&
                  test_space.GetVDim() == TestVDim,
              "PA spaces must match the declared operation");
  VerifyCoefficientDimensions(trial_space.GetMesh()->SpaceDimension());
  BlfCoefficientStorage::DeviceView first{nullptr, 0, 0, true};
  BlfCoefficientStorage::DeviceView second{nullptr, 0, 0, true};
  int rows = 0;
  bool diagonal = false;
  bool identity = false;
  constexpr auto storage = mfem::CoefficientStorage::CONSTANTS;
  if constexpr (PAStorage == BlfPAStorage::Kronecker) {
    auto &tensor = *static_cast<BlfKroneckerTensorCoefficient *>(matrix_coefficient);
    diagonal = tensor.FieldDiagonal() != nullptr;
    rows = tensor.GetHeight() / tensor.Dimension();
    if (diagonal) {
      first = coefficient_storage.Prepare(*tensor.FieldDiagonal(), trial_space,
                                          rule, use_device, storage);
    } else {
      first = coefficient_storage.Prepare(*tensor.FieldMatrix(), trial_space,
                                          rule, use_device, storage);
    }
    identity = tensor.Directions() == nullptr;
    if (!identity) {
      second = pa_factor_storage.Prepare(*tensor.Directions(), trial_space,
                                         rule, use_device, storage);
    }
  } else if constexpr (PAStorage == BlfPAStorage::IsotropicSymmetric) {
    auto &tensor = *static_cast<BlfIsotropicSymmetricTensorCoefficient *>(matrix_coefficient);
    first = coefficient_storage.Prepare(tensor.Lambda(), trial_space, rule,
                                        use_device, storage);
    second = pa_factor_storage.Prepare(tensor.Mu(), trial_space, rule,
                                       use_device, storage);
  } else if constexpr (PAStorage == BlfPAStorage::Kelvin) {
    auto &tensor = *static_cast<BlfMinorSymmetricTensorCoefficient *>(matrix_coefficient);
    first = coefficient_storage.Prepare(tensor.Constitutive(), trial_space,
                                        rule, use_device, storage);
  } else {
    auto &tensor = *static_cast<BlfMinorMajorSymmetricTensorCoefficient *>(matrix_coefficient);
    first = coefficient_storage.Prepare(tensor.Constitutive(), trial_space,
                                        rule, use_device, storage);
  }
  return {{first, geometry_storage.Prepare(trial_space, rule, use_device)},
          second, rows, diagonal, identity};
}

template <int ReferenceDimensions, BlfFieldOperator TrialOperator,
          mfem::FiniteElement::MapType TrialMap, BlfCoefficient CoefficientKind,
          BlfFieldOperator TestOperator, mfem::FiniteElement::MapType TestMap,
          bool IsMixed, int TrialVDim, int TestVDim,
          bool TrialOutOfPlane, bool TestOutOfPlane>
void BilinearFormOperator<ReferenceDimensions, TrialOperator, TrialMap,
                     CoefficientKind, TestOperator, TestMap, IsMixed,
                     TrialVDim, TestVDim, TrialOutOfPlane, TestOutOfPlane>::AssembleQDataHost(mfem::ElementTransformation &transformation,
                         const mfem::IntegrationPoint &ip,
                         mfem::DenseMatrix &qdata) {
  VerifyCoefficientDimensions(transformation.GetSpaceDim());
  mfem::Vector coefficient_values;
  if constexpr (CoefficientKind == BlfCoefficient::Scalar) {
    coefficient_values.SetSize(1);
    coefficient_values[0] = scalar_coefficient->Eval(transformation, ip);
  } else if constexpr (CoefficientKind == BlfCoefficient::Vector ||
                       CoefficientKind == BlfCoefficient::CrossProduct ||
                       CoefficientKind == BlfCoefficient::DiagonalMatrix) {
    vector_coefficient->Eval(coefficient_values, transformation, ip);
  } else if constexpr (UsesMatrixCoefficient) {
    mfem::DenseMatrix value(matrix_coefficient->GetHeight(),
                            matrix_coefficient->GetWidth());
    matrix_coefficient->Eval(value, transformation, ip);
    coefficient_values.SetSize(value.Height() * value.Width());
    std::copy(value.GetData(), value.GetData() + coefficient_values.Size(),
              coefficient_values.HostWrite());
  }

  const mfem::real_t *jacobian = nullptr;
  if constexpr (NeedsJacobian) {
    jacobian = transformation.Jacobian().GetData();
  }
  mfem::real_t determinant = 1.0;
  if constexpr (NeedsDeterminant) {
    determinant = transformation.Weight();
  }
  typename BlfGeometryFactors<GeometryFlags, Dims>::DeviceView geometry{
      nullptr, jacobian, NeedsDeterminant ? &determinant : nullptr, 1,
      transformation.GetSpaceDim()};
  PreparedQData prepared{
      {coefficient_values.HostRead(), 1, coefficient_values.Size(), true},
      geometry};
  mfem::real_t row_major_qdata[TestComponents * TrialComponents] = {};
  prepared.AssembleQData(
      0, 0, BlfQDataMatrix<TestComponents, TrialComponents>(row_major_qdata));
  for (int test = 0; test < TestComponents; ++test) {
    for (int trial = 0; trial < TrialComponents; ++trial) {
      qdata(test, trial) = row_major_qdata[trial + TrialComponents * test];
    }
  }
}
} // namespace mfem

#endif
