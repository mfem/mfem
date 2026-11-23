#ifndef LINEARINTEGS_HPP
#define LINEARINTEGS_HPP

#include "bilinearintegs.hpp"
#include "linearintegs_impl.hpp"

#include <type_traits>
#include <utility>

namespace mfem {

/// Coordinate-driven quadrature operation for a standalone domain linear form.
/// @tparam Function Trivially copyable host/device callable. Optional geometry
/// flags are detected directly on this type; see LinearFormQuadraturePoint.
/// Physical output is pulled back and multiplied by geometric measure.
/// With Function::reference_output enabled, return measure-weighted reference
/// channels instead. Neither convention includes the quadrature rule weight.
template <int ReferenceDimensions, BlfFieldOperator TestOperator,
          mfem::FiniteElement::MapType TestMap, class Function,
          int TestVDim = 1, bool TestOutOfPlane = false>
struct LinearFormOperator {
  constexpr static int Dims = ReferenceDimensions;
  constexpr static auto test_operator = TestOperator;
  constexpr static auto test_type = TestMap;
  constexpr static int test_vdim = TestVDim;
  constexpr static bool test_out_of_plane = TestOutOfPlane;
  using TestField = internal::BlfFieldDescriptor<Dims, TestOperator, TestMap,
                                                 TestVDim, TestOutOfPlane>;
  constexpr static int TestComponents = TestField::Components;
  constexpr static int MaxPhysicalComponents = TestField::PhysicalComponents(3);

  using QuadraturePoint = LinearFormQuadraturePoint<Dims, Function>;
  constexpr static bool FunctionNeedsJacobian = internal::Extractjacobian<Function>::value ||
      internal::Extractmetric<Function>::value ||
      internal::Extractinverse_metric<Function>::value ||
      internal::Extractadjugate<Function>::value;
  constexpr static int GeometryFlags =
      mfem::GeometricFactors::COORDINATES | mfem::GeometricFactors::DETERMINANTS |
      ((FunctionNeedsJacobian ||
        (!internal::Extractreference_output<Function>::value && TestField::NeedsJacobian))
           ? mfem::GeometricFactors::JACOBIANS : 0);

  static_assert(std::is_trivially_copyable_v<Function>,
                "quadrature function must be trivially copyable for device assembly");
  static_assert(std::is_invocable_r_v<void, const Function &, const QuadraturePoint &,
                                     mfem::real_t *> ||
                std::is_invocable_r_v<void, const Function &, const mfem::real_t *,
                                     int, mfem::real_t *>,
                "quadrature function must accept (point, values) or "
                "(coordinates, space_dimension, values)");
  static_assert(!internal::Extractreference_output<Function>::value ||
                std::is_invocable_v<const Function &, const QuadraturePoint &,
                                    mfem::real_t *>,
                "reference-output quadrature functions must accept the point context");

  Function function;
  BlfGeometryFactors<GeometryFlags, Dims> geometry_storage;

  explicit LinearFormOperator(Function function) : function(std::move(function)) {}

  struct PreparedQData {
    Function function;
    typename BlfGeometryFactors<GeometryFlags, Dims>::DeviceView geometry;

    /// Evaluate the original q-function without pullbacks or integration weights.
    /// @param element Local mesh element index.
    /// @param point Index in the prepared quadrature rule.
    /// @param[out] values Storage for every component written by Function.
    /// @param element_number Optional element index exposed to the q-function.
    /// @param point_number Optional quadrature index exposed to the q-function.
    /// @param inverse Optional precomputed column-major inverse metric.
    MFEM_HOST_DEVICE void EvaluateQData(int element, int point,
                                       mfem::real_t *values,
                                       int element_number = -1,
                                       int point_number = -1,
                                       const mfem::real_t *inverse = nullptr) const {
      mfem::real_t inverse_storage[9] = {};
      if constexpr (internal::Extractinverse_metric<Function>::value) {
        if (!inverse) {
          geometry.InverseMetric(element, point, inverse_storage);
          inverse = inverse_storage;
        }
      }
      const mfem::real_t measure = geometry.Determinant(element, point);
      QuadraturePoint context;
      context.space_dimension = geometry.space_dimensions;
      context.element = element_number < 0 ? element : element_number;
      context.quadrature_point = point_number < 0 ? point : point_number;
      context.measure = measure;
      for (int component = 0; component < geometry.space_dimensions; ++component) {
        context.coordinates[component] = geometry.Coordinate(element, point, component);
      }
      if constexpr (internal::Extractjacobian<Function>::value) {
        for (int column = 0; column < Dims; ++column) {
          for (int row = 0; row < geometry.space_dimensions; ++row) {
            context.jacobian[row + geometry.space_dimensions * column] =
                geometry.Jacobian(element, point, row, column);
          }
        }
      }
      if constexpr (internal::Extractmetric<Function>::value) {
        geometry.Metric(element, point, context.metric.data());
      }
      if constexpr (internal::Extractinverse_metric<Function>::value) {
        for (int index = 0; index < Dims * Dims; ++index) {
          context.inverse_metric[index] = inverse[index];
        }
      }
      if constexpr (internal::Extractadjugate<Function>::value) {
        if constexpr (Dims == 1) {
          context.adjugate[0] = 1.0;
        } else if constexpr (Dims == 2) {
          context.adjugate = {geometry.Jacobian(element, point, 1, 1),
                             -geometry.Jacobian(element, point, 1, 0),
                             -geometry.Jacobian(element, point, 0, 1),
                             geometry.Jacobian(element, point, 0, 0)};
        } else {
          for (int column = 0; column < Dims; ++column) {
            for (int row = 0; row < Dims; ++row) {
              context.adjugate[row + Dims * column] =
                  geometry.Jacobian(element, point, (column + 1) % 3, (row + 1) % 3) *
                  geometry.Jacobian(element, point, (column + 2) % 3, (row + 2) % 3) -
                  geometry.Jacobian(element, point, (column + 1) % 3, (row + 2) % 3) *
                  geometry.Jacobian(element, point, (column + 2) % 3, (row + 1) % 3);
            }
          }
        }
      }
      if constexpr (std::is_invocable_v<const Function &, const QuadraturePoint &,
                                        mfem::real_t *>) {
        function(context, values);
      } else {
        function(context.coordinates.data(), context.space_dimension, values);
      }
    }

    MFEM_HOST_DEVICE void AssembleQData(int element, int point,
                                       mfem::real_t *qdata,
                                       int element_number = -1,
                                       int point_number = -1) const {
      mfem::real_t inverse[9] = {};
      if constexpr ((!internal::Extractreference_output<Function>::value &&
                     TestField::NeedsInverseMetric) ||
                    internal::Extractinverse_metric<Function>::value) {
        geometry.InverseMetric(element, point, inverse);
      }
      const mfem::real_t measure = geometry.Determinant(element, point);
      if constexpr (internal::Extractreference_output<Function>::value) {
        EvaluateQData(element, point, qdata, element_number, point_number, inverse);
        return;
      }
      mfem::real_t physical[MaxPhysicalComponents] = {};
      EvaluateQData(element, point, physical, element_number, point_number, inverse);
      const int physical_components = TestField::PhysicalComponents(geometry.space_dimensions);
      for (int reference = 0; reference < TestComponents; ++reference) {
        mfem::real_t value = 0.0;
        for (int component = 0; component < physical_components; ++component) {
          value += physical[component] *
              TestField::Pullback(geometry, element, point, component, reference, inverse, measure);
        }
        qdata[reference] = measure * value;
      }
    }
  };

  PreparedQData PrepareQData(const mfem::FiniteElementSpace &space,
                             const mfem::IntegrationRule &rule, bool use_device) {
    if constexpr (internal::Extractadjugate<Function>::value) {
      MFEM_VERIFY(space.GetMesh()->SpaceDimension() == Dims,
                  "adjugate requires a square mesh Jacobian");
    }
    return {function, geometry_storage.Prepare(space, rule, use_device)};
  }

  void AssembleQDataHost(mfem::ElementTransformation &transformation,
                        const mfem::IntegrationPoint &point,
                        mfem::Vector &qdata) const {
    MFEM_VERIFY(transformation.GetDimension() == Dims &&
                    transformation.GetSpaceDim() >= Dims &&
                    transformation.GetSpaceDim() <= 3,
                "element transformation has incompatible dimensions");
    transformation.SetIntPoint(&point);
    mfem::Vector coordinates(transformation.GetSpaceDim());
    transformation.Transform(point, coordinates);
    const mfem::real_t measure = transformation.Weight();
    const mfem::real_t *jacobian = nullptr;
    if constexpr ((!internal::Extractreference_output<Function>::value && TestField::NeedsJacobian) ||
                  FunctionNeedsJacobian) {
      jacobian = transformation.Jacobian().GetData();
    }
    if constexpr (internal::Extractadjugate<Function>::value) {
      MFEM_VERIFY(transformation.GetSpaceDim() == Dims,
                  "adjugate requires a square mesh Jacobian");
    }
    PreparedQData prepared{
        function, {coordinates.HostRead(), jacobian, &measure, 1,
                   transformation.GetSpaceDim()}};
    qdata.SetSize(TestComponents);
    prepared.AssembleQData(0, 0, qdata.HostWrite(), transformation.ElementNo, point.index);
  }
};

/// Build an operation by deducing its quadrature-function type and geometry flags.
/// @param function Owned by value; mutate operation.function between assemblies.
/// No separate geometry-traits template argument is accepted.
template <int Dims, BlfFieldOperator TestOperator,
          mfem::FiniteElement::MapType TestMap, int TestVDim = 1,
          bool TestOutOfPlane = false, class Function>
auto MakeLinearFormOperator(Function function) {
  return LinearFormOperator<Dims, TestOperator, TestMap, Function,
                            TestVDim, TestOutOfPlane>(std::move(function));
}
} // namespace mfem

#endif
