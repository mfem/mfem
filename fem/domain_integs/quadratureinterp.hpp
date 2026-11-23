#ifndef QUADRATUREINTERP_HPP
#define QUADRATUREINTERP_HPP

#include "linearintegs.hpp"
#include "quadratureinterp_impl.hpp"

#include <utility>

namespace mfem {

/// Coordinate-driven q-function evaluation without linear-form integration.
/// The callable and its optional geometry flags are exactly those accepted by
/// LinearFormOperator. Returned values are stored unchanged, including when
/// Function::reference_output is true: no additional measure, quadrature weight,
/// physical pullback, or test-basis contraction is applied.
/// @tparam ReferenceDimensions Mesh topological dimension, from 1 through 3.
/// @tparam Function Trivially copyable host/device callable taking either
/// (LinearFormQuadraturePoint<ReferenceDimensions, Function>, values) or
/// (coordinates, space_dimension, values).
/// @tparam OutputComponents Number of values written by the callable per point.
template <int ReferenceDimensions, class Function, int OutputComponents = 1>
struct QuadratureInterpolationOperator
    : LinearFormOperator<ReferenceDimensions, BlfFieldOperator::Value,
                         mfem::FiniteElement::VALUE, Function> {
  using Base = LinearFormOperator<ReferenceDimensions, BlfFieldOperator::Value,
                                 mfem::FiniteElement::VALUE, Function>;
  using PreparedQData = typename Base::PreparedQData;
  constexpr static int Components = OutputComponents;

  static_assert(Components > 0, "quadrature output component count must be positive");

  /// @param function Callable owned by value; mutate this->function to reevaluate.
  explicit QuadratureInterpolationOperator(Function function)
      : Base(std::move(function)) {}

  /// Prepare the same coordinate/geometry context used by linear forms.
  /// @param mesh Mesh supplying the coordinate field, including curved geometry.
  /// @param rule Common element quadrature rule.
  /// @param use_device Select device-accessible versus host read pointers.
  /// @return Borrowed mesh geometry and a snapshot of the current callable.
  PreparedQData PrepareQData(mfem::Mesh &mesh,
                            const mfem::IntegrationRule &rule, bool use_device) {
    if constexpr (internal::Extractadjugate<Function>::value) {
      MFEM_VERIFY(mesh.SpaceDimension() == ReferenceDimensions,
                  "adjugate requires a square mesh Jacobian");
    }
    return {this->function, this->geometry_storage.Prepare(mesh, rule, use_device)};
  }
};

/// Deduce the callable type for pointwise quadrature interpolation.
/// @tparam Dims Mesh reference dimension.
/// @tparam Components Number of values written per quadrature point.
/// @tparam Function Callable type, deduced from function.
/// @param function Callable owned by the returned operation.
template <int Dims, int Components = 1, class Function>
auto MakeQuadratureInterpolationOperator(Function function) {
  return QuadratureInterpolationOperator<Dims, Function, Components>(
      std::move(function));
}
} // namespace mfem

#endif
