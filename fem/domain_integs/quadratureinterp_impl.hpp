#ifndef QUADRATUREINTERP_IMPL_HPP
#define QUADRATUREINTERP_IMPL_HPP

#include "linearintegs.hpp"

#include <type_traits>
#include <utility>

namespace mfem {

/// Evaluate a coordinate-driven q-function into an MFEM QuadratureFunction.
/// This interpolates the mesh coordinate field and evaluates a callable, not
/// a solution GridFunction. No test finite-element space or basis is needed.
/// @tparam T Operation with Dims, Components, and PrepareQData(mesh, rule,
/// use_device). Its trivially copyable result provides a host/device
/// EvaluateQData(element, point, values) method.
template <class T>
class DomainQuadratureInterpolator {
  T &operation_;

  using PreparedQData = decltype(std::declval<T &>().PrepareQData(
      std::declval<mfem::Mesh &>(), std::declval<const mfem::IntegrationRule &>(),
      false));
  static_assert(T::Dims >= 1 && T::Dims <= 3);
  static_assert(T::Components > 0);
  static_assert(std::is_trivially_copyable_v<PreparedQData>,
                "prepared quadrature operation must be trivially copyable");

public:
  /// @param operation Borrowed operation; it must outlive the interpolator.
  explicit DomainQuadratureInterpolator(T &operation) : operation_(operation) {}

  /// Overwrite every value at the output space's element quadrature points.
  /// @param[out] output Initialized quadrature function with T::Components vdim.
  /// Its space must be an element QuadratureSpace on a homogeneous mesh with
  /// one shared rule. Faces and mixed element geometries are not supported.
  /// Empty local meshes are a no-op; no geometry preparation/kernel is launched.
  void Interpolate(mfem::QuadratureFunction &output) {
    auto *space = dynamic_cast<mfem::QuadratureSpace *>(output.GetSpace());
    MFEM_VERIFY(space, "quadrature interpolation requires an element QuadratureSpace");
    auto &mesh = *space->GetMesh();
    MFEM_VERIFY(mesh.Dimension() == T::Dims &&
                    mesh.SpaceDimension() >= T::Dims && mesh.SpaceDimension() <= 3,
                "mesh has incompatible quadrature interpolation dimensions");
    MFEM_VERIFY(output.GetVDim() == T::Components &&
                    output.Size() == T::Components * space->GetSize(),
                "quadrature output has the wrong component count or size");
    if (mesh.GetNE() == 0) { return; }

    const auto geometry = mesh.GetTypicalElementGeometry();
    const auto &rule = space->GetIntRule(0);
    for (int element = 0; element < mesh.GetNE(); ++element) {
      MFEM_VERIFY(mesh.GetElementGeometry(element) == geometry &&
                      &space->GetIntRule(element) == &rule,
                  "quadrature interpolation requires homogeneous geometry and one rule");
    }
    const bool use_device =
        mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                             mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP);
    const auto prepared = operation_.PrepareQData(mesh, rule, use_device);
    output.UseDevice(true);
    auto *data = output.Write(use_device);
    const int points = rule.GetNPoints();
    mfem::forall(points, mesh.GetNE(), [=] MFEM_HOST_DEVICE(int point, int element) {
      mfem::real_t values[T::Components] = {};
      prepared.EvaluateQData(element, point, values);
      for (int component = 0; component < T::Components; ++component) {
        data[component + T::Components * (point + points * element)] = values[component];
      }
    });
  }

  /// Allocate and evaluate an output borrowing the supplied quadrature space.
  /// @param space Homogeneous element quadrature space; must outlive the result.
  /// @return QuadratureFunction with T::Components values per point.
  mfem::QuadratureFunction Interpolate(mfem::QuadratureSpace &space) {
    mfem::QuadratureFunction output(space, T::Components);
    Interpolate(output);
    return output;
  }
};
} // namespace mfem

#endif
