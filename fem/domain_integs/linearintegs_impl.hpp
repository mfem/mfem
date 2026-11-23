#ifndef LINEARINTEGS_IMPL_HPP
#define LINEARINTEGS_IMPL_HPP

#include "formintegs_impl.hpp"
#include "bilinearintegs_impl.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>

namespace mfem {

/// Geometry passed to a coordinate-driven quadrature function.
/// @tparam Dims Mesh reference dimension, from 1 through 3.
/// @tparam T Quadrature-function type declaring optional constexpr static bool
/// jacobian, metric, inverse_metric, adjugate, and reference_output members.
/// Missing flags default to false. Disabled matrices have zero-length storage
/// and unavailable accessors; enabled matrices use column-major storage.
template <int Dims, class T>
struct LinearFormQuadraturePoint {
  constexpr static int reference_dimension = Dims;
  std::array<mfem::real_t, 3> coordinates{};
  int space_dimension;
  int element;
  int quadrature_point;
  mfem::real_t measure;
  std::array<mfem::real_t, internal::Extractjacobian<T>::value ? 3 * Dims : 0> jacobian{};
  std::array<mfem::real_t, internal::Extractmetric<T>::value ? Dims * Dims : 0> metric{};
  std::array<mfem::real_t, internal::Extractinverse_metric<T>::value ? Dims * Dims : 0> inverse_metric{};
  std::array<mfem::real_t, internal::Extractadjugate<T>::value ? Dims * Dims : 0> adjugate{};

  MFEM_HOST_DEVICE mfem::real_t Jacobian(int row, int column) const
    requires(internal::Extractjacobian<T>::value) { return jacobian[row + space_dimension * column]; }
  MFEM_HOST_DEVICE mfem::real_t Metric(int row, int column) const
    requires(internal::Extractmetric<T>::value) { return metric[row + Dims * column]; }
  MFEM_HOST_DEVICE mfem::real_t InverseMetric(int row, int column) const
    requires(internal::Extractinverse_metric<T>::value) { return inverse_metric[row + Dims * column]; }
  MFEM_HOST_DEVICE mfem::real_t Adjugate(int row, int column) const
    requires(internal::Extractadjugate<T>::value) { return adjugate[row + Dims * column]; }
};

namespace internal {

template <class T> struct LfIntegratorTraits {
  constexpr static int Dims = T::Dims;
  constexpr static auto test_operator = T::test_operator;
  constexpr static auto test_type = T::test_type;
  constexpr static int TestVDim = ExtractTestVDim<T>();
  constexpr static bool TestOutOfPlane = ExtractTestOutOfPlane<T>();
  using TestField = BlfFieldDescriptor<Dims, test_operator, test_type,
                                       TestVDim, TestOutOfPlane>;
  constexpr static auto TestBasisOperator = TestField::BasisOperator;
  constexpr static int TestComponents = TestField::Components;

  static_assert(Dims >= 1 && Dims <= 3);
  static_assert(IsValidFieldOperator(test_operator, test_type, TestVDim,
                                    TestOutOfPlane));
  static_assert(!IsReducedVectorMap(test_type));
};

template <class Traits>
struct LfStoredQData {
  const mfem::real_t *data;
  int nq;

  MFEM_HOST_DEVICE mfem::real_t Value(int element, int point, int channel) const {
    return data[point + nq * (channel + Traits::TestComponents * element)];
  }
};

template <class Traits, class PreparedQData> struct LfMatrixFreeQData {
  PreparedQData prepared;
  const mfem::real_t *weights;

  MFEM_HOST_DEVICE void Evaluate(int element, int point, mfem::real_t *values) const {
    prepared.AssembleQData(element, point, values);
    for (int channel = 0; channel < Traits::TestComponents; ++channel) {
      values[channel] *= weights[point];
    }
  }

  MFEM_HOST_DEVICE mfem::real_t Value(int element, int point, int channel) const {
    mfem::real_t values[Traits::TestComponents] = {};
    Evaluate(element, point, values);
    return values[channel];
  }
};

template <class Traits, class QData> struct LfTensorAssembleFunctor {
  const mfem::real_t *closed_basis;
  const mfem::real_t *open_basis;
  const mfem::real_t *gradient_basis;
  QData qdata;
  const int *markers;
  const int *dof_map;
  mfem::real_t *output;
  int d1d, o1d, q1d, scalar_dofs, dofs, buffer_size;

  MFEM_HOST_DEVICE void operator()(int begin, int end, char *workspace) const {
    auto *buffer0 = reinterpret_cast<mfem::real_t *>(workspace);
    auto *buffer1 = buffer0 + buffer_size;
    for (int element = begin; element < end; ++element) {
      if (markers && !markers[element]) { continue; }
      for (int channel = 0; channel < Traits::TestComponents; ++channel) {
        constexpr bool scalar = internal::IsScalarMap(Traits::test_type);
        constexpr auto basis_operator = Traits::TestBasisOperator;
        constexpr int terms = scalar ? 1 : internal::TensorOperatorTermCount(
            basis_operator, Traits::Dims);
        for (int term_index = 0; term_index < terms; ++term_index) {
          internal::BlfTensorOperatorTerm term;
          if constexpr (scalar) {
            if constexpr (basis_operator == BlfFieldOperator::Gradient) {
              term = {channel / Traits::Dims, channel % Traits::Dims, 1.0};
            } else {
              term = {channel, -1, 1.0};
            }
          } else {
            term = internal::TensorOperatorTerm(basis_operator, Traits::Dims,
                                                 channel, term_index);
          }
          int axis_dofs[3] = {1, 1, 1};
          int component_dofs = 1;
          for (int axis = 0; axis < Traits::Dims; ++axis) {
            axis_dofs[axis] = internal::TensorFieldUsesClosedMap(
                Traits::test_type, term.field_component, axis) ? d1d : o1d;
            component_dofs *= axis_dofs[axis];
          }
          const mfem::real_t *source = nullptr;
          int preceding = 1;
          int following = internal::TensorIntPow(q1d, Traits::Dims - 1);
          for (int axis = 0; axis < Traits::Dims; ++axis) {
            const bool closed = internal::TensorFieldUsesClosedMap(
                Traits::test_type, term.field_component, axis);
            const auto *basis = term.derivative_dimension == axis
                                    ? gradient_basis
                                    : closed ? closed_basis : open_basis;
            const int nd = axis_dofs[axis];
            const int count = preceding * nd * following;
            auto *destination = axis % 2 == 0 ? buffer0 : buffer1;
            MFEM_FOREACH_THREAD(index, x, count) {
              const int before = index % preceding;
              const int dof = (index / preceding) % nd;
              const int after = index / (preceding * nd);
              mfem::real_t value = 0.0;
              for (int point = 0; point < q1d; ++point) {
                const int source_point = before + preceding * (point + q1d * after);
                value += basis[point + q1d * dof] *
                         (axis == 0 ? qdata.Value(element, source_point, channel)
                                    : source[source_point]);
              }
              if (axis + 1 == Traits::Dims) {
                const int lex_dof = term.field_component * component_dofs + index;
                const int field_offset = scalar ? (lex_dof / scalar_dofs) * scalar_dofs : 0;
                const int local_dof = scalar ? lex_dof % scalar_dofs : lex_dof;
                const int signed_dof = dof_map ? dof_map[local_dof] : local_dof;
                const int native_dof = signed_dof < 0 ? -1 - signed_dof : signed_dof;
                output[native_dof + field_offset + dofs * element] +=
                    (signed_dof < 0 ? -term.sign : term.sign) * value;
              } else {
                destination[index] = value;
              }
            }
            MFEM_SYNC_THREAD;
            source = destination;
            preceding *= nd;
            if (axis + 1 < Traits::Dims) { following /= q1d; }
          }
        }
      }
    }
  }
};

template <class Traits, class Prepared>
void AssembleLinearQData(int nq, int elements, const int *active,
                          Prepared prepared, mfem::real_t *data,
                          const mfem::real_t *weights) {
  mfem::forall(nq, elements, [active, prepared, data, weights,
                             nq] MFEM_HOST_DEVICE(int point, int index) {
    if (active && !active[index]) { return; }
    mfem::real_t values[Traits::TestComponents];
    prepared.AssembleQData(index, point, values);
    for (int channel = 0; channel < Traits::TestComponents; ++channel) {
      data[point + nq * (channel + Traits::TestComponents * index)] =
          weights[point] * values[channel];
    }
  });
}

template <bool MatrixFree, class Traits, class TestBasis, class QData>
void AssembleLinearFull(int dofs, int elements, int nq, const int *active,
                         QData qdata, const mfem::real_t *basis,
                         mfem::real_t *values) {
  mfem::forall(dofs, elements, [active, qdata, basis, values, nq,
                               dofs] MFEM_HOST_DEVICE(int dof, int index) {
    if (active && !active[index]) { return; }
    mfem::real_t value = 0.0;
    if constexpr (MatrixFree) {
      for (int point = 0; point < nq; ++point) {
        mfem::real_t point_values[Traits::TestComponents] = {};
        qdata.Evaluate(index, point, point_values);
        for (int channel = 0; channel < Traits::TestComponents; ++channel) {
          value += TestBasis::Eval(basis, point, channel, dof, nq) *
                   point_values[channel];
        }
      }
    } else {
      for (int channel = 0; channel < Traits::TestComponents; ++channel) {
        for (int point = 0; point < nq; ++point) {
          value += TestBasis::Eval(basis, point, channel, dof, nq) *
                   qdata.Value(index, point, channel);
        }
      }
    }
    values[dof + dofs * index] += value;
  });
}

}

/// Standalone assembly of integral F(mesh coordinates) dot test_operator(v) dx.
/// This is neither an MFEM LinearFormIntegrator nor a bilinear trial-vector
/// operator. The operation and integration rule are borrowed and must outlive
/// the integrator. Batches assume homogeneous, fixed-order finite elements;
/// individual element layouts are not checked.
/// @tparam T Operation supplying test-field metadata, PrepareQData, and
/// AssembleQDataHost, such as LinearFormOperator.
template <class T>
class DomainLinearFormIntegrator {
public:
  using lf_traits = internal::LfIntegratorTraits<T>;

private:
  using TestBasis = BlfReferenceBasis<
      lf_traits::TestBasisOperator, lf_traits::test_type,
      lf_traits::TestComponents, lf_traits::TestVDim>;

  T &integrator_;
  const mfem::IntegrationRule &integration_rule_;
  BlfTensorProductMode tensor_product_mode_;
  TestBasis test_basis_;
  mfem::Vector qdata_;
  mfem::Vector workspace_;
  mfem::Vector element_vectors_;

  static void VerifyElement(const mfem::FiniteElement &element) {
    MFEM_VERIFY(element.GetDim() == lf_traits::Dims,
                "test finite element has the wrong reference dimension");
    MFEM_VERIFY(element.GetMapType() == lf_traits::test_type,
                "test finite element has the wrong MapType");
  }
  using StoredQData = internal::LfStoredQData<lf_traits>;

  template <class PreparedQData>
  using MatrixFreeQData = internal::LfMatrixFreeQData<lf_traits, PreparedQData>;

  template <class QData>
  using TensorAssembleFunctor =
      internal::LfTensorAssembleFunctor<lf_traits, QData>;

  template <class QData>
  void AssembleTensor(const mfem::FiniteElement &element,
                      const mfem::Array<int> *markers, mfem::Vector &output,
                      mfem::ElementDofOrdering ordering, int elements,
                      const QData &qdata) {
    const auto &closed = element.GetDofToQuad(integration_rule_, mfem::DofToQuad::TENSOR);
    const mfem::DofToQuad *open = nullptr;
    if constexpr (internal::IsVectorTensorMap(lf_traits::test_type)) {
      const auto &vector = dynamic_cast<const mfem::VectorTensorFiniteElement &>(element);
      open = &vector.GetDofToQuadOpen(integration_rule_, mfem::DofToQuad::TENSOR);
    }
    const int buffer_size = internal::TensorIntPow(std::max(closed.ndof, closed.nqpt),
                                                  lf_traits::Dims);
    const std::size_t bytes = 2 * std::size_t(buffer_size) * sizeof(mfem::real_t);
    MFEM_VERIFY(bytes <= std::numeric_limits<int>::max(), "linear-form workspace too large");
    const int blocks =
        std::min(elements, forall_nested_max_concurrent_blocks());
    const bool global =
        mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                             mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP) &&
        bytes > forall_nested_shared_memory_limit;
    const std::size_t required = global
        ? forall_nested_global_workspace_size(elements, blocks, static_cast<int>(bytes)) : 0;
    MFEM_VERIFY(required / sizeof(mfem::real_t) <= std::numeric_limits<int>::max(),
                "linear-form global workspace too large");
    workspace_.SetSize(static_cast<int>((required + sizeof(mfem::real_t) - 1) /
                                        sizeof(mfem::real_t)));
    workspace_.UseDevice(true);
    const auto *map = ordering == mfem::ElementDofOrdering::NATIVE ? internal::LexicographicDofMap(element) : nullptr;
    const int dofs = element.GetDof() * lf_traits::TestVDim;
    TensorAssembleFunctor<QData> kernel{
        closed.B.Read(), open ? open->B.Read() : nullptr, closed.G.Read(),
        qdata, markers ? markers->Read() : nullptr,
        map && map->Size() ? map->Read() : nullptr, output.ReadWrite(),
        closed.ndof, open ? open->ndof : 0, closed.nqpt, element.GetDof(), dofs,
        buffer_size};
    forall_nested_workspace(kernel, elements, std::min(256, buffer_size), 1, 1,
                            blocks, static_cast<int>(bytes),
                            global ? reinterpret_cast<char *>(workspace_.Write()) : nullptr,
                            workspace_.Size() * sizeof(mfem::real_t));
  }

  template <bool MatrixFree = false>
  void AssembleElements(const mfem::FiniteElementSpace &space,
                const mfem::Array<int> *markers, mfem::Vector &output, bool add,
                mfem::ElementDofOrdering ordering) {
    MFEM_VERIFY(space.GetVDim() == lf_traits::TestVDim,
                "test space vdim does not match the linear form");
    MFEM_VERIFY(space.GetMesh()->Dimension() == lf_traits::Dims,
                "mesh dimension does not match the linear form");
    MFEM_VERIFY(!space.IsVariableOrder(), "variable-order linear assembly is not supported");
    const int elements = space.GetNE();
    MFEM_VERIFY(!markers || markers->Size() == elements,
                "linear-form element marker size is incorrect");
    if (!elements) {
      if constexpr (MatrixFree) { qdata_.Destroy(); }
      if (!add) { output.SetSize(0); }
      MFEM_VERIFY(output.Size() == 0, "empty linear-form output must be empty");
      return;
    }
    const auto &element = *space.GetFE(0);
    VerifyElement(element);
    const int dofs = element.GetDof() * lf_traits::TestVDim;
    if (add) {
      MFEM_VERIFY(output.Size() == dofs * elements, "linear-form output size is incorrect");
    } else {
      output.SetSize(dofs * elements);
      output.UseDevice(true);
      output = 0.0;
    }
    const int nq = integration_rule_.GetNPoints();
    const auto prepared = integrator_.PrepareQData(space, integration_rule_, true);
    static_assert(std::is_trivially_copyable_v<decltype(prepared)>);
    const auto *weights = integration_rule_.GetWeights().Read();
    const auto *active = markers ? markers->Read() : nullptr;
    if constexpr (MatrixFree) {
      qdata_.Destroy();
    } else {
      qdata_.SetSize(elements * nq * lf_traits::TestComponents);
      qdata_.UseDevice(true);
      auto *data = qdata_.Write();
      internal::AssembleLinearQData<lf_traits>(
          nq, elements, active, prepared, data, weights);
    }
    const auto qdata = [this, prepared, weights, nq]() {
      if constexpr (MatrixFree) {
        return MatrixFreeQData<decltype(prepared)>{prepared, weights};
      } else {
        return StoredQData{qdata_.Read(), nq};
      }
    }();
    const bool supported = internal::SupportsTensorField<
        lf_traits::Dims, lf_traits::TestBasisOperator, lf_traits::test_type>(
            element, integration_rule_);
    MFEM_VERIFY(tensor_product_mode_ != BlfTensorProductMode::Force || supported,
                "forced tensor linear assembly requires tensor bases and quadrature");
    if (supported && tensor_product_mode_ != BlfTensorProductMode::Disable) {
      AssembleTensor(element, markers, output, ordering, elements, qdata);
    } else {
      test_basis_.Reset(element, integration_rule_, ordering);
      const auto *basis = test_basis_.Read();
      auto *values = output.ReadWrite();
      internal::AssembleLinearFull<MatrixFree, lf_traits, TestBasis>(
          dofs, elements, nq, active, qdata, basis, values);
    }
  }

public:
  DomainLinearFormIntegrator(
      T &integrator, const mfem::IntegrationRule &integration_rule,
      BlfTensorProductMode tensor_product_mode = BlfTensorProductMode::Auto)
      : integrator_(integrator), integration_rule_(integration_rule),
        tensor_product_mode_(tensor_product_mode) {}

  /// Assemble one native-order element vector on the host, overwriting output.
  void AssembleElementVector(const mfem::FiniteElement &element,
                             mfem::ElementTransformation &transformation,
                             mfem::Vector &output) {
    VerifyElement(element);
    TestBasis basis(element, integration_rule_);
    output.SetSize(basis.Dofs());
    output = 0.0;
    auto *values = output.HostReadWrite();
    const auto *table = basis.Read(false);
    mfem::Vector qvalues(lf_traits::TestComponents);
    for (int point = 0; point < integration_rule_.GetNPoints(); ++point) {
      const auto &ip = integration_rule_.IntPoint(point);
      transformation.SetIntPoint(&ip);
      integrator_.AssembleQDataHost(transformation, ip, qvalues);
      for (int dof = 0; dof < basis.Dofs(); ++dof) {
        for (int channel = 0; channel < lf_traits::TestComponents; ++channel) {
          values[dof] += ip.weight * qvalues[channel] *
                         TestBasis::Eval(table, point, channel, dof, basis.QuadraturePoints());
        }
      }
    }
  }

  /// Assemble an E-vector on the configured device in the requested ordering.
  /// @param add Add into pre-sized output instead of overwriting it.
  void AssembleElementVectors(
      const mfem::FiniteElementSpace &space, mfem::Vector &output, bool add = false,
      mfem::ElementDofOrdering ordering = mfem::ElementDofOrdering::NATIVE) {
    AssembleElements(space, nullptr, output, add, ordering);
  }

  /// Add marked element contributions into a pre-sized E-vector.
  /// @param markers Per-element 0/1 mask, not a boundary or attribute selector.
  void AddElementVectors(
      const mfem::FiniteElementSpace &space, const mfem::Array<int> &markers,
      mfem::Vector &output,
      mfem::ElementDofOrdering ordering = mfem::ElementDofOrdering::LEXICOGRAPHIC) {
    AssembleElements(space, &markers, output, true, ordering);
  }

  /// Assemble into the test space's local V-vector, including DOF orientations.
  /// Parallel true-DOF reduction remains the caller's responsibility.
  /// @param space Homogeneous fixed-order test space.
  /// @param[out] output Local V-vector; resized/cleared unless add is true.
  /// @param markers Optional per-element 0/1 mask; null includes all elements.
  /// @param add Add into pre-sized output instead of overwriting it.
  void Assemble(const mfem::FiniteElementSpace &space, mfem::Vector &output,
                const mfem::Array<int> *markers = nullptr, bool add = false) {
    AssembleVector<false>(space, output, markers, add);
  }

  /// Fuse q-function evaluation and test-basis contraction into a local V-vector.
  /// No weighted q-data array is allocated; previous cached q-data is released.
  /// Geometry/basis caches and tensor contraction scratch are still used.
  /// @param space Homogeneous fixed-order test space.
  /// @param[out] output Local V-vector; true-DOF reduction is the caller's job.
  /// @param markers Optional per-element 0/1 mask; null includes all elements.
  /// @param add Add into pre-sized output instead of overwriting it.
  void AssembleMF(const mfem::FiniteElementSpace &space, mfem::Vector &output,
                  const mfem::Array<int> *markers = nullptr, bool add = false) {
    AssembleVector<true>(space, output, markers, add);
  }

  /// Fused assembly directly into an E-vector without storing q-data.
  /// @param space Homogeneous fixed-order test space.
  /// @param[out] output E-vector in the requested local ordering.
  /// @param add Add into pre-sized output instead of overwriting it.
  /// @param ordering Native or lexicographic element DOF ordering.
  void AssembleElementVectorsMF(
      const mfem::FiniteElementSpace &space, mfem::Vector &output, bool add = false,
      mfem::ElementDofOrdering ordering = mfem::ElementDofOrdering::NATIVE) {
    AssembleElements<true>(space, nullptr, output, add, ordering);
  }

  /// Add fused contributions from marked elements without storing q-data.
  /// @param space Homogeneous fixed-order test space.
  /// @param markers Per-element 0/1 mask.
  /// @param[in,out] output Pre-sized E-vector to accumulate into.
  /// @param ordering Native or lexicographic element DOF ordering.
  void AddElementVectorsMF(
      const mfem::FiniteElementSpace &space, const mfem::Array<int> &markers,
      mfem::Vector &output,
      mfem::ElementDofOrdering ordering = mfem::ElementDofOrdering::LEXICOGRAPHIC) {
    AssembleElements<true>(space, &markers, output, true, ordering);
  }

  /// @return Number of stored weighted q-data entries; zero after MF assembly.
  int QDataSize() const { return qdata_.Size(); }

private:
  template <bool MatrixFree>
  void AssembleVector(const mfem::FiniteElementSpace &space, mfem::Vector &output,
                      const mfem::Array<int> *markers, bool add) {
    if (add) {
      MFEM_VERIFY(output.Size() == space.GetVSize(), "global output size is incorrect");
    } else {
      output.SetSize(space.GetVSize());
      output.UseDevice(true);
      output = 0.0;
    }
    if (space.GetNE() == 0) {
      AssembleElements<MatrixFree>(space, markers, element_vectors_, false,
                       mfem::ElementDofOrdering::NATIVE);
      return;
    }
    const auto &element = *space.GetFE(0);
    constexpr bool scalar = internal::IsScalarMap(lf_traits::test_type);
    const bool use_restriction = scalar ||
        dynamic_cast<const mfem::VectorTensorFiniteElement *>(&element);
    const auto *map = internal::LexicographicDofMap(element);
    const auto ordering = use_restriction && map && map->Size()
                              ? mfem::ElementDofOrdering::LEXICOGRAPHIC
                              : mfem::ElementDofOrdering::NATIVE;
    AssembleElements<MatrixFree>(space, markers, element_vectors_, false, ordering);
    if (use_restriction) {
      const auto *restriction = space.GetElementRestriction(ordering);
      restriction->AddMultTranspose(element_vectors_, output);
    } else {
      mfem::Array<int> vdofs;
      mfem::DofTransformation transformation;
      const int dofs = element.GetDof() * lf_traits::TestVDim;
      const auto *data = element_vectors_.HostRead();
      const auto *active = markers ? markers->HostRead() : nullptr;
      mfem::Vector local(dofs);
      for (int index = 0; index < space.GetNE(); ++index) {
        if (active && !active[index]) { continue; }
        std::copy_n(data + dofs * index, dofs, local.HostWrite());
        space.GetElementVDofs(index, vdofs, transformation);
        transformation.TransformDual(local);
        output.AddElementVector(vdofs, local);
      }
    }
  }
};
} // namespace mfem

#endif
