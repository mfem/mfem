#ifndef BILINEARINTEGS_IMPL_HPP
#define BILINEARINTEGS_IMPL_HPP

/// @file
/// Reference contractions, EA/PA kernels, and out-of-line template definitions
/// for the public bilinear-form API in bilinearintegs.hpp.

#include "bilinearintegs.hpp"

namespace mfem {

namespace internal {

/// Inlined pointwise policy shared by full and tensor-product basis kernels.
/// Representation and space dimension are compile-time kernel properties.
/// @tparam Storage Payload representation.
/// @tparam Traits Trial/test reference component and field metadata.
/// @tparam SpaceDimensions Cartesian space dimension for structured physical actions.
template <BlfPAStorage Storage, class Traits, int SpaceDimensions = Traits::Dims>
struct BlfPAAction {
  const mfem::real_t *data;
  int nq;
  constexpr static int trial_components = Traits::TrialComponents;
  constexpr static int test_components = Traits::TestComponents;
  constexpr static int dimensions = Traits::Dims;
  constexpr static int kelvin = SpaceDimensions * (SpaceDimensions + 1) / 2;
  using Layout = BlfPADataLayout<Storage, Traits>;
  constexpr static bool Pointwise = Layout::Pointwise;
  constexpr static bool Diagonal = Storage == BlfPAStorage::Diagonal ||
                                   Storage == BlfPAStorage::UniformDiagonal;
  constexpr static bool Symmetric = Storage == BlfPAStorage::Symmetric;
  constexpr static int Components = Layout::Components(SpaceDimensions);

  std::array<bool, (Storage == BlfPAStorage::Dense &&
                           !Traits::HasComponentFields && Components > 1
                       ? 1
                       : 0)> ordering;

  /// @param payload Device-readable quadrature payload.
  /// @param quadrature_points Quadrature points per element.
  /// @param component_major Dense ordering; ignored for statically known layouts.
  MFEM_HOST_DEVICE constexpr BlfPAAction(const mfem::real_t *payload,
                                        int quadrature_points,
                                        bool component_major)
      : data(payload), nq(quadrature_points) {
    if constexpr (Storage == BlfPAStorage::Dense &&
                  !Traits::HasComponentFields && Components > 1) {
      ordering[0] = component_major;
    }
  }

  /// @param element Element index.
  /// @param point Quadrature index.
  /// @param component Stored payload index.
  /// @return Payload entry in either point-major or component-major storage.
  MFEM_HOST_DEVICE mfem::real_t Load(int element, int point, int component) const {
    if constexpr (Storage != BlfPAStorage::Dense) {
      return data[point + nq * (component + Components * element)];
    } else if constexpr (Components == 1) {
      return data[point + nq * element];
    } else if constexpr (Traits::HasComponentFields) {
      return data[component + Components * (point + nq * element)];
    } else {
      return data[ordering[0]
                      ? point + nq * (component + Components * element)
                      : component + Components * (point + nq * element)];
    }
  }

  /// @param element Element index.
  /// @param point Quadrature index.
  /// @param row Reference test component.
  /// @param column Reference trial component.
  /// @return Dense or packed reference action entry.
  MFEM_HOST_DEVICE mfem::real_t Entry(int element, int point, int row,
                                      int column) const {
    if constexpr (Storage == BlfPAStorage::UniformDiagonal) {
      return row == column ? Load(element, point, 0) : 0.0;
    } else if constexpr (Storage == BlfPAStorage::Diagonal) {
      return row == column ? Load(element, point, row) : 0.0;
    } else if constexpr (Storage == BlfPAStorage::Symmetric) {
      const int lower = row < column ? row : column;
      const int upper = row < column ? column : row;
      return Load(element, point, lower + upper * (upper + 1) / 2);
    } else {
      return Load(element, point, column + trial_components * row);
    }
  }

  /// Apply factored coefficients in physical/Kelvin coordinates, with no
  /// Hessians and no dense reference tensor reconstruction.
  /// @tparam Transpose Swap input/output restrictions and transpose material.
  /// @param element Element index.
  /// @param point Quadrature index.
  /// @param input Reference input components, with stride nq.
  /// @param[out] output Reference output components, with stride nq.
  template <bool Transpose, int StaticStride = 0>
  MFEM_HOST_DEVICE void ApplyPoint(int element, int point,
                                   const mfem::real_t *input,
                                   mfem::real_t *output) const {
    const int component_stride = StaticStride ? StaticStride : nq;
    constexpr int input_fields = Transpose ? Traits::TestVDim : Traits::TrialVDim;
    constexpr int output_fields = Transpose ? Traits::TrialVDim : Traits::TestVDim;
    constexpr bool input_oop = Transpose ? Traits::TestOutOfPlane : Traits::TrialOutOfPlane;
    constexpr bool output_oop = Transpose ? Traits::TrialOutOfPlane : Traits::TestOutOfPlane;
    if constexpr (Storage == BlfPAStorage::Kronecker) {
      for (int field = 0; field < output_fields; ++field) {
        for (int direction = 0; direction < dimensions; ++direction) {
          mfem::real_t value = 0.0;
          for (int source = 0; source < input_fields; ++source) {
            const int row = Transpose ? source : field;
            const int column = Transpose ? field : source;
            value += Load(element, point, column + Traits::TrialVDim * row) *
                     input[component_stride * (direction + dimensions * source)];
          }
          output[component_stride * (direction + dimensions * field)] = value;
        }
      }
      for (int field = 0; field < output_fields; ++field) {
        mfem::real_t result[dimensions] = {};
        for (int direction = 0; direction < dimensions; ++direction) {
          for (int source = 0; source < dimensions; ++source) {
            const int row = Transpose ? source : direction;
            const int column = Transpose ? direction : source;
            result[direction] += Load(element, point,
                Traits::TrialVDim * Traits::TestVDim + column + dimensions * row) *
                output[component_stride * (source + dimensions * field)];
          }
        }
        for (int direction = 0; direction < dimensions; ++direction) {
          output[component_stride * (direction + dimensions * field)] = result[direction];
        }
      }
    } else {
      mfem::real_t gradient[SpaceDimensions * SpaceDimensions] = {};
      for (int field = 0; field < input_fields; ++field) {
        const int cartesian_field = input_oop ? 2 : field;
        for (int physical = 0; physical < SpaceDimensions; ++physical) {
          for (int reference = 0; reference < dimensions; ++reference) {
            gradient[physical + SpaceDimensions * cartesian_field] +=
                Load(element, point, physical + SpaceDimensions * reference) *
                input[component_stride * (reference + dimensions * field)];
          }
        }
      }
      mfem::real_t stress[SpaceDimensions * SpaceDimensions] = {};
      constexpr int material_offset = SpaceDimensions * dimensions;
      if constexpr (Storage == BlfPAStorage::IsotropicSymmetric) {
        mfem::real_t trace = 0.0;
        for (int axis = 0; axis < SpaceDimensions; ++axis) {
          trace += gradient[axis + SpaceDimensions * axis];
        }
        const mfem::real_t lambda = Load(element, point, material_offset);
        const mfem::real_t mu = Load(element, point, material_offset + 1);
        for (int field = 0; field < SpaceDimensions; ++field) {
          for (int direction = 0; direction < SpaceDimensions; ++direction) {
            stress[direction + SpaceDimensions * field] =
                mu * (gradient[direction + SpaceDimensions * field] +
                      gradient[field + SpaceDimensions * direction]) +
                (field == direction ? lambda * trace : 0.0);
          }
        }
      } else {
        constexpr mfem::real_t inverse_sqrt_two = 0.70710678118654752440;
        mfem::real_t strain[kelvin] = {};
        mfem::real_t response[kelvin] = {};
        for (int axis = 0; axis < SpaceDimensions; ++axis) {
          strain[axis] = gradient[axis + SpaceDimensions * axis];
        }
        int index = SpaceDimensions;
        for (int row = 0; row < SpaceDimensions; ++row) {
          for (int column = row + 1; column < SpaceDimensions; ++column) {
            strain[index++] = inverse_sqrt_two *
                (gradient[row + SpaceDimensions * column] + gradient[column + SpaceDimensions * row]);
          }
        }
        for (int row = 0; row < kelvin; ++row) {
          for (int column = 0; column < kelvin; ++column) {
            int entry;
            if constexpr (Storage == BlfPAStorage::SymmetricKelvin) {
              const int lower = row < column ? row : column;
              const int upper = row < column ? column : row;
              entry = lower + upper * (upper + 1) / 2;
            } else {
              entry = Transpose ? row + kelvin * column : column + kelvin * row;
            }
            response[row] += Load(element, point, material_offset + entry) * strain[column];
          }
        }
        for (int axis = 0; axis < SpaceDimensions; ++axis) {
          stress[axis + SpaceDimensions * axis] = response[axis];
        }
        index = SpaceDimensions;
        for (int row = 0; row < SpaceDimensions; ++row) {
          for (int column = row + 1; column < SpaceDimensions; ++column) {
            stress[row + SpaceDimensions * column] = stress[column + SpaceDimensions * row] =
                inverse_sqrt_two * response[index++];
          }
        }
      }
      for (int field = 0; field < output_fields; ++field) {
        const int cartesian_field = output_oop ? 2 : field;
        for (int reference = 0; reference < dimensions; ++reference) {
          mfem::real_t value = 0.0;
          for (int physical = 0; physical < SpaceDimensions; ++physical) {
            value += Load(element, point, physical + SpaceDimensions * reference) *
                     stress[physical + SpaceDimensions * cartesian_field];
          }
          output[component_stride * (reference + dimensions * field)] = value;
        }
      }
    }
  }
};

/// Evaluate reference shapes without any physical geometry pullback.
/// @tparam Operator Reference operation (Gradient for component curl/divergence).
/// @tparam Map MFEM finite-element map determining the shape evaluator.
/// @tparam Components Total reference output count, including VDim copies.
/// @tparam VDim Scalar component count; native vector FEs require 1.
/// @param fe Reference finite element.
/// @param ip Reference integration point.
/// @param[out] shape Resized (fe.GetDof() * VDim, Components) matrix with
/// component-major DOFs and reference outputs.
template <BlfFieldOperator Operator, mfem::FiniteElement::MapType Map,
          int Components, int VDim = 1>
void CalcReferenceShape(const mfem::FiniteElement &fe,
                        const mfem::IntegrationPoint &ip,
                        mfem::DenseMatrix &shape) {
  if constexpr (VDim > 1) {
    static_assert(IsScalarMap(Map));
    constexpr int scalar_components = Components / VDim;
    static_assert(Components % VDim == 0);
    mfem::DenseMatrix scalar_shape;
    CalcReferenceShape<Operator, Map, scalar_components>(fe, ip, scalar_shape);
    shape.SetSize(fe.GetDof() * VDim, Components);
    shape = 0.0;
    for (int field = 0; field < VDim; ++field) {
      for (int dof = 0; dof < fe.GetDof(); ++dof) {
        for (int component = 0; component < scalar_components; ++component) {
          shape(dof + fe.GetDof() * field,
                component + scalar_components * field) =
              scalar_shape(dof, component);
        }
      }
    }
  } else {
    shape.SetSize(fe.GetDof(), Components);
    if constexpr (Operator == BlfFieldOperator::Value) {
      if constexpr (IsScalarMap(Map)) {
        static_assert(Components == 1);
        mfem::Vector scalar_shape(shape.GetData(), fe.GetDof());
        fe.CalcShape(ip, scalar_shape);
      } else {
        MFEM_VERIFY(fe.GetRangeDim() == Components,
                    "finite-element range dimension does not match its "
                    "BlfIntegratorTraits");
        fe.CalcVShape(ip, shape);
      }
    } else if constexpr (Operator == BlfFieldOperator::Gradient) {
      MFEM_VERIFY(fe.GetDim() == Components,
                  "finite-element dimension does not match its gradient "
                  "BlfIntegratorTraits");
      fe.CalcDShape(ip, shape);
    } else if constexpr (Operator == BlfFieldOperator::Divergence) {
      static_assert(Components == 1);
      mfem::Vector div_shape(shape.GetData(), fe.GetDof());
      fe.CalcDivShape(ip, div_shape);
    } else {
      static_assert(Operator == BlfFieldOperator::Curl);
      MFEM_VERIFY(fe.GetCurlDim() == Components,
                  "finite-element curl dimension does not match its "
                  "BlfIntegratorTraits");
      fe.CalcCurlShape(ip, shape);
    }
  }
}

/// Constant-time interface checks for native matrices and direct EA output.
/// Batch callers supply uniform element geometry/order; do not scan elements.
template <class Traits>
void VerifyEAElementSpaces(const mfem::FiniteElementSpace &trial,
                           const mfem::FiniteElementSpace &test) {
  MFEM_VERIFY(trial.GetVDim() == Traits::TrialVDim &&
                  test.GetVDim() == Traits::TestVDim,
              "finite-element space vdim does not match the operation");
  MFEM_VERIFY(trial.GetMesh() == test.GetMesh(),
              "trial and test spaces must use the same mesh");
  MFEM_VERIFY(trial.GetNE() == test.GetNE(),
              "trial and test spaces have different element counts");
  if (trial.GetNE() == 0) { return; }
  const auto *trial_fe = trial.GetFE(0);
  const auto *test_fe = test.GetFE(0);
  MFEM_VERIFY(trial_fe && test_fe, "finite-element space returned a null FE");
  MFEM_VERIFY(trial_fe->GetDim() == Traits::Dims &&
                  trial_fe->GetMapType() == Traits::trial_type,
              "trial finite element has the wrong dimension or MapType");
  MFEM_VERIFY(test_fe->GetDim() == Traits::Dims &&
                  test_fe->GetMapType() == Traits::test_type,
              "test finite element has the wrong dimension or MapType");
}

/// Borrow a map only when the basis and requested EA output orders differ.
inline const int *ReadEAOutputDofMap(const mfem::FiniteElementSpace &space,
    bool same_space_interface, bool use_device, bool lexicographic_basis) {
  const bool native_output = same_space_interface &&
      mfem::GetEVectorOrdering(space) == mfem::ElementDofOrdering::NATIVE;
  return native_output == lexicographic_basis
      ? ReadLexicographicDofMap(*space.GetFE(0), use_device) : nullptr;
}

inline mfem::real_t *PrepareEAOutput(mfem::Vector &emat, int size,
                                    bool add, bool use_device) {
  if (add) {
    MFEM_VERIFY(emat.Size() == size,
                "existing element-assembly vector has the wrong size");
  } else { emat.SetSize(size); }
  return add ? emat.ReadWrite(use_device) : emat.Write(use_device);
}

}

namespace internal {


template <class Traits, class Action, bool ADAt, int FixedTrialDofs,
          int FixedTestDofs, int FixedNQ>
struct FullDiagonalRowFunctor {
  using TrialBasis = BlfReferenceBasis<Traits::TrialBasisOperator,
      Traits::trial_type, Traits::TrialComponents, Traits::TrialVDim>;
  using TestBasis = BlfReferenceBasis<Traits::TestBasisOperator,
      Traits::test_type, Traits::TestComponents, Traits::TestVDim>;
  int runtime_trial_dofs, runtime_test_dofs, runtime_nq;
  const mfem::real_t *trial_basis, *test_basis, *weights;
  mfem::real_t *output;
  mfem::real_t scale;
  Action action;

  MFEM_HOST_DEVICE void operator()(int index) const {
    const int trial_dofs = FixedTrialDofs ? FixedTrialDofs : runtime_trial_dofs;
    const int test_dofs = FixedTestDofs ? FixedTestDofs : runtime_test_dofs;
    const int nq = FixedNQ ? FixedNQ : runtime_nq;
    const int element = index / test_dofs;
    const int row = index % test_dofs;
    mfem::real_t diagonal = 0.0;
    const int first_column = ADAt ? 0 : row;
    const int last_column = ADAt ? trial_dofs : row + 1;
    for (int column = first_column; column < last_column; ++column) {
      mfem::real_t entry = 0.0;
      for (int point = 0; point < nq; ++point) {
        mfem::real_t input[Traits::TrialComponents];
        mfem::real_t result[Traits::TestComponents];
        for (int component = 0; component < Traits::TrialComponents; ++component) {
          input[component] = TrialBasis::Eval(
              trial_basis, point, component, column, nq);
        }
        if constexpr (Action::Pointwise) {
          action.template ApplyPoint<false, 1>(element, point, input, result);
        } else {
          for (int test = 0; test < Traits::TestComponents; ++test) {
            result[test] = 0.0;
            for (int trial = 0; trial < Traits::TrialComponents; ++trial) {
              result[test] += action.Entry(element, point, test, trial) * input[trial];
            }
          }
        }
        for (int component = 0; component < Traits::TestComponents; ++component) {
          entry += TestBasis::Eval(test_basis, point, component, row, nq) *
                   result[component];
        }
      }
      if constexpr (ADAt) {
        diagonal += scale * scale * entry * entry *
                    weights[column + trial_dofs * element];
      } else {
        diagonal += scale * entry;
      }
    }
    output[index] += diagonal;
  }
};

/// Basis policies shared by direct diagonals and fused ADAt columns.
template <int Components, int FixedDofs, int FixedNQ>
struct FullDiagonalBasis {
  const mfem::real_t *values;
  int runtime_dofs, runtime_nq;
  MFEM_HOST_DEVICE int Dofs() const { return FixedDofs ? FixedDofs : runtime_dofs; }
  MFEM_HOST_DEVICE int NQ() const { return FixedNQ ? FixedNQ : runtime_nq; }
  MFEM_HOST_DEVICE int StageReals() const { return 0; }
  MFEM_HOST_DEVICE void Stage(mfem::real_t *) {}
  MFEM_HOST_DEVICE mfem::real_t Eval(int q, int c, int d) const {
    return values[q + NQ() * (c + Components * d)];
  }
};

/// Evaluate a single tensor DOF without constructing a probe E-vector.
template <class Traits, bool Trial, int FixedD1D, int FixedQ1D>
struct TensorDiagonalBasis {
  constexpr static int Dims = Traits::Dims;
  constexpr static auto Map = Trial ? Traits::trial_type : Traits::test_type;
  constexpr static auto Operator = Trial ? Traits::TrialBasisOperator : Traits::TestBasisOperator;
  constexpr static int Fields = IsVectorTensorMap(Map) ? Dims :
      (Trial ? Traits::TrialVDim : Traits::TestVDim);
  const mfem::real_t *values, *derivatives, *open;
  int runtime_d1d, runtime_q1d;
  MFEM_HOST_DEVICE int ND() const { return FixedD1D ? FixedD1D : runtime_d1d; }
  MFEM_HOST_DEVICE int Q1D() const { return FixedQ1D ? FixedQ1D : runtime_q1d; }
  MFEM_HOST_DEVICE int NQ() const { return TensorIntPow(Q1D(), Dims); }
  MFEM_HOST_DEVICE int OpenND() const { return ND() - (IsVectorTensorMap(Map) ? 1 : 0); }
  MFEM_HOST_DEVICE int ComponentDofs() const {
    if constexpr (Map == mfem::FiniteElement::H_CURL) {
      return OpenND() * TensorIntPow(ND(), Dims - 1);
    } else if constexpr (Map == mfem::FiniteElement::H_DIV) {
      return ND() * TensorIntPow(OpenND(), Dims - 1);
    } else { return TensorIntPow(ND(), Dims); }
  }
  MFEM_HOST_DEVICE int Dofs() const { return Fields * ComponentDofs(); }
  MFEM_HOST_DEVICE int StageReals() const {
    return Q1D() * (2 * ND() + (IsVectorTensorMap(Map) ? OpenND() : 0));
  }
  MFEM_HOST_DEVICE void Stage(mfem::real_t *buffer) {
    const int size = ND() * Q1D();
    MFEM_FOREACH_THREAD(i, x, size) {
      buffer[i] = values[i];
      buffer[i + size] = derivatives ? derivatives[i] : 0.0;
    }
    if constexpr (IsVectorTensorMap(Map)) {
      MFEM_FOREACH_THREAD(i, x, OpenND() * Q1D()) { buffer[i + 2 * size] = open[i]; }
    }
    values = buffer;
    derivatives = buffer + size;
    open = buffer + 2 * size;
  }
  MFEM_HOST_DEVICE int AxisWidth(const BlfTensorOperatorTerm &term, int axis) const {
    return TensorFieldUsesClosedMap(Map, term.field_component, axis) ? ND() : OpenND();
  }
  MFEM_HOST_DEVICE mfem::real_t Factor(int q, int d,
      const BlfTensorOperatorTerm &term, int axis) const {
    const auto *table = term.derivative_dimension == axis ? derivatives :
        TensorFieldUsesClosedMap(Map, term.field_component, axis) ? values : open;
    return table[q + Q1D() * d];
  }
  MFEM_HOST_DEVICE mfem::real_t Eval(int point, int component, int dof) const {
    const int field = dof / ComponentDofs();
    const int local = dof % ComponentDofs();
    mfem::real_t value = 0.0;
    constexpr int terms = TensorOperatorTermCount(Operator, Dims);
    for (int t = 0; t < terms; ++t) {
      const auto term = TensorOperatorTerm(Operator, Dims, component, t);
      if (term.field_component != field) { continue; }
      int q = point, d = local;
      mfem::real_t product = term.sign;
      for (int axis = 0; axis < Dims; ++axis) {
        const int width = AxisWidth(term, axis);
        product *= Factor(q % Q1D(), d % width, term, axis);
        q /= Q1D();
        d /= width;
      }
      value += product;
    }
    return value;
  }
};

template <class Traits, class Action>
MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
ApplyDiagonalPoint(const Action &action, int q, const mfem::real_t *input,
                   mfem::real_t *result) {
  if constexpr (Action::Pointwise) {
    action.template ApplyPoint<false, 1>(0, q, input, result);
  } else if constexpr (Action::Diagonal) {
    for (int c = 0; c < Traits::TestComponents; ++c) {
      result[c] = c < Traits::TrialComponents ? action.Entry(0, q, c, c) * input[c] : 0.0;
    }
  } else {
    for (int test = 0; test < Traits::TestComponents; ++test) {
      result[test] = 0.0;
      for (int trial = 0; trial < Traits::TrialComponents; ++trial) {
        result[test] += action.Entry(0, q, test, trial) * input[trial];
      }
    }
  }
}

/// Direct regular diagonal: quadrature lanes cooperate on matching DOFs only.
template <class Traits, class TrialBasis, class TestBasis, class Action>
struct DirectDiagonalFunctor {
  TrialBasis trial_basis;
  TestBasis test_basis;
  Action action;
  mfem::real_t *output;
  int lanes;
  bool cache_qdata;

  MFEM_HOST_DEVICE int ScratchReals() const {
    return std::min(256, test_basis.Dofs() * lanes);
  }

  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *workspace) const {
    auto trial = trial_basis;
    auto test = test_basis;
    auto *buffer = reinterpret_cast<mfem::real_t *>(workspace);
    trial.Stage(buffer);
    buffer += trial.StageReals();
    test.Stage(buffer);
    buffer += test.StageReals();
    auto *qdata = buffer;
    buffer += cache_qdata ? action.nq * Action::Components : 0;
    auto *partial = buffer;
    MFEM_SYNC_THREAD;
    const int rows_per_tile = 256 / lanes;
    for (int e = begin; e < end; ++e) {
      auto local_action = action;
      local_action.data += e * action.nq * Action::Components;
      if (cache_qdata) {
        MFEM_FOREACH_THREAD(i, x, action.nq * Action::Components) {
          qdata[i] = local_action.data[i];
        }
        local_action.data = qdata;
        MFEM_SYNC_THREAD;
      }
      for (int start = 0; start < test.Dofs(); start += rows_per_tile) {
        const int rows = std::min(rows_per_tile, test.Dofs() - start);
        MFEM_FOREACH_THREAD_2D(lane, r, x, lanes, rows) {
          mfem::real_t value = 0.0;
          for (int q = lane; q < action.nq; q += lanes) {
            mfem::real_t input[Traits::TrialComponents], result[Traits::TestComponents];
            for (int c = 0; c < Traits::TrialComponents; ++c) {
              input[c] = trial.Eval(q, c, start + r);
            }
            ApplyDiagonalPoint<Traits>(local_action, q, input, result);
            for (int c = 0; c < Traits::TestComponents; ++c) {
              value += test.Eval(q, c, start + r) * result[c];
            }
          }
          partial[lane + lanes * r] = value;
        }
        MFEM_SYNC_THREAD;
        MFEM_FOREACH_THREAD(r, x, rows) {
          mfem::real_t value = 0.0;
          for (int lane = 0; lane < lanes; ++lane) { value += partial[lane + lanes * r]; }
          output[start + r + test.Dofs() * e] += value;
        }
        MFEM_SYNC_THREAD;
      }
    }
  }
};

/// Preserve sum factorization for tensor ADAt, including derivative/open maps.
template <class Traits, class Basis> struct TensorDiagonalProjection {
  int d1d, q1d;
  MFEM_HOST_DEVICE int ScratchReals() const {
    return Traits::Dims >= 2 ? d1d * TensorIntPow(q1d, Traits::Dims - 1) +
        (Traits::Dims == 3 ? d1d * d1d * q1d : 0) : 0;
  }
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  Project(const Basis &basis, const mfem::real_t *values,
          mfem::real_t *column, mfem::real_t *scratch) const {
    constexpr int dims = Traits::Dims;
    const int nq = basis.NQ(), q1d = basis.Q1D();
    auto *first = scratch;
    auto *second = first + basis.ND() * TensorIntPow(q1d, dims - 1);
    MFEM_FOREACH_THREAD(r, x, basis.Dofs()) { column[r] = 0.0; }
    MFEM_SYNC_THREAD;
    for (int c = 0; c < Traits::TestComponents; ++c) {
      constexpr int terms = TensorOperatorTermCount(Basis::Operator, dims);
      for (int t = 0; t < terms; ++t) {
        const auto term = TensorOperatorTerm(Basis::Operator, dims, c, t);
        const int nx = basis.AxisWidth(term, 0);
        const int ny = dims >= 2 ? basis.AxisWidth(term, 1) : 1;
        const int nz = dims == 3 ? basis.AxisWidth(term, 2) : 1;
        if constexpr (dims == 1) {
          MFEM_FOREACH_THREAD(dx, x, nx) {
            mfem::real_t value = 0.0;
            for (int q = 0; q < q1d; ++q) {
              value += basis.Factor(q, dx, term, 0) * values[q + nq * c];
            }
            column[dx + basis.ComponentDofs() * term.field_component] += term.sign * value;
          }
        } else {
          MFEM_FOREACH_THREAD_2D(dx, yz, x, nx, TensorIntPow(q1d, dims - 1)) {
            mfem::real_t value = 0.0;
            for (int q = 0; q < q1d; ++q) {
              value += basis.Factor(q, dx, term, 0) * values[q + q1d * yz + nq * c];
            }
            first[dx + nx * yz] = value;
          }
          MFEM_SYNC_THREAD;
          MFEM_FOREACH_THREAD_3D(dx, dy, z, x, nx, ny, dims == 3 ? q1d : 1) {
            mfem::real_t value = 0.0;
            for (int q = 0; q < q1d; ++q) {
              value += basis.Factor(q, dy, term, 1) * first[dx + nx * (q + q1d * z)];
            }
            if constexpr (dims == 2) {
              column[dx + nx * dy + basis.ComponentDofs() * term.field_component] += term.sign * value;
            } else { second[dx + nx * (dy + ny * z)] = value; }
          }
          if constexpr (dims == 3) {
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD_3D(dx, dy, dz, x, nx, ny, nz) {
              mfem::real_t value = 0.0;
              for (int q = 0; q < q1d; ++q) {
                value += basis.Factor(q, dz, term, 2) * second[dx + nx * (dy + ny * q)];
              }
              column[dx + nx * (dy + ny * dz) + basis.ComponentDofs() * term.field_component] += term.sign * value;
            }
          }
        }
        MFEM_SYNC_THREAD;
      }
    }
  }
};

/// One launch for all ADAt columns; square only after the complete projection.
template <class Traits, class TrialBasis, class TestBasis, class Projection, class Action>
struct ColumnDiagonalFunctor {
  TrialBasis trial_basis;
  TestBasis test_basis;
  Projection projection;
  Action action;
  const mfem::real_t *weights;
  mfem::real_t *output;
  bool cache_qdata;

  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *workspace) const {
    auto trial = trial_basis;
    auto test = test_basis;
    auto *buffer = reinterpret_cast<mfem::real_t *>(workspace);
    trial.Stage(buffer);
    buffer += trial.StageReals();
    test.Stage(buffer);
    buffer += test.StageReals();
    auto *qdata = buffer;
    buffer += cache_qdata ? action.nq * Action::Components : 0;
    auto *values = buffer;
    auto *column = values + action.nq * Traits::TestComponents;
    auto *diagonal = column + test.Dofs();
    auto *scratch = diagonal + test.Dofs();
    MFEM_SYNC_THREAD;
    for (int e = begin; e < end; ++e) {
      auto local_action = action;
      local_action.data += e * action.nq * Action::Components;
      if (cache_qdata) {
        MFEM_FOREACH_THREAD(i, x, action.nq * Action::Components) { qdata[i] = local_action.data[i]; }
        local_action.data = qdata;
      }
      MFEM_FOREACH_THREAD(r, x, test.Dofs()) { diagonal[r] = 0.0; }
      MFEM_SYNC_THREAD;
      for (int d = 0; d < trial.Dofs(); ++d) {
        MFEM_FOREACH_THREAD(q, x, action.nq) {
          mfem::real_t input[Traits::TrialComponents], result[Traits::TestComponents];
          for (int c = 0; c < Traits::TrialComponents; ++c) { input[c] = trial.Eval(q, c, d); }
          ApplyDiagonalPoint<Traits>(local_action, q, input, result);
          for (int c = 0; c < Traits::TestComponents; ++c) { values[q + action.nq * c] = result[c]; }
        }
        MFEM_SYNC_THREAD;
        projection.Project(test, values, column, scratch);
        MFEM_FOREACH_THREAD(r, x, test.Dofs()) {
          diagonal[r] += column[r] * column[r] * weights[d + trial.Dofs() * e];
        }
        MFEM_SYNC_THREAD;
      }
      MFEM_FOREACH_THREAD(r, x, test.Dofs()) { output[r + test.Dofs() * e] += diagonal[r]; }
      MFEM_SYNC_THREAD;
    }
  }
};

/// Separable regular diagonal when matching DOFs share tensor axis sizes.
template <class Traits, class TrialBasis, class TestBasis, class Action>
struct TensorFactorDiagonalFunctor {
  TrialBasis trial_basis;
  TestBasis test_basis;
  Action action;
  mfem::real_t *output;
  bool cache_qdata;

  MFEM_HOST_DEVICE int ScratchReals() const {
    constexpr int dims = Traits::Dims;
    const int nd = test_basis.ND(), nq = test_basis.Q1D();
    return dims * nd * (nq + 1) +
        (dims >= 2 ? nd * TensorIntPow(nq, dims - 1) : 0) +
        (dims == 3 ? nd * nd * nq : 0);
  }
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *workspace) const {
    constexpr int dims = Traits::Dims;
    auto trial = trial_basis;
    auto test = test_basis;
    const int nd = test.ND(), q1d = test.Q1D(), nq = test.NQ(), stride = q1d + 1;
    constexpr bool can_be_symmetric = Action::Symmetric &&
        Traits::TrialBasisOperator == Traits::TestBasisOperator &&
        Traits::TrialVDim == Traits::TestVDim && TrialBasis::Map == TestBasis::Map;
    const bool symmetric = can_be_symmetric && trial.values == test.values &&
        trial.derivatives == test.derivatives && trial.open == test.open;
    auto *buffer = reinterpret_cast<mfem::real_t *>(workspace);
    trial.Stage(buffer);
    buffer += trial.StageReals();
    test.Stage(buffer);
    buffer += test.StageReals();
    auto *qdata = buffer;
    buffer += cache_qdata ? nq * Action::Components : 0;
    auto *products = buffer;
    auto *first = products + dims * nd * stride;
    auto *second = first + (dims >= 2 ? nd * TensorIntPow(q1d, dims - 1) : 0);
    MFEM_SYNC_THREAD;
    for (int e = begin; e < end; ++e) {
      auto local_action = action;
      local_action.data += e * nq * Action::Components;
      if (cache_qdata) {
        MFEM_FOREACH_THREAD(i, x, nq * Action::Components) { qdata[i] = local_action.data[i]; }
        local_action.data = qdata;
        MFEM_SYNC_THREAD;
      }
      for (int tc = 0; tc < Traits::TestComponents; ++tc) {
        for (int rc = 0; rc < Traits::TrialComponents; ++rc) {
          if constexpr (Action::Diagonal) { if (tc != rc) { continue; } }
          if (symmetric && rc > tc) { continue; }
          constexpr int tt = TensorOperatorTermCount(TestBasis::Operator, dims);
          constexpr int rt = TensorOperatorTermCount(TrialBasis::Operator, dims);
          for (int ti = 0; ti < tt; ++ti) {
            const auto test_term = TensorOperatorTerm(TestBasis::Operator, dims, tc, ti);
            for (int ri = 0; ri < rt; ++ri) {
              const auto trial_term = TensorOperatorTerm(TrialBasis::Operator, dims, rc, ri);
              if (test_term.field_component != trial_term.field_component) { continue; }
              const int nx = test.AxisWidth(test_term, 0);
              const int ny = dims >= 2 ? test.AxisWidth(test_term, 1) : 1;
              const int nz = dims == 3 ? test.AxisWidth(test_term, 2) : 1;
              const int field_offset = test.ComponentDofs() * test_term.field_component;
              const auto sign = test_term.sign * trial_term.sign *
                  (symmetric && tc != rc ? 2.0 : 1.0);
              MFEM_FOREACH_THREAD_3D(q, d, axis, x, q1d, nd, dims) {
                if (d < test.AxisWidth(test_term, axis)) {
                  products[q + stride * (d + nd * axis)] =
                      test.Factor(q, d, test_term, axis) * trial.Factor(q, d, trial_term, axis);
                }
              }
              MFEM_SYNC_THREAD;
              if constexpr (dims == 1) {
                MFEM_FOREACH_THREAD(d, x, nx) {
                  mfem::real_t value = 0.0;
                  for (int q = 0; q < q1d; ++q) {
                    value += products[q + stride * d] * local_action.Entry(0, q, tc, rc);
                  }
                  output[d + field_offset + test.Dofs() * e] += sign * value;
                }
              } else {
                MFEM_FOREACH_THREAD_2D(dx, yz, x, nx, TensorIntPow(q1d, dims - 1)) {
                  mfem::real_t value = 0.0;
                  for (int q = 0; q < q1d; ++q) {
                    value += products[q + stride * dx] * local_action.Entry(0, q + q1d * yz, tc, rc);
                  }
                  first[dx + nx * yz] = value;
                }
                MFEM_SYNC_THREAD;
                MFEM_FOREACH_THREAD_3D(dx, dy, z, x, nx, ny, dims == 3 ? q1d : 1) {
                  mfem::real_t value = 0.0;
                  for (int q = 0; q < q1d; ++q) {
                    value += products[q + stride * (dy + nd)] * first[dx + nx * (q + q1d * z)];
                  }
                  if constexpr (dims == 2) {
                    output[dx + nx * dy + field_offset + test.Dofs() * e] += sign * value;
                  } else { second[dx + nx * (dy + ny * z)] = value; }
                }
                if constexpr (dims == 3) {
                  MFEM_SYNC_THREAD;
                  MFEM_FOREACH_THREAD_3D(dx, dy, dz, x, nx, ny, nz) {
                    mfem::real_t value = 0.0;
                    for (int q = 0; q < q1d; ++q) {
                      value += products[q + stride * (dz + 2 * nd)] * second[dx + nx * (dy + ny * q)];
                    }
                    output[dx + nx * (dy + ny * dz) + field_offset + test.Dofs() * e] += sign * value;
                  }
                }
              }
              MFEM_SYNC_THREAD;
            }
          }
        }
      }
    }
  }
};

/// Parallel full-basis entries, reduced to ADAt without an element matrix.
template <class Traits, class TrialBasis, class TestBasis, class Action>
struct FullPairDiagonalFunctor {
  TrialBasis trial_basis;
  TestBasis test_basis;
  Action action;
  const mfem::real_t *weights;
  mfem::real_t *output;
  int lanes, row_tile;
  bool cache_qdata;

  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *workspace) const {
    auto *qdata = reinterpret_cast<mfem::real_t *>(workspace);
    auto *partial = qdata + (cache_qdata ? action.nq * Action::Components : 0);
    auto *entries = partial + lanes * row_tile * trial_basis.Dofs();
    const int tiles = (test_basis.Dofs() + row_tile - 1) / row_tile;
    for (int group = begin; group < end; ++group) {
      const int e = group / tiles, start = (group % tiles) * row_tile;
      const int rows = std::min(row_tile, test_basis.Dofs() - start);
      auto local_action = action;
      local_action.data += e * action.nq * Action::Components;
      if (cache_qdata) {
        MFEM_FOREACH_THREAD(i, x, action.nq * Action::Components) { qdata[i] = local_action.data[i]; }
        local_action.data = qdata;
        MFEM_SYNC_THREAD;
      }
      MFEM_FOREACH_THREAD_2D(lane, pair, x, lanes, rows * trial_basis.Dofs()) {
        const int column = pair % trial_basis.Dofs(), row = start + pair / trial_basis.Dofs();
        mfem::real_t value = 0.0;
        for (int q = lane; q < action.nq; q += lanes) {
          mfem::real_t input[Traits::TrialComponents], result[Traits::TestComponents];
          for (int c = 0; c < Traits::TrialComponents; ++c) { input[c] = trial_basis.Eval(q, c, column); }
          ApplyDiagonalPoint<Traits>(local_action, q, input, result);
          for (int c = 0; c < Traits::TestComponents; ++c) { value += test_basis.Eval(q, c, row) * result[c]; }
        }
        partial[lane + lanes * pair] = value;
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(pair, x, rows * trial_basis.Dofs()) {
        mfem::real_t value = 0.0;
        for (int lane = 0; lane < lanes; ++lane) { value += partial[lane + lanes * pair]; }
        entries[pair] = value * value * weights[pair % trial_basis.Dofs() + trial_basis.Dofs() * e];
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(r, x, rows) {
        mfem::real_t value = 0.0;
        for (int c = 0; c < trial_basis.Dofs(); ++c) { value += entries[c + trial_basis.Dofs() * r]; }
        output[start + r + test_basis.Dofs() * e] += value;
      }
      MFEM_SYNC_THREAD;
    }
  }
};

/// Block workspace and batching for the diagonal's tensor contractions.
template <int Dims> struct TensorDiagonalSizes {
  int basis_stride, basis_size, first_size, second_size, dofs, element_batch;

  MFEM_HOST_DEVICE constexpr TensorDiagonalSizes(int nd, int nq) {
    basis_stride = nq + 1;
    basis_size = nd * basis_stride;
    first_size = Dims >= 2 ? nd * TensorIntPow(nq, Dims - 1) : 0;
    second_size = Dims == 3 ? nd * nd * nq : 0;
    dofs = TensorIntPow(nd, Dims);
    const int scratch = first_size + second_size;
    constexpr int shared_reals = 32 * 1024 / sizeof(mfem::real_t);
    const int shared_batch = scratch ? (shared_reals - basis_size) / scratch : 256;
    const int work = std::max({first_size, second_size, dofs});
    element_batch = std::max(1, std::min({64, 128 / work, shared_batch}));
  }
};

/// Fuse all axes in one launch, retaining intermediate contractions in scratch.
template <int Dims, int FixedD1D, int FixedQ1D, int QDataVDim = 1>
struct ScalarDiagonalTensorFunctor {
  int runtime_d1d, runtime_q1d;
  const mfem::real_t *trial, *test, *input;
  mfem::real_t *output;

  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *workspace) const {
    const int nd = FixedD1D ? FixedD1D : runtime_d1d;
    const int nq = FixedQ1D ? FixedQ1D : runtime_q1d;
    const TensorDiagonalSizes<Dims> sizes(nd, nq);
    auto *basis = reinterpret_cast<mfem::real_t *>(workspace);
    auto *first = basis + sizes.basis_size;
    auto *second = first + sizes.element_batch * sizes.first_size;
    MFEM_FOREACH_THREAD_2D(q, d, x, nq, nd) {
      basis[q + sizes.basis_stride * d] = trial[q + nq * d] * test[q + nq * d];
    }
    MFEM_SYNC_THREAD;
    for (int start = begin; start < end; start += sizes.element_batch) {
      const int batch = std::min(sizes.element_batch, end - start);
      if constexpr (Dims == 1) {
        MFEM_FOREACH_THREAD_2D(d, e, x, nd, batch) {
          mfem::real_t value = 0.0;
          for (int q = 0; q < nq; ++q) {
            value += basis[q + sizes.basis_stride * d] *
                     input[q + nq * ((start + e) / QDataVDim)];
          }
          output[d + nd * (start + e)] += value;
        }
      } else {
        MFEM_FOREACH_THREAD_2D(i, e, x, sizes.first_size, batch) {
          const int dx = i % nd, yz = i / nd;
          mfem::real_t value = 0.0;
          for (int q = 0; q < nq; ++q) {
            value += basis[q + sizes.basis_stride * dx] *
                     input[q + nq * yz + TensorIntPow(nq, Dims) *
                           ((start + e) / QDataVDim)];
          }
          first[i + sizes.first_size * e] = value;
        }
        MFEM_SYNC_THREAD;
        const int count = Dims == 2 ? sizes.dofs : sizes.second_size;
        MFEM_FOREACH_THREAD_2D(i, e, x, count, batch) {
          const int dx = i % nd, dy = (i / nd) % nd, z = i / (nd * nd);
          mfem::real_t value = 0.0;
          for (int q = 0; q < nq; ++q) {
            value += basis[q + sizes.basis_stride * dy] *
                     first[dx + nd * (q + nq * z) + sizes.first_size * e];
          }
          if constexpr (Dims == 2) {
            output[i + sizes.dofs * (start + e)] += value;
          } else {
            second[i + sizes.second_size * e] = value;
          }
        }
        if constexpr (Dims == 3) {
          MFEM_SYNC_THREAD;
          MFEM_FOREACH_THREAD_2D(i, e, x, sizes.dofs, batch) {
            const int xy = i % (nd * nd), dz = i / (nd * nd);
            mfem::real_t value = 0.0;
            for (int q = 0; q < nq; ++q) {
              value += basis[q + sizes.basis_stride * dz] *
                       second[xy + nd * nd * q + sizes.second_size * e];
            }
            output[i + sizes.dofs * (start + e)] += value;
          }
        }
      }
      MFEM_SYNC_THREAD;
    }
  }
};

/// Choose a one-dimensional value or derivative factor for a scalar operator.
/// @tparam Operator Reference value or gradient operation.
/// @tparam Pointer Indexable scalar basis buffer type.
/// @param values One-dimensional value table, quadrature-fastest.
/// @param derivatives Derivative table in the same layout.
/// @param q One-dimensional quadrature index.
/// @param dof One-dimensional DOF index.
/// @param component Reference output component (gradient direction).
/// @param dimension Axis of this tensor factor.
/// @param q1d Quadrature points per axis.
template <BlfFieldOperator Operator, class Pointer>
MFEM_HOST_DEVICE mfem::real_t
TensorFactor(Pointer values, Pointer derivatives,
             int q, int dof, int component, int dimension, int q1d) {
  if constexpr (Operator == BlfFieldOperator::Gradient) {
    return (component == dimension ? derivatives : values)[q + q1d * dof];
  } else {
    return values[q + q1d * dof];
  }
}

/// Expand isotropic elasticity from its PA pullback and two Lame parameters.
/// Avoid projecting a dense Cartesian fourth-order tensor or contracting its
/// structural zeros. Works also for embedded component fields.
template <class Traits, class Prepared> struct IsotropicEAQData {
  Prepared prepared;
  int space_dimensions;
  template <class QData>
  MFEM_HOST_DEVICE void AssembleQData(int e, int q, QData data) const {
    constexpr int dims = Traits::Dims;
    mfem::real_t payload[11];
    prepared.AssemblePAData(e, q, payload, 1.0);
    const int offset = space_dimensions * dims;
    const auto lambda = payload[offset], mu = payload[offset + 1];
    for (int test = 0; test < Traits::TestComponents; ++test) {
      const int f = Traits::TestOutOfPlane ? 2 : test / dims;
      const int a = test % dims;
      for (int trial = 0; trial < Traits::TrialComponents; ++trial) {
        if (!QData::StoresEntry(test, trial)) { continue; }
        const int g = Traits::TrialOutOfPlane ? 2 : trial / dims;
        const int b = trial % dims;
        auto value = lambda * payload[f + space_dimensions * a] *
                              payload[g + space_dimensions * b] +
                     mu * payload[g + space_dimensions * a] *
                          payload[f + space_dimensions * b];
        if (f == g) {
          for (int p = 0; p < space_dimensions; ++p) {
            value += mu * payload[p + space_dimensions * a] *
                          payload[p + space_dimensions * b];
          }
        }
        data(test, trial) = value;
      }
    }
  }
};

template <class T>
auto PrepareEAQData(T &operation, const mfem::FiniteElementSpace &trial,
                    const mfem::FiniteElementSpace &test,
                    const mfem::IntegrationRule &rule, bool use_device) {
  if constexpr (PreferredPAStorage<T> == BlfPAStorage::IsotropicSymmetric &&
                requires { operation.PreparePAData(trial, test, rule, use_device); }) {
    const auto prepared = operation.PreparePAData(trial, test, rule, use_device);
    return IsotropicEAQData<BlfIntegratorTraits<T>, decltype(prepared)>{
        prepared, trial.GetMesh()->SpaceDimension()};
  } else {
    return operation.PrepareQData(trial, test, rule, use_device);
  }
}

/// Stage weighted quadrature data in the contraction's compile-time layout.
/// The caller supplies the synchronization before consuming the staged data.
template <class Traits, bool ComponentMajor, class Prepared>
MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
StageEAQData(int e, int nq, const Prepared &prepared,
             const mfem::real_t *weights, mfem::real_t *staged) {
  constexpr int tc = Traits::TestComponents, rc = Traits::TrialComponents;
  MFEM_FOREACH_THREAD(q, x, nq) {
    mfem::real_t point[tc * rc];
    prepared.AssembleQData(e, q, BlfQDataMatrix<tc, rc>(point));
    for (int c = 0; c < tc * rc; ++c) {
      const int index = ComponentMajor ? q + nq * c : c + tc * rc * q;
      staged[index] = point[c] * weights[q];
    }
  }
}

/// Scalar tensor EA extents derived from the independent axis sizes.
template <class Traits>
struct TensorEASizes {
  int trial_basis_size, staged_trial_size, test_basis_size, staged_test_size;
  int qdata_size, stage0_size, stage1_size, matrix_size;
  int quadrature_points, pair_dofs;

  /// @param trial_d1d Trial one-dimensional basis size.
  /// @param test_d1d Test one-dimensional basis size.
  /// @param q1d One-dimensional quadrature size.
  MFEM_HOST_DEVICE constexpr TensorEASizes(int trial_d1d, int test_d1d,
                                          int q1d) {
    constexpr int dims = Traits::Dims;
    constexpr bool trial_uses_derivatives =
        Traits::trial_operator == BlfFieldOperator::Gradient;
    constexpr bool test_uses_derivatives =
        Traits::test_operator == BlfFieldOperator::Gradient;
    trial_basis_size = q1d * trial_d1d;
    test_basis_size = q1d * test_d1d;
    staged_trial_size = trial_basis_size * (trial_uses_derivatives ? 2 : 1);
    staged_test_size = test_basis_size * (test_uses_derivatives ? 2 : 1);
    quadrature_points = TensorIntPow(q1d, dims);
    qdata_size = quadrature_points * Traits::TrialComponents *
                 Traits::TestComponents;
    pair_dofs = test_d1d * trial_d1d;
    matrix_size = TensorIntPow(pair_dofs, dims);
    stage0_size = dims == 1 ? 0 : pair_dofs * TensorIntPow(q1d, dims - 1);
    stage1_size = dims == 3 ? pair_dofs * pair_dofs * q1d : 0;
  }
};

/// Per-element scalar tensor EA kernel; stages basis/qdata in block workspace.
/// @tparam Prepared Trivially copyable device quadrature assembler.
/// @tparam Entry Device callable accumulating one contracted matrix entry.
template <class Traits, class Prepared, class Entry, int FixedD1D, int FixedQ1D,
          int FixedTestD1D>
struct TensorEAWorkFunctor {
  int test_d1d;
  int trial_d1d;
  int q1d;
  const mfem::real_t *trial_values;
  const mfem::real_t *trial_derivatives;
  const mfem::real_t *test_values;
  const mfem::real_t *test_derivatives;
  const mfem::real_t *quadrature_weights;
  Prepared prepared_qdata;
  Entry add_entry;
  int element_stride = 1;

  /// @param begin First element index assigned to this block.
  /// @param end Exclusive end of the block's element range.
  /// @param raw_workspace Per-block scratch buffer sized by the EA launcher.
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *raw_workspace) const {
    const int trial_d1d = FixedD1D ? FixedD1D : this->trial_d1d;
    const int test_d1d = FixedTestD1D ? FixedTestD1D : this->test_d1d;
    const int q1d = FixedQ1D ? FixedQ1D : this->q1d;
    constexpr int dims = Traits::Dims;
    constexpr int trial_components = Traits::TrialComponents;
    constexpr int test_components = Traits::TestComponents;
    constexpr bool trial_uses_derivatives =
        Traits::trial_operator == BlfFieldOperator::Gradient;
    constexpr bool test_uses_derivatives =
        Traits::test_operator == BlfFieldOperator::Gradient;
    const auto [trial_basis_size, staged_trial_size, test_basis_size,
                staged_test_size, qdata_size, stage0_size, stage1_size,
                matrix_size, quadrature_points, pair_dofs] =
        TensorEASizes<Traits>(trial_d1d, test_d1d, q1d);
    auto *workspace = reinterpret_cast<mfem::real_t *>(raw_workspace);
    auto *staged_trial = workspace;
    auto *staged_trial_derivatives = staged_trial + trial_basis_size;
    auto *staged_test = workspace + staged_trial_size;
    auto *staged_test_derivatives = staged_test + test_basis_size;
    auto *staged_qdata = workspace + staged_trial_size + staged_test_size;
    auto *stage0 = staged_qdata + qdata_size;
    auto *stage1 = stage0 + stage0_size;
    MFEM_FOREACH_THREAD(i, x, trial_basis_size) {
      staged_trial[i] = trial_values[i];
      if constexpr (trial_uses_derivatives) {
        staged_trial_derivatives[i] = trial_derivatives[i];
      }
    }
    MFEM_FOREACH_THREAD(i, x, test_basis_size) {
      staged_test[i] = test_values[i];
      if constexpr (test_uses_derivatives) {
        staged_test_derivatives[i] = test_derivatives[i];
      }
    }
    MFEM_SYNC_THREAD;
    for (int element = begin; element < end; element += element_stride) {
      add_entry.Initialize(element);
      StageEAQData<Traits, false>(element, quadrature_points, prepared_qdata,
                                  quadrature_weights, staged_qdata);
      MFEM_SYNC_THREAD;
      for (int test_component = 0; test_component < test_components;
           ++test_component) {
        for (int trial_component = 0; trial_component < trial_components;
             ++trial_component) {
          if constexpr (dims == 1) {
            MFEM_FOREACH_THREAD_2D(test_x, trial_x, x, test_d1d, trial_d1d) {
              if (!add_entry.StoresEntry(test_x, trial_x)) { continue; }
              mfem::real_t value = 0.0;
              for (int qx = 0; qx < q1d; ++qx) {
                const int qdata_index =
                    trial_component +
                    trial_components *
                        (test_component + test_components * qx);
                value += TensorFactor<Traits::test_operator>(
                             staged_test, staged_test_derivatives, qx, test_x,
                             test_component, 0, q1d) *
                         staged_qdata[qdata_index] *
                         TensorFactor<Traits::trial_operator>(
                             staged_trial, staged_trial_derivatives, qx,
                             trial_x, trial_component, 0, q1d);
              }
              add_entry(element, test_x, trial_x, value);
            }
          } else {
            const int nqrest = TensorIntPow(q1d, dims - 1);
            MFEM_FOREACH_THREAD_3D(test_x, trial_x, qrest, x, test_d1d,
                                   trial_d1d, nqrest) {
              mfem::real_t value = 0.0;
              for (int qx = 0; qx < q1d; ++qx) {
                const int q = qx + q1d * qrest;
                const int qdata_index =
                    trial_component +
                    trial_components * (test_component + test_components * q);
                value += TensorFactor<Traits::test_operator>(
                             staged_test, staged_test_derivatives, qx, test_x,
                             test_component, 0, q1d) *
                         staged_qdata[qdata_index] *
                         TensorFactor<Traits::trial_operator>(
                             staged_trial, staged_trial_derivatives, qx,
                             trial_x, trial_component, 0, q1d);
              }
              stage0[test_x + test_d1d * (trial_x + trial_d1d * qrest)] =
                  value;
            }
            MFEM_SYNC_THREAD;
            if constexpr (dims == 2) {
              MFEM_FOREACH_THREAD_4D(test_x, test_y, trial_x, trial_y, x,
                                     test_d1d, test_d1d, trial_d1d,
                                     trial_d1d) {
                const int test_lex = test_x + test_y * test_d1d;
                const int trial_lex = trial_x + trial_y * trial_d1d;
                if (!add_entry.StoresEntry(test_lex, trial_lex)) { continue; }
                const int pair_x = test_x + test_d1d * trial_x;
                mfem::real_t value = 0.0;
                for (int qy = 0; qy < q1d; ++qy) {
                  value += TensorFactor<Traits::test_operator>(
                               staged_test, staged_test_derivatives, qy,
                               test_y, test_component, 1, q1d) *
                           stage0[pair_x + pair_dofs * qy] *
                           TensorFactor<Traits::trial_operator>(
                               staged_trial, staged_trial_derivatives, qy,
                               trial_y, trial_component, 1, q1d);
                }
                add_entry(element, test_lex, trial_lex, value);
              }
            } else {
              MFEM_FOREACH_THREAD_4D(pair_x, test_y, trial_y, qz, x,
                                     pair_dofs, test_d1d, trial_d1d, q1d) {
                mfem::real_t value = 0.0;
                for (int qy = 0; qy < q1d; ++qy) {
                  value += TensorFactor<Traits::test_operator>(
                               staged_test, staged_test_derivatives, qy,
                               test_y, test_component, 1, q1d) *
                           stage0[pair_x + pair_dofs * (qy + q1d * qz)] *
                           TensorFactor<Traits::trial_operator>(
                               staged_trial, staged_trial_derivatives, qy,
                               trial_y, trial_component, 1, q1d);
                }
                stage1[pair_x +
                       pair_dofs *
                           (test_y + test_d1d * (trial_y + trial_d1d * qz))] =
                    value;
              }
              MFEM_SYNC_THREAD;
              MFEM_FOREACH_THREAD_6D(test_x, test_y, test_z, trial_x, trial_y,
                                     trial_z, x, test_d1d, test_d1d, test_d1d,
                                     trial_d1d, trial_d1d, trial_d1d) {
                const int test_lex =
                    test_x + test_d1d * (test_y + test_d1d * test_z);
                const int trial_lex =
                    trial_x + trial_d1d * (trial_y + trial_d1d * trial_z);
                if (!add_entry.StoresEntry(test_lex, trial_lex)) { continue; }
                const int pair_x = test_x + test_d1d * trial_x;
                const int pair_y = test_y + test_d1d * trial_y;
                mfem::real_t value = 0.0;
                for (int qz = 0; qz < q1d; ++qz) {
                  value +=
                      TensorFactor<Traits::test_operator>(
                          staged_test, staged_test_derivatives, qz, test_z,
                          test_component, 2, q1d) *
                      stage1[pair_x + pair_dofs * (pair_y + pair_dofs * qz)] *
                      TensorFactor<Traits::trial_operator>(
                          staged_trial, staged_trial_derivatives, qz, trial_z,
                          trial_component, 2, q1d);
                }
                add_entry(element, test_lex, trial_lex, value);
              }
            }
          }
          MFEM_SYNC_THREAD;
        }
      }
    }
  }
};

/// Batch scalar value/value elements in independent y lanes. Every lane has
/// the same number of elements, so block-wide contraction barriers coincide.
template <class Kernel> struct BatchedTensorEAWorkFunctor {
  Kernel kernel;
  int batch;
  int workspace_reals;
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *workspace) const {
    MFEM_FOREACH_THREAD(slot, y, batch) {
      kernel(begin * batch + slot, end * batch + slot,
             workspace + slot * workspace_reals * sizeof(mfem::real_t));
    }
  }
};

/// Native-vector/scalar tensor EA extents derived from closed axis sizes.
template <class Traits>
struct OperatorTensorEASizes {
  int trial_o1d, test_o1d, trial_component_dofs, test_component_dofs;
  int nq, matrix_size, qdata_size, stage0_size, stage1_size;

  /// @param trial_d1d Trial closed one-dimensional basis size.
  /// @param test_d1d Test closed one-dimensional basis size.
  /// @param q1d One-dimensional quadrature size.
  MFEM_HOST_DEVICE constexpr OperatorTensorEASizes(int trial_d1d,
                                                  int test_d1d, int q1d) {
    constexpr int dims = Traits::Dims;
    trial_o1d =
        trial_d1d -
        (internal::IsVectorTensorMap(Traits::trial_type) ? 1 : 0);
    test_o1d =
        test_d1d -
        (internal::IsVectorTensorMap(Traits::test_type) ? 1 : 0);
    if constexpr (Traits::trial_type == mfem::FiniteElement::H_CURL) {
      trial_component_dofs = trial_o1d * TensorIntPow(trial_d1d, dims - 1);
    } else if constexpr (Traits::trial_type == mfem::FiniteElement::H_DIV) {
      trial_component_dofs = trial_d1d * TensorIntPow(trial_o1d, dims - 1);
    } else {
      trial_component_dofs = TensorIntPow(trial_d1d, dims);
    }
    if constexpr (Traits::test_type == mfem::FiniteElement::H_CURL) {
      test_component_dofs = test_o1d * TensorIntPow(test_d1d, dims - 1);
    } else if constexpr (Traits::test_type == mfem::FiniteElement::H_DIV) {
      test_component_dofs = test_d1d * TensorIntPow(test_o1d, dims - 1);
    } else {
      test_component_dofs = TensorIntPow(test_d1d, dims);
    }
    constexpr int trial_fields =
        internal::IsVectorTensorMap(Traits::trial_type) ? dims : Traits::TrialVDim;
    constexpr int test_fields =
        internal::IsVectorTensorMap(Traits::test_type) ? dims : Traits::TestVDim;
    matrix_size = trial_component_dofs * trial_fields * test_component_dofs *
                  test_fields;
    nq = TensorIntPow(q1d, dims);
    qdata_size = nq * Traits::TrialComponents * Traits::TestComponents;
    const int max_pair = trial_d1d * test_d1d;
    stage0_size = max_pair * TensorIntPow(q1d, dims - 1);
    stage1_size = dims == 3 ? max_pair * max_pair * q1d : 0;
  }
};

/// Tensor EA kernel using operator terms and independent closed axis sizes.
/// @tparam Prepared Trivially copyable device quadrature assembler.
/// @tparam Entry Device callable accumulating one contracted matrix entry.
template <class Traits, class Prepared, class Entry> struct OperatorTensorEAWorkFunctor {
  int test_d1d;
  int trial_d1d;
  int q1d;
  const mfem::real_t *weights;
  const mfem::real_t *test_closed_derivatives;
  const mfem::real_t *test_closed_values;
  const mfem::real_t *test_open_values;
  const mfem::real_t *trial_closed_derivatives;
  const mfem::real_t *trial_closed_values;
  const mfem::real_t *trial_open_values;
  Prepared prepared;
  Entry add_entry;

  /// @param begin First element index assigned to this block.
  /// @param end Exclusive end of the block's element range.
  /// @param raw_workspace Per-block scratch buffer sized by the EA launcher.
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *raw_workspace) const {
    constexpr int dims = Traits::Dims;
    constexpr int trial_components = Traits::TrialComponents;
    constexpr int test_components = Traits::TestComponents;
    const auto [trial_o1d, test_o1d, trial_component_dofs,
                test_component_dofs, nq, matrix_size, qdata_size,
                stage0_size, stage1_size] =
        OperatorTensorEASizes<Traits>(trial_d1d, test_d1d, q1d);
    auto *qdata = reinterpret_cast<mfem::real_t *>(raw_workspace);
    auto *stage0 = qdata + qdata_size;
    auto *stage1 = stage0 + stage0_size;
    for (int e = begin; e < end; ++e) {
      add_entry.Initialize(e);
      StageEAQData<Traits, false>(e, nq, prepared, weights, qdata);
      MFEM_SYNC_THREAD;

      for (int test_component = 0; test_component < test_components;
           ++test_component) {
        constexpr auto test_operator = ReferenceFieldOperator(
            Traits::test_operator, Traits::test_type, Traits::TestVDim,
            Traits::TestOutOfPlane);
        const int test_terms = internal::TensorOperatorTermCount(
            test_operator, dims);
        for (int test_term_index = 0; test_term_index < test_terms;
             ++test_term_index) {
          const auto test_term =
              internal::TensorOperatorTerm(test_operator, dims,
                                           test_component, test_term_index);
          const bool test_closed_x = internal::TensorFieldUsesClosedMap(
              Traits::test_type, test_term.field_component, 0);
          const bool test_closed_y = internal::TensorFieldUsesClosedMap(
              Traits::test_type, test_term.field_component, 1);
          const bool test_closed_z =
              dims == 3 &&
              internal::TensorFieldUsesClosedMap(
                  Traits::test_type, test_term.field_component, 2);
          const int tx = test_closed_x ? test_d1d : test_o1d;
          const int ty = test_closed_y ? test_d1d : test_o1d;
          const int tz =
              dims == 3 ? (test_closed_z ? test_d1d : test_o1d) : 1;
          const auto *tbx =
              test_term.derivative_dimension == 0
                  ? test_closed_derivatives
                  : (test_closed_x ? test_closed_values : test_open_values);
          const auto *tby =
              test_term.derivative_dimension == 1
                  ? test_closed_derivatives
                  : (test_closed_y ? test_closed_values : test_open_values);
          const auto *tbz =
              test_term.derivative_dimension == 2
                  ? test_closed_derivatives
                  : (test_closed_z ? test_closed_values : test_open_values);

          for (int trial_component = 0; trial_component < trial_components;
               ++trial_component) {
            constexpr auto trial_operator = ReferenceFieldOperator(
                Traits::trial_operator, Traits::trial_type, Traits::TrialVDim,
                Traits::TrialOutOfPlane);
            const int trial_terms = internal::TensorOperatorTermCount(
                trial_operator, dims);
            for (int trial_term_index = 0; trial_term_index < trial_terms;
                 ++trial_term_index) {
              const auto trial_term = internal::TensorOperatorTerm(
                  trial_operator, dims, trial_component,
                  trial_term_index);
              const bool trial_closed_x = internal::TensorFieldUsesClosedMap(
                  Traits::trial_type, trial_term.field_component, 0);
              const bool trial_closed_y = internal::TensorFieldUsesClosedMap(
                  Traits::trial_type, trial_term.field_component, 1);
              const bool trial_closed_z =
                  dims == 3 &&
                  internal::TensorFieldUsesClosedMap(
                      Traits::trial_type, trial_term.field_component, 2);
              const int rx = trial_closed_x ? trial_d1d : trial_o1d;
              const int ry = trial_closed_y ? trial_d1d : trial_o1d;
              const int rz =
                  dims == 3 ? (trial_closed_z ? trial_d1d : trial_o1d) : 1;
              const auto *rbx = trial_term.derivative_dimension == 0
                                    ? trial_closed_derivatives
                                    : (trial_closed_x ? trial_closed_values
                                                      : trial_open_values);
              const auto *rby = trial_term.derivative_dimension == 1
                                    ? trial_closed_derivatives
                                    : (trial_closed_y ? trial_closed_values
                                                      : trial_open_values);
              const auto *rbz = trial_term.derivative_dimension == 2
                                    ? trial_closed_derivatives
                                    : (trial_closed_z ? trial_closed_values
                                                      : trial_open_values);
              const int pair_x_size = tx * rx;
              const int nqrest = TensorIntPow(q1d, dims - 1);
              MFEM_FOREACH_THREAD_3D(test_x, trial_x, qrest, x, tx, rx,
                                     nqrest) {
                mfem::real_t value = 0.0;
                for (int qx = 0; qx < q1d; ++qx) {
                  const int q = qx + q1d * qrest;
                  const int qdata_index =
                      trial_component +
                      trial_components *
                          (test_component + test_components * q);
                  value += tbx[qx + q1d * test_x] * qdata[qdata_index] *
                           rbx[qx + q1d * trial_x];
                }
                stage0[test_x + (trial_x + qrest * rx) * tx] = value;
              }
              MFEM_SYNC_THREAD;

              if constexpr (dims == 1) {
                MFEM_FOREACH_THREAD_2D(test_x, trial_x, x, tx, rx) {
                  add_entry(e,
                      test_x + test_component_dofs * test_term.field_component,
                      trial_x + trial_component_dofs * trial_term.field_component,
                      test_term.sign * trial_term.sign *
                          stage0[test_x + tx * trial_x]);
                }
              } else if constexpr (dims == 2) {
                MFEM_FOREACH_THREAD_4D(test_x, test_y, trial_x, trial_y, x,
                                       tx, ty, rx, ry) {
                  const int pair_x = test_x + tx * trial_x;
                  mfem::real_t value = 0.0;
                  for (int qy = 0; qy < q1d; ++qy) {
                    value += tby[qy + q1d * test_y] *
                             stage0[pair_x + pair_x_size * qy] *
                             rby[qy + q1d * trial_y];
                  }
                  const int test_local = test_x + tx * test_y;
                  const int trial_local = trial_x + rx * trial_y;
                  add_entry(e,
                            test_local + test_component_dofs *
                                             test_term.field_component,
                            trial_local + trial_component_dofs *
                                              trial_term.field_component,
                            test_term.sign * trial_term.sign * value);
                }
              } else {
                const int pair_y_size = ty * ry;
                MFEM_FOREACH_THREAD_4D(pair_x, test_y, trial_y, qz, x,
                                       pair_x_size, ty, ry, q1d) {
                  mfem::real_t value = 0.0;
                  for (int qy = 0; qy < q1d; ++qy) {
                    value += tby[qy + q1d * test_y] *
                             stage0[pair_x + pair_x_size * (qy + q1d * qz)] *
                             rby[qy + q1d * trial_y];
                  }
                  stage1[pair_x + (test_y + (trial_y + qz * ry) * ty) *
                                      pair_x_size] = value;
                }
                MFEM_SYNC_THREAD;
                MFEM_FOREACH_THREAD_6D(test_x, test_y, test_z, trial_x,
                                       trial_y, trial_z, x, tx, ty, tz, rx,
                                       ry, rz) {
                  const int pair_x = test_x + tx * trial_x;
                  const int pair_y = test_y + ty * trial_y;
                  mfem::real_t value = 0.0;
                  for (int qz = 0; qz < q1d; ++qz) {
                    value +=
                        tbz[qz + q1d * test_z] *
                        stage1[pair_x +
                               pair_x_size * (pair_y + pair_y_size * qz)] *
                        rbz[qz + q1d * trial_z];
                  }
                  const int test_local = test_x + tx * (test_y + ty * test_z);
                  const int trial_local =
                      trial_x + rx * (trial_y + ry * trial_z);
                  add_entry(e,
                            test_local + test_component_dofs *
                                             test_term.field_component,
                            trial_local + trial_component_dofs *
                                              trial_term.field_component,
                            test_term.sign * trial_term.sign * value);
                }
              }
              MFEM_SYNC_THREAD;
            }
          }
        }
      }
    }
  }
};

/// Launch any EA kernel with a common shared/global scratch policy.
template <class Kernel>
void LaunchEA(Kernel kernel, int groups, int threads, int batch,
                  std::size_t workspace_bytes, bool use_device) {
  MFEM_VERIFY(workspace_bytes <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "per-block element-assembly workspace exceeds INT_MAX");
  const int blocks = std::min(forall_nested_max_concurrent_blocks(), groups);
  const auto global_bytes = forall_nested_global_workspace_size(
      groups, blocks, static_cast<int>(workspace_bytes));
  const auto global_reals = (global_bytes + sizeof(mfem::real_t) - 1) / sizeof(mfem::real_t);
  MFEM_VERIFY(global_reals <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "global element-assembly workspace exceeds INT_MAX");
  const int reals = use_device && workspace_bytes > forall_nested_shared_memory_limit
      ? static_cast<int>(global_reals) : 0;
  mfem::Vector workspace(reals);
  workspace.UseDevice(true);
  auto *buffer = reals ? reinterpret_cast<char *>(workspace.Write(true)) : nullptr;
  forall_nested_launch_bounds_workspace<256>(
      kernel, kernel, groups, threads, batch, 1, blocks,
      static_cast<int>(workspace_bytes), buffer, reals * sizeof(mfem::real_t));
}

/// Evaluate native basis rows while storing directly in the requested EA order.
/// Signed lexicographic maps apply independently within every vdim copy.
struct DirectFullEAOutput {
  const int *test_map;
  const int *trial_map;
  int test_scalar_dofs;
  int trial_scalar_dofs;
  bool add;

  MFEM_HOST_DEVICE int Signed(int dof, const int *map, int scalar_dofs) const {
    return map ? map[dof % scalar_dofs] : dof;
  }
  MFEM_HOST_DEVICE int Native(int dof, const int *map, int scalar_dofs) const {
    if (!map) { return dof; }
    const int signed_dof = Signed(dof, map, scalar_dofs);
    return dof / scalar_dofs * scalar_dofs +
        (signed_dof >= 0 ? signed_dof : -1 - signed_dof);
  }
  MFEM_HOST_DEVICE int TestDof(int dof) const {
    return Native(dof, test_map, test_scalar_dofs);
  }
  MFEM_HOST_DEVICE int TrialDof(int dof) const {
    return Native(dof, trial_map, trial_scalar_dofs);
  }
  MFEM_HOST_DEVICE void Store(mfem::real_t *matrices, int e, int test,
      int trial, int test_dofs, int trial_dofs, mfem::real_t value) const {
    value *= (Signed(test, test_map, test_scalar_dofs) >= 0 ? 1.0 : -1.0) *
             (Signed(trial, trial_map, trial_scalar_dofs) >= 0 ? 1.0 : -1.0);
    const int index = trial + trial_dofs * (test + test_dofs * e);
    if (add) { matrices[index] += value; }
    else { matrices[index] = value; }
  }
  MFEM_HOST_DEVICE void operator()(mfem::real_t *matrices, int e, int test,
      int trial, int test_dofs, int trial_dofs, bool symmetric,
      mfem::real_t value) const {
    Store(matrices, e, test, trial, test_dofs, trial_dofs, value);
    if (symmetric && test != trial) {
      Store(matrices, e, trial, test, test_dofs, trial_dofs, value);
    }
  }
};

/// Contract one full-basis test/trial DOF pair against staged weighted qdata.
template <class Traits, class Output,
          int FixedTrialDofs = 0, int FixedTestDofs = 0, int FixedNQ = 0>
struct DirectFullEAEntryFunctor {
  using TrialBasis =
      BlfReferenceBasis<Traits::TrialBasisOperator, Traits::trial_type,
                        Traits::TrialComponents, Traits::TrialVDim>;
  using TestBasis =
      BlfReferenceBasis<Traits::TestBasisOperator, Traits::test_type,
                        Traits::TestComponents, Traits::TestVDim>;

  int quadrature_points;
  int test_dofs;
  int trial_dofs;
  mfem::real_t *matrices;
  bool symmetric;
  Output output{};

  MFEM_HOST_DEVICE int TrialDofs() const { return FixedTrialDofs ? FixedTrialDofs : trial_dofs; }
  MFEM_HOST_DEVICE int TestDofs() const { return FixedTestDofs ? FixedTestDofs : test_dofs; }
  MFEM_HOST_DEVICE int NQ() const { return FixedNQ ? FixedNQ : quadrature_points; }

  /// @tparam Pointer Indexable scalar scratch-buffer type.
  /// @param e Mesh element index.
  /// @param test_dof Local test row.
  /// @param trial_dof Local trial column.
  /// @param workspace Staged trial/test bases followed by weighted qdata.
  template <class Pointer>
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int e, int test_dof, int trial_dof, Pointer workspace) const {
    if (symmetric && test_dof > trial_dof) { return; }
    const int trial_dofs = TrialDofs(), test_dofs = TestDofs(), quadrature_points = NQ();
    const int test_native = output.TestDof(test_dof);
    const int trial_native = output.TrialDof(trial_dof);
    constexpr int trial_components = Traits::TrialComponents;
    constexpr int test_components = Traits::TestComponents;
    const int trial_basis_size =
        trial_dofs * trial_components * quadrature_points;
    const int test_basis_size =
        test_dofs * test_components * quadrature_points;
    const auto *staged_trial = workspace;
    const auto *staged_test = staged_trial + trial_basis_size;
    const auto *staged_qdata = staged_test + test_basis_size;
    mfem::real_t value = 0.0;
    for (int q = 0; q < quadrature_points; ++q) {
      for (int test_component = 0; test_component < test_components;
           ++test_component) {
        const auto test_value = TestBasis::Eval(
            staged_test, q, test_component, test_native, quadrature_points);
        for (int trial_component = 0; trial_component < trial_components;
             ++trial_component) {
          const int qdata_index =
              trial_component +
              trial_components * (test_component + test_components * q);
          value += test_value * staged_qdata[qdata_index] *
                   TrialBasis::Eval(staged_trial, q, trial_component,
                                    trial_native, quadrature_points);
        }
      }
    }
    output(matrices, e, test_dof, trial_dof, test_dofs, trial_dofs, symmetric, value);
  }
};

/// Full-basis EA block kernel staging quadrature data before entry contraction.
/// @tparam Prepared Trivially copyable device quadrature assembler.
template <class Traits, class Prepared, class Output,
          int FixedTrialDofs = 0, int FixedTestDofs = 0, int FixedNQ = 0>
struct DirectFullEAWorkFunctor {
  const mfem::real_t *test_values;
  const mfem::real_t *quadrature_weights;
  const mfem::real_t *trial_values;
  Prepared prepared_qdata;
  DirectFullEAEntryFunctor<Traits, Output, FixedTrialDofs, FixedTestDofs, FixedNQ> assemble_entry;

  /// @param begin First element index assigned to this block.
  /// @param end Exclusive end of the block's element range.
  /// @param raw_workspace Per-block scratch buffer sized by the EA launcher.
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *raw_workspace) const {
    constexpr int trial_components = Traits::TrialComponents;
    constexpr int test_components = Traits::TestComponents;
    const int test_dofs = assemble_entry.TestDofs();
    const int trial_dofs = assemble_entry.TrialDofs();
    const int quadrature_points = assemble_entry.NQ();
    const int test_basis_size =
        test_dofs * test_components * quadrature_points;
    const int trial_basis_size =
        trial_dofs * trial_components * quadrature_points;
    auto *workspace = reinterpret_cast<mfem::real_t *>(raw_workspace);
    MFEM_FOREACH_THREAD(i, x, trial_basis_size) { workspace[i] = trial_values[i]; }
    MFEM_FOREACH_THREAD(i, x, test_basis_size) { workspace[trial_basis_size + i] = test_values[i]; }
    MFEM_SYNC_THREAD;
    for (int e = begin; e < end; ++e) {
      auto *staged_qdata = workspace + trial_basis_size + test_basis_size;
      StageEAQData<Traits, false>(e, quadrature_points, prepared_qdata,
                                  quadrature_weights, staged_qdata);
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_2D(test_dof, trial_dof, x, test_dofs, trial_dofs) {
        assemble_entry(e, test_dof, trial_dof, workspace);
      }
      MFEM_SYNC_THREAD;
    }
  }
};

/// Tile test rows contiguously while retaining quadrature/component ordering.
/// Use the same layout for staging, contraction, and host workspace sizing.
template <int Tile> struct EARowLayout {
  MFEM_HOST_DEVICE static constexpr int Index(int dof, int q, int c,
                                             int nq, int components) {
    return dof % Tile + Tile * (q + nq * (c + components * (dof / Tile)));
  }

  template <class SizeType = int>
  MFEM_HOST_DEVICE static constexpr SizeType Size(int dofs, int nq,
                                                  int components) {
    return static_cast<SizeType>((dofs + Tile - 1) / Tile) * Tile * components * nq;
  }
};

using FullEARowLayout = EARowLayout<8>;

/// Contract a test basis against the reusable quadrature action D B_trial.
template <class Traits, class Output, int FixedTrialDofs, int FixedTestDofs, int FixedNQ>
struct FullEAEntryFunctor {
  using TestBasis = BlfReferenceBasis<Traits::TestBasisOperator,
      Traits::test_type, Traits::TestComponents, Traits::TestVDim>;
  int quadrature_points;
  int test_dofs;
  int trial_dofs;
  mfem::real_t *matrices;
  bool symmetric;
  Output output{};

  MFEM_HOST_DEVICE int TrialDofs() const { return FixedTrialDofs ? FixedTrialDofs : trial_dofs; }
  MFEM_HOST_DEVICE int TestDofs() const { return FixedTestDofs ? FixedTestDofs : test_dofs; }
  MFEM_HOST_DEVICE int NQ() const { return FixedNQ ? FixedNQ : quadrature_points; }

  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int e, int test_dof, int trial_dof,
             const mfem::real_t *test_basis,
             const mfem::real_t *action) const {
    if (symmetric && test_dof > trial_dof) { return; }
    const int trial_dofs = TrialDofs(), test_dofs = TestDofs(), quadrature_points = NQ();
    const int test_native = output.TestDof(test_dof);
    const int trial_native = output.TrialDof(trial_dof);
    mfem::real_t value = 0.0;
    for (int c = 0; c < Traits::TestComponents; ++c) {
      for (int q = 0; q < quadrature_points; ++q) {
        const int index = FullEARowLayout::Index(
            test_native, q, c, quadrature_points, Traits::TestComponents);
        const auto basis = test_basis[index];
        // Native vector and component-field bases contain entire zero blocks.
        if (basis != 0.0) {
          value += basis * action[q + quadrature_points *
                                      (c + Traits::TestComponents * trial_native)];
        }
      }
    }
    output(matrices, e, test_dof, trial_dof, test_dofs, trial_dofs, symmetric, value);
  }
};

/// Full-basis EA reuses staged bases and each trial quadrature action.
template <class Traits, class Prepared, class Output,
          int FixedTrialDofs, int FixedTestDofs, int FixedNQ>
struct FullEAWorkFunctor {
  const mfem::real_t *test_values;
  const mfem::real_t *quadrature_weights;
  const mfem::real_t *trial_values;
  Prepared prepared_qdata;
  FullEAEntryFunctor<Traits, Output, FixedTrialDofs, FixedTestDofs, FixedNQ> assemble_entry;

  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *raw_workspace) const {
    using TrialBasis = BlfReferenceBasis<Traits::TrialBasisOperator,
        Traits::trial_type, Traits::TrialComponents, Traits::TrialVDim>;
    constexpr int tc = Traits::TestComponents, rc = Traits::TrialComponents;
    const int nq = assemble_entry.NQ();
    const int trial_dofs = assemble_entry.TrialDofs();
    const int test_dofs = assemble_entry.TestDofs();
    const int trial_size = trial_dofs * rc * nq;
    const int test_size = FullEARowLayout::Size(test_dofs, nq, tc);
    auto *trial_basis = reinterpret_cast<mfem::real_t *>(raw_workspace);
    auto *test_basis = trial_basis + trial_size;
    auto *qdata = test_basis + test_size;
    auto *action = qdata + nq * rc * tc;
    MFEM_FOREACH_THREAD(i, x, trial_size) { trial_basis[i] = trial_values[i]; }
    MFEM_FOREACH_THREAD_3D(q, c, dof, x, nq, tc, test_dofs) {
      const int index = FullEARowLayout::Index(dof, q, c, nq, tc);
      test_basis[index] = test_values[q + nq * (c + tc * dof)];
    }
    MFEM_SYNC_THREAD;
    for (int e = begin; e < end; ++e) {
      StageEAQData<Traits, true>(e, nq, prepared_qdata, quadrature_weights, qdata);
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_3D(q, c, dof, x, nq, tc, trial_dofs) {
        mfem::real_t value = 0.0;
        for (int r = 0; r < rc; ++r) {
          const auto basis = TrialBasis::Eval(trial_basis, q, r, dof, nq);
          if (basis != 0.0) { value += qdata[q + nq * (r + rc * c)] * basis; }
        }
        action[q + nq * (c + tc * dof)] = value;
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_2D(test_dof, trial_dof, x, test_dofs, trial_dofs) {
        assemble_entry(e, test_dof, trial_dof, test_basis, action);
      }
      MFEM_SYNC_THREAD;
    }
  }
};

/// Common native/direct-output full-basis launcher. Fixed extents are supplied
/// by the EA dispatch registry; the same contractions serve runtime sizes.
template <class Traits, class Prepared, class Output,
          int FixedTrialDofs = 0, int FixedTestDofs = 0, int FixedNQ = 0>
void LaunchFullEA(int elements, int trial_dofs, int test_dofs, int nq,
    const mfem::real_t *trial, const mfem::real_t *test,
    const mfem::real_t *weights, Prepared prepared, mfem::real_t *matrices,
    bool symmetric, Output output, bool use_device) {
  constexpr int rc = Traits::TrialComponents, tc = Traits::TestComponents;
  const int matrix_size = test_dofs * trial_dofs;
  const std::size_t trial_size = static_cast<std::size_t>(nq) * rc * trial_dofs;
  const std::size_t qdata_size = static_cast<std::size_t>(nq) * rc * tc;
  if constexpr (rc == 1 && tc == 1) {
    DirectFullEAEntryFunctor<Traits, Output, FixedTrialDofs, FixedTestDofs, FixedNQ>
        entry{nq, test_dofs, trial_dofs, matrices, symmetric, output};
    const auto bytes = (trial_size + static_cast<std::size_t>(nq) * tc * test_dofs +
                        qdata_size) * sizeof(mfem::real_t);
    DirectFullEAWorkFunctor<Traits, Prepared, Output, FixedTrialDofs, FixedTestDofs, FixedNQ>
        kernel{test, weights, trial, prepared, entry};
    LaunchEA(kernel, elements, std::min(256, std::max(matrix_size, nq)), 1, bytes, use_device);
  } else {
    const int action_size = nq * tc * trial_dofs;
    const auto bytes = (trial_size + FullEARowLayout::Size<std::size_t>(test_dofs, nq, tc) +
                        qdata_size + action_size) * sizeof(mfem::real_t);
    FullEAEntryFunctor<Traits, Output, FixedTrialDofs, FixedTestDofs, FixedNQ>
        entry{nq, test_dofs, trial_dofs, matrices, symmetric, output};
    FullEAWorkFunctor<Traits, Prepared, Output, FixedTrialDofs, FixedTestDofs, FixedNQ>
        kernel{test, weights, trial, prepared, entry};
    LaunchEA(kernel, elements, std::min(256, std::max({matrix_size, nq, action_size})),
             1, bytes, use_device);
  }
}

/// Store only independent runtime PA sizes and, when needed, dense ordering.
/// @tparam FixedInput Registered input size, or zero for runtime sizing.
/// @tparam FixedOutput Registered output size, or zero for runtime sizing.
/// @tparam FixedQuadrature Registered quadrature size, or zero for runtime.
/// @tparam RuntimeOrdering Whether dense ordering depends on setup strategy.
template <class Traits, int FixedInput, int FixedOutput, int FixedQuadrature,
          bool RuntimeOrdering = false>
struct RuntimeApplyParameters {
  static_assert(Traits::Mixed || !FixedInput || !FixedOutput ||
                FixedInput == FixedOutput);
  constexpr static bool runtime_input =
      FixedInput == 0 && (Traits::Mixed || FixedOutput == 0);
  constexpr static bool runtime_output =
      Traits::Mixed && FixedOutput == 0;
  constexpr static bool runtime_quadrature = FixedQuadrature == 0;
  std::array<int, runtime_input + runtime_output + runtime_quadrature> sizes;
  std::array<bool, RuntimeOrdering ? 1 : 0> ordering;

  /// @param input Input DOFs or closed-axis size; ignored when fixed.
  /// @param output Output size; ignored when fixed or non-mixed.
  /// @param quadrature Quadrature size; ignored when fixed.
  /// @param component_major Dense ordering; ignored when statically known.
  MFEM_HOST_DEVICE constexpr RuntimeApplyParameters(
      int input, int output, int quadrature, bool component_major = false) {
    if constexpr (runtime_input) { sizes[0] = input; }
    if constexpr (runtime_output) { sizes[runtime_input] = output; }
    if constexpr (runtime_quadrature) {
      sizes[runtime_input + runtime_output] = quadrature;
    }
    if constexpr (RuntimeOrdering) { ordering[0] = component_major; }
  }

  /// @return Input size, fixed or obtained from independent runtime storage.
  MFEM_HOST_DEVICE constexpr int Input() const {
    if constexpr (FixedInput != 0) { return FixedInput; }
    else if constexpr (!Traits::Mixed && FixedOutput != 0) {
      return FixedOutput;
    } else { return sizes[0]; }
  }

  /// @return Output size; non-mixed forms reuse the input size.
  MFEM_HOST_DEVICE constexpr int Output() const {
    if constexpr (!Traits::Mixed) { return Input(); }
    else if constexpr (FixedOutput != 0) { return FixedOutput; }
    else { return sizes[runtime_input]; }
  }

  /// @return Quadrature size, fixed or runtime.
  MFEM_HOST_DEVICE constexpr int Quadrature() const {
    if constexpr (FixedQuadrature != 0) { return FixedQuadrature; }
    else { return sizes[runtime_input + runtime_output]; }
  }

  /// @return Runtime dense ordering, or false when no ordering is stored.
  MFEM_HOST_DEVICE constexpr bool ComponentMajor() const {
    if constexpr (RuntimeOrdering) { return ordering[0]; }
    else { return false; }
  }
};

/// Resolve shared extents, explicitly constant for fully registered kernels.
/// @tparam Sizes Extent type with a constexpr constructor.
/// @tparam FixedSizes Registered constructor arguments, or zero for runtime.
/// @tparam RuntimeSizes Deduced integer argument types.
/// @param runtime_sizes Arguments already resolved individually against the
/// registered sizes; used when any dimension remains runtime-sized.
template <class Sizes, int... FixedSizes, class... RuntimeSizes>
MFEM_HOST_DEVICE constexpr Sizes
ResolveApplySizes(RuntimeSizes... runtime_sizes) {
  if constexpr (((FixedSizes != 0) && ...)) {
    constexpr Sizes sizes(FixedSizes...);
    return sizes;
  } else {
    return Sizes(runtime_sizes...);
  }
}

/// Shared full-basis launch/kernel extents, constant for fixed-size kernels.
/// @tparam Transpose Swap trial/test roles.
/// @tparam Scalar Use unpadded scalar tables and scalar batching limits.
/// @tparam FixedNQ Registered quadrature size, or zero for runtime sizing.
template <class Traits, bool Transpose, bool Scalar, int FixedNQ> struct FullApplySizes {
  int input_values, output_values, basis_stride;
  int staged_input_basis_size, staged_output_basis_size;
  bool cache_basis;
  int element_batch;

  /// @param input_dofs Input DOFs per element.
  /// @param output_dofs Output DOFs per element.
  /// @param nq Quadrature points per element.
  MFEM_HOST_DEVICE constexpr FullApplySizes(int input_dofs, int output_dofs,
                                           int nq) {
    constexpr int input_components =
        Transpose ? Traits::TestComponents : Traits::TrialComponents;
    constexpr int output_components =
        Transpose ? Traits::TrialComponents : Traits::TestComponents;
    constexpr int shared_reals = 32 * 1024 / sizeof(mfem::real_t);
    input_values = nq * input_components;
    output_values = nq * output_components;
    basis_stride = Scalar ? nq : nq + 1;
    staged_input_basis_size = input_dofs * input_components * basis_stride;
    staged_output_basis_size =
        BlfShareFullBasis<Traits> ? 0 : output_dofs * output_components * basis_stride;
    const int staged_basis_size =
        staged_input_basis_size + staged_output_basis_size;
    cache_basis = staged_basis_size <= shared_reals / (Scalar ? 2 : 4);
    const int available_reals =
        shared_reals - (cache_basis ? staged_basis_size : 0);
    if constexpr (Scalar) {
      constexpr int target_work = FixedNQ == 24 ? 768 : 1024;
      const int max_batch =
          std::max(1, std::min(256, target_work / std::max(nq, output_dofs)));
      element_batch = std::max(1, std::min(max_batch, available_reals / nq));
    } else {
      element_batch = std::max(
          1, std::min(256, available_reals / (input_values + output_values)));
    }
  }
};

template <class Traits, bool Transpose> struct ScalarTensorFieldSizes {
  constexpr static int InputFields =
      Transpose ? Traits::TestVDim : Traits::TrialVDim;
  constexpr static int OutputFields =
      Transpose ? Traits::TrialVDim : Traits::TestVDim;
  constexpr static bool InputGradient =
      (Transpose ? Traits::test_operator : Traits::trial_operator) !=
      BlfFieldOperator::Value;
  constexpr static bool OutputGradient =
      (Transpose ? Traits::trial_operator : Traits::test_operator) !=
      BlfFieldOperator::Value;
  constexpr static int InputXChannels = InputGradient ? 2 : 1;
  constexpr static int InputYChannels = InputGradient ? 3 : 1;
  constexpr static int OutputXChannels = OutputGradient ? 2 : 1;
  constexpr static int OutputYChannels = OutputGradient ? 3 : 1;
  int input_dofs, output_dofs, nq, input_stage_x, input_stage_y;
  int output_stage_x, output_stage_y, buffer_x_size, buffer_y_size;
  int element_batch;

  MFEM_HOST_DEVICE constexpr ScalarTensorFieldSizes(int input_d1d,
                                                    int output_d1d, int q1d)
      : input_dofs(input_d1d * input_d1d * input_d1d),
        output_dofs(output_d1d * output_d1d * output_d1d), nq(q1d * q1d * q1d),
        input_stage_x(q1d * input_d1d * input_d1d),
        input_stage_y(q1d * q1d * input_d1d),
        output_stage_x(q1d * output_d1d * output_d1d),
        output_stage_y(q1d * q1d * output_d1d),
        buffer_x_size(std::max({InputFields * InputXChannels * input_stage_x,
                                OutputFields * OutputXChannels * output_stage_x,
                                OutputFields * OutputYChannels * nq})),
        buffer_y_size(
            std::max(InputFields * InputYChannels * input_stage_y,
                     OutputFields * OutputYChannels * output_stage_y)) {
    constexpr int shared_reals = 32 * 1024 / sizeof(mfem::real_t);
    const int parallel_work = std::max({nq, input_dofs, output_dofs});
    const int preferred_batch = std::max(1, std::min(32, 256 / parallel_work));
    element_batch = std::max(
        1, std::min(preferred_batch,
                    shared_reals / (buffer_x_size + buffer_y_size)));
  }
};

template <class Traits, bool Transpose, int FixedD1DIn = 0, int FixedD1DOut = 0,
          int FixedQ1D = 0, BlfPAStorage Storage = BlfPAStorage::Dense>
struct ScalarTensorFieldApplyFunctor {
  const mfem::real_t *input_basis, *input_derivative, *output_basis,
      *output_derivative, *qdata, *input;
  mfem::real_t *output;
  mfem::real_t multiplier;
  RuntimeApplyParameters<Traits, FixedD1DIn, FixedD1DOut, FixedQ1D> parameters;

  MFEM_HOST_DEVICE void operator()(int begin, int end, char *workspace) const {
    const int input_d1d = parameters.Input();
    const int output_d1d = parameters.Output();
    const int q1d = parameters.Quadrature();
    using Sizes = ScalarTensorFieldSizes<Traits, Transpose>;
    const auto sizes =
        ResolveApplySizes<Sizes, FixedD1DIn, FixedD1DOut, FixedQ1D>(
            input_d1d, output_d1d, q1d);
    auto *batched_buffer_x = reinterpret_cast<mfem::real_t *>(workspace);
    auto *batched_buffer_y =
        batched_buffer_x + sizes.element_batch * sizes.buffer_x_size;
    internal::BlfPAAction<Storage, Traits, 3> action(qdata, sizes.nq, true);
    for (int batch_begin = begin; batch_begin < end;
         batch_begin += sizes.element_batch) {
      const int batch_size = std::min(sizes.element_batch, end - batch_begin);
      MFEM_FOREACH_THREAD_4D(qx, dy, dz, local_element, x, q1d, input_d1d,
                             input_d1d, batch_size) {
        const int element = batch_begin + local_element;
        auto *buffer_x = batched_buffer_x + local_element * sizes.buffer_x_size;
        const int point = qx + q1d * (dy + input_d1d * dz);
        for (int field = 0; field < Sizes::InputFields; ++field) {
          mfem::real_t value = 0.0, gradient = 0.0;
          for (int dx = 0; dx < input_d1d; ++dx) {
            const auto entry =
                input[dx + input_d1d * (dy + input_d1d * dz) +
                      sizes.input_dofs *
                          (field + Sizes::InputFields * element)];
            value += input_basis[qx + q1d * dx] * entry;
            if constexpr (Sizes::InputGradient) {
              gradient += input_derivative[qx + q1d * dx] * entry;
            }
          }
          buffer_x[point + sizes.input_stage_x *
                               (Sizes::InputXChannels * field)] = value;
          if constexpr (Sizes::InputGradient) {
            buffer_x[point + sizes.input_stage_x * (2 * field + 1)] = gradient;
          }
        }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_4D(qx, qy, dz, local_element, x, q1d, q1d, input_d1d,
                             batch_size) {
        const auto *buffer_x =
            batched_buffer_x + local_element * sizes.buffer_x_size;
        auto *buffer_y = batched_buffer_y + local_element * sizes.buffer_y_size;
        const int point = qx + q1d * (qy + q1d * dz);
        for (int field = 0; field < Sizes::InputFields; ++field) {
          mfem::real_t value = 0.0, gradient_x = 0.0, gradient_y = 0.0;
          for (int dy = 0; dy < input_d1d; ++dy) {
            const int source = qx + q1d * (dy + input_d1d * dz);
            const auto entry =
                buffer_x[source +
                         sizes.input_stage_x * (Sizes::InputXChannels * field)];
            value += input_basis[qy + q1d * dy] * entry;
            if constexpr (Sizes::InputGradient) {
              gradient_y += input_derivative[qy + q1d * dy] * entry;
              gradient_x +=
                  input_basis[qy + q1d * dy] *
                  buffer_x[source + sizes.input_stage_x * (2 * field + 1)];
            }
          }
          if constexpr (Sizes::InputGradient) {
            buffer_y[point + sizes.input_stage_y * (3 * field)] = gradient_x;
            buffer_y[point + sizes.input_stage_y * (3 * field + 1)] =
                gradient_y;
          }
          buffer_y[point +
                   sizes.input_stage_y * (Sizes::InputYChannels * field +
                                          Sizes::InputYChannels - 1)] = value;
        }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_4D(qx, qy, qz, local_element, x, q1d, q1d, q1d,
                             batch_size) {
        const int element = batch_begin + local_element;
        auto *buffer_x = batched_buffer_x + local_element * sizes.buffer_x_size;
        const auto *buffer_y =
            batched_buffer_y + local_element * sizes.buffer_y_size;
        const int point = qx + q1d * (qy + q1d * qz);
        mfem::real_t input_values[Sizes::InputFields * Sizes::InputYChannels] =
            {};
        mfem::real_t
            output_values[Sizes::OutputFields * Sizes::OutputYChannels] = {};
        for (int field = 0; field < Sizes::InputFields; ++field) {
          for (int dz = 0; dz < input_d1d; ++dz) {
            const int source = qx + q1d * (qy + q1d * dz);
            if constexpr (Sizes::InputGradient) {
              input_values[3 * field] +=
                  input_basis[qz + q1d * dz] *
                  buffer_y[source + sizes.input_stage_y * (3 * field)];
              input_values[3 * field + 1] +=
                  input_basis[qz + q1d * dz] *
                  buffer_y[source + sizes.input_stage_y * (3 * field + 1)];
              input_values[3 * field + 2] +=
                  input_derivative[qz + q1d * dz] *
                  buffer_y[source + sizes.input_stage_y * (3 * field + 2)];
            } else {
              input_values[field] +=
                  input_basis[qz + q1d * dz] *
                  buffer_y[source + sizes.input_stage_y * field];
            }
          }
        }
        if constexpr (decltype(action)::Pointwise) {
          action.template ApplyPoint<Transpose, 1>(element, point, input_values,
                                                   output_values);
        } else {
          for (int row = 0; row < Sizes::OutputFields * Sizes::OutputYChannels;
               ++row) {
            for (int column = 0;
                 column < Sizes::InputFields * Sizes::InputYChannels;
                 ++column) {
              output_values[row] +=
                  action.Entry(element, point, Transpose ? column : row,
                               Transpose ? row : column) *
                  input_values[column];
            }
          }
        }
        for (int component = 0;
             component < Sizes::OutputFields * Sizes::OutputYChannels;
             ++component) {
          buffer_x[point + sizes.nq * component] = output_values[component];
        }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_4D(qx, qy, dz, local_element, x, q1d, q1d, output_d1d,
                             batch_size) {
        const auto *buffer_x =
            batched_buffer_x + local_element * sizes.buffer_x_size;
        auto *buffer_y = batched_buffer_y + local_element * sizes.buffer_y_size;
        const int point = qx + q1d * (qy + q1d * dz);
        for (int component = 0;
             component < Sizes::OutputFields * Sizes::OutputYChannels;
             ++component) {
          mfem::real_t value = 0.0;
          const auto *table = Sizes::OutputGradient && component % 3 == 2
                                  ? output_derivative
                                  : output_basis;
          for (int qz = 0; qz < q1d; ++qz) {
            value +=
                table[qz + q1d * dz] *
                buffer_x[qx + q1d * (qy + q1d * qz) + sizes.nq * component];
          }
          buffer_y[point + sizes.output_stage_y * component] = value;
        }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_4D(qx, dy, dz, local_element, x, q1d, output_d1d,
                             output_d1d, batch_size) {
        auto *buffer_x = batched_buffer_x + local_element * sizes.buffer_x_size;
        const auto *buffer_y =
            batched_buffer_y + local_element * sizes.buffer_y_size;
        const int point = qx + q1d * (dy + output_d1d * dz);
        for (int field = 0; field < Sizes::OutputFields; ++field) {
          mfem::real_t value = 0.0, gradient = 0.0;
          for (int qy = 0; qy < q1d; ++qy) {
            const int source = qx + q1d * (qy + q1d * dz);
            if constexpr (Sizes::OutputGradient) {
              gradient += output_basis[qy + q1d * dy] *
                          buffer_y[source + sizes.output_stage_y * (3 * field)];
              value +=
                  output_derivative[qy + q1d * dy] *
                      buffer_y[source +
                               sizes.output_stage_y * (3 * field + 1)] +
                  output_basis[qy + q1d * dy] *
                      buffer_y[source + sizes.output_stage_y * (3 * field + 2)];
            } else {
              value += output_basis[qy + q1d * dy] *
                       buffer_y[source + sizes.output_stage_y * field];
            }
          }
          buffer_x[point + sizes.output_stage_x *
                               (Sizes::OutputXChannels * field)] = value;
          if constexpr (Sizes::OutputGradient) {
            buffer_x[point + sizes.output_stage_x * (2 * field + 1)] = gradient;
          }
        }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_4D(dx, dy, dz, local_element, x, output_d1d, output_d1d,
                             output_d1d, batch_size) {
        const int element = batch_begin + local_element;
        const auto *buffer_x =
            batched_buffer_x + local_element * sizes.buffer_x_size;
        for (int field = 0; field < Sizes::OutputFields; ++field) {
          mfem::real_t value = 0.0;
          for (int qx = 0; qx < q1d; ++qx) {
            const int source = qx + q1d * (dy + output_d1d * dz);
            value += output_basis[qx + q1d * dx] *
                     buffer_x[source + sizes.output_stage_x *
                                           (Sizes::OutputXChannels * field)];
            if constexpr (Sizes::OutputGradient) {
              value +=
                  output_derivative[qx + q1d * dx] *
                  buffer_x[source + sizes.output_stage_x * (2 * field + 1)];
            }
          }
          output[dx + output_d1d * (dy + output_d1d * dz) +
                 sizes.output_dofs * (field + Sizes::OutputFields * element)] +=
              multiplier * value;
        }
      }
      MFEM_SYNC_THREAD;
    }
  }
};
/// Shared differential tensor launch/kernel extents.
/// @tparam Transpose Swap trial/test roles.
template <class Traits, bool Transpose> struct TensorOperatorSizes {
  int input_component_dofs, input_dofs, output_component_dofs, output_dofs;
  int nq, input_values, output_values, intermediate_size, element_batch;

  /// @param d1d_in Input closed-basis DOFs per axis.
  /// @param o1d_in Input open-basis DOFs per axis.
  /// @param d1d_out Output closed-basis DOFs per axis.
  /// @param o1d_out Output open-basis DOFs per axis.
  /// @param q1d Quadrature points per axis.
  MFEM_HOST_DEVICE constexpr TensorOperatorSizes(
      int d1d_in, int o1d_in, int d1d_out, int o1d_out, int q1d) {
    constexpr int dims = Traits::Dims;
    constexpr auto input_map =
        Transpose ? Traits::test_type : Traits::trial_type;
    constexpr auto output_map =
        Transpose ? Traits::trial_type : Traits::test_type;
    constexpr int input_components =
        Transpose ? Traits::TestComponents : Traits::TrialComponents;
    constexpr int output_components =
        Transpose ? Traits::TrialComponents : Traits::TestComponents;
    if constexpr (input_map == mfem::FiniteElement::H_CURL) {
      input_component_dofs = o1d_in * TensorIntPow(d1d_in, dims - 1);
    } else if constexpr (input_map == mfem::FiniteElement::H_DIV) {
      input_component_dofs = d1d_in * TensorIntPow(o1d_in, dims - 1);
    } else {
      input_component_dofs = TensorIntPow(d1d_in, dims);
    }
    if constexpr (output_map == mfem::FiniteElement::H_CURL) {
      output_component_dofs = o1d_out * TensorIntPow(d1d_out, dims - 1);
    } else if constexpr (output_map == mfem::FiniteElement::H_DIV) {
      output_component_dofs = d1d_out * TensorIntPow(o1d_out, dims - 1);
    } else {
      output_component_dofs = TensorIntPow(d1d_out, dims);
    }
    input_dofs = input_component_dofs *
                 (internal::IsVectorTensorMap(input_map) ? dims :
                  (Transpose ? Traits::TestVDim : Traits::TrialVDim));
    output_dofs = output_component_dofs *
                  (internal::IsVectorTensorMap(output_map) ? dims :
                   (Transpose ? Traits::TrialVDim : Traits::TestVDim));
    nq = TensorIntPow(q1d, dims);
    input_values = nq * input_components;
    output_values = nq * output_components;
    const int max_d1d = std::max({d1d_in, o1d_in, d1d_out, o1d_out});
    if constexpr (dims == 1) {
      intermediate_size = 1;
    } else if constexpr (dims == 2) {
      intermediate_size = q1d * max_d1d;
    } else {
      intermediate_size =
          std::max(q1d * max_d1d * max_d1d, q1d * q1d * max_d1d);
    }
    constexpr int shared_reals = 32 * 1024 / sizeof(mfem::real_t);
    element_batch = std::max(
        1, std::min(64, shared_reals /
                            (input_values + output_values +
                             2 * intermediate_size)));
  }
};

/// Shared scalar tensor launch/kernel extents.
template <class Traits>
struct TensorApplySizes {
  int input_basis_size, output_basis_size, element_batch, buffer0_size;
  int basis_stride, input_dofs, nq, output_dofs, buffer1_size, max_stage_size;

  /// @param d1d_in Input DOFs per axis.
  /// @param d1d_out Output DOFs per axis.
  /// @param q1d Quadrature points per axis.
  MFEM_HOST_DEVICE constexpr TensorApplySizes(int d1d_in, int d1d_out,
                                             int q1d) {
    constexpr int dims = Traits::Dims;
    input_dofs = TensorIntPow(d1d_in, dims);
    output_dofs = TensorIntPow(d1d_out, dims);
    nq = TensorIntPow(q1d, dims);
    const int stage0_size = q1d * TensorIntPow(d1d_in, dims - 1);
    const int stage1_size =
        dims == 2 ? nq : (dims == 3 ? q1d * q1d * d1d_in : 0);
    const int stage2_size =
        dims == 2 ? d1d_out * q1d : (dims == 3 ? nq : 0);
    const int stage3_size = dims == 3 ? d1d_out * q1d * q1d : 0;
    const int stage4_size = dims == 3 ? d1d_out * d1d_out * q1d : 0;
    buffer0_size = std::max({stage0_size, stage2_size, stage4_size});
    buffer1_size = std::max(stage1_size, stage3_size);
    const int element_workspace_size = buffer0_size + buffer1_size;
    basis_stride = q1d + 1;
    input_basis_size = d1d_in * basis_stride;
    output_basis_size = BlfShareTensorBasis<Traits> ? 0 : d1d_out * basis_stride;
    constexpr int shared_reals = 32 * 1024 / sizeof(mfem::real_t);
    const int available_reals =
        std::max(0, shared_reals - input_basis_size - output_basis_size);
    const int max_shared_batch = element_workspace_size > 0
                                     ? available_reals / element_workspace_size
                                     : 1;
    const int max_1d = std::max({d1d_in, d1d_out, q1d});
    int preferred_batch;
    if constexpr (dims == 2) {
      preferred_batch = std::max(1, std::min(16, 128 / (max_1d * max_1d)));
    } else if constexpr (dims == 3) {
      preferred_batch = std::max(1, std::min(64, 128 / (max_1d * max_1d)));
    } else {
      preferred_batch = std::max(1, std::min(256, 256 / max_1d));
    }
    element_batch = std::max(1, std::min(preferred_batch, max_shared_batch));
    if constexpr (dims == 3) {
      max_stage_size = std::max(
          {q1d * d1d_in, q1d * q1d, d1d_out * q1d, d1d_out * d1d_out});
    } else {
      max_stage_size = std::max({stage0_size, stage1_size, stage2_size,
                                 stage3_size, stage4_size, output_dofs});
    }
  }
};

/// Direct full-basis PA application for the one-input/one-output-DOF case.
struct FullApplySingleFunctor {
  const mfem::real_t *input_basis;
  const mfem::real_t *output_basis;
  const mfem::real_t *qdata;
  const mfem::real_t *input;
  mfem::real_t *output;
  mfem::real_t multiplier;

  /// @param element Zero-based element index for the one-DOF contraction.
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void operator()(int element) const {
    output[element] +=
        multiplier * output_basis[0] * qdata[element] * input_basis[0] *
        input[element];
  }
};

/// Scalar-output full-basis PA kernel with optional compile-time sizes.
/// @tparam Transpose Swap trial/test roles for evaluated-basis sharing.
/// @tparam FixedInputDofs Input DOFs per element, or zero for runtime sizing.
/// @tparam FixedOutputDofs Output DOFs per element, or zero for runtime sizing.
/// @tparam FixedNQ Quadrature points per element, or zero for runtime sizing.
template <class Traits, bool Transpose, int FixedInputDofs = 0,
          int FixedOutputDofs = 0, int FixedNQ = 0>
struct FullScalarApplyFunctor {
  const mfem::real_t *input_basis;
  const mfem::real_t *output_basis;
  const mfem::real_t *qdata;
  const mfem::real_t *input;
  mfem::real_t *output;
  mfem::real_t multiplier;
  RuntimeApplyParameters<Traits, FixedInputDofs, FixedOutputDofs, FixedNQ> parameters;

  /// @param begin First element index assigned to this block.
  /// @param end Exclusive end of the block's element range.
  /// @param raw_workspace Per-block scratch buffer sized by Launch().
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *raw_workspace) const {
    const int input_dofs = parameters.Input();
    const int nq = parameters.Quadrature();
    const int output_dofs = parameters.Output();
    const auto [input_values, output_values, basis_stride,
                staged_input_basis_size, staged_output_basis_size,
                cache_basis, element_batch] =
        ResolveApplySizes<FullApplySizes<Traits, Transpose, true, FixedNQ>,
                          FixedInputDofs, FixedOutputDofs, FixedNQ>(
            input_dofs, output_dofs, nq);
    auto *staged_input_basis = reinterpret_cast<mfem::real_t *>(raw_workspace);
    auto *staged_output_basis =
        staged_input_basis + (cache_basis ? staged_input_basis_size : 0);
    auto *values = staged_output_basis +
                   (cache_basis ? staged_output_basis_size : 0);
    if (cache_basis) {
      MFEM_FOREACH_THREAD_2D(q, dof, x, nq, input_dofs) {
        staged_input_basis[q + nq * dof] = input_basis[q + nq * dof];
      }
      if constexpr (!BlfShareFullBasis<Traits>) {
        MFEM_FOREACH_THREAD_2D(q, dof, x, nq, output_dofs) {
          staged_output_basis[q + nq * dof] = output_basis[q + nq * dof];
        }
      }
      MFEM_SYNC_THREAD;
    }
    const auto *cached_output_basis =
        BlfShareFullBasis<Traits> ? staged_input_basis : staged_output_basis;
    for (int batch_begin = begin; batch_begin < end;
         batch_begin += element_batch) {
      const int batch_size = std::min(element_batch, end - batch_begin);
      MFEM_FOREACH_THREAD_2D(q, local_element, x, nq, batch_size) {
        const int element = batch_begin + local_element;
        mfem::real_t value = 0.0;
        if constexpr (FixedInputDofs) {
          MFEM_UNROLL(FixedInputDofs)
          for (int dof = 0; dof < FixedInputDofs; ++dof) {
            value += (cache_basis ? staged_input_basis[q + nq * dof]
                                  : input_basis[q + nq * dof]) *
                     input[dof + input_dofs * element];
          }
        } else {
          for (int dof = 0; dof < input_dofs; ++dof) {
            value += (cache_basis ? staged_input_basis[q + nq * dof]
                                  : input_basis[q + nq * dof]) *
                     input[dof + input_dofs * element];
          }
        }
        values[local_element * nq + q] = qdata[q + nq * element] * value;
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_2D(dof, local_element, x, output_dofs,
                             batch_size) {
        const int element = batch_begin + local_element;
        mfem::real_t value = 0.0;
        if constexpr (FixedNQ) {
          MFEM_UNROLL(FixedNQ < 8 ? FixedNQ : 8)
          for (int q = 0; q < FixedNQ; ++q) {
            value += (cache_basis ? cached_output_basis[q + nq * dof]
                                  : output_basis[q + nq * dof]) *
                     values[local_element * nq + q];
          }
        } else {
          for (int q = 0; q < nq; ++q) {
            value += (cache_basis ? cached_output_basis[q + nq * dof]
                                  : output_basis[q + nq * dof]) *
                     values[local_element * nq + q];
          }
        }
        output[dof + output_dofs * element] += multiplier * value;
      }
      MFEM_SYNC_THREAD;
    }
  }
};

/// Full-basis PA contraction with arbitrary reference output components.
/// @tparam Transpose Apply the transposed quadrature action and swap sides.
/// @tparam Storage Inlined coefficient action layout.
/// @tparam SpaceDimensions Cartesian space dimension for factored tensor actions.
/// @tparam FixedInputDofs Input DOFs per element, or zero for runtime sizing.
/// @tparam FixedOutputDofs Output DOFs per element, or zero for runtime sizing.
/// @tparam FixedNQ Quadrature points per element, or zero for runtime sizing.
template <class Traits, bool Transpose, BlfPAStorage Storage = BlfPAStorage::Dense,
          int SpaceDimensions = Traits::Dims, int FixedInputDofs = 0,
          int FixedOutputDofs = 0, int FixedNQ = 0> struct FullApplyFunctor {
  const mfem::real_t *input_basis;
  const mfem::real_t *output_basis;
  const mfem::real_t *qdata;
  const mfem::real_t *input;
  mfem::real_t *output;
  mfem::real_t multiplier;
  constexpr static int input_components =
      Transpose ? Traits::TestComponents : Traits::TrialComponents;
  constexpr static int output_components =
      Transpose ? Traits::TrialComponents : Traits::TestComponents;
  RuntimeApplyParameters<
      Traits, FixedInputDofs, FixedOutputDofs, FixedNQ,
      (Storage == BlfPAStorage::Dense && !Traits::HasComponentFields &&
       input_components * output_components > 1)> parameters;

  /// @param begin First element index assigned to this block.
  /// @param end Exclusive end of the block's element range.
  /// @param raw_workspace Per-block scratch buffer sized by Launch().
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *raw_workspace) const {
    const int input_dofs = parameters.Input();
    const int output_dofs = parameters.Output();
    const int nq = parameters.Quadrature();
    const auto [input_values, output_values, basis_stride,
                staged_input_basis_size, staged_output_basis_size,
                cache_basis, element_batch] =
        ResolveApplySizes<FullApplySizes<Traits, Transpose, false, FixedNQ>,
                          FixedInputDofs, FixedOutputDofs, FixedNQ>(
            input_dofs, output_dofs, nq);
    const bool component_major =
        Storage != BlfPAStorage::Dense ||
        parameters.ComponentMajor();
    auto *staged_input_basis = reinterpret_cast<mfem::real_t *>(raw_workspace);
    auto *staged_output_basis =
        staged_input_basis + (cache_basis ? staged_input_basis_size : 0);
    auto *input_at_q =
        staged_output_basis + (cache_basis ? staged_output_basis_size : 0);
    auto *output_at_q = input_at_q + element_batch * input_values;
    if (cache_basis) {
      MFEM_FOREACH_THREAD_2D(q, column, x, nq,
                             input_components * input_dofs) {
        staged_input_basis[q + basis_stride * column] =
            input_basis[q + column * nq];
      }
      if constexpr (!BlfShareFullBasis<Traits>) {
        MFEM_FOREACH_THREAD_2D(q, column, x, nq,
                               output_components * output_dofs) {
          staged_output_basis[q + basis_stride * column] =
              output_basis[q + nq * column];
        }
      }
      MFEM_SYNC_THREAD;
    }
    const auto *cached_output_basis =
        BlfShareFullBasis<Traits> ? staged_input_basis : staged_output_basis;
    for (int batch_begin = begin; batch_begin < end;
         batch_begin += element_batch) {
      const int batch_size = std::min(element_batch, end - batch_begin);
      MFEM_FOREACH_THREAD_3D(q, component, local_element, x, nq,
                             input_components, batch_size) {
        const int index =
            q + nq * (component + input_components * local_element);
        const int local_index = index - local_element * input_values;
        const int element = batch_begin + local_element;
        mfem::real_t value = 0.0;
        for (int dof = 0; dof < input_dofs; ++dof) {
          const int column = component + input_components * dof;
          const mfem::real_t basis_value =
              cache_basis ? staged_input_basis[q + basis_stride * column]
                          : input_basis[q + nq * column];
          value += basis_value * input[dof + input_dofs * element];
        }
        input_at_q[local_element * input_values + local_index] = value;
      }
      MFEM_SYNC_THREAD;
      const internal::BlfPAAction<Storage, Traits, SpaceDimensions> action{
          qdata, nq, component_major};
      if constexpr (decltype(action)::Pointwise) {
        MFEM_FOREACH_THREAD_2D(q, local_element, x, nq, batch_size) {
          action.template ApplyPoint<Transpose>(
              batch_begin + local_element, q,
              input_at_q + local_element * input_values + q,
              output_at_q + local_element * output_values + q);
        }
      } else {
        MFEM_FOREACH_THREAD_3D(q, output_component, local_element, x, nq,
                               output_components, batch_size) {
          const int index =
              q + nq * (output_component + output_components * local_element);
          const int local_index = index - local_element * output_values;
          const int element = batch_begin + local_element;
          mfem::real_t value = 0.0;
          for (int input_component = 0; input_component < input_components;
               ++input_component) {
            const int trial_component =
                Transpose ? output_component : input_component;
            const int test_component = Transpose ? input_component : output_component;
            value +=
                action.Entry(element, q, test_component, trial_component) *
                input_at_q[local_element * input_values + q + nq * input_component];
          }
          output_at_q[local_element * output_values + local_index] = value;
        }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_2D(dof, local_element, x, output_dofs, batch_size) {
        const int element = batch_begin + local_element;
        mfem::real_t value = 0.0;
        for (int component = 0; component < output_components; ++component) {
          for (int q = 0; q < nq; ++q) {
            const int column = component + output_components * dof;
            const mfem::real_t basis_value =
                cache_basis ? cached_output_basis[q + basis_stride * column]
                            : output_basis[q + nq * column];
            value += basis_value *
                     output_at_q[local_element * output_values + q +
                                 nq * component];
          }
        }
        output[dof + output_dofs * element] += multiplier * value;
      }
      MFEM_SYNC_THREAD;
    }
  }
};

/// Scalar/vector tensor PA block kernel with separable differential terms.
/// @tparam Transpose Transpose the quadrature action and swap trial/test roles.
/// @tparam Direct Use direct CPU iteration rather than cooperative GPU threads.
/// @tparam FixedD1DIn Input closed-basis size, or zero for runtime sizing.
/// @tparam FixedD1DOut Output closed-basis size, or zero for runtime sizing.
/// @tparam FixedQ1D Quadrature points per axis, or zero for runtime sizing.
/// @tparam Storage Inlined dense, packed, or factored coefficient action.
template <class Traits, bool Transpose, bool Direct, int FixedD1DIn = 0,
          int FixedD1DOut = 0, int FixedQ1D = 0,
          BlfPAStorage Storage = BlfPAStorage::Dense,
          int SpaceDimensions = Traits::Dims>
struct TensorOperatorApplyFunctor {
  const mfem::real_t *input_gradient;
  const mfem::real_t *input_closed;
  const mfem::real_t *input_open;
  const mfem::real_t *input;
  const mfem::real_t *qdata;
  const mfem::real_t *output_gradient;
  const mfem::real_t *output_closed;
  const mfem::real_t *output_open;
  mfem::real_t *output;
  mfem::real_t multiplier;
  RuntimeApplyParameters<Traits, FixedD1DIn, FixedD1DOut, FixedQ1D> parameters;

  /// @param begin First element index assigned to this block.
  /// @param end Exclusive end of the block's element range.
  /// @param raw_workspace Per-block scratch buffer sized by Launch().
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *raw_workspace) const {
    const int d1d_in = parameters.Input();
    const int d1d_out = parameters.Output();
    const int q1d = parameters.Quadrature();
    constexpr int dims = Traits::Dims;
    constexpr auto input_operator = ReferenceFieldOperator(
        Transpose ? Traits::test_operator : Traits::trial_operator,
        Transpose ? Traits::test_type : Traits::trial_type,
        Transpose ? Traits::TestVDim : Traits::TrialVDim,
        Transpose ? Traits::TestOutOfPlane : Traits::TrialOutOfPlane);
    constexpr auto output_operator = ReferenceFieldOperator(
        Transpose ? Traits::trial_operator : Traits::test_operator,
        Transpose ? Traits::trial_type : Traits::test_type,
        Transpose ? Traits::TrialVDim : Traits::TestVDim,
        Transpose ? Traits::TrialOutOfPlane : Traits::TestOutOfPlane);
    constexpr auto input_map_type =
        Transpose ? Traits::test_type : Traits::trial_type;
    constexpr auto output_map_type =
        Transpose ? Traits::trial_type : Traits::test_type;
    constexpr int input_components =
        Transpose ? Traits::TestComponents : Traits::TrialComponents;
    constexpr int output_components =
        Transpose ? Traits::TrialComponents : Traits::TestComponents;

    constexpr int FixedO1DIn =
        FixedD1DIn == 0 ? 0 : (internal::IsVectorTensorMap(input_map_type)
                                  ? FixedD1DIn - 1 : FixedD1DIn);
    constexpr int FixedO1DOut =
        FixedD1DOut == 0 ? 0 : (internal::IsVectorTensorMap(output_map_type)
                                   ? FixedD1DOut - 1 : FixedD1DOut);
    const int o1d_in =
        internal::IsVectorTensorMap(input_map_type) ? d1d_in - 1 : d1d_in;
    const int o1d_out =
        internal::IsVectorTensorMap(output_map_type) ? d1d_out - 1 : d1d_out;
    const auto [input_component_dofs, input_dofs, output_component_dofs,
                output_dofs, nq, input_values, output_values,
                intermediate_size, element_batch] =
        ResolveApplySizes<TensorOperatorSizes<Traits, Transpose>, FixedD1DIn,
                          FixedO1DIn, FixedD1DOut, FixedO1DOut, FixedQ1D>(
            d1d_in, o1d_in, d1d_out, o1d_out, q1d);
    const int input_closed_size = q1d * d1d_in;
    const int input_open_size = q1d * o1d_in;
    const int output_closed_size = q1d * d1d_out;
    const int output_open_size = q1d * o1d_out;
    auto *cached_input_gradient = reinterpret_cast<mfem::real_t *>(raw_workspace);
    auto *cached_input_closed = cached_input_gradient + input_closed_size;
    auto *cached_input_open = cached_input_closed + input_closed_size;
    auto *cached_output_gradient = cached_input_open + input_open_size;
    auto *cached_output_closed = cached_output_gradient + output_closed_size;
    auto *cached_output_open = cached_output_closed + output_closed_size;
    if constexpr (BlfShareTensorBasis<Traits>) {
      cached_output_gradient = cached_input_gradient;
      cached_output_closed = cached_input_closed;
      cached_output_open = cached_input_open;
    }
    MFEM_FOREACH_THREAD_1D_MODE(Direct, i, x, input_closed_size) {
      cached_input_gradient[i] = input_gradient[i];
      cached_input_closed[i] = input_closed[i];
    }
    MFEM_FOREACH_THREAD_1D_MODE(Direct, i, x, input_open_size) {
      cached_input_open[i] = input_open[i];
    }
    if constexpr (!BlfShareTensorBasis<Traits>) {
      MFEM_FOREACH_THREAD_1D_MODE(Direct, i, x, output_closed_size) {
        cached_output_gradient[i] = output_gradient[i];
        cached_output_closed[i] = output_closed[i];
      }
      MFEM_FOREACH_THREAD_1D_MODE(Direct, i, x, output_open_size) {
        cached_output_open[i] = output_open[i];
      }
    }
    MFEM_SYNC_THREAD;
    auto *input_at_q =
        cached_input_gradient + 2 * input_closed_size + input_open_size +
        (BlfShareTensorBasis<Traits> ? 0 : 2 * output_closed_size + output_open_size);
    auto *output_at_q = input_at_q + element_batch * input_values;
    auto *buffer0 = output_at_q + element_batch * output_values;
    auto *buffer1 = buffer0 + element_batch * intermediate_size;

    for (int batch_begin = begin; batch_begin < end;
         batch_begin += element_batch) {
      const int batch_size = std::min(element_batch, end - batch_begin);
      MFEM_FOREACH_THREAD_1D_MODE(Direct, i, x, batch_size * input_values) {
        input_at_q[i] = 0.0;
      }
      MFEM_SYNC_THREAD;

      for (int component = 0; component < input_components; ++component) {
        constexpr int terms =
            internal::TensorOperatorTermCount(input_operator, dims);
        for (int term_index = 0; term_index < terms; ++term_index) {
          const auto term = internal::TensorOperatorTerm(
              input_operator, dims, component, term_index);
          const bool closed_x = internal::TensorFieldUsesClosedMap(
              input_map_type, term.field_component, 0);
          const bool closed_y = internal::TensorFieldUsesClosedMap(
              input_map_type, term.field_component, 1);
          const bool closed_z =
              dims == 3 && internal::TensorFieldUsesClosedMap(
                               input_map_type, term.field_component, 2);
          const int nx = closed_x ? d1d_in : o1d_in;
          const int ny = closed_y ? d1d_in : o1d_in;
          const int nz = dims == 3 ? (closed_z ? d1d_in : o1d_in) : 1;
          const auto *basis_x = term.derivative_dimension == 0
                                    ? cached_input_gradient
                                    : (closed_x ? cached_input_closed
                                                : cached_input_open);
          const auto *basis_y = term.derivative_dimension == 1
                                    ? cached_input_gradient
                                    : (closed_y ? cached_input_closed
                                                : cached_input_open);
          const auto *basis_z = term.derivative_dimension == 2
                                    ? cached_input_gradient
                                    : (closed_z ? cached_input_closed
                                                : cached_input_open);
          if constexpr (dims == 1) {
            MFEM_FOREACH_THREAD_2D_MODE(Direct, qx, local_element, x, q1d,
                                        batch_size) {
              const int e = batch_begin + local_element;
              const int x_offset =
                  input_component_dofs * term.field_component +
                  input_dofs * e;
              mfem::real_t value = 0.0;
              for (int dx = 0; dx < nx; ++dx) {
                value += basis_x[qx + q1d * dx] * input[x_offset + dx];
              }
              input_at_q[local_element * input_values + qx +
                         nq * component] += term.sign * value;
            }
          } else {
            MFEM_FOREACH_THREAD_4D_MODE(Direct, qx, dy, dz, local_element, x,
                                        q1d, ny, nz, batch_size) {
              const int index = qx + q1d * (dy + ny * dz);
              const int e = batch_begin + local_element;
              const int x_offset =
                  input_component_dofs * term.field_component +
                  input_dofs * e;
              mfem::real_t value = 0.0;
              for (int dx = 0; dx < nx; ++dx) {
                value += basis_x[qx + q1d * dx] *
                         input[x_offset + dx + nx * (dy + ny * dz)];
              }
              buffer0[local_element * intermediate_size + index] = value;
            }
            MFEM_SYNC_THREAD;
          }

          if constexpr (dims == 2) {
            MFEM_FOREACH_THREAD_3D_MODE(Direct, qx, qy, local_element, x, q1d,
                                        q1d, batch_size) {
              const int point = qx + q1d * qy;
              mfem::real_t value = 0.0;
              for (int dy = 0; dy < ny; ++dy) {
                value += basis_y[qy + q1d * dy] *
                         buffer0[local_element * intermediate_size + qx +
                                 q1d * dy];
              }
              input_at_q[local_element * input_values + point +
                         nq * component] += term.sign * value;
            }
          } else if constexpr (dims == 3) {
            MFEM_FOREACH_THREAD_4D_MODE(Direct, qx, qy, dz, local_element, x,
                                        q1d, q1d, nz, batch_size) {
              const int index = qx + q1d * (qy + q1d * dz);
              mfem::real_t value = 0.0;
              for (int dy = 0; dy < ny; ++dy) {
                value += basis_y[qy + q1d * dy] *
                         buffer0[local_element * intermediate_size + qx +
                                 q1d * (dy + ny * dz)];
              }
              buffer1[local_element * intermediate_size + index] = value;
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD_4D_MODE(Direct, qx, qy, qz, local_element, x,
                                        q1d, q1d, q1d, batch_size) {
              const int point = qx + q1d * (qy + q1d * qz);
              mfem::real_t value = 0.0;
              for (int dz = 0; dz < nz; ++dz) {
                value += basis_z[qz + q1d * dz] *
                         buffer1[local_element * intermediate_size + qx +
                                 q1d * (qy + q1d * dz)];
              }
              input_at_q[local_element * input_values + point +
                         nq * component] += term.sign * value;
            }
          }
          MFEM_SYNC_THREAD;
        }
      }

      const internal::BlfPAAction<Storage, Traits, SpaceDimensions> action{
          qdata, nq, true};
      if constexpr (decltype(action)::Pointwise) {
        MFEM_FOREACH_THREAD_2D_MODE(Direct, point, local_element, x, nq,
                                    batch_size) {
          action.template ApplyPoint<Transpose>(batch_begin + local_element,
              point, input_at_q + local_element * input_values + point,
              output_at_q + local_element * output_values + point);
        }
      } else {
      MFEM_FOREACH_THREAD_3D_MODE(Direct, point, output_component,
                                  local_element, x, nq, output_components,
                                  batch_size) {
        const int index = point + output_component * nq;
        const int e = batch_begin + local_element;
        mfem::real_t value = 0.0;
        for (int input_component = 0; input_component < input_components;
             ++input_component) {
          const int trial_component =
              Transpose ? output_component : input_component;
          const int test_component =
              Transpose ? input_component : output_component;
          value += action.Entry(e, point, test_component, trial_component) * input_at_q[local_element * input_values +
                                              point + nq * input_component];
        }
        output_at_q[local_element * output_values + index] = value;
      }
      }
      MFEM_SYNC_THREAD;

      for (int component = 0; component < output_components; ++component) {
        constexpr int terms =
            internal::TensorOperatorTermCount(output_operator, dims);
        for (int term_index = 0; term_index < terms; ++term_index) {
          const auto term = internal::TensorOperatorTerm(
              output_operator, dims, component, term_index);
          const bool closed_x = internal::TensorFieldUsesClosedMap(
              output_map_type, term.field_component, 0);
          const bool closed_y = internal::TensorFieldUsesClosedMap(
              output_map_type, term.field_component, 1);
          const bool closed_z =
              dims == 3 && internal::TensorFieldUsesClosedMap(
                               output_map_type, term.field_component, 2);
          const int nx = closed_x ? d1d_out : o1d_out;
          const int ny = closed_y ? d1d_out : o1d_out;
          const int nz = dims == 3 ? (closed_z ? d1d_out : o1d_out) : 1;
          const auto *basis_x =
              term.derivative_dimension == 0
                  ? cached_output_gradient
                  : (closed_x ? cached_output_closed : cached_output_open);
          const auto *basis_y =
              term.derivative_dimension == 1
                  ? cached_output_gradient
                  : (closed_y ? cached_output_closed : cached_output_open);
          const auto *basis_z =
              term.derivative_dimension == 2
                  ? cached_output_gradient
                  : (closed_z ? cached_output_closed : cached_output_open);

          if constexpr (dims == 1) {
            MFEM_FOREACH_THREAD_2D_MODE(Direct, dx, local_element, x, nx,
                                        batch_size) {
              const int e = batch_begin + local_element;
              const int component_offset =
                  output_component_dofs * term.field_component +
                  output_dofs * e;
              mfem::real_t value = 0.0;
              for (int qx = 0; qx < q1d; ++qx) {
                value += basis_x[qx + q1d * dx] *
                         output_at_q[local_element * output_values + qx +
                                     nq * component];
              }
              output[component_offset + dx] += multiplier * term.sign * value;
            }
          } else if constexpr (dims == 2) {
            MFEM_FOREACH_THREAD_3D_MODE(Direct, dx, qy, local_element, x, nx,
                                        q1d, batch_size) {
              const int index = dx + nx * qy;
              mfem::real_t value = 0.0;
              for (int qx = 0; qx < q1d; ++qx) {
                value += basis_x[qx + q1d * dx] *
                         output_at_q[local_element * output_values + qx +
                                     q1d * qy + nq * component];
              }
              buffer0[local_element * intermediate_size + index] = value;
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD_3D_MODE(Direct, dx, dy, local_element, x, nx,
                                        ny, batch_size) {
              const int index = dx + nx * dy;
              const int e = batch_begin + local_element;
              const int component_offset =
                  output_component_dofs * term.field_component +
                  output_dofs * e;
              mfem::real_t value = 0.0;
              for (int qy = 0; qy < q1d; ++qy) {
                value +=
                    basis_y[qy + q1d * dy] *
                    buffer0[local_element * intermediate_size + dx + nx * qy];
              }
              output[component_offset + index] += multiplier * term.sign * value;
            }
          } else {
            MFEM_FOREACH_THREAD_4D_MODE(Direct, dx, qy, qz, local_element, x,
                                        nx, q1d, q1d, batch_size) {
              const int index = dx + nx * (qy + q1d * qz);
              mfem::real_t value = 0.0;
              for (int qx = 0; qx < q1d; ++qx) {
                value += basis_x[qx + q1d * dx] *
                         output_at_q[local_element * output_values + qx +
                                     q1d * (qy + q1d * qz) + nq * component];
              }
              buffer0[local_element * intermediate_size + index] = value;
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD_4D_MODE(Direct, dx, dy, qz, local_element, x,
                                        nx, ny, q1d, batch_size) {
              const int index = dx + nx * (dy + ny * qz);
              mfem::real_t value = 0.0;
              for (int qy = 0; qy < q1d; ++qy) {
                value += basis_y[qy + q1d * dy] *
                         buffer0[local_element * intermediate_size + dx +
                                 nx * (qy + q1d * qz)];
              }
              buffer1[local_element * intermediate_size + index] = value;
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD_4D_MODE(Direct, dx, dy, dz, local_element, x,
                                        nx, ny, nz, batch_size) {
              const int index = dx + nx * (dy + ny * dz);
              const int e = batch_begin + local_element;
              const int component_offset =
                  output_component_dofs * term.field_component +
                  output_dofs * e;
              mfem::real_t value = 0.0;
              for (int qz = 0; qz < q1d; ++qz) {
                value += basis_z[qz + q1d * dz] *
                         buffer1[local_element * intermediate_size + dx +
                                 nx * (dy + ny * qz)];
              }
              output[component_offset + index] += multiplier * term.sign * value;
            }
          }
          MFEM_SYNC_THREAD;
        }
      }
    }
  }
};

/// Scalar value/value tensor PA block kernel with staged separable contractions.
/// @tparam Direct Use direct CPU iteration rather than cooperative GPU threads.
/// @tparam FixedD1DIn Input basis DOFs per axis, or zero for runtime sizing.
/// @tparam FixedD1DOut Output basis DOFs per axis, or zero for runtime sizing.
/// @tparam FixedQ1D Quadrature points per axis, or zero for runtime sizing.
template <class Traits, bool Direct, int FixedD1DIn = 0, int FixedD1DOut = 0,
          int FixedQ1D = 0, int QDataVDim = 1>
struct TensorApplyFunctor {
  const mfem::real_t *basis;
  const mfem::real_t *basis_transpose;
  const mfem::real_t *input;
  const mfem::real_t *qdata;
  mfem::real_t *output;
  mfem::real_t multiplier;
  RuntimeApplyParameters<Traits, FixedD1DIn, FixedD1DOut, FixedQ1D> parameters;

  /// Compute a strided dot product.
  /// @tparam LeftPointer Indexable scalar buffer type on the left.
  /// @tparam RightPointer Indexable scalar buffer type on the right.
  /// @param count Number of entries to contract.
  /// @param left Left buffer start.
  /// @param left_stride Left entry stride.
  /// @param right Right buffer start.
  /// @param right_stride Right entry stride.
  template <class LeftPointer, class RightPointer>
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE static mfem::real_t
  Contract(int count, LeftPointer left, int left_stride, RightPointer right,
           int right_stride) {
    mfem::real_t value = 0.0;
    for (int i = 0; i < count; ++i) {
      value += left[i * left_stride] * right[i * right_stride];
    }
    return value;
  }

  /// @param begin First element index assigned to this block.
  /// @param end Exclusive end of the block's element range.
  /// @param raw_workspace Per-block scratch buffer sized by Launch().
  MFEM_HOST_DEVICE MFEM_NESTED_ALWAYS_INLINE void
  operator()(int begin, int end, char *raw_workspace) const {
    const int d1d_in = parameters.Input();
    const int q1d = parameters.Quadrature();
    const int d1d_out = parameters.Output();
    const auto [input_basis_size, output_basis_size, element_batch,
                buffer0_size, basis_stride, input_dofs, nq, output_dofs,
                buffer1_size, max_stage_size] =
        ResolveApplySizes<TensorApplySizes<Traits>, FixedD1DIn, FixedD1DOut, FixedQ1D>(
            d1d_in, d1d_out, q1d);
    auto *shared_basis = reinterpret_cast<mfem::real_t *>(raw_workspace);
    auto *shared_basis_transpose = shared_basis + input_basis_size;
    auto *buffer0 = shared_basis_transpose + output_basis_size;
    auto *buffer1 = buffer0 + element_batch * buffer0_size;

    MFEM_FOREACH_THREAD_2D_MODE(Direct, qx, dx, x, q1d, d1d_in) {
      shared_basis[qx + basis_stride * dx] = basis[qx + q1d * dx];
    }
    if constexpr (!BlfShareTensorBasis<Traits>) {
      MFEM_FOREACH_THREAD_2D_MODE(Direct, qx, dx, x, q1d, d1d_out) {
        shared_basis_transpose[qx + basis_stride * dx] =
            basis_transpose[dx + d1d_out * qx];
      }
    }
    MFEM_SYNC_THREAD;
    const auto *output_basis =
        BlfShareTensorBasis<Traits> ? shared_basis : shared_basis_transpose;

    for (int batch_begin = begin; batch_begin < end;
         batch_begin += element_batch) {
      const int batch_size = end - batch_begin < element_batch
                                 ? end - batch_begin
                                 : element_batch;
      if constexpr (Traits::Dims == 1) {
        MFEM_FOREACH_THREAD_2D_MODE(Direct, qx, local_element, x, q1d,
                                    batch_size) {
          const int e = batch_begin + local_element;
          const mfem::real_t value = Contract(
              d1d_in, shared_basis + qx, basis_stride, input + input_dofs * e, 1);
          buffer0[local_element * buffer0_size + qx] =
              qdata[qx + nq * (e / QDataVDim)] * value;
        }
        MFEM_SYNC_THREAD;
        MFEM_FOREACH_THREAD_2D_MODE(Direct, dx, local_element, x, d1d_out,
                                    batch_size) {
          const int e = batch_begin + local_element;
          const mfem::real_t value =
              Contract(q1d, output_basis + basis_stride * dx, 1,
                       buffer0 + local_element * buffer0_size, 1);
          output[dx + output_dofs * e] += multiplier * value;
        }
      } else if constexpr (Traits::Dims == 2) {
        MFEM_FOREACH_THREAD_3D_MODE(Direct, qx, dy, local_element, x, q1d, d1d_in,
                                    batch_size) {
          const int point = qx + dy * q1d;
          const int e = batch_begin + local_element;
          const mfem::real_t value =
              Contract(d1d_in, shared_basis + qx, basis_stride,
                       input + d1d_in * dy + input_dofs * e, 1);
          buffer0[local_element * buffer0_size + point] = value;
        }
        MFEM_SYNC_THREAD;
        MFEM_FOREACH_THREAD_3D_MODE(Direct, qx, qy, local_element, x, q1d, q1d,
                                    batch_size) {
          const int point = qx + q1d * qy;
          const int e = batch_begin + local_element;
          const mfem::real_t value =
              Contract(d1d_in, shared_basis + qy, basis_stride,
                       buffer0 + local_element * buffer0_size + qx, q1d);
          buffer1[local_element * buffer1_size + point] =
              qdata[point + nq * (e / QDataVDim)] * value;
        }
        MFEM_SYNC_THREAD;
        MFEM_FOREACH_THREAD_3D_MODE(Direct, dx, qy, local_element, x, d1d_out, q1d,
                                    batch_size) {
          const int point = dx + qy * d1d_out;
          const mfem::real_t value =
              Contract(q1d, output_basis + basis_stride * dx, 1,
                       buffer1 + local_element * buffer1_size + q1d * qy, 1);
          buffer0[local_element * buffer0_size + point] = value;
        }
        MFEM_SYNC_THREAD;
        MFEM_FOREACH_THREAD_3D_MODE(Direct, dx, dy, local_element, x, d1d_out,
                                    d1d_out, batch_size) {
          const int point = dx + d1d_out * dy;
          const int e = batch_begin + local_element;
          const mfem::real_t value =
              Contract(q1d, output_basis + basis_stride * dy, 1,
                       buffer0 + local_element * buffer0_size + dx, d1d_out);
          output[point + output_dofs * e] += multiplier * value;
        }
      } else {
        MFEM_FOREACH_THREAD_3D_MODE(Direct, qx, dy, local_element, x, q1d, d1d_in,
                                    batch_size) {
          const int e = batch_begin + local_element;
          for (int dz = 0; dz < d1d_in; ++dz) {
            const int point = qx + q1d * (dy + d1d_in * dz);
            const mfem::real_t value =
                Contract(d1d_in, shared_basis + qx, basis_stride,
                         input + d1d_in * (dy + d1d_in * dz) + input_dofs * e, 1);
            buffer0[local_element * buffer0_size + point] = value;
          }
        }
        MFEM_SYNC_THREAD;
        MFEM_FOREACH_THREAD_3D_MODE(Direct, qx, qy, local_element, x, q1d, q1d,
                                    batch_size) {
          for (int dz = 0; dz < d1d_in; ++dz) {
            const int point = qx + q1d * (qy + q1d * dz);
            const mfem::real_t value = Contract(
                d1d_in, shared_basis + qy, basis_stride,
                buffer0 + local_element * buffer0_size + qx + q1d * d1d_in * dz, q1d);
            buffer1[local_element * buffer1_size + point] = value;
          }
        }
        MFEM_SYNC_THREAD;
        MFEM_FOREACH_THREAD_3D_MODE(Direct, qx, qy, local_element, x, q1d, q1d,
                                    batch_size) {
          const int e = batch_begin + local_element;
          for (int qz = 0; qz < q1d; ++qz) {
            const int point = qx + q1d * (qy + q1d * qz);
            const mfem::real_t value = Contract(
                d1d_in, shared_basis + qz, basis_stride,
                buffer1 + local_element * buffer1_size + qx + q1d * qy, q1d * q1d);
            buffer0[local_element * buffer0_size + point] =
                qdata[point + nq * (e / QDataVDim)] * value;
          }
        }
        MFEM_SYNC_THREAD;
        MFEM_FOREACH_THREAD_3D_MODE(Direct, dx, qy, local_element, x, d1d_out, q1d,
                                    batch_size) {
          for (int qz = 0; qz < q1d; ++qz) {
            const int point = dx + d1d_out * (qy + q1d * qz);
            const mfem::real_t value = Contract(
                q1d, output_basis + basis_stride * dx, 1,
                buffer0 + local_element * buffer0_size + q1d * (qy + q1d * qz),
                1);
            buffer1[local_element * buffer1_size + point] = value;
          }
        }
        MFEM_SYNC_THREAD;
        MFEM_FOREACH_THREAD_3D_MODE(Direct, dx, dy, local_element, x, d1d_out,
                                    d1d_out, batch_size) {
          for (int qz = 0; qz < q1d; ++qz) {
            const int point = dx + d1d_out * (dy + d1d_out * qz);
            const mfem::real_t value = Contract(
                q1d, output_basis + basis_stride * dy, 1,
                buffer1 + local_element * buffer1_size + dx + d1d_out * q1d * qz,
                d1d_out);
            buffer0[local_element * buffer0_size + point] = value;
          }
        }
        MFEM_SYNC_THREAD;
        MFEM_FOREACH_THREAD_3D_MODE(Direct, dx, dy, local_element, x, d1d_out,
                                    d1d_out, batch_size) {
          const int e = batch_begin + local_element;
          for (int dz = 0; dz < d1d_out; ++dz) {
            const int point = dx + d1d_out * (dy + d1d_out * dz);
            const mfem::real_t value = Contract(
                q1d, output_basis + basis_stride * dz, 1,
                buffer0 + local_element * buffer0_size + dx + d1d_out * dy,
                d1d_out * d1d_out);
            output[point + output_dofs * e] += multiplier * value;
          }
        }
      }
    MFEM_SYNC_THREAD;
    }
  }
};

/// Direct scalar EA output. A triangle owner stores both contributions without
/// reading or mirroring the accumulated destination, which may be nonsymmetric.
struct BlfEADirectEntry {
  const int *test_dof_map;
  const int *trial_dof_map;
  mfem::real_t *matrices;
  int test_dofs;
  int trial_dofs;
  bool symmetric;
  bool add;

  MFEM_HOST_DEVICE void Initialize(int) const {}

  MFEM_HOST_DEVICE bool StoresEntry(int test, int trial) const {
    return !symmetric || test <= trial;
  }

  MFEM_HOST_DEVICE void Store(int element, int test, int trial,
                              mfem::real_t value) const {
    const int signed_test = test_dof_map ? test_dof_map[test] : test;
    const int signed_trial = trial_dof_map ? trial_dof_map[trial] : trial;
    const int output_test = signed_test >= 0 ? signed_test : -1 - signed_test;
    const int output_trial = signed_trial >= 0 ? signed_trial : -1 - signed_trial;
    value *= (signed_test >= 0 ? 1.0 : -1.0) *
             (signed_trial >= 0 ? 1.0 : -1.0);
    const int index = output_trial + trial_dofs * (output_test + test_dofs * element);
    if (add) { matrices[index] += value; }
    else { matrices[index] = value; }
  }
  MFEM_HOST_DEVICE void operator()(int element, int test, int trial,
                                   mfem::real_t value) const {
    Store(element, test, trial, value);
    if (symmetric && test != trial) { Store(element, trial, test, value); }
  }
};

/// Accumulate tensor operator terms directly into the requested EA order.
/// Mirroring applies only to each new contribution, preserving asymmetric input.
struct BlfEAAccumulatingEntry {
  const int *test_dof_map;
  const int *trial_dof_map;
  mfem::real_t *matrices;
  int test_dofs, trial_dofs, test_scalar_dofs, trial_scalar_dofs;
  bool symmetric, add;

  MFEM_HOST_DEVICE void Initialize(int element) const {
    if (!add) {
      const int size = test_dofs * trial_dofs;
      MFEM_FOREACH_THREAD(i, x, size) { matrices[i + size * element] = 0.0; }
    }
  }
  MFEM_HOST_DEVICE bool StoresEntry(int test, int trial) const {
    return !symmetric || test <= trial;
  }

  MFEM_HOST_DEVICE void Store(int element, int test, int trial,
                              mfem::real_t value) const {
    const int ts = test % test_scalar_dofs, rs = trial % trial_scalar_dofs;
    const int signed_test = test_dof_map ? test_dof_map[ts] : ts;
    const int signed_trial = trial_dof_map ? trial_dof_map[rs] : rs;
    const int output_test = test / test_scalar_dofs * test_scalar_dofs +
        (signed_test >= 0 ? signed_test : -1 - signed_test);
    const int output_trial = trial / trial_scalar_dofs * trial_scalar_dofs +
        (signed_trial >= 0 ? signed_trial : -1 - signed_trial);
    value *= (signed_test >= 0 ? 1.0 : -1.0) *
             (signed_trial >= 0 ? 1.0 : -1.0);
    matrices[output_trial + trial_dofs * (output_test + test_dofs * element)] += value;
  }
  MFEM_HOST_DEVICE void operator()(int element, int test, int trial,
                                   mfem::real_t value) const {
    Store(element, test, trial, value);
    if (symmetric && test != trial) { Store(element, trial, test, value); }
  }
};

template <class Traits, int QDataVDim = 1>
void ApplyScalarTensorOrderZero(
    int elements, const mfem::real_t *basis,
    const mfem::real_t *basis_transpose, const mfem::real_t *qdata,
    const mfem::real_t *input, mfem::real_t *output,
    mfem::real_t apply_multiplier) {
  mfem::forall(elements, [basis, basis_transpose, qdata, input, output,
                          apply_multiplier] MFEM_HOST_DEVICE(int e) {
    mfem::real_t scale = 1.0;
    for (int dimension = 0; dimension < Traits::Dims; ++dimension) {
      scale *= basis[0] * basis_transpose[0];
    }
    output[e] += apply_multiplier * scale * qdata[e / QDataVDim] * input[e];
  });
}

template <class Traits, BlfPAStorage Storage, int SpaceDimensions,
          class Prepared>
void AssembleCompactBilinearQData(int nq, int elements, Prepared prepared,
                                  mfem::real_t *qdata,
                                  const mfem::real_t *weights) {
  using Action = BlfPAAction<Storage, Traits, SpaceDimensions>;
  constexpr int components = Action::Components;
  mfem::forall(nq, elements,
               [prepared, qdata, weights, nq] MFEM_HOST_DEVICE(int point,
                                                              int element) {
    mfem::real_t payload[components];
    if constexpr (Storage == BlfPAStorage::UniformDiagonal) {
      mfem::real_t diagonal[Traits::TrialComponents];
      prepared.AssembleQData(element, point,
          BlfPackedQDataMatrix<Traits::TrialComponents, true>(diagonal));
      payload[0] = diagonal[0];
    } else if constexpr (Storage == BlfPAStorage::Symmetric ||
                  Storage == BlfPAStorage::Diagonal) {
      prepared.AssembleQData(
          element, point,
          BlfPackedQDataMatrix<Traits::TrialComponents,
                              Storage == BlfPAStorage::Diagonal>(payload));
    } else {
      prepared.AssemblePAData(element, point, payload, weights[point]);
    }
    for (int component = 0; component < components; ++component) {
      const int index = point + nq * (component + components * element);
      if constexpr (Storage == BlfPAStorage::Symmetric ||
                    Storage == BlfPAStorage::Diagonal ||
                    Storage == BlfPAStorage::UniformDiagonal) {
        qdata[index] = weights[point] * payload[component];
      } else {
        qdata[index] = payload[component];
      }
    }
  });
}

template <class Traits, class Prepared>
void AssembleDenseBilinearQData(int nq, int elements, Prepared prepared,
                                mfem::real_t *qdata,
                                const mfem::real_t *weights,
                                bool component_major) {
  constexpr int qdata_components = Traits::TrialComponents * Traits::TestComponents;
  if (component_major) {
    // Component-major storage matches tensor contractions: threads for
    // adjacent quadrature points read contiguous matrix entries.
    mfem::forall(nq, elements, [prepared, qdata, weights,
                                nq] MFEM_HOST_DEVICE(int q, int e) {
      mfem::real_t point[qdata_components];
      prepared.AssembleQData(
          e, q,
          BlfQDataMatrix<Traits::TestComponents,
                         Traits::TrialComponents>(point));
      for (int component = 0; component < qdata_components; ++component) {
        qdata[q + nq * (component + qdata_components * e)] =
            point[component] * weights[q];
      }
    });
    return;
  }
  mfem::forall(nq, elements, [prepared, qdata, weights,
                              nq] MFEM_HOST_DEVICE(int q, int e) {
    const int index = q + nq * e;
    auto *point = qdata + qdata_components * index;
    prepared.AssembleQData(
        e, q,
        BlfQDataMatrix<Traits::TestComponents,
                       Traits::TrialComponents>(point));
    for (int component = 0; component < qdata_components; ++component) {
      point[component] *= weights[q];
    }
  });
}

}



template <BlfFieldOperator Operator, mfem::FiniteElement::MapType Map,
          int Components, int VDim>
void
  BlfReferenceBasis<Operator, Map, Components, VDim>::Reset(const mfem::FiniteElement &fe,
        const mfem::IntegrationRule &integration_rule,
        mfem::ElementDofOrdering ordering) {
  const int scalar_dofs = fe.GetDof();
  dofs_ = scalar_dofs * VDim;
  quadrature_points_ = integration_rule.GetNPoints();
  values_.SetSize(dofs_ * Components * quadrature_points_);
  const auto *dof_map = ordering == mfem::ElementDofOrdering::LEXICOGRAPHIC
      ? internal::LexicographicDofMap(fe) : nullptr;
  MFEM_VERIFY(!dof_map || dof_map->Size() == 0 ||
                  dof_map->Size() == scalar_dofs,
              "invalid finite-element dof ordering map");
  mfem::DenseMatrix shape;
  auto *values = values_.HostWrite();
  for (int q = 0; q < quadrature_points_; ++q) {
    internal::CalcReferenceShape<Operator, Map, Components, VDim>(
        fe, integration_rule.IntPoint(q), shape);
    for (int ordered_dof = 0; ordered_dof < dofs_; ++ordered_dof) {
      const int scalar_dof = ordered_dof % scalar_dofs;
      const int signed_native =
          dof_map && dof_map->Size() ? (*dof_map)[scalar_dof] : scalar_dof;
      const int native_dof =
          ordered_dof / scalar_dofs * scalar_dofs +
          (signed_native >= 0 ? signed_native : -1 - signed_native);
      const mfem::real_t sign = signed_native >= 0 ? 1.0 : -1.0;
      for (int component = 0; component < Components; ++component) {
        values[Index(q, component, ordered_dof, quadrature_points_)] =
            sign * shape(native_dof, component);
      }
    }
  }
  values_.UseDevice(true);
}

template <class T>
template <class Entry>
void
  DomainBilinearFormIntegrator<T>::LaunchScalarTensorEA(T &integrator,
                                const mfem::FiniteElementSpace &trial_space,
                                const mfem::FiniteElementSpace &test_space,
                                const mfem::IntegrationRule &integration_rule,
                                Entry add_entry) {
  const auto &trial_fe = *trial_space.GetFE(0);
  const auto &test_fe = *test_space.GetFE(0);
  const auto &trial_map =
      trial_fe.GetDofToQuad(integration_rule, mfem::DofToQuad::TENSOR);
  const auto &test_map =
      test_fe.GetDofToQuad(integration_rule, mfem::DofToQuad::TENSOR);
  const int trial_d1d = trial_map.ndof;
  const int test_d1d = test_map.ndof;
  const int q1d = trial_map.nqpt;
  const int quadrature_points = integration_rule.GetNPoints();
  const int trial_dofs = trial_fe.GetDof() * blf_traits::TrialVDim;
  const int test_dofs = test_fe.GetDof() * blf_traits::TestVDim;
  const int matrix_size = trial_dofs * test_dofs;
  const int element_count = trial_space.GetNE();

  const bool use_device =
      mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                           mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP);
  const auto prepared_qdata = internal::PrepareEAQData(integrator,
      trial_space, test_space, integration_rule, use_device);
  const auto *trial_values = trial_map.B.Read(use_device);
  const auto *trial_derivatives = trial_map.G.Read(use_device);
  const auto *test_values = test_map.B.Read(use_device);
  const auto *test_derivatives = test_map.G.Read(use_device);
  const auto *quadrature_weights =
      integration_rule.GetWeights().Read(use_device);
  const auto sizes = TensorEASizes(trial_d1d, test_d1d, q1d);
  const int staged_trial_size = sizes.staged_trial_size;
  const int staged_test_size = sizes.staged_test_size;
  const int qdata_size = sizes.qdata_size;
  const int stage0_size = sizes.stage0_size;
  const int stage1_size = sizes.stage1_size;
  const int workspace_reals = staged_trial_size + staged_test_size +
                              qdata_size + stage0_size + stage1_size;
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(workspace_reals) * sizeof(mfem::real_t);
  internal::TensorEAWorkFunctor<blf_traits, decltype(prepared_qdata), Entry> kernel{
      test_d1d, trial_d1d, q1d, trial_values, trial_derivatives, test_values,
      test_derivatives, quadrature_weights, prepared_qdata, add_entry};
  const int threads = std::min(256,
      std::max({matrix_size, quadrature_points, stage0_size, stage1_size}));
  internal::LaunchEA(kernel, element_count, threads, 1, workspace_bytes, use_device);

}

template <class T>
template <class Entry>
void DomainBilinearFormIntegrator<T>::LaunchOperatorTensorEA(
      T &integrator, const mfem::FiniteElementSpace &trial_space,
      const mfem::FiniteElementSpace &test_space,
      const mfem::IntegrationRule &integration_rule,
      Entry add_entry) {
  const auto &trial_fe = *trial_space.GetFE(0);
  const auto &test_fe = *test_space.GetFE(0);
  const auto &trial_closed =
      trial_fe.GetDofToQuad(integration_rule, mfem::DofToQuad::TENSOR);
  const auto &test_closed =
      test_fe.GetDofToQuad(integration_rule, mfem::DofToQuad::TENSOR);
  const mfem::DofToQuad *trial_open_ptr = &trial_closed;
  const mfem::DofToQuad *test_open_ptr = &test_closed;
  if constexpr (internal::IsVectorTensorMap(blf_traits::trial_type)) {
    const auto *trial_vector =
        dynamic_cast<const mfem::VectorTensorFiniteElement *>(&trial_fe);
    MFEM_VERIFY(trial_vector, "missing vector tensor trial element");
    trial_open_ptr = &trial_vector->GetDofToQuadOpen(integration_rule,
                                                     mfem::DofToQuad::TENSOR);
  }
  if constexpr (internal::IsVectorTensorMap(blf_traits::test_type)) {
    const auto *test_vector =
        dynamic_cast<const mfem::VectorTensorFiniteElement *>(&test_fe);
    MFEM_VERIFY(test_vector, "missing vector tensor test element");
    test_open_ptr = &test_vector->GetDofToQuadOpen(integration_rule,
                                                   mfem::DofToQuad::TENSOR);
  }
  const auto &trial_open = *trial_open_ptr;
  const auto &test_open = *test_open_ptr;
  const int trial_d1d = trial_closed.ndof;
  const int test_d1d = test_closed.ndof;
  const int q1d = trial_closed.nqpt;
  const int nq = integration_rule.GetNPoints();
  const int trial_dofs = trial_fe.GetDof() * blf_traits::TrialVDim;
  const int test_dofs = test_fe.GetDof() * blf_traits::TestVDim;
  const int matrix_size = trial_dofs * test_dofs;
  const int elements = trial_space.GetNE();

  const bool use_device =
      mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                           mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP);
  const auto prepared = internal::PrepareEAQData(integrator, trial_space, test_space,
                                                integration_rule, use_device);
  const auto *weights = integration_rule.GetWeights().Read(use_device);
  const auto *trial_closed_values = trial_closed.B.Read(use_device);
  const auto *trial_closed_derivatives = trial_closed.G.Read(use_device);
  const auto *trial_open_values = trial_open.B.Read(use_device);
  const auto *test_closed_values = test_closed.B.Read(use_device);
  const auto *test_closed_derivatives = test_closed.G.Read(use_device);
  const auto *test_open_values = test_open.B.Read(use_device);
  const auto sizes = OperatorTensorEASizes(trial_d1d, test_d1d, q1d);
  const int qdata_size = sizes.qdata_size;
  const int stage0_size = sizes.stage0_size;
  const int stage1_size = sizes.stage1_size;
  const int workspace_reals = qdata_size + stage0_size + stage1_size;
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(workspace_reals) * sizeof(mfem::real_t);
  OperatorTensorEAWorkFunctor<decltype(prepared), decltype(add_entry)> kernel{
      test_d1d, trial_d1d, q1d,
      weights, test_closed_derivatives, test_closed_values, test_open_values,
      trial_closed_derivatives, trial_closed_values, trial_open_values,
      prepared, add_entry};

  constexpr int launch_bound = 256;
  const int threads = std::min(
      launch_bound, std::max({matrix_size, nq, stage0_size, stage1_size}));
  internal::LaunchEA(kernel, elements, threads, 1, workspace_bytes, use_device);
}

template <class T>
template <int TrialD1D, int TestD1D, int Q1D>
void DomainBilinearFormIntegrator<T>::AssembleFusedTensorEA(
    T &integrator, const mfem::FiniteElementSpace &trial_space,
    const mfem::FiniteElementSpace &test_space,
    const mfem::IntegrationRule &rule, mfem::Vector &emat,
    bool add, bool same_space_interface) {
  static_assert(blf_traits::TrialComponents == 1 &&
                blf_traits::TestComponents == 1 &&
                blf_traits::trial_operator == BlfFieldOperator::Value &&
                blf_traits::test_operator == BlfFieldOperator::Value,
                "direct tensor EA requires scalar value/value fields");
  const auto &trial_fe = *trial_space.GetFE(0);
  const auto &test_fe = *test_space.GetFE(0);
  const auto &trial_map = trial_fe.GetDofToQuad(rule, mfem::DofToQuad::TENSOR);
  const auto &test_map = test_fe.GetDofToQuad(rule, mfem::DofToQuad::TENSOR);
  const int trial_d1d = TrialD1D ? TrialD1D : trial_map.ndof;
  const int test_d1d = TestD1D ? TestD1D : test_map.ndof;
  const int q1d = Q1D ? Q1D : trial_map.nqpt;
  MFEM_VERIFY(trial_d1d == trial_map.ndof && test_d1d == test_map.ndof &&
                  q1d == trial_map.nqpt && q1d == test_map.nqpt,
              "registered EA kernel does not match the basis sizes");
  const int trial_dofs = trial_fe.GetDof();
  const int test_dofs = test_fe.GetDof();
  const int elements = trial_space.GetNE();
  const int matrix_size = trial_dofs * test_dofs;
  const int size = elements * matrix_size;
  const bool use_device =
      mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                           mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP);
  auto *matrices = internal::PrepareEAOutput(emat, size, add, use_device);
  // EA has no dependency on the PA cache: coefficients and geometry are
  // refreshed here, then weighted qdata is formed inside the contraction block.
  const auto prepared = internal::PrepareEAQData(integrator, trial_space,
                                                 test_space, rule, use_device);
  const internal::BlfEADirectEntry entry{
      internal::ReadEAOutputDofMap(test_space, same_space_interface, use_device, true),
      internal::ReadEAOutputDofMap(trial_space, same_space_interface, use_device, true), matrices,
      test_dofs, trial_dofs,
      internal::HasSymmetricEA<T> && &trial_space == &test_space, add};
  const auto sizes = TensorEASizes(trial_d1d, test_d1d, q1d);
  const int workspace_reals = sizes.staged_trial_size + sizes.staged_test_size +
      sizes.qdata_size + sizes.stage0_size + sizes.stage1_size;
  const int batch = matrix_size <= 256 && elements % 4 == 0 &&
      elements >= forall_nested_max_concurrent_blocks() &&
      4 * workspace_reals * sizeof(mfem::real_t) <= forall_nested_shared_memory_limit
      ? 4 : 1;
  internal::TensorEAWorkFunctor<blf_traits, decltype(prepared), decltype(entry),
                              TrialD1D, Q1D, TestD1D> kernel{
      test_d1d, trial_d1d, q1d, trial_map.B.Read(use_device), nullptr,
      test_map.B.Read(use_device), nullptr, rule.GetWeights().Read(use_device),
      prepared, entry, batch};
  internal::BatchedTensorEAWorkFunctor<decltype(kernel)> batched{
      kernel, batch, workspace_reals};
  const int threads = std::min(256 / batch,
      std::max({matrix_size, sizes.quadrature_points, sizes.stage0_size, sizes.stage1_size}));
  internal::LaunchEA(batched, elements / batch, threads, batch,
                     static_cast<std::size_t>(batch) * workspace_reals * sizeof(mfem::real_t),
                     use_device);
}

template <class T>
template <int TrialD1D, int TestD1D, int Q1D>
typename DomainBilinearFormIntegrator<T>::AssembleEATensorKernelType
DomainBilinearFormIntegrator<T>::AssembleEATensorKernels::Kernel() {
  return &DomainBilinearFormIntegrator<T>::template AssembleFusedTensorEA<
      TrialD1D, TestD1D, Q1D>;
}

template <class T>
typename DomainBilinearFormIntegrator<T>::AssembleEATensorKernelType
DomainBilinearFormIntegrator<T>::AssembleEATensorKernels::Fallback(int, int, int) {
  return &DomainBilinearFormIntegrator<T>::template AssembleFusedTensorEA<0, 0, 0>;
}

template <class T>
template <int TrialDofs, int TestDofs, int NQ>
void DomainBilinearFormIntegrator<T>::AssembleFusedFullEA(
    T &integrator, const mfem::FiniteElementSpace &trial_space,
    const mfem::FiniteElementSpace &test_space,
    const mfem::IntegrationRule &rule, mfem::Vector &emat,
    bool add, bool same_space_interface) {
  const auto &trial_fe = *trial_space.GetFE(0);
  const auto &test_fe = *test_space.GetFE(0);
  const int trial_dofs = TrialDofs ? TrialDofs : trial_fe.GetDof() * blf_traits::TrialVDim;
  const int test_dofs = TestDofs ? TestDofs : test_fe.GetDof() * blf_traits::TestVDim;
  const int nq = NQ ? NQ : rule.GetNPoints();
  MFEM_VERIFY(trial_dofs == trial_fe.GetDof() * blf_traits::TrialVDim &&
                  test_dofs == test_fe.GetDof() * blf_traits::TestVDim && nq == rule.GetNPoints(),
              "registered full EA kernel does not match the basis sizes");
  const int elements = trial_space.GetNE();
  const int size = elements * trial_dofs * test_dofs;
  const bool use_device =
      mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                           mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP);
  auto *matrices = internal::PrepareEAOutput(emat, size, add, use_device);
  TrialBasis trial_basis(trial_fe, rule);
  TestBasis test_basis(test_fe, rule);
  const auto prepared = internal::PrepareEAQData(integrator, trial_space, test_space, rule, use_device);
  const internal::DirectFullEAOutput output{
      internal::ReadEAOutputDofMap(test_space, same_space_interface, use_device, false),
      internal::ReadEAOutputDofMap(trial_space, same_space_interface, use_device, false),
      test_fe.GetDof(), trial_fe.GetDof(), add};
  internal::LaunchFullEA<blf_traits, decltype(prepared), decltype(output), TrialDofs, TestDofs, NQ>(
      elements, trial_dofs, test_dofs, nq, trial_basis.Read(use_device), test_basis.Read(use_device),
      rule.GetWeights().Read(use_device), prepared,
      matrices,
      internal::HasSymmetricEA<T> && &trial_space == &test_space, output, use_device);
}

template <class T>
template <int TrialDofs, int TestDofs, int NQ>
typename DomainBilinearFormIntegrator<T>::AssembleEAFullKernelType
DomainBilinearFormIntegrator<T>::AssembleEAFullKernels::Kernel() {
  return &DomainBilinearFormIntegrator<T>::template AssembleFusedFullEA<TrialDofs, TestDofs, NQ>;
}

template <class T>
typename DomainBilinearFormIntegrator<T>::AssembleEAFullKernelType
DomainBilinearFormIntegrator<T>::AssembleEAFullKernels::Fallback(int, int, int) {
  return &DomainBilinearFormIntegrator<T>::template AssembleFusedFullEA<0, 0, 0>;
}

template <class T>
void DomainBilinearFormIntegrator<T>::AssembleEAImpl(
    const mfem::FiniteElementSpace &trial_space,
    const mfem::FiniteElementSpace &test_space, mfem::Vector &emat,
    bool add, bool same_space_interface) {
  internal::VerifyEAElementSpaces<blf_traits>(trial_space, test_space);
  const int elements = trial_space.GetNE();
  if (elements == 0) {
    if (!add) { emat.SetSize(0); }
    MFEM_VERIFY(emat.Size() == 0, "existing element-assembly vector has the wrong size");
    return;
  }
  const auto &trial_fe = *trial_space.GetFE(0);
  const auto &test_fe = *test_space.GetFE(0);
  const bool supports = SupportsTensorProduct(trial_fe, test_fe, integration_rule_);
  MFEM_VERIFY(tensor_product_mode_ != BlfTensorProductMode::Force || supports,
              "forced tensor-product EA is unsupported for these elements, "
              "operators, or quadrature rule");
  const bool use_tensor = supports && tensor_product_mode_ != BlfTensorProductMode::Disable &&
      (tensor_product_mode_ == BlfTensorProductMode::Force ||
       WithinTensorOptimizationLimits(trial_fe, test_fe, integration_rule_, tensor_product_limits_));
  if (!use_tensor) {
    AssembleEAFullKernels::Run(trial_fe.GetDof() * blf_traits::TrialVDim,
        test_fe.GetDof() * blf_traits::TestVDim, integration_rule_.GetNPoints(),
        integrator_, trial_space, test_space, integration_rule_, emat, add, same_space_interface);
    return;
  }
  if constexpr (UsesScalarMassTensorEA) {
    const auto &trial_map = trial_fe.GetDofToQuad(integration_rule_, mfem::DofToQuad::TENSOR);
    const auto &test_map = test_fe.GetDofToQuad(integration_rule_, mfem::DofToQuad::TENSOR);
    AssembleEATensorKernels::Run(trial_map.ndof, test_map.ndof, trial_map.nqpt,
        integrator_, trial_space, test_space, integration_rule_, emat, add, same_space_interface);
  } else {
    const int trial_dofs = trial_fe.GetDof() * blf_traits::TrialVDim;
    const int test_dofs = test_fe.GetDof() * blf_traits::TestVDim;
    const bool use_device =
        mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                             mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP);
    const internal::BlfEAAccumulatingEntry entry{
        internal::ReadEAOutputDofMap(test_space, same_space_interface, use_device, true),
        internal::ReadEAOutputDofMap(trial_space, same_space_interface, use_device, true),
        internal::PrepareEAOutput(emat, elements * trial_dofs * test_dofs, add, use_device),
        test_dofs, trial_dofs, test_fe.GetDof(), trial_fe.GetDof(),
        !UsesOperatorTensorEA && internal::HasSymmetricEA<T> && &trial_space == &test_space,
        add};
    LaunchTensorEA(integrator_, trial_space, test_space, integration_rule_, entry);
  }
}

template <class T>
void DomainBilinearFormIntegrator<T>::AssembleElementMatrix2Impl(
    const mfem::FiniteElement &trial_fe, const mfem::FiniteElement &test_fe,
    mfem::ElementTransformation &transformation, mfem::DenseMatrix &element_matrix) {
  static_assert(internal::BlfHostQDataAssembler<T>,
                "single-element assembly requires "
                "AssembleQDataHost(ElementTransformation &, const "
                "IntegrationPoint &, DenseMatrix &)");
  VerifyElement(trial_fe, blf_traits::trial_type, "trial");
  VerifyElement(test_fe, blf_traits::test_type, "test");
  MFEM_VERIFY(transformation.GetDimension() == blf_traits::Dims,
              "element transformation has the wrong reference dimension");
  MFEM_VERIFY(transformation.GetSpaceDim() >= blf_traits::Dims &&
                  transformation.GetSpaceDim() <= 3,
              "element transformation has an invalid physical dimension");

  constexpr int trial_components = blf_traits::TrialComponents;
  constexpr int test_components = blf_traits::TestComponents;
  const int trial_dofs = trial_fe.GetDof() * blf_traits::TrialVDim;
  const int test_dofs = test_fe.GetDof() * blf_traits::TestVDim;
  mfem::DenseMatrix trial_shape(trial_dofs, trial_components);
  mfem::DenseMatrix test_shape(test_dofs, test_components);
  mfem::DenseMatrix qdata(test_components, trial_components);
  mfem::DenseMatrix test_qdata(test_dofs, trial_components);

  element_matrix.SetSize(test_dofs, trial_dofs);
  element_matrix = 0.0;
  for (int q = 0; q < integration_rule_.GetNPoints(); ++q) {
    const auto &ip = integration_rule_.IntPoint(q);
    transformation.SetIntPoint(&ip);
    internal::CalcReferenceShape<blf_traits::TrialBasisOperator,
                                 blf_traits::trial_type, trial_components,
                                 blf_traits::TrialVDim>(trial_fe, ip,
                                                        trial_shape);
    internal::CalcReferenceShape<blf_traits::TestBasisOperator,
                                 blf_traits::test_type, test_components,
                                 blf_traits::TestVDim>(test_fe, ip,
                                                       test_shape);

    qdata.SetSize(test_components, trial_components);
    qdata = 0.0;
    integrator_.AssembleQDataHost(transformation, ip, qdata);
    MFEM_VERIFY(qdata.Height() == test_components &&
                    qdata.Width() == trial_components,
                "AssembleQData changed the quadrature-tensor dimensions");
    qdata *= ip.weight;

    mfem::Mult(test_shape, qdata, test_qdata);
    mfem::AddMultABt(test_qdata, trial_shape, element_matrix);
  }
}

template <class T>
template <bool ADAt, class TrialBasisType, class TestBasisType, class Projection>
void DomainBilinearFormIntegrator<T>::DiagonalContract(
    const mfem::Vector &diagonal_weights, mfem::Vector &diag,
    TrialBasisType trial, TestBasisType test, Projection projection) const {
  const bool gpu =
      mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                           mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP);
  const int nq = trial.NQ();
  WithPALayout([&]<BlfPAStorage Storage, int SpaceDimensions>() {
    using Action = internal::BlfPAAction<Storage, blf_traits, SpaceDimensions>;
    const Action action(qdata_.Read(), nq, UsesComponentMajorQData());
    const bool cache_qdata = gpu && nq * Action::Components * sizeof(mfem::real_t) <= 16 * 1024;
    const int basis_reals = trial.StageReals() + test.StageReals();
    const int qdata_reals = cache_qdata ? nq * Action::Components : 0;
    if constexpr (ADAt) {
      internal::ColumnDiagonalFunctor<blf_traits, TrialBasisType, TestBasisType, Projection, Action> kernel{
          trial, test, projection, action, diagonal_weights.Read(), diag.ReadWrite(), cache_qdata};
      const int workspace_reals = basis_reals + qdata_reals +
          nq * blf_traits::TestComponents + 2 * test.Dofs() + projection.ScratchReals();
      Launch(kernel, std::min(256, std::max(nq, test.Dofs())),
             static_cast<std::size_t>(workspace_reals) * sizeof(mfem::real_t), elements_);
    } else {
      if constexpr (requires { trial.ND(); test.ND(); }) {
        if constexpr (TrialBasisType::Map == TestBasisType::Map &&
                      TrialBasisType::Fields == TestBasisType::Fields && !Action::Pointwise) {
          // Measured low-order GPU cases favor direct quadrature; CPU
          // contractions and larger GPU bases benefit from sum factorization.
          const bool use_factors = gpu ? trial.ND() >= 3 : trial.ND() >= 2;
          if (trial.ND() == test.ND() && use_factors) {
            internal::TensorFactorDiagonalFunctor<blf_traits, TrialBasisType, TestBasisType, Action> kernel{
                trial, test, action, diag.ReadWrite(), cache_qdata};
            const int reals = basis_reals + qdata_reals + kernel.ScratchReals();
            Launch(kernel, 256, static_cast<std::size_t>(reals) * sizeof(mfem::real_t), elements_);
            return;
          }
        }
      }
      internal::DirectDiagonalFunctor<blf_traits, TrialBasisType, TestBasisType, Action> kernel{
          trial, test, action, diag.ReadWrite(), gpu ? 8 : 1, cache_qdata};
      Launch(kernel, 256, static_cast<std::size_t>(basis_reals + qdata_reals + kernel.ScratchReals()) *
             sizeof(mfem::real_t), elements_);
    }
  });
}

template <class T>
template <bool ADAt, int FixedTrialDofs, int FixedTestDofs, int FixedNQ>
void DomainBilinearFormIntegrator<T>::DiagonalFull(
    const mfem::Vector &diagonal_weights, mfem::Vector &diag) const {
  using Trial = internal::FullDiagonalBasis<blf_traits::TrialComponents, FixedTrialDofs, FixedNQ>;
  using Test = internal::FullDiagonalBasis<blf_traits::TestComponents, FixedTestDofs, FixedNQ>;
  const Trial trial{trial_basis_->Read(), trial_dofs_, quadrature_points_};
  const Test test{test_basis_->Read(), test_dofs_, quadrature_points_};
  const bool gpu =
      mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                           mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP);
  if (!gpu || (ADAt && trial.Dofs() <= 4 && test.Dofs() <= 4)) {
    // CPU rows and tiny GPU ADAt matrices favor the direct, scratch-free path.
    WithPALayout([&]<BlfPAStorage Storage, int SpaceDimensions>() {
      using Action = internal::BlfPAAction<Storage, blf_traits, SpaceDimensions>;
      const Action action(qdata_.Read(), trial.NQ(), UsesComponentMajorQData());
      mfem::forall(elements_ * test.Dofs(),
          internal::FullDiagonalRowFunctor<blf_traits, Action, ADAt,
              FixedTrialDofs, FixedTestDofs, FixedNQ>{
              trial.Dofs(), test.Dofs(), trial.NQ(), trial_basis_->Read(), test_basis_->Read(),
              ADAt ? diagonal_weights.Read() : nullptr, diag.ReadWrite(), 1.0, action});
    });
    return;
  }
  if constexpr (ADAt) {
    WithPALayout([&]<BlfPAStorage Storage, int SpaceDimensions>() {
      using Action = internal::BlfPAAction<Storage, blf_traits, SpaceDimensions>;
      const Action action(qdata_.Read(), trial.NQ(), UsesComponentMajorQData());
      const int lanes = trial.NQ() >= 16 ? 4 : 1;
      const int rows = std::max(1, std::min(16, 256 / (trial.Dofs() * lanes)));
      const int groups = elements_ * ((test.Dofs() + rows - 1) / rows);
      const bool cache = trial.NQ() * Action::Components * sizeof(mfem::real_t) <= 16 * 1024;
      internal::FullPairDiagonalFunctor<blf_traits, Trial, Test, Action> kernel{
          trial, test, action, diagonal_weights.Read(), diag.ReadWrite(), lanes, rows, cache};
      const int reals = (cache ? trial.NQ() * Action::Components : 0) +
                       (lanes + 1) * rows * trial.Dofs();
      Launch(kernel, 256, static_cast<std::size_t>(reals) * sizeof(mfem::real_t), groups, groups);
    });
  } else {
    DiagonalContract<false>(diagonal_weights, diag, trial, test, 0);
  }
}

template <class T>
template <bool ADAt, int FixedD1D, int FixedTestD1D, int FixedQ1D>
void DomainBilinearFormIntegrator<T>::DiagonalTensor(
    const mfem::Vector &diagonal_weights, mfem::Vector &diag) const {
  constexpr int dims = blf_traits::Dims;
  const int nd = FixedD1D ? FixedD1D : trial_maps_->ndof;
  const int nq = FixedQ1D ? FixedQ1D : trial_maps_->nqpt;
  if constexpr (!ADAt &&
                blf_traits::trial_operator == BlfFieldOperator::Value &&
                blf_traits::test_operator == BlfFieldOperator::Value &&
                blf_traits::TrialVDim == blf_traits::TestVDim &&
                (SelectedPAStorage<dims> == BlfPAStorage::Diagonal ||
                 SelectedPAStorage<dims> == BlfPAStorage::UniformDiagonal ||
                 (blf_traits::TrialComponents == 1 && blf_traits::TestComponents == 1)) &&
                !internal::IsVectorTensorMap(blf_traits::trial_type) &&
                !internal::IsVectorTensorMap(blf_traits::test_type)) {
    const int elements = elements_ * blf_traits::TrialVDim;
    const internal::TensorDiagonalSizes<dims> sizes(nd, nq);
    internal::ScalarDiagonalTensorFunctor<dims, FixedD1D, FixedQ1D,
        SelectedPAStorage<dims> == BlfPAStorage::UniformDiagonal ? blf_traits::TrialVDim : 1>
        assemble{nd, nq, trial_maps_->B.Read(), test_maps_->B.Read(), qdata_.Read(), diag.ReadWrite()};
    const int work = std::max({sizes.basis_size, sizes.element_batch * sizes.first_size,
                              sizes.element_batch * sizes.second_size, sizes.element_batch * sizes.dofs});
    Launch(assemble, std::min(256, work),
           (sizes.basis_size + sizes.element_batch * (sizes.first_size + sizes.second_size)) *
               sizeof(mfem::real_t),
           (elements + sizes.element_batch - 1) / sizes.element_batch, elements);
  } else {
    using Trial = internal::TensorDiagonalBasis<blf_traits, true, FixedD1D, FixedQ1D>;
    using Test = internal::TensorDiagonalBasis<blf_traits, false, FixedTestD1D, FixedQ1D>;
    const Trial trial{trial_maps_->B.Read(), trial_maps_->G.Read(), trial_open_maps_->B.Read(),
                      trial_maps_->ndof, trial_maps_->nqpt};
    const Test test{test_maps_->B.Read(), test_maps_->G.Read(), test_open_maps_->B.Read(),
                    test_maps_->ndof, test_maps_->nqpt};
    DiagonalContract<ADAt>(diagonal_weights, diag, trial, test,
        internal::TensorDiagonalProjection<blf_traits, Test>{test_maps_->ndof, nq});
  }
}

template <class T>
template <class F>
decltype(auto) DomainBilinearFormIntegrator<T>::WithPALayout(F &&action) const {
  constexpr auto preferred = internal::PreferredPAStorage<T>;
  if constexpr (preferred == BlfPAStorage::Kelvin ||
                preferred == BlfPAStorage::SymmetricKelvin ||
                preferred == BlfPAStorage::IsotropicSymmetric) {
    if constexpr (blf_traits::Dims <= 1 && blf_traits::TrialVDim <= 1 &&
                  blf_traits::TestVDim <= 1 &&
                  !blf_traits::TrialOutOfPlane && !blf_traits::TestOutOfPlane) {
      if (space_dimensions_ == 1) {
        return action.template operator()<SelectedPAStorage<1>, 1>();
      }
    }
    if constexpr (blf_traits::Dims <= 2 && blf_traits::TrialVDim <= 2 &&
                  blf_traits::TestVDim <= 2 &&
                  !blf_traits::TrialOutOfPlane && !blf_traits::TestOutOfPlane) {
      if (space_dimensions_ == 2) {
        return action.template operator()<SelectedPAStorage<2>, 2>();
      }
    }
    return action.template operator()<SelectedPAStorage<3>, 3>();
  } else {
    return action.template operator()<SelectedPAStorage<blf_traits::Dims>,
                                       blf_traits::Dims>();
  }
}

template <class T>
template <class F>
void DomainBilinearFormIntegrator<T>::Launch(F &kernel, int threads, std::size_t workspace_bytes,
              int block_limit, int elements) const {
  if (elements == 0) { elements = elements_; }
  MFEM_VERIFY(workspace_bytes <=
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "per-block partial-assembly workspace exceeds INT_MAX");
  constexpr int launch_bound = 256;
  const int bytes = static_cast<int>(workspace_bytes);
  const bool use_gpu =
      mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                           mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP);
  const int max_concurrent_blocks = std::min(
      {forall_nested_max_concurrent_blocks(), block_limit, elements});

  if (!use_gpu) {
    const auto required = forall_nested_global_workspace_size(
        elements, max_concurrent_blocks, bytes);
    const auto workspace_reals =
        (required + sizeof(mfem::real_t) - 1) / sizeof(mfem::real_t);
    MFEM_VERIFY(workspace_reals <=
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "partial-assembly workspace exceeds INT_MAX");
    workspace_.SetSize(static_cast<int>(workspace_reals));
    auto *workspace = reinterpret_cast<char *>(workspace_.HostReadWrite());
    forall_nested_launch_bounds<launch_bound>(
        kernel, kernel, elements, threads, 1, 1, max_concurrent_blocks,
        bytes, true, workspace, forall_nested_workspace_stride(bytes));
    return;
  }

  const auto required = forall_nested_global_workspace_size(
      elements, max_concurrent_blocks, bytes);
  const auto workspace_reals =
      (required + sizeof(mfem::real_t) - 1) / sizeof(mfem::real_t);
  MFEM_VERIFY(workspace_reals <=
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "partial-assembly workspace exceeds INT_MAX");
  const int global_reals = bytes > forall_nested_shared_memory_limit
                               ? static_cast<int>(workspace_reals)
                               : 0;
  workspace_.SetSize(global_reals);
  workspace_.UseDevice(true);
  auto *workspace = global_reals
                        ? reinterpret_cast<char *>(workspace_.Write(true))
                        : nullptr;
  forall_nested_launch_bounds_workspace<launch_bound>(
      kernel, kernel, elements, threads, 1, 1, max_concurrent_blocks, bytes,
      workspace, global_reals * sizeof(mfem::real_t));
}

template <class T>
template <bool Transpose, int FixedInputDofs, int FixedOutputDofs,
            int FixedNQ, BlfPAStorage Storage,
            int SpaceDimensions>
void DomainBilinearFormIntegrator<T>::ApplyFullImpl(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const {
  constexpr int input_components =
      Transpose ? blf_traits::TestComponents : blf_traits::TrialComponents;
  constexpr int output_components =
      Transpose ? blf_traits::TrialComponents : blf_traits::TestComponents;

  const int input_dofs =
      FixedInputDofs ? FixedInputDofs : (Transpose ? test_dofs_ : trial_dofs_);
  const int output_dofs = FixedOutputDofs
                              ? FixedOutputDofs
                              : (Transpose ? trial_dofs_ : test_dofs_);
  const int nq = FixedNQ ? FixedNQ : quadrature_points_;
  const auto *input_basis =
      Transpose ? test_basis_->Read() : trial_basis_->Read();
  const auto *output_basis =
      Transpose ? trial_basis_->Read() : test_basis_->Read();
  const auto *qdata = qdata_.Read();
  const auto *input = x.Read();
  auto *output = y.ReadWrite();

  if constexpr (input_components == 1 && output_components == 1) {
    if (input_dofs == 1 && output_dofs == 1 && nq == 1) {
      mfem::forall(elements_, FullApplySingleFunctor{
                                input_basis, output_basis, qdata, input, output,
                                apply_scale});
      return;
    }

    const auto [input_values, output_values, basis_stride,
                staged_input_basis_size, staged_output_basis_size,
                cache_basis, element_batch] =
        FullApplySizes<Transpose, true, FixedNQ>(input_dofs, output_dofs, nq);
    const int staged_basis_size =
        staged_input_basis_size + staged_output_basis_size;
    FullScalarApplyFunctor<Transpose, FixedInputDofs, FixedOutputDofs, FixedNQ>
        scalar_apply{input_basis,
                     output_basis,
                     qdata,
                     input,
                     output,
                     apply_scale,
                     {input_dofs, output_dofs, nq}};
    const int threads =
        std::min(256, element_batch * std::max(nq, output_dofs));
    Launch(scalar_apply, threads,
           static_cast<std::size_t>(
               (cache_basis ? staged_basis_size : 0) + element_batch * nq) *
               sizeof(mfem::real_t),
           (elements_ + element_batch - 1) / element_batch);
    return;
  }

  const auto [input_values, output_values, basis_stride,
              staged_input_basis_size, staged_output_basis_size,
              cache_basis, element_batch] =
      FullApplySizes<Transpose, false, FixedNQ>(input_dofs, output_dofs, nq);
  const int basis_workspace_size =
      cache_basis ? staged_input_basis_size + staged_output_basis_size : 0;
  const int element_workspace_size = input_values + output_values;
  const bool component_major_qdata = UsesComponentMajorQData();

  FullApplyFunctor<Transpose, Storage, SpaceDimensions, FixedInputDofs,
                   FixedOutputDofs, FixedNQ> apply{
      input_basis,
      output_basis,
      qdata,
      input,
      output,
      apply_scale,
      {input_dofs, output_dofs, nq, component_major_qdata}};

  const int threads =
      std::min(256, element_batch *
                        std::max({input_values, output_values, output_dofs}));
  Launch(apply, threads,
         static_cast<std::size_t>(basis_workspace_size +
                                  element_batch * element_workspace_size) *
             sizeof(mfem::real_t),
         (elements_ + element_batch - 1) / element_batch);
}

template <class T>
template <bool Transpose, int FixedD1DIn, int FixedD1DOut,
            int FixedQ1D, BlfPAStorage Storage,
            int SpaceDimensions>
void DomainBilinearFormIntegrator<T>::ApplyVectorTensorImpl(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const {
  static_assert(blf_traits::Dims >= 1 && blf_traits::Dims <= 3);

  const auto &input_closed_map = Transpose ? *test_maps_ : *trial_maps_;
  const auto &input_open_map =
      Transpose ? *test_open_maps_ : *trial_open_maps_;
  const auto &output_closed_map = Transpose ? *trial_maps_ : *test_maps_;
  const auto &output_open_map =
      Transpose ? *trial_open_maps_ : *test_open_maps_;
  const int d1d_in = input_closed_map.ndof;
  const int o1d_in = input_open_map.ndof;
  const int d1d_out = output_closed_map.ndof;
  const int o1d_out = output_open_map.ndof;
  const int q1d = input_closed_map.nqpt;
  const auto [input_component_dofs, input_dofs, output_component_dofs,
          output_dofs, nq, input_values, output_values,
          intermediate_size, element_batch] =
      TensorOperatorSizes<Transpose>(d1d_in, o1d_in, d1d_out, o1d_out, q1d);

  const auto *input_closed = input_closed_map.B.Read();
  const auto *input_gradient = input_closed_map.G.Read();
  const auto *input_open = input_open_map.B.Read();
  const auto *output_closed = output_closed_map.B.Read();
  const auto *output_gradient = output_closed_map.G.Read();
  const auto *output_open = output_open_map.B.Read();
  const auto *qdata = qdata_.Read();
  const auto *input = x.Read();
  auto *output = y.ReadWrite();

  const int basis_workspace_size =
      q1d * (2 * d1d_in + o1d_in) +
      (ShareTensorBasis ? 0 : q1d * (2 * d1d_out + o1d_out));
  const int element_workspace_size =
      input_values + output_values + 2 * intermediate_size;
  const int workspace_reals =
      basis_workspace_size + element_batch * element_workspace_size;

  TensorOperatorApplyFunctor<Transpose, false, FixedD1DIn, FixedD1DOut,
                             FixedQ1D, Storage, SpaceDimensions>
      apply{input_gradient,
            input_closed,
            input_open,
            input,
            qdata,
            output_gradient,
            output_closed,
            output_open,
            output,
            apply_scale,
            {d1d_in, d1d_out, q1d}};

  const int per_element_parallel_work =
      std::max({input_values, output_values, intermediate_size,
                output_component_dofs});
  const int parallel_work = element_batch * per_element_parallel_work;
  const int threads = std::min(256, parallel_work);
  const auto workspace_bytes =
      static_cast<std::size_t>(workspace_reals) * sizeof(mfem::real_t);
  if (parallel_work <= threads) {
    TensorOperatorApplyFunctor<Transpose, true, FixedD1DIn, FixedD1DOut,
                               FixedQ1D, Storage, SpaceDimensions>
        direct_apply{input_gradient,
                     input_closed,
                     input_open,
                     input,
                     qdata,
                     output_gradient,
                     output_closed,
                     output_open,
                     output,
                     apply_scale,
                     {d1d_in, d1d_out, q1d}};
    Launch(direct_apply, threads, workspace_bytes, elements_);
  } else {
    Launch(apply, threads, workspace_bytes, elements_);
  }
}

template <class T>
template <bool Transpose, int FixedD1DIn, int FixedD1DOut,
            int FixedQ1D>
void DomainBilinearFormIntegrator<T>::ApplyScalarTensorGeneric(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const {
  const int elements = elements_ * blf_traits::TrialVDim;
  const auto &input_map = Transpose ? *test_maps_ : *trial_maps_;
  const auto &output_map = Transpose ? *trial_maps_ : *test_maps_;
  const int d1d_in = input_map.ndof;
  const int d1d_out = output_map.ndof;
  const int q1d = input_map.nqpt;
  const auto [input_basis_size, output_basis_size, element_batch,
          buffer0_size, basis_stride, input_dofs, nq, output_dofs,
          buffer1_size, max_stage_size] =
      TensorApplySizes(d1d_in, d1d_out, q1d);
  const auto *basis = input_map.B.Read();
  const auto *basis_transpose = output_map.Bt.Read();
  const auto *qdata = qdata_.Read();
  const auto *input = x.Read();
  auto *output = y.ReadWrite();
  const mfem::real_t apply_multiplier = apply_scale;

  // The order-zero tensor operator is a scalar multiply per element.  A
  // staged block kernel would add several barriers without exposing any
  // contraction parallelism, so fuse the complete operation here.
  if (d1d_in == 1 && d1d_out == 1 && q1d == 1) {
    internal::ApplyScalarTensorOrderZero<blf_traits,
        SelectedPAStorage<blf_traits::Dims> == BlfPAStorage::UniformDiagonal
            ? blf_traits::TrialVDim : 1>(
        elements, basis, basis_transpose, qdata, input, output,
        apply_multiplier);
    return;
  }

  const int workspace_reals = input_basis_size + output_basis_size +
                              element_batch * (buffer0_size + buffer1_size);

  TensorApplyFunctor<false, FixedD1DIn, FixedD1DOut, FixedQ1D> apply{
      basis, basis_transpose, input, qdata, output, apply_scale,
      {d1d_in, d1d_out, q1d}};

  const int threads = std::min(256, element_batch * max_stage_size);
  const auto workspace_bytes =
      static_cast<std::size_t>(workspace_reals) * sizeof(mfem::real_t);
  const int block_limit = (elements + element_batch - 1) / element_batch;
  if (element_batch * max_stage_size <= threads) {
    TensorApplyFunctor<true, FixedD1DIn, FixedD1DOut, FixedQ1D> direct_apply{
        basis, basis_transpose, input, qdata, output, apply_scale,
        {d1d_in, d1d_out, q1d}};
    Launch(direct_apply, threads, workspace_bytes, block_limit, elements);
  } else {
    Launch(apply, threads, workspace_bytes, block_limit, elements);
  }
}

template <class T>
template <bool Transpose, int FixedD1DIn, int FixedD1DOut,
            int FixedQ1D>
void DomainBilinearFormIntegrator<T>::ApplyTensorGeneric(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const {
  if constexpr (blf_traits::HasComponentFields &&
                blf_traits::trial_operator == BlfFieldOperator::Value &&
                blf_traits::test_operator == BlfFieldOperator::Value &&
                internal::IsScalarMap(blf_traits::trial_type) &&
                internal::IsScalarMap(blf_traits::test_type) &&
                (SelectedPAStorage<blf_traits::Dims> == BlfPAStorage::Diagonal ||
                 SelectedPAStorage<blf_traits::Dims> == BlfPAStorage::UniformDiagonal) &&
                (blf_traits::Dims < 3 ||
                 (FixedD1DIn == 3 && FixedD1DOut == 3 && FixedQ1D == 4))) {
    return ApplyScalarTensorGeneric<Transpose, FixedD1DIn, FixedD1DOut,
                                    FixedQ1D>(x, y, apply_scale);
  } else if constexpr (blf_traits::Dims == 3 &&
                internal::IsScalarMap(blf_traits::trial_type) &&
                internal::IsScalarMap(blf_traits::test_type) &&
                (blf_traits::HasComponentFields ||
                 blf_traits::trial_operator != BlfFieldOperator::Value ||
                 blf_traits::test_operator != BlfFieldOperator::Value)) {
    const auto &input_map = Transpose ? *test_maps_ : *trial_maps_;
    const auto &output_map = Transpose ? *trial_maps_ : *test_maps_;
    const int input_d1d = input_map.ndof;
    const int output_d1d = output_map.ndof;
    const int q1d = input_map.nqpt;
    const internal::ScalarTensorFieldSizes<blf_traits, Transpose> sizes(
        input_d1d, output_d1d, q1d);
    WithPALayout([&]<BlfPAStorage Storage, int SpaceDimensions>() {
      internal::ScalarTensorFieldApplyFunctor<blf_traits, Transpose,
          FixedD1DIn, FixedD1DOut, FixedQ1D, Storage> apply{
          input_map.B.Read(), input_map.G.Read(), output_map.B.Read(),
          output_map.G.Read(), qdata_.Read(), x.Read(), y.ReadWrite(),
          apply_scale, {input_d1d, output_d1d, q1d}};
      Launch(apply, std::min(256, sizes.element_batch *
                                    std::max({sizes.nq, sizes.input_dofs,
                                              sizes.output_dofs})),
             sizes.element_batch * (sizes.buffer_x_size + sizes.buffer_y_size) *
                 sizeof(mfem::real_t),
             (elements_ + sizes.element_batch - 1) / sizes.element_batch);
    });
  } else if constexpr (blf_traits::HasComponentFields ||
                blf_traits::trial_operator != BlfFieldOperator::Value ||
                blf_traits::test_operator != BlfFieldOperator::Value ||
                ((internal::IsVectorTensorMap(blf_traits::trial_type) ||
                  internal::IsVectorTensorMap(blf_traits::test_type)) &&
                 (blf_traits::Dims == 2 || blf_traits::Dims == 3))) {
    return ApplyVectorTensorGeneric<Transpose, FixedD1DIn, FixedD1DOut,
                                    FixedQ1D>(x, y, apply_scale);
  } else {
    return ApplyScalarTensorGeneric<Transpose, FixedD1DIn, FixedD1DOut,
                                    FixedQ1D>(x, y, apply_scale);
  }
}

template <class T>
template <BlfPAStorage Storage, int SpaceDimensions>
void DomainBilinearFormIntegrator<T>::AssembleCompactPA(const mfem::FiniteElementSpace &trial_space,
                          const mfem::FiniteElementSpace &test_space,
                          bool use_device) {
  const int nq = quadrature_points_;
  constexpr int components =
      internal::BlfPAAction<Storage, blf_traits, SpaceDimensions>::Components;
  qdata_.SetSize(elements_ * nq * components,
      pa_mt == mfem::MemoryType::DEFAULT ? mfem::Device::GetDeviceMemoryType() : pa_mt);
  qdata_.UseDevice(true);
  auto *qdata = qdata_.Write(use_device);
  const auto *weights = integration_rule_.GetWeights().Read(use_device);
  if constexpr (Storage == BlfPAStorage::Symmetric ||
                Storage == BlfPAStorage::Diagonal ||
                Storage == BlfPAStorage::UniformDiagonal) {
    const auto prepared = integrator_.PrepareQData(
        trial_space, test_space, integration_rule_, use_device);
    internal::AssembleCompactBilinearQData<blf_traits, Storage, SpaceDimensions>(
        nq, elements_, prepared, qdata, weights);
  } else {
    const auto prepared = integrator_.PreparePAData(
        trial_space, test_space, integration_rule_, use_device);
    internal::AssembleCompactBilinearQData<blf_traits, Storage, SpaceDimensions>(
        nq, elements_, prepared, qdata, weights);
  }
}

template <class T>
template <bool Transpose>
void DomainBilinearFormIntegrator<T>::Apply(const mfem::Vector &x, mfem::Vector &y, mfem::real_t apply_scale) const {
  MFEM_VERIFY(kernel_ != Kernel::Unassembled,
              "AssemblePA must be called before applying the operator");
  const int input_dofs = Transpose ? test_dofs_ : trial_dofs_;
  const int output_dofs = Transpose ? trial_dofs_ : test_dofs_;
  MFEM_VERIFY(x.Size() == elements_ * input_dofs &&
                  y.Size() == elements_ * output_dofs,
              "partial-assembly element vector has the wrong size");
  if (elements_ == 0) { return; }
  if (kernel_ == Kernel::Tensor && !tensor_uses_full_basis_) {
    ApplyTensor<Transpose>(x, y, apply_scale);
  } else if constexpr (Transpose) {
    ApplyFullTransposeKernels::Run(input_dofs, output_dofs,
                                   quadrature_points_, *this, x, y, apply_scale);
  } else {
    ApplyFullKernels::Run(input_dofs, output_dofs, quadrature_points_, *this,
                          x, y, apply_scale);
  }
}

template <class T>
void DomainBilinearFormIntegrator<T>::Assemble(const mfem::FiniteElementSpace &trial_space,
                const mfem::FiniteElementSpace &test_space,
                bool same_space_interface) {
  MFEM_VERIFY(trial_space.GetVDim() == blf_traits::TrialVDim &&
                  test_space.GetVDim() == blf_traits::TestVDim,
              "finite-element space vdim does not match the operation");
  MFEM_VERIFY(trial_space.GetMesh() == test_space.GetMesh(),
              "trial and test spaces must use the same mesh");
  MFEM_VERIFY(trial_space.GetNE() == test_space.GetNE(),
              "trial and test spaces have different element counts");
  const auto *trial_fe = trial_space.GetTypicalFE();
  const auto *test_fe = test_space.GetTypicalFE();
  MFEM_VERIFY(trial_fe && test_fe,
              "finite-element space returned a null typical element");
  VerifyElement(*trial_fe, blf_traits::trial_type, "trial");
  VerifyElement(*test_fe, blf_traits::test_type, "test");
  elements_ = trial_space.GetNE();
  same_space_ = same_space_interface;
  trial_dofs_ = trial_fe->GetDof() * blf_traits::TrialVDim;
  test_dofs_ = test_fe->GetDof() * blf_traits::TestVDim;
  quadrature_points_ = integration_rule_.GetNPoints();
  space_dimensions_ = trial_space.GetMesh()->SpaceDimension();

  const bool supports_tensor =
      SupportsTensorProduct(*trial_fe, *test_fe, integration_rule_);
  MFEM_VERIFY(tensor_product_mode_ != BlfTensorProductMode::Force ||
                  supports_tensor,
              "forced tensor-product partial assembly is unsupported for "
              "these elements, operators, or quadrature rule");
  if (elements_ == 0) {
    kernel_ = Kernel::Full;
    tensor_uses_full_basis_ = false;
    trial_maps_ = test_maps_ = nullptr;
    trial_open_maps_ = test_open_maps_ = nullptr;
    trial_basis_.reset();
    test_basis_.reset();
    qdata_.SetSize(0);
    workspace_.SetSize(0);
    return;
  }
  if (supports_tensor) {
    trial_maps_ =
        &trial_fe->GetDofToQuad(integration_rule_, mfem::DofToQuad::TENSOR);
    test_maps_ =
        &test_fe->GetDofToQuad(integration_rule_, mfem::DofToQuad::TENSOR);
    if constexpr (internal::IsVectorTensorMap(blf_traits::trial_type)) {
      const auto *trial_vector =
          dynamic_cast<const mfem::VectorTensorFiniteElement *>(trial_fe);
      MFEM_VERIFY(trial_vector, "missing vector tensor trial element");
      trial_open_maps_ = &trial_vector->GetDofToQuadOpen(
          integration_rule_, mfem::DofToQuad::TENSOR);
    } else {
      trial_open_maps_ = trial_maps_;
    }
    if constexpr (internal::IsVectorTensorMap(blf_traits::test_type)) {
      const auto *test_vector =
          dynamic_cast<const mfem::VectorTensorFiniteElement *>(test_fe);
      MFEM_VERIFY(test_vector, "missing vector tensor test element");
      test_open_maps_ = &test_vector->GetDofToQuadOpen(
          integration_rule_, mfem::DofToQuad::TENSOR);
    } else {
      test_open_maps_ = test_maps_;
    }
  } else {
    trial_maps_ = test_maps_ = nullptr;
    trial_open_maps_ = test_open_maps_ = nullptr;
  }
  const bool use_tensor =
      supports_tensor &&
      (tensor_product_mode_ == BlfTensorProductMode::Force ||
       (tensor_product_mode_ == BlfTensorProductMode::Auto &&
        WithinTensorOptimizationLimits(*trial_maps_, *test_maps_)));
  kernel_ = use_tensor ? Kernel::Tensor : Kernel::Full;

  constexpr int max_full_tensor_basis_reals = 1024;
  // At order one, direct contraction with a cached full basis is cheaper
  // than synchronizing between the separable terms of curl-curl and
  // div-div. The closed/open map sizes identify the order-one tensor basis
  // for both the ND and RT families (and avoid relying on collection names).
  const bool order_one_vector_basis = trial_maps_ && trial_open_maps_ &&
                                      trial_maps_->ndof == 2 &&
                                      trial_open_maps_->ndof == 1;
  tensor_uses_full_basis_ =
      mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                           mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP) &&
      kernel_ == Kernel::Tensor && (is_hcurl_curl_curl || is_hdiv_div_div) &&
      same_space_ && order_one_vector_basis &&
      quadrature_points_ *
              std::max(trial_dofs_ * blf_traits::TrialComponents,
                       test_dofs_ * blf_traits::TestComponents) <=
          max_full_tensor_basis_reals;

  if (kernel_ == Kernel::Full || tensor_uses_full_basis_) {
    const auto trial_ordering = same_space_interface
                                    ? mfem::GetEVectorOrdering(trial_space)
                                    : mfem::ElementDofOrdering::LEXICOGRAPHIC;
    const auto test_ordering = same_space_interface
                                   ? mfem::GetEVectorOrdering(test_space)
                                   : mfem::ElementDofOrdering::LEXICOGRAPHIC;
    trial_basis_ = std::make_unique<TrialBasis>(*trial_fe, integration_rule_,
                                                trial_ordering);
    test_basis_ = std::make_unique<TestBasis>(*test_fe, integration_rule_,
                                              test_ordering);
  } else {
    trial_basis_.reset();
    test_basis_.reset();
  }

  const bool use_device =
      mfem::Device::Allows(mfem::Backend::CUDA | mfem::Backend::HIP |
                           mfem::Backend::RAJA_CUDA | mfem::Backend::RAJA_HIP);
  WithPALayout([this, &trial_space, &test_space,
                use_device]<BlfPAStorage Storage, int SpaceDimensions>() {
    if constexpr (Storage != BlfPAStorage::Dense) {
      AssembleCompactPA<Storage, SpaceDimensions>(trial_space, test_space, use_device);
    } else {
      const auto prepared = integrator_.PrepareQData(
          trial_space, test_space, integration_rule_, use_device);
      const auto *weights = integration_rule_.GetWeights().Read(use_device);
      constexpr int qdata_components =
          blf_traits::TrialComponents * blf_traits::TestComponents;
      qdata_.SetSize(elements_ * quadrature_points_ * qdata_components,
          pa_mt == mfem::MemoryType::DEFAULT ? mfem::Device::GetDeviceMemoryType() : pa_mt);
      qdata_.UseDevice(true);
      auto *qdata = qdata_.Write(use_device);
      const int nq = quadrature_points_;
      internal::AssembleDenseBilinearQData<blf_traits>(
          nq, elements_, prepared, qdata, weights, UsesComponentMajorQData());
    }
  });
}

template <class T>
void DomainBilinearFormIntegrator<T>::AssembleDiagonalPA(mfem::Vector &diag) {
  MFEM_VERIFY(kernel_ != Kernel::Unassembled,
              "AssemblePA must be called before assembling the diagonal");
  MFEM_VERIFY(trial_dofs_ == test_dofs_ &&
                  diag.Size() == elements_ * test_dofs_,
              "diagonal assembly requires square element matrices and a "
              "matching test E-vector");
  if (!elements_) { return; }
  const mfem::Vector unused;
  if (kernel_ == Kernel::Full || tensor_uses_full_basis_) {
    DiagonalFullKernels::Run(trial_dofs_, test_dofs_, quadrature_points_,
                            *this, unused, diag);
  } else {
    DiagonalTensorKernels::Run(trial_maps_->ndof, test_maps_->ndof,
                              trial_maps_->nqpt, *this, unused, diag);
  }
}

#define BLF_DIAGONAL_REGISTRY(Registry, Method, ...)                           \
  template <class T>                                                          \
  template <int TrialDofs, int TestDofs, int NQ>                               \
  typename DomainBilinearFormIntegrator<T>::DiagonalKernelType                 \
  DomainBilinearFormIntegrator<T>::Registry::Kernel() {                        \
    return &DomainBilinearFormIntegrator<T>::template Method<                  \
        __VA_ARGS__ __VA_OPT__(,) TrialDofs, TestDofs, NQ>;                    \
  }                                                                          \
  template <class T>                                                          \
  typename DomainBilinearFormIntegrator<T>::DiagonalKernelType                 \
  DomainBilinearFormIntegrator<T>::Registry::Fallback(int, int, int) {         \
    return &DomainBilinearFormIntegrator<T>::template Method<                  \
        __VA_ARGS__ __VA_OPT__(,) 0, 0, 0>;                                   \
  }

BLF_DIAGONAL_REGISTRY(DiagonalTensorKernels, DiagonalTensor, false)
BLF_DIAGONAL_REGISTRY(DiagonalADAtTensorKernels, DiagonalTensor, true)
BLF_DIAGONAL_REGISTRY(DiagonalFullKernels, DiagonalFull, false)
BLF_DIAGONAL_REGISTRY(DiagonalADAtFullKernels, DiagonalFull, true)

#undef BLF_DIAGONAL_REGISTRY

/// @tparam T Operation type of the enclosing PA adapter.
/// @tparam D1DIn Trial closed-basis DOFs per axis.
/// @tparam D1DOut Test closed-basis DOFs per axis.
/// @tparam Q1D Quadrature points per axis.
/// @return Forward tensor member-function kernel for these fixed sizes.
template <class T>
template <int D1DIn, int D1DOut, int Q1D>
typename DomainBilinearFormIntegrator<T>::ApplyTensorKernelType
DomainBilinearFormIntegrator<T>::ApplyTensorKernels::Kernel() {
  return &DomainBilinearFormIntegrator<T>::template ApplyTensorGeneric<
      false, D1DIn, D1DOut, Q1D>;
}

/// @tparam T Operation type of the enclosing PA adapter.
/// The three runtime dispatch keys (input/output closed-basis DOFs per axis,
/// quadrature points per axis) are unused; the returned kernel reads setup data.
/// @return Runtime-sized forward tensor member-function kernel.
template <class T>
DomainBilinearFormIntegrator<T>::ApplyTensorKernelType
DomainBilinearFormIntegrator<T>::ApplyTensorKernels::Fallback(int, int, int) {
  return &DomainBilinearFormIntegrator<T>::template ApplyTensorGeneric<
      false, 0, 0, 0>;
}

/// @tparam T Operation type of the enclosing PA adapter.
/// @tparam D1DIn Test closed-basis DOFs per axis (transpose input).
/// @tparam D1DOut Trial closed-basis DOFs per axis (transpose output).
/// @tparam Q1D Quadrature points per axis.
/// @return Transpose tensor member-function kernel for these fixed sizes.
template <class T>
template <int D1DIn, int D1DOut, int Q1D>
typename DomainBilinearFormIntegrator<T>::ApplyTensorKernelType
DomainBilinearFormIntegrator<T>::ApplyTensorTransposeKernels::Kernel() {
  return &DomainBilinearFormIntegrator<T>::template ApplyTensorGeneric<
      true, D1DIn, D1DOut, Q1D>;
}

/// @tparam T Operation type of the enclosing PA adapter.
/// The three runtime keys are input/output closed-basis DOFs per axis and
/// quadrature points per axis; the runtime kernel reads the assembled sizes.
/// @return Runtime-sized transpose tensor member-function kernel.
template <class T>
DomainBilinearFormIntegrator<T>::ApplyTensorKernelType
DomainBilinearFormIntegrator<T>::ApplyTensorTransposeKernels::Fallback(
    int, int, int) {
  return &DomainBilinearFormIntegrator<T>::template ApplyTensorGeneric<
      true, 0, 0, 0>;
}

/// @tparam T Operation type of the enclosing PA adapter.
/// @tparam InputDofs Trial element DOFs including vdim copies.
/// @tparam OutputDofs Test element DOFs including vdim copies.
/// @tparam NQ Total quadrature points per element.
/// @return Forward full-basis member-function kernel for these fixed sizes.
template <class T>
template <int InputDofs, int OutputDofs, int NQ>
typename DomainBilinearFormIntegrator<T>::ApplyFullKernelType
DomainBilinearFormIntegrator<T>::ApplyFullKernels::Kernel() {
  return &DomainBilinearFormIntegrator<T>::template ApplyFull<
      false, InputDofs, OutputDofs, NQ>;
}

/// @tparam T Operation type of the enclosing PA adapter.
/// The runtime keys (input DOFs, output DOFs, quadrature count per element) are
/// unused; the returned kernel reads the assembled sizes.
/// @return Runtime-sized forward full-basis member-function kernel.
template <class T>
DomainBilinearFormIntegrator<T>::ApplyFullKernelType
DomainBilinearFormIntegrator<T>::ApplyFullKernels::Fallback(int, int, int) {
  return &DomainBilinearFormIntegrator<T>::template ApplyFull<false>;
}

/// @tparam T Operation type of the enclosing PA adapter.
/// @tparam InputDofs Test element DOFs including vdim copies (transpose input).
/// @tparam OutputDofs Trial element DOFs including vdim copies (transpose output).
/// @tparam NQ Total quadrature points per element.
/// @return Transpose full-basis member-function kernel for these fixed sizes.
template <class T>
template <int InputDofs, int OutputDofs, int NQ>
typename DomainBilinearFormIntegrator<T>::ApplyFullKernelType
DomainBilinearFormIntegrator<T>::ApplyFullTransposeKernels::Kernel() {
  return &DomainBilinearFormIntegrator<T>::template ApplyFull<
      true, InputDofs, OutputDofs, NQ>;
}

/// @tparam T Operation type of the enclosing PA adapter.
/// The runtime keys are input DOFs, output DOFs and quadrature count per element;
/// the returned kernel reads the assembled sizes.
/// @return Runtime-sized transpose full-basis member-function kernel.
template <class T>
DomainBilinearFormIntegrator<T>::ApplyFullKernelType
DomainBilinearFormIntegrator<T>::ApplyFullTransposeKernels::Fallback(
    int, int, int) {
  return &DomainBilinearFormIntegrator<T>::template ApplyFull<true>;
}
} // namespace mfem

#endif
