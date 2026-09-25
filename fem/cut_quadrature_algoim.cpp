// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#include "cut_quadrature.hpp"

#ifdef MFEM_USE_ALGOIM

#include "intrules.hpp"

#include <algoim/quadrature_general.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <vector>

namespace mfem
{

namespace
{

class AlgoimCutQuadratureWorkspace : public CutQuadratureWorkspace { };

bool HasMeasure(CutMeasure set, CutMeasure measure)
{
   return static_cast<unsigned>(set & measure) != 0u;
}

CutQuadratureStatus ValidateRequest(const CutQuadratureRequest &request)
{
   const unsigned measures = static_cast<unsigned>(request.measures);
   const unsigned allowed = static_cast<unsigned>(CutMeasure::Volume) |
                            static_cast<unsigned>(CutMeasure::Interface);
   if (request.order < 0 || measures == 0u || (measures & ~allowed) != 0u ||
       (request.region != CutRegion::Negative &&
        request.region != CutRegion::Positive) ||
       (request.weight_policy != QuadratureWeightPolicy::Unconstrained &&
        request.weight_policy != QuadratureWeightPolicy::Nonnegative))
   {
      return CutQuadratureStatus::InvalidRequest;
   }
   return CutQuadratureStatus::Success;
}

void Reset(ReferenceCutQuadrature &result, CutQuadratureStatus status)
{
   result.status = status;
   result.classification = CutCellClass::Unclassified;
   result.volume.SetSize(0);
   result.interface.rule.SetSize(0);
   result.interface.reference_normals.SetSize(0, 0);
}

int Dimension(Geometry::Type geometry)
{
   if (geometry == Geometry::SQUARE) { return 2; }
   if (geometry == Geometry::CUBE) { return 3; }
   return 0;
}

int CoefficientCount(Geometry::Type geometry, int order)
{
   if (order < 0) { return -1; }
   const int dimension = Dimension(geometry);
   if (dimension == 0) { return -1; }
   const std::uint64_t n = static_cast<std::uint64_t>(order) + 1u;
   std::uint64_t count = 1u;
   for (int d = 0; d < dimension; d++)
   {
      if (count > static_cast<std::uint64_t>(
                     std::numeric_limits<int>::max()) / n)
      {
         return -1;
      }
      count *= n;
   }
   return static_cast<int>(count);
}

template <int N>
class AlgoimBernsteinLevelSet
{
public:
   AlgoimBernsteinLevelSet(const Vector &coefficients, int order,
                           real_t selection_sign = 1.0)
      : order_(order), sign_(selection_sign)
   {
      const int n = order_ + 1;
      power_coefficients_ = coefficients;
      Vector next(coefficients.Size());
      int stride = 1;
      for (int direction = 0; direction < N; direction++)
      {
         for (int index = 0; index < power_coefficients_.Size(); index++)
         {
            const int degree = (index / stride) % n;
            const int base = index - degree*stride;
            real_t value = 0.0;
            for (int i = 0; i <= degree; i++)
            {
               const real_t sign = ((degree - i) & 1) ? -1.0 : 1.0;
               value += power_coefficients_(base + i*stride) *
                        Binomial(order_, i) * Binomial(order_ - i, degree - i) *
                        sign;
            }
            next(index) = value;
         }
         power_coefficients_ = next;
         stride *= n;
      }
   }

   template <typename T>
   T operator()(const algoim::uvector<T, N> &x) const
   {
      std::vector<T> powers[N];
      for (int d = 0; d < N; d++)
      {
         powers[d].resize(order_ + 1);
         powers[d][0] = T(1.0);
         for (int i = 1; i <= order_; i++)
         {
            powers[d][i] = powers[d][i - 1] * x(d);
         }
      }
      T value = T(0.0);
      int c = 0;
      if (N == 2)
      {
         for (int j = 0; j <= order_; j++)
         {
            for (int i = 0; i <= order_; i++, c++)
            {
               value += T(power_coefficients_(c)) * powers[0][i] *
                        powers[1][j];
            }
         }
      }
      else
      {
         for (int k = 0; k <= order_; k++)
         {
            for (int j = 0; j <= order_; j++)
            {
               for (int i = 0; i <= order_; i++, c++)
               {
                  value += T(power_coefficients_(c)) * powers[0][i] *
                           powers[1][j] * powers[2][k];
               }
            }
         }
      }
      return T(sign_) * value;
   }

   template <typename T>
   algoim::uvector<T, N> grad(const algoim::uvector<T, N> &x) const
   {
      std::vector<T> powers[N];
      for (int d = 0; d < N; d++)
      {
         powers[d].resize(order_ + 1);
         powers[d][0] = T(1.0);
         for (int i = 1; i <= order_; i++)
         {
            powers[d][i] = powers[d][i - 1] * x(d);
         }
      }
      algoim::uvector<T, N> gradient = T(0.0);
      const int n = order_ + 1;
      for (int c = 0; c < power_coefficients_.Size(); c++)
      {
         int remainder = c;
         int degree[N];
         for (int d = 0; d < N; d++)
         {
            degree[d] = remainder % n;
            remainder /= n;
         }
         for (int direction = 0; direction < N; direction++)
         {
            if (degree[direction] == 0) { continue; }
            T term = T(power_coefficients_(c) * degree[direction]);
            for (int d = 0; d < N; d++)
            {
               const int exponent = degree[d] - (d == direction ? 1 : 0);
               term *= powers[d][exponent];
            }
            gradient(direction) += term;
         }
      }
      for (int d = 0; d < N; d++) { gradient(d) *= T(sign_); }
      return gradient;
   }

private:
   static long long Binomial(int n, int k)
   {
      if (k < 0 || k > n) { return 0; }
      k = std::min(k, n - k);
      long long value = 1;
      for (int i = 1; i <= k; i++) { value = value*(n - k + i)/i; }
      return value;
   }

   Vector power_coefficients_;
   int order_;
   real_t sign_;
};

template <int N>
bool FaceIsZero(const ElementLevelSet &level_set, int direction, int side)
{
   const int n = level_set.order + 1;
   for (int k = 0; k < (N == 3 ? n : 1); k++)
   {
      for (int j = 0; j < n; j++)
      {
         for (int i = 0; i < n; i++)
         {
            const int index[3] = {i, j, k};
            if (index[direction] != (side ? level_set.order : 0)) { continue; }
            const int c = i + n*(j + n*k);
            if (level_set.coefficients(c) != 0.0) { return false; }
         }
      }
   }
   return true;
}

template <int N>
CutQuadratureStatus GenerateBoundaryInterface(
   const ElementLevelSet &level_set, const CutQuadratureRequest &request,
   ReferenceInterfaceRule &result)
{
   std::vector<std::pair<int, int> > faces;
   for (int direction = 0; direction < N; direction++)
   {
      for (int side = 0; side < 2; side++)
      {
         if (FaceIsZero<N>(level_set, direction, side))
         {
            faces.emplace_back(direction, side);
         }
      }
   }
   const Geometry::Type face_geometry = N == 2 ? Geometry::SEGMENT :
                                             Geometry::SQUARE;
   const IntegrationRule &face_rule = IntRules.Get(face_geometry, request.order);
   const int nq = static_cast<int>(faces.size()) * face_rule.GetNPoints();
   result.rule.SetSize(nq);
   result.rule.SetOrder(request.order);
   if (request.compute_reference_normals)
   {
      result.reference_normals.SetSize(N, nq);
   }
   AlgoimBernsteinLevelSet<N> original(level_set.coefficients, level_set.order);
   real_t scale = 0.0;
   for (int i = 0; i < level_set.coefficients.Size(); i++)
   {
      scale = std::max(scale, std::abs(level_set.coefficients(i)));
   }
   const real_t gradient_tolerance = 64.0 *
      std::numeric_limits<real_t>::epsilon() *
      std::max(1, level_set.order) * scale;
   int q = 0;
   for (const auto &face : faces)
   {
      for (int i = 0; i < face_rule.GetNPoints(); i++, q++)
      {
         algoim::uvector<algoim::real, N> point = 0.0;
         const IntegrationPoint &fp = face_rule.IntPoint(i);
         int tangent = 0;
         for (int d = 0; d < N; d++)
         {
            if (d == face.first) { point(d) = face.second; }
            else { point(d) = tangent++ == 0 ? fp.x : fp.y; }
         }
         IntegrationPoint &ip = result.rule.IntPoint(q);
         if (N == 2) { ip.Set2w(point(0), point(1), fp.weight); }
         else { ip.Set(point(0), point(1), point(2), fp.weight); }
         const auto gradient = original.grad(point);
         real_t norm_squared = 0.0;
         for (int d = 0; d < N; d++)
         {
            norm_squared += gradient(d)*gradient(d);
         }
         const real_t norm = std::sqrt(norm_squared);
         if (!std::isfinite(norm) || norm <= gradient_tolerance)
         {
            return CutQuadratureStatus::DegenerateInterface;
         }
         if (request.compute_reference_normals)
         {
            for (int d = 0; d < N; d++)
            {
               result.reference_normals(d, q) = gradient(d) / norm;
            }
         }
      }
   }
   result.rule.SetPointIndices();
   return CutQuadratureStatus::Success;
}

bool FiniteAndContained(const IntegrationPoint &ip, int dim)
{
   const real_t tolerance = 64.0 * std::numeric_limits<real_t>::epsilon();
   const real_t coordinates[3] = {ip.x, ip.y, ip.z};
   if (!std::isfinite(ip.weight)) { return false; }
   for (int d = 0; d < dim; d++)
   {
      if (!std::isfinite(coordinates[d]) || coordinates[d] < -tolerance ||
          coordinates[d] > 1.0 + tolerance)
      {
         return false;
      }
   }
   return true;
}

template <int N>
bool DegenerateInterfaceOnSampleGrid(const ElementLevelSet &level_set,
                                     real_t scale)
{
   AlgoimBernsteinLevelSet<N> polynomial(level_set.coefficients,
                                        level_set.order);
   const int subdivisions = std::max(2, 2*level_set.order);
   const int point_count = N == 2 ? (subdivisions + 1)*(subdivisions + 1) :
                           (subdivisions + 1)*(subdivisions + 1)*
                           (subdivisions + 1);
   const real_t value_tolerance = 64.0 *
                                  std::numeric_limits<real_t>::epsilon()*scale;
   const real_t gradient_tolerance = value_tolerance *
                                     std::max(1, level_set.order);
   bool found_zero = false;
   bool all_zero_gradients = true;
   for (int index = 0; index < point_count; index++)
   {
      int remainder = index;
      algoim::uvector<algoim::real, N> point;
      for (int d = 0; d < N; d++)
      {
         point(d) = real_t(remainder % (subdivisions + 1))/subdivisions;
         remainder /= subdivisions + 1;
      }
      if (std::abs(polynomial(point)) > value_tolerance) { continue; }
      found_zero = true;
      const auto gradient = polynomial.grad(point);
      real_t norm_squared = 0.0;
      for (int d = 0; d < N; d++)
      {
         norm_squared += gradient(d)*gradient(d);
      }
      all_zero_gradients = all_zero_gradients &&
                           std::sqrt(norm_squared) <= gradient_tolerance;
   }
   return found_zero && all_zero_gradients;
}

template <int N>
CutQuadratureStatus GenerateAlgoim(const ElementLevelSet &level_set,
                                   const CutQuadratureRequest &request,
                                   ReferenceCutQuadrature &result)
{
   real_t minimum = level_set.coefficients.Min();
   real_t maximum = level_set.coefficients.Max();
   real_t scale = 0.0;
   for (int i = 0; i < level_set.coefficients.Size(); i++)
   {
      scale = std::max(scale, std::abs(level_set.coefficients(i)));
   }

   // An identically zero Bernstein polynomial has a positive-measure zero set.
   if (scale == 0.0)
   {
      result.classification = CutCellClass::Degenerate;
      result.status = CutQuadratureStatus::DegenerateVolume;
      return result.status;
   }

   if (request.region == CutRegion::Negative)
   {
      result.classification = (maximum <= 0.0 && minimum < 0.0) ?
                              CutCellClass::Full :
                              (minimum >= 0.0 ? CutCellClass::Empty :
                               CutCellClass::Cut);
   }
   else
   {
      result.classification = (minimum >= 0.0 && maximum > 0.0) ?
                              CutCellClass::Full :
                              (maximum <= 0.0 ? CutCellClass::Empty :
                               CutCellClass::Cut);
   }

   if (HasMeasure(request.measures, CutMeasure::Interface) &&
       result.classification == CutCellClass::Cut &&
       DegenerateInterfaceOnSampleGrid<N>(level_set, scale))
   {
      result.status = CutQuadratureStatus::DegenerateInterface;
      return result.status;
   }

   const int qo = (request.order + 2) / 2;
   const algoim::HyperRectangle<algoim::real, N> box(0.0, 1.0);

   if (HasMeasure(request.measures, CutMeasure::Volume))
   {
      if (result.classification == CutCellClass::Full)
      {
         result.volume = IntRules.Get(level_set.geometry, request.order);
         result.volume.SetOrder(request.order);
      }
      else if (result.classification == CutCellClass::Empty)
      {
         result.volume.SetSize(0);
         result.volume.SetOrder(request.order);
      }
      else
      {
         const real_t sign = request.region == CutRegion::Negative ? 1.0 : -1.0;
         AlgoimBernsteinLevelSet<N> selected(level_set.coefficients,
                                             level_set.order, sign);
         const auto quadrature = algoim::quadGen<N>(selected, box, -1, -1, qo);
         result.volume.SetSize(static_cast<int>(quadrature.nodes.size()));
         result.volume.SetOrder(request.order);
         for (int i = 0; i < result.volume.GetNPoints(); i++)
         {
            IntegrationPoint &ip = result.volume.IntPoint(i);
            if (N == 2)
            {
               ip.Set2w(quadrature.nodes[i].x(0), quadrature.nodes[i].x(1),
                        quadrature.nodes[i].w);
            }
            else
            {
               ip.Set(quadrature.nodes[i].x(0), quadrature.nodes[i].x(1),
                      quadrature.nodes[i].x(2), quadrature.nodes[i].w);
            }
            if (!FiniteAndContained(ip, N))
            {
               result.status = CutQuadratureStatus::GenerationFailure;
               return result.status;
            }
         }
         result.volume.SetPointIndices();
      }
   }

   if (HasMeasure(request.measures, CutMeasure::Interface))
   {
      if (result.classification != CutCellClass::Cut)
      {
         const CutQuadratureStatus boundary_status =
            GenerateBoundaryInterface<N>(level_set, request, result.interface);
         if (boundary_status != CutQuadratureStatus::Success)
         {
            result.status = boundary_status;
            return result.status;
         }
      }
      else
      {
         AlgoimBernsteinLevelSet<N> original(level_set.coefficients,
                                             level_set.order);
         const auto quadrature = algoim::quadGen<N>(original, box, N, -1, qo);
         const int nq = static_cast<int>(quadrature.nodes.size());
         result.interface.rule.SetSize(nq);
         result.interface.rule.SetOrder(request.order);
         if (request.compute_reference_normals)
         {
            result.interface.reference_normals.SetSize(N, nq);
         }
         const real_t gradient_tolerance = 64.0 *
            std::numeric_limits<real_t>::epsilon() * level_set.order * scale;
         bool all_degenerate = nq > 0;
         bool any_degenerate = false;
         for (int i = 0; i < nq; i++)
         {
            IntegrationPoint &ip = result.interface.rule.IntPoint(i);
            algoim::uvector<algoim::real, N> point;
            for (int d = 0; d < N; d++) { point(d) = quadrature.nodes[i].x(d); }
            if (N == 2)
            {
               ip.Set2w(point(0), point(1), quadrature.nodes[i].w);
            }
            else
            {
               ip.Set(point(0), point(1), point(2), quadrature.nodes[i].w);
            }
            if (!FiniteAndContained(ip, N))
            {
               result.status = CutQuadratureStatus::GenerationFailure;
               return result.status;
            }
            const auto gradient = original.grad(point);
            real_t norm_squared = 0.0;
            for (int d = 0; d < N; d++)
            {
               norm_squared += gradient(d) * gradient(d);
            }
            const real_t norm = std::sqrt(norm_squared);
            const bool degenerate = !std::isfinite(norm) ||
                                    norm <= gradient_tolerance;
            all_degenerate = all_degenerate && degenerate;
            any_degenerate = any_degenerate || degenerate;
            if (request.compute_reference_normals && !degenerate)
            {
               for (int d = 0; d < N; d++)
               {
                  result.interface.reference_normals(d, i) = gradient(d) / norm;
               }
            }
         }
         if (all_degenerate)
         {
            result.status = CutQuadratureStatus::DegenerateInterface;
            return result.status;
         }
         if (any_degenerate)
         {
            result.status = CutQuadratureStatus::GenerationFailure;
            return result.status;
         }
         result.interface.rule.SetPointIndices();
      }
   }

   result.status = CutQuadratureStatus::Success;
   return result.status;
}

void Clear(BatchedReferenceCutQuadrature &result)
{
   result.status.SetSize(0);
   result.classification.SetSize(0);
   result.volume.points.SetSize(0, 0);
   result.volume.weights.SetSize(0);
   result.volume.normals.SetSize(0, 0);
   result.volume.offsets.SetSize(0);
   result.interface.points.SetSize(0, 0);
   result.interface.weights.SetSize(0);
   result.interface.normals.SetSize(0, 0);
   result.interface.offsets.SetSize(0);
}

bool AllowedExtractionStatus(CutQuadratureStatus status)
{
   return status == CutQuadratureStatus::Success ||
          status == CutQuadratureStatus::UnsupportedSourceBasis ||
          status == CutQuadratureStatus::InvalidLevelSet;
}

void PackRules(const std::vector<ReferenceCutQuadrature> &local,
               bool interface, int dimension, bool normals,
               PackedReferenceRules &packed)
{
   const int size = static_cast<int>(local.size());
   packed.offsets.SetSize(size + 1);
   packed.offsets[0] = 0;
   for (int i = 0; i < size; i++)
   {
      int count = 0;
      if (local[i].status == CutQuadratureStatus::Success)
      {
         count = interface ? local[i].interface.rule.GetNPoints() :
                 local[i].volume.GetNPoints();
      }
      packed.offsets[i + 1] = packed.offsets[i] + count;
   }
   const int total = packed.offsets[size];
   packed.points.SetSize(dimension, total);
   packed.weights.SetSize(total);
   if (normals) { packed.normals.SetSize(dimension, total); }
   else { packed.normals.SetSize(0, 0); }

   for (int e = 0; e < size; e++)
   {
      if (local[e].status != CutQuadratureStatus::Success) { continue; }
      const IntegrationRule &rule = interface ? local[e].interface.rule :
                                    local[e].volume;
      for (int j = 0; j < rule.GetNPoints(); j++)
      {
         const int p = packed.offsets[e] + j;
         const IntegrationPoint &ip = rule.IntPoint(j);
         packed.points(0, p) = ip.x;
         if (dimension > 1) { packed.points(1, p) = ip.y; }
         if (dimension > 2) { packed.points(2, p) = ip.z; }
         packed.weights(p) = ip.weight;
         if (normals)
         {
            for (int d = 0; d < dimension; d++)
            {
               packed.normals(d, p) =
                  local[e].interface.reference_normals(d, j);
            }
         }
      }
   }
}

} // namespace

AlgoimCutQuadratureGenerator::AlgoimCutQuadratureGenerator()
{
   capabilities_.geometries.Append(Geometry::SQUARE);
   capabilities_.geometries.Append(Geometry::CUBE);
   capabilities_.bases.Append(PolynomialBasis::BernsteinTensor);
   capabilities_.min_order = 0;
   capabilities_.max_order = 19;
   capabilities_.volume = true;
   capabilities_.interface = true;
   capabilities_.negative_phase = true;
   capabilities_.positive_phase = true;
   capabilities_.unconstrained_weights = true;
   capabilities_.nonnegative_weights = true;
   capabilities_.normals = true;
   capabilities_.host_scalar = true;
   capabilities_.host_batch = true;
}

std::unique_ptr<CutQuadratureWorkspace>
AlgoimCutQuadratureGenerator::CreateWorkspace() const
{
   return std::unique_ptr<CutQuadratureWorkspace>(
             new AlgoimCutQuadratureWorkspace);
}

CutQuadratureStatus AlgoimCutQuadratureGenerator::GenerateReference(
   const ElementLevelSet &level_set, const CutQuadratureRequest &request,
   ReferenceCutQuadrature &result, CutQuadratureWorkspace &) const
{
   CutQuadratureStatus status = ValidateRequest(request);
   Reset(result, status);
   if (status != CutQuadratureStatus::Success) { return result.status; }
   if (request.execution != CutExecutionMode::Host)
   {
      result.status = CutQuadratureStatus::UnsupportedExecutionMode;
      return result.status;
   }
   if (level_set.basis != PolynomialBasis::BernsteinTensor)
   {
      result.status = CutQuadratureStatus::UnsupportedPolynomialBasis;
      return result.status;
   }
   if (Dimension(level_set.geometry) == 0)
   {
      result.status = CutQuadratureStatus::UnsupportedGeometry;
      return result.status;
   }
   if (request.order < capabilities_.min_order ||
       request.order > capabilities_.max_order)
   {
      result.status = CutQuadratureStatus::UnsupportedOrder;
      return result.status;
   }
   if (level_set.order < 0 ||
       level_set.coefficients.Size() !=
       CoefficientCount(level_set.geometry, level_set.order))
   {
      result.status = CutQuadratureStatus::InvalidLevelSet;
      return result.status;
   }
   for (int i = 0; i < level_set.coefficients.Size(); i++)
   {
      if (!std::isfinite(level_set.coefficients(i)))
      {
         result.status = CutQuadratureStatus::InvalidLevelSet;
         return result.status;
      }
   }

   try
   {
      return level_set.geometry == Geometry::SQUARE ?
             GenerateAlgoim<2>(level_set, request, result) :
             GenerateAlgoim<3>(level_set, request, result);
   }
   catch (...)
   {
      result.status = CutQuadratureStatus::GenerationFailure;
      return result.status;
   }
}

CutQuadratureStatus AlgoimCutQuadratureGenerator::GenerateReferenceBatch(
   const ElementLevelSetBatch &level_sets,
   const CutQuadratureRequest &request,
   BatchedReferenceCutQuadrature &result,
   CutQuadratureWorkspace &workspace) const
{
   Clear(result);
   CutQuadratureStatus call_status = ValidateRequest(request);
   if (call_status != CutQuadratureStatus::Success) { return call_status; }
   if (request.execution != CutExecutionMode::Host)
   {
      return CutQuadratureStatus::UnsupportedExecutionMode;
   }

   const int size = level_sets.coefficients.Width();
   if (level_sets.element_descriptors.Size() != size ||
       level_sets.extraction_status.Size() != size ||
       (level_sets.descriptor.basis == PolynomialBasis::BernsteinTensor &&
        CoefficientCount(level_sets.descriptor.geometry,
                         level_sets.descriptor.order) >= 0 &&
        level_sets.coefficients.Height() !=
        CoefficientCount(level_sets.descriptor.geometry,
                         level_sets.descriptor.order)))
   {
      return CutQuadratureStatus::InvalidBatch;
   }
   for (int i = 0; i < size; i++)
   {
      if (!AllowedExtractionStatus(level_sets.extraction_status[i]))
      {
         return CutQuadratureStatus::InvalidBatch;
      }
   }
   for (int i = 0; i < size; i++)
   {
      if (level_sets.extraction_status[i] == CutQuadratureStatus::Success &&
          level_sets.element_descriptors[i] != level_sets.descriptor)
      {
         return CutQuadratureStatus::HeterogeneousBatch;
      }
   }

   try
   {
      std::vector<ReferenceCutQuadrature> local(size);
      result.status.SetSize(size);
      result.classification.SetSize(size);
      for (int i = 0; i < size; i++)
      {
         if (level_sets.extraction_status[i] != CutQuadratureStatus::Success)
         {
            local[i].status = level_sets.extraction_status[i];
            local[i].classification = CutCellClass::Unclassified;
         }
         else
         {
            ElementLevelSet level_set;
            level_set.geometry = level_sets.descriptor.geometry;
            level_set.basis = level_sets.descriptor.basis;
            level_set.order = level_sets.descriptor.order;
            level_set.coefficients.SetSize(level_sets.coefficients.Height());
            level_sets.coefficients.GetColumn(i, level_set.coefficients);
            GenerateReference(level_set, request, local[i], workspace);
         }
         result.status[i] = local[i].status;
         result.classification[i] = local[i].classification;
      }
      const int dimension = Dimension(level_sets.descriptor.geometry);
      PackRules(local, false, dimension, false, result.volume);
      PackRules(local, true, dimension,
                request.compute_reference_normals &&
                HasMeasure(request.measures, CutMeasure::Interface),
                result.interface);
   }
   catch (...)
   {
      Clear(result);
      return CutQuadratureStatus::ExecutionFailure;
   }
   return CutQuadratureStatus::Success;
}

} // namespace mfem

#endif // MFEM_USE_ALGOIM
