// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#include "cut_quadrature.hpp"

#ifdef MFEM_USE_ALGOIM

#include "intrules.hpp"

#include <algoim/quadrature_general.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
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
   struct Tensor
   {
      std::array<int, N> degree;
      std::vector<algoim::real> coefficients;
   };

   struct EvaluationData
   {
      Tensor polynomial;
      std::array<Tensor, N> derivatives;
      std::array<std::array<algoim::real, N>, N> hessian_bounds;
   };

public:
   AlgoimBernsteinLevelSet(const Vector &coefficients, int order,
                           real_t selection_sign = 1.0)
      : sign_(selection_sign)
   {
      Tensor polynomial;
      polynomial.degree.fill(order);
      polynomial.coefficients.resize(coefficients.Size());
      for (int i = 0; i < coefficients.Size(); i++)
      {
         polynomial.coefficients[i] = coefficients(i);
      }
      ReduceConstantDirections(polynomial);
      value_ = Prepare(polynomial);
      for (int d = 0; d < N; d++)
      {
         gradient_[d] = Prepare(value_.derivatives[d]);
      }
   }

   template <typename T>
   T operator()(const algoim::uvector<T, N> &x) const
   {
      return Evaluate(value_, x) * sign_;
   }

   template <typename T>
   algoim::uvector<T, N> grad(const algoim::uvector<T, N> &x) const
   {
      algoim::uvector<T, N> gradient = T(0.0);
      for (int d = 0; d < N; d++)
      {
         gradient(d) = Evaluate(gradient_[d], x) * sign_;
      }
      return gradient;
   }

   int Degree(int direction) const
   {
      return value_.polynomial.degree[direction];
   }

   bool HasNonvanishingGradient(algoim::real tolerance) const
   {
      for (const auto &derivative : value_.derivatives)
      {
         const auto bounds = std::minmax_element(derivative.coefficients.begin(),
                                                 derivative.coefficients.end());
         // The convex hull bounds this gradient component throughout the cell.
         // One component bounded away from zero certifies a regular interface.
         if (*bounds.first > tolerance || *bounds.second < -tolerance)
         {
            return true;
         }
      }
      return false;
   }

private:
   // Remove only exactly constant coordinate directions, without approximating
   // or lowering a nonconstant polynomial's degree.
   static void ReduceConstantDirections(Tensor &polynomial)
   {
      int stride = 1;
      for (int d = 0; d < N; d++)
      {
         const int width = polynomial.degree[d] + 1;
         const int size = static_cast<int>(polynomial.coefficients.size());
         bool constant = true;
         for (int i = 0; i < size; i++)
         {
            const int base = i - ((i / stride) % width)*stride;
            if (polynomial.coefficients[i] != polynomial.coefficients[base])
            {
               constant = false;
               break;
            }
         }
         if (constant && width > 1)
         {
            std::vector<algoim::real> reduced(size / width);
            for (int i = 0; i < size / width; i++)
            {
               reduced[i] = polynomial.coefficients[
                               i % stride + (i / stride)*stride*width];
            }
            polynomial.coefficients.swap(reduced);
            polynomial.degree[d] = 0;
         }
         stride *= polynomial.degree[d] + 1;
      }
   }

   static Tensor Differentiate(const Tensor &polynomial, int direction)
   {
      Tensor derivative;
      const int degree = polynomial.degree[direction];
      if (degree == 0)
      {
         derivative.degree.fill(0);
         derivative.coefficients.assign(1, 0.0);
         return derivative;
      }
      derivative.degree = polynomial.degree;
      derivative.degree[direction]--;
      int stride = 1;
      for (int d = 0; d < direction; d++)
      {
         stride *= polynomial.degree[d] + 1;
      }
      const int size = static_cast<int>(polynomial.coefficients.size()) /
                       (degree + 1)*degree;
      derivative.coefficients.resize(size);
      for (int i = 0; i < size; i++)
      {
         const int source = i % stride +
                            ((i / stride) % degree)*stride +
                            (i / (stride*degree))*stride*(degree + 1);
         derivative.coefficients[i] = algoim::real(degree) *
                                      (polynomial.coefficients[source + stride] -
                                       polynomial.coefficients[source]);
         if (!std::isfinite(derivative.coefficients[i]))
         {
            throw std::overflow_error("Nonfinite Bernstein derivative");
         }
      }
      ReduceConstantDirections(derivative);
      return derivative;
   }

   static EvaluationData Prepare(const Tensor &polynomial)
   {
      EvaluationData data;
      data.polynomial = polynomial;
      for (int d = 0; d < N; d++)
      {
         data.derivatives[d] = Differentiate(polynomial, d);
         for (int e = 0; e < N; e++)
         {
            const Tensor second = Differentiate(data.derivatives[d], e);
            algoim::real bound = 0.0;
            for (const auto coefficient : second.coefficients)
            {
               bound = std::max(bound, std::abs(coefficient));
            }
            // Bernstein coefficients bound each derivative on [0,1]^N.
            data.hessian_bounds[d][e] = bound;
         }
      }
      return data;
   }

   template <typename T>
   T EvaluateTensor(const Tensor &polynomial,
                    const algoim::uvector<T, N> &x) const
   {
      const int size = static_cast<int>(polynomial.coefficients.size());
      if (size == 1) { return T(polynomial.coefficients[0]); }
      auto &values = Scratch(T(0.0));
      if (values.size() < polynomial.coefficients.size())
      {
         values.resize(polynomial.coefficients.size());
      }
      std::copy(polynomial.coefficients.begin(), polynomial.coefficients.end(),
                values.begin());
      int remaining_size = size;
      for (int d = 0; d < N; d++)
      {
         const int width = polynomial.degree[d] + 1;
         if (width == 1) { continue; }
         const int fibers = remaining_size / width;
         for (int f = 0; f < fibers; f++)
         {
            const int base = f*width;
            for (int remaining = width - 1; remaining > 0; remaining--)
            {
               for (int i = 0; i < remaining; i++)
               {
                  values[base + i] = values[base + i] +
                                     x(d)*(values[base + i + 1] -
                                           values[base + i]);
               }
            }
            values[f] = values[base];
         }
         remaining_size = fibers;
      }
      return values[0];
   }

   template <typename T>
   T Evaluate(const EvaluationData &data,
              const algoim::uvector<T, N> &x) const
   {
      return EvaluateTensor(data.polynomial, x);
   }

   algoim::Interval<N> Evaluate(
      const EvaluationData &data,
      const algoim::uvector<algoim::Interval<N>, N> &x) const
   {
      algoim::uvector<algoim::real, N> center = 0.0;
      algoim::uvector<algoim::real, N> radius = 0.0;
      for (int d = 0; d < N; d++)
      {
         center(d) = x(d).alpha;
         radius(d) = x(d).maxDeviation();
         if (center(d) - radius(d) < 0.0 ||
             center(d) + radius(d) > 1.0)
         {
            // Convex-hull bounds below apply on the reference element only.
            return EvaluateTensor(data.polynomial, x);
         }
      }

      // A centered Taylor enclosure avoids the repeated interval dependency
      // introduced by directly applying de Casteljau to interval coordinates.
      algoim::uvector<algoim::real, N> beta = 0.0;
      algoim::real remainder = 0.0;
      for (int d = 0; d < N; d++)
      {
         const algoim::real derivative = EvaluateTensor(data.derivatives[d],
                                                        center);
         for (int e = 0; e < N; e++)
         {
            beta(e) += derivative*x(d).beta(e);
            remainder += 0.5*data.hessian_bounds[d][e]*radius(d)*radius(e);
         }
         remainder += std::abs(derivative)*x(d).eps;
      }
      return algoim::Interval<N>(EvaluateTensor(data.polynomial, center),
                                 beta, remainder);
   }

   std::vector<algoim::real> &Scratch(algoim::real) const
   {
      return scalar_scratch_;
   }

   std::vector<algoim::Interval<N>> &Scratch(const algoim::Interval<N> &) const
   {
      return interval_scratch_;
   }

   EvaluationData value_;
   std::array<EvaluationData, N> gradient_;
   algoim::real sign_;
   // Each adapter belongs to one generation call. Algoim evaluates it
   // sequentially, so scalar and interval buffers can be reused across values
   // and gradient components without sharing mutable state between callers.
   mutable std::vector<algoim::real> scalar_scratch_;
   mutable std::vector<algoim::Interval<N>> interval_scratch_;
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
void DeflateZeroFace(ElementLevelSet &polynomial, int direction, int side)
{
   // Divide a Bernstein polynomial by x_d or (1-x_d), then elevate the
   // quotient back to the original degree to preserve the tensor layout.
   // The caller has established that the restriction to this face is zero.
   const int p = polynomial.order;
   const int n = p + 1;
   int stride = 1;
   for (int d = 0; d < direction; d++) { stride *= n; }
   std::vector<algoim::real> quotient(p);
   for (int base = 0; base < polynomial.coefficients.Size(); base++)
   {
      if ((base / stride) % n != 0) { continue; }
      for (int i = 0; i < p; i++)
      {
         const int source = side == 0 ? i + 1 : i;
         const int divisor = side == 0 ? i + 1 : p - i;
         quotient[i] = algoim::real(p) *
                       polynomial.coefficients(base + source*stride) / divisor;
      }
      for (int i = 0; i <= p; i++)
      {
         algoim::real value = 0.0;
         if (i > 0) { value += algoim::real(i)/p * quotient[i - 1]; }
         if (i < p) { value += algoim::real(p - i)/p * quotient[i]; }
         polynomial.coefficients(base + i*stride) = static_cast<real_t>(value);
      }
   }
}

template <int N>
CutQuadratureStatus DeflateZeroFaces(const ElementLevelSet &original,
                                     ElementLevelSet &interior)
{
   interior = original;
   for (int direction = 0; direction < N; direction++)
   {
      for (int side = 0; side < 2; side++)
      {
         if (!FaceIsZero<N>(original, direction, side)) { continue; }
         // Remove all multiplicities, without allowing an unbounded loop.
         for (int multiplicity = 0; multiplicity < original.order &&
              FaceIsZero<N>(interior, direction, side); multiplicity++)
         {
            DeflateZeroFace<N>(interior, direction, side);
         }
         if (FaceIsZero<N>(interior, direction, side))
         {
            return CutQuadratureStatus::GenerationFailure;
         }
      }
   }
   for (int i = 0; i < interior.coefficients.Size(); i++)
   {
      if (!std::isfinite(interior.coefficients(i)))
      {
         return CutQuadratureStatus::GenerationFailure;
      }
   }
   return CutQuadratureStatus::Success;
}

void AppendInterfaceRule(ReferenceInterfaceRule &destination,
                         const ReferenceInterfaceRule &source, bool normals)
{
   const int first = destination.rule.GetNPoints();
   const int second = source.rule.GetNPoints();
   if (first == 0)
   {
      // Also preserve the dim-by-zero normal matrix for an empty interface.
      destination = source;
      return;
   }
   if (second == 0) { return; }

   IntegrationRule combined(first + second);
   combined.SetOrder(source.rule.GetOrder());
   for (int i = 0; i < first; i++)
   {
      combined.IntPoint(i) = destination.rule.IntPoint(i);
   }
   for (int i = 0; i < second; i++)
   {
      combined.IntPoint(first + i) = source.rule.IntPoint(i);
   }
   if (normals)
   {
      const int dim = source.reference_normals.Height();
      DenseMatrix combined_normals(dim, first + second);
      for (int d = 0; d < dim; d++)
      {
         for (int i = 0; i < first; i++)
         {
            combined_normals(d, i) = destination.reference_normals(d, i);
         }
         for (int i = 0; i < second; i++)
         {
            combined_normals(d, first + i) = source.reference_normals(d, i);
         }
      }
      destination.reference_normals = combined_normals;
   }
   destination.rule = combined;
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
   if (nq == 0) { return CutQuadratureStatus::Success; }
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
   const real_t value_tolerance = 64.0 *
                                  std::numeric_limits<real_t>::epsilon()*scale;
   const real_t gradient_tolerance = value_tolerance *
                                     std::max(1, level_set.order);
   if (polynomial.HasNonvanishingGradient(gradient_tolerance)) { return false; }
   std::array<int, N> widths;
   int point_count = 1;
   for (int d = 0; d < N; d++)
   {
      // Constant directions contribute identical values and gradients at all
      // grid coordinates. Keep the original grid in every dependent direction.
      widths[d] = polynomial.Degree(d) == 0 ? 1 : subdivisions + 1;
      point_count *= widths[d];
   }
   bool found_zero = false;
   for (int index = 0; index < point_count; index++)
   {
      int remainder = index;
      algoim::uvector<algoim::real, N> point;
      for (int d = 0; d < N; d++)
      {
         point(d) = real_t(remainder % widths[d])/subdivisions;
         remainder /= widths[d];
      }
      if (std::abs(polynomial(point)) > value_tolerance) { continue; }
      found_zero = true;
      const auto gradient = polynomial.grad(point);
      real_t norm_squared = 0.0;
      for (int d = 0; d < N; d++)
      {
         norm_squared += gradient(d)*gradient(d);
      }
      if (!(std::sqrt(norm_squared) <= gradient_tolerance)) { return false; }
   }
   return found_zero;
}

template <int N>
CutQuadratureStatus GenerateAlgoim(const ElementLevelSet &input,
                                   const CutQuadratureRequest &request,
                                   ReferenceCutQuadrature &result)
{
   const real_t minimum = input.coefficients.Min();
   const real_t maximum = input.coefficients.Max();
   real_t coefficient_norm = 0.0;
   for (int i = 0; i < input.coefficients.Size(); i++)
   {
      coefficient_norm = std::max(coefficient_norm,
                                  std::abs(input.coefficients(i)));
   }

   // An identically zero Bernstein polynomial has a positive-measure zero set.
   if (coefficient_norm == 0.0)
   {
      result.classification = CutCellClass::Degenerate;
      result.status = CutQuadratureStatus::DegenerateVolume;
      return result.status;
   }

   // A positive rescaling preserves the zero set, phase signs, and gradient
   // orientation. Normalize before deflation, Algoim evaluation, and gradient
   // norm checks so large/small overall scales cannot overflow/underflow them.
   // Divide directly: the reciprocal of a tiny finite norm can overflow.
   ElementLevelSet level_set(input);
   for (int i = 0; i < level_set.coefficients.Size(); i++)
   {
      level_set.coefficients(i) /= coefficient_norm;
   }
   const real_t scale = 1.0;

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

   ElementLevelSet interior;
   if (result.classification == CutCellClass::Cut)
   {
      // x_d and (1-x_d) are strictly positive inside the reference cell.
      // Removing these boundary factors preserves both open volume phases and
      // also prevents Algoim from root-finding an identically zero face
      // restriction during volume dimension reduction.
      const CutQuadratureStatus deflation_status =
         DeflateZeroFaces<N>(level_set, interior);
      if (deflation_status != CutQuadratureStatus::Success)
      {
         result.status = deflation_status;
         return result.status;
      }
   }

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
         AlgoimBernsteinLevelSet<N> selected(interior.coefficients,
                                             interior.order, sign);
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
      // Boundary components are independent of the volume classification.
      ReferenceInterfaceRule boundary;
      const CutQuadratureStatus boundary_status =
         GenerateBoundaryInterface<N>(level_set, request, boundary);
      if (boundary_status != CutQuadratureStatus::Success)
      {
         result.status = boundary_status;
         return result.status;
      }
      if (result.classification == CutCellClass::Cut)
      {
         // The same deflated polynomial generates only the interior interface;
         // known zero faces are assigned to the boundary rule exactly once.
         AlgoimBernsteinLevelSet<N> interior_phi(interior.coefficients,
                                                 interior.order);
         // Normal orientation and degeneracy checks still use the original phi.
         AlgoimBernsteinLevelSet<N> original(level_set.coefficients,
                                             level_set.order);
         const auto quadrature = algoim::quadGen<N>(interior_phi, box, N, -1, qo);
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
      }
      AppendInterfaceRule(result.interface, boundary,
                          request.compute_reference_normals);
      result.interface.rule.SetOrder(request.order);
      result.interface.rule.SetPointIndices();
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

AlgoimCutQuadratureConstructor::AlgoimCutQuadratureConstructor()
{
   capabilities_.geometries.Append(Geometry::SQUARE);
   capabilities_.geometries.Append(Geometry::CUBE);
   capabilities_.bases.Append(PolynomialBasis::BernsteinTensor);
   capabilities_.min_order = 0;
   capabilities_.max_order = 19;
   capabilities_.min_polynomial_degree = 0;
   // Resource policy for tensor evaluation, also bounding signed-int indexing
   // and the (2*degree + 1)^N interface sampling grid before generation starts.
   capabilities_.max_polynomial_degree = 64;
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
AlgoimCutQuadratureConstructor::CreateWorkspace() const
{
   return std::unique_ptr<CutQuadratureWorkspace>(
             new AlgoimCutQuadratureWorkspace);
}

CutQuadratureStatus AlgoimCutQuadratureConstructor::GenerateReference(
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
   if (level_set.order < 0)
   {
      result.status = CutQuadratureStatus::InvalidLevelSet;
      return result.status;
   }
   if (level_set.order < capabilities_.min_polynomial_degree ||
       level_set.order > capabilities_.max_polynomial_degree)
   {
      result.status = CutQuadratureStatus::UnsupportedPolynomialDegree;
      return result.status;
   }
   if (level_set.coefficients.Size() !=
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

CutQuadratureStatus AlgoimCutQuadratureConstructor::GenerateReferenceBatch(
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
