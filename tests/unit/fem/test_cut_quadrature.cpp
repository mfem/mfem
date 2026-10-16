// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

using namespace mfem;

namespace
{

ElementLevelSet SquareLinear(real_t left, real_t right)
{
   ElementLevelSet level_set;
   level_set.geometry = Geometry::SQUARE;
   level_set.order = 1;
   level_set.basis = PolynomialBasis::BernsteinTensor;
   level_set.coefficients.SetSize(4);
   level_set.coefficients[0] = left;
   level_set.coefficients[1] = right;
   level_set.coefficients[2] = left;
   level_set.coefficients[3] = right;
   return level_set;
}

#ifdef MFEM_USE_ALGOIM
real_t WeightSum(const IntegrationRule &rule)
{
   real_t sum = 0.0;
   for (int i = 0; i < rule.GetNPoints(); i++) { sum += rule[i].weight; }
   return sum;
}
#endif

class TestExtractor : public ElementLevelSetExtractor
{
public:
   explicit TestExtractor(LevelSetRevision revision = 0) : revision_(revision) { }
   CutQuadratureStatus GetElementLevelSet(
      int, ElementTransformation &, ElementLevelSet &) const override
   { return CutQuadratureStatus::InvalidLevelSet; }
   LevelSetRevision Revision() const override { return revision_; }
   void Bump() { revision_++; }
private:
   LevelSetRevision revision_;
};

class MockWorkspace : public CutQuadratureWorkspace { };

class MockCutConstructor : public CutQuadratureConstructor
{
public:
   MockCutConstructor(bool nonnegative, bool fail_batch)
      : fail_batch_(fail_batch)
   {
      caps_.geometries.Append(Geometry::SQUARE);
      caps_.bases.Append(PolynomialBasis::BernsteinTensor);
      caps_.min_order = 0;
      caps_.max_order = 20;
      caps_.volume = true;
      caps_.negative_phase = caps_.positive_phase = true;
      caps_.unconstrained_weights = true;
      caps_.nonnegative_weights = nonnegative;
      caps_.host_scalar = caps_.host_batch = true;
   }
   const CutQuadratureCapabilities &Capabilities() const override
   { return caps_; }
   std::unique_ptr<CutQuadratureWorkspace> CreateWorkspace() const override
   { return std::unique_ptr<CutQuadratureWorkspace>(new MockWorkspace); }
   CutQuadratureStatus GenerateReference(
      const ElementLevelSet &, const CutQuadratureRequest &request,
      ReferenceCutQuadrature &result, CutQuadratureWorkspace &) const override
   {
      result = ReferenceCutQuadrature();
      if (request.weight_policy == QuadratureWeightPolicy::Nonnegative &&
          !caps_.nonnegative_weights)
      {
         result.status = CutQuadratureStatus::UnsupportedWeightPolicy;
         return result.status;
      }
      result.classification = CutCellClass::Cut;
      result.volume.SetSize(1);
      result.volume[0].Set2w(0.5, 0.5,
                             caps_.nonnegative_weights ? 1.0 : -1.0);
      result.volume.SetPointIndices();
      return result.status;
   }
   CutQuadratureStatus GenerateReferenceBatch(
      const ElementLevelSetBatch &batch, const CutQuadratureRequest &request,
      BatchedReferenceCutQuadrature &result,
      CutQuadratureWorkspace &) const override
   {
      result.status.SetSize(0);
      result.classification.SetSize(0);
      const unsigned measures = static_cast<unsigned>(request.measures);
      const unsigned allowed = static_cast<unsigned>(CutMeasure::Volume) |
                               static_cast<unsigned>(CutMeasure::Interface);
      if (request.order < 0 || measures == 0u || (measures & ~allowed) != 0u)
      {
         return CutQuadratureStatus::InvalidRequest;
      }
      if (request.execution != CutExecutionMode::Host)
      {
         return CutQuadratureStatus::UnsupportedExecutionMode;
      }
      const int size = batch.coefficients.Width();
      if (batch.element_descriptors.Size() != size ||
          batch.extraction_status.Size() != size)
      {
         return CutQuadratureStatus::InvalidBatch;
      }
      for (int i = 0; i < size; i++)
      {
         const CutQuadratureStatus status = batch.extraction_status[i];
         if (status != CutQuadratureStatus::Success &&
             status != CutQuadratureStatus::UnsupportedSourceBasis &&
             status != CutQuadratureStatus::InvalidLevelSet)
         {
            return CutQuadratureStatus::InvalidBatch;
         }
      }
      for (int i = 0; i < size; i++)
      {
         if (batch.extraction_status[i] == CutQuadratureStatus::Success &&
             batch.element_descriptors[i] != batch.descriptor)
         {
            return CutQuadratureStatus::HeterogeneousBatch;
         }
      }
      return fail_batch_ ? CutQuadratureStatus::ExecutionFailure :
             CutQuadratureStatus::Success;
   }
private:
   CutQuadratureCapabilities caps_;
   bool fail_batch_;
};

} // namespace

static_assert(!std::is_copy_constructible<TestExtractor>::value,
              "extractor copy");
static_assert(!std::is_move_constructible<TestExtractor>::value,
              "extractor move");

TEST_CASE("Cut quadrature value semantics and retention", "[CutQuadrature]")
{
   const CutMeasure both = CutMeasure::Volume | CutMeasure::Interface;
   REQUIRE(static_cast<unsigned>(both & CutMeasure::Volume) != 0u);
   REQUIRE(static_cast<unsigned>(both & CutMeasure::Interface) != 0u);

   ElementLevelSetDescriptor a =
   { Geometry::SQUARE, PolynomialBasis::BernsteinTensor, 2 };
   ElementLevelSetDescriptor b = a;
   REQUIRE(a == b);
   b.geometry = Geometry::CUBE; REQUIRE(a != b); b = a;
   b.basis = PolynomialBasis::BernsteinSimplex; REQUIRE(a != b); b = a;
   b.order++; REQUIRE(a != b);

   CutQuadratureRequest request, changed;
   REQUIRE(request == changed);
   changed.order++; REQUIRE(request != changed); changed = request;
   changed.region = CutRegion::Positive; REQUIRE(request != changed);
   changed = request; changed.measures = both; REQUIRE(request != changed);
   changed = request;
   changed.weight_policy = QuadratureWeightPolicy::Nonnegative;
   REQUIRE(request != changed);
   changed = request; changed.execution = CutExecutionMode::Device;
   REQUIRE(request != changed);
   changed = request; changed.compute_reference_normals = true;
   REQUIRE(request != changed);

   TestExtractor extractor(7), other(7);
   RetainedCutQuadrature retained;
   retained.extractor_id = extractor.Id();
   retained.element = 3;
   retained.revision = extractor.Revision();
   retained.request = request;
   REQUIRE(retained.IsValid(extractor, 3, request));
   REQUIRE_FALSE(retained.IsValid(other, 3, request));
   REQUIRE_FALSE(retained.IsValid(extractor, 4, request));
   extractor.Bump();
   REQUIRE_FALSE(retained.IsValid(extractor, 3, request));

   RetainedBatchedCutQuadrature retained_batch;
   retained_batch.extractor_id = other.Id();
   retained_batch.revision = other.Revision();
   retained_batch.request = request;
   retained_batch.elements.Append(1);
   retained_batch.elements.Append(4);
   Array<int> same_elements, reordered_elements;
   same_elements.Append(1); same_elements.Append(4);
   reordered_elements.Append(4); reordered_elements.Append(1);
   REQUIRE(retained_batch.IsValid(other, same_elements, request));
   REQUIRE_FALSE(retained_batch.IsValid(other, reordered_elements, request));

   // A missed bump is intentionally undetectable and permits stale reuse.
   TestExtractor missed_bump(4);
   retained.extractor_id = missed_bump.Id();
   retained.element = 0;
   retained.revision = 4;
   REQUIRE(retained.IsValid(missed_bump, 0, request));

   std::vector<std::uint64_t> ids(64);
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel for
#endif
   for (int i = 0; i < 64; i++)
   {
      TestExtractor p;
      ids[i] = p.Id();
   }
   std::sort(ids.begin(), ids.end());
   REQUIRE(std::adjacent_find(ids.begin(), ids.end()) == ids.end());
}

TEST_CASE("Cut quadrature mock weight and execution contracts",
          "[CutQuadrature]")
{
   ElementLevelSet level_set = SquareLinear(-0.5, 0.5);
   CutQuadratureRequest request;
   MockCutConstructor signed_constructor(false, false);
   auto workspace = signed_constructor.CreateWorkspace();
   ReferenceCutQuadrature result;
   REQUIRE(signed_constructor.GenerateReference(level_set, request, result,
                                                *workspace) == result.status);
   REQUIRE(result.volume[0].weight < 0.0);
   request.weight_policy = QuadratureWeightPolicy::Nonnegative;
   REQUIRE(signed_constructor.GenerateReference(level_set, request, result,
                                                *workspace) ==
           CutQuadratureStatus::UnsupportedWeightPolicy);
   REQUIRE(result.classification == CutCellClass::Unclassified);

   MockCutConstructor nonnegative_constructor(true, false);
   workspace = nonnegative_constructor.CreateWorkspace();
   REQUIRE(nonnegative_constructor.GenerateReference(level_set, request, result,
                                                     *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.volume[0].weight >= 0.0);

   MockCutConstructor failing_constructor(true, true);
   ElementLevelSetBatch batch;
   batch.descriptor =
   { Geometry::SQUARE, PolynomialBasis::BernsteinTensor, 1 };
   batch.coefficients.SetSize(4, 1);
   batch.coefficients.SetCol(0, level_set.coefficients);
   batch.element_descriptors.Append(batch.descriptor);
   batch.extraction_status.Append(CutQuadratureStatus::Success);
   BatchedReferenceCutQuadrature batch_result;
   workspace = failing_constructor.CreateWorkspace();
   REQUIRE(failing_constructor.GenerateReferenceBatch(batch, request,
                                                      batch_result,
                                                      *workspace) ==
           CutQuadratureStatus::ExecutionFailure);
   REQUIRE(batch_result.status.Size() == 0);
}

#ifdef MFEM_USE_ALGOIM

TEST_CASE("Algoim cut quadrature scalar contracts", "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   ReferenceCutQuadrature result;
   CutQuadratureRequest request;

   REQUIRE(constructor.Capabilities().min_order == 0);
   REQUIRE(constructor.Capabilities().max_order == 19);
   REQUIRE_FALSE(constructor.Capabilities().device_batch);

   ElementLevelSet cut = SquareLinear(-0.5, 0.5);
   REQUIRE(constructor.GenerateReference(cut, request, result, *workspace) ==
           result.status);
   REQUIRE(result.status == CutQuadratureStatus::Success);
   REQUIRE(result.classification == CutCellClass::Cut);
   REQUIRE(WeightSum(result.volume) == MFEM_Approx(0.5));
   for (int i = 0; i < result.volume.GetNPoints(); i++)
   {
      REQUIRE(result.volume[i].index == i);
      REQUIRE(std::isfinite(result.volume[i].weight));
      REQUIRE(result.volume[i].weight >= 0.0);
      REQUIRE(result.volume[i].x >= 0.0);
      REQUIRE(result.volume[i].x <= 1.0);
      REQUIRE(result.volume[i].y >= 0.0);
      REQUIRE(result.volume[i].y <= 1.0);
   }

   request.region = CutRegion::Positive;
   REQUIRE(constructor.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(WeightSum(result.volume) == MFEM_Approx(0.5));

   ElementLevelSet full = SquareLinear(-1.0, -1.0);
   request.region = CutRegion::Negative;
   REQUIRE(constructor.GenerateReference(full, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.classification == CutCellClass::Full);
   REQUIRE(WeightSum(result.volume) == MFEM_Approx(1.0));

   ElementLevelSet empty = SquareLinear(1.0, 1.0);
   REQUIRE(constructor.GenerateReference(empty, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.classification == CutCellClass::Empty);
   REQUIRE(result.volume.GetNPoints() == 0);

   ElementLevelSet zero = SquareLinear(0.0, 0.0);
   REQUIRE(constructor.GenerateReference(zero, request, result, *workspace) ==
           CutQuadratureStatus::DegenerateVolume);
   REQUIRE(result.classification == CutCellClass::Degenerate);
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   REQUIRE(constructor.GenerateReference(zero, request, result, *workspace) ==
           CutQuadratureStatus::DegenerateVolume);
   REQUIRE(result.classification == CutCellClass::Degenerate);

   request = CutQuadratureRequest();
   request.order = -1;
   REQUIRE(constructor.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::InvalidRequest);
   REQUIRE(result.classification == CutCellClass::Unclassified);
   request.order = 20;
   REQUIRE(constructor.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::UnsupportedOrder);
   REQUIRE(result.classification == CutCellClass::Unclassified);
   request.order = 4;
   request.execution = CutExecutionMode::Device;
   REQUIRE(constructor.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::UnsupportedExecutionMode);
   REQUIRE(result.classification == CutCellClass::Unclassified);
   request.execution = CutExecutionMode::Host;
   request.measures = static_cast<CutMeasure>(0u);
   REQUIRE(constructor.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::InvalidRequest);
   request.measures = static_cast<CutMeasure>(8u);
   REQUIRE(constructor.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::InvalidRequest);

   request = CutQuadratureRequest();
   cut.basis = PolynomialBasis::BernsteinSimplex;
   REQUIRE(constructor.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::UnsupportedPolynomialBasis);
   REQUIRE(result.classification == CutCellClass::Unclassified);
   cut = SquareLinear(-0.5, 0.5);
   cut.geometry = Geometry::TRIANGLE;
   REQUIRE(constructor.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::UnsupportedGeometry);
   REQUIRE(result.classification == CutCellClass::Unclassified);
   cut = SquareLinear(-0.5, 0.5);
   cut.coefficients.SetSize(3);
   REQUIRE(constructor.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::InvalidLevelSet);
   REQUIRE(result.classification == CutCellClass::Unclassified);
}

TEST_CASE("Algoim interface and normal contracts", "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   ReferenceCutQuadrature result;
   CutQuadratureRequest request;
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;

   ElementLevelSet boundary = SquareLinear(0.0, 1.0);
   REQUIRE(constructor.GenerateReference(boundary, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.classification == CutCellClass::Empty);
   REQUIRE(result.interface.rule.GetNPoints() > 0);
   REQUIRE(WeightSum(result.interface.rule) == MFEM_Approx(1.0));
   for (int i = 0; i < result.interface.rule.GetNPoints(); i++)
   {
      REQUIRE(result.interface.reference_normals(0, i) > 0.0);
   }

   ElementLevelSet cut = SquareLinear(-0.5, 0.5);
   REQUIRE(constructor.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   const DenseMatrix negative_normals(result.interface.reference_normals);
   request.region = CutRegion::Positive;
   REQUIRE(constructor.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   for (int i = 0; i < result.interface.rule.GetNPoints(); i++)
   {
      REQUIRE(result.interface.reference_normals(0, i) ==
              MFEM_Approx(negative_normals(0, i)));
   }
}

TEST_CASE("Algoim preserves cuts under extreme coefficient rescaling",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   // The reported scales require double storage. Use representable extremes
   // that also expose gradient-norm underflow/overflow in single precision.
   const bool double_precision = std::numeric_limits<real_t>::digits > 24;
   const real_t tiny = static_cast<real_t>(double_precision ? 1e-200 : 1e-20);
   const real_t large = static_cast<real_t>(double_precision ? 1e200 : 1e20);
   const real_t tolerance = std::max(real_t(1e-11),
                                     128*std::numeric_limits<real_t>::epsilon());
   for (const int dim : {2, 3})
   {
      for (const int example : {0, 1, 2})
      {
         // x-0.4; (x-0.25)(x-1); x. Cover interior interfaces, a
         // combined interior/boundary interface, and Full/Empty zero faces.
         const int degree = example == 1 ? 2 : 1;
         const real_t row[3] =
         {
            example == 0 ? real_t(-0.4) :
            (example == 1 ? real_t(0.25) : real_t(0.0)),
            example == 0 ? real_t(0.6) :
            (example == 1 ? real_t(-0.375) : real_t(1.0)), 0.0
         };
         for (const real_t scale :
              {
                 real_t(1.0), tiny, large, -tiny, -large,
                 std::numeric_limits<real_t>::min(),
                 std::numeric_limits<real_t>::max()
              })
         {
            CAPTURE(dim, example, scale);
            ElementLevelSet polynomial;
            polynomial.geometry = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
            polynomial.order = degree;
            const int width = degree + 1;
            polynomial.coefficients.SetSize(dim == 2 ? width*width :
                                            width*width*width);
            for (int i = 0; i < polynomial.coefficients.Size(); i++)
            {
               polynomial.coefficients(i) = scale*row[i % width];
            }
            const Vector input_coefficients(polynomial.coefficients);
            CutQuadratureRequest request;
            request.order = 2; // All interface components are planar.
            request.measures = CutMeasure::Volume | CutMeasure::Interface;
            request.compute_reference_normals = true;
            for (const auto region : {CutRegion::Negative, CutRegion::Positive})
            {
               CAPTURE(region);
               request.region = region;
               ReferenceCutQuadrature result;
               REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                                     *workspace) ==
                       CutQuadratureStatus::Success);
               const bool original_negative =
                  (region == CutRegion::Negative) == (scale > 0.0);
               const real_t negative_volume = example == 0 ? real_t(0.4) :
                                              (example == 1 ? real_t(0.75) :
                                               real_t(0.0));
               REQUIRE(WeightSum(result.volume) ==
                       MFEM_Approx(original_negative ? negative_volume :
                                   1.0 - negative_volume, tolerance, tolerance));
               const CutCellClass classification = example != 2 ? CutCellClass::Cut :
                                                   (original_negative ? CutCellClass::Empty :
                                                    CutCellClass::Full);
               REQUIRE(result.classification == classification);
               REQUIRE(WeightSum(result.interface.rule) ==
                       MFEM_Approx(example == 1 ? 2.0 : 1.0, tolerance, tolerance));
               for (int q = 0; q < result.interface.rule.GetNPoints(); q++)
               {
                  const auto &ip = result.interface.rule[q];
                  const bool boundary = example == 1 && ip.x == real_t(1.0);
                  const real_t coordinate = example == 0 ? real_t(0.4) :
                                            (example == 1 ? (boundary ? real_t(1.0) : real_t(0.25)) :
                                             real_t(0.0));
                  REQUIRE(ip.x == MFEM_Approx(coordinate, tolerance, tolerance));
                  const real_t normal = (scale > 0.0 ? 1.0 : -1.0) *
                                        (example == 1 && !boundary ? -1.0 : 1.0);
                  for (int d = 0; d < dim; d++)
                  {
                     REQUIRE(result.interface.reference_normals(d, q) ==
                             MFEM_Approx(d == 0 ? normal : real_t(0.0),
                                         tolerance, tolerance));
                  }
               }
               for (int i = 0; i < input_coefficients.Size(); i++)
               {
                  REQUIRE(polynomial.coefficients(i) == input_coefficients(i));
               }
            }
         }
      }
   }
}

TEST_CASE("Algoim preserves degree-elevated linear cuts",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   const real_t tolerance = std::max(real_t(1e-11),
                                     128*std::numeric_limits<real_t>::epsilon());
   for (const int degree : {36, 62, 64})
   {
      for (int direction = 0; direction < 2; direction++)
      {
         CAPTURE(degree, direction);
         ElementLevelSet polynomial;
         polynomial.geometry = Geometry::SQUARE;
         polynomial.order = degree;
         const int width = degree + 1;
         polynomial.coefficients.SetSize(width*width);
         for (int j = 0; j < width; j++)
         {
            for (int i = 0; i < width; i++)
            {
               // Exact Bernstein degree elevation of phi = x_d - 0.4,
               // up to storage rounding; no interpolation or basis conversion.
               const int index = direction == 0 ? i : j;
               polynomial.coefficients(i + width*j) =
                  real_t(index)/degree - real_t(0.4);
            }
         }
         CutQuadratureRequest request;
         // The cut is planar; a low integration order still exercises the full
         // elevated level-set degree while avoiding redundant rule points.
         request.order = 2;
         request.measures = CutMeasure::Volume | CutMeasure::Interface;
         request.compute_reference_normals = true;
         ReferenceCutQuadrature result;
         for (const auto region : {CutRegion::Negative, CutRegion::Positive})
         {
            CAPTURE(region);
            request.region = region;
            if (region == CutRegion::Positive)
            {
               // Interface geometry is phase-independent and its orientation
               // is checked separately below and in the normal contracts test.
               // Preserve both phase volumes without regenerating the interface.
               request.measures = CutMeasure::Volume;
               request.compute_reference_normals = false;
            }
            REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                                  *workspace) ==
                    CutQuadratureStatus::Success);
            REQUIRE(WeightSum(result.volume) ==
                    MFEM_Approx(region == CutRegion::Negative ? 0.4 : 0.6,
                                tolerance, tolerance));
            if (region == CutRegion::Positive) { continue; }
            REQUIRE(WeightSum(result.interface.rule) ==
                    MFEM_Approx(1.0, tolerance, tolerance));
            REQUIRE(result.interface.rule.GetNPoints() > 0);
            for (int q = 0; q < result.interface.rule.GetNPoints(); q++)
            {
               const auto &ip = result.interface.rule[q];
               REQUIRE((direction == 0 ? ip.x : ip.y) ==
                       MFEM_Approx(0.4, tolerance, tolerance));
               for (int d = 0; d < 2; d++)
               {
                  REQUIRE(result.interface.reference_normals(d, q) ==
                          MFEM_Approx(d == direction ? 1.0 : 0.0,
                                      tolerance, tolerance));
               }
            }
         }
      }
   }
}

TEST_CASE("Algoim detects singular interfaces in constant tensor directions",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   const real_t row[4] = {-0.125, 0.125, -0.125, 0.125};
   CutQuadratureRequest request;
   request.measures = CutMeasure::Interface;
   for (int dim = 2; dim <= 3; dim++)
   {
      for (int direction = 0; direction < dim; direction++)
      {
         CAPTURE(dim, direction);
         ElementLevelSet polynomial;
         polynomial.geometry = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
         polynomial.order = 3;
         polynomial.coefficients.SetSize(dim == 2 ? 16 : 64);
         int stride = 1;
         for (int d = 0; d < direction; d++) { stride *= 4; }
         for (int i = 0; i < polynomial.coefficients.Size(); i++)
         {
            // phi = (x_d - 0.5)^3, independent of every other coordinate.
            polynomial.coefficients(i) = row[(i / stride) % 4];
         }
         ReferenceCutQuadrature result;
         REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                               *workspace) ==
                 CutQuadratureStatus::DegenerateInterface);
         REQUIRE(result.classification == CutCellClass::Cut);
      }
   }
}

TEST_CASE("Algoim evaluates elevated multivariate Bernstein planes",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   const real_t tolerance = std::max(real_t(1e-11),
                                     128*std::numeric_limits<real_t>::epsilon());
   for (int dim = 2; dim <= 3; dim++)
   {
      const int degree = dim == 2 ? 12 : 4;
      CAPTURE(dim, degree);
      ElementLevelSet polynomial;
      polynomial.geometry = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
      polynomial.order = degree;
      const int width = degree + 1;
      polynomial.coefficients.SetSize(dim == 2 ? width*width :
                                      width*width*width);
      for (int i = 0; i < polynomial.coefficients.Size(); i++)
      {
         int index = i;
         real_t coefficient = -0.8;
         for (int d = 0; d < dim; d++)
         {
            coefficient += real_t(index % width)/degree;
            index /= width;
         }
         polynomial.coefficients(i) = coefficient;
      }
      CutQuadratureRequest request;
      // Two Gauss points suffice for these planar simplex measures in 2D/3D.
      request.order = 2;
      request.measures = CutMeasure::Volume | CutMeasure::Interface;
      request.compute_reference_normals = true;
      ReferenceCutQuadrature result;
      REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                            *workspace) ==
              CutQuadratureStatus::Success);
      // x+y(+z) < 0.8 is a reference simplex wholly inside the element.
      REQUIRE(WeightSum(result.volume) ==
              MFEM_Approx(dim == 2 ? 0.8*0.8/2 : 0.8*0.8*0.8/6,
                          tolerance, tolerance));
      REQUIRE(WeightSum(result.interface.rule) ==
              MFEM_Approx(dim == 2 ? std::sqrt(2.0)*0.8 :
                          std::sqrt(3.0)*0.8*0.8/2, tolerance, tolerance));
      for (int q = 0; q < result.interface.rule.GetNPoints(); q++)
      {
         for (int d = 0; d < dim; d++)
         {
            REQUIRE(result.interface.reference_normals(d, q) ==
                    MFEM_Approx(1/std::sqrt(real_t(dim)), tolerance, tolerance));
         }
      }
   }
}

TEST_CASE("Algoim preserves an elevated quadratic cut",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   ElementLevelSet polynomial;
   polynomial.geometry = Geometry::SQUARE;
   polynomial.order = 36;
   const int degree = polynomial.order;
   const int width = degree + 1;
   polynomial.coefficients.SetSize(width*width);
   for (int j = 0; j < width; j++)
   {
      for (int i = 0; i < width; i++)
      {
         // Elevated Bernstein coefficients of (x-0.2)(x-0.7).
         const real_t quadratic = real_t(i)*(i - 1)/(degree*(degree - 1));
         polynomial.coefficients(i + width*j) =
            real_t(0.14) - real_t(0.9)*i/degree + quadratic;
      }
   }
   CutQuadratureRequest request;
   // Both zero sets are straight lines, irrespective of the elevated degree.
   request.order = 2;
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;
   ReferenceCutQuadrature result;
   REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                         *workspace) ==
           CutQuadratureStatus::Success);
   const real_t tolerance = std::max(real_t(1e-11),
                                     128*std::numeric_limits<real_t>::epsilon());
   REQUIRE(WeightSum(result.volume) == MFEM_Approx(0.5, tolerance, tolerance));
   REQUIRE(WeightSum(result.interface.rule) ==
           MFEM_Approx(2.0, tolerance, tolerance));
   real_t left_measure = 0.0, right_measure = 0.0;
   for (int q = 0; q < result.interface.rule.GetNPoints(); q++)
   {
      const auto &ip = result.interface.rule[q];
      const bool left = ip.x < 0.45;
      REQUIRE(ip.x == MFEM_Approx(left ? 0.2 : 0.7, tolerance, tolerance));
      REQUIRE(result.interface.reference_normals(0, q) ==
              MFEM_Approx(left ? -1.0 : 1.0, tolerance, tolerance));
      REQUIRE(result.interface.reference_normals(1, q) ==
              MFEM_Approx(0.0, tolerance, tolerance));
      (left ? left_measure : right_measure) += ip.weight;
   }
   REQUIRE(left_measure == MFEM_Approx(1.0, tolerance, tolerance));
   REQUIRE(right_measure == MFEM_Approx(1.0, tolerance, tolerance));
}

TEST_CASE("Algoim validates polynomial degrees independently of quadrature order",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   const auto &caps = constructor.Capabilities();
   REQUIRE(caps.min_polynomial_degree == 0);
   REQUIRE(caps.max_polynomial_degree == 64);
   CutQuadratureRequest request;
   ElementLevelSetDescriptor descriptor =
   { Geometry::SQUARE, PolynomialBasis::BernsteinTensor, 64 };
   REQUIRE(caps.Supports(request, descriptor));
   REQUIRE(caps.Supports(request, descriptor, true));
   descriptor.order = 65;
   REQUIRE_FALSE(caps.Supports(request, descriptor));
   REQUIRE_FALSE(caps.Supports(request, descriptor, true));
   descriptor.order = -1;
   REQUIRE_FALSE(caps.Supports(request, descriptor));
   descriptor.order = 0;
   REQUIRE(caps.Supports(request, descriptor));
   request.order = 20;
   REQUIRE_FALSE(caps.Supports(request, descriptor));
   request.order = 4;

   ElementLevelSet polynomial = SquareLinear(-0.4, 0.6);
   ReferenceCutQuadrature result;
   for (const int degree : {65, std::numeric_limits<int>::max()})
   {
      CAPTURE(degree);
      // Deliberately retain the small coefficient array: reject metadata before
      // coefficient-count arithmetic, allocations, or polynomial evaluation.
      polynomial.order = degree;
      REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                            *workspace) ==
              CutQuadratureStatus::UnsupportedPolynomialDegree);
      REQUIRE(result.status == CutQuadratureStatus::UnsupportedPolynomialDegree);
      REQUIRE(result.classification == CutCellClass::Unclassified);
      REQUIRE(result.volume.GetNPoints() == 0);
      REQUIRE(result.interface.rule.GetNPoints() == 0);
   }
   polynomial.order = -1;
   REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                         *workspace) ==
           CutQuadratureStatus::InvalidLevelSet);
   polynomial.order = 0;
   polynomial.coefficients.SetSize(1);
   polynomial.coefficients(0) = -1.0;
   REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                         *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(WeightSum(result.volume) == MFEM_Approx(1.0));

   ElementLevelSetBatch batch;
   batch.descriptor =
   { Geometry::SQUARE, PolynomialBasis::BernsteinTensor, 65 };
   batch.coefficients.SetSize(66*66, 2);
   batch.coefficients = 1.0;
   batch.element_descriptors.SetSize(2);
   batch.element_descriptors[0] = batch.element_descriptors[1] = batch.descriptor;
   batch.extraction_status.SetSize(2);
   batch.extraction_status[0] = CutQuadratureStatus::Success;
   batch.extraction_status[1] = CutQuadratureStatus::UnsupportedSourceBasis;
   BatchedReferenceCutQuadrature packed;
   REQUIRE(constructor.GenerateReferenceBatch(batch, request, packed,
                                              *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(packed.status[0] == CutQuadratureStatus::UnsupportedPolynomialDegree);
   REQUIRE(packed.status[1] == CutQuadratureStatus::UnsupportedSourceBasis);
   REQUIRE(packed.classification[0] == CutCellClass::Unclassified);
   REQUIRE(packed.classification[1] == CutCellClass::Unclassified);
   REQUIRE(packed.volume.weights.Size() == 0);
   REQUIRE(packed.interface.weights.Size() == 0);
}

TEST_CASE("Algoim combines interior and boundary interfaces without duplication",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   for (int dim = 2; dim <= 3; dim++)
   {
      for (int direction = 0; direction < dim; direction++)
      {
         for (int side = 0; side < 2; side++)
         {
            CAPTURE(dim, direction, side);
            // side 1: (t-0.25)(t-1); side 0: t(t-0.75).
            // Both have negative volume 0.75 and two unit interface components.
            const real_t row[3] =
            {
               side == 0 ? real_t(0.0) : real_t(0.25), real_t(-0.375),
               side == 0 ? real_t(0.25) : real_t(0.0)
            };
            ElementLevelSet polynomial;
            polynomial.geometry = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
            polynomial.basis = PolynomialBasis::BernsteinTensor;
            polynomial.order = 2;
            polynomial.coefficients.SetSize(dim == 2 ? 9 : 27);
            int stride = 1;
            for (int d = 0; d < direction; d++) { stride *= 3; }
            for (int i = 0; i < polynomial.coefficients.Size(); i++)
            {
               polynomial.coefficients(i) = row[(i / stride) % 3];
            }

            CutQuadratureRequest request;
            request.measures = CutMeasure::Volume | CutMeasure::Interface;
            request.compute_reference_normals = true;
            ReferenceCutQuadrature result;
            REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                                  *workspace) ==
                    CutQuadratureStatus::Success);
            REQUIRE(result.classification == CutCellClass::Cut);
            REQUIRE(WeightSum(result.volume) == MFEM_Approx(0.75));
            REQUIRE(WeightSum(result.interface.rule) == MFEM_Approx(2.0));
            REQUIRE(result.interface.rule.GetOrder() == request.order);
            REQUIRE(result.interface.reference_normals.Height() == dim);
            REQUIRE(result.interface.reference_normals.Width() ==
                    result.interface.rule.GetNPoints());

            real_t boundary_measure = 0.0, interior_measure = 0.0;
            for (int i = 0; i < result.interface.rule.GetNPoints(); i++)
            {
               const IntegrationPoint &ip = result.interface.rule.IntPoint(i);
               const real_t coordinate = direction == 0 ? ip.x :
                                         (direction == 1 ? ip.y : ip.z);
               const bool boundary = coordinate == real_t(side);
               if (boundary) { boundary_measure += ip.weight; }
               else
               {
                  REQUIRE(coordinate == MFEM_Approx(side == 0 ? 0.75 : 0.25));
                  interior_measure += ip.weight;
               }
               const real_t normal = boundary ? (side == 0 ? -1.0 : 1.0) :
                                     (side == 0 ? 1.0 : -1.0);
               for (int d = 0; d < dim; d++)
               {
                  REQUIRE(result.interface.reference_normals(d, i) ==
                          MFEM_Approx(d == direction ? normal : real_t(0.0))
                          .margin(64*std::numeric_limits<real_t>::epsilon()));
               }
               REQUIRE(ip.index == i);
            }
            REQUIRE(boundary_measure == MFEM_Approx(1.0));
            REQUIRE(interior_measure == MFEM_Approx(1.0));

            // Phase selection changes volume, not the interface or its normals.
            request.region = CutRegion::Positive;
            ReferenceCutQuadrature positive;
            REQUIRE(constructor.GenerateReference(polynomial, request, positive,
                                                  *workspace) ==
                    CutQuadratureStatus::Success);
            REQUIRE(WeightSum(positive.volume) == MFEM_Approx(0.25));
            REQUIRE(WeightSum(positive.interface.rule) == MFEM_Approx(2.0));
            REQUIRE(positive.interface.rule.GetNPoints() ==
                    result.interface.rule.GetNPoints());
            for (int i = 0; i < positive.interface.rule.GetNPoints(); i++)
            {
               for (int d = 0; d < dim; d++)
               {
                  REQUIRE(positive.interface.reference_normals(d, i) ==
                          MFEM_Approx(result.interface.reference_normals(d, i)));
               }
            }

            // Pack phi and -phi to check point/normal alignment after merging.
            request.region = CutRegion::Negative;
            ElementLevelSetBatch batch;
            batch.descriptor =
            { polynomial.geometry, polynomial.basis, polynomial.order };
            batch.coefficients.SetSize(polynomial.coefficients.Size(), 2);
            batch.element_descriptors.SetSize(2);
            batch.extraction_status.SetSize(2);
            for (int e = 0; e < 2; e++)
            {
               batch.element_descriptors[e] = batch.descriptor;
               batch.extraction_status[e] = CutQuadratureStatus::Success;
               for (int i = 0; i < polynomial.coefficients.Size(); i++)
               {
                  batch.coefficients(i, e) =
                     (e == 0 ? 1.0 : -1.0)*polynomial.coefficients(i);
               }
            }
            BatchedReferenceCutQuadrature packed;
            ElementLevelSet negated = polynomial;
            negated.coefficients *= -1.0;
            ReferenceCutQuadrature reversed;
            REQUIRE(constructor.GenerateReference(negated, request, reversed,
                                                  *workspace) ==
                    CutQuadratureStatus::Success);
            REQUIRE(constructor.GenerateReferenceBatch(batch, request, packed,
                                                       *workspace) ==
                    CutQuadratureStatus::Success);
            for (int e = 0; e < 2; e++)
            {
               REQUIRE(packed.status[e] == CutQuadratureStatus::Success);
               real_t volume = 0.0, surface = 0.0;
               for (int q = packed.volume.offsets[e];
                    q < packed.volume.offsets[e + 1]; q++)
               {
                  volume += packed.volume.weights(q);
               }
               REQUIRE(volume == MFEM_Approx(e == 0 ? 0.75 : 0.25));
               const ReferenceInterfaceRule &expected = e == 0 ?
                                                        result.interface :
                                                        reversed.interface;
               const int begin = packed.interface.offsets[e];
               const int end = packed.interface.offsets[e + 1];
               REQUIRE(end - begin == expected.rule.GetNPoints());
               for (int q = begin; q < end; q++)
               {
                  surface += packed.interface.weights(q);
                  const IntegrationPoint &ip = expected.rule[q - begin];
                  REQUIRE(packed.interface.points(0, q) == MFEM_Approx(ip.x));
                  REQUIRE(packed.interface.points(1, q) == MFEM_Approx(ip.y));
                  REQUIRE(packed.interface.weights(q) == MFEM_Approx(ip.weight));
                  if (dim == 3)
                  {
                     REQUIRE(packed.interface.points(2, q) == MFEM_Approx(ip.z));
                  }
                  for (int d = 0; d < dim; d++)
                  {
                     REQUIRE(packed.interface.normals(d, q) ==
                             MFEM_Approx(expected.reference_normals(
                                            d, q - begin)));
                  }
                  const real_t coordinate =
                     packed.interface.points(direction, q);
                  const bool boundary = coordinate == real_t(side);
                  const real_t normal = boundary ? (side == 0 ? -1.0 : 1.0) :
                                        (side == 0 ? 1.0 : -1.0);
                  REQUIRE(packed.interface.normals(direction, q) ==
                          MFEM_Approx((e == 0 ? 1.0 : -1.0)*normal));
               }
               REQUIRE(surface == MFEM_Approx(2.0));
            }

            // Boundary weights must also be included without requesting normals.
            request.measures = CutMeasure::Interface;
            request.compute_reference_normals = false;
            REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                                  *workspace) ==
                    CutQuadratureStatus::Success);
            REQUIRE(result.volume.GetNPoints() == 0);
            REQUIRE(WeightSum(result.interface.rule) == MFEM_Approx(2.0));
            REQUIRE(result.interface.reference_normals.Width() == 0);
         }
      }
   }
}

TEST_CASE("Algoim deflates both boundary sides and preserves singular interfaces",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   ElementLevelSet polynomial;
   polynomial.geometry = Geometry::SQUARE;
   polynomial.basis = PolynomialBasis::BernsteinTensor;
   polynomial.order = 3;
   polynomial.coefficients.SetSize(16);
   CutQuadratureRequest request;
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;
   ReferenceCutQuadrature result;

   SECTION("two zero faces and one interior component")
   {
      // phi = x(1-x)(x-0.25): three distinct unit-length components.
      const real_t row[4] = {0.0, real_t(-1.0/12.0), 0.25, 0.0};
      for (int j = 0; j < 4; j++)
      {
         for (int i = 0; i < 4; i++)
         {
            polynomial.coefficients(i + 4*j) = row[i];
         }
      }
      REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                            *workspace) ==
              CutQuadratureStatus::Success);
      REQUIRE(WeightSum(result.volume) == MFEM_Approx(0.25));
      REQUIRE(WeightSum(result.interface.rule) == MFEM_Approx(3.0));
   }

   SECTION("conservative Cut classification with only a boundary interface")
   {
      // phi = (1-x)*(24*(x-0.5)^2+3) is positive inside, despite mixed bounds.
      const real_t row[4] = {9.0, -2.0, 3.0, 0.0};
      for (int j = 0; j < 4; j++)
      {
         for (int i = 0; i < 4; i++)
         {
            polynomial.coefficients(i + 4*j) = row[i];
         }
      }
      REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                            *workspace) ==
              CutQuadratureStatus::Success);
      REQUIRE(result.classification == CutCellClass::Cut);
      REQUIRE(WeightSum(result.volume) == MFEM_Approx(0.0));
      REQUIRE(WeightSum(result.interface.rule) == MFEM_Approx(1.0));
      REQUIRE(result.interface.reference_normals.Width() ==
              result.interface.rule.GetNPoints());
      for (int i = 0; i < result.interface.rule.GetNPoints(); i++)
      {
         REQUIRE(result.interface.rule[i].x == MFEM_Approx(1.0));
         REQUIRE(result.interface.reference_normals(0, i) == MFEM_Approx(-1.0));
      }
   }

   SECTION("repeated boundary factor remains degenerate")
   {
      // phi = (x-0.25)(x-1)^2: the original gradient vanishes at x=1.
      const real_t row[4] = {-0.25, 0.25, 0.0, 0.0};
      for (int j = 0; j < 4; j++)
      {
         for (int i = 0; i < 4; i++)
         {
            polynomial.coefficients(i + 4*j) = row[i];
         }
      }
      REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                            *workspace) ==
              CutQuadratureStatus::DegenerateInterface);
      request.measures = CutMeasure::Volume;
      REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                            *workspace) ==
              CutQuadratureStatus::Success);
      REQUIRE(WeightSum(result.volume) == MFEM_Approx(0.25));
   }
}

TEST_CASE("Algoim removes boundary factors for volume-only requests",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   const real_t rows[3][4] =
   {
      {0.0, real_t(-1.0/12.0), 0.25, 0.0},  // t(1-t)(t-0.25)
      {0.0, 0.0, real_t(-1.0/12.0), 0.75},  // t^2(t-0.25)
      {-0.25, 0.25, 0.0, 0.0}       // (1-t)^2(t-0.25)
   };
   for (int dim = 2; dim <= 3; dim++)
   {
      for (int direction = 0; direction < dim; direction++)
      {
         for (int factor = 0; factor < 3; factor++)
         {
            CAPTURE(dim, direction, factor);
            ElementLevelSet polynomial;
            polynomial.geometry = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
            polynomial.order = 3;
            polynomial.coefficients.SetSize(dim == 2 ? 16 : 64);
            int stride = 1;
            for (int d = 0; d < direction; d++) { stride *= 4; }
            for (int i = 0; i < polynomial.coefficients.Size(); i++)
            {
               polynomial.coefficients(i) = rows[factor][(i / stride) % 4];
            }
            // Boundary factors are positive inside the cell, including repeated
            // factors whose boundary gradients vanish. Both open volume phases
            // agree with those of t-0.25, without requesting an interface.
            CutQuadratureRequest request;
            request.order = 2;
            request.measures = CutMeasure::Volume;
            ReferenceCutQuadrature result;
            for (const auto region : {CutRegion::Negative, CutRegion::Positive})
            {
               CAPTURE(region);
               request.region = region;
               REQUIRE(constructor.GenerateReference(polynomial, request, result,
                                                     *workspace) ==
                       CutQuadratureStatus::Success);
               REQUIRE(result.classification == CutCellClass::Cut);
               REQUIRE(WeightSum(result.volume) ==
                       MFEM_Approx(region == CutRegion::Negative ? 0.25 : 0.75));
               REQUIRE(result.interface.rule.GetNPoints() == 0);
            }
         }
      }
   }
}

TEST_CASE("Algoim separates interface and volume degeneracy",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   ElementLevelSet cubic;
   cubic.geometry = Geometry::SQUARE;
   cubic.order = 3;
   cubic.basis = PolynomialBasis::BernsteinTensor;
   cubic.coefficients.SetSize(16);
   const real_t row[4] = {-0.125, 0.125, -0.125, 0.125};
   for (int j = 0; j < 4; j++)
   {
      for (int i = 0; i < 4; i++) { cubic.coefficients[i + 4*j] = row[i]; }
   }

   CutQuadratureRequest request;
   ReferenceCutQuadrature result;
   REQUIRE(constructor.GenerateReference(cubic, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.classification == CutCellClass::Cut);
   REQUIRE(WeightSum(result.volume) == MFEM_Approx(0.5));

   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;
   REQUIRE(constructor.GenerateReference(cubic, request, result, *workspace) ==
           CutQuadratureStatus::DegenerateInterface);
   REQUIRE(result.classification == CutCellClass::Cut);
}

TEST_CASE("Algoim circle and sphere rules", "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   CutQuadratureRequest request;
   request.order = 8;
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;
   ConstantCoefficient one(1.0);

   SECTION("circle")
   {
      Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
      FunctionCoefficient phi([](const Vector &x)
      {
         return (x(0) - 0.5)*(x(0) - 0.5) +
                (x(1) - 0.5)*(x(1) - 0.5) - 0.0625;
      });
      CoefficientLevelSetExtractor extractor(phi, 2);
      ElementTransformation &Tr = *mesh.GetElementTransformation(0);
      ElementLevelSet local;
      REQUIRE(extractor.GetElementLevelSet(0, Tr, local) ==
              CutQuadratureStatus::Success);
      ReferenceCutQuadrature result;
      REQUIRE(constructor.GenerateReference(local, request, result, *workspace) ==
              CutQuadratureStatus::Success);
      REQUIRE(CutQuadratureIntegrator::IntegrateVolume(one, Tr, result) ==
              MFEM_Approx(3.14159265358979323846/16.0).epsilon(2e-4));
      REQUIRE(CutQuadratureIntegrator::IntegrateInterface(one, Tr, result) ==
              MFEM_Approx(3.14159265358979323846/2.0).epsilon(2e-4));
   }

   SECTION("sphere")
   {
      Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
      FunctionCoefficient phi([](const Vector &x)
      {
         return (x(0) - 0.5)*(x(0) - 0.5) +
                (x(1) - 0.5)*(x(1) - 0.5) +
                (x(2) - 0.5)*(x(2) - 0.5) - 0.0625;
      });
      CoefficientLevelSetExtractor extractor(phi, 2);
      ElementTransformation &Tr = *mesh.GetElementTransformation(0);
      ElementLevelSet local;
      REQUIRE(extractor.GetElementLevelSet(0, Tr, local) ==
              CutQuadratureStatus::Success);
      ReferenceCutQuadrature result;
      REQUIRE(constructor.GenerateReference(local, request, result, *workspace) ==
              CutQuadratureStatus::Success);
      REQUIRE(CutQuadratureIntegrator::IntegrateVolume(one, Tr, result) ==
              MFEM_Approx(3.14159265358979323846/48.0).epsilon(5e-4));
      REQUIRE(CutQuadratureIntegrator::IntegrateInterface(one, Tr, result) ==
              MFEM_Approx(3.14159265358979323846/4.0).epsilon(5e-4));
   }
}

TEST_CASE("Cut rules retain reference data under non-affine deformation",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   CutQuadratureRequest request;
   request.order = 6;
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;
   ConstantCoefficient one(1.0);

   SECTION("quadrilateral")
   {
      Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
      FunctionCoefficient phi([](const Vector &x) { return x(0) - 0.5; });
      CoefficientLevelSetExtractor extractor(phi, 1, 2);
      ElementTransformation &Tr = *mesh.GetElementTransformation(0);
      ElementLevelSet local;
      REQUIRE(extractor.GetElementLevelSet(0, Tr, local) ==
              CutQuadratureStatus::Success);
      ReferenceCutQuadrature result;
      REQUIRE(constructor.GenerateReference(local, request, result,
                                            *workspace) ==
              CutQuadratureStatus::Success);

      VectorFunctionCoefficient deform(2, [](const Vector &x, Vector &y)
      {
         y.SetSize(2);
         y(0) = x(0);
         y(1) = x(1)*(1.0 + 0.4*x(0));
      });
      mesh.Transform(deform);
      ElementTransformation &deformed = *mesh.GetElementTransformation(0);
      REQUIRE(CutQuadratureIntegrator::IntegrateVolume(one, deformed, result) ==
              MFEM_Approx(0.55));
      REQUIRE(CutQuadratureIntegrator::IntegrateInterface(one, deformed,
                                                          result) ==
              MFEM_Approx(1.2));
   }

   SECTION("hexahedron")
   {
      Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
      FunctionCoefficient phi([](const Vector &x) { return x(0) - 0.5; });
      CoefficientLevelSetExtractor extractor(phi, 1, 2);
      ElementTransformation &Tr = *mesh.GetElementTransformation(0);
      ElementLevelSet local;
      REQUIRE(extractor.GetElementLevelSet(0, Tr, local) ==
              CutQuadratureStatus::Success);
      ReferenceCutQuadrature result;
      REQUIRE(constructor.GenerateReference(local, request, result,
                                            *workspace) ==
              CutQuadratureStatus::Success);

      VectorFunctionCoefficient deform(3, [](const Vector &x, Vector &y)
      {
         y.SetSize(3);
         y(0) = x(0);
         y(1) = x(1);
         y(2) = x(2)*(1.0 + 0.4*x(0));
      });
      mesh.Transform(deform);
      ElementTransformation &deformed = *mesh.GetElementTransformation(0);
      REQUIRE(CutQuadratureIntegrator::IntegrateVolume(one, deformed, result) ==
              MFEM_Approx(0.55));
      REQUIRE(CutQuadratureIntegrator::IntegrateInterface(one, deformed,
                                                          result) ==
              MFEM_Approx(1.2));
   }
}

TEST_CASE("Algoim packed batch validation and equivalence",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   CutQuadratureRequest request;
   ElementLevelSet cut = SquareLinear(-0.5, 0.5);
   ElementLevelSet full = SquareLinear(-1.0, -1.0);

   ElementLevelSetBatch batch;
   batch.descriptor =
   { Geometry::SQUARE, PolynomialBasis::BernsteinTensor, 1 };
   batch.coefficients.SetSize(4, 4);
   batch.element_descriptors.SetSize(4);
   batch.extraction_status.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      batch.element_descriptors[i] = batch.descriptor;
      batch.extraction_status[i] = CutQuadratureStatus::Success;
   }
   batch.coefficients.SetCol(0, cut.coefficients);
   batch.coefficients.SetCol(1, full.coefficients);
   batch.extraction_status[2] = CutQuadratureStatus::UnsupportedSourceBasis;
   batch.extraction_status[3] = CutQuadratureStatus::InvalidLevelSet;
   batch.element_descriptors[2] =
   { Geometry::TRIANGLE, PolynomialBasis::BernsteinSimplex, 99 };
   for (int r = 0; r < 4; r++)
   {
      batch.coefficients(r, 2) = std::numeric_limits<real_t>::quiet_NaN();
   }

   BatchedReferenceCutQuadrature result;
   REQUIRE(constructor.GenerateReferenceBatch(batch, request, result,
                                              *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.status.Size() == 4);
   REQUIRE(result.status[0] == CutQuadratureStatus::Success);
   REQUIRE(result.status[1] == CutQuadratureStatus::Success);
   REQUIRE(result.status[2] == CutQuadratureStatus::UnsupportedSourceBasis);
   REQUIRE(result.classification[2] == CutCellClass::Unclassified);
   REQUIRE(result.volume.offsets.Size() == 5);
   REQUIRE(result.volume.offsets[4] == result.volume.weights.Size());
   real_t packed_sum = 0.0;
   for (int i = result.volume.offsets[0]; i < result.volume.offsets[1]; i++)
   {
      packed_sum += result.volume.weights[i];
   }
   ReferenceCutQuadrature scalar;
   REQUIRE(constructor.GenerateReference(cut, request, scalar, *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(packed_sum == MFEM_Approx(WeightSum(scalar.volume)));

   CutQuadratureRequest high_order = request;
   high_order.order = 20;
   REQUIRE(constructor.GenerateReferenceBatch(batch, high_order, result,
                                              *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.status[0] == CutQuadratureStatus::UnsupportedOrder);
   REQUIRE(result.classification[0] == CutCellClass::Unclassified);

   batch.extraction_status[2] = CutQuadratureStatus::ExecutionFailure;
   REQUIRE(constructor.GenerateReferenceBatch(batch, request, result,
                                              *workspace) ==
           CutQuadratureStatus::InvalidBatch);
   REQUIRE(result.status.Size() == 0);
   batch.extraction_status[2] = CutQuadratureStatus::UnsupportedSourceBasis;
   batch.element_descriptors[0].order = 2;
   REQUIRE(constructor.GenerateReferenceBatch(batch, request, result,
                                              *workspace) ==
           CutQuadratureStatus::HeterogeneousBatch);
   REQUIRE(result.status.Size() == 0);
   batch.element_descriptors[0] = batch.descriptor;
   batch.coefficients.SetSize(3, 4);
   REQUIRE(constructor.GenerateReferenceBatch(batch, request, result,
                                              *workspace) ==
           CutQuadratureStatus::InvalidBatch);

   batch.coefficients.SetSize(4, 4);

   batch.descriptor.basis = PolynomialBasis::BernsteinSimplex;
   batch.element_descriptors[0].basis = PolynomialBasis::BernsteinSimplex;
   batch.extraction_status[1] = CutQuadratureStatus::UnsupportedSourceBasis;
   REQUIRE(constructor.GenerateReferenceBatch(batch, request, result,
                                              *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.status[0] ==
           CutQuadratureStatus::UnsupportedPolynomialBasis);
   REQUIRE(result.classification[0] == CutCellClass::Unclassified);

   request.order = -1;
   request.execution = CutExecutionMode::Device;
   batch.extraction_status.SetSize(0);
   REQUIRE(constructor.GenerateReferenceBatch(batch, request, result,
                                              *workspace) ==
           CutQuadratureStatus::InvalidRequest);
   request.order = 4;
   REQUIRE(constructor.GenerateReferenceBatch(batch, request, result,
                                              *workspace) ==
           CutQuadratureStatus::UnsupportedExecutionMode);
}

TEST_CASE("Cut level-set extractors and physical mapping",
          "[CutQuadrature][Algoim]")
{
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL,
                                     true, 2.0, 3.0);
   H1_FECollection collection(2, 2);
   FiniteElementSpace space(&mesh, &collection);
   GridFunction field(&space);
   FunctionCoefficient level_set([](const Vector &x) { return x(0) - 1.0; });
   field.ProjectCoefficient(level_set);
   ElementTransformation &Tr = *mesh.GetElementTransformation(0);

   GridFunctionLevelSetExtractor grid_extractor(field, 3);
   CoefficientLevelSetExtractor coefficient_extractor(level_set, 2, 3);
   ElementLevelSet grid_local, coefficient_local;
   REQUIRE(grid_extractor.GetElementLevelSet(0, Tr, grid_local) ==
           CutQuadratureStatus::Success);
   REQUIRE(coefficient_extractor.GetElementLevelSet(0, Tr, coefficient_local) ==
           CutQuadratureStatus::Success);

   FiniteElementSpace vector_space(&mesh, &collection, 2);
   GridFunction vector_field(&vector_space);
   GridFunctionLevelSetExtractor unsupported_extractor(vector_field);
   ElementLevelSet unused;
   REQUIRE(unsupported_extractor.GetElementLevelSet(0, Tr, unused) ==
           CutQuadratureStatus::UnsupportedSourceBasis);
   FunctionCoefficient not_finite([](const Vector &)
   {
      return std::numeric_limits<real_t>::quiet_NaN();
   });
   CoefficientLevelSetExtractor invalid_extractor(not_finite, 1);
   REQUIRE(invalid_extractor.GetElementLevelSet(0, Tr, unused) ==
           CutQuadratureStatus::InvalidLevelSet);
   REQUIRE(grid_local.coefficients.Size() == 9);
   for (int j = 0; j < 3; j++)
   {
      REQUIRE(grid_local.coefficients[3*j] == MFEM_Approx(-1.0));
      REQUIRE(grid_local.coefficients[3*j + 1] == MFEM_Approx(0.0).margin(1e-12));
      REQUIRE(grid_local.coefficients[3*j + 2] == MFEM_Approx(1.0));
   }

   AlgoimCutQuadratureConstructor constructor;
   auto workspace = constructor.CreateWorkspace();
   CutQuadratureRequest request;
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;
   ReferenceCutQuadrature result;
   REQUIRE(constructor.GenerateReference(grid_local, request, result,
                                         *workspace) ==
           CutQuadratureStatus::Success);

   ConstantCoefficient one(1.0);
   REQUIRE(CutQuadratureIntegrator::IntegrateVolume(one, Tr, result) ==
           MFEM_Approx(3.0));
   IntegrationRule mapped_volume;
   MapReferenceVolumeRule(Tr, result.volume, mapped_volume);
   REQUIRE(WeightSum(mapped_volume) == MFEM_Approx(3.0));
   DenseMatrix physical_normals;
   REQUIRE(CutQuadratureIntegrator::IntegrateInterface(
              one, Tr, result, &physical_normals) ==
           MFEM_Approx(3.0));
   REQUIRE(physical_normals.Height() == 2);
   REQUIRE(physical_normals.Width() == result.interface.rule.GetNPoints());
   for (int i = 0; i < physical_normals.Width(); i++)
   {
      REQUIRE(physical_normals(0, i) == MFEM_Approx(1.0));
      REQUIRE(physical_normals(1, i) == MFEM_Approx(0.0).margin(1e-12));
   }

   RetainedCutQuadrature retained;
   retained.extractor_id = grid_extractor.Id();
   retained.element = 0;
   retained.revision = grid_extractor.Revision();
   retained.request = request;
   retained.result = result;
   REQUIRE(retained.IsValid(grid_extractor, 0, request));
   field = 2.0; // Without a bump this deliberately remains a stale hit.
   REQUIRE(retained.IsValid(grid_extractor, 0, request));
   grid_extractor.IncrementRevision();
   REQUIRE_FALSE(retained.IsValid(grid_extractor, 0, request));

   AlgoimIntegrationRules legacy(4, level_set, 1);
   IntegrationRule legacy_volume, legacy_surface;
   legacy.GetVolumeIntegrationRule(Tr, legacy_volume);
   legacy.GetSurfaceIntegrationRule(Tr, legacy_surface);
   Vector legacy_surface_metric;
   legacy.GetSurfaceWeights(Tr, legacy_surface,
                            legacy_surface_metric);
   REQUIRE(legacy_volume.GetNPoints() > 0);
   REQUIRE(legacy_surface.GetNPoints() == legacy_surface_metric.Size());
}

TEST_CASE("Legacy Algoim validates orders before rule generation",
          "[CutQuadrature][Algoim]")
{
   class LegacyRuleState : public AlgoimIntegrationRules
   {
   public:
      using AlgoimIntegrationRules::AlgoimIntegrationRules;
      int TargetOrder() const { return Order; }
      int ProjectionOrder() const { return lsOrder; }
   };

   AlgoimCutQuadratureConstructor backend;
   const auto &caps = backend.Capabilities();
   const int minimum_order = std::max(1, caps.min_order);
   const int minimum_degree = std::max(1, caps.min_polynomial_degree);
   FunctionCoefficient phi([](const Vector &x) { return x(0) - 0.4; });
   LegacyRuleState maximum(caps.max_order, phi, caps.max_polynomial_degree);
   REQUIRE(maximum.TargetOrder() == caps.max_order);
   REQUIRE(maximum.ProjectionOrder() == caps.max_polynomial_degree);

   LegacyRuleState legacy(minimum_order, phi, minimum_degree);
   CutIntegrationRules &base = legacy;
   base.SetOrder(caps.max_order);
   base.SetLevelSetProjectionOrder(caps.max_polynomial_degree);
   REQUIRE(legacy.TargetOrder() == caps.max_order);
   REQUIRE(legacy.ProjectionOrder() == caps.max_polynomial_degree);
   // Acceptance at the maximum degree does not require expensive high-degree
   // interpolation. Generate a linear cut at the maximum target order instead.
   base.SetLevelSetProjectionOrder(1);
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   ElementTransformation &Tr = *mesh.GetElementTransformation(0);
   IntegrationRule volume, surface;
   legacy.GetVolumeIntegrationRule(Tr, volume);
   legacy.GetSurfaceIntegrationRule(Tr, surface);
   REQUIRE(volume.GetOrder() == caps.max_order);
   REQUIRE(surface.GetOrder() == caps.max_order);
   REQUIRE(WeightSum(volume) == MFEM_Approx(0.6));
   REQUIRE(WeightSum(surface) == MFEM_Approx(1.0));

#ifdef MFEM_USE_EXCEPTIONS
   struct ErrorActionGuard
   {
      ErrorAction previous;
      ErrorActionGuard() : previous(get_error_action())
      { set_error_action(MFEM_ERROR_THROW); }
      ~ErrorActionGuard() { set_error_action(previous); }
   } error_action_guard;

   for (const int order :
        {
           0, -1, caps.max_order + 1,
           std::numeric_limits<int>::max()
        })
   {
      CAPTURE(order);
      REQUIRE_THROWS_AS(LegacyRuleState(order, phi, 1), ErrorException);
      REQUIRE_THROWS_AS(base.SetOrder(order), ErrorException);
      REQUIRE(legacy.TargetOrder() == caps.max_order);
   }
   for (const int degree :
        {
           0, -1, caps.max_polynomial_degree + 1,
           std::numeric_limits<int>::max()
        })
   {
      CAPTURE(degree);
      REQUIRE_THROWS_AS(LegacyRuleState(minimum_order, phi, degree),
                        ErrorException);
      REQUIRE_THROWS_AS(base.SetLevelSetProjectionOrder(degree), ErrorException);
      REQUIRE(legacy.ProjectionOrder() == 1);
   }
   // The old extractor and configuration remain usable after rejected setters.
   base.SetLevelSetCoefficient(phi);
   legacy.GetVolumeIntegrationRule(Tr, volume);
   REQUIRE(volume.GetOrder() == caps.max_order);
   REQUIRE(WeightSum(volume) == MFEM_Approx(0.6));
#endif
}

TEST_CASE("Algoim shared constructor uses per-thread workspaces",
          "[CutQuadrature][Algoim]")
{
   const AlgoimCutQuadratureConstructor constructor;
   const ElementLevelSet cut = SquareLinear(-0.5, 0.5);
   const CutQuadratureRequest request;
   std::vector<CutQuadratureStatus> statuses(4);
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel for
#endif
   for (int i = 0; i < 4; i++)
   {
      auto workspace = constructor.CreateWorkspace();
      ReferenceCutQuadrature result;
      statuses[i] = constructor.GenerateReference(cut, request, result,
                                                  *workspace);
   }
   for (auto status : statuses)
   {
      REQUIRE(status == CutQuadratureStatus::Success);
   }
}

#endif // MFEM_USE_ALGOIM
