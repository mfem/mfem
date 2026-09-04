// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <algorithm>
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

class TestProvider : public ElementLevelSetProvider
{
public:
   explicit TestProvider(LevelSetRevision revision = 0) : revision_(revision) { }
   CutQuadratureStatus GetElementLevelSet(
      int, ElementTransformation &, ElementLevelSet &) const override
   { return CutQuadratureStatus::InvalidLevelSet; }
   LevelSetRevision Revision() const override { return revision_; }
   void Bump() { revision_++; }
private:
   LevelSetRevision revision_;
};

class MockWorkspace : public CutQuadratureWorkspace { };

class MockCutGenerator : public CutQuadratureGenerator
{
public:
   MockCutGenerator(bool nonnegative, bool fail_batch)
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

static_assert(!std::is_copy_constructible<TestProvider>::value, "provider copy");
static_assert(!std::is_move_constructible<TestProvider>::value, "provider move");

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

   TestProvider provider(7), other(7);
   RetainedCutQuadrature retained;
   retained.provider_id = provider.Id();
   retained.element = 3;
   retained.revision = provider.Revision();
   retained.request = request;
   REQUIRE(retained.IsValid(provider, 3, request));
   REQUIRE_FALSE(retained.IsValid(other, 3, request));
   REQUIRE_FALSE(retained.IsValid(provider, 4, request));
   provider.Bump();
   REQUIRE_FALSE(retained.IsValid(provider, 3, request));

   RetainedBatchedCutQuadrature retained_batch;
   retained_batch.provider_id = other.Id();
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
   TestProvider missed_bump(4);
   retained.provider_id = missed_bump.Id();
   retained.element = 0;
   retained.revision = 4;
   REQUIRE(retained.IsValid(missed_bump, 0, request));

   std::vector<std::uint64_t> ids(64);
#ifdef MFEM_USE_OPENMP
#pragma omp parallel for
#endif
   for (int i = 0; i < 64; i++)
   {
      TestProvider p;
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
   MockCutGenerator signed_generator(false, false);
   auto workspace = signed_generator.CreateWorkspace();
   ReferenceCutQuadrature result;
   REQUIRE(signed_generator.GenerateReference(level_set, request, result,
                                               *workspace) == result.status);
   REQUIRE(result.volume[0].weight < 0.0);
   request.weight_policy = QuadratureWeightPolicy::Nonnegative;
   REQUIRE(signed_generator.GenerateReference(level_set, request, result,
                                               *workspace) ==
           CutQuadratureStatus::UnsupportedWeightPolicy);
   REQUIRE(result.classification == CutCellClass::Unclassified);

   MockCutGenerator nonnegative_generator(true, false);
   workspace = nonnegative_generator.CreateWorkspace();
   REQUIRE(nonnegative_generator.GenerateReference(level_set, request, result,
                                                    *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.volume[0].weight >= 0.0);

   MockCutGenerator failing_generator(true, true);
   ElementLevelSetBatch batch;
   batch.descriptor =
   { Geometry::SQUARE, PolynomialBasis::BernsteinTensor, 1 };
   batch.coefficients.SetSize(4, 1);
   batch.coefficients.SetCol(0, level_set.coefficients);
   batch.element_descriptors.Append(batch.descriptor);
   batch.extraction_status.Append(CutQuadratureStatus::Success);
   BatchedReferenceCutQuadrature batch_result;
   workspace = failing_generator.CreateWorkspace();
   REQUIRE(failing_generator.GenerateReferenceBatch(batch, request,
                                                     batch_result,
                                                     *workspace) ==
           CutQuadratureStatus::ExecutionFailure);
   REQUIRE(batch_result.status.Size() == 0);
}

#ifdef MFEM_USE_ALGOIM

TEST_CASE("Algoim cut quadrature scalar contracts", "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureGenerator generator;
   auto workspace = generator.CreateWorkspace();
   ReferenceCutQuadrature result;
   CutQuadratureRequest request;

   REQUIRE(generator.Capabilities().min_order == 0);
   REQUIRE(generator.Capabilities().max_order == 19);
   REQUIRE_FALSE(generator.Capabilities().device_batch);

   ElementLevelSet cut = SquareLinear(-0.5, 0.5);
   REQUIRE(generator.GenerateReference(cut, request, result, *workspace) ==
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
   REQUIRE(generator.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(WeightSum(result.volume) == MFEM_Approx(0.5));

   ElementLevelSet full = SquareLinear(-1.0, -1.0);
   request.region = CutRegion::Negative;
   REQUIRE(generator.GenerateReference(full, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.classification == CutCellClass::Full);
   REQUIRE(WeightSum(result.volume) == MFEM_Approx(1.0));

   ElementLevelSet empty = SquareLinear(1.0, 1.0);
   REQUIRE(generator.GenerateReference(empty, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.classification == CutCellClass::Empty);
   REQUIRE(result.volume.GetNPoints() == 0);

   ElementLevelSet zero = SquareLinear(0.0, 0.0);
   REQUIRE(generator.GenerateReference(zero, request, result, *workspace) ==
           CutQuadratureStatus::DegenerateVolume);
   REQUIRE(result.classification == CutCellClass::Degenerate);
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   REQUIRE(generator.GenerateReference(zero, request, result, *workspace) ==
           CutQuadratureStatus::DegenerateVolume);
   REQUIRE(result.classification == CutCellClass::Degenerate);

   request = CutQuadratureRequest();
   request.order = -1;
   REQUIRE(generator.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::InvalidRequest);
   REQUIRE(result.classification == CutCellClass::Unclassified);
   request.order = 20;
   REQUIRE(generator.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::UnsupportedOrder);
   REQUIRE(result.classification == CutCellClass::Unclassified);
   request.order = 4;
   request.execution = CutExecutionMode::Device;
   REQUIRE(generator.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::UnsupportedExecutionMode);
   REQUIRE(result.classification == CutCellClass::Unclassified);
   request.execution = CutExecutionMode::Host;
   request.measures = static_cast<CutMeasure>(0u);
   REQUIRE(generator.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::InvalidRequest);
   request.measures = static_cast<CutMeasure>(8u);
   REQUIRE(generator.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::InvalidRequest);

   request = CutQuadratureRequest();
   cut.basis = PolynomialBasis::BernsteinSimplex;
   REQUIRE(generator.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::UnsupportedPolynomialBasis);
   REQUIRE(result.classification == CutCellClass::Unclassified);
   cut = SquareLinear(-0.5, 0.5);
   cut.geometry = Geometry::TRIANGLE;
   REQUIRE(generator.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::UnsupportedGeometry);
   REQUIRE(result.classification == CutCellClass::Unclassified);
   cut = SquareLinear(-0.5, 0.5);
   cut.coefficients.SetSize(3);
   REQUIRE(generator.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::InvalidLevelSet);
   REQUIRE(result.classification == CutCellClass::Unclassified);
}

TEST_CASE("Algoim interface and normal contracts", "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureGenerator generator;
   auto workspace = generator.CreateWorkspace();
   ReferenceCutQuadrature result;
   CutQuadratureRequest request;
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;

   ElementLevelSet boundary = SquareLinear(0.0, 1.0);
   REQUIRE(generator.GenerateReference(boundary, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.classification == CutCellClass::Empty);
   REQUIRE(result.interface.rule.GetNPoints() > 0);
   REQUIRE(WeightSum(result.interface.rule) == MFEM_Approx(1.0));
   for (int i = 0; i < result.interface.rule.GetNPoints(); i++)
   {
      REQUIRE(result.interface.reference_normals(0, i) > 0.0);
   }

   ElementLevelSet cut = SquareLinear(-0.5, 0.5);
   REQUIRE(generator.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   const DenseMatrix negative_normals(result.interface.reference_normals);
   request.region = CutRegion::Positive;
   REQUIRE(generator.GenerateReference(cut, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   for (int i = 0; i < result.interface.rule.GetNPoints(); i++)
   {
      REQUIRE(result.interface.reference_normals(0, i) ==
              MFEM_Approx(negative_normals(0, i)));
   }
}

TEST_CASE("Algoim separates interface and volume degeneracy",
          "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureGenerator generator;
   auto workspace = generator.CreateWorkspace();
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
   REQUIRE(generator.GenerateReference(cubic, request, result, *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.classification == CutCellClass::Cut);
   REQUIRE(WeightSum(result.volume) == MFEM_Approx(0.5));

   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;
   REQUIRE(generator.GenerateReference(cubic, request, result, *workspace) ==
           CutQuadratureStatus::DegenerateInterface);
   REQUIRE(result.classification == CutCellClass::Cut);
}

TEST_CASE("Algoim circle and sphere rules", "[CutQuadrature][Algoim]")
{
   AlgoimCutQuadratureGenerator generator;
   auto workspace = generator.CreateWorkspace();
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
      CoefficientLevelSetProvider provider(phi, 2);
      ElementTransformation &Tr = *mesh.GetElementTransformation(0);
      ElementLevelSet local;
      REQUIRE(provider.GetElementLevelSet(0, Tr, local) ==
              CutQuadratureStatus::Success);
      ReferenceCutQuadrature result;
      REQUIRE(generator.GenerateReference(local, request, result, *workspace) ==
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
      CoefficientLevelSetProvider provider(phi, 2);
      ElementTransformation &Tr = *mesh.GetElementTransformation(0);
      ElementLevelSet local;
      REQUIRE(provider.GetElementLevelSet(0, Tr, local) ==
              CutQuadratureStatus::Success);
      ReferenceCutQuadrature result;
      REQUIRE(generator.GenerateReference(local, request, result, *workspace) ==
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
   AlgoimCutQuadratureGenerator generator;
   auto workspace = generator.CreateWorkspace();
   CutQuadratureRequest request;
   request.order = 6;
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;
   ConstantCoefficient one(1.0);

   SECTION("quadrilateral")
   {
      Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
      FunctionCoefficient phi([](const Vector &x) { return x(0) - 0.5; });
      CoefficientLevelSetProvider provider(phi, 1, 2);
      ElementTransformation &Tr = *mesh.GetElementTransformation(0);
      ElementLevelSet local;
      REQUIRE(provider.GetElementLevelSet(0, Tr, local) ==
              CutQuadratureStatus::Success);
      ReferenceCutQuadrature result;
      REQUIRE(generator.GenerateReference(local, request, result,
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
      CoefficientLevelSetProvider provider(phi, 1, 2);
      ElementTransformation &Tr = *mesh.GetElementTransformation(0);
      ElementLevelSet local;
      REQUIRE(provider.GetElementLevelSet(0, Tr, local) ==
              CutQuadratureStatus::Success);
      ReferenceCutQuadrature result;
      REQUIRE(generator.GenerateReference(local, request, result,
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
   AlgoimCutQuadratureGenerator generator;
   auto workspace = generator.CreateWorkspace();
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
   REQUIRE(generator.GenerateReferenceBatch(batch, request, result,
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
   REQUIRE(generator.GenerateReference(cut, request, scalar, *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(packed_sum == MFEM_Approx(WeightSum(scalar.volume)));

   CutQuadratureRequest high_order = request;
   high_order.order = 20;
   REQUIRE(generator.GenerateReferenceBatch(batch, high_order, result,
                                             *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.status[0] == CutQuadratureStatus::UnsupportedOrder);
   REQUIRE(result.classification[0] == CutCellClass::Unclassified);

   batch.extraction_status[2] = CutQuadratureStatus::ExecutionFailure;
   REQUIRE(generator.GenerateReferenceBatch(batch, request, result,
                                             *workspace) ==
           CutQuadratureStatus::InvalidBatch);
   REQUIRE(result.status.Size() == 0);
   batch.extraction_status[2] = CutQuadratureStatus::UnsupportedSourceBasis;
   batch.element_descriptors[0].order = 2;
   REQUIRE(generator.GenerateReferenceBatch(batch, request, result,
                                             *workspace) ==
           CutQuadratureStatus::HeterogeneousBatch);
   REQUIRE(result.status.Size() == 0);
   batch.element_descriptors[0] = batch.descriptor;
   batch.coefficients.SetSize(3, 4);
   REQUIRE(generator.GenerateReferenceBatch(batch, request, result,
                                             *workspace) ==
           CutQuadratureStatus::InvalidBatch);

   batch.coefficients.SetSize(4, 4);

   batch.descriptor.basis = PolynomialBasis::BernsteinSimplex;
   batch.element_descriptors[0].basis = PolynomialBasis::BernsteinSimplex;
   batch.extraction_status[1] = CutQuadratureStatus::UnsupportedSourceBasis;
   REQUIRE(generator.GenerateReferenceBatch(batch, request, result,
                                             *workspace) ==
           CutQuadratureStatus::Success);
   REQUIRE(result.status[0] ==
           CutQuadratureStatus::UnsupportedPolynomialBasis);
   REQUIRE(result.classification[0] == CutCellClass::Unclassified);

   request.order = -1;
   request.execution = CutExecutionMode::Device;
   batch.extraction_status.SetSize(0);
   REQUIRE(generator.GenerateReferenceBatch(batch, request, result,
                                             *workspace) ==
           CutQuadratureStatus::InvalidRequest);
   request.order = 4;
   REQUIRE(generator.GenerateReferenceBatch(batch, request, result,
                                             *workspace) ==
           CutQuadratureStatus::UnsupportedExecutionMode);
}

TEST_CASE("Cut level-set providers and physical mapping",
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

   GridFunctionLevelSetProvider grid_provider(field, 3);
   CoefficientLevelSetProvider coefficient_provider(level_set, 2, 3);
   ElementLevelSet grid_local, coefficient_local;
   REQUIRE(grid_provider.GetElementLevelSet(0, Tr, grid_local) ==
           CutQuadratureStatus::Success);
   REQUIRE(coefficient_provider.GetElementLevelSet(0, Tr, coefficient_local) ==
           CutQuadratureStatus::Success);

   FiniteElementSpace vector_space(&mesh, &collection, 2);
   GridFunction vector_field(&vector_space);
   GridFunctionLevelSetProvider unsupported_provider(vector_field);
   ElementLevelSet unused;
   REQUIRE(unsupported_provider.GetElementLevelSet(0, Tr, unused) ==
           CutQuadratureStatus::UnsupportedSourceBasis);
   FunctionCoefficient not_finite([](const Vector &)
   {
      return std::numeric_limits<real_t>::quiet_NaN();
   });
   CoefficientLevelSetProvider invalid_provider(not_finite, 1);
   REQUIRE(invalid_provider.GetElementLevelSet(0, Tr, unused) ==
           CutQuadratureStatus::InvalidLevelSet);
   REQUIRE(grid_local.coefficients.Size() == 9);
   for (int j = 0; j < 3; j++)
   {
      REQUIRE(grid_local.coefficients[3*j] == MFEM_Approx(-1.0));
      REQUIRE(grid_local.coefficients[3*j + 1] == MFEM_Approx(0.0).margin(1e-12));
      REQUIRE(grid_local.coefficients[3*j + 2] == MFEM_Approx(1.0));
   }

   AlgoimCutQuadratureGenerator generator;
   auto workspace = generator.CreateWorkspace();
   CutQuadratureRequest request;
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;
   ReferenceCutQuadrature result;
   REQUIRE(generator.GenerateReference(grid_local, request, result,
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
   retained.provider_id = grid_provider.Id();
   retained.element = 0;
   retained.revision = grid_provider.Revision();
   retained.request = request;
   retained.result = result;
   REQUIRE(retained.IsValid(grid_provider, 0, request));
   field = 2.0; // Without a bump this deliberately remains a stale hit.
   REQUIRE(retained.IsValid(grid_provider, 0, request));
   grid_provider.IncrementRevision();
   REQUIRE_FALSE(retained.IsValid(grid_provider, 0, request));

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

TEST_CASE("Algoim shared generator uses per-thread workspaces",
          "[CutQuadrature][Algoim]")
{
   const AlgoimCutQuadratureGenerator generator;
   const ElementLevelSet cut = SquareLinear(-0.5, 0.5);
   const CutQuadratureRequest request;
   std::vector<CutQuadratureStatus> statuses(4);
#ifdef MFEM_USE_OPENMP
#pragma omp parallel for
#endif
   for (int i = 0; i < 4; i++)
   {
      auto workspace = generator.CreateWorkspace();
      ReferenceCutQuadrature result;
      statuses[i] = generator.GenerateReference(cut, request, result,
                                                *workspace);
   }
   for (auto status : statuses)
   {
      REQUIRE(status == CutQuadratureStatus::Success);
   }
}

#endif // MFEM_USE_ALGOIM
