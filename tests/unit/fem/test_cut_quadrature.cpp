#include "mfem.hpp"
#include "unit_tests.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef MFEM_USE_ALGOIM
#include "fem/cut_quadrature_algoim.hpp"
#endif

using namespace mfem;

namespace
{
real_t Measure(const IntegrationRule &rule)
{
   real_t result = 0;
   for (int index = 0; index < rule.GetNPoints(); ++index)
   { result += rule.IntPoint(index).weight; }
   return result;
}

class TestCutBackend : public CutQuadratureBackend
{
public:
   IntegrationRule rule;
   CutQuadratureStatus status = CutQuadratureStatus::Success;
   std::string Name() const override { return "test"; }
   CutQuadratureReport CheckSupport(Geometry::Type,
                                   const Array<const ReferenceLevelSet *> &,
                                   const CutQuadratureRequest &) const override
   { return {}; }
   CutQuadratureReport Generate(Geometry::Type,
                               const Array<const ReferenceLevelSet *> &,
                               const CutQuadratureRequest &, IntegrationRule &output,
                               CutRegion &region) const override
   {
      output = rule;
      region = CutRegion::Cut;
      return {status, "test"};
   }
};
}

TEST_CASE("Cut domain validation is geometry independent", "[CutQuadrature]")
{
   for (auto geometry : {Geometry::TRIANGLE, Geometry::TETRAHEDRON,
                         Geometry::SQUARE, Geometry::CUBE, Geometry::PRISM})
   {
      REQUIRE(CutDomain::Surface().Validate(geometry, 1).Succeeded());
      REQUIRE(CutDomain().OnFace(0).Validate(geometry, 0).Succeeded());
      REQUIRE_FALSE(CutDomain().OnFace(Geometry::NumBdrArray[geometry])
                    .Validate(geometry, 0).Succeeded());
   }
   REQUIRE_FALSE(CutDomain::Intersection(0, 0).Validate(Geometry::CUBE, 1).Succeeded());
   REQUIRE_FALSE(CutDomain::Surface(1).Validate(Geometry::SQUARE, 1).Succeeded());
   REQUIRE_FALSE(CutDomain::Intersection(0, 1).Validate(Geometry::SEGMENT, 2).Succeeded());
   REQUIRE_FALSE(CutDomain().Validate(Geometry::INVALID, 0).Succeeded());
}

TEST_CASE("Polynomial cut level sets preserve coefficients and gradients", "[CutQuadrature]")
{
   for (int dimension : {2, 3})
   {
      const auto geometry = dimension == 2 ? Geometry::SQUARE : Geometry::CUBE;
      IsoparametricTransformation identity;
      identity.SetIdentityTransformation(geometry);
      FunctionCoefficient coefficient([](const Vector &point)
      {
         return 0.2 + point(0)*point(0) + 2*point(1) +
                (point.Size() == 3 ? 3*point(2) : 0);
      });
      for (int basis : {BasisType::GaussLegendre, BasisType::Positive})
      {
         L2_FECollection collection(2, dimension, basis);
         const auto &element = *collection.FiniteElementForGeometry(geometry);
         PolynomialLevelSet level_set;
         const auto *saved = identity.GetIntPointPointer();
         REQUIRE(level_set.Project(coefficient, identity, element).Succeeded());
         REQUIRE(identity.GetIntPointPointer() == saved);
         IntegrationPoint point;
         point.Set3(0.21, 0.43, 0.67);
         const real_t exact = 0.2 + point.x*point.x + 2*point.y +
                              (dimension == 3 ? 3*point.z : 0);
         REQUIRE(level_set.Eval(point) == MFEM_Approx(exact));
         Vector gradient;
         level_set.EvalGradient(point, gradient);
         REQUIRE(gradient(0) == MFEM_Approx(2*point.x));
         REQUIRE(gradient(1) == MFEM_Approx(2));
         if (dimension == 3) { REQUIRE(gradient(2) == MFEM_Approx(3)); }
         L2_FECollection positive(2, dimension, BasisType::Positive);
         Vector coefficients;
         const auto &positive_element = *positive.FiniteElementForGeometry(geometry);
         REQUIRE(level_set.GetBernsteinCoefficients(positive_element, coefficients).Succeeded());
         if (basis == BasisType::Positive)
         {
            for (int index = 0; index < coefficients.Size(); ++index)
            { REQUIRE(coefficients(index) == level_set.GetDofs()(index)); }
         }
         PolynomialLevelSet converted;
         REQUIRE(converted.Set(positive_element, coefficients).Succeeded());
         coefficients = 0.;
         REQUIRE(converted.Eval(point) == MFEM_Approx(exact));
      }
      Mesh mesh = dimension == 2 ?
                  Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL) :
                  Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
      H1_FECollection collection(2, dimension);
      FiniteElementSpace space(&mesh, &collection);
      GridFunction field(&space);
      field.ProjectCoefficient(coefficient);
      PolynomialLevelSet extracted;
      REQUIRE(extracted.Set(field, 0).Succeeded());
      const auto &point = Geometries.GetCenter(geometry);
      REQUIRE(extracted.Eval(point) == MFEM_Approx(coefficient.Eval(identity, point)));
      REQUIRE_FALSE(extracted.Set(field, -1).Succeeded());
      REQUIRE(extracted.AsPolynomial() == nullptr);
   }
}

TEST_CASE("Cut generator clears failures and supports simplex conormals", "[CutQuadrature]")
{
   TestCutBackend backend;
   backend.rule.SetSize(1);
   backend.rule.IntPoint(0).Set2w(0.25, 0.25, 1.0);
   CutQuadratureGenerator generator(backend);
   FunctionLevelSet level_set(Geometry::TRIANGLE,
                             [](const IntegrationPoint &point) { return point.x - 0.25; },
                             [](const IntegrationPoint &, Vector &normal)
   { normal = 0.; normal(0) = 1; });
   Array<const ReferenceLevelSet *> level_sets(1);
   level_sets[0] = &level_set;
   CutQuadratureRequest request;
   request.domain = CutDomain::Surface();
   CutQuadrature output;
   REQUIRE(generator.Generate(Geometry::TRIANGLE, level_sets, request, output).Succeeded());
   REQUIRE(output.conormals(0)(0, 0) == MFEM_Approx(1));
   backend.status = CutQuadratureStatus::NumericalFailure;
   REQUIRE_FALSE(generator.Generate(Geometry::TRIANGLE, level_sets, request, output).Succeeded());
   REQUIRE(output.GetNPoints() == 0);
   REQUIRE(output.geometry == Geometry::INVALID);
   REQUIRE(output.region == CutRegion::Unknown);
   backend.status = CutQuadratureStatus::Success;
   backend.rule.IntPoint(0).x = 1.1;
   REQUIRE_FALSE(generator.Generate(Geometry::TRIANGLE, level_sets, request, output).Succeeded());
   backend.rule.IntPoint(0).x = std::numeric_limits<real_t>::quiet_NaN();
   REQUIRE_FALSE(generator.Generate(Geometry::TRIANGLE, level_sets, request, output).Succeeded());
   backend.rule.IntPoint(0).Set2w(0.5, 0, 1);
   request.domain = CutDomain().OnFace(0);
   REQUIRE(generator.Generate(Geometry::TRIANGLE, level_sets, request, output).Succeeded());
   REQUIRE(output.conormals(0)(1, 0) == MFEM_Approx(-1));
}

TEST_CASE("Cut mapping uses generalized Nanson scaling and preserves state", "[CutQuadrature]")
{
   CutQuadrature reference;
   reference.geometry = Geometry::SQUARE;
   reference.domain = CutDomain::Surface();
   reference.region = CutRegion::Cut;
   reference.rule.SetSize(1);
   reference.rule.IntPoint(0).Set2w(0.25, 0.5, 1);
   reference.conormals.SetSize(2, 1, 1);
   reference.conormals = 0.;
   reference.conormals(0)(0, 0) = 1;
   IsoparametricTransformation transformation;
   transformation.SetIdentityTransformation(Geometry::SQUARE);
   DenseMatrix points(3, 4);
   points = 0.;
   const auto &vertices = *Geometries.GetVertices(Geometry::SQUARE);
   for (int vertex = 0; vertex < 4; ++vertex)
   {
      points(0, vertex) = 2*vertices.IntPoint(vertex).x;
      points(1, vertex) = 3*vertices.IntPoint(vertex).y;
      points(2, vertex) = 4*vertices.IntPoint(vertex).y;
   }
   transformation.SetPointMat(points);
   for (auto type : {CutWeightType::Reference, CutWeightType::Physical,
                     CutWeightType::Integrator})
   {
      CutMappingOptions options;
      options.weight_type = type;
      options.data = CQ_PHYSICAL_POINTS | CQ_PHYSICAL_CONORMALS | CQ_MEASURE_SCALING;
      MappedCutQuadrature mapped;
      REQUIRE(transformation.GetIntPointPointer() == nullptr);
      REQUIRE(MapCutQuadrature(transformation, reference, options, mapped).Succeeded());
      REQUIRE(transformation.GetIntPointPointer() == nullptr);
      REQUIRE(mapped.measure_scaling(0) == MFEM_Approx(5));
      REQUIRE(mapped.physical_points(2, 0) == MFEM_Approx(2));
      REQUIRE(mapped.physical_conormals(0)(0, 0) == MFEM_Approx(1));
      const real_t expected = type == CutWeightType::Reference ? 1 :
                              (type == CutWeightType::Physical ? 5 : 0.5);
      REQUIRE(Measure(mapped.rule) == MFEM_Approx(expected));
      REQUIRE(Measure(reference.rule) == MFEM_Approx(1));
   }
   const IntegrationPoint &saved = Geometries.GetCenter(Geometry::SQUARE);
   transformation.SetIntPoint(&saved);
   MappedCutQuadrature mapped;
   REQUIRE(MapCutQuadrature(transformation, reference, {}, mapped).Succeeded());
   REQUIRE(transformation.GetIntPointPointer() == &saved);
   points = 0.;
   transformation.SetPointMat(points);
   REQUIRE_FALSE(MapCutQuadrature(transformation, reference, {}, mapped).Succeeded());
   REQUIRE(mapped.rule.GetNPoints() == 0);
   reference.rule.IntPoint(0).weight = 0;
   REQUIRE(MapCutQuadrature(transformation, reference, {}, mapped).Succeeded());
}

TEST_CASE("Cut mapping handles curved maps and intersecting surfaces", "[CutQuadrature]")
{
   CutQuadrature reference;
   reference.geometry = Geometry::SQUARE;
   reference.domain = CutDomain::Surface();
   reference.region = CutRegion::Cut;
   reference.rule.SetSize(1);
   reference.rule.IntPoint(0).Set2w(0.25, 0.5, 1);
   reference.conormals.SetSize(2, 1, 1);
   reference.conormals = 0.;
   reference.conormals(0)(0, 0) = 1;
   IsoparametricTransformation transformation;
   transformation.SetIdentityTransformation(Geometry::SQUARE);
   DenseMatrix points(transformation.GetPointMat());
   for (int vertex = 0; vertex < points.Width(); ++vertex)
   { points(1, vertex) *= 1 + points(0, vertex); }
   transformation.SetPointMat(points);
   CutMappingOptions options;
   options.weight_type = CutWeightType::Physical;
   MappedCutQuadrature mapped;
   REQUIRE(MapCutQuadrature(transformation, reference, options, mapped).Succeeded());
   REQUIRE(Measure(mapped.rule) == MFEM_Approx(1.25));

   reference.geometry = Geometry::CUBE;
   reference.domain = CutDomain::Intersection(0, 1);
   reference.rule.IntPoint(0).Set3w(0.2, 0.3, 0.5, 1);
   reference.conormals.SetSize(3, 2, 1);
   reference.conormals = 0.;
   reference.conormals(0)(0, 0) = 1;
   reference.conormals(0)(1, 1) = 1;
   transformation.SetIdentityTransformation(Geometry::CUBE);
   points = transformation.GetPointMat();
   for (int vertex = 0; vertex < points.Width(); ++vertex)
      for (int axis = 0; axis < 3; ++axis)
      { points(axis, vertex) *= axis + 2; }
   transformation.SetPointMat(points);
   REQUIRE(MapCutQuadrature(transformation, reference, options, mapped).Succeeded());
   REQUIRE(Measure(mapped.rule) == MFEM_Approx(4));
   reference.conormals(0)(1, 1) = 0;
   reference.conormals(0)(0, 1) = 1;
   REQUIRE(MapCutQuadrature(transformation, reference, options, mapped).status ==
           CutQuadratureStatus::DegenerateGeometry);
   REQUIRE(mapped.GetNPoints() == 0);
}

#ifdef MFEM_USE_ALGOIM
TEST_CASE("ALGOIM cut phases and planes on tensor cells", "[CutQuadrature]")
{
   auto backend = CreateCutQuadratureBackend("algoim");
   REQUIRE(backend != nullptr);
   CutQuadratureGenerator generator(*backend);
   for (int dimension : {2, 3})
   {
      const auto geometry = dimension == 2 ? Geometry::SQUARE : Geometry::CUBE;
      IsoparametricTransformation identity;
      identity.SetIdentityTransformation(geometry);
      H1_FECollection collection(2, dimension);
      const auto &element = *collection.FiniteElementForGeometry(geometry);
      for (int axis = 0; axis < dimension; ++axis)
      {
         FunctionCoefficient coefficient([axis](const Vector &point)
         { return point(axis) - 0.37; });
         PolynomialLevelSet level_set;
         REQUIRE(level_set.Project(coefficient, identity, element).Succeeded());
         Array<const ReferenceLevelSet *> level_sets(1);
         level_sets[0] = &level_set;
         real_t phase_sum = 0;
         for (int domain = 0; domain < 3; ++domain)
         {
            CutQuadratureRequest request;
            request.order = 6;
            request.domain = domain == 0 ? CutDomain::Positive() :
                             (domain == 1 ? CutDomain::Negative() : CutDomain::Surface());
            CutQuadrature output;
            auto report = generator.Generate(geometry, level_sets, request, output);
            INFO(report.message);
            REQUIRE(report.Succeeded());
            REQUIRE(output.rule.GetOrder() == request.order);
            REQUIRE(Measure(output.rule) == MFEM_Approx(domain == 0 ? 0.63 :
                                                       (domain == 1 ? 0.37 : 1.0)));
            if (domain < 2) { phase_sum += Measure(output.rule); }
            else
            {
               for (int index = 0; index < output.GetNPoints(); ++index)
               { REQUIRE(output.conormals(index)(axis, 0) == MFEM_Approx(1)); }
            }
         }
         REQUIRE(phase_sum == MFEM_Approx(1));
      }
   }
}

TEST_CASE("ALGOIM classifies constants and rejects unsupported requests", "[CutQuadrature]")
{
   auto backend = CreateCutQuadratureBackend("algoim");
   CutQuadratureGenerator generator(*backend);
   L2_FECollection collection(0, 2);
   const auto &element = *collection.FiniteElementForGeometry(Geometry::SQUARE);
   Vector values(1);
   PolynomialLevelSet level_set;
   Array<const ReferenceLevelSet *> level_sets(1);
   level_sets[0] = &level_set;
   CutQuadratureRequest request;
   CutQuadrature output;
   for (real_t value : {-1., 0., 1.})
   {
      values = value;
      REQUIRE(level_set.Set(element, values).Succeeded());
      request.domain = CutDomain::Positive();
      REQUIRE(generator.Generate(Geometry::SQUARE, level_sets, request, output).Succeeded());
      REQUIRE(Measure(output.rule) == MFEM_Approx(value > 0 ? 1 : 0));
      REQUIRE(output.region == (value > 0 ? CutRegion::Full : CutRegion::Empty));
      request.domain = CutDomain::Surface();
      auto report = generator.Generate(Geometry::SQUARE, level_sets, request, output);
      REQUIRE(report.status == (value == 0 ? CutQuadratureStatus::DegenerateGeometry :
                                CutQuadratureStatus::Success));
      REQUIRE(output.GetNPoints() == 0);
   }
   request.domain = CutDomain::Positive();
   request.order = -1;
   REQUIRE(generator.Generate(Geometry::SQUARE, level_sets, request, output).status ==
           CutQuadratureStatus::InvalidInput);
   request.order = 1000;
   REQUIRE(generator.Generate(Geometry::SQUARE, level_sets, request, output).status ==
           CutQuadratureStatus::Unsupported);
   request.order = 4;
   IntegrationRule fixed(1);
   fixed.IntPoint(0).Set2(0.2, 0.3);
   request.fixed_points = &fixed;
   REQUIRE(generator.Generate(Geometry::SQUARE, level_sets, request, output).status ==
           CutQuadratureStatus::Unsupported);
   request.fixed_points = nullptr;
   request.domain.OnFace(0);
   REQUIRE(generator.Generate(Geometry::SQUARE, level_sets, request, output).status ==
           CutQuadratureStatus::Unsupported);
   request.domain = CutDomain();
   Array<const ReferenceLevelSet *> none;
   REQUIRE(generator.Generate(Geometry::TRIANGLE, none, request, output).status ==
           CutQuadratureStatus::Unsupported);
   REQUIRE(output.GetNPoints() == 0);
   REQUIRE(CreateCutQuadratureBackend("moment") == nullptr);
}

TEST_CASE("ALGOIM curved interfaces converge", "[CutQuadrature]")
{
   std::vector<std::string> names = {"algoim-general"};
#ifdef MFEM_USE_LAPACKE
   names.push_back("algoim-polynomial");
#endif
   for (const auto &name : names)
   {
      auto backend = CreateCutQuadratureBackend(name);
      CutQuadratureGenerator generator(*backend);
      for (int dimension : {2, 3})
      {
         INFO(name << ", dimension = " << dimension);
         const auto geometry = dimension == 2 ? Geometry::SQUARE : Geometry::CUBE;
         L2_FECollection collection(2, dimension);
         const auto &element = *collection.FiniteElementForGeometry(geometry);
         IsoparametricTransformation identity;
         identity.SetIdentityTransformation(geometry);
         FunctionCoefficient coefficient([](const Vector &point)
         {
            real_t value = 0.32*0.32;
            for (int axis = 0; axis < point.Size(); ++axis)
            { value -= std::pow(point(axis)-0.5, 2); }
            return value;
         });
         PolynomialLevelSet level_set;
         REQUIRE(level_set.Project(coefficient, identity, element).Succeeded());
         Array<const ReferenceLevelSet *> level_sets(1);
         level_sets[0] = &level_set;
         for (bool surface : {false, true})
         {
            real_t previous_error = 1;
            const int high_order = name == "algoim-general" ? 18 : 28;
            for (int order : {4, high_order})
            {
               CutQuadratureRequest request;
               request.order = order;
               request.domain = surface ? CutDomain::Surface() : CutDomain::Positive();
               CutQuadrature output;
               auto report = generator.Generate(geometry, level_sets, request, output);
               INFO(report.message);
               REQUIRE(report.Succeeded());
               const real_t exact = dimension == 2 ?
                                    std::acos(-1.0) * (surface ? 0.64 : 0.32*0.32) :
                                    4 * std::acos(-1.0) * 0.32*0.32 * (surface ? 1 : 0.32/3);
               const real_t error = std::abs(Measure(output.rule) - exact);
               REQUIRE(error < previous_error);
               previous_error = error;
            }
            REQUIRE(previous_error < 1e-4);
         }
      }
   }
}

namespace
{
struct SineLevelSet
{
   bool fail = false;

   template <typename Scalar>
   Scalar operator()(const algoim::uvector<Scalar, 2> &point) const
   {
      if (fail) { throw std::runtime_error("Requested evaluation failure."); }
      using std::sin;
      return point(1) - sin(point(0));
   }

   template <typename Scalar>
   algoim::uvector<Scalar, 2> grad(const algoim::uvector<Scalar, 2> &point) const
   {
      using std::cos;
      algoim::uvector<Scalar, 2> result;
      result(0) = -cos(point(0));
      result(1) = Scalar(1);
      return result;
   }
};
}

TEST_CASE("General ALGOIM integrates interval-capable analytic functions", "[CutQuadrature]")
{
   AlgoimFunctionLevelSet<2, SineLevelSet> level_set(SineLevelSet{});
   Array<const ReferenceLevelSet *> level_sets(1);
   level_sets[0] = &level_set;
   CutQuadratureRequest request;
   request.order = 12;
   const auto saved = algoim::Interval<2>::delta();
   algoim::Interval<2>::delta() = 0.123;
   for (const char *name : {"algoim-general", "algoim"})
   {
      auto backend = CreateCutQuadratureBackend(name);
      REQUIRE(backend != nullptr);
      CutQuadratureGenerator generator(*backend);
      CutQuadrature output;
      real_t phases = 0;
      for (bool positive : {false, true})
      {
         request.domain = positive ? CutDomain::Positive() : CutDomain::Negative();
         auto report = generator.Generate(Geometry::SQUARE, level_sets, request, output);
         INFO(report.message);
         REQUIRE(report.Succeeded());
         const real_t expected = positive ? std::cos(1.0) : 1 - std::cos(1.0);
         REQUIRE(Measure(output.rule) == MFEM_Approx(expected));
         phases += Measure(output.rule);
         REQUIRE(algoim::Interval<2>::delta(0) == 0.123);
      }
      REQUIRE(phases == MFEM_Approx(1));
      request.domain = CutDomain::Surface();
      REQUIRE(generator.Generate(Geometry::SQUARE, level_sets, request, output).Succeeded());
      real_t exact_surface = 0;
      const auto &line = IntRules.Get(Geometry::SEGMENT, 60);
      for (int index = 0; index < line.GetNPoints(); ++index)
      {
         const auto &point = line.IntPoint(index);
         exact_surface += point.weight * std::sqrt(1 + std::pow(std::cos(point.x), 2));
      }
      REQUIRE(Measure(output.rule) == MFEM_Approx(exact_surface));
      for (int index = 0; index < output.GetNPoints(); ++index)
      {
         REQUIRE(std::abs(level_set.Eval(output.rule.IntPoint(index))) < 1e-12);
         REQUIRE(output.conormals(index)(1, 0) > 0);
      }
      request.order = 20;
      REQUIRE(generator.Generate(Geometry::SQUARE, level_sets, request, output).status ==
              CutQuadratureStatus::Unsupported);
      REQUIRE(output.GetNPoints() == 0);
      request.order = 12;
      AlgoimFunctionLevelSet<2, SineLevelSet> failing(SineLevelSet{true});
      level_sets[0] = &failing;
      REQUIRE(generator.Generate(Geometry::SQUARE, level_sets, request, output).status ==
              CutQuadratureStatus::NumericalFailure);
      REQUIRE(output.GetNPoints() == 0);
      REQUIRE(algoim::Interval<2>::delta(1) == 0.123);
      level_sets[0] = &level_set;
   }
   algoim::Interval<2>::delta() = saved;
}

TEST_CASE("ALGOIM dispatch depends on configured capabilities and active inputs", "[CutQuadrature]")
{
   auto automatic = CreateCutQuadratureBackend("algoim");
   auto general = CreateCutQuadratureBackend("algoim-general");
   auto polynomial = CreateCutQuadratureBackend("algoim-polynomial");
   L2_FECollection collection(2, 2);
   PolynomialLevelSet level_set;
   Vector coefficients(9);
   coefficients = 1.;
   REQUIRE(level_set.Set(*collection.FiniteElementForGeometry(Geometry::SQUARE), coefficients).Succeeded());
   FunctionLevelSet callback(Geometry::SQUARE,
      [](const IntegrationPoint &point) { return point.x - 0.3; },
      [](const IntegrationPoint &, Vector &gradient) { gradient = 0.; gradient(0) = 1; });
   Array<const ReferenceLevelSet *> level_sets(2);
   level_sets[0] = &callback;
   level_sets[1] = &level_set;
   CutQuadratureRequest request;
   request.domain = CutDomain::Positive(1);
   request.order = 20;
   REQUIRE(general->CheckSupport(Geometry::SQUARE, level_sets, request).status ==
           CutQuadratureStatus::Unsupported);
#ifdef MFEM_USE_LAPACKE
   REQUIRE(polynomial != nullptr);
   REQUIRE(automatic->CheckSupport(Geometry::SQUARE, level_sets, request).Succeeded());
#else
   REQUIRE(polynomial == nullptr);
   REQUIRE(automatic->CheckSupport(Geometry::SQUARE, level_sets, request).status ==
           CutQuadratureStatus::Unsupported);
#endif
   request.order = 6;
   request.domain = CutDomain::Positive(0);
   REQUIRE(automatic->CheckSupport(Geometry::SQUARE, level_sets, request).status ==
           CutQuadratureStatus::Unsupported);
   request.domain = CutDomain::Positive(1);
   request.domain.Add(0, LevelSetSign::Positive);
   REQUIRE(general->CheckSupport(Geometry::SQUARE, level_sets, request).status ==
           CutQuadratureStatus::Unsupported);
}

TEST_CASE("General ALGOIM evaluates tensor polynomials independently of LAPACKE", "[CutQuadrature]")
{
   for (int dimension : {2, 3})
   {
      const auto geometry = dimension == 2 ? Geometry::SQUARE : Geometry::CUBE;
      IsoparametricTransformation identity;
      identity.SetIdentityTransformation(geometry);
      for (int basis : {BasisType::GaussLegendre, BasisType::Positive})
      {
         L2_FECollection collection(3, dimension, basis);
         FunctionCoefficient coefficient([](const Vector &point)
         { return point(1) - 0.2 - 0.4*point(0)*point(0); });
         PolynomialLevelSet level_set;
         REQUIRE(level_set.Project(coefficient, identity,
                 *collection.FiniteElementForGeometry(geometry)).Succeeded());
         Array<const ReferenceLevelSet *> level_sets(1);
         level_sets[0] = &level_set;
         CutQuadratureRequest request;
         request.order = 12;
         request.domain = CutDomain::Negative();
         auto general = CreateCutQuadratureBackend("algoim-general");
         CutQuadratureGenerator generator(*general);
         CutQuadrature output;
         auto report = generator.Generate(geometry, level_sets, request, output);
         INFO(report.message);
         REQUIRE(report.Succeeded());
         REQUIRE(Measure(output.rule) == MFEM_Approx(1.0/3.0));
         request.domain = CutDomain::Surface();
         REQUIRE(generator.Generate(geometry, level_sets, request, output).Succeeded());
         const real_t exact = 0.5*std::sqrt(1.64) + std::asinh(0.8)/1.6;
         REQUIRE(Measure(output.rule) == MFEM_Approx(exact));
#ifdef MFEM_USE_LAPACKE
         auto polynomial = CreateCutQuadratureBackend("algoim-polynomial");
         CutQuadratureGenerator other(*polynomial);
         CutQuadrature compared;
         REQUIRE(other.Generate(geometry, level_sets, request, compared).Succeeded());
         REQUIRE(Measure(compared.rule) == MFEM_Approx(Measure(output.rule)));
#endif
      }
   }
}

#ifdef MFEM_USE_LAPACKE
TEST_CASE("Polynomial ALGOIM partitions two level sets and preserves moments", "[CutQuadrature]")
{
   for (int dimension : {2, 3})
   {
      const auto geometry = dimension == 2 ? Geometry::SQUARE : Geometry::CUBE;
      H1_FECollection first_collection(1, dimension);
      L2_FECollection second_collection(3, dimension, BasisType::Positive);
      IsoparametricTransformation identity;
      identity.SetIdentityTransformation(geometry);
      FunctionCoefficient first_coefficient([](const Vector &point) { return point(0)-0.3; });
      FunctionCoefficient second_coefficient([](const Vector &point) { return point(1)-0.4; });
      PolynomialLevelSet first, second;
      REQUIRE(first.Project(first_coefficient, identity,
                            *first_collection.FiniteElementForGeometry(geometry)).Succeeded());
      REQUIRE(second.Project(second_coefficient, identity,
                             *second_collection.FiniteElementForGeometry(geometry)).Succeeded());
      Array<const ReferenceLevelSet *> level_sets(3);
      level_sets[0] = &second;
      level_sets[1] = &first;
      level_sets[2] = &second;
      for (const char *name : {"algoim", "algoim-polynomial"})
      {
         auto backend = CreateCutQuadratureBackend(name);
         CutQuadratureGenerator generator(*backend);
         for (bool reverse : {false, true})
         {
            real_t total = 0;
            for (bool positive_first : {false, true})
               for (bool positive_second : {false, true})
               {
                  const auto first_sign = positive_first ? LevelSetSign::Positive : LevelSetSign::Negative;
                  const auto second_sign = positive_second ? LevelSetSign::Positive : LevelSetSign::Negative;
                  CutQuadratureRequest request;
                  request.order = 6;
                  request.domain = CutDomain();
                  if (reverse) { request.domain.Add(2, second_sign).Add(1, first_sign); }
                  else { request.domain.Add(1, first_sign).Add(2, second_sign); }
                  CutQuadrature output;
                  auto report = generator.Generate(geometry, level_sets, request, output);
                  INFO(report.message);
                  REQUIRE(report.Succeeded());
                  const real_t measure = (positive_first ? 0.7 : 0.3) * (positive_second ? 0.6 : 0.4);
                  REQUIRE(Measure(output.rule) == MFEM_Approx(measure));
                  real_t first_moment = 0, second_moment = 0;
                  for (int index = 0; index < output.GetNPoints(); ++index)
                  {
                     const auto &point = output.rule.IntPoint(index);
                     REQUIRE((positive_first ? 1 : -1)*first.Eval(point) >= -1e-12);
                     REQUIRE((positive_second ? 1 : -1)*second.Eval(point) >= -1e-12);
                     first_moment += point.x*point.weight;
                     second_moment += point.y*point.weight;
                  }
                  REQUIRE(first_moment == MFEM_Approx(measure*(positive_first ? 0.65 : 0.15)));
                  REQUIRE(second_moment == MFEM_Approx(measure*(positive_second ? 0.7 : 0.2)));
                  total += Measure(output.rule);
               }
            REQUIRE(total == MFEM_Approx(1));
         }
         CutQuadratureRequest request;
         request.domain = CutDomain::Intersection(1, 2);
         REQUIRE(backend->CheckSupport(geometry, level_sets, request).status == CutQuadratureStatus::Unsupported);
         request.domain = CutDomain::Surface(1).Add(2, LevelSetSign::Positive);
         REQUIRE(backend->CheckSupport(geometry, level_sets, request).status == CutQuadratureStatus::Unsupported);
         request.domain = CutDomain::Positive(0).Add(1, LevelSetSign::Positive).Add(2, LevelSetSign::Positive);
         REQUIRE(backend->CheckSupport(geometry, level_sets, request).status == CutQuadratureStatus::Unsupported);
      }
   }
}

TEST_CASE("Polynomial ALGOIM handles empty, oblique and curved intersections", "[CutQuadrature]")
{
   auto backend = CreateCutQuadratureBackend("algoim-polynomial");
   CutQuadratureGenerator generator(*backend);
   L2_FECollection collection(2, 2);
   const auto &element = *collection.FiniteElementForGeometry(Geometry::SQUARE);
   IsoparametricTransformation identity;
   identity.SetIdentityTransformation(Geometry::SQUARE);
   for (int shape = 0; shape < 5; ++shape)
   {
      INFO("shape = " << shape);
      FunctionCoefficient first_coefficient([shape](const Vector &point)
      {
         if (shape == 0) { return point(0)-0.7; }
         if (shape == 1) { return point(0)+point(1)-0.8; }
         if (shape == 2) { return 0.32*0.32-std::pow(point(0)-0.5,2)-std::pow(point(1)-0.5,2); }
         return shape == 3 ? 1.0 : -1.0;
      });
      FunctionCoefficient second_coefficient([shape](const Vector &point)
      { return shape == 0 ? 0.3-point(0) : (shape == 2 ? point(0)-0.5 : point(1)-0.4); });
      PolynomialLevelSet first, second;
      REQUIRE(first.Project(first_coefficient, identity, element).Succeeded());
      REQUIRE(second.Project(second_coefficient, identity, element).Succeeded());
      Array<const ReferenceLevelSet *> level_sets(2);
      level_sets[0] = &first; level_sets[1] = &second;
      CutQuadratureRequest request;
      request.order = 28;
      request.domain = CutDomain::Positive(0).Add(1, LevelSetSign::Positive);
      CutQuadrature output;
      auto report = generator.Generate(Geometry::SQUARE, level_sets, request, output);
      INFO(report.message);
      REQUIRE(report.Succeeded());
      const real_t expected[] = {0, 0.52, std::acos(-1.0)*0.32*0.32/2, 0.6, 0};
      const real_t error = std::abs(Measure(output.rule)-expected[shape]);
      REQUIRE(error < (shape == 2 ? 1e-5 : 1e-12));
      if (shape == 2)
      {
         request.order = 4;
         REQUIRE(generator.Generate(Geometry::SQUARE, level_sets, request, output).Succeeded());
         REQUIRE(error < std::abs(Measure(output.rule)-expected[shape]));
      }
   }
}

TEST_CASE("Polynomial ALGOIM intersects genuinely three-dimensional cuts", "[CutQuadrature]")
{
   auto backend = CreateCutQuadratureBackend("algoim-polynomial");
   CutQuadratureGenerator generator(*backend);
   L2_FECollection collection(2, 3);
   const auto &element = *collection.FiniteElementForGeometry(Geometry::CUBE);
   IsoparametricTransformation identity;
   identity.SetIdentityTransformation(Geometry::CUBE);
   FunctionCoefficient first_coefficient([](const Vector &point)
   { return point(0) + point(1) + point(2) - 1; });
   FunctionCoefficient second_coefficient([](const Vector &point)
   { return point(0) + 2*point(1) + 3*point(2) - 1; });
   PolynomialLevelSet first, second;
   REQUIRE(first.Project(first_coefficient, identity, element).Succeeded());
   REQUIRE(second.Project(second_coefficient, identity, element).Succeeded());
   Array<const ReferenceLevelSet *> level_sets(2);
   level_sets[0] = &first; level_sets[1] = &second;
   CutQuadratureRequest request;
   request.order = 10;
   request.domain = CutDomain::Negative(0).Add(1, LevelSetSign::Positive);
   CutQuadrature output;
   auto report = generator.Generate(Geometry::CUBE, level_sets, request, output);
   INFO(report.message);
   REQUIRE(report.Succeeded());
   REQUIRE(Measure(output.rule) == MFEM_Approx(5.0/36.0));
}
#endif
#else
TEST_CASE("Disabled cut backend remains discoverable as unavailable", "[CutQuadrature]")
{
   REQUIRE(CreateCutQuadratureBackend("algoim") == nullptr);
   REQUIRE(GetCutQuadratureBackends().empty());
}
#endif
