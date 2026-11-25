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
   CutQuadratureReport::Status status = CutQuadratureReport::Status::Success;
   std::string Name() const override { return "test"; }
   CutQuadratureReport CheckSupport(Geometry::Type, const CutDomain &, int,
                                    const IntegrationRule *) const override
   { return {}; }
   CutQuadratureReport Generate(Geometry::Type, const CutDomain &, int,
                                CutQuadrature &output,
                                const IntegrationRule *) const override
   {
      output.Clear();
      if (status != CutQuadratureReport::Status::Success) { return {status, "test"}; }
      output.rule = rule;
      output.region = CutQuadrature::Region::Cut;
      return {status, "test"};
   }
};

class ThrowingCutTransformation : public IsoparametricTransformation
{
public:
   bool fail = false;

   explicit ThrowingCutTransformation(bool jacobian) : jacobian(jacobian)
   { SetIdentityTransformation(Geometry::SQUARE); }

   void Transform(const IntegrationPoint &point, Vector &physical) override
   {
      if (fail && !jacobian && point.x > 0.5)
      { throw std::runtime_error("Requested transform failure."); }
      IsoparametricTransformation::Transform(point, physical);
   }

protected:
   const DenseMatrix &EvalJacobian() override
   {
      if (fail && jacobian && GetIntPoint().x > 0.5)
      { throw std::runtime_error("Requested Jacobian failure."); }
      DenseMatrix derivatives(GetFE()->GetDof(), GetFE()->GetDim());
      GetFE()->CalcDShape(GetIntPoint(), derivatives);
      dFdx.SetSize(GetSpaceDim(), GetFE()->GetDim());
      Mult(GetPointMat(), derivatives, dFdx);
      EvalState |= JACOBIAN_MASK;
      return dFdx;
   }

private:
   bool jacobian;
};
}

TEST_CASE("Cut domain validation is geometry independent", "[CutQuadrature]")
{
   for (auto geometry :
        {
           Geometry::TRIANGLE, Geometry::TETRAHEDRON,
           Geometry::SQUARE, Geometry::CUBE, Geometry::PRISM
        })
   {
      FunctionLevelSet level_set(geometry,
      [](const IntegrationPoint &point) { return point.x; },
      [](const IntegrationPoint &, Vector &gradient)
      { gradient = 0.; });
      REQUIRE(CutDomain::Surface(level_set).Validate(geometry).Succeeded());
   }
   auto value = [](const IntegrationPoint &point) { return point.x; };
   auto gradient = [](const IntegrationPoint &, Vector &result) { result = 0.; };
   FunctionLevelSet first(Geometry::SEGMENT, value, gradient);
   FunctionLevelSet second(Geometry::SEGMENT, value, gradient);
   REQUIRE_FALSE(CutDomain::Surface(first).Add(first,
                                               CutDomain::LevelSetSign::Zero)
                 .Validate(Geometry::SEGMENT).Succeeded());
   REQUIRE_FALSE(CutDomain::Surface(first).Validate(Geometry::SQUARE).Succeeded());
   REQUIRE_FALSE(CutDomain::Surface(first).Add(second,
                                               CutDomain::LevelSetSign::Zero)
                 .Validate(Geometry::SEGMENT).Succeeded());
   REQUIRE_FALSE(CutDomain().Validate(Geometry::INVALID).Succeeded());
}

TEST_CASE("Polynomial cut level sets preserve coefficients and gradients",
          "[CutQuadrature]")
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
         level_set.Project(coefficient, identity, element);
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
      }
      Mesh mesh = dimension == 2 ?
                  Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL) :
                  Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
      H1_FECollection collection(2, dimension);
      FiniteElementSpace space(&mesh, &collection);
      GridFunction field(&space);
      field.ProjectCoefficient(coefficient);
      PolynomialLevelSet extracted;
      extracted.Set(field, 0);
      const auto &point = Geometries.GetCenter(geometry);
      REQUIRE(extracted.Eval(point) == MFEM_Approx(coefficient.Eval(identity,
                                                                    point)));
   }
}

#ifdef MFEM_USE_LAPACK
TEST_CASE("Moment cut quadrature preserves one-dimensional measures",
          "[CutQuadrature]")
{
   Mesh mesh = Mesh::MakeCartesian1D(1);
   H1_FECollection collection(2, 1);
   FiniteElementSpace space(&mesh, &collection);
   FunctionCoefficient coefficient([](const Vector &point)
   { return 0.55 - point(0); });
   GridFunction field(&space);
   field.ProjectCoefficient(coefficient);
   PolynomialLevelSet level_set;
   level_set.Set(field, 0);

   auto backend = CreateCutQuadratureBackend("moment");
   REQUIRE(backend != nullptr);
   CutQuadratureGenerator generator(*backend);
   CutQuadrature output;
   auto report = generator.Generate(
                    Geometry::SEGMENT, CutDomain::Positive(level_set), 2, output);
   REQUIRE(report.Succeeded());
   REQUIRE(report.moment_degree == 2);
   REQUIRE(Measure(output.rule) == MFEM_Approx(0.55).margin(2e-12));

   REQUIRE(generator.Generate(Geometry::SEGMENT,
                              CutDomain::Negative(level_set), 2,
                              output).Succeeded());
   REQUIRE(Measure(output.rule) == MFEM_Approx(0.45).margin(2e-12));

   REQUIRE(generator.Generate(Geometry::SEGMENT,
                              CutDomain::Surface(level_set), 2,
                              output).Succeeded());
   REQUIRE(Measure(output.rule) == MFEM_Approx(1.0).margin(2e-12));
   REQUIRE(output.conormals(0)(0, 0) == MFEM_Approx(-1.0));
}

TEST_CASE("Moment cut quadrature preserves prescribed volume nodes",
          "[CutQuadrature]")
{
   FunctionLevelSet level_set(
      Geometry::SQUARE,
   [](const IntegrationPoint &point) { return point.x - 0.5; },
   [](const IntegrationPoint &, Vector &gradient)
   { gradient = 0.; gradient(0) = 1.; });
   auto backend = CreateCutQuadratureBackend("moment");
   REQUIRE(backend != nullptr);
   CutQuadratureGenerator generator(*backend);
   const IntegrationRule &points = IntRules.Get(Geometry::SQUARE, 4);
   CutQuadrature output;
   REQUIRE(generator.Generate(Geometry::SQUARE, CutDomain(), 1,
                              output, &points).status ==
           CutQuadratureReport::Status::Unsupported);
   REQUIRE(output.GetNPoints() == 0);
   REQUIRE(generator.Generate(Geometry::SQUARE,
                              CutDomain::Positive(level_set), 1,
                              output, &points).Succeeded());
   REQUIRE(output.GetNPoints() == points.GetNPoints());
   REQUIRE(Measure(output.rule) == MFEM_Approx(0.5).margin(2e-12));
   for (int index = 0; index < output.GetNPoints(); ++index)
   {
      REQUIRE(output.rule.IntPoint(index).x == points.IntPoint(index).x);
      REQUIRE(output.rule.IntPoint(index).y == points.IntPoint(index).y);
   }

   FunctionLevelSet full(
      Geometry::SQUARE,
   [](const IntegrationPoint &) { return 1.0; },
   [](const IntegrationPoint &, Vector &gradient) { gradient = 0.; });
   REQUIRE(generator.Generate(Geometry::SQUARE, CutDomain::Positive(full), 1,
                              output, &points).Succeeded());
   REQUIRE(Measure(output.rule) == MFEM_Approx(1.0).margin(2e-12));
   for (int index = 0; index < output.GetNPoints(); ++index)
   {
      REQUIRE(output.rule.IntPoint(index).x == points.IntPoint(index).x);
      REQUIRE(output.rule.IntPoint(index).y == points.IntPoint(index).y);
   }
}
#endif

TEST_CASE("Cut generator clears failures and supports simplex conormals",
          "[CutQuadrature]")
{
   TestCutBackend backend;
   backend.rule.SetSize(1);
   backend.rule.IntPoint(0).Set2w(0.25, 0.25, 1.0);
   CutQuadratureGenerator generator(backend);
   FunctionLevelSet level_set(Geometry::TRIANGLE,
   [](const IntegrationPoint &point) { return point.x - 0.25; },
   [](const IntegrationPoint &, Vector &normal)
   { normal = 0.; normal(0) = 1; });
   const auto domain = CutDomain::Surface(level_set);
   CutQuadrature output;
   REQUIRE(generator.Generate(Geometry::TRIANGLE, domain, 2, output).Succeeded());
   REQUIRE(output.conormals(0)(0, 0) == MFEM_Approx(1));
   backend.status = CutQuadratureReport::Status::NumericalFailure;
   REQUIRE_FALSE(generator.Generate(Geometry::TRIANGLE, domain, 2,
                                    output).Succeeded());
   REQUIRE(output.GetNPoints() == 0);
   REQUIRE(output.geometry == Geometry::INVALID);
   REQUIRE(output.region == CutQuadrature::Region::Unknown);
   backend.status = CutQuadratureReport::Status::Success;
   backend.rule.IntPoint(0).x = 1.1;
   REQUIRE_FALSE(generator.Generate(Geometry::TRIANGLE, domain, 2,
                                    output).Succeeded());
   backend.rule.IntPoint(0).x = std::numeric_limits<real_t>::quiet_NaN();
   REQUIRE_FALSE(generator.Generate(Geometry::TRIANGLE, domain, 2,
                                    output).Succeeded());
}

TEST_CASE("Cut generator reports gradient exceptions and clears outputs",
          "[CutQuadrature]")
{
   TestCutBackend backend;
   backend.rule.SetSize(2);
   backend.rule.IntPoint(0).Set2w(0.25, 0.5, 0.5);
   backend.rule.IntPoint(1).Set2w(0.75, 0.5, 0.5);
   CutQuadratureGenerator generator(backend);
   bool throw_gradient = true;
   FunctionLevelSet level_set(Geometry::SQUARE,
   [](const IntegrationPoint &point) { return point.y - 0.5; },
   [&](const IntegrationPoint &point, Vector &normal)
   {
      if (throw_gradient && point.x > 0.5)
      { throw std::runtime_error("gradient callback failed"); }
      normal = 0.; normal(1) = 1;
   });
   const auto domain = CutDomain::Surface(level_set);
   CutQuadrature output;
   CutQuadratureReport report;
   REQUIRE_NOTHROW(report = generator.Generate(Geometry::SQUARE, domain, 2,
                                               output));
   REQUIRE(report.status == CutQuadratureReport::Status::NumericalFailure);
   REQUIRE(report.message == "gradient callback failed");
   REQUIRE(output.GetNPoints() == 0);
   REQUIRE(output.rule.GetOrder() == -1);
   REQUIRE(output.geometry == Geometry::INVALID);
   REQUIRE(output.region == CutQuadrature::Region::Unknown);
   REQUIRE(output.conormals.SizeK() == 0);

   throw_gradient = false;
   REQUIRE(generator.Generate(Geometry::SQUARE, domain, 2, output).Succeeded());
   REQUIRE(output.GetNPoints() == 2);
}

TEST_CASE("Cut mapping uses generalized Nanson scaling",
          "[CutQuadrature]")
{
   CutQuadrature reference;
   reference.geometry = Geometry::SQUARE;
   reference.codimension = 1;
   reference.region = CutQuadrature::Region::Cut;
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
   const int data = CutQuadratureMapping::PHYSICAL_POINTS |
                    CutQuadratureMapping::PHYSICAL_CONORMALS |
                    CutQuadratureMapping::MEASURE_SCALING;
   CutQuadratureMapping mapped;
   REQUIRE(MapCutQuadrature(transformation, reference, data, mapped).Succeeded());
   REQUIRE(mapped.measure_scaling(0) == MFEM_Approx(5));
   REQUIRE(mapped.physical_points(2, 0) == MFEM_Approx(2));
   REQUIRE(mapped.physical_conormals(0)(0, 0) == MFEM_Approx(1));
   REQUIRE(reference.rule.IntPoint(0).weight == MFEM_Approx(1));
   REQUIRE(reference.rule.IntPoint(0).weight * mapped.measure_scaling(0) ==
           MFEM_Approx(5));
   points = 0.;
   transformation.SetPointMat(points);
   REQUIRE_FALSE(MapCutQuadrature(transformation, reference,
                                  CutQuadratureMapping::MEASURE_SCALING,
                                  mapped).Succeeded());
   REQUIRE(mapped.measure_scaling.Size() == 0);
   reference.rule.IntPoint(0).weight = 0;
   REQUIRE(MapCutQuadrature(transformation, reference,
                            CutQuadratureMapping::MEASURE_SCALING,
                            mapped).Succeeded());
   REQUIRE(mapped.measure_scaling(0) == 0);
}

TEST_CASE("Cut mapping clears transformation exceptions",
          "[CutQuadrature]")
{
   TestCutBackend backend;
   backend.rule.SetSize(2);
   backend.rule.IntPoint(0).Set2w(0.25, 0.5, 0.5);
   backend.rule.IntPoint(1).Set2w(0.75, 0.5, 0.5);
   CutQuadratureGenerator generator(backend);
   FunctionLevelSet level_set(Geometry::SQUARE,
   [](const IntegrationPoint &point) { return point.y - 0.5; },
   [](const IntegrationPoint &, Vector &normal)
   { normal = 0.; normal(1) = 1; });
   const auto domain = CutDomain::Surface(level_set);
   CutQuadrature reference;
   REQUIRE(generator.Generate(Geometry::SQUARE, domain, 2, reference).Succeeded());
   const int data = CutQuadratureMapping::PHYSICAL_POINTS |
                    CutQuadratureMapping::PHYSICAL_CONORMALS |
                    CutQuadratureMapping::MEASURE_SCALING;
   for (bool jacobian : {false, true})
   {
      ThrowingCutTransformation transformation(jacobian);
      transformation.GetPointMat() *= 2;
      CutQuadratureMapping mapped;
      REQUIRE(MapCutQuadrature(transformation, reference, data,
                               mapped).Succeeded());
      REQUIRE(mapped.physical_points.Width() == 2);
      transformation.fail = true;
      CutQuadratureReport report;
      REQUIRE_NOTHROW(report = MapCutQuadrature(
                                  transformation, reference, data, mapped));
      REQUIRE(report.status == CutQuadratureReport::Status::NumericalFailure);
      REQUIRE(report.message == (jacobian ? "Requested Jacobian failure." :
                                 "Requested transform failure."));
      REQUIRE(mapped.physical_points.Height() == 0);
      REQUIRE(mapped.physical_points.Width() == 0);
      REQUIRE(mapped.physical_conormals.SizeI() == 0);
      REQUIRE(mapped.physical_conormals.SizeJ() == 0);
      REQUIRE(mapped.physical_conormals.SizeK() == 0);
      REQUIRE(mapped.measure_scaling.Size() == 0);
      REQUIRE(Measure(reference.rule) == MFEM_Approx(1));
   }
}

TEST_CASE("Cut measure scaling is positive under reflections",
          "[CutQuadrature]")
{
   for (int dimension : {2, 3})
      for (bool embedded : {false, true})
      {
         if (embedded && dimension == 3) { continue; }
         const auto geometry = dimension == 2 ? Geometry::SQUARE : Geometry::CUBE;
         IsoparametricTransformation transformation;
         transformation.SetIdentityTransformation(geometry);
         DenseMatrix points(dimension + (embedded ? 1 : 0),
                            transformation.GetPointMat().Width());
         const auto &vertices = *Geometries.GetVertices(geometry);
         for (int vertex = 0; vertex < points.Width(); ++vertex)
         {
            points(0, vertex) = -2 * vertices.IntPoint(vertex).x;
            points(1, vertex) = 3 * vertices.IntPoint(vertex).y;
            if (dimension == 3) { points(2, vertex) = 4 * vertices.IntPoint(vertex).z; }
            else if (embedded) { points(2, vertex) = 4 * vertices.IntPoint(vertex).y; }
         }
         transformation.SetPointMat(points);
         for (int codimension : {0, 1})
         {
            CutQuadrature reference;
            reference.geometry = geometry;
            reference.codimension = codimension;
            reference.region = codimension ? CutQuadrature::Region::Cut :
                               CutQuadrature::Region::Full;
            reference.rule.SetSize(1);
            reference.rule.IntPoint(0).Set3w(0.25, 0.5, 0.4, 0.7);
            if (codimension)
            {
               reference.conormals.SetSize(dimension, 1, 1);
               reference.conormals = 0.;
               reference.conormals(0)(0, 0) = 1;
            }
            transformation.SetIntPoint(&reference.rule.IntPoint(0));
            const real_t parent_weight = transformation.Weight();
            REQUIRE((parent_weight > 0) == embedded);
            const int data = CutQuadratureMapping::MEASURE_SCALING |
                             CutQuadratureMapping::PHYSICAL_CONORMALS;
            CutQuadratureMapping mapped;
            REQUIRE(MapCutQuadrature(transformation, reference, data,
                                     mapped).Succeeded());
            const real_t scaling = std::abs(parent_weight) / (codimension ? 2 : 1);
            REQUIRE(mapped.measure_scaling(0) == MFEM_Approx(scaling));
            REQUIRE(Measure(reference.rule) == MFEM_Approx(0.7));
            if (codimension)
            { REQUIRE(mapped.physical_conormals(0)(0, 0) == MFEM_Approx(-1)); }
            transformation.SetIntPoint(nullptr);
         }
      }
}

TEST_CASE("Cut mapping handles curved maps and intersecting surfaces",
          "[CutQuadrature]")
{
   CutQuadrature reference;
   reference.geometry = Geometry::SQUARE;
   reference.codimension = 1;
   reference.region = CutQuadrature::Region::Cut;
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
   CutQuadratureMapping mapped;
   REQUIRE(MapCutQuadrature(transformation, reference,
                            CutQuadratureMapping::MEASURE_SCALING,
                            mapped).Succeeded());
   REQUIRE(reference.rule.IntPoint(0).weight * mapped.measure_scaling(0) ==
           MFEM_Approx(1.25));

   reference.geometry = Geometry::CUBE;
   reference.codimension = 2;
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
   REQUIRE(MapCutQuadrature(transformation, reference,
                            CutQuadratureMapping::MEASURE_SCALING,
                            mapped).Succeeded());
   REQUIRE(reference.rule.IntPoint(0).weight * mapped.measure_scaling(0) ==
           MFEM_Approx(4));
   reference.conormals(0)(1, 1) = 0;
   reference.conormals(0)(0, 1) = 1;
   REQUIRE(MapCutQuadrature(transformation, reference,
                            CutQuadratureMapping::MEASURE_SCALING,
                            mapped).status ==
           CutQuadratureReport::Status::DegenerateGeometry);
   REQUIRE(mapped.measure_scaling.Size() == 0);
}

TEST_CASE("Cut mapping remains stable on sheared and embedded elements",
          "[CutQuadrature]")
{
   for (int configuration = 0; configuration < 3; ++configuration)
   {
      const int dimension = configuration == 2 ? 3 : 2;
      const int space_dimension = configuration == 0 ? 2 : 3;
      const bool embedded = dimension != space_dimension;
      const auto geometry = dimension == 2 ? Geometry::SQUARE : Geometry::CUBE;
      for (real_t thickness : {1e-7, 1e-8, 1e-10})
      {
         INFO("configuration = " << configuration << ", thickness = " << thickness);
         IsoparametricTransformation transformation;
         transformation.SetIdentityTransformation(geometry);
         const auto &vertices = *Geometries.GetVertices(geometry);
         DenseMatrix points(space_dimension, vertices.GetNPoints());
         for (int vertex = 0; vertex < vertices.GetNPoints(); ++vertex)
         {
            const auto &point = vertices.IntPoint(vertex);
            points(0, vertex) = point.x + point.y;
            points(1, vertex) = thickness * point.y;
            if (space_dimension == 3)
            { points(2, vertex) = embedded ? 2 * thickness * point.y : 2 * point.z; }
         }
         transformation.SetPointMat(points);
         const auto &center = Geometries.GetCenter(geometry);
         transformation.SetIntPoint(&center);
         const real_t height = embedded ? std::sqrt(5.0) * thickness : thickness;
         const real_t extrusion = dimension == 3 ? 2 : 1;
         const real_t volume = height * extrusion;
         REQUIRE(transformation.Weight() / volume == MFEM_Approx(1));
         for (int codimension = 0; codimension <= dimension; ++codimension)
         {
            INFO("codimension = " << codimension);
            CutQuadrature reference;
            reference.geometry = geometry;
            reference.codimension = codimension;
            reference.region = CutQuadrature::Region::Cut;
            reference.rule.SetSize(1);
            reference.rule.IntPoint(0) = center;
            reference.rule.IntPoint(0).weight = 1;
            if (codimension)
            {
               reference.conormals.SetSize(dimension, codimension, 1);
               reference.conormals = 0.;
               for (int column = 0; column < codimension; ++column)
               {
                  reference.conormals(0)(column, column) = 1;
               }
               if (codimension > 1)
               {
                  reference.conormals(0)(0, 1) = 1 / std::sqrt(2.0);
                  reference.conormals(0)(1, 1) = 1 / std::sqrt(2.0);
               }
            }
            const real_t measure = codimension == 0 ? volume :
                                   (codimension == 1 ? std::hypot(1.0, height) * extrusion :
                                    (codimension == dimension ? 1 : extrusion));
            const int data = CutQuadratureMapping::PHYSICAL_CONORMALS |
                             CutQuadratureMapping::MEASURE_SCALING;
            CutQuadratureMapping mapped;
            auto report = MapCutQuadrature(transformation, reference, data,
                                           mapped);
            INFO(report.message);
            REQUIRE(report.Succeeded());
            REQUIRE(mapped.measure_scaling(0) / measure == MFEM_Approx(1));
            REQUIRE(reference.rule.IntPoint(0).weight == MFEM_Approx(1));
            for (int column = 0; column < codimension; ++column)
            {
               Vector normal(space_dimension), expected_normal(space_dimension);
               mapped.physical_conormals(0).GetColumn(column, normal);
               expected_normal = 0.;
               if (column == 0)
               {
                  expected_normal(0) = thickness;
                  expected_normal(1) = embedded ? -0.2 : -1;
                  if (embedded) { expected_normal(2) = -0.4; }
               }
               else { expected_normal(column == 1 ? 0 : column) = 1; }
               expected_normal /= expected_normal.Norml2();
               REQUIRE(normal.Norml2() == MFEM_Approx(1));
               normal -= expected_normal;
               REQUIRE(normal.Norml2() < 1e-12);
            }
         }
      }
   }
}

#ifdef MFEM_USE_ALGOIM
TEST_CASE("ALGOIM converts polynomial level-set bases internally",
          "[CutQuadrature]")
{
   IsoparametricTransformation identity;
   identity.SetIdentityTransformation(Geometry::CUBE);
   FunctionCoefficient coefficient([](const Vector &point)
   { return point(0) + point(1) + point(2) - 0.9; });
   auto backend = CreateCutQuadratureBackend("algoim-general");
   CutQuadratureGenerator generator(*backend);
   for (int basis : {BasisType::GaussLegendre, BasisType::Positive})
   {
      L2_FECollection collection(4, 3, basis);
      PolynomialLevelSet level_set;
      level_set.Project(coefficient, identity,
                        *collection.FiniteElementForGeometry(Geometry::CUBE));
      CutQuadrature output;
      auto report = generator.Generate(Geometry::CUBE,
                                       CutDomain::Positive(level_set),
                                       8, output);
      INFO(report.message);
      REQUIRE(report.Succeeded());
      REQUIRE(Measure(output.rule) == MFEM_Approx(1.0 - std::pow(0.9, 3)/6));
   }
}

TEST_CASE("ALGOIM cut phases and planes on tensor cells", "[CutQuadrature]")
{
   auto backend = CreateCutQuadratureBackend("algoim-auto");
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
         level_set.Project(coefficient, identity, element);
         real_t phase_sum = 0;
         for (int domain = 0; domain < 3; ++domain)
         {
            const auto cut_domain = domain == 0 ? CutDomain::Positive(level_set) :
                                    (domain == 1 ? CutDomain::Negative(level_set) :
                                     CutDomain::Surface(level_set));
            CutQuadrature output;
            auto report = generator.Generate(geometry, cut_domain, 6, output);
            INFO(report.message);
            REQUIRE(report.Succeeded());
            REQUIRE(output.rule.GetOrder() == 6);
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

TEST_CASE("ALGOIM classifies constants and rejects unsupported requests",
          "[CutQuadrature]")
{
   auto backend = CreateCutQuadratureBackend("algoim-auto");
   CutQuadratureGenerator generator(*backend);
   L2_FECollection collection(0, 2);
   const auto &element = *collection.FiniteElementForGeometry(Geometry::SQUARE);
   Vector values(1);
   PolynomialLevelSet level_set;
   CutQuadrature output;
   for (real_t value : {-1., 0., 1.})
   {
      values = value;
      level_set.Set(element, values);
      const auto positive = CutDomain::Positive(level_set);
      REQUIRE(generator.Generate(Geometry::SQUARE, positive, 2, output).Succeeded());
      REQUIRE(Measure(output.rule) == MFEM_Approx(value > 0 ? 1 : 0));
      REQUIRE(output.region == (value > 0 ? CutQuadrature::Region::Full :
                                CutQuadrature::Region::Empty));
      const auto surface = CutDomain::Surface(level_set);
      auto report = generator.Generate(Geometry::SQUARE, surface, 2, output);
      REQUIRE(report.status == (value == 0 ?
                                CutQuadratureReport::Status::DegenerateGeometry :
                                CutQuadratureReport::Status::Success));
      REQUIRE(output.GetNPoints() == 0);
   }
   const auto positive = CutDomain::Positive(level_set);
   REQUIRE(generator.Generate(Geometry::SQUARE, positive, -1, output).status ==
           CutQuadratureReport::Status::InvalidInput);
   REQUIRE(generator.Generate(Geometry::SQUARE, positive, 1000, output).status ==
           CutQuadratureReport::Status::Unsupported);
   IntegrationRule fixed(1);
   fixed.IntPoint(0).Set2(0.2, 0.3);
   REQUIRE(generator.Generate(Geometry::SQUARE, positive, 4, output,
                              &fixed).status ==
           CutQuadratureReport::Status::Unsupported);
   REQUIRE(generator.Generate(Geometry::TRIANGLE, CutDomain(), 2, output).status ==
           CutQuadratureReport::Status::Unsupported);
   REQUIRE(output.GetNPoints() == 0);
#ifdef MFEM_USE_LAPACK
   REQUIRE(CreateCutQuadratureBackend("moment") != nullptr);
#else
   REQUIRE(CreateCutQuadratureBackend("moment") == nullptr);
#endif
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
         level_set.Project(coefficient, identity, element);
         for (bool surface : {false, true})
         {
            real_t previous_error = 1;
            const int high_order = name == "algoim-general" ? 18 : 28;
            for (int order : {4, high_order})
            {
               const auto domain = surface ? CutDomain::Surface(level_set) :
                                   CutDomain::Positive(level_set);
               CutQuadrature output;
               auto report = generator.Generate(geometry, domain, order, output);
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

TEST_CASE("General ALGOIM integrates interval-capable analytic functions",
          "[CutQuadrature]")
{
   AlgoimFunctionLevelSet<2, SineLevelSet> level_set(SineLevelSet{});
   const auto saved = algoim::Interval<2>::delta();
   algoim::Interval<2>::delta() = 0.123;
   for (const char *name : {"algoim-general", "algoim-auto"})
   {
      auto backend = CreateCutQuadratureBackend(name);
      REQUIRE(backend != nullptr);
      CutQuadratureGenerator generator(*backend);
      CutQuadrature output;
      real_t phases = 0;
      for (bool positive : {false, true})
      {
         const auto domain = positive ? CutDomain::Positive(level_set) :
                             CutDomain::Negative(level_set);
         auto report = generator.Generate(Geometry::SQUARE, domain, 12, output);
         INFO(report.message);
         REQUIRE(report.Succeeded());
         const real_t expected = positive ? std::cos(1.0) : 1 - std::cos(1.0);
         REQUIRE(Measure(output.rule) == MFEM_Approx(expected));
         phases += Measure(output.rule);
         REQUIRE(algoim::Interval<2>::delta(0) == 0.123);
      }
      REQUIRE(phases == MFEM_Approx(1));
      const auto surface = CutDomain::Surface(level_set);
      REQUIRE(generator.Generate(Geometry::SQUARE, surface, 12, output).Succeeded());
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
      REQUIRE(generator.Generate(Geometry::SQUARE, surface, 20, output).status ==
              CutQuadratureReport::Status::Unsupported);
      REQUIRE(output.GetNPoints() == 0);
      AlgoimFunctionLevelSet<2, SineLevelSet> failing(SineLevelSet{true});
      const auto failing_surface = CutDomain::Surface(failing);
      REQUIRE(generator.Generate(Geometry::SQUARE, failing_surface, 12,
                                 output).status ==
              CutQuadratureReport::Status::NumericalFailure);
      REQUIRE(output.GetNPoints() == 0);
      REQUIRE(algoim::Interval<2>::delta(1) == 0.123);
   }
   algoim::Interval<2>::delta() = saved;
}

TEST_CASE("ALGOIM dispatch depends on configured capabilities and active inputs",
          "[CutQuadrature]")
{
   auto automatic = CreateCutQuadratureBackend("algoim-auto");
   auto general = CreateCutQuadratureBackend("algoim-general");
   auto polynomial = CreateCutQuadratureBackend("algoim-polynomial");
   L2_FECollection collection(2, 2);
   PolynomialLevelSet level_set;
   Vector coefficients(9);
   coefficients = 1.;
   level_set.Set(*collection.FiniteElementForGeometry(Geometry::SQUARE),
                 coefficients);
   FunctionLevelSet callback(Geometry::SQUARE,
   [](const IntegrationPoint &point) { return point.x - 0.3; },
   [](const IntegrationPoint &, Vector &gradient) { gradient = 0.; gradient(0) = 1; });
   auto domain = CutDomain::Positive(level_set);
   REQUIRE(general->CheckSupport(Geometry::SQUARE, domain, 20).status ==
           CutQuadratureReport::Status::Unsupported);
#ifdef MFEM_USE_LAPACKE
   REQUIRE(polynomial != nullptr);
   REQUIRE(automatic->CheckSupport(Geometry::SQUARE, domain, 20).Succeeded());
#else
   REQUIRE(polynomial == nullptr);
   REQUIRE(automatic->CheckSupport(Geometry::SQUARE, domain, 20).status ==
           CutQuadratureReport::Status::Unsupported);
#endif
   domain = CutDomain::Positive(callback);
   REQUIRE(automatic->CheckSupport(Geometry::SQUARE, domain, 6).status ==
           CutQuadratureReport::Status::Unsupported);
   domain = CutDomain::Positive(level_set);
   domain.Add(callback, CutDomain::LevelSetSign::Positive);
   REQUIRE(general->CheckSupport(Geometry::SQUARE, domain, 6).status ==
           CutQuadratureReport::Status::Unsupported);
}

TEST_CASE("General ALGOIM evaluates tensor polynomials independently of LAPACKE",
          "[CutQuadrature]")
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
         level_set.Project(coefficient, identity,
                           *collection.FiniteElementForGeometry(geometry));
         auto general = CreateCutQuadratureBackend("algoim-general");
         CutQuadratureGenerator generator(*general);
         CutQuadrature output;
         const auto negative = CutDomain::Negative(level_set);
         auto report = generator.Generate(geometry, negative, 12, output);
         INFO(report.message);
         REQUIRE(report.Succeeded());
         REQUIRE(Measure(output.rule) == MFEM_Approx(1.0/3.0));
         const auto surface = CutDomain::Surface(level_set);
         REQUIRE(generator.Generate(geometry, surface, 12, output).Succeeded());
         const real_t exact = 0.5*std::sqrt(1.64) + std::asinh(0.8)/1.6;
         REQUIRE(Measure(output.rule) == MFEM_Approx(exact));
#ifdef MFEM_USE_LAPACKE
         auto polynomial = CreateCutQuadratureBackend("algoim-polynomial");
         CutQuadratureGenerator other(*polynomial);
         CutQuadrature compared;
         REQUIRE(other.Generate(geometry, surface, 12, compared).Succeeded());
         REQUIRE(Measure(compared.rule) == MFEM_Approx(Measure(output.rule)));
#endif
      }
   }
}

#ifdef MFEM_USE_LAPACKE
TEST_CASE("Polynomial ALGOIM simplifies equivalent sign constraints",
          "[CutQuadrature]")
{
   for (int dimension : {2, 3})
   {
      const auto geometry = dimension == 2 ? Geometry::SQUARE : Geometry::CUBE;
      IsoparametricTransformation identity;
      identity.SetIdentityTransformation(geometry);
      L2_FECollection collection(2, dimension);
      const auto &element = *collection.FiniteElementForGeometry(geometry);
      FunctionCoefficient coefficient([](const Vector &point)
      {
         real_t value = -0.9;
         for (int axis = 0; axis < point.Size(); ++axis) { value += point(axis); }
         return value;
      });
      PolynomialLevelSet first, second;
      first.Project(coefficient, identity, element);
      const real_t expected = dimension == 2 ? 0.405 : 0.1215;
      for (const char *name : {"algoim-auto", "algoim-polynomial"})
      {
         auto backend = CreateCutQuadratureBackend(name);
         CutQuadratureGenerator generator(*backend);
         for (real_t factor : {1., 2., -1., -2.})
         {
            Vector coefficients(first.GetDofs());
            coefficients *= factor;
            second.Set(element, coefficients);
            for (bool contradictory : {false, true})
            {
               const bool negative_second = (factor > 0) != contradictory;
               const auto domain = CutDomain::Negative(first).Add(
                                      second, negative_second ?
                                      CutDomain::LevelSetSign::Negative :
                                      CutDomain::LevelSetSign::Positive);
               CutQuadrature output;
               auto report = generator.Generate(geometry, domain, 8, output);
               INFO(report.message);
               REQUIRE(report.Succeeded());
               REQUIRE(Measure(output.rule) == MFEM_Approx(contradictory ? 0 : expected));
               if (contradictory)
               {
                  REQUIRE(output.region == CutQuadrature::Region::Empty);
                  REQUIRE(output.GetNPoints() == 0);
                  REQUIRE(output.rule.GetOrder() == 8);
               }
               else
               {
                  for (int axis = 0; axis < dimension; ++axis)
                  {
                     real_t moment = 0;
                     for (int index = 0; index < output.GetNPoints(); ++index)
                     {
                        const auto &point = output.rule.IntPoint(index);
                        moment += point.weight * (axis == 0 ? point.x :
                                                  (axis == 1 ? point.y : point.z));
                     }
                     REQUIRE(moment == MFEM_Approx(expected * 0.9 / (dimension + 1)));
                  }
               }
            }
         }
      }
   }
}

TEST_CASE("Polynomial ALGOIM keeps nearby sign constraints distinct",
          "[CutQuadrature]")
{
   L2_FECollection collection(1, 2, BasisType::Positive);
   const auto &element = *collection.FiniteElementForGeometry(Geometry::SQUARE);
   Vector coefficients(4);
   for (int index = 0; index < coefficients.Size(); ++index)
   { coefficients(index) = index % 2 - 0.5; }
   PolynomialLevelSet first, second;
   first.Set(element, coefficients);
   const real_t width = 1e-10;
   coefficients -= width;
   second.Set(element, coefficients);
   auto backend = CreateCutQuadratureBackend("algoim-polynomial");
   CutQuadratureGenerator generator(*backend);
   const auto domain = CutDomain::Positive(first).Add(
                          second, CutDomain::LevelSetSign::Negative);
   CutQuadrature output;
   auto report = generator.Generate(Geometry::SQUARE, domain, 8, output);
   INFO(report.message);
   REQUIRE(report.Succeeded());
   REQUIRE(output.region == CutQuadrature::Region::Cut);
   REQUIRE(std::abs(Measure(output.rule) / width - 1) < 1e-5);

   coefficients += width;
   coefficients -= 1e-15;
   second.Set(element, coefficients);
   report = generator.Generate(Geometry::SQUARE, domain, 8, output);
   REQUIRE(report.status == CutQuadratureReport::Status::NumericalFailure);
   REQUIRE(output.region == CutQuadrature::Region::Unknown);
   REQUIRE(output.GetNPoints() == 0);
}

TEST_CASE("Polynomial ALGOIM rejects unresolved nearly proportional constraints",
          "[CutQuadrature]")
{
   const auto geometry = Geometry::CUBE;
   IsoparametricTransformation identity;
   identity.SetIdentityTransformation(geometry);
   for (int first_degree : {1, 2, 3, 4})
   {
      FunctionCoefficient coefficient([first_degree](const Vector &point)
      {
         return first_degree == 2 ? (point(0) - 0.5) * (point(0) - 0.5) - 0.09 :
                point(0) + point(1) + point(2) - 0.9;
      });
      const int second_degree = first_degree <= 2 ? 4 : (first_degree == 3 ? 3 : 1);
      L2_FECollection first_collection(first_degree, 3);
      L2_FECollection second_collection(second_degree, 3);
      PolynomialLevelSet first, second;
      first.Project(coefficient, identity,
                    *first_collection.FiniteElementForGeometry(geometry));
      second.Project(coefficient, identity,
                     *second_collection.FiniteElementForGeometry(geometry));
      Vector original(second.GetDofs());
      for (real_t factor : {0.1, -0.1, 1.0})
      {
         if (factor == 1 && first_degree == second_degree) { continue; }
         CAPTURE(first_degree, second_degree, factor);
         Vector scaled(original);
         scaled *= factor;
         second.Set(second.GetFE(), scaled);
         for (const char *name : {"algoim-auto", "algoim-polynomial"})
         {
            auto backend = CreateCutQuadratureBackend(name);
            CutQuadratureGenerator generator(*backend);
            auto domain = CutDomain::Negative(first);
            CutQuadrature output;
            REQUIRE(generator.Generate(geometry, domain, 8, output).Succeeded());
            REQUIRE(Measure(output.rule) == MFEM_Approx(first_degree == 2 ? 0.6 : 0.1215));
            for (auto sign :
                 {
                    CutDomain::LevelSetSign::Negative,
                    CutDomain::LevelSetSign::Positive
                 })
            {
               domain = CutDomain::Negative(first).Add(second, sign);
               REQUIRE(backend->CheckSupport(geometry, domain, 8).Succeeded());
               auto report = generator.Generate(geometry, domain, 8, output);
               REQUIRE(report.status == CutQuadratureReport::Status::NumericalFailure);
               REQUIRE(report.message.find("Nearly proportional") != std::string::npos);
               REQUIRE(output.geometry == Geometry::INVALID);
               REQUIRE(output.region == CutQuadrature::Region::Unknown);
               REQUIRE(output.GetNPoints() == 0);
               REQUIRE(output.rule.GetOrder() == -1);
               REQUIRE(output.conormals.SizeK() == 0);
               CutQuadrature direct;
               direct.rule.SetSize(1);
               direct.region = CutQuadrature::Region::Full;
               report = backend->Generate(geometry, domain, 8, direct);
               REQUIRE(report.status == CutQuadratureReport::Status::NumericalFailure);
               REQUIRE(direct.region == CutQuadrature::Region::Unknown);
               REQUIRE(direct.GetNPoints() == 0);
               REQUIRE(direct.rule.GetOrder() == -1);
            }
         }
      }
   }
}

TEST_CASE("Polynomial ALGOIM partitions two level sets and preserves moments",
          "[CutQuadrature]")
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
      PolynomialLevelSet first, second, third;
      first.Project(first_coefficient, identity,
                    *first_collection.FiniteElementForGeometry(geometry));
      second.Project(second_coefficient, identity,
                     *second_collection.FiniteElementForGeometry(geometry));
      third.Set(second.GetFE(), second.GetDofs());
      for (const char *name : {"algoim-auto", "algoim-polynomial"})
      {
         auto backend = CreateCutQuadratureBackend(name);
         CutQuadratureGenerator generator(*backend);
         for (bool reverse : {false, true})
         {
            real_t total = 0;
            for (bool positive_first : {false, true})
               for (bool positive_second : {false, true})
               {
                  const auto first_sign = positive_first ?
                                          CutDomain::LevelSetSign::Positive :
                                          CutDomain::LevelSetSign::Negative;
                  const auto second_sign = positive_second ?
                                           CutDomain::LevelSetSign::Positive :
                                           CutDomain::LevelSetSign::Negative;
                  CutDomain domain;
                  if (reverse)
                  { domain.Add(second, second_sign).Add(first, first_sign); }
                  else
                  { domain.Add(first, first_sign).Add(second, second_sign); }
                  CutQuadrature output;
                  auto report = generator.Generate(geometry, domain, 6, output);
                  INFO(report.message);
                  REQUIRE(report.Succeeded());
                  const real_t measure = (positive_first ? 0.7 : 0.3) * (positive_second ? 0.6 :
                                                                         0.4);
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
         auto domain = CutDomain::Surface(first).Add(
                          second, CutDomain::LevelSetSign::Zero);
         REQUIRE(backend->CheckSupport(geometry, domain, 2).status ==
                 CutQuadratureReport::Status::Unsupported);
         domain = CutDomain::Surface(first).Add(
                     second, CutDomain::LevelSetSign::Positive);
         REQUIRE(backend->CheckSupport(geometry, domain, 2).status ==
                 CutQuadratureReport::Status::Unsupported);
         domain = CutDomain::Positive(first)
                  .Add(second, CutDomain::LevelSetSign::Positive)
                  .Add(third, CutDomain::LevelSetSign::Positive);
         REQUIRE(backend->CheckSupport(geometry, domain, 2).status ==
                 CutQuadratureReport::Status::Unsupported);
      }
   }
}

TEST_CASE("Polynomial ALGOIM handles empty, oblique and curved intersections",
          "[CutQuadrature]")
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
      first.Project(first_coefficient, identity, element);
      second.Project(second_coefficient, identity, element);
      const auto domain = CutDomain::Positive(first).Add(
                             second, CutDomain::LevelSetSign::Positive);
      CutQuadrature output;
      auto report = generator.Generate(Geometry::SQUARE, domain, 28, output);
      INFO(report.message);
      REQUIRE(report.Succeeded());
      const real_t expected[] = {0, 0.52, std::acos(-1.0)*0.32*0.32/2, 0.6, 0};
      const real_t error = std::abs(Measure(output.rule)-expected[shape]);
      REQUIRE(error < (shape == 2 ? 1e-5 : 1e-12));
      if (shape == 2)
      {
         REQUIRE(generator.Generate(Geometry::SQUARE, domain, 4, output).Succeeded());
         REQUIRE(error < std::abs(Measure(output.rule)-expected[shape]));
      }
   }
}

TEST_CASE("Polynomial ALGOIM intersects genuinely three-dimensional cuts",
          "[CutQuadrature]")
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
   first.Project(first_coefficient, identity, element);
   second.Project(second_coefficient, identity, element);
   const auto domain = CutDomain::Negative(first).Add(
                          second, CutDomain::LevelSetSign::Positive);
   CutQuadrature output;
   auto report = generator.Generate(Geometry::CUBE, domain, 10, output);
   INFO(report.message);
   REQUIRE(report.Succeeded());
   REQUIRE(Measure(output.rule) == MFEM_Approx(5.0/36.0));
}
#endif
#else
TEST_CASE("Disabled cut backend remains discoverable as unavailable",
          "[CutQuadrature]")
{
   REQUIRE(CreateCutQuadratureBackend("algoim-auto") == nullptr);
}
#endif
