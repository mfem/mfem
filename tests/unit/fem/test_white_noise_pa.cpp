#include "mfem.hpp"
#include "unit_tests.hpp"
#include "fem/qfunction_random.hpp"
#include "general/forall.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

using namespace mfem;

namespace
{

const real_t noise_tolerance = 2000 * std::numeric_limits<real_t>::epsilon();

Mesh NoiseMesh(Geometry::Type geometry)
{
   if (geometry == Geometry::TRIANGLE)
   {
      return Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE);
   }
   if (geometry == Geometry::SQUARE)
   {
      return Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
   }
   if (geometry == Geometry::TETRAHEDRON)
   {
      return Mesh::MakeCartesian3D(1, 1, 1, Element::TETRAHEDRON);
   }
   return Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
}

const IntegrationRule &NoiseRule(Geometry::Type geometry, int order)
{
   return geometry == Geometry::TRIANGLE || geometry == Geometry::TETRAHEDRON
          ? StroudIntRules.Get(geometry, order) : IntRules.Get(geometry, order);
}

std::unique_ptr<FiniteElementCollection> NoiseCollection(
   bool discontinuous, int order, int dimension)
{
   if (discontinuous)
   {
      return std::unique_ptr<FiniteElementCollection>(
                new L2_FECollection(order, dimension, BasisType::GaussLegendre,
                                    FiniteElement::VALUE));
   }
   return std::unique_ptr<FiniteElementCollection>(new H1_FECollection(order,
                                                                  dimension));
}

void NoiseTrueVector(const FiniteElementSpace &space, const Vector &local,
                     Vector &result)
{
   result.SetSize(space.GetTrueVSize());
   const Operator *prolongation = space.GetProlongationMatrix();
   if (prolongation) { prolongation->MultTranspose(local, result); }
   else { result = local; }
}

class NoiseReferenceIntegrator : public LinearFormIntegrator
{
   const QuadratureSpace &qspace;
   const QuadratureFunction &normal_q;

public:
   NoiseReferenceIntegrator(const QuadratureSpace &qspace_,
                            const QuadratureFunction &normal_q_)
      : qspace(qspace_), normal_q(normal_q_) { }

   using LinearFormIntegrator::AssembleRHSElementVect;
   void AssembleRHSElementVect(const FiniteElement &element,
                               ElementTransformation &transformation,
                               Vector &element_rhs) override
   {
      const int dofs = element.GetDof();
      const int components = normal_q.GetVDim();
      const int offset = qspace.Offset(transformation.ElementNo);
      const IntegrationRule &rule = qspace.GetIntRule(transformation.ElementNo);
      const real_t *normals = normal_q.HostRead();
      Vector shape(dofs);
      element_rhs.SetSize(dofs * components);
      real_t *result = element_rhs.HostWrite();
      std::fill(result, result + element_rhs.Size(), real_t(0.0));
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
         const IntegrationPoint &location = rule.IntPoint(point);
         transformation.SetIntPoint(&location);
         const real_t weight = std::sqrt(location.weight *
                                         transformation.Weight());
         element.CalcShape(location, shape);
         const int quad = offset + point;
         for (int component = 0; component < components; ++component)
         {
            const real_t value = weight *
                                normals[component + components * quad];
            for (int dof = 0; dof < dofs; ++dof)
            {
               result[dof + dofs * component] += shape[dof] * value;
            }
         }
      }
   }
};

void CheckNoiseCovariance(FiniteElementSpace &space, const IntegrationRule &rule,
                          Array<int> *attributes = nullptr,
                          bool essential_constraints = false,
                          bool use_fast_assembly = true,
                          bool use_basis_cache = true)
{
   QuadratureSpace qspace(*space.GetMesh(), rule);
   QuadratureFunction normal_q(qspace, space.GetVDim());
   LinearForm fast(&space);
   auto *noise = new QuadratureWhiteNoiseLFIntegrator(space, normal_q);
   noise->UseBasisCache(use_basis_cache);
   if (attributes) { fast.AddDomainIntegrator(noise, *attributes); }
   else { fast.AddDomainIntegrator(noise); }
   REQUIRE(fast.SupportsDevice());
   fast.UseFastAssembly(use_fast_assembly);

   BilinearForm mass(&space);
   auto *mass_integrator = new VectorMassIntegrator;
   mass_integrator->SetVDim(space.GetVDim());
   mass_integrator->SetIntRule(&rule);
   if (attributes) { mass.AddDomainIntegrator(mass_integrator, *attributes); }
   else { mass.AddDomainIntegrator(mass_integrator); }
   mass.Assemble();
   mass.Finalize();
   Array<int> empty;
   SparseMatrix true_mass;
   mass.FormSystemMatrix(empty, true_mass);
   DenseMatrix reference;
   true_mass.ToDenseMatrix(reference);

   Array<int> essential;
   if (essential_constraints)
   {
      Array<int> boundary(space.GetMesh()->bdr_attributes.Max());
      boundary = 0;
      boundary[0] = 1;
      space.GetEssentialTrueDofs(boundary, essential);
      for (int entry = 0; entry < essential.Size(); ++entry)
      {
         for (int other = 0; other < reference.Height(); ++other)
         {
            reference(essential[entry], other) = 0.0;
            reference(other, essential[entry]) = 0.0;
         }
      }
   }

   DenseMatrix covariance(space.GetTrueVSize());
   covariance = 0.0;
   Vector sample;
   for (int column = 0; column < normal_q.Size(); ++column)
   {
      normal_q = 0.0;
      normal_q.HostReadWrite()[column] = 1.0;
      fast.Assemble();
      NoiseTrueVector(space, fast, sample);
      sample.SetSubVector(essential, 0.0);
      sample.HostRead();
      AddMultVVt(sample, covariance);
   }
   const real_t scale = std::max(real_t(1.0), reference.MaxMaxNorm());
   covariance -= reference;
   REQUIRE(covariance.MaxMaxNorm() <= noise_tolerance * scale);
}

}

TEST_CASE("Quadrature Gaussian Philox known answer", "[WhiteNoisePA][GPU]")
{
   const std::uint32_t expected[] =
   {0x6627e8d5U, 0xe169c58dU, 0xbc57ac4cU, 0x9b00dbd8U};
   const auto host = internal::Philox4x32(0, 0, 0, 0, 0, 0);
   Array<std::uint32_t> device_words(4);
   auto *output = device_words.Write();
   mfem::forall(1, [=] MFEM_HOST_DEVICE(int entry)
   {
      const auto result = internal::Philox4x32(0, 0, 0, 0, 0, 0);
      for (int word = 0; word < 4; ++word)
      {
         output[4 * entry + word] = result.words[word];
      }
   });
   const auto *actual = device_words.HostRead();
   for (int word = 0; word < 4; ++word)
   {
      REQUIRE(host.words[word] == expected[word]);
      REQUIRE(actual[word] == expected[word]);
   }
}

TEST_CASE("Quadrature Gaussian replay and odd lengths", "[WhiteNoisePA][GPU]")
{
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   QuadratureSpace qspace(mesh, IntRules.Get(Geometry::SQUARE, 0));
   QuadratureFunction normal_q(qspace, 3);
   REQUIRE(normal_q.Size() == 3);
   const std::uint64_t seed = (std::uint64_t(1) << 40) + 73;
   const std::uint64_t sample_id = (std::uint64_t(1) << 42) + 9;
   QuadratureGaussianSampler sampler(seed, 7);
   sampler.Generate(sample_id, normal_q);
   Vector saved(normal_q);
   sampler.Generate(sample_id + 1, normal_q);
   Vector difference(normal_q);
   difference -= saved;
   REQUIRE(difference.Normlinf() > 0.0);
   sampler.Generate(sample_id, normal_q);
   difference = normal_q;
   difference -= saved;
   REQUIRE(difference.Normlinf() == 0.0);
   sampler.Generate(std::uint32_t(sample_id), normal_q);
   difference = normal_q;
   difference -= saved;
   REQUIRE(difference.Normlinf() > 0.0);
   QuadratureGaussianSampler(std::uint32_t(seed), 7).Generate(sample_id, normal_q);
   difference = normal_q;
   difference -= saved;
   REQUIRE(difference.Normlinf() > 0.0);
   QuadratureGaussianSampler(seed, 8).Generate(sample_id, normal_q);
   difference = normal_q;
   difference -= saved;
   REQUIRE(difference.Normlinf() > 0.0);
   QuadratureGaussianSampler(seed + 1, 7).Generate(sample_id, normal_q);
   difference = normal_q;
   difference -= saved;
   REQUIRE(difference.Normlinf() > 0.0);
   const real_t *values = saved.HostRead();
   for (int entry = 0; entry < saved.Size(); ++entry)
   {
      REQUIRE(std::isfinite(values[entry]));
   }
}

TEST_CASE("White-noise fast and ordinary assembly", "[WhiteNoisePA][GPU]")
{
   const auto geometry = GENERATE(Geometry::TRIANGLE, Geometry::SQUARE,
                                  Geometry::TETRAHEDRON, Geometry::CUBE);
   const bool discontinuous = GENERATE(false, true);
   const int order = GENERATE(1, 2, 3, 4, 5, 6);
   const int components = GENERATE(1, 3);
   const auto ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);
   CAPTURE(geometry, discontinuous, order, components, ordering);
   Mesh mesh = NoiseMesh(geometry);
   auto collection = NoiseCollection(discontinuous, order, mesh.Dimension());
   FiniteElementSpace space(&mesh, collection.get(), components, ordering);
   QuadratureSpace qspace(mesh, NoiseRule(geometry, 2 * order));
   QuadratureFunction normal_q(qspace, components);
   QuadratureGaussianSampler sampler(731, 0);
   LinearForm fast(&space), ordinary(&space), reference(&space);
   fast.AddDomainIntegrator(
      new QuadratureWhiteNoiseLFIntegrator(space, normal_q));
   auto *ordinary_noise = new QuadratureWhiteNoiseLFIntegrator(space, normal_q);
   ordinary.AddDomainIntegrator(ordinary_noise);
   reference.AddDomainIntegrator(new NoiseReferenceIntegrator(qspace, normal_q));
   REQUIRE(fast.SupportsDevice());
   fast.UseFastAssembly(true);
   ordinary.UseFastAssembly(false);
   for (std::uint64_t sample_id = 0; sample_id < 3; ++sample_id)
   {
      sampler.Generate(sample_id, normal_q);
      ordinary_noise->UseBasisCache(sample_id != 1);
      fast.Assemble();
      ordinary.Assemble();
      reference.Assemble();
      Vector difference(fast);
      difference -= ordinary;
      REQUIRE(difference.Normlinf() <= noise_tolerance *
              std::max(real_t(1.0), ordinary.Normlinf()));
      difference = ordinary;
      difference -= reference;
      REQUIRE(difference.Normlinf() <= noise_tolerance *
              std::max(real_t(1.0), reference.Normlinf()));
   }
}

TEST_CASE("White-noise deterministic vector covariance", "[WhiteNoisePA][GPU]")
{
   const bool use_fast_assembly = GENERATE(false, true);
   const bool use_basis_cache = GENERATE(false, true);
   const auto geometry = GENERATE(Geometry::TRIANGLE, Geometry::SQUARE,
                                  Geometry::TETRAHEDRON, Geometry::CUBE);
   const bool discontinuous = GENERATE(false, true);
   const int components = GENERATE(1, 3);
   const auto ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);
   CAPTURE(geometry, discontinuous, components, ordering);
   Mesh mesh = NoiseMesh(geometry);
   auto collection = NoiseCollection(discontinuous, 1, mesh.Dimension());
   FiniteElementSpace space(&mesh, collection.get(), components, ordering);
   CheckNoiseCovariance(space, NoiseRule(geometry, 2), nullptr, false,
                        use_fast_assembly, use_basis_cache);
}

TEST_CASE("White-noise curved geometry and refreshed weights",
          "[WhiteNoisePA][GPU]")
{
   const bool use_fast_assembly = GENERATE(false, true);
   const bool use_basis_cache = GENERATE(false, true);
   const auto geometry = GENERATE(Geometry::TRIANGLE, Geometry::SQUARE,
                                  Geometry::TETRAHEDRON, Geometry::CUBE);
   Mesh mesh = NoiseMesh(geometry);
   mesh.SetCurvature(2);
   mesh.Transform([](const Vector &input, Vector &output)
   {
      output = input;
      output[0] += 0.1 * input[0] * input[1];
   });
   H1_FECollection collection(2, mesh.Dimension());
   FiniteElementSpace space(&mesh, &collection, 2);
   const IntegrationRule &rule = NoiseRule(geometry, 6);
   QuadratureSpace qspace(mesh, rule);
   QuadratureFunction normal_q(qspace, 2);
   QuadratureGaussianSampler(129).Generate(4, normal_q);
   LinearForm existing(&space);
   auto *noise = new QuadratureWhiteNoiseLFIntegrator(space, normal_q);
   noise->UseBasisCache(use_basis_cache);
   existing.AddDomainIntegrator(noise);
   existing.UseFastAssembly(use_fast_assembly);
   existing.Assemble();
   Vector before(existing);
   mesh.Transform([](const Vector &input, Vector &output)
   {
      output = input;
      output[0] *= 2.0;
   });
   existing.Assemble();
   before *= std::sqrt(real_t(2.0));
   before -= existing;
   REQUIRE(before.Normlinf() <= noise_tolerance);
   CheckNoiseCovariance(space, rule, nullptr, false, use_fast_assembly,
                        use_basis_cache);
}

TEST_CASE("White-noise positive simplex bases", "[WhiteNoisePA][GPU]")
{
   const bool use_fast_assembly = GENERATE(false, true);
   const bool use_basis_cache = GENERATE(false, true);
   const auto geometry = GENERATE(Geometry::TRIANGLE, Geometry::TETRAHEDRON);
   Mesh mesh = NoiseMesh(geometry);
   H1_FECollection collection(2, mesh.Dimension(), BasisType::Positive);
   FiniteElementSpace space(&mesh, &collection, 2, Ordering::byVDIM);
   CheckNoiseCovariance(space, NoiseRule(geometry, 4), nullptr, false,
                        use_fast_assembly, use_basis_cache);
}

TEST_CASE("White-noise combines with deterministic tensor loads",
          "[WhiteNoisePA][GPU]")
{
   const auto geometry = GENERATE(Geometry::SQUARE, Geometry::CUBE);
   Mesh mesh = NoiseMesh(geometry);
   H1_FECollection collection(2, mesh.Dimension());
   FiniteElementSpace space(&mesh, &collection, 3);
   const IntegrationRule &rule = NoiseRule(geometry, 4);
   QuadratureSpace qspace(mesh, rule);
   QuadratureFunction normal_q(qspace, 3);
   QuadratureGaussianSampler(93).Generate(11, normal_q);
   Vector constant(3);
   constant = 2.0;
   VectorConstantCoefficient coefficient(constant);
   LinearForm combined(&space), noise(&space), deterministic(&space);
   combined.AddDomainIntegrator(
      new QuadratureWhiteNoiseLFIntegrator(space, normal_q));
   combined.AddDomainIntegrator(new VectorDomainLFIntegrator(coefficient));
   noise.AddDomainIntegrator(
      new QuadratureWhiteNoiseLFIntegrator(space, normal_q));
   deterministic.AddDomainIntegrator(new VectorDomainLFIntegrator(coefficient));
   combined.UseFastAssembly(true);
   combined.Assemble();
   noise.Assemble();
   deterministic.Assemble();
   Vector difference(combined);
   difference -= noise;
   difference -= deterministic;
   REQUIRE(difference.Normlinf() <= noise_tolerance);
}

TEST_CASE("White-noise constrained and marked covariance", "[WhiteNoisePA][GPU]")
{
   const bool use_fast_assembly = GENERATE(false, true);
   const bool use_basis_cache = GENERATE(false, true);
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   mesh.EnsureNCMesh();
   Array<int> refine(1);
   refine[0] = 0;
   mesh.GeneralRefinement(refine, 1, 0);
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      mesh.GetElement(element)->SetAttribute(1 + element % 2);
   }
   mesh.SetAttributes();
   H1_FECollection collection(2, 2);
   FiniteElementSpace space(&mesh, &collection, 3, Ordering::byVDIM);
   Array<int> attributes(2);
   attributes[0] = 1;
   attributes[1] = 0;
   CheckNoiseCovariance(space, NoiseRule(Geometry::SQUARE, 4), &attributes, true,
                        use_fast_assembly, use_basis_cache);
}

TEST_CASE("White-noise positive underintegration", "[WhiteNoisePA][GPU]")
{
   const bool use_fast_assembly = GENERATE(false, true);
   const bool use_basis_cache = GENERATE(false, true);
   const auto geometry = GENERATE(Geometry::TRIANGLE, Geometry::SQUARE,
                                  Geometry::TETRAHEDRON, Geometry::CUBE);
   Mesh mesh = NoiseMesh(geometry);
   H1_FECollection collection(2, mesh.Dimension());
   FiniteElementSpace space(&mesh, &collection, 2);
   CheckNoiseCovariance(space, NoiseRule(geometry, 0), nullptr, false,
                        use_fast_assembly, use_basis_cache);
}

TEST_CASE("White-noise device capability preserves legacy limits",
          "[WhiteNoisePA][GPU]")
{
   Mesh mesh = NoiseMesh(Geometry::TRIANGLE);
   H1_FECollection collection(1, 2);
   FiniteElementSpace space(&mesh, &collection);
   QuadratureSpace qspace(mesh, NoiseRule(Geometry::TRIANGLE, 2));
   QuadratureFunction normal_q(qspace);
   ConstantCoefficient coefficient(1.0);
   LinearForm existing(&space);
   existing.AddDomainIntegrator(new DomainLFIntegrator(coefficient));
   REQUIRE_FALSE(existing.SupportsDevice());
   LinearForm mixed(&space);
   mixed.AddDomainIntegrator(
      new QuadratureWhiteNoiseLFIntegrator(space, normal_q));
   mixed.AddDomainIntegrator(new DomainLFIntegrator(coefficient));
   REQUIRE_FALSE(mixed.SupportsDevice());
   LinearForm boundary(&space);
   boundary.AddDomainIntegrator(
      new QuadratureWhiteNoiseLFIntegrator(space, normal_q));
   boundary.AddBoundaryIntegrator(new BoundaryLFIntegrator(coefficient));
   REQUIRE_FALSE(boundary.SupportsDevice());
}

TEST_CASE("Positive simplex quadrature integrates monomials", "[WhiteNoisePA]")
{
   const auto geometry = GENERATE(Geometry::TRIANGLE, Geometry::TETRAHEDRON);
   const int order = GENERATE(0, 2, 4, 6);
   const int dimension = geometry == Geometry::TRIANGLE ? 2 : 3;
   const IntegrationRule &rule = NoiseRule(geometry, order);
   for (int power_x = 0; power_x <= order; ++power_x)
   {
      for (int power_y = 0; power_y <= order - power_x; ++power_y)
      {
         const int max_z = dimension == 2 ? 0 : order - power_x - power_y;
         for (int power_z = 0; power_z <= max_z; ++power_z)
         {
            double integral = 0.0;
            for (int point = 0; point < rule.GetNPoints(); ++point)
            {
               const IntegrationPoint &location = rule.IntPoint(point);
               REQUIRE(location.weight > 0.0);
               integral += location.weight * std::pow(location.x, power_x) *
                           std::pow(location.y, power_y) *
                           (dimension == 3 ? std::pow(location.z, power_z) : 1.0);
            }
            const double exact = std::tgamma(power_x + 1.0) *
                                 std::tgamma(power_y + 1.0) *
                                 std::tgamma(power_z + 1.0) /
                                 std::tgamma(power_x + power_y + power_z +
                                             dimension + 1.0);
            REQUIRE(std::abs(integral - exact) <= noise_tolerance);
         }
      }
   }
}

TEST_CASE("White-noise projected Gaussian moments",
          "[WhiteNoisePA][GPU][Statistical]")
{
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   H1_FECollection collection(1, 2);
   FiniteElementSpace space(&mesh, &collection, 2, Ordering::byVDIM);
   QuadratureSpace qspace(mesh, NoiseRule(Geometry::SQUARE, 2));
   QuadratureFunction normal_q(qspace, 2);
   QuadratureGaussianSampler sampler(923);
   LinearForm rhs(&space);
   rhs.AddDomainIntegrator(
      new QuadratureWhiteNoiseLFIntegrator(space, normal_q));
   rhs.UseFastAssembly(true);
   const int samples = 2048;
   double mean[2] = {0.0, 0.0};
   double second[2] = {0.0, 0.0};
   double cross = 0.0;
   for (int sample = 0; sample < samples; ++sample)
   {
      sampler.Generate(sample, normal_q);
      rhs.Assemble();
      const real_t *values = rhs.HostRead();
      double projection[2] = {0.0, 0.0};
      for (int component = 0; component < 2; ++component)
      {
         for (int dof = 0; dof < space.GetNDofs(); ++dof)
         {
            projection[component] += values[space.DofToVDof(dof, component)];
         }
         mean[component] += projection[component];
         second[component] += projection[component] * projection[component];
      }
      cross += projection[0] * projection[1];
   }
   for (int component = 0; component < 2; ++component)
   {
      REQUIRE(std::abs(mean[component] / samples) < 6.0 / std::sqrt(samples));
      REQUIRE(std::abs(second[component] / samples - 1.0) <
              6.0 * std::sqrt(2.0 / samples));
   }
   REQUIRE(std::abs(cross / samples) < 6.0 / std::sqrt(samples));
}

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("White-noise rejects invalid inputs and stale bindings",
          "[WhiteNoisePA][GPU]")
{
   Mesh mesh = NoiseMesh(Geometry::SQUARE);
   H1_FECollection collection(1, 2);
   FiniteElementSpace space(&mesh, &collection, 2);
   const IntegrationRule &rule = NoiseRule(Geometry::SQUARE, 2);
   QuadratureSpace qspace(mesh, rule);
   QuadratureFunction normal_q(qspace, 2);
   QuadratureWhiteNoiseLFIntegrator valid(space, normal_q);
   REQUIRE_THROWS_AS(valid.SetIntRule(&rule), ErrorException);
   REQUIRE_NOTHROW(valid.SetIntRule(nullptr));

   SECTION("vector dimension")
   {
      QuadratureFunction wrong(qspace, 1);
      REQUIRE_THROWS_AS(QuadratureWhiteNoiseLFIntegrator(space, wrong),
                        ErrorException);
   }
   SECTION("integral mapping")
   {
      L2_FECollection integral_collection(1, 2, BasisType::GaussLegendre,
                                           FiniteElement::INTEGRAL);
      FiniteElementSpace integral_space(&mesh, &integral_collection, 2);
      REQUIRE_THROWS_AS(QuadratureWhiteNoiseLFIntegrator(integral_space, normal_q),
                        ErrorException);
   }
   SECTION("negative quadrature weight")
   {
      IntegrationRule invalid(rule);
      invalid.IntPoint(0).weight = -invalid.IntPoint(0).weight;
      QuadratureSpace bad_space(mesh, invalid);
      QuadratureFunction bad_normal(bad_space, 2);
      bad_normal = 0.0;
      LinearForm rhs(&space);
      rhs.AddDomainIntegrator(
         new QuadratureWhiteNoiseLFIntegrator(space, bad_normal));
      REQUIRE_THROWS_AS(rhs.Assemble(), ErrorException);
   }
   SECTION("incompatible tensor points")
   {
      IntegrationRule invalid(rule);
      invalid.IntPoint(1).y += 0.01;
      QuadratureSpace bad_space(mesh, invalid);
      QuadratureFunction bad_normal(bad_space, 2);
      bad_normal = 0.0;
      LinearForm rhs(&space);
      rhs.AddDomainIntegrator(
         new QuadratureWhiteNoiseLFIntegrator(space, bad_normal));
      rhs.UseFastAssembly(true);
      REQUIRE_THROWS_AS(rhs.Assemble(), ErrorException);
   }
   SECTION("nonfinite quadrature point")
   {
      IntegrationRule invalid(rule);
      invalid.IntPoint(0).x = std::numeric_limits<real_t>::quiet_NaN();
      QuadratureSpace bad_space(mesh, invalid);
      QuadratureFunction bad_normal(bad_space, 2);
      bad_normal = 0.0;
      LinearForm rhs(&space);
      rhs.AddDomainIntegrator(
         new QuadratureWhiteNoiseLFIntegrator(space, bad_normal));
      REQUIRE_THROWS_AS(rhs.Assemble(), ErrorException);
   }
   SECTION("stale vector dimension")
   {
      normal_q.SetVDim(1);
      REQUIRE_FALSE(valid.SupportsDevice(space));
      Vector output;
      REQUIRE_THROWS_AS(valid.AssembleRHSElementVect(*space.GetFE(0),
                        *space.GetElementTransformation(0), output), ErrorException);
   }
   SECTION("stale mesh")
   {
      mesh.UniformRefinement();
      space.Update();
      REQUIRE_FALSE(valid.SupportsDevice(space));
      Vector output;
      REQUIRE_THROWS_AS(valid.AssembleRHSElementVect(*space.GetFE(0),
                        *space.GetElementTransformation(0), output), ErrorException);
   }
}
#endif

#ifdef MFEM_USE_MPI
namespace
{

bool NoiseAllRanks(bool condition)
{
   const int local = condition ? 1 : 0;
   int global;
   MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
   return global != 0;
}

}

TEST_CASE("Parallel white-noise covariance action and replay",
          "[WhiteNoisePA][Parallel][GPU]")
{
   const bool use_basis_cache = GENERATE(false, true);
   const auto geometry = GENERATE(Geometry::TRIANGLE, Geometry::SQUARE,
                                  Geometry::TETRAHEDRON, Geometry::CUBE);
   const bool discontinuous = GENERATE(false, true);
   const auto ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);
   CAPTURE(geometry, discontinuous, ordering);
   Mesh serial_mesh = NoiseMesh(geometry);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   auto collection = NoiseCollection(discontinuous, 1, mesh.Dimension());
   ParFiniteElementSpace space(&mesh, collection.get(), 2, ordering);
   const IntegrationRule &rule = NoiseRule(geometry, 2);
   QuadratureSpace qspace(mesh, rule);
   QuadratureFunction normal_q(qspace, 2);
   QuadratureGaussianSampler sampler(918, std::uint32_t(space.GetMyRank()));
   ParLinearForm fast(&space), ordinary(&space);
   fast.AddDomainIntegrator(
      new QuadratureWhiteNoiseLFIntegrator(space, normal_q));
   auto *ordinary_noise = new QuadratureWhiteNoiseLFIntegrator(space, normal_q);
   ordinary_noise->UseBasisCache(use_basis_cache);
   ordinary.AddDomainIntegrator(ordinary_noise);
   REQUIRE(NoiseAllRanks(fast.SupportsDevice()));
   fast.UseFastAssembly(true);
   sampler.Generate(3, normal_q);
   fast.Assemble();
   Vector initial(space.GetTrueVSize()), sample(space.GetTrueVSize());
   fast.ParallelAssemble(initial);
   ordinary.Assemble();
   ordinary.ParallelAssemble(sample);
   sample -= initial;
   REQUIRE(NoiseAllRanks(sample.Normlinf() <= noise_tolerance));
   sampler.Generate(4, normal_q);
   sampler.Generate(3, normal_q);
   fast.Assemble();
   fast.ParallelAssemble(sample);
   sample -= initial;
   REQUIRE(NoiseAllRanks(sample.Normlinf() <= noise_tolerance));

   ParBilinearForm mass(&space);
   auto *mass_integrator = new VectorMassIntegrator;
   mass_integrator->SetVDim(2);
   mass_integrator->SetIntRule(&rule);
   mass.AddDomainIntegrator(mass_integrator);
   mass.Assemble();
   mass.Finalize();
   Array<int> empty;
   HypreParMatrix true_mass;
   mass.FormSystemMatrix(empty, true_mass);
   Vector probe(space.GetTrueVSize());
   real_t *probe_values = probe.HostWrite();
   for (int entry = 0; entry < probe.Size(); ++entry)
   {
      probe_values[entry] = std::sin(real_t(1 + entry + 7 * space.GetMyRank()));
   }
   Vector reference(probe.Size()), covariance_action(probe.Size());
   true_mass.Mult(probe, reference);
   covariance_action = 0.0;
   Array<int> local_sizes(Mpi::WorldSize());
   const int local_size = normal_q.Size();
   MPI_Allgather(&local_size, 1, MPI_INT, local_sizes.GetData(), 1, MPI_INT,
                 MPI_COMM_WORLD);
   for (int owner = 0; owner < local_sizes.Size(); ++owner)
   {
      for (int column = 0; column < local_sizes[owner]; ++column)
      {
         normal_q = 0.0;
         if (space.GetMyRank() == owner)
         {
            normal_q.HostReadWrite()[column] = 1.0;
         }
         fast.Assemble();
         fast.ParallelAssemble(sample);
         const real_t projection = InnerProduct(MPI_COMM_WORLD, probe, sample);
         covariance_action.Add(projection, sample);
      }
   }
   covariance_action -= reference;
   REQUIRE(NoiseAllRanks(covariance_action.Normlinf() <= noise_tolerance));
}

TEST_CASE("Parallel white-noise empty ranks", "[WhiteNoisePA][Parallel][GPU]")
{
   const bool use_basis_cache = GENERATE(false, true);
   const auto geometry = GENERATE(Geometry::TRIANGLE, Geometry::SQUARE,
                                  Geometry::TETRAHEDRON, Geometry::CUBE);
   const bool discontinuous = GENERATE(false, true);
   const auto ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);
   const int components = GENERATE(1, 3);
   const bool use_fast_assembly = GENERATE(false, true);
   CAPTURE(geometry, discontinuous, ordering, components, use_fast_assembly);
   Mesh serial_mesh = NoiseMesh(geometry);
   Array<int> partition(serial_mesh.GetNE());
   partition = 0;
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh, partition.GetData());
   auto collection = NoiseCollection(discontinuous, 1, mesh.Dimension());
   ParFiniteElementSpace space(&mesh, collection.get(), components, ordering);
   QuadratureSpace qspace(mesh, NoiseRule(geometry, 2));
   QuadratureFunction normal_q(qspace, components);
   QuadratureGaussianSampler sampler(71, std::uint32_t(space.GetMyRank()));
   sampler.Generate(0, normal_q);
   REQUIRE(NoiseAllRanks(mesh.GetNodes() == nullptr));
   ParLinearForm rhs(&space);
   auto *noise = new QuadratureWhiteNoiseLFIntegrator(space, normal_q);
   noise->UseBasisCache(use_basis_cache);
   rhs.AddDomainIntegrator(noise);
   REQUIRE(NoiseAllRanks(rhs.SupportsDevice()));
   rhs.UseFastAssembly(use_fast_assembly);
   rhs.Assemble();
   Vector true_rhs(space.GetTrueVSize());
   rhs.ParallelAssemble(true_rhs);
   REQUIRE(InnerProduct(MPI_COMM_WORLD, true_rhs, true_rhs) > 0.0);

   QuadratureFunction coefficient_q(qspace, components);
   const real_t *weights = qspace.GetWeights().HostRead();
   const real_t *normals = normal_q.HostRead();
   real_t *values = coefficient_q.HostWrite();
   for (int entry = 0; entry < coefficient_q.Size(); ++entry)
   {
      values[entry] = normals[entry] / std::sqrt(weights[entry / components]);
   }
   VectorQuadratureFunctionCoefficient coefficient(coefficient_q);
   ParLinearForm reference(&space);
   reference.AddDomainIntegrator(new VectorQuadratureLFIntegrator(coefficient));
   reference.Assemble();
   Vector true_reference(space.GetTrueVSize());
   reference.ParallelAssemble(true_reference);
   true_rhs -= true_reference;
   REQUIRE(NoiseAllRanks(true_rhs.Normlinf() <= noise_tolerance));

   sampler.Generate(1, normal_q);
   sampler.Generate(0, normal_q);
   rhs.Assemble();
   rhs.ParallelAssemble(true_rhs);
   true_rhs -= true_reference;
   REQUIRE(NoiseAllRanks(true_rhs.Normlinf() <= noise_tolerance));

   bool valid_empty_rank = true;
   if (space.GetMyRank() != 0)
   {
      valid_empty_rank = mesh.GetNE() == 0 && normal_q.Size() == 0 &&
                         rhs.Size() == 0 && true_rhs.Size() == 0;
   }
   REQUIRE(NoiseAllRanks(valid_empty_rank));
}
#endif
