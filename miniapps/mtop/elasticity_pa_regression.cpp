// Regression driver for MFEM's linear isotropic elasticity PA implementation.
//
// Build this program against the current MFEM tree and run it in both supported
// spatial dimensions with -dim 2 and -dim 3.

#include "mfem.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

using namespace mfem;
using namespace std;

namespace
{

constexpr real_t pi =
   static_cast<real_t>(3.141592653589793238462643383279502884L);

real_t MaterialFunction(const Vector &x)
{
   real_t value = static_cast<real_t>(1.45)
                  + static_cast<real_t>(0.19) * sin(2.0*pi*x[0])
                  + static_cast<real_t>(0.13) * cos(3.0*pi*x[1]);
   if (x.Size() == 3)
   {
      value += static_cast<real_t>(0.08) * x[2]
               + static_cast<real_t>(0.05) * sin(pi*x[2]);
   }
   return value;
}

void FillVector(Vector &v, const int seed)
{
   for (int i = 0; i < v.Size(); ++i)
   {
      const real_t k = static_cast<real_t>(i + 1);
      const real_t s = static_cast<real_t>(seed + 1);
      v(i) = sin((static_cast<real_t>(0.31) +
                  static_cast<real_t>(0.017)*s)*k)
             + static_cast<real_t>(0.23)*cos(
                (static_cast<real_t>(0.13) +
                 static_cast<real_t>(0.011)*s)*k)
             + static_cast<real_t>(0.001)*s;
   }
}

struct ErrorMetrics
{
   real_t relative_l2;
   real_t absolute_l2;
   real_t absolute_linf;
};

ErrorMetrics CompareVectors(const Vector &actual, const Vector &expected)
{
   MFEM_VERIFY(actual.Size() == expected.Size(), "Vector size mismatch.");
   Vector difference(actual);
   difference -= expected;

   const real_t absolute_l2 = difference.Norml2();
   const real_t reference_l2 = expected.Norml2();
   const real_t relative_l2 = reference_l2 > 0.0 ?
                              absolute_l2/reference_l2 :
                              (absolute_l2 == 0.0 ? 0.0 :
                               std::numeric_limits<real_t>::infinity());
   return {relative_l2, absolute_l2, difference.Normlinf()};
}

class TestSuite
{
private:
   real_t relative_tolerance;
   real_t absolute_tolerance;
   int checks = 0;
   int failures = 0;

public:
   explicit TestSuite(const real_t relative_tolerance_,
                      const real_t absolute_tolerance_)
      : relative_tolerance(relative_tolerance_),
        absolute_tolerance(absolute_tolerance_) { }

   void Check(const string &name, const Vector &actual, const Vector &expected)
   {
      const ErrorMetrics error = CompareVectors(actual, expected);
      const bool pass = std::isfinite(error.absolute_l2) &&
                        (error.absolute_l2 <= absolute_tolerance ||
                         (std::isfinite(error.relative_l2) &&
                          error.relative_l2 <= relative_tolerance));
      ++checks;
      failures += pass ? 0 : 1;

      cout << "  " << left << setw(57) << name
           << (pass ? "PASS" : "FAIL")
           << "  rel-L2=" << scientific << setprecision(3)
           << error.relative_l2
           << "  abs-L2=" << error.absolute_l2
           << "  abs-Linf(diag)=" << error.absolute_linf << '\n';
   }

   int Finish() const
   {
      cout << "\n" << (failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED")
           << "  (" << checks - failures << "/" << checks << " passed)\n";
      return failures == 0 ? 0 : 2;
   }
};

Mesh MakeMesh(const int dim, const int elements_per_direction)
{
   if (dim == 2)
   {
      return Mesh::MakeCartesian2D(elements_per_direction,
                                   elements_per_direction,
                                   Element::QUADRILATERAL,
                                   1,
                                   static_cast<real_t>(1.2),
                                   static_cast<real_t>(0.9));
   }

   return Mesh::MakeCartesian3D(elements_per_direction,
                                elements_per_direction,
                                elements_per_direction,
                                Element::HEXAHEDRON,
                                static_cast<real_t>(1.2),
                                static_cast<real_t>(0.9),
                                static_cast<real_t>(1.1));
}

void AddElasticityAndMass(BilinearForm &form,
                          Coefficient &material,
                          Coefficient &mass,
                          const real_t q_lambda,
                          const real_t q_mu,
                          const bool elasticity_first)
{
   if (elasticity_first)
   {
      form.AddDomainIntegrator(
         new ElasticityIntegrator(material, q_lambda, q_mu));
      form.AddDomainIntegrator(new VectorMassIntegrator(mass));
   }
   else
   {
      form.AddDomainIntegrator(new VectorMassIntegrator(mass));
      form.AddDomainIntegrator(
         new ElasticityIntegrator(material, q_lambda, q_mu));
   }
}

void TestScaledConstructorActionAndDiagonal(FiniteElementSpace &vector_fes,
                                            Coefficient &material,
                                            const real_t q_lambda,
                                            const real_t q_mu,
                                            TestSuite &tests)
{
   BilinearForm pa_form(&vector_fes);
   pa_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_form.AddDomainIntegrator(
      new ElasticityIntegrator(material, q_lambda, q_mu));
   pa_form.Assemble();

   BilinearForm full_form(&vector_fes);
   full_form.AddDomainIntegrator(
      new ElasticityIntegrator(material, q_lambda, q_mu));
   full_form.Assemble();
   full_form.Finalize();

   Vector x(vector_fes.GetVSize());
   Vector y_pa(vector_fes.GetVSize());
   Vector y_full(vector_fes.GetVSize());
   FillVector(x, 1);
   y_pa = 0.0;
   y_full = 0.0;
   pa_form.Mult(x, y_pa);
   full_form.Mult(x, y_full);
   tests.Check("scaled constructor: PA action vs full assembly", y_pa, y_full);

   Vector diagonal_pa(vector_fes.GetVSize());
   Vector diagonal_full(vector_fes.GetVSize());
   pa_form.AssembleDiagonal(diagonal_pa);
   full_form.SpMat().GetDiag(diagonal_full);
   tests.Check("scaled constructor: PA diagonal vs full assembly",
               diagonal_pa, diagonal_full);
}

void TestDirectDiagonalAccumulation(FiniteElementSpace &vector_fes,
                                    Coefficient &material,
                                    const int dim,
                                    const real_t q_lambda,
                                    const real_t q_mu,
                                    TestSuite &tests)
{
   ElasticityIntegrator integrator(material, q_lambda, q_mu);
   integrator.AssemblePA(vector_fes);

   const int scalar_dofs = vector_fes.GetTypicalFE()->GetDof();
   const int e_size = scalar_dofs * dim * vector_fes.GetNE();

   Vector contribution(e_size);
   contribution = 0.0;
   integrator.AssembleDiagonalPA(contribution);

   Vector accumulated(e_size);
   FillVector(accumulated, 3);
   Vector expected(accumulated);
   expected += contribution;
   integrator.AssembleDiagonalPA(accumulated);

   tests.Check("direct AssembleDiagonalPA adds to a nonzero E-vector",
               accumulated, expected);
}

void TestMultiIntegratorDiagonal(FiniteElementSpace &vector_fes,
                                 Coefficient &material,
                                 const real_t q_lambda,
                                 const real_t q_mu,
                                 TestSuite &tests)
{
   ConstantCoefficient mass_coefficient(static_cast<real_t>(0.37));

   BilinearForm full_form(&vector_fes);
   AddElasticityAndMass(full_form, material, mass_coefficient,
                        q_lambda, q_mu, true);
   full_form.Assemble();
   full_form.Finalize();
   Vector diagonal_full(vector_fes.GetVSize());
   full_form.SpMat().GetDiag(diagonal_full);

   BilinearForm pa_elasticity_first(&vector_fes);
   pa_elasticity_first.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   AddElasticityAndMass(pa_elasticity_first, material, mass_coefficient,
                        q_lambda, q_mu, true);
   pa_elasticity_first.Assemble();
   Vector diagonal_elasticity_first(vector_fes.GetVSize());
   pa_elasticity_first.AssembleDiagonal(diagonal_elasticity_first);

   BilinearForm pa_mass_first(&vector_fes);
   pa_mass_first.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   AddElasticityAndMass(pa_mass_first, material, mass_coefficient,
                        q_lambda, q_mu, false);
   pa_mass_first.Assemble();
   Vector diagonal_mass_first(vector_fes.GetVSize());
   pa_mass_first.AssembleDiagonal(diagonal_mass_first);

   tests.Check("elasticity + vector mass diagonal (elasticity first)",
               diagonal_elasticity_first, diagonal_full);
   tests.Check("elasticity + vector mass diagonal (elasticity second)",
               diagonal_mass_first, diagonal_full);
   tests.Check("multi-integrator diagonal is insertion-order independent",
               diagonal_mass_first, diagonal_elasticity_first);
}

void TestComponentPartialAssembly(FiniteElementSpace &scalar_fes,
                                  FiniteElementSpace &vector_fes,
                                  Coefficient &material,
                                  const int dim,
                                  const real_t q_lambda,
                                  const real_t q_mu,
                                  TestSuite &tests)
{
   BilinearForm full_form(&vector_fes);
   full_form.AddDomainIntegrator(
      new ElasticityIntegrator(material, q_lambda, q_mu));
   full_form.Assemble();
   full_form.Finalize();

   // Deliberately do not call parent.AssemblePA(vector_fes).  The first
   // component assembly must initialize the shared parent state correctly.
   ElasticityIntegrator parent(material, q_lambda, q_mu);

   Vector x_scalar(scalar_fes.GetVSize());
   Vector x_vector(vector_fes.GetVSize());
   Vector y_vector(vector_fes.GetVSize());
   Vector y_block(scalar_fes.GetVSize());
   Vector y_reference(scalar_fes.GetVSize());

   for (int j = 0; j < dim; ++j)
   {
      FillVector(x_scalar, 10 + j);
      x_vector = 0.0;
      for (int k = 0; k < scalar_fes.GetVSize(); ++k)
      {
         x_vector(vector_fes.DofToVDof(k, j)) = x_scalar(k);
      }

      y_vector = 0.0;
      full_form.Mult(x_vector, y_vector);

      for (int i = 0; i < dim; ++i)
      {
         BilinearForm component_form(&scalar_fes);
         component_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         component_form.AddDomainIntegrator(
            new ElasticityComponentIntegrator(parent, i, j));
         component_form.Assemble();

         y_block = 0.0;
         component_form.Mult(x_scalar, y_block);
         for (int k = 0; k < scalar_fes.GetVSize(); ++k)
         {
            y_reference(k) = y_vector(vector_fes.DofToVDof(k, i));
         }

         ostringstream name;
         name << "component PA cold start: block (" << i << ',' << j << ')';
         tests.Check(name.str(), y_block, y_reference);
      }
   }
}

void TestComponentElementAssembly(FiniteElementSpace &scalar_fes,
                                  FiniteElementSpace &vector_fes,
                                  Coefficient &material,
                                  const int dim,
                                  const real_t q_lambda,
                                  const real_t q_mu,
                                  TestSuite &tests)
{
   BilinearForm full_elasticity(&vector_fes);
   full_elasticity.AddDomainIntegrator(
      new ElasticityIntegrator(material, q_lambda, q_mu));
   full_elasticity.Assemble();
   full_elasticity.Finalize();

   ConstantCoefficient mass_coefficient(static_cast<real_t>(0.41));
   BilinearForm full_mass(&scalar_fes);
   full_mass.AddDomainIntegrator(new MassIntegrator(mass_coefficient));
   full_mass.Assemble();
   full_mass.Finalize();

   // Deliberately share one parent across all blocks. The first EA call must
   // initialize it, and later calls must reuse compatible PA data.
   ElasticityIntegrator parent(material, q_lambda, q_mu);

   Vector x_scalar(scalar_fes.GetVSize());
   Vector x_vector(vector_fes.GetVSize());
   Vector y_vector(vector_fes.GetVSize());
   Vector y_block(scalar_fes.GetVSize());
   Vector y_mass(scalar_fes.GetVSize());
   Vector expected_with_mass(scalar_fes.GetVSize());
   Vector y_component_only(scalar_fes.GetVSize());
   Vector y_component_first(scalar_fes.GetVSize());
   Vector y_component_second(scalar_fes.GetVSize());

   for (int j_block = 0; j_block < dim; ++j_block)
   {
      FillVector(x_scalar, 30 + j_block);
      x_vector = 0.0;
      for (int k = 0; k < scalar_fes.GetVSize(); ++k)
      {
         x_vector(vector_fes.DofToVDof(k, j_block)) = x_scalar(k);
      }

      y_vector = 0.0;
      full_elasticity.Mult(x_vector, y_vector);
      full_mass.Mult(x_scalar, y_mass);

      for (int i_block = 0; i_block < dim; ++i_block)
      {
         for (int k = 0; k < scalar_fes.GetVSize(); ++k)
         {
            y_block(k) = y_vector(vector_fes.DofToVDof(k, i_block));
         }
         expected_with_mass = y_block;
         expected_with_mass += y_mass;

         BilinearForm component_only(&scalar_fes);
         component_only.SetAssemblyLevel(AssemblyLevel::ELEMENT);
         component_only.AddDomainIntegrator(
            new ElasticityComponentIntegrator(parent, i_block, j_block));
         component_only.Assemble();
         component_only.Mult(x_scalar, y_component_only);

         ostringstream action_name;
         action_name << "component EA action/layout: block ("
                     << i_block << ',' << j_block << ')';
         tests.Check(action_name.str(), y_component_only, y_block);

         BilinearForm component_first(&scalar_fes);
         component_first.SetAssemblyLevel(AssemblyLevel::ELEMENT);
         component_first.AddDomainIntegrator(
            new ElasticityComponentIntegrator(parent, i_block, j_block));
         component_first.AddDomainIntegrator(new MassIntegrator(mass_coefficient));
         component_first.Assemble();
         component_first.Mult(x_scalar, y_component_first);

         ostringstream first_name;
         first_name << "component EA + mass (component first): block ("
                    << i_block << ',' << j_block << ')';
         tests.Check(first_name.str(), y_component_first, expected_with_mass);

         BilinearForm component_second(&scalar_fes);
         component_second.SetAssemblyLevel(AssemblyLevel::ELEMENT);
         component_second.AddDomainIntegrator(new MassIntegrator(mass_coefficient));
         component_second.AddDomainIntegrator(
            new ElasticityComponentIntegrator(parent, i_block, j_block));
         component_second.Assemble();
         component_second.Mult(x_scalar, y_component_second);

         ostringstream second_name;
         second_name << "component EA + mass (component second): block ("
                     << i_block << ',' << j_block << ')';
         tests.Check(second_name.str(), y_component_second,
                     expected_with_mass);
      }
   }
}


double SecondsSince(const std::chrono::steady_clock::time_point &start)
{
   return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - start).count();
}

void RunPerformanceBenchmark(FiniteElementSpace &vector_fes,
                             Coefficient &material,
                             const int dim,
                             const real_t q_lambda,
                             const real_t q_mu,
                             const int warmup_repetitions,
                             const int apply_repetitions,
                             const int diagonal_repetitions)
{
   cout << "\nElasticity PA performance benchmark\n"
        << "  Applies include BilinearForm restriction/prolongation overhead.\n"
        << "  Tensor-product quads/hexes attempt the fused kernel; unsupported\n"
        << "  D1D/Q1D pairs automatically use the optimized generic fallback.\n";

   MFEM_DEVICE_SYNC;
   const auto setup_start = std::chrono::steady_clock::now();
   BilinearForm pa_form(&vector_fes);
   pa_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_form.AddDomainIntegrator(
      new ElasticityIntegrator(material, q_lambda, q_mu));
   pa_form.Assemble();
   MFEM_DEVICE_SYNC;
   const double setup_seconds = SecondsSince(setup_start);

   Vector x(vector_fes.GetVSize());
   Vector y(vector_fes.GetVSize());
   FillVector(x, 101 + dim);
   x.UseDevice(true);
   y.UseDevice(true);

   for (int i = 0; i < warmup_repetitions; ++i)
   {
      pa_form.Mult(x, y);
   }
   MFEM_DEVICE_SYNC;

   const auto apply_start = std::chrono::steady_clock::now();
   for (int i = 0; i < apply_repetitions; ++i)
   {
      pa_form.Mult(x, y);
   }
   MFEM_DEVICE_SYNC;
   const double apply_seconds = SecondsSince(apply_start);
   const double seconds_per_apply = apply_seconds / apply_repetitions;
   const double mdofs_per_second =
      static_cast<double>(vector_fes.GetVSize()) /
      seconds_per_apply / 1.0e6;
   const real_t output_norm = y.Norml2();

   cout << fixed << setprecision(6)
        << "  PA setup:             " << setup_seconds << " s\n"
        << "  timed applications:   " << apply_repetitions << '\n'
        << "  total apply time:     " << apply_seconds << " s\n"
        << "  time/application:     " << 1.0e3*seconds_per_apply << " ms\n"
        << "  vector-dof throughput:" << setw(12) << setprecision(3)
        << mdofs_per_second << " MDOF/s\n"
        << scientific << setprecision(12)
        << "  final output norm:    " << output_norm << '\n';

   if (diagonal_repetitions > 0)
   {
      Vector diagonal(vector_fes.GetVSize());
      const int diagonal_warmups = std::min(warmup_repetitions, 3);
      for (int i = 0; i < diagonal_warmups; ++i)
      {
         pa_form.AssembleDiagonal(diagonal);
      }
      MFEM_DEVICE_SYNC;

      const auto diagonal_start = std::chrono::steady_clock::now();
      for (int i = 0; i < diagonal_repetitions; ++i)
      {
         pa_form.AssembleDiagonal(diagonal);
      }
      MFEM_DEVICE_SYNC;
      const double diagonal_seconds = SecondsSince(diagonal_start);
      const double seconds_per_diagonal =
         diagonal_seconds / diagonal_repetitions;

      cout << fixed << setprecision(6)
           << "  timed diagonals:      " << diagonal_repetitions << '\n'
           << "  time/diagonal:        "
           << 1.0e3*seconds_per_diagonal << " ms\n"
           << scientific << setprecision(12)
           << "  diagonal norm:        " << diagonal.Norml2() << '\n';
   }
}

} // namespace

int main(int argc, char *argv[])
{
   int dim = 2;
   int order = 2;
   int elements_per_direction = 2;
   const char *device_config = "cpu";
#ifdef MFEM_USE_SINGLE
   real_t tolerance = static_cast<real_t>(5.0e-5);
   real_t absolute_tolerance = static_cast<real_t>(5.0e-6);
#else
   real_t tolerance = static_cast<real_t>(5.0e-10);
   real_t absolute_tolerance = static_cast<real_t>(5.0e-12);
#endif
   real_t q_lambda = static_cast<real_t>(2.3);
   real_t q_mu = static_cast<real_t>(0.7);
   bool run_checks = true;
   int benchmark_repetitions = 0;
   int benchmark_warmups = 5;
   int diagonal_repetitions = 20;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension",
                  "Spatial dimension: 2 or 3.");
   args.AddOption(&order, "-o", "--order",
                  "H1 finite element order.");
   args.AddOption(&elements_per_direction, "-n", "--elements",
                  "Number of Cartesian elements per coordinate direction.");
   args.AddOption(&device_config, "-d", "--device",
                  "MFEM device configuration, e.g. cpu, cuda, hip, debug.");
   args.AddOption(&tolerance, "-tol", "--tolerance",
                  "Relative L2 error tolerance.");
   args.AddOption(&absolute_tolerance, "-atol", "--absolute-tolerance",
                  "Absolute L2 error tolerance.");
   args.AddOption(&q_lambda, "-ql", "--q-lambda",
                  "Scale in lambda = q_lambda*m.");
   args.AddOption(&q_mu, "-qm", "--q-mu",
                  "Scale in mu = q_mu*m.");
   args.AddOption(&run_checks,
                  "-checks", "--run-checks",
                  "-no-checks", "--skip-checks",
                  "Run correctness regression checks.");
   args.AddOption(&benchmark_repetitions, "-b", "--benchmark-repetitions",
                  "Number of timed PA applications; zero disables timing.");
   args.AddOption(&benchmark_warmups, "-bw", "--benchmark-warmups",
                  "Number of untimed warm-up applications.");
   args.AddOption(&diagonal_repetitions, "-bd", "--benchmark-diagonals",
                  "Number of timed diagonal assemblies; zero disables them.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if ((dim != 2 && dim != 3) || order < 1 ||
       elements_per_direction < 1 || tolerance <= 0.0 ||
       absolute_tolerance <= 0.0 ||
       benchmark_repetitions < 0 || benchmark_warmups < 0 ||
       diagonal_repetitions < 0 ||
       (!run_checks && benchmark_repetitions == 0))
   {
      cerr << "Invalid options: use -dim 2 or -dim 3, positive order, "
           << "element count, and tolerance; benchmark counts must be "
           << "nonnegative, and at least checks or timing must be enabled.\n";
      return 1;
   }

   Device device(device_config);
   device.Print();
   args.PrintOptions(cout);

   Mesh mesh = MakeMesh(dim, elements_per_direction);
   H1_FECollection h1_collection(order, dim);
   FiniteElementSpace scalar_fes(&mesh, &h1_collection, 1,
                                 Ordering::byNODES);
   FiniteElementSpace vector_fes(&mesh, &h1_collection, dim,
                                 Ordering::byNODES);
   FunctionCoefficient material(MaterialFunction);

   cout << "\nElasticity PA test configuration\n"
        << "  dimension:   " << dim << '\n'
        << "  elements:    " << mesh.GetNE() << '\n'
        << "  order:       " << order << '\n'
        << "  scalar dofs: " << scalar_fes.GetVSize() << '\n'
        << "  vector dofs: " << vector_fes.GetVSize() << "\n";

   int status = 0;
   if (run_checks)
   {
      cout << "\nElasticity PA regression checks\n\n";
      TestSuite tests(tolerance, absolute_tolerance);
      TestScaledConstructorActionAndDiagonal(vector_fes, material,
                                             q_lambda, q_mu, tests);
      TestDirectDiagonalAccumulation(vector_fes, material, dim,
                                     q_lambda, q_mu, tests);
      TestMultiIntegratorDiagonal(vector_fes, material,
                                  q_lambda, q_mu, tests);
      TestComponentPartialAssembly(scalar_fes, vector_fes, material,
                                   dim, q_lambda, q_mu, tests);
      TestComponentElementAssembly(scalar_fes, vector_fes, material,
                                   dim, q_lambda, q_mu, tests);
      status = tests.Finish();
   }

   if (status == 0 && benchmark_repetitions > 0)
   {
      RunPerformanceBenchmark(vector_fes, material, dim, q_lambda, q_mu,
                              benchmark_warmups, benchmark_repetitions,
                              diagonal_repetitions);
   }

   return status;
}
