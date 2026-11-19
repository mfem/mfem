#include "mfem.hpp"
#include "general/backends.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

#ifdef MFEM_USE_MPI
using BenchMesh = ParMesh;
using BenchSpace = ParFiniteElementSpace;
using BenchForm = ParLinearForm;
using BenchGridFunction = ParGridFunction;
using BenchMass = ParBilinearForm;
#else
using BenchMesh = Mesh;
using BenchSpace = FiniteElementSpace;
using BenchForm = LinearForm;
using BenchGridFunction = GridFunction;
using BenchMass = BilinearForm;
#endif

int Rank()
{
#ifdef MFEM_USE_MPI
   return Mpi::WorldRank();
#else
   return 0;
#endif
}

int Ranks()
{
#ifdef MFEM_USE_MPI
   return Mpi::WorldSize();
#else
   return 1;
#endif
}

double GlobalMax(double local)
{
#ifdef MFEM_USE_MPI
   double global;
   MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return global;
#else
   return local;
#endif
}

long long GlobalSum(long long local)
{
#ifdef MFEM_USE_MPI
   long long global;
   MPI_Allreduce(&local, &global, 1, MPI_LONG_LONG_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   return global;
#else
   return local;
#endif
}

real_t Dot(const Vector &left, const Vector &right)
{
#ifdef MFEM_USE_MPI
   return InnerProduct(MPI_COMM_WORLD, left, right);
#else
   return left * right;
#endif
}

void Synchronize()
{
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      MFEM_DEVICE_SYNC;
   }
}

template <typename Operation>
double Measure(Operation operation)
{
   Synchronize();
#ifdef MFEM_USE_MPI
   MPI_Barrier(MPI_COMM_WORLD);
#endif
   const auto start = std::chrono::steady_clock::now();
   operation();
   Synchronize();
   const double seconds = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - start).count();
   return GlobalMax(seconds);
}

struct Settings
{
   const char *geometry = "quad";
   const char *space = "h1";
   const char *ordering = "nodes";
   const char *device = "cpu";
   const char *method = "all";
   const char *csv = "-";
   int cells = 0;
   int order = 2;
   int components = 1;
   int samples = 100;
   int batches = 5;
   int warmup = 10;
   int verification_samples = 4096;
   int seed = 20261115;
   int device_id = 0;

   int Dimension() const
   {
      return std::string(geometry) == "tri" ||
             std::string(geometry) == "quad" ? 2 : 3;
   }

   bool Simplex() const
   {
      return std::string(geometry) == "tri" ||
             std::string(geometry) == "tet";
   }

   bool Valid() const
   {
      const std::string shape(geometry), family(space), layout(ordering);
      const std::string algorithm(method), backend(device);
      return (shape == "quad" || shape == "hex" || shape == "tri" ||
              shape == "tet") && (family == "h1" || family == "l2") &&
             (layout == "nodes" || layout == "vdim") &&
             (algorithm == "all" || algorithm == "legacy" ||
              algorithm == "uncached" || algorithm == "ordinary" ||
              algorithm == "fast") &&
             cells >= 0 && order >= (family == "h1" ? 1 : 0) &&
             order <= (std::numeric_limits<int>::max() - 1) / 2 &&
             components > 0 && samples > 0 && batches > 0 && warmup >= 0 &&
             verification_samples >= 256 && seed > 0 && device_id >= 0 &&
             seed <= std::numeric_limits<int>::max() - (Ranks() - 1) &&
             (algorithm != "legacy" || components == 1) &&
             (backend == "cpu" || algorithm == "all" || algorithm == "fast");
   }
};

Mesh MakeMesh(const Settings &settings, int cells)
{
   if (settings.Dimension() == 2)
   {
      return Mesh::MakeCartesian2D(cells, cells, settings.Simplex() ?
                                  Element::TRIANGLE : Element::QUADRILATERAL);
   }
   return Mesh::MakeCartesian3D(cells, cells, cells, settings.Simplex() ?
                               Element::TETRAHEDRON : Element::HEXAHEDRON);
}

struct Problem
{
   std::unique_ptr<BenchMesh> mesh;
   std::unique_ptr<FiniteElementCollection> collection;
   std::unique_ptr<BenchSpace> space;

   Problem(const Settings &settings, int cells)
   {
      Mesh serial_mesh = MakeMesh(settings, cells);
#ifdef MFEM_USE_MPI
      mesh.reset(new ParMesh(MPI_COMM_WORLD, serial_mesh));
#else
      mesh.reset(new Mesh(std::move(serial_mesh)));
#endif
      if (std::string(settings.space) == "h1")
      {
         collection.reset(new H1_FECollection(settings.order,
                                             settings.Dimension()));
      }
      else
      {
         collection.reset(new L2_FECollection(settings.order,
                                             settings.Dimension(),
                                             BasisType::GaussLegendre,
                                             FiniteElement::VALUE));
      }
      const auto ordering = std::string(settings.ordering) == "nodes" ?
                            Ordering::byNODES : Ordering::byVDIM;
      space.reset(new BenchSpace(mesh.get(), collection.get(),
                                 settings.components, ordering));
   }
};

const IntegrationRule &Rule(const Settings &settings, const Problem &problem)
{
   const auto geometry = problem.space->GetTypicalFE()->GetGeomType();
   return settings.Simplex() ?
          StroudIntRules.Get(geometry, 2 * settings.order) :
          IntRules.Get(geometry, 2 * settings.order);
}

struct SampleMethod
{
   const std::string method;
   std::unique_ptr<QuadratureSpace> quadrature;
   std::unique_ptr<QuadratureFunction> normals;
   QuadratureGaussianSampler sampler;
   BenchForm rhs;
   Vector true_rhs;

   SampleMethod(const Settings &settings, Problem &problem,
                const std::string &method_)
      : method(method_), sampler(std::uint64_t(settings.seed),
                                 std::uint32_t(Rank())),
        rhs(problem.space.get()), true_rhs(problem.space->GetTrueVSize())
   {
      true_rhs.UseDevice(true);
      if (method == "legacy")
      {
#ifdef MFEM_USE_MPI
         auto *integrator = new WhiteGaussianNoiseDomainLFIntegrator(
            MPI_COMM_WORLD, settings.seed);
#else
         auto *integrator = new WhiteGaussianNoiseDomainLFIntegrator(
            settings.seed);
#endif
         integrator->SaveFactors(problem.mesh->GetNE());
         rhs.AddDomainIntegrator(integrator);
      }
      else
      {
         quadrature.reset(new QuadratureSpace(*problem.mesh,
                                             Rule(settings, problem)));
         normals.reset(new QuadratureFunction(*quadrature,
                                             settings.components));
         auto *integrator = new QuadratureWhiteNoiseLFIntegrator(
            *problem.space, *normals);
         integrator->UseBasisCache(method != "uncached");
         rhs.AddDomainIntegrator(integrator);
      }
      rhs.UseFastAssembly(method == "fast");
   }

   void Generate(std::uint64_t sample)
   {
      if (normals) { sampler.Generate(sample, *normals); }
   }

   void AssembleTrue()
   {
#ifdef MFEM_USE_MPI
      rhs.ParallelAssemble(true_rhs);
#else
      const Operator *prolongation = rhs.FESpace()->GetProlongationMatrix();
      if (prolongation) { prolongation->MultTranspose(rhs, true_rhs); }
      else { true_rhs = rhs; }
#endif
   }

   void Draw(std::uint64_t sample)
   {
      Generate(sample);
      rhs.Assemble();
      AssembleTrue();
   }
};

struct Verification
{
   double relative_error = 0.0;
   double mean_z = 0.0;
   double covariance_z = 0.0;
};

bool Verify(const Settings &settings, const std::vector<std::string> &methods,
            std::vector<Verification> &results)
{
   Problem problem(settings, 1);
   SampleMethod fast(settings, problem, "fast");
   SampleMethod ordinary(settings, problem, "ordinary");
   SampleMethod uncached(settings, problem, "uncached");
   double relative_error = 0.0;
   for (int sample = 0; sample < 3; ++sample)
   {
      fast.Draw(std::uint64_t(sample));
      const double scale = GlobalMax(fast.true_rhs.Normlinf());
      const double minimum_scale = std::numeric_limits<real_t>::min();
      const double denominator = std::max(scale, minimum_scale);
      for (SampleMethod *reference : {&ordinary, &uncached})
      {
         reference->Draw(std::uint64_t(sample));
         Vector difference(fast.true_rhs);
         difference -= reference->true_rhs;
         const double error = GlobalMax(difference.Normlinf()) / denominator;
         if (!std::isfinite(error)) { return false; }
         relative_error = std::max(relative_error, error);
      }
   }
   if (relative_error > 5000 * std::numeric_limits<real_t>::epsilon())
   {
      if (Rank() == 0)
      {
         std::cerr << "Cached/uncached/fast mismatch: " << relative_error << '\n';
      }
      return false;
   }

   constexpr int probe_count = 3;
   std::array<Vector, probe_count> probes;
   for (int probe = 0; probe < probe_count; ++probe)
   {
      VectorFunctionCoefficient coefficient(settings.components,
      [=](const Vector &position, Vector &values)
      {
         values.SetSize(settings.components);
         for (int component = 0; component < settings.components; ++component)
         {
            if (probe == 0)
            {
               values[component] = real_t(1.0) / (component + 1);
            }
            else if (probe == 1)
            {
               values[component] = (component % 2 ? -1.0 : 1.0) *
                                   (0.5 + position[0]);
            }
            else
            {
               values[component] = (component + 1) *
                                   (0.25 + position[position.Size() - 1]);
            }
         }
      });
      BenchGridFunction function(problem.space.get());
      function.ProjectCoefficient(coefficient);
      function.GetTrueDofs(probes[probe]);
   }
   BenchMass mass(problem.space.get());
   auto *integrator = new VectorMassIntegrator;
   integrator->SetVDim(settings.components);
   integrator->SetIntRule(&Rule(settings, problem));
   mass.AddDomainIntegrator(integrator);
   mass.Assemble();
   mass.Finalize();
   Array<int> empty;
   OperatorHandle true_mass;
   mass.FormSystemMatrix(empty, true_mass);
   double expected[probe_count][probe_count];
   for (int column = 0; column < probe_count; ++column)
   {
      Vector action(problem.space->GetTrueVSize());
      true_mass->Mult(probes[column], action);
      for (int row = 0; row < probe_count; ++row)
      {
         expected[row][column] = Dot(probes[row], action);
      }
      if (!std::isfinite(expected[column][column]) ||
          expected[column][column] <= 0.0) { return false; }
   }

   for (const std::string &method : methods)
   {
      SampleMethod sampling(settings, problem, method);
      double mean[probe_count] = {};
      double moment[probe_count][probe_count] = {};
      for (int sample = 0; sample < settings.verification_samples; ++sample)
      {
         sampling.Draw(std::uint64_t(sample) + 1000);
         double values[probe_count], delta[probe_count];
         for (int probe = 0; probe < probe_count; ++probe)
         {
            values[probe] = Dot(probes[probe], sampling.true_rhs);
            delta[probe] = values[probe] - mean[probe];
            mean[probe] += delta[probe] / (sample + 1);
         }
         for (int row = 0; row < probe_count; ++row)
         {
            for (int column = 0; column < probe_count; ++column)
            {
               moment[row][column] += delta[row] *
                                      (values[column] - mean[column]);
            }
         }
      }
      Verification verification;
      verification.relative_error = relative_error;
      for (int row = 0; row < probe_count; ++row)
      {
         const double mean_error = std::sqrt(expected[row][row] /
                                             settings.verification_samples);
         const double mean_z = std::abs(mean[row]) / mean_error;
         if (!std::isfinite(mean_z)) { return false; }
         verification.mean_z = std::max(verification.mean_z, mean_z);
         for (int column = 0; column < probe_count; ++column)
         {
            const double covariance = moment[row][column] /
                                      (settings.verification_samples - 1);
            const double variance =
               expected[row][row] * expected[column][column] +
               expected[row][column] * expected[row][column];
            const double standard_error =
               std::sqrt(variance / (settings.verification_samples - 1));
            const double covariance_z =
               std::abs(covariance - expected[row][column]) / standard_error;
            if (!std::isfinite(covariance_z)) { return false; }
            verification.covariance_z = std::max(verification.covariance_z,
                                                covariance_z);
         }
      }
      if (Rank() == 0)
      {
         std::cerr << "Verification " << method << ": relative_error="
                   << relative_error << ", mean_z=" << verification.mean_z
                   << ", covariance_z=" << verification.covariance_z << '\n';
      }
      if (verification.mean_z > 7.0 || verification.covariance_z > 7.0)
      {
         return false;
      }
      results.push_back(verification);
   }
   return true;
}

struct BatchTiming
{
   double median, minimum, maximum;
};

template <typename Operation>
BatchTiming TimeBatches(const Settings &settings, Operation operation)
{
   std::vector<double> times;
   for (int batch = 0; batch < settings.batches; ++batch)
   {
      times.push_back(Measure([&]()
      {
         for (int sample = 0; sample < settings.samples; ++sample)
         {
            operation();
         }
      }) * 1000.0 / settings.samples);
   }
   std::sort(times.begin(), times.end());
   const int middle = settings.batches / 2;
   const double median = settings.batches % 2 ? times[middle] :
                         0.5 * (times[middle - 1] + times[middle]);
   return {median, times.front(), times.back()};
}

void Benchmark(const Settings &settings, const std::string &method,
               const Verification &verification, std::ostream &csv)
{
   Problem problem(settings, settings.cells);
   std::unique_ptr<SampleMethod> sampling;
   const double setup_ms = 1000.0 * Measure([&]()
   {
      sampling.reset(new SampleMethod(settings, problem, method));
      sampling->Draw(0);
   });
   std::uint64_t sample_id = 1;
   for (int sample = 0; sample < settings.warmup; ++sample)
   {
      sampling->Draw(sample_id++);
   }
   const BatchTiming total = TimeBatches(settings, [&]()
   {
      sampling->Draw(sample_id++);
   });
   BatchTiming generation = {};
   if (method != "legacy")
   {
      generation = TimeBatches(settings, [&]()
      {
         sampling->Generate(sample_id++);
      });
   }
   const BatchTiming assembly = TimeBatches(settings, [&]()
   {
      sampling->rhs.Assemble();
   });
   const BatchTiming true_assembly = TimeBatches(settings, [&]()
   {
      sampling->AssembleTrue();
   });
   const double norm_squared = Dot(sampling->true_rhs, sampling->true_rhs);
   MFEM_VERIFY(std::isfinite(norm_squared) && norm_squared > 0.0,
               "The timed sample must have a finite, nonzero norm.");

   const long long elements = problem.mesh->GetNE();
   const long long dofs = problem.space->GetTypicalFE()->GetDof();
   const long long scalar_bytes = sizeof(real_t);
   long long factor_bytes = 0, quadrature_bytes = 0, work_bytes = 0;
   if (method == "legacy")
   {
      factor_bytes = elements * dofs * dofs * scalar_bytes;
   }
   else
   {
      const long long quadrature_size = sampling->quadrature->GetSize();
      quadrature_bytes = quadrature_size * scalar_bytes *
                         (static_cast<long long>(settings.components) + 2);
      if (elements > 0)
      {
         if (method == "uncached") { work_bytes += dofs * scalar_bytes; }
         if (!settings.Simplex())
         {
            const FiniteElement &element = *problem.space->GetTypicalFE();
            const IntegrationRule &rule = Rule(settings, problem);
            const DofToQuad &maps = element.GetDofToQuad(rule, DofToQuad::TENSOR);
            const long long normal_size = sampling->normals->Size();
            const long long first = normal_size / maps.nqpt * maps.ndof;
            work_bytes += scalar_bytes * first;
            if (settings.Dimension() == 3)
            {
               work_bytes += scalar_bytes * (first / maps.nqpt * maps.ndof);
            }
         }
      }
   }
   const long long local_bytes = factor_bytes + quadrature_bytes + work_bytes;
   const double max_rank_bytes = GlobalMax(double(local_bytes));
   factor_bytes = GlobalSum(factor_bytes);
   quadrature_bytes = GlobalSum(quadrature_bytes);
   work_bytes = GlobalSum(work_bytes);
   const long long global_elements = GlobalSum(elements);
   const long long true_dofs = GlobalSum(problem.space->GetTrueVSize());
   if (Rank() == 0)
   {
      csv << std::setprecision(17) << settings.geometry << ',' << settings.space
          << ',' << settings.order << ',' << settings.components << ','
          << settings.ordering << ',' << '"' << settings.device << '"' << ','
          << Ranks() << ',' << global_elements << ',' << true_dofs << ','
          << method << ',' << settings.seed << ',' << settings.warmup << ','
          << settings.samples << ',' << settings.batches << ',' << setup_ms
          << ',' << total.median << ',' << total.minimum << ',' << total.maximum
          << ',';
      if (method != "legacy") { csv << generation.median; }
      csv << ',' << assembly.median << ',' << true_assembly.median << ','
          << 1000.0 / total.median << ',' << factor_bytes << ','
          << quadrature_bytes << ',' << work_bytes << ','
          << factor_bytes + quadrature_bytes + work_bytes << ','
          << max_rank_bytes
          << ',' << settings.verification_samples << ','
          << verification.relative_error << ',' << verification.mean_z << ','
          << verification.covariance_z << '\n';
      csv.flush();
      std::cerr << method << ": setup " << setup_ms << " ms; sample "
                << total.median << " ms [" << total.minimum << ", "
                << total.maximum << "]; " << 1000.0 / total.median
                << " samples/s\n";
   }
}

}

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   Mpi::Init(argc, argv);
   Hypre::Init();
#endif
   out.SetStream(std::cerr);
   Settings settings;
   OptionsParser args(argc, argv);
   args.AddOption(&settings.geometry, "-g", "--geometry",
                  "quad, hex, tri, or tet.");
   args.AddOption(&settings.space, "-s", "--space", "h1 or value-mapped l2.");
   args.AddOption(&settings.cells, "-n", "--cells",
                  "Cartesian cells per axis; 0 selects 64 in 2D, 16 in 3D.");
   args.AddOption(&settings.order, "-o", "--order", "Finite element degree.");
   args.AddOption(&settings.components, "-vd", "--vdim", "Vector dimension.");
   args.AddOption(&settings.ordering, "-ord", "--ordering", "nodes or vdim.");
   args.AddOption(&settings.device, "-d", "--device",
                  "MFEM device configuration.");
   args.AddOption(&settings.device_id, "-id", "--device-id",
                  "Visible device index.");
   args.AddOption(&settings.method, "-m", "--method",
                  "all, legacy (Cholesky), uncached, ordinary (cached), or fast.");
   args.AddOption(&settings.samples, "-ns", "--samples",
                  "Samples per timed batch.");
   args.AddOption(&settings.batches, "-nb", "--batches",
                  "Number of timed batches.");
   args.AddOption(&settings.warmup, "-nw", "--warmup",
                  "Untimed warm-up samples.");
   args.AddOption(&settings.verification_samples, "-nv", "--verify-samples",
                  "Samples for untimed small-mesh verification; at least 256.");
   args.AddOption(&settings.seed, "-seed", "--seed", "Positive random seed.");
   args.AddOption(&settings.csv, "-csv", "--csv",
                  "CSV output path; - means stdout.");
   args.Parse();
   if (!args.Good())
   {
      if (Rank() == 0) { args.PrintUsage(std::cerr); }
      return args.Help() ? 0 : 1;
   }
   if (!settings.Valid())
   {
      if (Rank() == 0)
      {
         std::cerr << "Invalid options. Legacy requires scalar spaces; timed "
                   << "legacy/uncached/ordinary require --device cpu. "
                   << "Counts must be "
                   << "positive (warmup/cells may be zero); "
                   << "verify-samples >= 256. "
                   << "Seed plus rank must fit in int.\n";
      }
      return 1;
   }
   if (settings.cells == 0)
   {
      settings.cells = settings.Dimension() == 2 ? 64 : 16;
   }
   const int dimension = settings.Dimension();
   const double elements = std::pow(double(settings.cells), dimension) *
                           (settings.Simplex() ? (dimension == 2 ? 2 : 6) : 1);
   const double vertices = std::pow(double(settings.cells) + 1.0, dimension);
   if (std::max(elements, vertices) > std::numeric_limits<int>::max())
   {
      if (Rank() == 0)
      {
         std::cerr << "Mesh exceeds the supported element/vertex count.\n";
      }
      return 1;
   }
   Device device(settings.device, settings.device_id);
   if (Rank() == 0) { device.Print(std::cerr); }
   std::vector<std::string> methods;
   if (std::string(settings.method) == "all")
   {
      if (std::string(settings.device) == "cpu")
      {
         if (settings.components == 1) { methods.push_back("legacy"); }
         methods.push_back("uncached");
         methods.push_back("ordinary");
      }
      methods.push_back("fast");
   }
   else { methods.push_back(settings.method); }
   std::vector<Verification> verification;
   if (!Verify(settings, methods, verification))
   {
      if (Rank() == 0)
      {
         std::cerr << "Verification failed; no timings recorded.\n";
      }
      return 2;
   }
   std::ofstream file;
   std::ostream *csv = &std::cout;
   if (Rank() == 0 && std::string(settings.csv) != "-")
   {
      file.open(settings.csv);
      csv = &file;
   }
   if (GlobalMax(Rank() == 0 && !*csv ? 1.0 : 0.0) != 0.0)
   {
      if (Rank() == 0) { std::cerr << "Cannot open CSV output.\n"; }
      return 1;
   }
   if (Rank() == 0)
   {
      *csv << "geometry,space,degree,vdim,ordering,backend,ranks,elements,"
           << "true_dofs,method,seed,warmup,samples,batches,setup_ms,sample_ms,"
           << "sample_min_ms,sample_max_ms,rng_ms,local_rhs_ms,"
           << "true_assembly_ms,"
           << "samples_per_second,factor_bytes_est,quadrature_bytes_est,"
           << "work_bytes_est,sampling_bytes_est,max_rank_sampling_bytes_est,"
           << "verify_samples,relative_error,mean_z,covariance_z\n";
   }
   for (std::size_t method = 0; method < methods.size(); ++method)
   {
      Benchmark(settings, methods[method], verification[method], *csv);
   }
   if (GlobalMax(Rank() == 0 && !*csv ? 1.0 : 0.0) != 0.0)
   {
      if (Rank() == 0) { std::cerr << "Cannot write CSV output.\n"; }
      return 1;
   }
   return 0;
}
