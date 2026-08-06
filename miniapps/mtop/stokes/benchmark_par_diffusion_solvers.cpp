/**
 * @file benchmark_par_diffusion_solvers.cpp
 * @brief Time assembled ParDiffusionSolver against ParDiffusionSolverPA.
 *
 * Setup and repeated Mult() times use identical finite-element data. Times are
 * maximum wall times over MPI ranks. An untimed warm-up solve precedes timing,
 * which removes most one-time GPU kernel initialization from the measurement.
 */

#include "mfem.hpp"
#include "par_diffusion_solver.hpp"
#include "par_diffusion_solver_pa.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

using namespace mfem;
using namespace std;

namespace
{

/// Evaluate the benchmark load coefficient.
real_t Source(const Vector &x)
{
   real_t value = 1.0;
   for (int d = 0; d < x.Size(); d++) { value += x(d); }
   return value;
}

/// Evaluate the positive benchmark diffusion coefficient.
real_t Diffusion(const Vector &x)
{
   real_t value = 1.0;
   for (int d = 0; d < x.Size(); d++) { value += 0.1*x(d); }
   return value;
}

/// Add zero Dirichlet conditions on every boundary attribute.
template <typename SolverType>
void AddZeroBoundaryConditions(const ParMesh &mesh, SolverType &solver)
{
   for (int i = 0; i < mesh.bdr_attributes.Size(); i++)
   {
      solver.AddBoundaryCondition(mesh.bdr_attributes[i], 0.0);
   }
}

/// Reduce a local wall time to the maximum over all ranks.
double GlobalMaximumTime(MPI_Comm comm, double local_time)
{
   double global_time = 0.0;
   MPI_Allreduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, comm);
   return global_time;
}

/// Warm up and time repeated collective solver applications.
template <typename SolverType>
double TimeSolves(SolverType &solver, const Vector &rhs, int repetitions,
                  MPI_Comm comm, Vector &solution)
{
   solver.Mult(rhs, solution);
   MPI_Barrier(comm);
   StopWatch watch;
   watch.Start();
   for (int i = 0; i < repetitions; i++) { solver.Mult(rhs, solution); }
   watch.Stop();
   return GlobalMaximumTime(comm, watch.RealTime());
}

/// Write assembled, PA, and difference fields to one ParaView collection.
void SaveParaView(ParMesh &mesh,
                  ParFiniteElementSpace &fes,
                  const Vector &assembled,
                  const Vector &pa,
                  const char *prefix,
                  int order)
{
   ParGridFunction assembled_gf(&fes);
   assembled_gf.SetFromTrueDofs(assembled);
   ParGridFunction pa_gf(&fes);
   pa_gf.SetFromTrueDofs(pa);
   ParGridFunction difference_gf(&fes);
   difference_gf = pa_gf;
   difference_gf -= assembled_gf;

   ParaViewDataCollection collection("diffusion_solver_benchmark", &mesh);
   collection.SetPrefixPath(prefix);
   collection.RegisterField("assembled_solution", &assembled_gf);
   collection.RegisterField("pa_solution", &pa_gf);
   collection.RegisterField("difference", &difference_gf);
   collection.SetLevelsOfDetail(order);
   collection.SetHighOrderOutput(true);
   collection.SetDataFormat(VTKFormat::BINARY);
   collection.SetCycle(0);
   collection.SetTime(0.0);
   collection.Save();
}

}

/// Run the assembled-versus-PA timing comparison.
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;
   const int rank = Mpi::WorldRank();

   const char *device_config = "cpu";
   int dim = 2, elements = 12, order = 3;
   int serial_refinements = 0, parallel_refinements = 1;
   int repetitions = 10, max_iterations = 1000, print_level = 0;
   real_t relative_tolerance = 1e-10;
   bool paraview = false;
   const char *paraview_prefix = "ParaView";

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device", "MFEM device.");
   args.AddOption(&dim, "-dim", "--dimension", "Dimension: 2 or 3.");
   args.AddOption(&elements, "-n", "--num-elements", "Elements per direction.");
   args.AddOption(&order, "-o", "--order", "H1 polynomial order.");
   args.AddOption(&serial_refinements, "-rs", "--serial-refinements",
                  "Refinements before mesh partitioning.");
   args.AddOption(&parallel_refinements, "-rp", "--parallel-refinements",
                  "Refinements after mesh partitioning.");
   args.AddOption(&repetitions, "-rep", "--repetitions", "Timed solves.");
   args.AddOption(&relative_tolerance, "-rtol", "--relative-tolerance",
                  "CG relative tolerance.");
   args.AddOption(&max_iterations, "-mi", "--max-iterations",
                  "CG maximum iterations.");
   args.AddOption(&print_level, "-pl", "--print-level", "CG print level.");
   args.AddOption(&paraview, "-pv", "--paraview",
                  "-no-pv", "--no-paraview", "Write ParaView results.");
   args.AddOption(&paraview_prefix, "-pv-prefix", "--paraview-prefix",
                  "ParaView output directory.");
   args.Parse();
   if (!args.Good())
   {
      if (rank == 0) { args.PrintUsage(cout); }
      return 1;
   }
   const int local_options_ok =
      (dim == 2 || dim == 3) && elements > 0 && order > 0 &&
      serial_refinements >= 0 && parallel_refinements >= 0 &&
      repetitions > 0 && relative_tolerance >= 0.0 &&
      max_iterations > 0;
   int global_options_ok = 0;
   MPI_Allreduce(&local_options_ok, &global_options_ok, 1, MPI_INT, MPI_MIN,
                 comm);
   if (!global_options_ok)
   {
      if (rank == 0) { cerr << "Invalid benchmark options.\n"; }
      return 2;
   }

   Device device(device_config);
   if (rank == 0) { device.Print(); }

   Mesh serial_mesh = dim == 2
      ? Mesh::MakeCartesian2D(elements, elements, Element::QUADRILATERAL,
                              true, 1.0, 1.0)
      : Mesh::MakeCartesian3D(elements, elements, elements,
                              Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   for (int i = 0; i < serial_refinements; i++)
   {
      serial_mesh.UniformRefinement();
   }
   ParMesh mesh(comm, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < parallel_refinements; i++)
   {
      mesh.UniformRefinement();
   }
   H1_FECollection collection(order, dim);
   ParFiniteElementSpace fes(&mesh, &collection);

   FunctionCoefficient source(Source);
   ParLinearForm load(&fes);
   load.AddDomainIntegrator(new DomainLFIntegrator(source));
   load.Assemble();
   Vector rhs(fes.GetTrueVSize());
   rhs.UseDevice(true);
   load.ParallelAssemble(rhs);

   FunctionCoefficient coefficient(Diffusion);
   StopWatch watch;

   MPI_Barrier(comm);
   watch.Start();
   unique_ptr<ParDiffusionSolver> assembled_solver(new ParDiffusionSolver(
      fes, coefficient, relative_tolerance, max_iterations, print_level));
   AddZeroBoundaryConditions(mesh, *assembled_solver);
   assembled_solver->Assemble();
   watch.Stop();
   const double assembled_setup =
      GlobalMaximumTime(comm, watch.RealTime());

   MPI_Barrier(comm);
   watch.Clear();
   watch.Start();
   unique_ptr<ParDiffusionSolverPA> pa_solver(new ParDiffusionSolverPA(
      fes, coefficient, relative_tolerance, max_iterations, print_level));
   AddZeroBoundaryConditions(mesh, *pa_solver);
   pa_solver->Assemble();
   watch.Stop();
   const double pa_setup = GlobalMaximumTime(comm, watch.RealTime());

   Vector assembled_solution, pa_solution;
   const double assembled_solve =
      TimeSolves(*assembled_solver, rhs, repetitions, comm, assembled_solution);
   const double pa_solve =
      TimeSolves(*pa_solver, rhs, repetitions, comm, pa_solution);

   Vector difference(pa_solution);
   difference -= assembled_solution;
   const real_t difference_norm =
      sqrt(InnerProduct(comm, difference, difference));
   const real_t reference_norm =
      sqrt(InnerProduct(comm, assembled_solution, assembled_solution));
   const real_t relative_difference =
      difference_norm/max(reference_norm, real_t(1.0));
   const HYPRE_BigInt global_dofs = fes.GlobalTrueVSize();

   if (rank == 0)
   {
      cout << "Global true DOFs: " << global_dofs << '\n'
           << "Repetitions: " << repetitions << '\n'
           << "Assembled setup [s]:  " << assembled_setup << '\n'
           << "PA setup [s]:         " << pa_setup << '\n'
           << "Assembled solves [s]: " << assembled_solve << '\n'
           << "PA solves [s]:        " << pa_solve << '\n'
           << "Assembled avg [s]:    " << assembled_solve/repetitions << '\n'
           << "PA avg [s]:           " << pa_solve/repetitions << '\n'
           << "Relative solution difference: " << relative_difference << '\n';
   }

   if (paraview)
   {
      SaveParaView(mesh, fes, assembled_solution, pa_solution,
                   paraview_prefix, order);
   }
   return 0;
}
