/**
 * @file test_par_diffusion_solver_pa.cpp
 * @brief Correctness tests for ParDiffusionSolverPA on CPU or GPU.
 *
 * A known discrete H1 vector is applied through the unconstrained PA operator
 * to manufacture the load. The same field supplies nonzero Dirichlet data.
 * The test exercises real_t, borrowed Coefficient, and shared_ptr<Coefficient>
 * constructor interfaces.
 */

#include "mfem.hpp"
#include "par_diffusion_solver_pa.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>

using namespace mfem;
using namespace std;

namespace
{

/// Evaluate the smooth field used as an exactly recoverable discrete solution.
real_t ExactField(const Vector &x)
{
   real_t value = 1.0;
   for (int d = 0; d < x.Size(); d++) { value *= 1.0 + x(d); }
   return value;
}

/// Evaluate a positive variable diffusion coefficient.
real_t VariableCoefficient(const Vector &x)
{
   real_t sum = 0.0;
   for (int d = 0; d < x.Size(); d++) { sum += x(d); }
   return 1.0 + 0.2*sum;
}

/// Compute the distributed Euclidean norm of a true-DOF vector.
real_t ParallelNorm(MPI_Comm comm, const Vector &x)
{
   return sqrt(InnerProduct(comm, x, x));
}

/// Copy all one-based boundary attributes from a parallel mesh.
void GetBoundaryAttributes(const ParMesh &mesh, Array<int> &attributes)
{
   attributes.SetSize(mesh.bdr_attributes.Size());
   for (int i = 0; i < attributes.Size(); i++)
   {
      attributes[i] = mesh.bdr_attributes[i];
   }
}

/// Run one ownership-interface test and return the global pass status.
bool RunCase(ParDiffusionSolverPA &solver,
             const Array<int> &attributes,
             Coefficient &boundary_value,
             const Vector &x_exact,
             MPI_Comm comm,
             real_t tolerance,
             Vector &solution,
             real_t &relative_error,
             real_t &relative_residual)
{
   Vector rhs(x_exact.Size());
   rhs.UseDevice(true);
   solver.GetFullOperator().Mult(x_exact, rhs);

   for (int i = 0; i < attributes.Size(); i++)
   {
      solver.AddBoundaryCondition(attributes[i], boundary_value);
   }
   solver.Assemble();

   solution.SetSize(solver.Width());
   solution.UseDevice(true);
   solver.Mult(rhs, solution);

   Vector error(solution);
   error.UseDevice(true);
   error -= x_exact;
   relative_error =
      ParallelNorm(comm, error)/max(ParallelNorm(comm, x_exact), real_t(1.0));

   Vector system_rhs;
   solver.FormSystemRHS(rhs, system_rhs);
   Vector residual(rhs.Size());
   residual.UseDevice(true);
   solver.GetSystemOperator().Mult(solution, residual);
   residual -= system_rhs;
   relative_residual =
      ParallelNorm(comm, residual)/
      max(ParallelNorm(comm, system_rhs), real_t(1.0));

   const int local_pass =
      solver.GetConverged() && relative_error <= tolerance &&
      relative_residual <= tolerance;
   int global_pass = 0;
   MPI_Allreduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN, comm);
   return global_pass == 1;
}

/// Write one solution, exact field, and error field to ParaView.
void SaveParaView(ParMesh &mesh,
                  ParFiniteElementSpace &fes,
                  const Vector &solution,
                  const Vector &exact,
                  const string &name,
                  const char *prefix,
                  int order)
{
   ParGridFunction solution_gf(&fes);
   solution_gf.SetFromTrueDofs(solution);
   ParGridFunction exact_gf(&fes);
   exact_gf.SetFromTrueDofs(exact);
   ParGridFunction error_gf(&fes);
   error_gf = solution_gf;
   error_gf -= exact_gf;

   ParaViewDataCollection collection(name, &mesh);
   collection.SetPrefixPath(prefix);
   collection.RegisterField("solution", &solution_gf);
   collection.RegisterField("exact", &exact_gf);
   collection.RegisterField("error", &error_gf);
   collection.SetLevelsOfDetail(order);
   collection.SetHighOrderOutput(true);
   collection.SetDataFormat(VTKFormat::BINARY);
   collection.SetCycle(0);
   collection.SetTime(0.0);
   collection.Save();
}

}

/// Run all PA diffusion constructor and ownership tests.
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;
   const int rank = Mpi::WorldRank();

   const char *device_config = "cpu";
   int dim = 2, elements = 6, order = 2;
   int serial_refinements = 0, parallel_refinements = 1;
   real_t tolerance = 1e-8, solver_tolerance = 1e-12;
   int max_iterations = 1000, print_level = 0;
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
   args.AddOption(&tolerance, "-tol", "--tolerance", "Pass tolerance.");
   args.AddOption(&solver_tolerance, "-rtol", "--relative-tolerance",
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
      tolerance > 0.0 && solver_tolerance >= 0.0 &&
      max_iterations > 0;
   int global_options_ok = 0;
   MPI_Allreduce(&local_options_ok, &global_options_ok, 1, MPI_INT, MPI_MIN,
                 comm);
   if (!global_options_ok)
   {
      if (rank == 0) { cerr << "Invalid test options.\n"; }
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
   Array<int> attributes;
   GetBoundaryAttributes(mesh, attributes);

   FunctionCoefficient exact_coefficient(ExactField);
   ParGridFunction exact_grid_function(&fes);
   exact_grid_function.ProjectCoefficient(exact_coefficient);
   Vector x_exact(fes.GetTrueVSize());
   x_exact.UseDevice(true);
   exact_grid_function.GetTrueDofs(x_exact);

   real_t errors[3] = {0.0, 0.0, 0.0};
   real_t residuals[3] = {0.0, 0.0, 0.0};
   bool passes[3] = {false, false, false};
   Vector solutions[3];

   {
      ParDiffusionSolverPA solver(
         fes, real_t(2.0), solver_tolerance, max_iterations, print_level);
      passes[0] = RunCase(solver, attributes, exact_coefficient, x_exact, comm,
                          tolerance, solutions[0], errors[0], residuals[0]);
   }
   {
      FunctionCoefficient coefficient(VariableCoefficient);
      ParDiffusionSolverPA solver(
         fes, coefficient, solver_tolerance, max_iterations, print_level);
      passes[1] = RunCase(solver, attributes, exact_coefficient, x_exact, comm,
                          tolerance, solutions[1], errors[1], residuals[1]);
   }
   {
      shared_ptr<Coefficient> coefficient =
         make_shared<FunctionCoefficient>(VariableCoefficient);
      ParDiffusionSolverPA solver(
         fes, coefficient, solver_tolerance, max_iterations, print_level);
      coefficient.reset();
      passes[2] = RunCase(solver, attributes, exact_coefficient, x_exact, comm,
                          tolerance, solutions[2], errors[2], residuals[2]);
   }

   if (paraview)
   {
      const char *names[3] = {"pa_real", "pa_coefficient", "pa_shared"};
      for (int i = 0; i < 3; i++)
      {
         SaveParaView(mesh, fes, solutions[i], x_exact, names[i],
                      paraview_prefix, order);
      }
   }

   if (rank == 0)
   {
      const char *names[3] = {"real_t", "Coefficient", "shared_ptr"};
      for (int i = 0; i < 3; i++)
      {
         cout << names[i] << ": error=" << errors[i]
              << ", residual=" << residuals[i]
              << ", " << (passes[i] ? "PASS" : "FAIL") << '\n';
      }
   }
   return passes[0] && passes[1] && passes[2] ? 0 : 3;
}
