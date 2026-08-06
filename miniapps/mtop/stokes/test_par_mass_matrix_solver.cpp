/**
 * @file test_par_mass_matrix_solver.cpp
 * @brief Parallel manufactured-vector tests for ParMassMatrixSolver.
 *
 * The driver builds a Cartesian parallel H1 space and tests all ownership
 * interfaces:
 *
 * 1. an internally owned constant coefficient;
 * 2. a caller-owned borrowed coefficient;
 * 3. a coefficient whose ownership is shared with the solver.
 *
 * Each test constructs an explicitly assembled reference mass matrix, forms
 * rhs = M*x_exact, solves through mfem::Solver::Mult(), and checks the parallel
 * relative solution error and residual. All parallel assembly, solver
 * construction, Mult(), and destruction calls are made by every rank in the
 * same order. Rank-zero branches are used only for output.
 *
 * Use `-d cuda`, `-d hip`, or another MFEM device configuration to exercise
 * the partial-assembly solve on a supported accelerator. The driver itself
 * calls Mult() from the host; MFEM dispatches the numerical kernels to the
 * selected backend.
 */

#include "mfem.hpp"
#include "ParMassMatrixSolver.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace mfem;
using namespace std;

namespace
{

/**
 * @brief Evaluate the smooth manufactured finite-element field.
 *
 * @param x Physical coordinate.
 * @return Product of 1+x[d] over all coordinate directions.
 */
real_t ExactSolution(const Vector &x)
{
   real_t value = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      value *= 1.0 + x(d);
   }
   return value;
}

/**
 * @brief Evaluate a positive spatially varying mass coefficient.
 *
 * @param x Physical coordinate.
 * @return One plus one quarter of the coordinate sum.
 */
real_t VariableDensity(const Vector &x)
{
   real_t sum = 0.0;
   for (int d = 0; d < x.Size(); d++)
   {
      sum += x(d);
   }
   return 1.0 + 0.25*sum;
}

/**
 * @brief Compute the distributed Euclidean norm of a true-DOF vector.
 *
 * This function is collective over @a comm.
 *
 * @param comm MPI communicator defining the vector distribution.
 * @param x Local portion of a parallel true-DOF vector.
 * @return Global Euclidean norm.
 */
real_t ParallelNorml2(MPI_Comm comm, const Vector &x)
{
   return sqrt(InnerProduct(comm, x, x));
}

/**
 * @brief Test one mass solver against an explicitly assembled reference matrix.
 *
 * The routine collectively forms a manufactured RHS, invokes the generic
 * mfem::Solver interface, computes distributed error metrics, and reduces the
 * pass/fail result across all ranks.
 *
 * @param solver Mass solver under test.
 * @param reference_matrix Explicitly assembled true-DOF mass matrix.
 * @param x_exact Manufactured exact true-DOF vector.
 * @param comm MPI communicator used by all distributed objects.
 * @param tolerance Maximum accepted relative error and residual.
 * @param relative_error Returned global relative solution error.
 * @param relative_residual Returned global relative residual.
 * @return True only when every rank participates and both checks pass.
 */
bool TestSolver(Solver &solver,
                const HypreParMatrix &reference_matrix,
                const Vector &x_exact,
                MPI_Comm comm,
                real_t tolerance,
                real_t &relative_error,
                real_t &relative_residual)
{
   Vector rhs(x_exact.Size());
   rhs.UseDevice(true);
   reference_matrix.Mult(x_exact, rhs);

   Vector x(solver.Width());
   x.UseDevice(true);
   solver.Mult(rhs, x);

   Vector error(x);
   error.UseDevice(true);
   error -= x_exact;
   relative_error =
      ParallelNorml2(comm, error)/
      max(ParallelNorml2(comm, x_exact), real_t(1.0));

   Vector residual(rhs.Size());
   residual.UseDevice(true);
   reference_matrix.Mult(x, residual);
   residual -= rhs;
   relative_residual =
      ParallelNorml2(comm, residual)/
      max(ParallelNorml2(comm, rhs), real_t(1.0));

   const int local_pass =
      relative_error <= tolerance && relative_residual <= tolerance;
   int global_pass = 0;
   MPI_Allreduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN, comm);
   return global_pass == 1;
}

}

/**
 * @brief Run constant, borrowed, and shared-coefficient parallel mass tests.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument array.
 * @return 0 on success, 1 for parser failure, 2 for invalid options, or 3 when
 *         a numerical test fails.
 */
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   MPI_Comm comm = MPI_COMM_WORLD;
   const int myid = Mpi::WorldRank();

   int dim = 2;
   int elements = 6;
   int order = 2;
   int serial_refinements = 0;
   int parallel_refinements = 1;
   real_t constant_density = 2.0;
   real_t solver_tolerance = 1e-12;
   real_t check_tolerance = 1e-9;
   int max_iterations = 500;
   int print_level = 0;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension", "Spatial dimension: 2 or 3.");
   args.AddOption(&elements, "-n", "--num-elements",
                  "Elements per coordinate direction.");
   args.AddOption(&order, "-o", "--order", "H1 finite element order.");
   args.AddOption(&serial_refinements, "-rs", "--serial-refinements",
                  "Uniform refinements before partitioning.");
   args.AddOption(&parallel_refinements, "-rp", "--parallel-refinements",
                  "Uniform refinements after partitioning.");
   args.AddOption(&constant_density, "-c", "--constant-density",
                  "Constant coefficient used by the first test.");
   args.AddOption(&solver_tolerance, "-rtol", "--relative-tolerance",
                  "CG relative tolerance.");
   args.AddOption(&check_tolerance, "-tol", "--check-tolerance",
                  "Relative error and residual pass tolerance.");
   args.AddOption(&max_iterations, "-mi", "--max-iterations",
                  "CG maximum iterations.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "CG print level.");
   args.AddOption(&device_config, "-d", "--device",
                  "MFEM device configuration, e.g. cpu, cuda, or hip.");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }

   const bool valid_options =
      (dim == 2 || dim == 3) && elements > 0 && order > 0 &&
      serial_refinements >= 0 && parallel_refinements >= 0 &&
      constant_density > 0.0 && solver_tolerance >= 0.0 &&
      check_tolerance > 0.0 && max_iterations > 0;
   int local_options_ok = valid_options ? 1 : 0;
   int global_options_ok = 0;
   MPI_Allreduce(&local_options_ok, &global_options_ok, 1, MPI_INT, MPI_MIN,
                 comm);
   if (!global_options_ok)
   {
      if (myid == 0) { cerr << "Invalid test options.\n"; }
      return 2;
   }

   if (myid == 0) { args.PrintOptions(cout); }

   // Device initialization is collective in application control flow and must
   // precede construction of mesh, finite-element, operator, and vector data.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(elements, elements,
                                   Element::QUADRILATERAL, true, 1.0, 1.0);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(elements, elements, elements,
                                   Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }

   for (int level = 0; level < serial_refinements; level++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(comm, mesh);
   mesh.Clear();

   for (int level = 0; level < parallel_refinements; level++)
   {
      pmesh.UniformRefinement();
   }

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&pmesh, &fec);

   // GlobalTrueVSize is collective, so all ranks call it before rank 0 prints.
   const HYPRE_BigInt global_true_dofs = fes.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Global true DOFs: " << global_true_dofs << '\n';
   }

   FunctionCoefficient exact_coefficient(ExactSolution);
   ParGridFunction exact_grid_function(&fes);
   exact_grid_function.ProjectCoefficient(exact_coefficient);
   Vector x_exact(fes.GetTrueVSize());
   x_exact.UseDevice(true);
   exact_grid_function.GetTrueDofs(x_exact);

   bool constant_pass = false;
   real_t constant_error = 0.0;
   real_t constant_residual = 0.0;
   {
      ConstantCoefficient density(constant_density);
      ParBilinearForm reference_form(&fes);
      reference_form.AddDomainIntegrator(new MassIntegrator(density));
      reference_form.Assemble();
      reference_form.Finalize();
      unique_ptr<HypreParMatrix> reference_matrix(
         reference_form.ParallelAssemble());

      // Construction, Mult(), and destruction all occur on every rank.
      ParMassMatrixSolver mass_solver(
         fes, constant_density, solver_tolerance, max_iterations, print_level);
      Solver &solver_interface = mass_solver;
      constant_pass = TestSolver(
         solver_interface, *reference_matrix, x_exact, comm, check_tolerance,
         constant_error, constant_residual);
   }

   bool variable_pass = false;
   real_t variable_error = 0.0;
   real_t variable_residual = 0.0;
   {
      FunctionCoefficient density(VariableDensity);
      ParBilinearForm reference_form(&fes);
      reference_form.AddDomainIntegrator(new MassIntegrator(density));
      reference_form.Assemble();
      reference_form.Finalize();
      unique_ptr<HypreParMatrix> reference_matrix(
         reference_form.ParallelAssemble());

      // The borrowed coefficient outlives the solver and all ranks execute the
      // same object lifetime and collective call sequence.
      ParMassMatrixSolver mass_solver(
         fes, density, solver_tolerance, max_iterations, print_level);
      variable_pass = TestSolver(
         mass_solver, *reference_matrix, x_exact, comm, check_tolerance,
         variable_error, variable_residual);
   }

   bool shared_pass = false;
   real_t shared_error = 0.0;
   real_t shared_residual = 0.0;
   {
      shared_ptr<Coefficient> density =
         make_shared<FunctionCoefficient>(VariableDensity);
      ParBilinearForm reference_form(&fes);
      reference_form.AddDomainIntegrator(new MassIntegrator(*density));
      reference_form.Assemble();
      reference_form.Finalize();
      unique_ptr<HypreParMatrix> reference_matrix(
         reference_form.ParallelAssemble());

      // Every rank passes a non-null shared pointer. The solver retains one
      // owner while it exists, independently of the caller's shared owner.
      ParMassMatrixSolver mass_solver(
         fes, density, solver_tolerance, max_iterations, print_level);
      density.reset();
      shared_pass = TestSolver(
         mass_solver, *reference_matrix, x_exact, comm, check_tolerance,
         shared_error, shared_residual);
   }

   if (myid == 0)
   {
      cout << setprecision(16)
           << "\nConstant-coefficient test\n"
           << "  relative solution error = " << constant_error << '\n'
           << "  relative residual       = " << constant_residual << '\n'
           << "  result                  = "
           << (constant_pass ? "PASS" : "FAIL") << '\n'
           << "\nVariable-coefficient test\n"
           << "  relative solution error = " << variable_error << '\n'
           << "  relative residual       = " << variable_residual << '\n'
           << "  result                  = "
           << (variable_pass ? "PASS" : "FAIL") << '\n'
           << "\nShared-coefficient test\n"
           << "  relative solution error = " << shared_error << '\n'
           << "  relative residual       = " << shared_residual << '\n'
           << "  result                  = "
           << (shared_pass ? "PASS" : "FAIL") << '\n';
   }

   return (constant_pass && variable_pass && shared_pass) ? 0 : 3;
}
