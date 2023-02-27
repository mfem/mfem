//                                MFEM Example 34
//
// Compile with: make ex34
//
// Sample runs:  ex34
//
// Description: This example code demonstrates the usage of MFEM to define
//              Hyperbolic Conservation Laws
//              ∂ₜu + ∇⋅F(u) = 0
//              with initial condition and periodic boundary conditions.
//              Currently, advection equation, Burgers' equation, Compressible
//              Euler equations are implemented. Other equations can be
//              implemented by defining flux functions.

#include "ex34.hpp"

#include <fstream>
#include <iostream>

#include "mfem.hpp"

using namespace std;
using namespace mfem;

inline double square(const double x) { return x * x; }

ODESolver *getODESolver(const int ode_solver_type);

typedef std::function<void(const Vector &, Vector &)> InitialCondition;
InitialCondition getInitCond(const int problem);

Mesh getMesh(const int problem);

int main(int argc, char *argv[]) {
  // 1. Parse command line options.
  const char *mesh_file = NULL;
  int order = 3;
  int ref_levels = 1;
  int ode_solver_type = 4;  // RK4
  int problem = 1;
  double cfl = 0.3;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use (default: problem dependent).");
  args.AddOption(&order, "-o", "--order",
                 "Finite element polynomial degree (default: 1).");
  args.AddOption(
      &ref_levels, "-r", "--refine",
      "The number of uniform refinement to be performed (default: 5).");
  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                 "ODE solver: 1 - Forward Euler,\n\t"
                 "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
  args.AddOption(&problem, "-p", "--problem",
                 "Problem setup to use. See options in velocity_function().");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
//   args.PrintOptions(cout);

  Mesh mesh = (mesh_file != NULL) ? Mesh(mesh_file, 1, 1) : getMesh(problem);
  mesh.EnsureNCMesh();

  const int dim = mesh.Dimension();
  const int num_equations = dim + 2;

  for (int i = 0; i < ref_levels; i++) {
    mesh.UniformRefinement();
  }

  L2_FECollection dg(order, dim);
  // Scalar finite element space
  FiniteElementSpace sfes(&mesh, &dg, 1);
  // Vector finite element space for velocity
  FiniteElementSpace dfes(&mesh, &dg, dim);
  // Vector finite element space for state variable
  FiniteElementSpace vfes(&mesh, &dg, num_equations);
  Array<int> offsets(num_equations + 1);
  for (int k = 0; k <= num_equations; k++) {
    offsets[k] = k * vfes.GetNDofs();
  }
  BlockVector u_block(offsets);

  GridFunction density(&sfes, u_block.GetData() + offsets[0]);
  GridFunction momentum(&dfes, u_block.GetData() + offsets[1]);
  GridFunction energy(&sfes, u_block.GetData() + offsets[1 + dim]);

  // Initialize the state.
  VectorFunctionCoefficient u0(num_equations, getInitCond(problem));
  GridFunction sol(&vfes, u_block.GetData());
  sol.ProjectCoefficient(u0);

  // Output the initial solution.
  {
    ofstream mesh_ofs("vortex.mesh");
    mesh_ofs.precision(8);
    mesh_ofs << mesh;

    for (int k = 0; k < num_equations; k++) {
      GridFunction uk(&sfes, u_block.GetBlock(k));
      ostringstream sol_name;
      sol_name << "vortex-" << k << "-init.gf";
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      sol_ofs << uk;
    }
  }

  // Get Euler system
  HyperbolicConservationLaws *euler = getEulerSystem(vfes);
  euler->set_cfl(cfl);

  // Get ODE solver
  ODESolver *ode_solver = getODESolver(ode_solver_type);
  ode_solver->Init(*euler);
  double t = 0.0;
  euler->SetTime(t);

  return 0;
}

ODESolver *getODESolver(const int ode_solver_type) {
  switch (ode_solver_type) {
    case 1:
      return new ForwardEulerSolver;
    case 2:
      return new RK2Solver(1.0);
    case 3:
      return new RK3SSPSolver;
    case 4:
      return new RK4Solver;
    case 6:
      return new RK6Solver;
    default:
      cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      throw std::invalid_argument("Failed to create an ODE solver\n");
  }
}

Mesh getMesh(const int problem) {
  switch (problem) {
    case 1:  // 1D accuracy test
      // (-1, 1), Periodic
      return Mesh("../data/periodic-segment.mesh");
    case 2:  // 2D accuracy test
      // (-1, 1) x (-1 ,1), Doubly Periodic
      return Mesh("../data/periodic-square-4x4.mesh");
    case 3:  // 1D Sod's Shock Tube
      // (0, 1), Dirichlet
      return Mesh::MakeCartesian1D(2, 1.0);
    case 4:  // Intersection of Mach 3
      // (0, 10), Dirichlet
      return Mesh::MakeCartesian1D(10, 10.0);
    case 5:  // 2D Sod's shock tube
      // (0, 1) x (0, 0.1), Dirichlet
      return Mesh::MakeCartesian2D(10, 1, mfem::Element::QUADRILATERAL, true,
                                   1.0, 0.1);
    case 6:  // 2D fast vortex
      // (-1, 1) x (-1 ,1), Doubly Periodic
      return Mesh("../data/periodic-square-4x4.mesh");
    case 7:  // 2D slow vortex
      // (-1, 1) x (-1 ,1), Doubly Periodic
      return Mesh("../data/periodic-square-4x4.mesh");
    default:
      mfem_error("Default mesh for the current problem is undefined!");
      throw std::invalid_argument("Invalid Problem.");
  }
}
// Get initial condition!
InitialCondition getInitCond(const int problem) {
  switch (problem) {
    case 1:  // 1D accuracy test
      std::cout << "1D Accuracy Test." << std::endl;
      std::cout << "domain = (-1, 1)" << std::endl;
      return [](const Vector &x, Vector &y) {
        const double density = 1.0 + 0.2 * __sinpi(x(0));
        const double velocity = 1.0;
        const double pressure = 1.0;
        const double energy =
            pressure / (1.4 - 1.0) + density * 0.5 * square(velocity);

        y(0) = density;
        y(1) = density * velocity;
        y(2) = energy;
      };

    case 2:  // 2D accuracy test
      std::cout << "2D Accuracy Test." << std::endl;
      std::cout << "domain = (-1, 1) x (-1, 1)" << std::endl;
      return [](const Vector &x, Vector &y) {
        const double density = 1.0 + 0.2 * __sinpi(x(0) + x(1));
        const double velocity_x = 0.7;
        const double velocity_y = 0.3;
        const double pressure = 1.0;
        const double energy =
            pressure / (1.4 - 1.0) +
            density * 0.5 * (square(velocity_x) + square(velocity_y));

        y(0) = density;
        y(1) = density * velocity_x;
        y(2) = density * velocity_y;
        y(3) = energy;
      };

    case 3:  // 1D Sod's Shock Tube
      std::cout << "1D Sod's Shock Tube." << std::endl;
      std::cout << "domain = (0, 1)" << std::endl;
      return [](const Vector &x, Vector &y) {
        const double density = x(0) < 0.3 ? 1.0 : 0.125;
        const double velocity = x(0) < 0.3 ? 0.75 : 0.0;
        const double pressure = x(0) < 0.3 ? 1 : 0.1;
        const double energy =
            pressure / (1.4 - 1.0) + density * 0.5 * square(velocity);

        y(0) = density;
        y(1) = density * velocity;
        y(2) = energy;
      };

    case 4:  // Intersection of Mach 3
      return [](const Vector &x, Vector &y) {
        const double density =
            x(0) < 1.0 ? 27 / 7 : 1 + 0.2 * sin(5 * (x(0) + 5.0));
        const double velocity = x(0) < 1.0 ? 4.0 * sqrt(35) : 0.0;
        const double pressure = x(0) < 1.0 ? 31 / 3 : 1.0;
        const double energy =
            pressure / (1.4 - 1.0) + density * 0.5 * square(velocity);

        y(0) = density;
        y(1) = density * velocity;
        y(2) = energy;
      };

    case 5:  // 2D Sod's Shock Tube
      return [](const Vector &x, Vector &y) {
        const double density = x(0) < 0.5 ? 1.0 : 0.125;
        const double velocity_x = 0.0;
        const double velocity_y = 0.0;
        const double pressure = x(0) < 0.5 ? 1.0 : 0.1;
        const double energy =
            pressure / (1.4 - 1.0) +
            density * 0.5 * (square(velocity_x) + square(velocity_y));

        y(0) = density;
        y(1) = density * velocity_x;
        y(2) = density * velocity_y;
        y(3) = energy;
      };

    case 6:  // Fast Vortex
      return [](const Vector &x, Vector &y) {
        MFEM_ASSERT(x.Size() == 2, "");

        // "Fast vortex"
        const double radius = 0.2;
        const double Minf = 0.5;
        const double beta = 1. / 5.;

        const double xc = 0.0, yc = 0.0;

        // Nice units
        const double vel_inf = 1.;
        const double den_inf = 1.;

        // Derive remainder of background state from this and Minf
        const double pres_inf = (den_inf / 1.4) * square(vel_inf / Minf);
        const double temp_inf = pres_inf / (den_inf * 1.0);

        double r2rad = 0.0;
        r2rad += square(x(0) - xc);
        r2rad += square(x(1) - yc);
        r2rad /= square(radius * radius);

        const double shrinv1 = 1.0 / (1.4 - 1.0);

        const double velX =
            vel_inf * (1 - beta * (x(1) - yc) / radius * exp(-0.5 * r2rad));
        const double velY =
            vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
        const double vel2 = velX * velX + velY * velY;

        const double specific_heat = 1.0 * 1.4 * shrinv1;
        const double temp = temp_inf - 0.5 * square(vel_inf * beta) /
                                           specific_heat * exp(-r2rad);

        const double den = den_inf * pow(temp / temp_inf, shrinv1);
        const double pres = den * 1.0 * temp;
        const double energy = shrinv1 * pres / den + 0.5 * vel2;

        y(0) = den;
        y(1) = den * velX;
        y(2) = den * velY;
        y(3) = den * energy;
      };

    case 7:  // Slow Vortex
      return [](const Vector &x, Vector &y) {
        MFEM_ASSERT(x.Size() == 2, "");

        // "Slow vortex"
        const double radius = 0.2;
        const double Minf = 0.05;
        const double beta = 1. / 50.;

        const double xc = 0.0, yc = 0.0;

        // Nice units
        const double vel_inf = 1.;
        const double den_inf = 1.;

        // Derive remainder of background state from this and Minf
        const double pres_inf = (den_inf / 1.4) * square(vel_inf / Minf);
        const double temp_inf = pres_inf / (den_inf * 1.0);

        double r2rad = 0.0;
        r2rad += square(x(0) - xc);
        r2rad += square(x(1) - yc);
        r2rad /= square(radius * radius);

        const double shrinv1 = 1.0 / (1.4 - 1.0);

        const double velX =
            vel_inf * (1 - beta * (x(1) - yc) / radius * exp(-0.5 * r2rad));
        const double velY =
            vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
        const double vel2 = velX * velX + velY * velY;

        const double specific_heat = 1.0 * 1.4 * shrinv1;
        const double temp = temp_inf - 0.5 * square(vel_inf * beta) /
                                           specific_heat * exp(-r2rad);

        const double den = den_inf * pow(temp / temp_inf, shrinv1);
        const double pres = den * 1.0 * temp;
        const double energy = shrinv1 * pres / den + 0.5 * vel2;

        y(0) = den;
        y(1) = den * velX;
        y(2) = den * velY;
        y(3) = den * energy;
      };

    default:
      mfem_error("Initial condition for the current problem is undefined!");
      throw std::invalid_argument("Invalid Problem.");
  }
}