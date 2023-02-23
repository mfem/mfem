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

int main(int argc, char *argv[]) {
  // 1. Parse command line options.
  const char *mesh_file = NULL;
  int order = 3;
  int ref_levels = 1;
  int ode_solver_type = 4;  // RK4
  double cfl = 0.3;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use (default: 2x2 rectangular mesh).");
  args.AddOption(&order, "-o", "--order",
                 "Finite element polynomial degree (default: 1).");
  args.AddOption(
      &ref_levels, "-r", "--refine",
      "The number of uniform refinement to be performed (default: 5).");
  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                 "ODE solver: 1 - Forward Euler,\n\t"
                 "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");

  Mesh mesh =
      (mesh_file != NULL)
          ? Mesh(mesh_file, 1, 1)
          : Mesh::MakeCartesian2D(2, 2, mfem::Element::Type::QUADRILATERAL);

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

  GridFunction density(&dfes, u_block.GetData() + offsets[0]);
  GridFunction momentum(&dfes, u_block.GetData() + offsets[1]);
  GridFunction energy(&dfes, u_block.GetData() + offsets[1 + dim]);

  // Get Euler system
  HyperbolicConservationLaws *euler = getEulerSystem(vfes);
  euler->set_cfl(cfl);

  ODESolver *ode_solver = getODESolver(ode_solver_type);
  ode_solver->Init(*euler);

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