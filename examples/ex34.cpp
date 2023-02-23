//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

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

  GridFunction mom(&dfes, u_block.GetData() + offsets[1]);

  // Get Euler system
  HyperbolicConservationLaws *euler = getEulerSystem(vfes);

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