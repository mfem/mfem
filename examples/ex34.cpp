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
    int order = 1;
    int ref_levels = 5;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use (default: 2x2 rectangular mesh).");
    args.AddOption(&order, "-o", "--order",
                   "Finite element polynomial degree (default: 1).");
    args.AddOption(
        &ref_levels, "-r", "--refine",
        "The number of uniform refinement to be performed (default: 5).");

    Mesh mesh =
        (mesh_file != NULL)
            ? Mesh(mesh_file, 1, 1)
            : Mesh::MakeCartesian2D(2, 2, mfem::Element::Type::QUADRILATERAL);

    const int dim = mesh.Dimension();
    for (int i = 0; i < ref_levels; i++) {
        mesh.UniformRefinement();
    }

    return 0;
}
