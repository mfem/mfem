//                                MFEM Example 0
//
// Compile with: make getsubmesh
//
// Sample runs:  getsubmesh -m ../miniapps/meshing/blade.mesh -p 4 -o bladebdr.mesh
//               getsubmesh -m ../data/inline-hex.mesh -p 1 -o hexbdr.mesh -rs 1
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "../mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../miniapps/meshing/blade.mesh";
   int polynomial_order = 1;
   int rs_levels         = 0;
   string out_file = "submesh.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&polynomial_order, "-p", "--polynomial_order", "Finite element polynomial degree");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&out_file, "-o", "--out", "output file to use.");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   for (int lev = 0; lev < rs_levels; lev++)
   {
      mesh.UniformRefinement();
   }
   mesh.SetCurvature(polynomial_order, false, -1, 0);

   int nattr = mesh.bdr_attributes.Max();
   Array<int> subdomain_attributes(nattr);
   for (int i = 0; i < nattr; i++) {
       subdomain_attributes[i] = i+1;
   }

   auto submesh = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);

   submesh.Save(out_file);

   return 0;
}
