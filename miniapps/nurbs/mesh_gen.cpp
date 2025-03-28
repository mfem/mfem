// Generate simple single-patch nurbs mesh

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   // const char *mesh_file = "../../data/cube-nurbs.mesh";
   const char *mesh_file =
      "../../../miniapps/nurbs/meshes/beam-hex-nurbs-onepatch.mesh";
   bool low_order = false;
   int ref_levels = 1;
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&low_order, "-lo", "--low-order", "-ho", "--high-order",
                  "Make a low order mesh with the same number of DOFs.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use as a template.");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "Number of uniform mesh refinements.");
   args.AddOption(&order, "-o", "--order",
                  "Order of NURBS basis.");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);

   // Read the mesh
   Mesh mesh(mesh_file, 1, 1);

   // Assuming template mesh is order 1
   if (order < 1)
   {
      cout << "Order must be at least 1" << endl;
      return 1;
   }
   else if (order > 1 && !low_order)
   {
      mesh.DegreeElevate(order-1);
   }

   // Refine the mesh to increase the resolution.
   // In the high-order mesh, we want this many divisions
   int divisions = pow(2,ref_levels);
   if (!low_order)
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.NURBSUniformRefinement();
      }
   }
   else
   {
      // In the low-order mesh, we want the same number of dofs
      mesh.NURBSUniformRefinement(divisions + (order-1));
   }

   // Write to file
   string lo_or_ho = low_order ? "lo" : "ho";
   string filename = lo_or_ho +
                     "_p" + std::to_string(order) +
                     "_i" + std::to_string(divisions) +
                     ".mesh";
   ofstream mesh_ofs(filename);
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);

   return 0;
}