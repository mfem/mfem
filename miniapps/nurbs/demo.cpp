// Script to demonstrate differences in edge_to_knot map

#include "mfem.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // Parse options.
   const char *mesh_file = "../../data/square-nurbs.mesh";
   int nurbs_degree_increase = 0;
   int ref_levels = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "Number of uniform mesh refinements.");
   args.AddOption(&nurbs_degree_increase, "-incdeg", "--nurbs-degree-increase",
                  "Elevate NURBS mesh degree by this amount.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   // Read the mesh; elevate degree and refine
   Mesh mesh(mesh_file, 1, 1);
   if (nurbs_degree_increase > 0) { mesh.DegreeElevate(nurbs_degree_increase); }
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }
   // Make a deep copy of patch topology and patches
   Mesh patch_topology = Mesh(*mesh.NURBSext->GetPatchTopology());
   Array<NURBSPatch*> patches = mesh.GetNURBSPatches();


   // Print out edge_to_knot map
   cout << endl << "Original Mesh: edge to knot map:" << endl;
   mesh.NURBSext->PrintEdgeToKnot();


   // Reconstruct mesh using patch topology and patches
   NURBSExtension ext(&patch_topology, patches);
   Mesh mesh_copy(ext);

   // // Print out edge_to_knot map for the reconstructed mesh
   // cout << endl << "Reconstructed Mesh: edge to knot map:" << endl;
   // mesh_copy.NURBSext->PrintEdgeToKnot();



   return 0;
}