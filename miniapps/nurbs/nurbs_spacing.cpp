#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/nc3-nurbs.mesh";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   mesh.NURBSext->ConvertToPatches(*mesh.GetNodes());

   Array<NURBSPatch*> patches;
   mesh.NURBSext->GetPatches(patches);

   mesh.NURBSext->PhysicalSpacing(*mesh.GetNodes());
   mesh.NURBSext->SetCoordsFromPatches(*mesh.GetNodes(), 2);

   mesh.NURBSext->ConvertToPatches(*mesh.GetNodes());
   mesh.NURBSext->GetPatches(patches);

   ofstream mesh_ofss("spaced.mesh");
   mesh_ofss.precision(8);
   mesh.Print(mesh_ofss);

   return 0;
}
