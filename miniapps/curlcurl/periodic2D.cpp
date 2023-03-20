// 
// make periodic2D
//      a simple code to create a periodic mesh in 2D
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int nx = 4, ny = 4; 
   double Lx = 1.0, Ly = 1.0;

   // Parse command line
   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--num-elem-in-x",
                  "Number of elements in the x-direction.");
   args.AddOption(&ny, "-ny", "--num-elem-in-y",
                  "Number of elements in the y-direction.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Create an initial mesh
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                     0.0, Lx, Ly, true);

   // Create translation vectors defining the periodicity
   Vector x_translation({Lx, 0.0, 0.0});
   Vector y_translation({0.0, Ly, 0.0});
   std::vector<Vector> translations = {x_translation, y_translation};
   Mesh newmesh = Mesh::MakePeriodic(
                  mesh,
                  mesh.CreatePeriodicVertexMapping(translations));
   newmesh.RemoveInternalBoundaries();

   // Save the final mesh
   ofstream mesh_ofs("periodic2D.mesh");
   mesh_ofs.precision(8);
   newmesh.Print(mesh_ofs);

   return 0;
}

