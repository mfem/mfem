
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options
   const char *mesh_file = "../../data/periodic-annulus-sector.msh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh orig_mesh(mesh_file);
   orig_mesh.CheckElementOrientation(true);
   orig_mesh.CheckBdrElementOrientation(true);

   // mesh.EnsureNodes();
   // mesh.UniformRefinement();
   // Array<int> elems({1,3,21,10,20,2,0});
   int nel = orig_mesh.GetNE();
   // int nel = elems.Size();
   Array<int> elems(nel);
   for (int i = 0; i<nel; i++)
   {
      elems[i] = i;
   }
   elems.Print();

   Mesh new_mesh = Mesh::ExtractMesh(orig_mesh,elems);
   new_mesh.CheckElementOrientation(true);
   new_mesh.CheckBdrElementOrientation(true);

   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mesh0_sock(vishost, visport);
      mesh0_sock.precision(8);
      mesh0_sock << "mesh\n" << orig_mesh << "keys n \n" << flush;

      socketstream mesh1_sock(vishost, visport);
      mesh1_sock.precision(8);
      mesh1_sock << "mesh\n" << new_mesh << "keys n \n" << flush;
   }

   return 0;
}

