#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "meshes/two-squares-nurbs.mesh";
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.Parse();
   args.PrintOptions(cout);
   Mesh mesh(mesh_file, 1, 1);

   // CheckKVDirection()
   cout << endl << "CheckKVDirection()" << endl;
   Array<int> patchvert, edges, orient, edgevert;
   Mesh patchTopo = mesh.NURBSext->GetPatchTopology();
   int NP = patchTopo.GetNE();
   for (int p = 0; p < NP; p++)
   {
      patchTopo.GetElementVertices(p, patchvert);
      patchTopo.GetElementEdges(p, edges, orient);

      cout << "Patch " << p << endl;
      cout << "pv: " << patchvert[0] << ", " << patchvert[1] << ", " << patchvert[2]
           << ", " << patchvert[3] << endl;

      cout << "i, e, ev0, ev1, pv[eev0], pv[eev1], oe" << endl;
      for (int i = 0; i < edges.Size(); i++)
      {
         const int edge = edges[i];
         patchTopo.GetEdgeVertices(edge, edgevert); // edge -> vert
         const int *eev = patchTopo.GetElement(p)->GetEdgeVertices(
                             i); // el -> edge -> vert

         cout << i << ", " << edges[i] << ", "
              << edgevert[0] << ", " << edgevert[1] << ", "
              << patchvert[eev[0]] << ", " << patchvert[eev[1]] << ", "
              << orient[i] << endl;
      }
   }

   cout << "Mesh is consistent? " << mesh.NURBSext->CheckPatches() << endl;

}