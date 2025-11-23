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

   // mesh.Print(cout);

   // Array<int> edge_to_ukv;
   // Array<int> ukv_to_rpkv;
   // mesh.GetEdgeToUniqueKnotvector(edge_to_ukv, ukv_to_rpkv);


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
      cout << "pv: " << patchvert[0] << ", " << patchvert[1] << ", " << patchvert[2] << ", " << patchvert[3] << endl;

      cout << "i, e, ev0, ev1, pv[eev0], pv[eev1], oe" << endl;
      for (int i = 0; i < edges.Size(); i++)
      {
         const int edge = edges[i];
         patchTopo.GetEdgeVertices(edge, edgevert); // edge -> vert
         const int *eev = patchTopo.GetElement(p)->GetEdgeVertices(i); // el -> edge -> vert

         cout << i << ", " << edges[i] << ", "
              << edgevert[0] << ", " << edgevert[1] << ", "
              << patchvert[eev[0]] << ", " << patchvert[eev[1]] << ", "
              << orient[i] << endl;
      }
   }


   cout << "Mesh is consistent? " << mesh.NURBSext->CheckPatches() << endl;


   // Check edge 7 specifically
   // const int i = 2; // el 2
   // const int j = 0; // local edge 0
   // const int *v = patchTopo.GetElement(i)->GetVertices();
   // const int *e = patchTopo.GetElement(i)->GetEdgeVertices(j);
   // cout << "edge 7:" << endl;
   // cout << "ev0, ev1, v0, v1" << endl;
   // cout << e[0] << ", " << e[1] << ", " << v[e[0]] << ", " << v[e[1]] << endl;



}