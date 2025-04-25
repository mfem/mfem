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



   // cout << endl << "Mesh" << endl;
   // mesh.Print(mfem::out);
   // cout << endl << "end Mesh" << endl;


   // Print out the vertex indices for each edge of the mesh
   // cout << endl << "Mesh Edges and their Vertex Indices:" << endl;
   // for (int i = 0; i < mesh.GetNEdges(); i++) {
   //    Array<int> v(2); // Array to store the two vertex indices of the edge
   //    mesh.GetEdgeVertices(i, v); // Get the vertices of edge i
   //    cout << "Edge " << i << ": Vertex " << v[0] << " - Vertex " << v[1] << endl;
   // }


   // Print out edge_to_knot map
   cout << endl << "Original Mesh: edge to knot map:" << endl;
   mesh.NURBSext->PrintEdgeToKnot();
   // mesh.NURBSext->Che
   // Array<int> e2k({0, -1, 1, 1});


   // Generate edge_to_knot map
   // cout << endl << "Original Mesh: generated edge to knot map:" << endl;
   // Array<int> edge_to_knot;
   // mesh.GeneratePatchTopologyMap(edge_to_knot);
   // edge_to_knot.Print();


   // cout << endl << "PatchTopo" << endl;
   // patch_topology.Print(mfem::out);
   // cout << endl << "end PatchTopo" << endl;

   // Patch topology
   cout << endl << "PatchTopo: Mesh Edges and their Vertex Indices:" << endl;
   int p = 0; // patch index
   int kvidx;
   int gidx=0; // global kv index
   int ne=patch_topology.GetNEdges();
   Array<int> kvs(3);
   Array<int> edges, oedges;
   Array<int> v(2); // Array to store the two vertex indices of the edge
   Array<int> e2k(ne);
   for (int p = 0; p < mesh.NURBSext->GetNP(); p++)
   {
      cout << "Patch " << p << ":" << endl;
      // kvs
      for (int ikv = 0; ikv < patches[p]->GetNKV(); ++ikv)
      {
         kvs[ikv] = gidx;
         gidx++;
      }
      patch_topology.GetElementEdges(p, edges, oedges);
      // patches[p]->
      for (int i = 0; i < edges.Size(); i++) {
         kvidx = (i<8) ? ((i & 1) ? kvs[1] : kvs[0]) : kvs[2];
         patch_topology.GetEdgeVertices(edges[i], v); // Get the vertices of edge i
         kvidx = (v[1] > v[0]) ? kvidx : -1 - kvidx;

         e2k[edges[i]] = kvidx;

         // reverse mapping (as used by checkpatches)
         int kvidx_ = (oedges[i] > 0) ? kvidx : -1 - kvidx;

         cout << "  - i=" << i << ", edge=" << edges[i]
              << ", v1=" << v[0] << ", v2=" << v[1]
              << ", oedge=" << oedges[i]
              << ", kvidx=" << kvidx
              << ", kvidx_=" << kvidx_
              << endl;
      }
   }

   cout << endl << "Original Mesh: generated edge to knot map:" << endl;
   e2k.Print();

   // Reconstruct mesh using patch topology and patches
   NURBSExtension ext(&patch_topology, patches);
   Mesh mesh_copy(ext);

   // // Print out edge_to_knot map for the reconstructed mesh
   // cout << endl << "Reconstructed Mesh: edge to knot map:" << endl;
   // mesh_copy.NURBSext->PrintEdgeToKnot();



   return 0;
}