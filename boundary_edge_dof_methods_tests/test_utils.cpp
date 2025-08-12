#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <set>
#include <vector>
#include <mpi.h>
#include <numeric>

using namespace std;
using namespace mfem;

// Helper function to create test mesh with oriented triangular face (copied from ../tests/unit/mesh/mesh_test_utils.cpp)
Mesh OrientedTriFaceMesh(int orientation, bool add_extbdr)
{
   if (!(orientation == 1 || orientation == 3 || orientation == 5)) {
       MFEM_ABORT("Invalid orientation: must be 1, 3, or 5");
   }

   Mesh mesh(3, 5, 2);
   mesh.AddVertex(-1.0, 0.0, 0.0);
   mesh.AddVertex(0.0, 0.0, 0.0);
   mesh.AddVertex(0.0, 1.0, 0.0);
   mesh.AddVertex(0.0, 0.0, 1.0);
   mesh.AddVertex(1.0, 0.0, 0.0);

   mesh.AddTet(0, 1, 2, 3, 1);

   switch (orientation)
   {
      case 1:
         mesh.AddTet(4,2,1,3,2); break;
      case 3:
         mesh.AddTet(4,3,2,1,2); break;
      case 5:
         mesh.AddTet(4,1,3,2,2); break;
   }

   mesh.FinalizeTopology(add_extbdr);
   mesh.SetAttributes();

   auto *bdr = new Triangle(1,2,3,
                            mesh.bdr_attributes.Size() == 0 ? 1 : mesh.bdr_attributes.Max() + 1);
   mesh.AddBdrElement(bdr);

   mesh.FinalizeTopology(false);
   mesh.Finalize();
   return mesh;
}

Mesh* RefinedTetPairMesh(int orientation, int pair_index)
{
   if (!(orientation == 1 || orientation == 3 || orientation == 5)) {
       MFEM_ABORT("Invalid orientation: must be 1, 3, or 5");
   }
   if (pair_index < 0) {
       MFEM_ABORT("Invalid pair_index: must be >= 0");
   }

   Mesh *mesh = new Mesh(OrientedTriFaceMesh(orientation, true));
   mesh->UniformRefinement();
   
   // Generate all possible pairs (C(16,2) = 120)
   std::vector<std::pair<int, int>> all_pairs;
   for (int i = 0; i < mesh->GetNE(); i++) 
   {
       for (int j = i + 1; j < mesh->GetNE(); j++) 
       {
           all_pairs.push_back({i, j});
       }
   }
   
   auto selected_pair = all_pairs[pair_index];
   
   // Check if this pair is adjacent (shares a face)
   bool is_adjacent = false;
   Array<int> faces_i, faces_j, ori_i, ori_j;
   mesh->GetElementFaces(selected_pair.first, faces_i, ori_i);
   mesh->GetElementFaces(selected_pair.second, faces_j, ori_j);
   for (int fi = 0; fi < faces_i.Size() && !is_adjacent; fi++) 
   {
       for (int fj = 0; fj < faces_j.Size(); fj++) 
       {
           if (faces_i[fi] == faces_j[fj]) 
           {
               is_adjacent = true;
               break;
           }
       }
   }
   
   if (!is_adjacent) 
   {
       // Skip non-adjacent pairs - return a simple mesh without the special boundary
       return mesh;
   }
   int tet1 = selected_pair.first;
   int tet2 = selected_pair.second;
   
   int shared_face = -1;
   Array<int> faces1, faces2, ori1, ori2;
   mesh->GetElementFaces(tet1, faces1, ori1);
   mesh->GetElementFaces(tet2, faces2, ori2);
   for (int fi = 0; fi < faces1.Size(); fi++) 
   {
       for (int fj = 0; fj < faces2.Size(); fj++) 
       {
           if (faces1[fi] == faces2[fj]) 
           {
               shared_face = faces1[fi];
               break;
           }
       }
       if (shared_face != -1) break;
   }

   
   Array<int> face_verts;
   mesh->GetFaceVertices(shared_face, face_verts);
   
   auto *bdr = new Triangle(face_verts[0], face_verts[1], face_verts[2], 3);
   mesh->AddBdrElement(bdr);
   
   mesh->FinalizeTopology(false);
   mesh->Finalize();
   return mesh;
}

// Generate all possible partitionings of n_elements into num_procs
void GeneratePartitionings(int n_elements, int num_procs, 
                          vector<vector<int>>& all_partitionings)
{
    vector<int> partition(n_elements);
    
    function<void(int)> generate = [&](int elem) 
    {
        if (elem == n_elements) 
        {
            all_partitionings.push_back(partition);
            return;
        }
        
        for (int proc = 0; proc < num_procs; proc++) 
        {
            partition[elem] = proc;
            generate(elem + 1);
        }
    };
    
    generate(0);
}