// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//             ---------------------------------------------------
//                       External Mesh Mapping Miniapp
//             ---------------------------------------------------
//
// This miniapp starts with a serial non-conforming mesh in a dummy format and demonstrates 
// how to build up a corresponding non-conforming MFEM Mesh and then decompose it 
// into a parallel ParMesh.  As part of this process we will demonstrate how to obtain 
// and compose the vertex ID mappings from the various steps that can shuffle the 
// vertices.  In the end this will let us map between the vertex ID numbers in the 
// external dummy mesh and the parallel non-conforming mesh constructed in MFEM.  This
// is the sort of thing you will have to do if you intend to add MFEM meshes to an existing
// simulation code and need them to exist and share data with other kinds of meshes in that 
// code.  If you are starting a new MFEM code, it makes much more sense to do everything with
// MFEM meshes.
//
// Compile with: make ext-mesh-mapping
//
// Sample runs:   mpirun -np 4 ext_mesh_mapping

#include "mfem.hpp"
#include "ext-mesh-mapping.hpp"

using namespace mfem;

Mesh *build_mfem_mesh(DummyMesh *dmesh);
void create_pmesh_to_mesh_emaps(Array<int> &partition, ParMesh *pmesh, Array<int> &emap);
void create_pmesh_to_mesh_vmaps(ParMesh *pmesh, Mesh *mesh, Array<int> &emap, Array<int> &vmap);

void print_dmesh_verts(DummyMesh *dmesh);
void print_mesh_verts(Mesh *mesh, const Array<int> &vmap = Array<int>());
void print_pmesh_verts(ParMesh *pmesh, const Array<int> &vmap = Array<int>());

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   //Initilize our dummy mesh and display the coordinates of the vertices
   DummyMesh *dmesh = new DummyMesh();
   print_dmesh_verts(dmesh);

   //Now build the mfem Mesh object with all of the elements in it on all processors and capture the vertex id mapping
   //that occurs when non-conforming meshes are finalized.
   Mesh * mesh = build_mfem_mesh(dmesh);
   Array<int> mesh_to_dmesh_vmap;
   const Array<int> vmap = mesh->ncmesh->GetVertexIDMap();
   mesh_to_dmesh_vmap = vmap;
   print_mesh_verts(mesh, mesh_to_dmesh_vmap);     //Print the vertices reordered using the vmap

   //Now enable parallel, given the following partition of the elements.
   //Note that we only have local vertex ids in the pmesh object.
   //The partition is the MPI Rank that each element lives on.
   Array<int> partition({3,3,1,2,0});
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partition.GetData());

   //Compute the mappings between the local element ids on each processor in pmesh
   //and the original global IDs in the mesh.  This can be computed from the element
   //partition array.
   Array<int> pmesh_to_mesh_emap;
   create_pmesh_to_mesh_emaps(partition, pmesh, pmesh_to_mesh_emap);

   //Now compute the mappings between the local vertex ids on each processor in pmesh
   //and the global ids in mesh.
   Array<int> pmesh_to_mesh_vmap;
   create_pmesh_to_mesh_vmaps(pmesh, mesh, pmesh_to_mesh_emap, pmesh_to_mesh_vmap);

   //Now compose the maps to create a final map between the local vertex numbering in the
   //pmesh object on this processor and the original dmesh vertex numbering
   Array<int> final_vmap(pmesh->GetNV());
   for (int local_vi = 0; local_vi < pmesh->GetNV(); ++local_vi)
   {
      final_vmap[local_vi] = mesh_to_dmesh_vmap[pmesh_to_mesh_vmap[local_vi]];
   }
   print_pmesh_verts(pmesh, final_vmap);           //Print the vertices with global IDs from the final_vmap
}

// Build up an MFEM Mesh from the data in the Dummy Mesh.  Since we are 
// building the vertex and element lists in the same order as we found them
// in the dmesh, the element id and vertex id mappings will be the identity maps.
Mesh *build_mfem_mesh(DummyMesh *dmesh)
{
   //Initilize the the dimension and memory for the mesh
   Mesh *mesh = new Mesh(2,    // The dimension of the mesh
                         dmesh->num_vertices, 
                         dmesh->num_elements, 
                         dmesh->num_belements, 
                         2    // The dimension of the space the mesh lives in (different for surface meshes)
                        );


   //Add the vertices into the mesh in the same order as they are in the dmesh
   for (int vid = 0; vid < dmesh->num_vertices; ++vid)
   {
      mesh->AddVertex(dmesh->V[vid].x, dmesh->V[vid].y);
   }

   //Add the elements into the mesh in the same order as they are in the dmesh
   //The dmesh is purely quads, but if it had other types we would use
   //the other mesh.Add* methods here as well.
   for (int eid = 0; eid < dmesh->num_elements; ++eid)
   {
      std::array<int,4> vid = dmesh->E[eid].vertex_ids;

      //The order of vertices in the MFEM quad elements is different from the
      //order of the vertices in out DummyMesh format so we must permute them
      //here.  See the ref-*.mesh files in mfem/data to establish to ordering
      //of the vertices in the MFEM elements.
      //  Dummy    MFEM
      //  2--3     3--2
      //  |  |     |  |
      //  0--1     0--1
      mesh->AddQuad(vid[0], vid[1], vid[3], vid[2]);
   }

   //Add the boundary elements into the mesh in the same order they are in the dmesh
   for (int bid = 0; bid < dmesh->num_belements; ++bid)
   {
      std::array<int,2> vid = dmesh->B[bid].vertex_ids;
      mesh->AddBdrSegment(vid[0], vid[1]);
   }

   //Finally add vertex parents to mark element 0 for anisotropic refinement
   for (int vpi = 0; vpi < dmesh->num_vparents; ++vpi)
   {
      mesh->AddVertexParents(std::get<0>(dmesh->VP[vpi]),std::get<1>(dmesh->VP[vpi]),std::get<2>(dmesh->VP[vpi]));
   }

   //This will make the mesh usable
   mesh->FinalizeMesh();

   return mesh;
}


// We can use the partition array to compute the mapping between the local
// element ids on each processor in pmesh and the global element ids in mesh.
void create_pmesh_to_mesh_emaps(Array<int> &partition, ParMesh *pmesh, Array<int> &emap)
{
   int my_rank = Mpi::WorldRank();
   emap.SetSize(pmesh->GetNE());

   int local_eid = 0;
   std::vector<int> indices;
   auto it = std::find(partition.begin(), partition.end(), my_rank);
   while (it != partition.end())
   {
      emap[local_eid] = std::distance(partition.begin(), it);
      local_eid++;
      it++;
      it = std::find(it, partition.end(), my_rank);
   }   
}


// We can use the local elements with local vertex ids defined in pmesh and the global
// elements defined with global vertex ids in mesh to define a mapping from the 
// local vertex ids on each processor to their global vertex id numbers. 
void create_pmesh_to_mesh_vmaps(ParMesh *pmesh, Mesh *mesh, Array<int> &emap, Array<int> &vmap)
{
   vmap.SetSize(pmesh->GetNV());
   for (int local_eid = 0; local_eid < pmesh->GetNE(); ++local_eid)
   {
      int global_eid = emap[local_eid];
      Array<int> local_elem_verts, global_elem_verts;
      pmesh->GetElement(local_eid)->GetVertices(local_elem_verts);
      mesh->GetElement(global_eid)->GetVertices(global_elem_verts);
      for (int vi = 0; vi < local_elem_verts.Size(); ++vi)
      {
         vmap[local_elem_verts[vi]] = global_elem_verts[vi];
      }
   }
}


void print_dmesh_verts(DummyMesh *dmesh)
{
   if (Mpi::Root())
   {
      std::cout << "6-----7-----8" << std::endl;
      std::cout << "|     |     |" << std::endl;
      std::cout << "|  3  |  4  |" << std::endl;
      std::cout << "|     |     |" << std::endl;
      std::cout << "3-----4-----5" << std::endl;
      std::cout << "|  1  |     |" << std::endl;
      std::cout << "9----10  2  |" << std::endl;
      std::cout << "|  0  |     |" << std::endl;
      std::cout << "0-----1-----2" << std::endl;      

      std::cout << "DummyMesh vertices:  "  << std::endl;
      for (int vid = 0; vid < dmesh->num_vertices; ++vid)
      {
         std::cout << vid << ":  " << dmesh->V[vid].x << ", " << dmesh->V[vid].y << std::endl;
      }
      std::cout << std::endl;
   }
}

void print_mesh_verts(Mesh *mesh, const Array<int> &vmap)
{
   if (Mpi::Root())
   {
      std::cout << std::endl << "MFEM Mesh Vertices:  "  << std::endl;
      if (vmap.Size() > 0)
      {
         std::cout << "(Remapped vertex ids)" << std::endl;
      }
      for (int vid = 0; vid < mesh->GetNV(); ++vid)
      {
         int id = vmap.Size() > 0 ? vmap[vid] : vid;
         double *vertex = mesh->GetVertex(id);
         std::cout << vid << ":  " <<  vertex[0] << ", " << vertex[1] << std::endl;
      }
      std::cout << std::endl;
   }   
}


void print_pmesh_verts(ParMesh *pmesh, const Array<int> &vmap)
{
   int my_rank = Mpi::WorldRank();
   int num_rank = Mpi::WorldSize();

   Array<int> num_verts(num_rank);
   int my_num_verts = pmesh->GetNV();
   MPI_Allgather(&my_num_verts, 1, MPI_INTEGER,
                 num_verts.GetData(), 1, MPI_INTEGER, MPI_COMM_WORLD);

   int max_num_verts = *std::max_element(num_verts.begin(), num_verts.end());
   Array<int> id_data(max_num_verts);
   Array<double> x_data(max_num_verts);
   Array<double> y_data(max_num_verts);

   //Send the data to rank 0
   for (int vid = 0; vid < pmesh->GetNV(); ++vid)
   {
      int id = vmap.Size() > 0 ? vmap[vid] : vid;
      double *vertex = pmesh->GetVertex(vid);
      id_data[vid] = id;
      x_data[vid] =  vertex[0];
      y_data[vid] =  vertex[1];
   }
   if (my_rank != 0)
   {
      MPI_Send(id_data.GetData(), pmesh->GetNV(), MPI_INTEGER, 0, 0, MPI_COMM_WORLD);
      MPI_Send(x_data.GetData(), pmesh->GetNV(), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
      MPI_Send(y_data.GetData(), pmesh->GetNV(), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
   }


   if (my_rank == 0)
   {
      std::cout << "MFEM ParMesh Vertices on each processor:  "  << std::endl;
      if (vmap.Size() > 0)
      {
         std::cout << "(Remapped vertex ids)" << std::endl;
      }      

      for (int p = 0; p < num_rank; ++p)
      {
         if (p != 0)
         {
            MPI_Status status;
            MPI_Recv(id_data.GetData(), num_verts[p], MPI_INTEGER, p, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(x_data.GetData(), num_verts[p], MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y_data.GetData(), num_verts[p], MPI_DOUBLE, p, 2, MPI_COMM_WORLD, &status);
         }

         for (int vid = 0; vid < num_verts[p]; ++vid)
         {
            std::cout << "rank (" << p << ") id (" << id_data[vid] << "):  " 
                      << x_data[vid]  << ", " << y_data[vid] << std::endl;
         }
      }
      std::cout << std::endl;
   }

   MPI_Barrier(MPI_COMM_WORLD);
}
