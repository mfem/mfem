// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"
#include "fem/gslib.hpp"

using namespace mfem;

TEST_CASE("BBoxTensorGridMap 2D", "[BBoxTensorGridMap]")
{
   // Create a 2x2x2 mesh on [0,1]^D
   int dim  = GENERATE(2, 3);
   CAPTURE(dim);
   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   }
   else if (dim == 3)
   {
      mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   }

   // Create map with 4x4x4 grid
   int nx = 4;
   BBoxTensorGridMap map(mesh, nx);

   // Test each element's center
   Vector center(dim);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementCenter(e, center);
      Array<int> elems = map.MapPointToElements(center);
      REQUIRE(elems.Size() > 0);
      REQUIRE(elems[0] == e);
   }

   // Test point outside
   Vector pt(dim);
   pt = 0.0;
   pt(0) = 1.5;
   Array<int> elems = map.MapPointToElements(pt);
   REQUIRE(elems.Size() == 0);
}

TEST_CASE("BBoxTensorGridMap Boundary", "[BBoxTensorGridMap]")
{
   // Create a 1x1 quad mesh on [0,1]x[0,1]
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);

   // Create map with 2x2 grid
   int nx = 2;
   BBoxTensorGridMap map(mesh, nx);

   // Point exactly on the boundary of all 4 tensor grid cells
   Vector pt(2);
   pt(0) = 0.5; pt(1) = 0.5;
   int grid_cell = map.GetGridCellFromPoint(pt);
   REQUIRE(grid_cell == 3); // cell index should be 3 (top-right)

   // This should map to mesh element 2 (top-right)
   Array<int> elems = map.MapPointToElements(pt);
   REQUIRE(elems.Size() > 0);
   REQUIRE(elems[0] == 2);
}

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)
TEST_CASE("GlobalBBoxTensorGridMap Parallel",
          "[GlobalBBoxTensorGridMap][Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Create a mesh on [0,1]^D
   int dim  = GENERATE(2, 3);
   CAPTURE(dim);
   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   }
   else if (dim == 3)
   {
      mesh = Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON);
   }

   int nel = mesh.GetNE();
   Array<int> partitioning(mesh.GetNE());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      partitioning[e] = e % num_procs;
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh, partitioning.GetData());

   // Setup a list of points to find - center of each element
   Vector centers(nel*dim);
   for (int e = 0; e < nel; e++)
   {
      Vector center_el(centers.GetData() + e*dim, dim);
      mesh.GetElementCenter(e, center_el);
   }

   // Create map with 4x4x4 global grid
   int nx = 4;
   GlobalBBoxTensorGridMap map(pmesh, nx);

   // Test MapPointsToProcs - each point should map to the processor owning the
   // element
   std::map<int, std::vector<int>> pt_to_procs;
   map.MapPointsToProcs(centers, 1, pt_to_procs);

   REQUIRE(pt_to_procs.size() == nel);
   for (int i = 0; i < nel; i++)
   {
      std::vector<int> procs = pt_to_procs[i];
      REQUIRE(procs.size() > 0);
      REQUIRE(procs[0] == i % num_procs);
   }
}
#endif
