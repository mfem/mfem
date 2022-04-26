// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
using namespace mfem;

#include "unit_tests.hpp"

TEST_CASE("Expression Construction", "[EntityIndices]")
{
   Array<int> a = {1,2,3};
   MeshConnections::EntityIndices idx = {1, false, a};
   MeshConnections::EntityIndices idx2 = {1, false, {1,2,3}};
   REQUIRE(idx.indices[2] == 3);
   REQUIRE(idx2.indices[2] == 3);
}

TEST_CASE("Matching Hand Picked Indices", "[MeshConnections]")
{
   SECTION("2x2 Quad Mesh")
   {
      Mesh mesh("./data/quad_2x2.mesh");
      MeshConnections::EntityIndices result;

      SECTION("Cells to Vertices")
      {
         result.dim = 0; result.boundary = false;
         mesh.connect.ChildrenOfEntity({2,false,0}, result);
         REQUIRE(result.indices == Array<int> {0,1,3,4});
         mesh.connect.ChildrenOfEntity({2,false,1}, result);
         REQUIRE(result.indices == Array<int> {1,2,4,5});
         mesh.connect.ChildrenOfEntity({2,false,2}, result);
         REQUIRE(result.indices == Array<int> {3,4,7,8});
         mesh.connect.ChildrenOfEntity({2,false,3}, result);
         REQUIRE(result.indices == Array<int> {4,5,8,9});
      }

      SECTION("Cells to Boundary Edges")
      {
         result.dim = 1; result.boundary = true;
         mesh.connect.ChildrenOfEntity({2,false,0}, result);
         REQUIRE(result.indices == Array<int> {0,4});
         mesh.connect.ChildrenOfEntity({2,false,1}, result);
         REQUIRE(result.indices == Array<int> {1,5});
         mesh.connect.ChildrenOfEntity({2,false,2}, result);
         REQUIRE(result.indices == Array<int> {2,6});
         mesh.connect.ChildrenOfEntity({2,false,3}, result);
         REQUIRE(result.indices == Array<int> {3,7});
      }

      SECTION("Cell Neighbors Across Vertices")
      {
         result.dim = 2; result.boundary = false;
         mesh.connect.NeighborsOfEntity({2,false,0}, 0, result);
         REQUIRE(result.indices == Array<int> {1,2,3});
         mesh.connect.NeighborsOfEntity({2,false,1}, 0, result);
         REQUIRE(result.indices == Array<int> {0,2,3});
         mesh.connect.NeighborsOfEntity({2,false,2}, 0, result);
         REQUIRE(result.indices == Array<int> {0,1,3});
         mesh.connect.NeighborsOfEntity({2,false,3}, 0, result);
         REQUIRE(result.indices == Array<int> {0,1,2});
      }

      SECTION("Boundary Edge Neighbors Across Vertices")
      {
         result.dim = 1; result.boundary = true;
         mesh.connect.NeighborsOfEntity({1,true,0}, 0, result);
         REQUIRE(result.indices == Array<int> {1,4});
         mesh.connect.NeighborsOfEntity({1,true,1}, 0, result);
         REQUIRE(result.indices == Array<int> {0,5});
         mesh.connect.NeighborsOfEntity({1,true,2}, 0, result);
         REQUIRE(result.indices == Array<int> {3,6});
         mesh.connect.NeighborsOfEntity({1,true,3}, 0, result);
         REQUIRE(result.indices == Array<int> {2,7});
         mesh.connect.NeighborsOfEntity({1,true,4}, 0, result);
         REQUIRE(result.indices == Array<int> {0,6});
         mesh.connect.NeighborsOfEntity({1,true,5}, 0, result);
         REQUIRE(result.indices == Array<int> {1,7});
         mesh.connect.NeighborsOfEntity({1,true,6}, 0, result);
         REQUIRE(result.indices == Array<int> {2,4});
         mesh.connect.NeighborsOfEntity({1,true,7}, 0, result);
         REQUIRE(result.indices == Array<int> {3,5});
      }
   }
}

