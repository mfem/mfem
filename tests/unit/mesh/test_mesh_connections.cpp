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

bool cmp_set(const Array<int> &a, const Array<int> &b);


TEST_CASE("Expression Construction", "[EntityIndices]")
{
   Array<int> a = {1,2,3};
   EntityIndices idx = {1, AllIdx, a};
   EntityIndices idx2 = {1, AllIdx, Array<int>{1,2,3}};
   REQUIRE(idx.indices[2] == 3);
   REQUIRE(idx2.indices[2] == 3);
}

TEST_CASE("Matching Hand Picked Indices", "[MeshConnections]")
{
   SECTION("2x2 Quad Mesh")
   {
      Mesh mesh("./data/quad_2x2.mesh");
      EntityIndices result;

      SECTION("Cells to Vertices")
      {
         result = {0, AllIdx, Array<int>{}};
         mesh.connect->ChildrenOfEntity({2,AllIdx,0}, result);
         REQUIRE(result.indices == Array<int>{0,1,4,3});
         mesh.connect->ChildrenOfEntity({2,AllIdx,1}, result);
         REQUIRE(result.indices == Array<int>{1,2,5,4});
         mesh.connect->ChildrenOfEntity({2,AllIdx,2}, result);
         REQUIRE(result.indices == Array<int>{3,4,7,6});
         mesh.connect->ChildrenOfEntity({2,AllIdx,3}, result);
         REQUIRE(result.indices == Array<int>{4,5,8,7});
      }

      SECTION("Cells to Boundary Edges")
      {
         result = {1, BdrIdx, Array<int>{}};
         mesh.connect->ChildrenOfEntity({2,AllIdx,0}, result);
         REQUIRE(result.indices == Array<int>{4,0});
         mesh.connect->ChildrenOfEntity({2,AllIdx,1}, result);
         REQUIRE(result.indices == Array<int>{1,5});
         mesh.connect->ChildrenOfEntity({2,AllIdx,2}, result);
         REQUIRE(result.indices == Array<int>{2,6});
         mesh.connect->ChildrenOfEntity({2,AllIdx,3}, result);
         REQUIRE(result.indices == Array<int>{7,3});
      }

      SECTION("Boundary Edges to Verts")
      {
         result = {0, AllIdx, Array<int>{}};
         mesh.connect->ChildrenOfEntity({1,BdrIdx,0}, result);
         REQUIRE(result.indices == Array<int>{0,1});
         mesh.connect->ChildrenOfEntity({1,BdrIdx,1}, result);
         REQUIRE(result.indices == Array<int>{1,2});
         mesh.connect->ChildrenOfEntity({1,BdrIdx,2}, result);
         REQUIRE(result.indices == Array<int>{7,6});
         mesh.connect->ChildrenOfEntity({1,BdrIdx,3}, result);
         REQUIRE(result.indices == Array<int>{8,7});
         mesh.connect->ChildrenOfEntity({1,BdrIdx,4}, result);
         REQUIRE(result.indices == Array<int>{3,0});
         mesh.connect->ChildrenOfEntity({1,BdrIdx,5}, result);
         REQUIRE(result.indices == Array<int>{2,5});
         mesh.connect->ChildrenOfEntity({1,BdrIdx,6}, result);
         REQUIRE(result.indices == Array<int>{6,3});
         mesh.connect->ChildrenOfEntity({1,BdrIdx,7}, result);
         REQUIRE(result.indices == Array<int>{5,8});         
      }

      SECTION("Verts to Cells")
      {
         result = {2, AllIdx, Array<int>{}};
         mesh.connect->ParentsOfEntity({0,AllIdx,0}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0}));
         mesh.connect->ParentsOfEntity({0,AllIdx,1}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,1}));
         mesh.connect->ParentsOfEntity({0,AllIdx,2}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1}));
         mesh.connect->ParentsOfEntity({0,AllIdx,3}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,2}));
         mesh.connect->ParentsOfEntity({0,AllIdx,4}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,1,2,3}));
         mesh.connect->ParentsOfEntity({0,AllIdx,5}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,3}));
         mesh.connect->ParentsOfEntity({0,AllIdx,6}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{2}));
         mesh.connect->ParentsOfEntity({0,AllIdx,7}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{2,3}));
         mesh.connect->ParentsOfEntity({0,AllIdx,8}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{3}));
      }

      SECTION("Verts to Boundary Edges")
      {
         result = {1, BdrIdx, Array<int>{}};
         mesh.connect->ParentsOfEntity({0,AllIdx,0}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,4}));
         mesh.connect->ParentsOfEntity({0,AllIdx,1}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,1}));
         mesh.connect->ParentsOfEntity({0,AllIdx,2}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,5}));
         mesh.connect->ParentsOfEntity({0,AllIdx,3}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{4,6}));
         mesh.connect->ParentsOfEntity({0,AllIdx,4}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{}));
         mesh.connect->ParentsOfEntity({0,AllIdx,5}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{5,7}));
         mesh.connect->ParentsOfEntity({0,AllIdx,6}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{2,6}));
         mesh.connect->ParentsOfEntity({0,AllIdx,7}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{2,3}));
         mesh.connect->ParentsOfEntity({0,AllIdx,8}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{3,7}));
      }

      SECTION("Boundary Edges to Cells")
      {
         result = {2, AllIdx, Array<int>{}};
         mesh.connect->ParentsOfEntity({1,BdrIdx,0}, result);
         REQUIRE(result.indices == Array<int>{0});
         mesh.connect->ParentsOfEntity({1,BdrIdx,1}, result);
         REQUIRE(result.indices == Array<int>{1});
         mesh.connect->ParentsOfEntity({1,BdrIdx,2}, result);
         REQUIRE(result.indices == Array<int>{2});
         mesh.connect->ParentsOfEntity({1,BdrIdx,3}, result);
         REQUIRE(result.indices == Array<int>{3});
         mesh.connect->ParentsOfEntity({1,BdrIdx,4}, result);
         REQUIRE(result.indices == Array<int>{0});
         mesh.connect->ParentsOfEntity({1,BdrIdx,5}, result);
         REQUIRE(result.indices == Array<int>{1});
         mesh.connect->ParentsOfEntity({1,BdrIdx,6}, result);
         REQUIRE(result.indices == Array<int>{2});
         mesh.connect->ParentsOfEntity({1,BdrIdx,7}, result);
         REQUIRE(result.indices == Array<int>{3});         
      }      

      SECTION("Cell Neighbors Across Vertices")
      {
         result = {2, AllIdx, Array<int>{}};
         mesh.connect->NeighborsOfEntity({2,AllIdx,0}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,2,3}));
         mesh.connect->NeighborsOfEntity({2,AllIdx,1}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,2,3}));
         mesh.connect->NeighborsOfEntity({2,AllIdx,2}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,1,3}));
         mesh.connect->NeighborsOfEntity({2,AllIdx,3}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,1,2}));
      }

      SECTION("Boundary Edge Neighbors Across Vertices")
      {
         result = {1, BdrIdx, Array<int>{}};
         mesh.connect->NeighborsOfEntity({1,BdrIdx,0}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,4}));
         mesh.connect->NeighborsOfEntity({1,BdrIdx,1}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,5}));
         mesh.connect->NeighborsOfEntity({1,BdrIdx,2}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{3,6}));
         mesh.connect->NeighborsOfEntity({1,BdrIdx,3}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{2,7}));
         mesh.connect->NeighborsOfEntity({1,BdrIdx,4}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,6}));
         mesh.connect->NeighborsOfEntity({1,BdrIdx,5}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,7}));
         mesh.connect->NeighborsOfEntity({1,BdrIdx,6}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{2,4}));
         mesh.connect->NeighborsOfEntity({1,BdrIdx,7}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{3,5}));
      }

      SECTION("Vertex Neighbors Across Cells")
      {
         result = {0, AllIdx, Array<int>{}};
         mesh.connect->NeighborsOfEntity({0,AllIdx,0}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,3,4}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,1}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,2,3,4,5}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,2}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,4,5}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,3}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,1,4,6,7}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,4}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,1,2,3,5,6,7,8}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,5}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,2,4,7,8}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,6}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{3,4,7}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,7}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{3,4,5,6,8}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,8}, 0, result);
         REQUIRE(cmp_set(result.indices, Array<int>{4,5,7}));  
      }

      SECTION("Vertex Neighbors Across Edges")
      {
         result = {0, AllIdx, Array<int>{}};
         mesh.connect->NeighborsOfEntity({0,AllIdx,0}, 1, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,3}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,1}, 1, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,2,4}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,2}, 1, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,5}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,3}, 1, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,4,6}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,4}, 1, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,3,5,7}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,5}, 1, result);
         REQUIRE(cmp_set(result.indices, Array<int>{2,4,8}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,6}, 1, result);
         REQUIRE(cmp_set(result.indices, Array<int>{3,7}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,7}, 1, result);
         REQUIRE(cmp_set(result.indices, Array<int>{4,6,8}));
         mesh.connect->NeighborsOfEntity({0,AllIdx,8}, 1, result);
         REQUIRE(cmp_set(result.indices, Array<int>{5,7}));  
      }

      SECTION("Children of Multiple Entities")
      {
         result = {0, AllIdx, Array<int>{}};
         mesh.connect->ChildrenOfEntities({1,BdrIdx,{2,3}}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{6,7,8}));
         mesh.connect->ChildrenOfEntities({2,AllIdx,{0,1}}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{0,1,2,3,4,5}));

         result = {1, BdrIdx, Array<int>{}};
         mesh.connect->ChildrenOfEntities({2,AllIdx,{2,3}}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{2,3,6,7}));
      }

      SECTION("Parents of Any Entities")
      {
         result = {2, AllIdx, Array<int>{}};
         mesh.connect->ParentsOfAnyEntities({1,BdrIdx,{1,2}}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,2}));
         mesh.connect->ParentsOfAnyEntities({0,AllIdx,{5,6,7,8}}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{1,2,3}));

         result = {1, BdrIdx, Array<int>{}};
         mesh.connect->ParentsOfAnyEntities({0,AllIdx,{3,4,6,7}}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{2,3,4,6}));
      }

      SECTION("Parents Covered By Entities")
      {
         result = {2, AllIdx, Array<int>{}};
         mesh.connect->ParentsCoveredByEntities({0,AllIdx,{3,4,5,6,7}}, result);
         REQUIRE(cmp_set(result.indices, Array<int>{2}));
      }
   }
}


bool cmp_set(const Array<int> &a, const Array<int> &b)
{
   if (a.Size() != b.Size())
   {
      return false;
   }

   bool same_set = true;
   for (int i = 0; i < a.Size(); ++i)
   {
      if (std::find(b.begin(), b.end(), a[i]) == b.end())
      {
         same_set = false;
         break;
      }
   }
   return same_set;
}