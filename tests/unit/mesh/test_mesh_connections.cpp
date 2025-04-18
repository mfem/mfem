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
using namespace mfem;

#include "unit_tests.hpp"

bool cmp_set(const Array<int> &a, const Array<int> &b);

TEST_CASE("Matching Hand Picked Indices", "[Mesh]")
{
   SECTION("2x2 Quad Mesh")
   {
      Mesh mesh("./data/quad_2x2.mesh");
      Array<int> result;

      SECTION("Vertex to Elems")
      {
         mesh.ElemsWithVert(0, result);
         REQUIRE(cmp_set(result, Array<int> {0}));
         mesh.ElemsWithVert(1, result);
         REQUIRE(cmp_set(result, Array<int> {0,1}));
         mesh.ElemsWithVert(2, result);
         REQUIRE(cmp_set(result, Array<int> {1}));
         mesh.ElemsWithVert(3, result);
         REQUIRE(cmp_set(result, Array<int> {0,2}));
         mesh.ElemsWithVert(4, result);
         REQUIRE(cmp_set(result, Array<int> {0,1,2,3}));
         mesh.ElemsWithVert(5, result);
         REQUIRE(cmp_set(result, Array<int> {1,3}));
         mesh.ElemsWithVert(6, result);
         REQUIRE(cmp_set(result, Array<int> {2}));
         mesh.ElemsWithVert(7, result);
         REQUIRE(cmp_set(result, Array<int> {2,3}));
         mesh.ElemsWithVert(8, result);
         REQUIRE(cmp_set(result, Array<int> {3}));
      }

      SECTION("Vertex to Edges")
      {
         mesh.EdgesWithVert(0, result);
         REQUIRE(cmp_set(result, Array<int> {0,3}));
         mesh.EdgesWithVert(1, result);
         REQUIRE(cmp_set(result, Array<int> {0,1,4}));
         mesh.EdgesWithVert(2, result);
         REQUIRE(cmp_set(result, Array<int> {4,5}));
         mesh.EdgesWithVert(3, result);
         REQUIRE(cmp_set(result, Array<int> {2,3,9}));
         mesh.EdgesWithVert(4, result);
         REQUIRE(cmp_set(result, Array<int> {1,2,6,7}));
         mesh.EdgesWithVert(5, result);
         REQUIRE(cmp_set(result, Array<int> {5,6,10}));
         mesh.EdgesWithVert(6, result);
         REQUIRE(cmp_set(result, Array<int> {8,9}));
         mesh.EdgesWithVert(7, result);
         REQUIRE(cmp_set(result, Array<int> {7,8,11}));
         mesh.EdgesWithVert(8, result);
         REQUIRE(cmp_set(result, Array<int> {10,11}));
      }

      SECTION("Elements Covered by Vertices")
      {
         Array<int> elems;

         elems = {};
         mesh.ElemsWithAllVerts(Array<int> {0,1,2,3}, elems);
         REQUIRE(cmp_set(elems, Array<int> {}));
         mesh.ElemsWithAllVerts(Array<int> {0,1,2,3,4}, elems);
         REQUIRE(cmp_set(elems, Array<int> {0}));
         mesh.ElemsWithAllVerts(Array<int> {0,1,2,3,4,5}, elems);
         REQUIRE(cmp_set(elems, Array<int> {0,1}));
         mesh.ElemsWithAllVerts(Array<int> {0,1,2,3,4,5,6}, elems);
         REQUIRE(cmp_set(elems, Array<int> {0,1}));
         mesh.ElemsWithAllVerts(Array<int> {0,1,2,3,4,5,6,7}, elems);
         REQUIRE(cmp_set(elems, Array<int> {0,1,2}));
         mesh.ElemsWithAllVerts(Array<int> {0,1,2,3,4,5,6,7,8}, elems);
         REQUIRE(cmp_set(elems, Array<int> {0,1,2,3}));
      }

      SECTION("Edges Covered by Vertices")
      {
         Array<int> edges;

         edges = {};
         mesh.EdgesWithAllVerts(Array<int> {0}, edges);
         REQUIRE(cmp_set(edges, Array<int> {}));
         mesh.EdgesWithAllVerts(Array<int> {0,1}, edges);
         REQUIRE(cmp_set(edges, Array<int> {0}));
         mesh.EdgesWithAllVerts(Array<int> {0,1,2}, edges);
         REQUIRE(cmp_set(edges, Array<int> {0,4}));
         mesh.EdgesWithAllVerts(Array<int> {0,1,2,3}, edges);
         REQUIRE(cmp_set(edges, Array<int> {0,3,4}));
         mesh.EdgesWithAllVerts(Array<int> {0,1,2,3,4}, edges);
         REQUIRE(cmp_set(edges, Array<int> {0,1,2,3,4}));
         mesh.EdgesWithAllVerts(Array<int> {0,1,2,3,4,5}, edges);
         REQUIRE(cmp_set(edges, Array<int> {0,1,2,3,4,5,6}));
         mesh.EdgesWithAllVerts(Array<int> {0,1,2,3,4,5,6}, edges);
         REQUIRE(cmp_set(edges, Array<int> {0,1,2,3,4,5,6,9}));
         mesh.EdgesWithAllVerts(Array<int> {0,1,2,3,4,5,6,7}, edges);
         REQUIRE(cmp_set(edges, Array<int> {0,1,2,3,4,5,6,7,8,9}));
         mesh.EdgesWithAllVerts(Array<int> {0,1,2,3,4,5,6,7,8}, edges);
         REQUIRE(cmp_set(edges, Array<int> {0,1,2,3,4,5,6,7,8,9,10,11}));
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