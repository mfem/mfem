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
#include "unit_tests.hpp"
#include <algorithm>

using namespace mfem;

template<typename T>
bool InArray(const T* begin, size_t sz, T i)
{
   const T *end = begin + sz;
   return std::find(begin, end, i) != end;
}

bool IndicesAreConnected(const Table &t, int i, int j)
{
   return InArray(t.GetRow(i), t.RowSize(i), j)
          && InArray(t.GetRow(j), t.RowSize(j), i);
}

TEST_CASE("Periodic mesh", "[Mesh]")
{
   int n = 3;
   SECTION("1D periodic mesh")
   {
      Mesh orig_mesh = Mesh::MakeCartesian1D(n);
      std::vector<Vector> translations = {Vector({1.0})};
      Mesh mesh = Mesh::MakePeriodic(
                     orig_mesh,
                     orig_mesh.CreatePeriodicVertexMapping(translations));
      REQUIRE(mesh.GetNV() == n);
      const Table &e2e = mesh.ElementToElementTable();
      REQUIRE(IndicesAreConnected(e2e, 0, 2));
      REQUIRE(IndicesAreConnected(e2e, 0, 1));
      REQUIRE(IndicesAreConnected(e2e, 1, 2));
   }
   SECTION("2D periodic mesh")
   {
      auto el = GENERATE(Element::TRIANGLE, Element::QUADRILATERAL);
      bool sfc = false; // <-- Lexicographic instead of SFC ordering
      Mesh orig_mesh = Mesh::MakeCartesian2D(n, n, el, false, 1.0, 1.0, sfc);
      std::vector<Vector> translations = {Vector({1.0,0.0}), Vector({0.0,1.0})};
      Mesh mesh = Mesh::MakePeriodic(
                     orig_mesh,
                     orig_mesh.CreatePeriodicVertexMapping(translations));
      REQUIRE(mesh.GetNV() == pow(n-1,2) + 2*(n-1) + 1);
      if (el == Element::QUADRILATERAL)
      {
         const Table &e2e = mesh.ElementToElementTable();
         for (int i=0; i<n; ++i)
         {
            // Bottom row connected to top row
            REQUIRE(IndicesAreConnected(e2e, i, i + n*(n-1)));
            // Left column connected to right column
            REQUIRE(IndicesAreConnected(e2e, i*n, n-1 + i*n));
         }
      }
   }
   SECTION("3D periodic mesh")
   {
      auto el = GENERATE(Element::TETRAHEDRON, Element::HEXAHEDRON, Element::WEDGE);
      bool sfc = false; // <-- Lexicographic instead of SFC ordering
      Mesh orig_mesh = Mesh::MakeCartesian3D(n, n, n, el, 1.0, 1.0, 1.0, sfc);
      std::vector<Vector> translations =
      {
         Vector({1.0, 0.0, 0.0}),
         Vector({0.0, 1.0, 0.0}),
         Vector({0.0, 0.0, 1.0})
      };
      Mesh mesh = Mesh::MakePeriodic(
                     orig_mesh,
                     orig_mesh.CreatePeriodicVertexMapping(translations));
      REQUIRE(mesh.GetNV() == pow(n-1,3) + 3*pow(n-1,2) + 3*(n-1) + 1);
      if (el == Element::HEXAHEDRON)
      {
         const Table &e2e = mesh.ElementToElementTable();
         int n2 = n*n;
         for (int j=0; j<n; ++j)
         {
            for (int i=0; i<n; ++i)
            {
               // z=0 face connected to z=1 face
               REQUIRE(IndicesAreConnected(e2e, i + j*n, i + j*n + n2*(n-1)));
               // y=0 face connected to y=1 face
               REQUIRE(IndicesAreConnected(e2e, i + j*n2, i + j*n2 + n*(n-1)));
               // x=0 face connected to x=1 face
               REQUIRE(IndicesAreConnected(e2e, i*n + j*n2, i*n + j*n2 + n-1));
            }
         }
      }
   }
}
