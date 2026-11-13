// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#include <sstream>
#include <string>

using namespace mfem;

namespace
{

// Nodes of the reference tetrahedron (tags 1-4) and of its edge midpoints (tags
// 5-10), in Gmsh 2.2 format.
const std::string gmsh22_header_and_nodes = R"($MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
10
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 0.5 0 0
6 0.5 0.5 0
7 0 0.5 0
8 0 0 0.5
9 0 0.5 0.5
10 0.5 0 0.5
$EndNodes
)";

// Linear tetrahedron with second-order (6-node) boundary triangles, as produced
// for example by Gmsh when optimizing a second-order mesh with Netgen.
const std::string gmsh22_tet4_tri6 = gmsh22_header_and_nodes + R"($Elements
5
1 4 2 1 1 1 2 3 4
2 9 2 2 1 1 3 2 7 6 5
3 9 2 3 2 1 2 4 5 10 8
4 9 2 4 3 1 4 3 8 9 7
5 9 2 5 4 2 3 4 6 9 10
$EndElements
)";

// Second-order (10-node) tetrahedron with linear boundary triangles.
const std::string gmsh22_tet10_tri3 = gmsh22_header_and_nodes + R"($Elements
5
1 11 2 1 1 1 2 3 4 5 6 7 8 9 10
2 2 2 2 1 1 3 2
3 2 2 3 2 1 2 4
4 2 2 4 3 1 4 3
5 2 2 5 4 2 3 4
$EndElements
)";

// Linear tetrahedron with second-order boundary triangles in Gmsh 4.1 format,
// with the boundary elements listed before the volume element.
const std::string gmsh41_tet4_tri6 = R"($MeshFormat
4.1 0 8
$EndMeshFormat
$Entities
0 0 1 1
1 0 0 0 1 1 1 1 2 0
1 0 0 0 1 1 1 1 1 1 1
$EndEntities
$Nodes
1 10 1 10
3 1 0 10
1
2
3
4
5
6
7
8
9
10
0 0 0
1 0 0
0 1 0
0 0 1
0.5 0 0
0.5 0.5 0
0 0.5 0
0 0 0.5
0 0.5 0.5
0.5 0 0.5
$EndNodes
$Elements
2 5 1 5
2 1 9 4
2 1 3 2 7 6 5
3 1 2 4 5 10 8
4 1 4 3 8 9 7
5 2 3 4 6 9 10
3 1 4 1
1 1 2 3 4
$EndElements
)";

Mesh ReadMesh(const std::string &mesh_str)
{
   std::istringstream iss(mesh_str);
   return Mesh(iss);
}

} // namespace

TEST_CASE("Gmsh mesh with boundary elements of different order",
          "[Mesh][Gmsh]")
{
   SECTION("Linear volume, second-order boundary")
   {
      const std::string &mesh_str = GENERATE(gmsh22_tet4_tri6,
                                             gmsh41_tet4_tri6);
      Mesh mesh = ReadMesh(mesh_str);

      // The mesh is linear: the high-order boundary nodes are discarded.
      REQUIRE(mesh.Dimension() == 3);
      REQUIRE(mesh.GetNE() == 1);
      REQUIRE(mesh.GetNBE() == 4);
      REQUIRE(mesh.GetNV() == 4);
      REQUIRE(mesh.GetNodes() == nullptr);
      REQUIRE(mesh.GetElementVolume(0) == MFEM_Approx(1.0/6.0));
   }

   SECTION("Second-order volume, linear boundary")
   {
      Mesh mesh = ReadMesh(gmsh22_tet10_tri3);

      // The mesh is second-order, following the volume element.
      REQUIRE(mesh.Dimension() == 3);
      REQUIRE(mesh.GetNE() == 1);
      REQUIRE(mesh.GetNBE() == 4);
      REQUIRE(mesh.GetNV() == 4);
      REQUIRE(mesh.GetNodes() != nullptr);
      REQUIRE(mesh.GetNodes()->FESpace()->GetMaxElementOrder() == 2);
      REQUIRE(mesh.GetElementVolume(0) == MFEM_Approx(1.0/6.0));
   }
}
