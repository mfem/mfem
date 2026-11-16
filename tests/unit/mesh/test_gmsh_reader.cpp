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

#include <array>
#include <iomanip>
#include <sstream>
#include <vector>

using namespace mfem;

namespace
{

/// For each node of the Gmsh pyramid of order @a p, in the Gmsh ordering, the
/// indices (i, j, k) such that the node is the point (i/p, j/p, k/p) of the
/// MFEM reference pyramid. The tables are the reference node coordinates
/// reported by Gmsh 4.13.1 (gmsh::model::mesh::getElementProperties) for the
/// element types 14 (order 2), 118 (order 3) and 119 (order 4), mapped from
/// the Gmsh reference pyramid (base [-1,1]^2, apex (0,0,1)) to the MFEM one.
const std::vector<std::array<int,3>> &GmshPyramidNodes(int p)
{
   static const std::vector<std::array<int,3>> nodes_2 =
   {
      {0,0,0}, {2,0,0}, {2,2,0}, {0,2,0}, {0,0,2}, {1,0,0}, {0,1,0}, {0,0,1},
      {2,1,0}, {1,0,1}, {1,2,0}, {1,1,1}, {0,1,1}, {1,1,0}
   };
   static const std::vector<std::array<int,3>> nodes_3 =
   {
      {0,0,0}, {3,0,0}, {3,3,0}, {0,3,0}, {0,0,3}, {1,0,0}, {2,0,0}, {0,1,0},
      {0,2,0}, {0,0,1}, {0,0,2}, {3,1,0}, {3,2,0}, {2,0,1}, {1,0,2}, {2,3,0},
      {1,3,0}, {2,2,1}, {1,1,2}, {0,2,1}, {0,1,2}, {1,0,1}, {0,1,1}, {2,1,1},
      {1,2,1}, {1,1,0}, {1,2,0}, {2,2,0}, {2,1,0}, {1,1,1}
   };
   static const std::vector<std::array<int,3>> nodes_4 =
   {
      {0,0,0}, {4,0,0}, {4,4,0}, {0,4,0}, {0,0,4}, {1,0,0}, {2,0,0}, {3,0,0},
      {0,1,0}, {0,2,0}, {0,3,0}, {0,0,1}, {0,0,2}, {0,0,3}, {4,1,0}, {4,2,0},
      {4,3,0}, {3,0,1}, {2,0,2}, {1,0,3}, {3,4,0}, {2,4,0}, {1,4,0}, {3,3,1},
      {2,2,2}, {1,1,3}, {0,3,1}, {0,2,2}, {0,1,3}, {1,0,1}, {2,0,1}, {1,0,2},
      {0,2,1}, {0,1,1}, {0,1,2}, {3,1,1}, {3,2,1}, {2,1,2}, {2,3,1}, {1,3,1},
      {1,2,2}, {1,1,0}, {1,3,0}, {3,3,0}, {3,1,0}, {1,2,0}, {2,3,0}, {3,2,0},
      {2,1,0}, {2,2,0}, {1,1,1}, {2,1,1}, {2,2,1}, {1,2,1}, {1,1,2}
   };
   switch (p)
   {
      case 2: return nodes_2;
      case 3: return nodes_3;
      case 4: return nodes_4;
   }
   MFEM_ABORT("No Gmsh pyramid table for order " << p);
}

/// A quadratic, orientation-preserving map of the reference pyramid. Every H1
/// pyramid element of order 2 and above represents it exactly, so the curved
/// pyramid read from Gmsh must reproduce it at every point.
void CurvedPyramidMap(real_t x, real_t y, real_t z, Vector &X)
{
   X.SetSize(3);
   X(0) = x + 0.1*y*z + 0.05*x*x;
   X(1) = y - 0.1*x*z + 0.05*x*y;
   X(2) = z + 0.1*x*y + 0.05*y*y;
}

/// Gmsh file (format 4.1 or 2.2, ASCII) with one curved pyramid of order @a p
/// whose nodes are the images of the reference nodes by CurvedPyramidMap.
std::string CurvedPyramidGmsh(int p, bool v41)
{
   const std::vector<std::array<int,3>> &nodes = GmshPyramidNodes(p);
   const int n = static_cast<int>(nodes.size());
   const int type = (p == 2) ? 14 : 115 + p;
   std::ostringstream msh;
   msh << std::setprecision(17);
   std::vector<Vector> coords(n);
   for (int i = 0; i < n; i++)
   {
      CurvedPyramidMap(real_t(nodes[i][0])/p, real_t(nodes[i][1])/p,
                       real_t(nodes[i][2])/p, coords[i]);
   }
   if (v41)
   {
      msh << "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
          << "$Entities\n0 0 0 1\n1 -1 -1 -1 2 2 2 1 1 0\n$EndEntities\n"
          << "$Nodes\n1 " << n << " 1 " << n << "\n3 1 0 " << n << "\n";
      for (int i = 0; i < n; i++) { msh << i + 1 << "\n"; }
      for (int i = 0; i < n; i++)
      {
         msh << coords[i](0) << " " << coords[i](1) << " " << coords[i](2) << "\n";
      }
      msh << "$EndNodes\n$Elements\n1 1 1 1\n3 1 " << type << " 1\n1";
   }
   else
   {
      msh << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n" << n << "\n";
      for (int i = 0; i < n; i++)
      {
         msh << i + 1 << " " << coords[i](0) << " " << coords[i](1) << " "
             << coords[i](2) << "\n";
      }
      msh << "$EndNodes\n$Elements\n1\n1 " << type << " 2 1 1";
   }
   for (int i = 0; i < n; i++) { msh << " " << i + 1; }
   msh << "\n$EndElements\n";
   return msh.str();
}

} // namespace

TEST_CASE("Gmsh curved pyramid", "[Mesh][Gmsh]")
{
   const int p = GENERATE(2, 3, 4);
   const bool v41 = GENERATE(true, false);
   CAPTURE(p, v41);

   std::istringstream msh(CurvedPyramidGmsh(p, v41));
   Mesh mesh(msh);

   REQUIRE(mesh.GetNE() == 1);
   REQUIRE(mesh.GetElementGeometry(0) == Geometry::PYRAMID);
   REQUIRE(mesh.GetNodes() != nullptr);
   REQUIRE(mesh.GetNodalFESpace()->GetElementOrder(0) == p);

   ElementTransformation &T = *mesh.GetElementTransformation(0);
   Vector X, X_exact;
   auto check_point = [&](const IntegrationPoint &ip)
   {
      CAPTURE(ip.x, ip.y, ip.z);
      T.SetIntPoint(&ip);
      T.Transform(ip, X);
      CurvedPyramidMap(ip.x, ip.y, ip.z, X_exact);
      X -= X_exact;
      REQUIRE(X.Normlinf() == MFEM_Approx(0.0));
   };

   // The curved pyramid passes through the Gmsh nodes ...
   for (const std::array<int,3> &node : GmshPyramidNodes(p))
   {
      IntegrationPoint ip;
      ip.Set3(real_t(node[0])/p, real_t(node[1])/p, real_t(node[2])/p);
      check_point(ip);
   }
   // ... and matches the map everywhere in between.
   const IntegrationRule &ir = IntRules.Get(Geometry::PYRAMID, 2*p);
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      check_point(ir.IntPoint(i));
   }

   // Exact volume of the image of the reference pyramid (the Jacobian
   // determinant is a cubic polynomial).
   const IntegrationRule &ir_vol = IntRules.Get(Geometry::PYRAMID, 8);
   real_t volume = 0.0;
   for (int i = 0; i < ir_vol.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir_vol.IntPoint(i);
      T.SetIntPoint(&ip);
      volume += ip.weight * T.Weight();
   }
   REQUIRE(volume == MFEM_Approx(5651.0/16000.0));
}

TEST_CASE("Gmsh periodic linear pyramids", "[Mesh][Gmsh]")
{
   // Two unit cubes along x, each split into 6 pyramids with the apex at the
   // cube center, periodic in x. The nodes of periodic meshes are set in an L2
   // space, whose pyramid (8 dofs at order 1) also has no dof per Gmsh node.
   const real_t nodes[14][3] =
   {
      {0,0,0}, {1,0,0}, {2,0,0}, {0,1,0}, {1,1,0}, {2,1,0}, {0,0,1}, {1,0,1},
      {2,0,1}, {0,1,1}, {1,1,1}, {2,1,1}, {0.5,0.5,0.5}, {1.5,0.5,0.5}
   };
   const int pyramids[12][5] =
   {
      {1,4,10,7,13}, {2,8,11,5,13}, {1,7,8,2,13}, {4,5,11,10,13},
      {1,2,5,4,13}, {7,10,11,8,13}, {2,5,11,8,14}, {3,9,12,6,14},
      {2,8,9,3,14}, {5,6,12,11,14}, {2,3,6,5,14}, {8,11,12,9,14}
   };
   std::ostringstream msh;
   msh << "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
       << "$Entities\n0 0 0 1\n1 0 0 0 2 1 1 1 1 0\n$EndEntities\n"
       << "$Nodes\n1 14 1 14\n3 1 0 14\n";
   for (int i = 1; i <= 14; i++) { msh << i << "\n"; }
   for (const auto &x : nodes) { msh << x[0] << " " << x[1] << " " << x[2] << "\n"; }
   msh << "$EndNodes\n$Elements\n1 12 1 12\n3 1 7 12\n";
   for (int e = 0; e < 12; e++)
   {
      msh << e + 1;
      for (int v : pyramids[e]) { msh << " " << v; }
      msh << "\n";
   }
   // Nodes 3, 6, 9, 12 (x = 2) are periodic images of 1, 4, 7, 10 (x = 0).
   msh << "$EndElements\n$Periodic\n1\n2 2 1\n0\n4\n3 1\n6 4\n9 7\n12 10\n"
       << "$EndPeriodic\n";

   std::istringstream iss(msh.str());
   Mesh mesh(iss);

   REQUIRE(mesh.GetNE() == 12);
   REQUIRE(mesh.GetNodes() != nullptr);

   // Every pyramid keeps the positions of its own vertices.
   const IntegrationRule &vertices = *Geometries.GetVertices(Geometry::PYRAMID);
   Vector X;
   for (int e = 0; e < 12; e++)
   {
      REQUIRE(mesh.GetElementGeometry(e) == Geometry::PYRAMID);
      ElementTransformation &T = *mesh.GetElementTransformation(e);
      for (int i = 0; i < 5; i++)
      {
         CAPTURE(e, i);
         const IntegrationPoint &ip = vertices.IntPoint(i);
         T.SetIntPoint(&ip);
         T.Transform(ip, X);
         const real_t *x = nodes[pyramids[e][i] - 1];
         REQUIRE(X(0) == MFEM_Approx(x[0]));
         REQUIRE(X(1) == MFEM_Approx(x[1]));
         REQUIRE(X(2) == MFEM_Approx(x[2]));
      }
   }
}
