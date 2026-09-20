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

using namespace mfem;

/// Create a mesh with one triangle, with different element vertices
static Mesh *BuildSingleTriMesh(const int shift)
{
   // Reproduce mfem/data/ref-triangle.mesh layout but with custom vertex order
   std::ostringstream os;
   os << "MFEM mesh v1.0\n\n";
   os << "dimension\n2\n\n";
   os << "elements\n1\n";
   os << "1 2";
   for (int i = 0; i < 3; i++)
   {
      os << " "<<(i + shift)%3;
   }
   os << "\n\n";
   os << "boundary\n3\n";
   os << "1 1 0 1\n";
   os << "2 1 1 2\n";
   os << "3 1 2 0\n\n";
   os << "vertices\n3\n2\n";
   os << "0 0\n";
   os << "1 0\n";
   os << "0 1\n";
   os << "\n";
   std::istringstream is(os.str());
   return new Mesh(is, 1, 0, false);
}

/// Create a mesh with one tetrahadron, with different element vertices
static Mesh *BuildSingleTetMesh(const int shift)
{
   // Reproduce mfem/data/ref-tetrahedron.mesh layout but with custom vertex order
   std::ostringstream os;
   os << "MFEM mesh v1.0\n\n";
   os << "dimension\n3\n\n";
   os << "elements\n1\n";
   // Element: attribute=1, geometry=4 (TETRAHEDRON), vertices 0 1 2 3
   os << "1 4";
   for (int i = 0; i < 4; i++)
   {
      os << " "<<(i + shift)%4;
   }
   os << "\n\n";
   // Boundary: 4 triangular faces with attributes 1..4
   os << "boundary\n4\n";
   os << "1 2 1 2 3\n";
   os << "2 2 0 3 2\n";
   os << "3 2 0 1 3\n";
   os << "4 2 0 2 1\n\n";
   // Vertices (4) with dimension flag 3
   os << "vertices\n4\n3\n";
   os << "0 0 0\n";
   os << "1 0 0\n";
   os << "0 1 0\n";
   os << "0 0 1\n";
   os << "\n";
   std::istringstream is(os.str());
   return new Mesh(is, 1, 0, false);
}

/// Create a mesh with one wedge, with different element vertices
static Mesh *BuildSingleWedgeMesh(const int shift)
{
   // Reproduce mfem/data/ref-tetrahedron.mesh layout but with custom vertex order
   std::ostringstream os;
   os << "MFEM mesh v1.0\n\n";
   os << "dimension\n3\n\n";
   os << "elements\n1\n";
   // Element: attribute=1, geometry=6 (PRISM), vertices 0 1 2 3 4 5
   os << "1 6";
   for (int i = 0; i < 3; i++)
   {
      os << " "<<(i + shift)%3;
   }
   for (int i = 0; i < 3; i++)
   {
      os << " "<<((i + shift)%3) + 3;
   }
   os << "\n\n";
   // Boundary: 2 triangular faces & 3 quad faces
   os << "boundary\n5\n";
   os << "1 2 0 1 2\n";
   os << "2 3 0 3 4 1\n";
   os << "3 3 1 4 5 2\n";
   os << "4 3 2 5 3 0\n";
   os << "5 2 3 4 5\n\n";
   // Vertices (4) with dimension flag 3
   os << "vertices\n6\n3\n";
   os << "0 0 0\n";
   os << "1 0 0\n";
   os << "0 1 0\n";
   os << "0 0 1\n";
   os << "1 0 1\n";
   os << "0 1 1\n";
   os << "\n";
   std::istringstream is(os.str());
   return new Mesh(is, 1, 0, false);
}

/// Create a mesh with one pyramid, with different element vertices
static Mesh *BuildSinglePyrMesh(const int shift)
{
   std::ostringstream os;
   os << "MFEM mesh v1.0\n\n";
   os << "dimension\n3\n\n";
   os << "elements\n1\n";
   // Element: attribute=1, geometry=7 (PYRAMID), vertices 0 1 2 3 4
   os << "1 7";
   for (int i = 0; i < 4; i++)
   {
      os << " "<<(i + shift)%4;
   }
   os << " 4 \n\n";
   // Boundary:1 quad face & 4 triangular faces
   os << "boundary\n5\n";
   os << "1 3 0 1 2 3\n";
   os << "2 2 0 1 4\n";
   os << "3 2 1 2 4\n";
   os << "4 2 2 3 4\n";
   os << "5 2 3 0 4\n\n";
   // Vertices (4) with dimension flag 3
   os << "vertices\n5\n3\n";
   os << "0 0 0\n";
   os << "1 0 0\n";
   os << "1 1 0\n";
   os << "0 1 0\n";
   os << "0 0 1\n";
   os << "\n";
   std::istringstream is(os.str());
   return new Mesh(is, 1, 0, false);
}

static const DenseMatrix & GetMetric(Mesh *mesh)
{
   // One element mesh, pick centroid of reference element (triangle/tetra)
   ElementTransformation *T = mesh->GetElementTransformation(0);
   IntegrationPoint ip;
   if (mesh->Dimension() == 2)
   {
      ip.Set2(1.0/3.0, 1.0/3.0);
   }
   else if (mesh->Dimension() == 3)
   {
      ip.Set3(1.0/4.0, 1.0/4.0, 1.0/4.0);
   }
   else
   {
      MFEM_ABORT("Unsupported mesh dimension in tau_variance test.");
   }
   T->SetIntPoint(&ip);
   return T->Metric();
}


TEST_CASE("Metrics for different Finite Elements",
          "[Triangle]"
          "[Tetrahedron]"
          "[Wedge]"
          "[Pyramid]")
{
   Mesh *mesh = nullptr;
   SECTION("Triangle")
   {
      mesh = BuildSingleTriMesh(0);
      DenseMatrix Gij_ref = GetMetric(mesh);
      delete mesh;
      for (size_t i = 1; i < 3; i++)
      {
         mesh = BuildSingleTriMesh(i);
         DenseMatrix Gij = GetMetric(mesh);
         Gij.Print();
         Gij -= Gij_ref;
         Gij.Print();
         REQUIRE(Gij.FNorm() == MFEM_Approx(0.0));
         delete mesh;
      }
   }

   SECTION("Tetrahedron")
   {
      mesh = BuildSingleTetMesh(0);
      DenseMatrix Gij_ref = GetMetric(mesh);
      delete mesh;
      for (size_t i = 1; i < 4; i++)
      {
         mesh = BuildSingleTetMesh(i);
         DenseMatrix Gij = GetMetric(mesh);
         delete mesh;
         Gij -= Gij_ref;
         REQUIRE(Gij.FNorm() == MFEM_Approx(0.0));
      }
   }

   SECTION("Wedge")
   {
      mesh = BuildSingleWedgeMesh(0);
      DenseMatrix Gij_ref = GetMetric(mesh);
      delete mesh;
      for (size_t i = 1; i < 3; i++)
      {
         mesh = BuildSingleWedgeMesh(i);
         DenseMatrix Gij = GetMetric(mesh);
         delete mesh;
         Gij -= Gij_ref;
         REQUIRE(Gij.FNorm() == MFEM_Approx(0.0));
      }
   }

   SECTION("Pyramid")
   {
      mesh = BuildSinglePyrMesh(0);
      DenseMatrix Gij_ref = GetMetric(mesh);
      delete mesh;
      for (size_t i = 1; i < 4; i++)
      {
         mesh = BuildSinglePyrMesh(i);
         DenseMatrix Gij = GetMetric(mesh);
         delete mesh;
         Gij -= Gij_ref;
         REQUIRE(Gij.FNorm() == MFEM_Approx(0.0));
      }
   }
}
