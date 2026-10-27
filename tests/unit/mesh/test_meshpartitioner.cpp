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

#include <algorithm>
#include <sstream>

#ifdef MFEM_USE_MPI

using namespace mfem;

static Mesh CreateMesh(Element::Type geom)
{
   switch (geom)
   {
      case Element::SEGMENT:
         return Mesh::MakeCartesian1D(16);
         break;
      case Element::TRIANGLE:
      case Element::QUADRILATERAL:
         return Mesh::MakeCartesian2D(4, 4, geom);
         break;
      case Element::TETRAHEDRON:
      case Element::HEXAHEDRON:
      case Element::WEDGE:
      case Element::PYRAMID:
         return Mesh::MakeCartesian3D(2, 2, 2, geom);
         break;
      default:
         MFEM_ABORT("");
   }
}

static void ComparePMeshes(ParMesh &pmesh1, ParMesh &pmesh2)
{
   REQUIRE(pmesh1.Conforming() == pmesh2.Conforming());
   REQUIRE(pmesh1.GetNE() == pmesh2.GetNE());
   if (pmesh1.GetNodes())
   {
      REQUIRE(pmesh2.GetNodes());
      REQUIRE(pmesh1.GetNodes()->Size() == pmesh2.GetNodes()->Size());
      Vector x = *pmesh1.GetNodes();
      x -= *pmesh2.GetNodes();
      REQUIRE(x.Normlinf() <= 1e-16);
   }
   else
   {
      REQUIRE(!pmesh2.GetNodes());
      Vector x1, x2;
      pmesh1.GetVertices(x1);
      pmesh2.GetVertices(x2);
      REQUIRE(x1.Size() == x2.Size());
      x1 -= x2;
      REQUIRE(x1.Normlinf() <= 1e-16);
   }
   auto &etoe1 = pmesh1.ElementToElementTable();
   auto &etoe2 = pmesh2.ElementToElementTable();
   REQUIRE(etoe1.Size() == etoe2.Size());
   REQUIRE(etoe1.Size_of_connections() == etoe2.Size_of_connections());
   REQUIRE(etoe1.Width() == etoe2.Width());
}

TEST_CASE("Conforming MakeFromSerial", "[Mesh] [MeshPartitioner] [Parallel]")
{
   // 1. Create a serial random conforming mesh
   // 2. Create a partition of the mesh, use that to make a pmesh
   // 3. Create a MeshPart for our local rank
   // 4. Compare the result
   auto geom = GENERATE(Element::SEGMENT, Element::TRIANGLE,
                        Element::QUADRILATERAL, Element::TETRAHEDRON,
                        Element::HEXAHEDRON, Element::WEDGE, Element::PYRAMID);
   int rank = Mpi::WorldRank();
   int nprocs = Mpi::WorldSize();
   CAPTURE(geom, rank);

   auto mesh = CreateMesh(geom);
   std::vector<int> partition(mesh.GetNE());
   // evenly divide all elements
   for (int i = 0, j = 0; j < mesh.GetNE(); ++j)
   {
      partition[j] = i;
      i = (i + 1) % nprocs;
   }
   // mesh which only exists on rank 0
   Mesh smesh;
   std::vector<int> spartition;
   if (rank == 0)
   {
      smesh = CreateMesh(geom);
      spartition.resize(mesh.GetNE());
      for (int i = 0, j = 0; j < smesh.GetNE(); ++j)
      {
         spartition[j] = i;
         i = (i + 1) % nprocs;
      }
   }
   SECTION("Low order")
   {
      REQUIRE(mesh.GetNE() > 0);
      ParMesh pmesh1(MPI_COMM_WORLD, mesh, partition.data());
      mesh.Clear();
      ParMesh pmesh2 =
         ParMesh::MakeFromSerial(MPI_COMM_WORLD, smesh, spartition.data());
      // shouldn't need to clear smesh
      REQUIRE(smesh.GetNE() == 0);
      ComparePMeshes(pmesh1, pmesh2);
   }
   SECTION("High order")
   {
      REQUIRE(mesh.GetNE() > 0);
      mesh.SetCurvature(2);
      if (rank == 0)
      {
         smesh.SetCurvature(2);
      }
      ParMesh pmesh1(MPI_COMM_WORLD, mesh, partition.data());
      mesh.Clear();
      ParMesh pmesh2 =
         ParMesh::MakeFromSerial(MPI_COMM_WORLD, smesh, spartition.data());
      // shouldn't need to clear smesh
      REQUIRE(smesh.GetNE() == 0);
      ComparePMeshes(pmesh1, pmesh2);
   }
}

TEST_CASE("Nonconforming MakeFromSerial", "[Mesh] [MeshPartitioner] [Parallel]")
{
   // 1. Create a serial random nonconforming mesh, with multiple levels of
   // refinement
   // 2. Create a partition of the mesh, use that to make a pmesh
   // 3. Create a MeshPart for our local rank
   // 4. Compare the result
   // TODO: can't create pyramid NC mesh yet?
   auto geom =
      GENERATE(Element::SEGMENT, Element::TRIANGLE, Element::QUADRILATERAL,
               Element::TETRAHEDRON, Element::HEXAHEDRON, Element::WEDGE);
   int rank = Mpi::WorldRank();
   int nprocs = Mpi::WorldSize();
   CAPTURE(geom, rank);
   auto mesh = CreateMesh(geom);
   auto ndims = mesh.Dimension();
   auto ref_type = ndims == 1 ? Refinement::X :
                   (ndims == 2 ? Refinement::XY : Refinement::XYZ);

   mesh.EnsureNCMesh(true);
   REQUIRE(mesh.Nonconforming());
   Array<Refinement> refinements;
   // non-conforming refinements
   for (int level = 0; level < 2; ++level)
   {
      refinements.SetSize(mesh.GetNE() / 2);
      for (int j = 0; j < refinements.Size(); ++j)
      {
         REQUIRE(2 * j < mesh.GetNE());
         refinements[j] = Refinement(2 * j, ref_type);
      }
      mesh.GeneralRefinement(refinements, 1);
   }

   std::vector<int> partition(mesh.GetNE());
   // evenly divide all elements
   for (int i = 0, j = 0; j < mesh.GetNE(); ++j)
   {
      partition[j] = i;
      i = (i + 1) % nprocs;
   }

   // mesh which only exists on rank 0
   Mesh smesh;
   std::vector<int> spartition;
   if (rank == 0)
   {
      smesh = CreateMesh(geom);
      smesh.EnsureNCMesh(true);
      REQUIRE(smesh.Nonconforming());
      // non-conforming refinements
      for (int level = 0; level < 2; ++level)
      {
         refinements.SetSize(smesh.GetNE() / 2);
         for (int j = 0; j < refinements.Size(); ++j)
         {
            REQUIRE(2 * j < smesh.GetNE());
            refinements[j] = Refinement(2 * j, ref_type);
         }
         smesh.GeneralRefinement(refinements, 1);
      }
      spartition.resize(smesh.GetNE());
      for (int i = 0, j = 0; j < smesh.GetNE(); ++j)
      {
         spartition[j] = i;
         i = (i + 1) % nprocs;
      }
   }

   SECTION("Low order")
   {
      ParMesh pmesh1(MPI_COMM_WORLD, mesh, partition.data());
      mesh.Clear();
      ParMesh pmesh2 =
         ParMesh::MakeFromSerial(MPI_COMM_WORLD, smesh, spartition.data());
      // shouldn't need to clear smesh
      REQUIRE(smesh.GetNE() == 0);
      ComparePMeshes(pmesh1, pmesh2);
   }
   SECTION("High order")
   {
      mesh.SetCurvature(2);
      if (rank == 0)
      {
         smesh.SetCurvature(2);
      }
      ParMesh pmesh1(MPI_COMM_WORLD, mesh, partition.data());
      mesh.Clear();
      ParMesh pmesh2 =
         ParMesh::MakeFromSerial(MPI_COMM_WORLD, smesh, spartition.data());
      // shouldn't need to clear smesh
      REQUIRE(smesh.GetNE() == 0);
      ComparePMeshes(pmesh1, pmesh2);
   }
}

TEST_CASE("MakeFromSerial preserves serialized mesh metadata",
          "[Mesh] [MeshPartitioner] [Parallel]")
{
   const int rank = Mpi::WorldRank();
   const int nprocs = Mpi::WorldSize();

   Mesh smesh;
   std::stringstream input;
   if (rank == 0)
   {
      smesh = Mesh::MakeCartesian2D(std::max(2, nprocs), 1,
                                    Element::QUADRILATERAL);
      smesh.EnsureNCMesh();
      Array<Refinement> ref(1);
      ref[0] = Refinement(0, Refinement::XY, 0.3);
      smesh.GeneralRefinement(ref);
      smesh.SetScaledNCMesh();
      smesh.attribute_sets.CreateAttributeSet("material").Append(1);
      smesh.bdr_attribute_sets.CreateAttributeSet("boundary").Append(1);
      smesh.Print(input);
      REQUIRE(input.str().rfind("MFEM NC mesh v1.2", 0) == 0);
   }

   ParMesh pmesh = ParMesh::MakeFromSerial(MPI_COMM_WORLD, input,
                                           1, 0, false);
   REQUIRE(pmesh.Nonconforming());
   REQUIRE(pmesh.attribute_sets.AttributeSetExists("material"));
   REQUIRE(pmesh.bdr_attribute_sets.AttributeSetExists("boundary"));
}

TEST_CASE("MakeFromSerial supports empty curved parts",
          "[Mesh] [MeshPartitioner] [Parallel]")
{
   if (Mpi::WorldSize() < 2) { return; }

   const int rank = Mpi::WorldRank();
   Mesh smesh;
   std::vector<int> partition;
   if (rank == 0)
   {
      smesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
      smesh.SetCurvature(2);
      partition.assign(smesh.GetNE(), 0);
   }

   ParMesh pmesh = ParMesh::MakeFromSerial(MPI_COMM_WORLD, smesh,
                                           partition.data());
   REQUIRE(pmesh.GetNodes());
   REQUIRE(pmesh.GetNodes()->VectorDim() == 2);
   if (rank != 0) { REQUIRE(pmesh.GetNE() == 0); }
}

TEST_CASE("MakeFromSerial prunes anisotropic 3D NC parts",
          "[Mesh] [MeshPartitioner] [Parallel]")
{
   const int rank = Mpi::WorldRank();
   const int nprocs = Mpi::WorldSize();
   if (nprocs < 2) { return; }

   Mesh smesh;
   std::vector<int> partition;
   int global_ne = 0;
   if (rank == 0)
   {
      smesh = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
      smesh.EnsureNCMesh();
      Array<Refinement> refs(1);
      refs[0] = Refinement(0, Refinement::X);
      smesh.GeneralRefinement(refs);
      refs[0] = Refinement(0, Refinement::Y);
      smesh.GeneralRefinement(refs);

      global_ne = smesh.GetNE();
      partition.resize(global_ne);
      for (int i = 0; i < global_ne; i++) { partition[i] = i % nprocs; }
   }
   MPI_Bcast(&global_ne, 1, MPI_INT, 0, MPI_COMM_WORLD);

   ParMesh pmesh = ParMesh::MakeFromSerial(MPI_COMM_WORLD, smesh,
                                           partition.data());
   REQUIRE(pmesh.Nonconforming());

   int local_ne = pmesh.GetNE();
   int distributed_ne;
   MPI_Allreduce(&local_ne, &distributed_ne, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   REQUIRE(distributed_ne == global_ne);
}

#endif
