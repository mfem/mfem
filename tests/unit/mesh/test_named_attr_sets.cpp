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

namespace mfem
{

Mesh CreateMesh()
{
   // Create simple cartesian mesh with two domains
   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   mesh.GetElement(2)->SetAttribute(2);
   mesh.GetElement(7)->SetAttribute(2);
   mesh.GetElement(8)->SetAttribute(2);
   mesh.GetElement(13)->SetAttribute(2);

   return mesh;
}

void CreateNCMesh(Mesh &mesh)
{
   Array<int> ref(4);
   ref[0] = 2; ref[1] = 7; ref[2] = 8; ref[3] = 13;

   mesh.EnsureNCMesh();
   mesh.GeneralRefinement(ref);
}

Mesh CreateNCMesh()
{
   Mesh mesh = CreateMesh();
   CreateNCMesh(mesh);
   return mesh;
}

void CreateNamedSets(Mesh &mesh)
{
   AttributeSets &attr_sets = mesh.attribute_sets;
   AttributeSets &bdr_attr_sets = mesh.bdr_attribute_sets;

   // Assign named attribute sets
   attr_sets.CreateAttributeSet("Frame");
   attr_sets.AddToAttributeSet("Frame", 1);
   attr_sets.CreateAttributeSet("Core");
   attr_sets.AddToAttributeSet("Core", 2);

   // Assign named boundary attribute sets
   bdr_attr_sets.CreateAttributeSet("Bottom");
   bdr_attr_sets.AddToAttributeSet("Bottom", 1);
   bdr_attr_sets.CreateAttributeSet("Top");
   bdr_attr_sets.AddToAttributeSet("Top", 3);
   bdr_attr_sets.CreateAttributeSet("Left");
   bdr_attr_sets.AddToAttributeSet("Left", 4);
   bdr_attr_sets.CreateAttributeSet("Right");
   bdr_attr_sets.AddToAttributeSet("Right", 2);
}

TEST_CASE("Named Attribute Sets in Mesh",
          "[Mesh]"
          "[NCMesh]")
{
   Mesh mesh = CreateMesh();

   // Define named attribute sets
   CreateNamedSets(mesh);

   AttributeSets &attr_sets = mesh.attribute_sets;
   AttributeSets &bdr_attr_sets = mesh.bdr_attribute_sets;

   // Copy the mesh and confirm that the sets copied correctly
   Mesh mesh_copy(mesh);

   REQUIRE(attr_sets.attr_sets == mesh_copy.attribute_sets.attr_sets);
   REQUIRE(bdr_attr_sets.attr_sets == mesh_copy.bdr_attribute_sets.attr_sets);

   // Save/Load the mesh and confirm the sets copied correctly
   std::ostringstream oss;
   mesh.Print(oss);

   std::istringstream iss(oss.str());
   Mesh mesh_load(iss);

   REQUIRE(attr_sets.attr_sets == mesh_load.attribute_sets.attr_sets);
   REQUIRE(bdr_attr_sets.attr_sets ==
           mesh_load.bdr_attribute_sets.attr_sets);

   // Create a non-conforming mesh and confirm that the sets are still there
   CreateNCMesh(mesh_copy);

   REQUIRE(attr_sets.attr_sets == mesh_copy.attribute_sets.attr_sets);
   REQUIRE(bdr_attr_sets.attr_sets == mesh_copy.bdr_attribute_sets.attr_sets);

   // Copy the non-conforming mesh and confirm the sets copied correctly
   Mesh ncmesh_copy(mesh_copy);

   REQUIRE(attr_sets.attr_sets == ncmesh_copy.attribute_sets.attr_sets);
   REQUIRE(bdr_attr_sets.attr_sets == ncmesh_copy.bdr_attribute_sets.attr_sets);
}

#ifdef MFEM_USE_MPI

TEST_CASE("Named Attribute Sets in ParMesh",
          "[Parallel]""[ParMesh]"
          "[ParNCMesh]")
{
   Mesh mesh = CreateMesh();

   // Define named sets in serial (or parallel)
   bool def_ser = GENERATE(false, true);

   if (def_ser)
   {
      // Define named attribute sets
      CreateNamedSets(mesh);
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   if (!def_ser)
   {
      // Define named attribute sets
      CreateNamedSets(pmesh);
   }

   AttributeSets &attr_sets = pmesh.attribute_sets;
   AttributeSets &bdr_attr_sets = pmesh.bdr_attribute_sets;

   // Copy the mesh and confirm that the sets copied correctly
   ParMesh pmesh_copy(pmesh);

   REQUIRE(attr_sets.attr_sets == pmesh_copy.attribute_sets.attr_sets);
   REQUIRE(bdr_attr_sets.attr_sets == pmesh_copy.bdr_attribute_sets.attr_sets);

   // Save/Load the mesh and confirm the sets copied correctly
   std::ostringstream oss;
   pmesh.ParPrint(oss);

   std::istringstream iss(oss.str());
   ParMesh pmesh_load(MPI_COMM_WORLD, iss);

   REQUIRE(attr_sets.attr_sets == pmesh_load.attribute_sets.attr_sets);
   REQUIRE(bdr_attr_sets.attr_sets ==
           pmesh_load.bdr_attribute_sets.attr_sets);
}

#endif // MFEM_USE_MPI

TEST_CASE("Named Attribute Sets in NCMesh",
          "[NCMesh]")
{
   Mesh ncmesh = CreateNCMesh();

   // Define named attribute sets
   CreateNamedSets(ncmesh);

   AttributeSets &attr_sets = ncmesh.attribute_sets;
   AttributeSets &bdr_attr_sets = ncmesh.bdr_attribute_sets;

   // Copy the ncmesh and confirm that the sets copied correctly
   Mesh ncmesh_copy(ncmesh);

   REQUIRE(attr_sets.attr_sets == ncmesh_copy.attribute_sets.attr_sets);
   REQUIRE(bdr_attr_sets.attr_sets == ncmesh_copy.bdr_attribute_sets.attr_sets);

   // Save/Load the ncmesh and confirm the sets copied correctly
   std::ostringstream oss;
   ncmesh.Print(oss);

   std::istringstream iss(oss.str());
   Mesh ncmesh_load(iss);

   REQUIRE(attr_sets.attr_sets == ncmesh_load.attribute_sets.attr_sets);
   REQUIRE(bdr_attr_sets.attr_sets ==
           ncmesh_load.bdr_attribute_sets.attr_sets);
}

#ifdef MFEM_USE_MPI

TEST_CASE("Named Attribute Sets in ParNCMesh",
          "[Parallel]"
          "[ParNCMesh]")
{
   Mesh ncmesh = CreateNCMesh();

   // Define named sets in serial (or parallel)
   bool def_ser = GENERATE(false, true);

   if (def_ser)
   {
      // Define named attribute sets
      CreateNamedSets(ncmesh);
   }

   ParMesh pncmesh(MPI_COMM_WORLD, ncmesh);
   ncmesh.Clear();

   if (!def_ser)
   {
      // Define named attribute sets
      CreateNamedSets(pncmesh);
   }

   AttributeSets &attr_sets = pncmesh.attribute_sets;
   AttributeSets &bdr_attr_sets = pncmesh.bdr_attribute_sets;

   // Copy the ncmesh and confirm that the sets copied correctly
   ParMesh pncmesh_copy(pncmesh);

   REQUIRE(attr_sets.attr_sets == pncmesh_copy.attribute_sets.attr_sets);
   REQUIRE(bdr_attr_sets.attr_sets ==
           pncmesh_copy.bdr_attribute_sets.attr_sets);

   // Save/Load the ncmesh and confirm the sets copied correctly
   std::ostringstream oss;
   pncmesh.ParPrint(oss);

   std::istringstream iss(oss.str());
   ParMesh pncmesh_load(MPI_COMM_WORLD, iss);

   REQUIRE(attr_sets.attr_sets == pncmesh_load.attribute_sets.attr_sets);
   REQUIRE(bdr_attr_sets.attr_sets ==
           pncmesh_load.bdr_attribute_sets.attr_sets);
}

#endif // MFEM_USE_MPI

} // namespace mfem
