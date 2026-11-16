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
#include <string>
#include <utility>
#include <vector>

using namespace mfem;

TEST_CASE("MFEM Mesh Named Attributes", "[Mesh]")
{
   // Path relative to the directory tests/unit
   Mesh mesh("data/compass-testing.mesh");

   REQUIRE(mesh.Dimension() == 2);
   REQUIRE(mesh.GetNE() == 12);
   REQUIRE(mesh.GetNV() == 13);

   REQUIRE(mesh.attribute_sets.attr_sets.Size() == 16);
   REQUIRE(mesh.bdr_attribute_sets.attr_sets.Size() == 13);

   std::vector<std::pair<std::string, std::vector<int>>> expected_attr_sets =
   {
      {"Base",  {9}},
      {"E Even",  {16}},
      {"E Odd",  {17}},
      {"East",  {16, 17}},
      {"N Even",  {10}},
      {"N Odd",  {11}},
      {"North",  {10, 11}},
      {"Rose",  {10, 11, 12, 13, 14, 15, 16, 17}},
      {"Rose Even",  {10, 12, 14, 16}},
      {"Rose Odd",  {11, 13, 15, 17}},
      {"S Even",  {14}},
      {"S Odd",  {15}},
      {"South",  {14, 15}},
      {"W Even",  {12}},
      {"W Odd",  {13}},
      {"West",  {12, 13}}
   };

   for (auto const &attr_name_index_pair: expected_attr_sets )
   {
      REQUIRE(mesh.attribute_sets.AttributeSetExists(
                 attr_name_index_pair.first));

      auto const &attr_set = mesh.attribute_sets.GetAttributeSet(
                                attr_name_index_pair.first);
      auto const &expected_attr_set = attr_name_index_pair.second;

      REQUIRE(static_cast<std::size_t>(attr_set.Size()) ==
              expected_attr_set.size());

      bool const elements_equal = std::equal(attr_set.begin(), attr_set.end(),
                                             expected_attr_set.begin());

      REQUIRE(elements_equal);
   }

   std::vector<std::pair<std::string, std::vector<int>>> expected_bdr_attr_sets
   =
   {
      {"Boundary",  {1, 2, 3, 4, 5, 6, 7, 8}},
      {"ENE",  { 1}},
      {"ESE",  { 8}},
      {"Eastern Boundary",  {1, 8}},
      {"NNE",  { 2}},
      {"NNW",  { 3}},
      {"Northern Boundary", {2, 3}},
      {"SSE",  { 7}},
      {"SSW",  { 6}},
      {"Southern Boundary", {6,7}},
      {"WNW",  { 4}},
      {"WSW",  { 5}},
      {"Western Boundary", {4,5}}
   };

   for (auto const &attr_bdr_name_index_pair: expected_bdr_attr_sets )
   {
      REQUIRE(mesh.bdr_attribute_sets.AttributeSetExists(
                 attr_bdr_name_index_pair.first));

      auto const &bdr_attr_set = mesh.bdr_attribute_sets.GetAttributeSet(
                                    attr_bdr_name_index_pair.first);
      auto const &expected_bdr_attr_set = attr_bdr_name_index_pair.second;

      REQUIRE(static_cast<std::size_t>(bdr_attr_set.Size()) ==
              expected_bdr_attr_set.size());

      bool const elements_equal = std::equal(bdr_attr_set.begin(),
                                             bdr_attr_set.end(),
                                             expected_bdr_attr_set.begin());

      REQUIRE(elements_equal);
   }
}

TEST_CASE("MFEM NC mesh v1.2 named attributes", "[Mesh]")
{
   Mesh file_mesh("data/nc-mesh-v1.2-attribute-sets.mesh");
   REQUIRE(file_mesh.Nonconforming());
   REQUIRE(file_mesh.attribute_sets.AttributeSetExists("materials"));
   REQUIRE(file_mesh.bdr_attribute_sets.AttributeSetExists("walls"));

   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
   mesh.EnsureNCMesh();
   const real_t scale = GENERATE(0.5, 0.3);
   const bool curved = GENERATE(false, true);
   Array<Refinement> refs(1);
   refs[0] = Refinement(0, Refinement::XY, scale);
   mesh.GeneralRefinement(refs);
   if (curved) { mesh.SetCurvature(2); }
   mesh.attribute_sets.CreateAttributeSet("materials").Append(1);
   mesh.bdr_attribute_sets.CreateAttributeSet("walls").Append(1);

   std::stringstream output;
   output.precision(17);
   mesh.Print(output);
   REQUIRE(output.str().rfind("MFEM NC mesh v1.2", 0) == 0);

   Mesh loaded(output);
   REQUIRE(loaded.Nonconforming());
   REQUIRE(loaded.attribute_sets.AttributeSetExists("materials"));
   REQUIRE(loaded.bdr_attribute_sets.AttributeSetExists("walls"));
   REQUIRE(loaded.attribute_sets.GetAttributeSet("materials").Size() == 1);
   REQUIRE(loaded.bdr_attribute_sets.GetAttributeSet("walls").Size() == 1);
   REQUIRE(loaded.GetNE() == mesh.GetNE());
   Vector vertices, loaded_vertices;
   mesh.GetVertices(vertices);
   loaded.GetVertices(loaded_vertices);
   REQUIRE(vertices.Size() == loaded_vertices.Size());
   vertices -= loaded_vertices;
   REQUIRE(vertices.Normlinf() < 1e-14);
}

TEST_CASE("MFEM NC mesh automatically serializes refinement scales", "[Mesh]")
{
   const real_t scale = GENERATE(0.5, 0.3);
   const bool force_scaled = GENERATE(false, true);
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
   mesh.EnsureNCMesh();
   Array<Refinement> refs(1);
   refs[0] = Refinement(0, Refinement::XY, scale);
   mesh.GeneralRefinement(refs);
   if (force_scaled) { mesh.SetScaledNCMesh(); }

   std::stringstream output;
   output.precision(17);
   mesh.Print(output);
   const char *version = (force_scaled || scale != 0.5) ?
                         "MFEM NC mesh v1.1" : "MFEM NC mesh v1.0";
   REQUIRE(output.str().rfind(version, 0) == 0);

   Mesh loaded(output);
   REQUIRE(loaded.GetNE() == mesh.GetNE());
   Vector vertices, loaded_vertices;
   mesh.GetVertices(vertices);
   loaded.GetVertices(loaded_vertices);
   REQUIRE(vertices.Size() == loaded_vertices.Size());
   vertices -= loaded_vertices;
   REQUIRE(vertices.Normlinf() < 1e-14);
}

TEST_CASE("MeshPart preserves named sets in its constructed mesh", "[Mesh]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
   mesh.attribute_sets.CreateAttributeSet("materials").Append(1);
   mesh.attribute_sets.AddToAttributeSet("materials", 2);
   mesh.bdr_attribute_sets.CreateAttributeSet("walls").Append(1);
   mesh.bdr_attribute_sets.AddToAttributeSet("walls", 3);
   if (GENERATE(false, true)) { mesh.SetCurvature(2); }

   int partition[] = {0, 1};
   MeshPartitioner partitioner(mesh, 2, partition);
   MeshPart part;
   for (int i = 0; i < 2; i++)
   {
      partitioner.ExtractPart(i, part);
      Mesh &local = part.GetMesh();
      std::stringstream output;
      part.Print(output);
      Mesh loaded(output);
      for (Mesh *result : {&local, &loaded})
      {
         REQUIRE(result->attribute_sets.AttributeSetExists("materials"));
         const Array<int> &materials =
            result->attribute_sets.GetAttributeSet("materials");
         REQUIRE(materials.Size() == 2);
         REQUIRE(materials[0] == 1);
         REQUIRE(materials[1] == 2);
         REQUIRE(result->bdr_attribute_sets.AttributeSetExists("walls"));
         const Array<int> &walls =
            result->bdr_attribute_sets.GetAttributeSet("walls");
         REQUIRE(walls.Size() == 2);
         REQUIRE(walls[0] == 1);
         REQUIRE(walls[1] == 3);
      }
   }
}
