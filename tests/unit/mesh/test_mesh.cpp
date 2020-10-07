// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

TEST_CASE("Element-wise construction", "[Mesh]")
{
   SECTION("Quadrilateral")
   {
      const int numVertices = 9;
      const int numElements = 4;

      Mesh mesh(2, numVertices, numElements);

      // Add each vertex by coordinates
      for (int i=0; i<3; ++i)
         for (int j=0; j<3; ++j)
         {
            mesh.AddVertex(i, j);
         }

      // Add each element by vertices
      const int geom = Geometry::SQUARE;

      Array<int> elvert(4);

      Element *el = mesh.NewElement(geom);
      elvert[0] = 0; elvert[1] = 1; elvert[2] = 3; elvert[3] = 4;
      el->SetVertices(elvert);
      mesh.AddElement(el);

      el = mesh.NewElement(geom);
      elvert[0] = 1; elvert[1] = 2; elvert[2] = 4; elvert[3] = 5;
      el->SetVertices(elvert);
      mesh.AddElement(el);

      el = mesh.NewElement(geom);
      elvert[0] = 3; elvert[1] = 4; elvert[2] = 6; elvert[3] = 7;
      el->SetVertices(elvert);
      mesh.AddElement(el);

      el = mesh.NewElement(geom);
      elvert[0] = 4; elvert[1] = 5; elvert[2] = 7; elvert[3] = 8;
      el->SetVertices(elvert);
      mesh.AddElement(el);

      mesh.FinalizeTopology();

      REQUIRE(numVertices == mesh.GetNV());
      REQUIRE(numElements == mesh.GetNE());
   }
}

TEST_CASE("Gecko integration in MFEM", "[Mesh]")
{
   Array<int> perm;

   SECTION("Permutation from Gecko is valid")
   {
      Array<bool> elem_covered;

      SECTION("Hex meshes")
      {
         Mesh mesh(3, 4, 5, Element::HEXAHEDRON);
         mesh.GetGeckoElementOrdering(perm);
         REQUIRE(perm.Size() == mesh.GetNE());
         REQUIRE(perm.Min() == 0);
         REQUIRE(perm.Max() == mesh.GetNE() - 1);

         elem_covered.SetSize(perm.Size(), false);
         for (int i = 0; i < perm.Size(); ++i)
         {
            elem_covered[perm[i]] = true;
         }

         bool all_elems_covered = true;
         for (int i = 0; i < perm.Size(); ++i)
         {
            all_elems_covered &= elem_covered[i];
         }
         REQUIRE(all_elems_covered);
      }

      SECTION("Tet meshes")
      {
         Mesh mesh(5, 4, 3, Element::TETRAHEDRON);
         mesh.GetGeckoElementOrdering(perm);
         REQUIRE(perm.Size() == mesh.GetNE());
         REQUIRE(perm.Min() == 0);
         REQUIRE(perm.Max() == mesh.GetNE() - 1);

         elem_covered.SetSize(perm.Size(), false);
         for (int i = 0; i < perm.Size(); ++i)
         {
            elem_covered[perm[i]] = true;
         }

         bool all_elems_covered = true;
         for (int i = 0; i < perm.Size(); ++i)
         {
            all_elems_covered &= elem_covered[i];
         }
         REQUIRE(all_elems_covered);
      }
   }

   SECTION("Reorder preserves physical vertex locations")
   {
      Mesh mesh(3, 4, 5, Element::HEXAHEDRON);
      Mesh mesh_reordered(3, 4, 5, Element::HEXAHEDRON);
      mesh_reordered.GetGeckoElementOrdering(perm);
      mesh_reordered.ReorderElements(perm);

      for (int old_elid = 0; old_elid < perm.Size(); ++old_elid)
      {
         int new_elid = perm[old_elid];
         Array<int> old_dofs, new_dofs;
         mesh.GetElementVertices(old_elid, old_dofs);
         mesh_reordered.GetElementVertices(new_elid, new_dofs);
         for (int dofi = 0; dofi < old_dofs.Size(); ++dofi)
         {
            for (int d = 0; d < 3; ++d)
            {
               REQUIRE(mesh.GetVertex(old_dofs[dofi])[d] == mesh_reordered.GetVertex(
                          new_dofs[dofi])[d]);
            }
         }
      }
   }
}
