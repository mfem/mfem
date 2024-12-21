// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

TEST_CASE("Element-wise construction", "[Mesh]")
{
   SECTION("Quadrilateral")
   {
      const int numVertices = 9;
      const int numElements = 4;

      Mesh mesh(2, numVertices, numElements);

      // Add each vertex by coordinates
      for (int j=0; j<3; ++j)
      {
         for (int i=0; i<3; ++i)
         {
            mesh.AddVertex(i, j);
         }
      }

      // Add each element by vertices
      const int geom = Geometry::SQUARE;

      Array<int> elvert(4);

      Element *el = mesh.NewElement(geom);
      elvert[0] = 0; elvert[1] = 1; elvert[2] = 4; elvert[3] = 3;
      el->SetVertices(elvert);
      REQUIRE(el->GetAttribute() == 1);
      mesh.AddElement(el);

      el = mesh.NewElement(geom);
      elvert[0] = 1; elvert[1] = 2; elvert[2] = 5; elvert[3] = 4;
      el->SetVertices(elvert);
      REQUIRE(el->GetAttribute() == 1);
      mesh.AddElement(el);

      el = mesh.NewElement(geom);
      elvert[0] = 3; elvert[1] = 4; elvert[2] = 7; elvert[3] = 6;
      el->SetVertices(elvert);
      REQUIRE(el->GetAttribute() == 1);
      mesh.AddElement(el);

      el = mesh.NewElement(geom);
      elvert[0] = 4; elvert[1] = 5; elvert[2] = 8; elvert[3] = 7;
      el->SetVertices(elvert);
      REQUIRE(el->GetAttribute() == 1);
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
         Mesh mesh = Mesh::MakeCartesian3D(3, 4, 5, Element::HEXAHEDRON);
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
         Mesh mesh = Mesh::MakeCartesian3D(5, 4, 3, Element::TETRAHEDRON);
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
      Mesh mesh = Mesh::MakeCartesian3D(3, 4, 5, Element::HEXAHEDRON);
      Mesh mesh_reordered = Mesh::MakeCartesian3D(3, 4, 5, Element::HEXAHEDRON);
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

TEST_CASE("MakeSimplicial", "[Mesh]")
{
   auto mesh_fname = GENERATE("../../data/star.mesh",
                              "../../data/inline-quad.mesh",
                              "../../data/inline-hex.mesh",
                              "../../data/inline-wedge.mesh",
                              "../../data/beam-wedge.mesh");

   Mesh orig_mesh(mesh_fname, 1, 1);
   Mesh simplex_mesh = Mesh::MakeSimplicial(orig_mesh);

   Geometry::Type orig_geom = orig_mesh.GetElementBaseGeometry(0);
   int factor;
   switch (orig_geom)
   {
      case Geometry::SQUARE: factor = 2; break;
      case Geometry::PRISM: factor = 3; break;
      case Geometry::CUBE: factor = 6; break;
      default: factor = 1;
   }

   int dim = orig_mesh.Dimension();
   Geometry::Type simplex_geom
      = (dim == 2) ? Geometry::TRIANGLE : Geometry::TETRAHEDRON;

   Array<Geometry::Type> geoms;
   simplex_mesh.GetGeometries(simplex_mesh.Dimension(), geoms);

   REQUIRE(geoms.Size() == 1);
   REQUIRE(geoms[0] == simplex_geom);
   // Note: assuming no hex is subdivided into 5 tets. This can happen depending
   // on the original mesh, but it doesn't happen for these test cases.
   REQUIRE(simplex_mesh.GetNE() == orig_mesh.GetNE()*factor);
}

TEST_CASE("MakeNurbs", "[Mesh]")
{
   Array<real_t> intervals_array({1, 1, 1});
   Vector intervals(intervals_array.GetData(), intervals_array.Size());
   Array<int> continuity({-1, 1, 1, -1});
   {
      const KnotVector kv(2, intervals, continuity);
      REQUIRE(kv.GetNE() == 3);
      REQUIRE(kv.GetNCP() == 5);
      REQUIRE(kv.GetOrder() == 2);
      REQUIRE(kv.Size() == 8);
   }
   {
      const KnotVector kv(3, intervals, continuity);
      REQUIRE(kv.GetNE() == 3);
      REQUIRE(kv.GetNCP() == 8);
      REQUIRE(kv.GetOrder() == 3);
      REQUIRE(kv.Size() == 12);
   }
   const KnotVector kv(2, intervals, continuity);
   Array<real_t> grev_pts({0.0, 1.0/6.0, 0.5, 5.0/6.0, 1.0});

   Array<NURBSPatch *> patches;

   // Will build and test on multiple NURBS meshes. Cleans up and
   // resets the patches array, which is assumed to be initially
   // populated for the particular test case.
   const auto test_nurbs_extension = [&](Mesh& patch_topology)
   {
      NURBSExtension ne(&patch_topology, patches);
      Mesh mesh(ne);
      GridFunction *nodes = mesh.GetNodes();
      REQUIRE(nodes != NULL);
      FiniteElementSpace *fe = nodes->FESpace();
      REQUIRE(fe != NULL);
      SparseMatrix p(fe->GetNDofs(), fe->GetNDofs());
      SparseMatrix r(fe->GetNDofs(), fe->GetNDofs());
      for (int i = 0; i < fe->GetNDofs(); ++i)
      {
         p.Add(i, i, 1);
         r.Add(i, i, 1);
      }
      p.Finalize();
      r.Finalize();
      fe->SetProlongation(p);
      fe->SetRestriction(r);
      for (int i=0; i<patches.Size(); i++) { delete patches[i]; }
      patches.SetSize(0);
   };

   // Bi-variate 2D test:
   Array<real_t> pts_2d(3 * kv.GetNCP() * kv.GetNCP());
   int count = 0;
   for (int j = 0; j < kv.GetNCP(); ++j)
      for (int i = 0; i < kv.GetNCP(); ++i)
      {
         pts_2d[count + 0] = grev_pts[i];
         pts_2d[count + 1] = grev_pts[j];
         pts_2d[count + 2] = 1;
         count += 3;
      }
   patches.Append(new NURBSPatch(&kv, &kv, 3, pts_2d.GetData()));
   Mesh patch_topology_2d =
      Mesh::MakeCartesian2D(1, 1, Element::Type::QUADRILATERAL);
   test_nurbs_extension(patch_topology_2d);

   // Tri-variate 3D test:
   Array<real_t> pts_3d(4 * kv.GetNCP() * kv.GetNCP() * kv.GetNCP());
   count = 0;
   for (int k = 0; k < kv.GetNCP(); ++k)
      for (int j = 0; j < kv.GetNCP(); ++j)
         for (int i = 0; i < kv.GetNCP(); ++i)
         {
            pts_3d[count + 0] = grev_pts[i];
            pts_3d[count + 1] = grev_pts[j];
            pts_3d[count + 2] = grev_pts[k];
            pts_3d[count + 3] = 1;
            count += 4;
         }
   patches.Append(new NURBSPatch(&kv, &kv, &kv, 4, pts_3d.GetData()));
   Mesh patch_topology_3d =
      Mesh::MakeCartesian3D(1, 1, 1, Element::Type::HEXAHEDRON);
   test_nurbs_extension(patch_topology_3d);
}
