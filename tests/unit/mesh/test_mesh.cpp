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
#include <array>
#include <cmath>
#include <random>
#include <vector>

using namespace mfem;

TEST_CASE("AttrToMarker", "[Mesh][AttributeSets]")
{
   SECTION("Empty attribute array")
   {
      Array<int> attrs;
      Array<int> marker = AttributeSets::AttrToMarker(5, attrs);
      REQUIRE(marker == Array<int>({0, 0, 0, 0, 0}));
   }

   SECTION("Non-empty attribute array")
   {
      Array<int> attrs({1, 3, 3, 5});
      Array<int> marker = AttributeSets::AttrToMarker(5, attrs);
      REQUIRE(marker == Array<int>({1, 0, 1, 0, 1}));
   }
}

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

TEST_CASE("Simplex meshes built and finalized marked by their longest edge",
          "[Mesh]")
{
   // FinalizeTriMesh() and FinalizeTetMesh() on a mesh built element by
   // element (as in ex7) mark its simplices for refinement: the edge between
   // the local vertices 0 and 1 of each is its longest, and a tetrahedron has
   // a refinement flag. A conforming refinement then bisects them, through
   // their longest edge, and keeps their volume.
   Mesh tri(2, 3, 1);
   tri.AddVertex(0.0, 0.0);
   tri.AddVertex(1.0, 0.0);
   tri.AddVertex(0.0, 3.0);
   tri.AddTriangle(0, 1, 2);
   tri.FinalizeTriMesh(1, 1, true);
   const int *v = tri.GetElement(0)->GetVertices();
   CHECK(((v[0] == 1 && v[1] == 2) || (v[0] == 2 && v[1] == 1)));

   Mesh tet(3, 4, 1);
   tet.AddVertex(0.0, 0.0, 0.0);
   tet.AddVertex(1.0, 0.0, 0.0);
   tet.AddVertex(0.0, 2.0, 0.0);
   tet.AddVertex(0.0, 0.0, 4.0);
   tet.AddTet(0, 1, 2, 3);
   tet.FinalizeTetMesh(1, 1, true);
   const int *w = tet.GetElement(0)->GetVertices();
   CHECK(((w[0] == 2 && w[1] == 3) || (w[0] == 3 && w[1] == 2)));
   const Tetrahedron *t = dynamic_cast<const Tetrahedron *>(tet.GetElement(0));
   REQUIRE(t != nullptr);
   CHECK(t->GetRefinementFlag() != 0);

   // The midpoints of the longest edges, and the volumes.
   const real_t mid[2][3] = {{0.5, 1.5, 0.0}, {0.0, 1.0, 2.0}};
   const real_t volume[2] = {1.5, 4.0/3.0};
   int m = 0;
   for (Mesh *mesh : {&tri, &tet})
   {
      Array<int> marked(1);
      marked[0] = 0;
      mesh->GeneralRefinement(marked, 0);
      CHECK(mesh->GetNE() > 1);
      bool found = false;
      for (int i = 0; i < mesh->GetNV(); i++)
      {
         const real_t *x = mesh->GetVertex(i);
         real_t d = 0.0;
         for (int k = 0; k < mesh->SpaceDimension(); k++)
         {
            d += (x[k] - mid[m][k])*(x[k] - mid[m][k]);
         }
         found = found || (d < 1e-24);
      }
      CHECK(found);
      real_t vol = 0.0;
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         vol += mesh->GetElementVolume(e);
      }
      CHECK((vol - volume[m])*(vol - volume[m]) < 1e-24);
      m++;
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

TEST_CASE("Hilbert reordering boundary element consistency", "[Mesh]")
{
   // After ReorderElements the boundary[] array must be sorted by the index of
   // the adjacent interior element (faces_info[be_to_face[i]].Elem1No).
   // This ensures spatial locality between volume and boundary elements.

   auto test = [](Mesh & mesh)
   {
      // Record the number of boundary elements before reordering
      const int nbe = mesh.GetNBE();
      REQUIRE(nbe > 0);

      Array<int> perm;
      mesh.GetHilbertElementOrdering(perm);
      mesh.ReorderElements(perm);

      REQUIRE(mesh.GetNBE() == nbe); // count must not change

      // Adjacent element indices must be non-decreasing across the boundary
      // element list.
      for (int i = 0; i < mesh.GetNBE() - 1; ++i)
      {
         int fi, fj, o;
         mesh.GetBdrElementFace(i,     &fi, &o);
         mesh.GetBdrElementFace(i + 1, &fj, &o);
         int eli, elj, dummy;
         mesh.GetFaceElements(fi, &eli, &dummy);
         mesh.GetFaceElements(fj, &elj, &dummy);
         REQUIRE(eli <= elj);
      }
   };

   SECTION("3D hex mesh boundary elements sorted after Hilbert reordering")
   {
      Mesh mesh = Mesh::MakeCartesian3D(3, 4, 5, Element::HEXAHEDRON);
      test(mesh);
   }

   SECTION("2D quad mesh boundary elements sorted after Hilbert reordering")
   {
      Mesh mesh = Mesh::MakeCartesian2D(4, 5, Element::QUADRILATERAL);
      test(mesh);
   }

   SECTION("3D tet mesh boundary elements sorted after Hilbert reordering")
   {
      Mesh mesh = Mesh::MakeCartesian3D(3, 4, 5, Element::TETRAHEDRON);
      test(mesh);
   }
}

TEST_CASE("MakeSimplicial", "[Mesh]")
{
   auto mesh_fname = GENERATE("../../data/star.mesh",
                              "../../data/star-surf.mesh",
                              "../../data/periodic-square.mesh",
                              "../../data/inline-tri.mesh",
                              "../../data/inline-quad.mesh",
                              "../../data/inline-hex.mesh",
                              "../../data/inline-tet.mesh",
                              "../../data/inline-wedge.mesh",
                              "../../data/periodic-cube.mesh",
                              "../../data/beam-wedge.mesh");
   CAPTURE(mesh_fname);

   Mesh orig_mesh(mesh_fname);
   Mesh simplex_mesh = Mesh::MakeSimplicial(orig_mesh);

   Geometry::Type orig_geom = orig_mesh.GetTypicalElementGeometry();
   const auto factor = [orig_geom]()
   {
      switch (orig_geom)
      {
         default: return 1; // No-op
         case Geometry::SQUARE: return 2;
         case Geometry::PRISM: return 3;
         case Geometry::CUBE: return 6;
      }
   }();

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

   auto *nodal_fes = orig_mesh.GetNodalFESpace();
   bool dg = nodal_fes != nullptr && nodal_fes->IsDGSpace();
   auto curvature = GENERATE(1,2,3);
   CAPTURE(curvature);
   orig_mesh.SetCurvature(curvature, dg, -1, GENERATE(0,1));
   auto ho_simplex_mesh = Mesh::MakeSimplicial(orig_mesh);

   CHECK(orig_mesh.GetNV() == simplex_mesh.GetNV());
   CHECK(orig_mesh.GetNV() == ho_simplex_mesh.GetNV());

   // Vertex locations should be unchanged after higher order transformation.
   Vector vert;
   for (int i = 0; i < ho_simplex_mesh.SpaceDimension(); i++)
   {
      ho_simplex_mesh.GetNodes()->GetNodalValues(vert, i+1);
      for (int j = 0; j < ho_simplex_mesh.GetNV(); j++)
      {
         REQUIRE(simplex_mesh.GetVertex(j)[i] - vert(j) == MFEM_Approx(0.0));
      }
   }
}

TEST_CASE("MakeMixedSimplicial", "[Mesh]")
{
   auto mesh_fname = "../../data/fichera-mixed-p2.mesh";

   Mesh orig_mesh(mesh_fname, 1, 1);
   Mesh simplex_mesh = Mesh::MakeSimplicial(orig_mesh);

   int ntet = 0, ncube = 0, nprism = 0;
   for (int i = 0; i < orig_mesh.GetNE(); i++)
   {
      auto geom_type = orig_mesh.GetElementGeometry(i);
      switch (geom_type)
      {
         case Geometry::Type::TETRAHEDRON : ntet++; break;
         case Geometry::Type::CUBE : ncube++; break;
         case Geometry::Type::PRISM : nprism++; break;
         default: break; // to calm compilers
      }
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
   REQUIRE(simplex_mesh.GetNE() == ntet + nprism * 3 + ncube * 6);

   CHECK(orig_mesh.GetNV() == simplex_mesh.GetNV());

   // Vertex locations should be unchanged after higher order transformation.
   Vector vert;
   constexpr real_t tol = 10*std::numeric_limits<real_t>::epsilon();
   for (int i = 0; i < simplex_mesh.SpaceDimension(); i++)
   {
      simplex_mesh.GetNodes()->GetNodalValues(vert, i+1);
      for (int j = 0; j < simplex_mesh.GetNV(); j++)
      {
         CHECK(std::abs(orig_mesh.GetVertex(j)[i] - vert(j)) < tol);
      }
   }
}

TEST_CASE("MakeSimplicial Surface Mesh", "[Mesh]")
{
   Mesh orig_mesh("../../data/star-surf.mesh");
   Mesh simplex_mesh = Mesh::MakeSimplicial(orig_mesh);
   for (int i = 0; i < orig_mesh.GetNV(); ++i)
   {
      for (int j = 0; j < 3; ++j)
      {
         REQUIRE(simplex_mesh.GetVertex(i)[j] == orig_mesh.GetVertex(i)[j]);
      }
   }
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

TEST_CASE("NURBS 1D curve in 2D from patches", "[Mesh]")
{
   // Build a 1D patch topology embedded in 2D physical space with
   // three segments of varying orders (linear, quadratic and cubic)
   constexpr int dim       = 1;
   constexpr int space_dim = 2;
   Mesh patch_topology(dim, 0, 0, 0, space_dim);

   constexpr int nv_input = 6;
   for (int i = 0; i < nv_input; i++)
   {
      patch_topology.AddVertex((real_t)i, 0.0, 0.0);
   }

   patch_topology.AddSegment(0, 1, 1);
   patch_topology.AddSegment(2, 3, 2);
   patch_topology.AddSegment(4, 5, 3);

   // Three 1D NURBS patches with variable order, each with control points
   // given in (x, y, w) format so that the physical dimension is 2 while the
   // topological dimension is 1.

   auto set_knots = [](KnotVector &kv, std::initializer_list<real_t> knots)
   {
      MFEM_VERIFY(kv.Size() == static_cast<int>(knots.size()),
                  "KnotVector and knot list must have the same size.");
      int i = 0;
      for (const auto &k : knots)
      {
         kv[i++] = k;
      }
   };

   auto set_cp = [](std::initializer_list<std::array<real_t, 3>> pts)
   {
      Array<real_t> cp(3 * static_cast<int>(pts.size()));
      int i = 0;
      for (const auto &[x, y, w] : pts)
      {
         cp[i++] = x;
         cp[i++] = y;
         cp[i++] = w;
      }
      return cp;
   };

   // Patch 0: order 1, 4 control points
   KnotVector kv1(1, 4);
   set_knots(kv1, {0.0, 0.0, 0.4, 0.6, 1.0, 1.0});
   kv1.GetElements();
   Array<real_t> cp1 = set_cp(
   {
      {0.0, 0.0, 1.0},
      {0.4, 0.6, 1.0},
      {0.6, 0.4, 1.0},
      {1.0, 1.0, 1.0}});
   Array<const KnotVector *> kvs1({&kv1});

   // Patch 1: order 2, 3 control points
   KnotVector kv2(2, 3);
   set_knots(kv2, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
   kv2.GetElements();
   Array<real_t> cp2 = set_cp(
   {
      {1.0, 0.0, 1.0},
      {1.0, 1.0, 1.2},
      {2.0, 1.0, 1.0}});
   Array<const KnotVector *> kvs2({&kv2});

   // Patch 2: order 3, 4 control points
   KnotVector kv3(3, 4);
   set_knots(kv3, {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0});
   kv3.GetElements();
   Array<real_t> cp3 = set_cp(
   {
      {2.0, 0.0, 1.0},
      {2.0, 0.9, 1.31},
      {2.1, 1.0, 1.32},
      {3.0, 1.0, 1.0}});
   Array<const KnotVector *> kvs3({&kv3});

   auto p1 = std::make_unique<NURBSPatch>(kvs1, 3, cp1.GetData());
   auto p2 = std::make_unique<NURBSPatch>(kvs2, 3, cp2.GetData());
   auto p3 = std::make_unique<NURBSPatch>(kvs3, 3, cp3.GetData());

   Array<const NURBSPatch *> patches(3);
   patches[0] = p1.get();
   patches[1] = p2.get();
   patches[2] = p3.get();

   NURBSExtension ne(&patch_topology, patches);
   Mesh mesh(ne);

   // Check that we created a 1D NURBS mesh embedded in 2D physical space and
   // that the associated finite element space uses the correct vector dimension.
   REQUIRE(mesh.Dimension() == dim);
   REQUIRE(mesh.SpaceDimension() == space_dim);
   REQUIRE(mesh.GetNE() == 5);
   REQUIRE(mesh.GetNV() == 8);

   GridFunction *nodes = mesh.GetNodes();
   REQUIRE(nodes != NULL);
   REQUIRE(nodes->FESpace() != NULL);
   REQUIRE(nodes->FESpace()->GetVDim() == space_dim);

   REQUIRE(mesh.NURBSext != NULL);
   REQUIRE(mesh.NURBSext->GetNP() == 3);

   // Additionally, exercise degree elevation and ensure basic invariants hold
   {
      const Array<int> &orders = mesh.NURBSext->GetOrders();
      const int max_order = orders.Max();
      mesh.DegreeElevate(max_order, max_order);

      REQUIRE(mesh.NURBSext != nullptr);
      REQUIRE(mesh.Dimension() == dim);
      REQUIRE(mesh.SpaceDimension() == space_dim);
      REQUIRE(mesh.NURBSext->Dimension() == dim);

      const Array<int> &new_orders = mesh.NURBSext->GetOrders();
      REQUIRE(new_orders.Size() == orders.Size());
      for (int i = 0; i < new_orders.Size(); ++i)
      {
         REQUIRE(new_orders[i] == max_order);
      }
   }
}

TEST_CASE("Mesh::Swap preserves named attribute sets", "[Mesh]")
{
   // Regression test for a bug where Mesh::Swap swapped attributes and
   // bdr_attributes but omitted the attr_sets maps, silently losing all named
   // element/boundary sets on any move or swap of an mfem::Mesh.

   Mesh a = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   Mesh b = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);

   Array<int> a_elem_attrs({1});
   Array<int> b_elem_attrs({1});
   Array<int> a_bdr_attrs({1, 2});
   Array<int> b_bdr_attrs({3, 4});

   a.attribute_sets.SetAttributeSet("elem_set_a", a_elem_attrs);
   a.bdr_attribute_sets.SetAttributeSet("bdr_set_a", a_bdr_attrs);
   b.attribute_sets.SetAttributeSet("elem_set_b", b_elem_attrs);
   b.bdr_attribute_sets.SetAttributeSet("bdr_set_b", b_bdr_attrs);

   a.Swap(b, true);

   // After swap, a should hold b's sets and b should hold a's sets.
   REQUIRE(a.attribute_sets.AttributeSetExists("elem_set_b"));
   REQUIRE(!a.attribute_sets.AttributeSetExists("elem_set_a"));
   REQUIRE(a.bdr_attribute_sets.AttributeSetExists("bdr_set_b"));
   REQUIRE(!a.bdr_attribute_sets.AttributeSetExists("bdr_set_a"));

   REQUIRE(b.attribute_sets.AttributeSetExists("elem_set_a"));
   REQUIRE(!b.attribute_sets.AttributeSetExists("elem_set_b"));
   REQUIRE(b.bdr_attribute_sets.AttributeSetExists("bdr_set_a"));
   REQUIRE(!b.bdr_attribute_sets.AttributeSetExists("bdr_set_b"));

   // Verify set contents survived the swap.
   Array<int> a_bdr_result = a.bdr_attribute_sets.GetAttributeSet("bdr_set_b");
   Array<int> b_bdr_result = b.bdr_attribute_sets.GetAttributeSet("bdr_set_a");
   REQUIRE(a_bdr_result == b_bdr_attrs);
   REQUIRE(b_bdr_result == a_bdr_attrs);
}

TEST_CASE("Tetrahedron and boundary triangle marked alike", "[Mesh]")
{
   // One tetrahedron, with a boundary triangle on each face. The edges of the
   // face 0-1-2 have squared lengths 1, 1 + 0.6e-6 and 1 + 1.2e-6: a
   // comparison with a relative tolerance of 1e-6 would take the first two
   // and the last two as equal, but not the first and the last, and its
   // result would depend on the order of the comparisons. Whatever the vertex
   // order of the tetrahedron and of the triangles, the refinement edge of
   // each boundary triangle must be the marked edge of its face in the
   // tetrahedron, so that the mesh can be refined.
   const real_t x = 0.5 - 0.3e-6;
   const real_t y = std::sqrt(1.0 + 0.6e-6 - x*x);
   const real_t p[4][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {x, y, 0.0},
      {0.5, 0.3, 0.8}
   };
   int tv[4] = {0, 1, 2, 3};
   int perms = 0, bad = 0;
   do
   {
      for (int rot = 0; rot < 3; rot++)
      {
         Mesh mesh(3, 4, 1, 4);
         for (int v = 0; v < 4; v++) { mesh.AddVertex(p[v]); }
         mesh.AddTet(tv);
         for (int f = 0; f < 4; f++)
         {
            const int *fv =
               Geometry::Constants<Geometry::TETRAHEDRON>::FaceVert[f];
            int bv[3];
            for (int k = 0; k < 3; k++) { bv[k] = tv[fv[(k + rot) % 3]]; }
            mesh.AddBdrTriangle(bv, 1 + f);
         }
         mesh.FinalizeTopology();
         mesh.Finalize(true);
         const Tetrahedron *tet =
            static_cast<const Tetrahedron *>(mesh.GetElement(0));
         Array<int> faces, ori;
         mesh.GetElementFaces(0, faces, ori);
         for (int b = 0; b < mesh.GetNBE(); b++)
         {
            const int *bv = mesh.GetBdrElement(b)->GetVertices();
            const int j = faces.Find(mesh.GetBdrElementFaceIndex(b));
            REQUIRE(j >= 0);
            int mv[3];
            tet->GetMarkedFace(j, mv);
            const bool same = (bv[0] == mv[0] && bv[1] == mv[1]) ||
                              (bv[0] == mv[1] && bv[1] == mv[0]);
            bad += !same;
         }
         // Refine (three bisections), which needs the two to agree.
         Array<int> marked(1);
         marked[0] = 0;
         mesh.GeneralRefinement(marked, 0);
         CHECK(mesh.GetNE() == 8);
         perms++;
      }
   }
   while (std::next_permutation(tv, tv + 4));
   INFO(bad << " boundary triangles marked differently, in " << perms
        << " vertex orders");
   CHECK(bad == 0);
}

TEST_CASE("Curved simplices with collinear corners marked alike", "[Mesh]")
{
   // The map z -> z^2 of the plane gives, on a straight simplex in the first
   // quadrant, a valid quadratic element whose corners can be collinear: the
   // triangle (1, 1), (2, 0.5), (0.5, 2) gets the corners (0, 2), (3.75, 2)
   // and (-3.75, 2), and its two short edges the same vector and length (also
   // in the tetrahedron with the fourth vertex (1, 1, 1)). Their keys differ
   // by their smaller vertices: in every vertex order the element is marked
   // (a debug build checks that the keys of an element differ), and the
   // marked edges of the element and of its faces are the same.
   for (int dim = 2; dim <= 3; dim++)
   {
      int order[4] = {0, 1, 2, 3};
      std::vector<std::vector<int>> marks;
      do
      {
         Mesh mesh(dim, dim + 1, 1, 0, dim);
         mesh.AddVertex(1.0, 1.0, 0.0);
         mesh.AddVertex(2.0, 0.5, 0.0);
         mesh.AddVertex(0.5, 2.0, 0.0);
         if (dim == 3)
         {
            mesh.AddVertex(1.0, 1.0, 1.0);
            mesh.AddTet(order);
         }
         else { mesh.AddTriangle(order); }
         mesh.FinalizeTopology();
         mesh.Finalize(false, true);
         mesh.SetCurvature(2);
         mesh.Transform([](const Vector &x, Vector &y)
         {
            y = x;
            y(0) = x(0)*x(0) - x(1)*x(1);
            y(1) = 2.0*x(0)*x(1);
         });
         REQUIRE(mesh.CheckElementOrientation(false) == 0);
         mesh.Finalize(true);
         // The marked edge, then those of the faces in the order of their
         // sorted vertices, as sorted pairs of vertex numbers.
         const int *v = mesh.GetElement(0)->GetVertices();
         std::vector<int> m = {std::min(v[0], v[1]), std::max(v[0], v[1])};
         if (dim == 3)
         {
            const Tetrahedron *tet =
               static_cast<const Tetrahedron *>(mesh.GetElement(0));
            std::vector<std::array<int, 5>> faces;
            for (int f = 0; f < 4; f++)
            {
               int fv[3];
               tet->GetMarkedFace(f, fv);
               std::array<int, 3> sv = {fv[0], fv[1], fv[2]};
               std::sort(sv.begin(), sv.end());
               faces.push_back({sv[0], sv[1], sv[2], std::min(fv[0], fv[1]),
                                std::max(fv[0], fv[1])
                               });
            }
            std::sort(faces.begin(), faces.end());
            for (const auto &f : faces)
            {
               m.insert(m.end(), f.begin(), f.end());
            }
         }
         marks.push_back(m);
      }
      while (std::next_permutation(order, order + dim + 1));
      INFO("dimension " << dim << ", " << marks.size() << " vertex orders");
      for (const auto &m : marks) { CHECK(m == marks[0]); }
   }
}

TEST_CASE("Uniform refinement after a local refinement of a curved mesh",
          "[Mesh]")
{
   // Uniform refinement takes the point matrices of the children from static
   // arrays, and local refinement computes its own: it must not write them
   // over the static arrays, which the next uniform refinements use for the
   // nodes of curved meshes. The point matrices of a local refinement that
   // refines few elements, here none, fit in these arrays. Straight nodes of
   // order 2 must keep the measure of the unit square or cube.
   const int dim = GENERATE(2, 3);
   Mesh mesh = (dim == 2) ?
               Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE) :
               Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
   mesh.SetCurvature(2);
   auto Measure = [&mesh]()
   {
      real_t measure = 0.0;
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         measure += mesh.GetElementVolume(e);
      }
      return measure;
   };
   for (int pass = 0; pass < 2; pass++)
   {
      mesh.UniformRefinement();
      CHECK(std::abs(Measure() - 1.0) < 1e-12);
      mesh.Finalize(true);
      Array<int> none;
      mesh.GeneralRefinement(none, 0);
      CHECK(std::abs(Measure() - 1.0) < 1e-12);
   }
}

namespace
{

/// The number of elements whose marked edge, between their local vertices 0
/// and 1, is shorter than one of their other edges by more than round-off.
/// The lengths are those of the chords between the vertices as mapped by each
/// element.
int MarkedNotLongest(const Mesh &mesh)
{
   int n = 0;
   IsoparametricTransformation T;
   DenseMatrix x;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      const Element *el = mesh.GetElement(e);
      mesh.GetElementTransformation(e, &T);
      T.Transform(*Geometries.GetVertices(el->GetGeometryType()), x);
      real_t longest = 0.0, marked = 0.0;
      for (int k = 0; k < el->GetNEdges(); k++)
      {
         const int *ev = el->GetEdgeVertices(k);
         real_t l = 0.0;
         for (int d = 0; d < x.Height(); d++)
         {
            l += (x(d, ev[0]) - x(d, ev[1]))*(x(d, ev[0]) - x(d, ev[1]));
         }
         longest = std::max(longest, l);
         if (ev[0] + ev[1] == 1) { marked = l; }
      }
      n += (marked < longest*(1.0 - 1e-12));
   }
   return n;
}

} // namespace

TEST_CASE("Triangle with discontinuous nodes marked by its longest edge",
          "[Mesh]")
{
   // Discontinuous (L2) nodes have no vertex DOFs: the lengths of the edges
   // come from the elements. A triangle with sides of lengths 100, 1 and
   // about 100.005 is marked by its longest side, also when Finalize() first
   // fixes its orientation, which its nodes must then follow.
   const bool inverted = GENERATE(false, true);
   INFO((inverted ? "inverted" : "positive") << " triangle");
   Mesh mesh(2, 3, 1);
   mesh.AddVertex(0.0, 0.0);
   mesh.AddVertex(100.0, 0.0);
   mesh.AddVertex(inverted ? 100.0 : 0.0, inverted ? -1.0 : 1.0);
   mesh.AddTriangle(0, 1, 2);
   mesh.FinalizeTopology();
   mesh.SetCurvature(2, true);
   mesh.Finalize(true, true);
   CHECK(MarkedNotLongest(mesh) == 0);
   CHECK(std::abs(mesh.GetElementVolume(0) - 50.0) < 1e-10);
}

TEST_CASE("Mesh with discontinuous nodes marked as without nodes", "[Mesh]")
{
   // Straight discontinuous nodes give the vertices their positions in every
   // element: marked again, the mesh is marked as the same mesh without
   // nodes. Its vertices are moved so that no two edges have the same length.
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(1, 2);
   INFO(dim << "D, nodes of order " << order);
   auto Make = [dim]()
   {
      Mesh mesh = (dim == 2) ?
                  Mesh::MakeCartesian2D(4, 4, Element::TRIANGLE) :
                  Mesh::MakeCartesian3D(3, 3, 3, Element::TETRAHEDRON);
      for (int v = 0; v < mesh.GetNV(); v++)
      {
         real_t *x = mesh.GetVertex(v);
         for (int d = 0; d < dim; d++)
         {
            x[d] += 0.02*std::sin(12.9898*v + 78.233*d);
         }
      }
      return mesh;
   };
   Mesh plain = Make(), nodal = Make();
   nodal.SetCurvature(order, true);
   plain.Finalize(true);
   nodal.Finalize(true);
   REQUIRE(plain.GetNE() == nodal.GetNE());
   REQUIRE(plain.GetNBE() == nodal.GetNBE());
   auto Same = [](const Element *a, const Element *b)
   {
      const int n = a->GetNVertices();
      return std::equal(a->GetVertices(), a->GetVertices() + n,
                        b->GetVertices());
   };
   int differ = 0;
   for (int e = 0; e < plain.GetNE(); e++)
   {
      differ += !Same(plain.GetElement(e), nodal.GetElement(e));
   }
   for (int b = 0; b < plain.GetNBE(); b++)
   {
      differ += !Same(plain.GetBdrElement(b), nodal.GetBdrElement(b));
   }
   CHECK(differ == 0);
   CHECK(MarkedNotLongest(nodal) == 0);
}

TEST_CASE("Periodic meshes marked by the longest edge of each element",
          "[Mesh]")
{
   // The nodes of a periodic mesh are discontinuous: a vertex on a periodic
   // boundary has a position in each of its elements. The cells are long in
   // one direction, so that the longest edge of each element is longer than
   // the others by much more than round-off. Marked again, each element is
   // marked by its longest edge.
   const int dim = GENERATE(2, 3);
   INFO(dim << "D");
   Mesh base = (dim == 2) ?
               Mesh::MakeCartesian2D(4, 3, Element::TRIANGLE, false, 1.0, 9.0) :
               Mesh::MakeCartesian3D(3, 3, 3, Element::TETRAHEDRON, 1.0, 4.0,
                                     13.0);
   std::vector<Vector> translations;
   for (int d = 0; d < dim; d++)
   {
      Vector t(dim);
      t = 0.0;
      t(d) = (d == 0) ? 1.0 : (d == 1) ? ((dim == 2) ? 9.0 : 4.0) : 13.0;
      translations.push_back(t);
   }
   Mesh mesh = Mesh::MakePeriodic(
                  base, base.CreatePeriodicVertexMapping(translations));
   mesh.Finalize(true);
   CHECK(MarkedNotLongest(mesh) == 0);
}

TEST_CASE("Inverted tetrahedra with discontinuous nodes marked by their "
          "lengths", "[Mesh]")
{
   // Finalize() fixes the orientation of inverted tetrahedra before it marks
   // them, and the marking reads discontinuous nodes through the elements: the
   // nodes must follow the fixed vertex orders. Half of the tetrahedra of a
   // mesh with moved vertices are inverted (two of their vertices swapped).
   // With discontinuous nodes of orders 1 and 2, the mesh is fixed and marked
   // as the same mesh without nodes, every tetrahedron is marked by its
   // longest edge, none is inverted any more, and the nodes map the vertices
   // of each tetrahedron to their coordinates.
   const int order = GENERATE(1, 2);
   INFO("nodes of order " << order);
   auto Make = []()
   {
      Mesh mesh = Mesh::MakeCartesian3D(3, 3, 3, Element::TETRAHEDRON);
      for (int v = 0; v < mesh.GetNV(); v++)
      {
         real_t *x = mesh.GetVertex(v);
         for (int d = 0; d < 3; d++)
         {
            x[d] += 0.02*std::sin(12.9898*v + 78.233*d);
         }
      }
      for (int e = 0; e < mesh.GetNE(); e += 2)
      {
         int *v = mesh.GetElement(e)->GetVertices();
         std::swap(v[2], v[3]);
      }
      return mesh;
   };
   Mesh plain = Make(), nodal = Make();
   nodal.SetCurvature(order, true);
   REQUIRE(nodal.CheckElementOrientation(false) == nodal.GetNE()/2);
   plain.Finalize(true, true);
   nodal.Finalize(true, true);
   REQUIRE(plain.GetNE() == nodal.GetNE());
   REQUIRE(plain.GetNBE() == nodal.GetNBE());
   auto Same = [](const Element *a, const Element *b)
   {
      const int n = a->GetNVertices();
      return std::equal(a->GetVertices(), a->GetVertices() + n,
                        b->GetVertices());
   };
   int differ = 0;
   for (int e = 0; e < plain.GetNE(); e++)
   {
      differ += !Same(plain.GetElement(e), nodal.GetElement(e));
   }
   for (int b = 0; b < plain.GetNBE(); b++)
   {
      differ += !Same(plain.GetBdrElement(b), nodal.GetBdrElement(b));
   }
   CHECK(differ == 0);
   CHECK(MarkedNotLongest(nodal) == 0);
   CHECK(nodal.CheckElementOrientation(false) == 0);
   real_t gap = 0.0;
   IsoparametricTransformation T;
   DenseMatrix x;
   for (int e = 0; e < nodal.GetNE(); e++)
   {
      nodal.GetElementTransformation(e, &T);
      T.Transform(*Geometries.GetVertices(Geometry::TETRAHEDRON), x);
      const int *v = nodal.GetElement(e)->GetVertices();
      for (int k = 0; k < 4; k++)
      {
         for (int d = 0; d < 3; d++)
         {
            gap = std::max(gap, std::abs(x(d, k) - plain.GetVertex(v[k])[d]));
         }
      }
   }
   CHECK(gap < 1e-12);
}

TEST_CASE("Uniform refinement of tetrahedra independent of round-off",
          "[Mesh]")
{
   // Prisms over equilateral triangles split into tetrahedra: the diagonals of
   // many octahedra of their uniform refinement tie. Moving every vertex by 1
   // to 6 ULP after the marking must not change the diagonals chosen.
   auto Refined = [](int seed)
   {
      Mesh wedges = Mesh::MakeCartesian3D(4, 4, 4, Element::WEDGE);
      for (int v = 0; v < wedges.GetNV(); v++)
      {
         real_t *x = wedges.GetVertex(v);
         x[0] -= 0.5*x[1];
         x[1] *= std::sqrt(real_t(0.75));
      }
      Mesh mesh = Mesh::MakeSimplicial(wedges);
      mesh.Finalize(true);
      std::mt19937 gen(seed);
      std::uniform_int_distribution<int> ulps(1, 6), up(0, 1);
      for (int v = 0; seed > 0 && v < mesh.GetNV(); v++)
      {
         real_t *x = mesh.GetVertex(v);
         for (int d = 0; d < 3; d++)
         {
            const real_t to = up(gen) ? 2.0 : -2.0;
            for (int k = ulps(gen); k > 0; k--)
            {
               x[d] = std::nextafter(x[d], to);
            }
         }
      }
      mesh.UniformRefinement();
      std::vector<std::array<int, 4>> children(mesh.GetNE());
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         const int *v = mesh.GetElement(e)->GetVertices();
         std::copy(v, v + 4, children[e].begin());
         std::sort(children[e].begin(), children[e].end());
      }
      return children;
   };
   const auto reference = Refined(0);
   for (int seed = 1; seed <= 4; seed++)
   {
      CAPTURE(seed);
      CHECK(Refined(seed) == reference);
   }
}
