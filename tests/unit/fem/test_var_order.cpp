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
#ifdef MFEM_USE_MPI
#include <array>
#include <cmath>
#include <map>
#endif

#include <array>
#include <map>
#include <set>
#include <vector>

namespace mfem
{

static real_t exact_sln(const Vector &p);
static void TestSolve(FiniteElementSpace &fespace);
static void TestSolveVec(FiniteElementSpace &fespace);
#ifdef MFEM_USE_MPI
static void TestSolvePar(ParFiniteElementSpace &fespace);
static void TestSolveParVec(ParFiniteElementSpace &fespace);
static void TestRandomPRefinement(Mesh & mesh);
#endif

namespace var_order_test { enum class SpaceType {RT, ND}; }

Mesh MakeCartesianMesh(int nx, int dim)
{
   if (dim == 2)
   {
      return Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL, true);
   }
   else
   {
      return Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON);
   }
}

namespace var_order_test
{

// The requested order is the collection order; RT constructors take one less.
static std::unique_ptr<RT_FECollection> MakeRTCollection(
   int kind, int order, int dim, int cb_type, int ob_type, int map_type)
{
   if (kind == 0)
   {
      return std::unique_ptr<RT_FECollection>(
                new RT_FECollection(order - 1, dim, cb_type, ob_type));
   }
   if (kind == 1)
   {
      return std::unique_ptr<RT_FECollection>(
                new RT_Trace_FECollection(order - 1, dim, map_type, ob_type));
   }
   return std::unique_ptr<RT_FECollection>(
             new DG_Interface_FECollection(order - 1, dim, map_type, ob_type));
}

} // namespace var_order_test

TEST_CASE("RT collection cloning preserves the requested collection order",
          "[FiniteElementCollection][RT][RTCloneOrder]")
{
   using namespace var_order_test;
   const int kind = GENERATE(0, 1, 2);
   const int dim = GENERATE(2, 3);
   const int base_order = GENERATE(1, 3);
   const int order = GENERATE(1, 2, 3);
   const int ob_type = GENERATE(BasisType::GaussLegendre,
                                BasisType::IntegratedGLL);
   const int map_type = GENERATE(FiniteElement::VALUE, FiniteElement::INTEGRAL);
   const int cb_type = BasisType::GaussLobatto;
   CAPTURE(kind, dim, base_order, order, ob_type, map_type);
   auto source = MakeRTCollection(kind, base_order, dim, cb_type, ob_type,
                                  map_type);
   auto expected = MakeRTCollection(kind, order, dim, cb_type, ob_type,
                                    map_type);
   std::unique_ptr<FiniteElementCollection> clone(source->Clone(order));
   REQUIRE(clone->GetOrder() == order);
   REQUIRE(clone->GetConstructorOrder() == order - 1);
   REQUIRE(std::string(clone->Name()) == expected->Name());
   const auto trace_geom = dim == 2 ? Geometry::SEGMENT : Geometry::SQUARE;
   const auto cell_geom = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   for (auto geom : {trace_geom, kind == 0 ? cell_geom : trace_geom})
   {
      const FiniteElement &fe = *source->GetFE(geom, order);
      const FiniteElement &ref = *expected->FiniteElementForGeometry(geom);
      REQUIRE(fe.GetOrder() == ref.GetOrder());
      REQUIRE(fe.GetDof() == ref.GetDof());
      REQUIRE(fe.GetMapType() == ref.GetMapType());
      REQUIRE(source->GetNumDof(geom, order) == expected->DofForGeometry(geom));
   }
   const int *actual = source->GetDofOrdering(trace_geom, order, 1);
   const int *reference = expected->DofOrderForOrientation(trace_geom, 1);
   for (int i = 0; i < expected->DofForGeometry(trace_geom); i++)
   {
      REQUIRE(actual[i] == reference[i]);
   }
}

TEST_CASE("RT LOR retains the lowest-order space after collection cloning",
          "[FiniteElementCollection][RT][LOR][RTLOROrder]")
{
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(1, 2, 3);
   CAPTURE(dim, order);
   Mesh mesh = MakeCartesianMesh(1, dim);
   RT_FECollection high(order - 1, dim, BasisType::GaussLobatto,
                        BasisType::IntegratedGLL);
   FiniteElementSpace high_space(&mesh, &high);
   LORDiscretization lor(high_space);
   FiniteElementSpace &low_space = lor.GetFESpace();
   REQUIRE(low_space.FEColl()->GetOrder() == 1);
   REQUIRE(low_space.FEColl()->GetConstructorOrder() == 0);
   RT_FECollection reference(0, dim, BasisType::GaussLobatto,
                             BasisType::IntegratedGLL);
   FiniteElementSpace reference_space(low_space.GetMesh(), &reference);
   REQUIRE(low_space.GetTrueVSize() == reference_space.GetTrueVSize());
   for (int e = 0; e < low_space.GetNE(); e++)
   {
      REQUIRE(low_space.GetFE(e)->GetOrder() == 1);
   }
}

// Check basic functioning of variable order spaces, hp interpolation and
// some corner cases.
TEST_CASE("Variable Order FiniteElementSpace",
          "[FiniteElementCollection]"
          "[FiniteElementSpace]"
          "[NCMesh]")
{
   SECTION("Quad mesh")
   {
      // 2-element quad mesh
      Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
      mesh.EnsureNCMesh();

      // standard H1 space with order 1 elements
      H1_FECollection fec(1, mesh.Dimension());
      FiniteElementSpace fespace(&mesh, &fec);

      REQUIRE(fespace.GetNDofs() == 6);
      REQUIRE(fespace.GetNConformingDofs() == 6);

      // convert to variable order space: p-refine second element
      fespace.SetElementOrder(1, 2);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 11);
      REQUIRE(fespace.GetNConformingDofs() == 10);

      // h-refine first element in the y axis
      Array<Refinement> refs;
      refs.Append(Refinement(0, 2));
      mesh.GeneralRefinement(refs);
      fespace.Update();

      REQUIRE(fespace.GetNDofs() == 13);
      REQUIRE(fespace.GetNConformingDofs() == 11);

      // relax the master edge to be quadratic
      fespace.SetRelaxedHpConformity(true);

      REQUIRE(fespace.GetNDofs() == 13);
      REQUIRE(fespace.GetNConformingDofs() == 12);

      // increase order
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         fespace.SetElementOrder(i, fespace.GetElementOrder(i) + 1);
      }
      fespace.Update(false);

      // 15 quadratic + 16 cubic DOFs - 2 shared vertices:
      REQUIRE(fespace.GetNDofs() == 29);
      // 3 constrained DOFs on slave side, inexact interpolation
      REQUIRE(fespace.GetNConformingDofs() == 26);

      // relaxed off
      fespace.SetRelaxedHpConformity(false);

      // new quadratic DOF on master edge:
      REQUIRE(fespace.GetNDofs() == 30);
      // 3 constrained DOFs on slave side, 2 on master side:
      REQUIRE(fespace.GetNConformingDofs() == 25);

      TestSolve(fespace);

      // refine
      mesh.UniformRefinement();
      fespace.Update();

      REQUIRE(fespace.GetNDofs() == 93);
      REQUIRE(fespace.GetNConformingDofs() == 83);

      TestSolve(fespace);
   }

   SECTION("Quad/hex mesh projection")
   {
      for (int dim=2; dim<=3; ++dim)
      {
         // 2-element mesh
         Mesh mesh = dim == 2 ? Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL) :
                     Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
         mesh.EnsureNCMesh();

         // h-refine element 1
         Array<Refinement> refinements;
         refinements.Append(Refinement(1));

         int nonconformity_limit = 0; // 0 meaning allow unlimited ratio
         mesh.GeneralRefinement(refinements, 1, nonconformity_limit);  // h-refinement

         // standard H1 space with order 2 elements
         H1_FECollection fec(2, mesh.Dimension());
         FiniteElementSpace fespace(&mesh, &fec);

         GridFunction x(&fespace);

         // p-refine element 0
         fespace.SetElementOrder(0, 3);

         fespace.Update(false);
         x.SetSpace(&fespace);

         // Test projection of the coefficient
         FunctionCoefficient exsol(exact_sln);
         x.ProjectCoefficient(exsol);

         // Enforce space constraints on locally interpolated GridFunction x
         const SparseMatrix *R = fespace.GetHpRestrictionMatrix();
         const SparseMatrix *P = fespace.GetConformingProlongation();
         Vector y(fespace.GetTrueVSize());
         R->Mult(x, y);
         P->Mult(y, x);

         const real_t error = x.ComputeL2Error(exsol);
         REQUIRE(error == MFEM_Approx(0.0));
      }
   }

   SECTION("Hex mesh")
   {
      // 2-element hex mesh
      Mesh mesh = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
      mesh.EnsureNCMesh();

      // standard H1 space with order 1 elements
      H1_FECollection fec(1, mesh.Dimension());
      FiniteElementSpace fespace(&mesh, &fec);

      REQUIRE(fespace.GetNDofs() == 12);
      REQUIRE(fespace.GetNConformingDofs() == 12);

      // convert to variable order space: p-refine second element
      fespace.SetElementOrder(1, 2);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 31);
      REQUIRE(fespace.GetNConformingDofs() == 26);

      // h-refine first element in the z axis
      Array<Refinement> refs;
      refs.Append(Refinement(0, 4));
      mesh.GeneralRefinement(refs);
      fespace.Update();

      REQUIRE(fespace.GetNDofs() == 35);
      REQUIRE(fespace.GetNConformingDofs() == 28);

      // relax the master face to be quadratic
      fespace.SetRelaxedHpConformity(true);

      REQUIRE(fespace.GetNDofs() == 35);
      REQUIRE(fespace.GetNConformingDofs() == 31);

      // increase order
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         fespace.SetElementOrder(i, fespace.GetElementOrder(i) + 1);
      }
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 105);
      REQUIRE(fespace.GetNConformingDofs() == 92);

      // relaxed off
      fespace.SetRelaxedHpConformity(false);

      REQUIRE(fespace.GetNDofs() == 108);
      REQUIRE(fespace.GetNConformingDofs() == 87);

      // refine one of the small elements into four
      refs[0].SetType(3);
      mesh.GeneralRefinement(refs);
      fespace.Update();

      REQUIRE(fespace.GetNDofs() == 162);
      REQUIRE(fespace.GetNConformingDofs() == 115);

      TestSolve(fespace);

      // The low-order child restricts a line in the second master face, not
      // its whole polynomial trace. Independent broken trace equations give
      // dimension 180 - rank(J) = 180 - 84 = 96 on this six-element mesh.
      fespace.SetElementOrder(0, 1);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 152);
      REQUIRE(fespace.GetNConformingDofs() == 96);
      FunctionCoefficient compatible([](const Vector &x)
      { return 1+x[0]+x[1]+x[2]+x[0]*x[1]*x[2]; });
      GridFunction field(&fespace);
      field = 0.0;
      field.ProjectCoefficient(compatible);
      Vector truth(fespace.GetTrueVSize());
      fespace.GetHpConformingRestriction()->Mult(field, truth);
      field.SetFromTrueDofs(truth);
      REQUIRE(field.ComputeL2Error(compatible) < 1e-12);
   }

   SECTION("Prism mesh")
   {
      // 2-element prism mesh
      Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::WEDGE);
      mesh.EnsureNCMesh();

      // standard H1 space with order 2 elements
      H1_FECollection fec(2, mesh.Dimension());
      FiniteElementSpace fespace(&mesh, &fec);

      REQUIRE(fespace.GetNDofs() == 27);
      REQUIRE(fespace.GetNConformingDofs() == 27);

      // convert to variable order space: p-refine first element
      fespace.SetElementOrder(0, 3);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 54);
      REQUIRE(fespace.GetNConformingDofs() == 42);

      // refine to form an edge-face constraint similar to
      // https://github.com/mfem/mfem/pull/713#issuecomment-495786362
      Array<Refinement> refs;
      refs.Append(Refinement(1, 3));
      mesh.GeneralRefinement(refs);
      fespace.Update(false);

      refs[0].SetType(4);
      refs.Append(Refinement(2, 4));
      mesh.GeneralRefinement(refs);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 113);
      REQUIRE(fespace.GetNConformingDofs() == 67);

      TestSolve(fespace);
   }

   SECTION("Quad/hex mesh ND/RT")
   {
      using namespace var_order_test;

      const auto space_type = GENERATE(SpaceType::RT, SpaceType::ND);
      const int dim = GENERATE(2, 3);
      CAPTURE(space_type);
      CAPTURE(dim);

      Mesh mesh = MakeCartesianMesh(dim == 2 ? 4 : 2, dim);
      mesh.EnsureNCMesh();

      int ndof0, ncdof1, ncdof2, ndof1;
      std::unique_ptr<FiniteElementCollection> fec;
      if (space_type == SpaceType::RT)
      {
         // Standard RT space with order 0 elements
         fec.reset(new RT_FECollection(0, dim));
         ndof0 = dim == 2 ? 40 : 36;
         // A direct RT_1 construction is independent of the variable-order
         // Clone path. The first Cartesian cell has dim boundary faces and
         // dim shared faces. Boundary traces gain (nf-1) modes; shared faces
         // keep their RT_0 trace and add an nf-DOF local variant. Cell-interior
         // modes are all free. This gives 50/46 DOFs in 2D and 69/57 in 3D.
         RT_FECollection reference(1, dim);
         FiniteElementSpace reference_space(&mesh, &reference);
         const auto face_geom = dim == 2 ? Geometry::SEGMENT : Geometry::SQUARE;
         const auto cell_geom = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
         const int nf = reference.DofForGeometry(face_geom);
         const int ni = reference.DofForGeometry(cell_geom);
         ncdof1 = ndof0 + dim*(nf - 1) + ni;
         ndof1 = ncdof1 + dim*nf;
         ncdof2 = reference_space.GetTrueVSize();
      }
      else
      {
         // Standard ND space with order 1 elements
         fec.reset(new ND_FECollection(1, dim));

         if (dim == 2)
         {
            ndof0 = 40;
            ndof1 = 50;
            ncdof1 = 46;
            ncdof2 = 144;
         }
         else
         {
            ndof0 = 54;
            ndof1 = 105;
            ncdof1 = 75;
            ncdof2 = 300;
         }
      }

      FiniteElementSpace fespace(&mesh, fec.get());

      REQUIRE(fespace.GetNDofs() == ndof0);
      REQUIRE(fespace.GetNConformingDofs() == ndof0);

      // Convert to variable order space: p-refine first element
      fespace.SetElementOrder(0, 2);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == ndof1);
      REQUIRE(fespace.GetNConformingDofs() == ncdof1);

      // p-refine all elements
      for (int i = 1; i < mesh.GetNE(); i++)
      {
         fespace.SetElementOrder(i, fespace.GetElementOrder(i) + 1);
      }
      fespace.Update(false);

      REQUIRE(fespace.GetNConformingDofs() == ncdof2);
      for (int e = 0; e < fespace.GetNE(); e++)
      {
         REQUIRE(fespace.GetFE(e)->GetOrder() == fespace.GetElementOrder(e));
      }

      TestSolveVec(fespace);
   }
}

namespace var_order_test
{

// Hex mesh 2x2x1 with the element at x > 1/2, y > 1/2 refined once: the coarse
// neighbors see master faces normal to x and to y, whose slave faces have
// interior edges in both directions of the master face.
static Mesh MakeLocallyRefinedHexMesh()
{
   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 1, Element::HEXAHEDRON);
   mesh.EnsureNCMesh();
   Array<int> refine;
   Vector center(3);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementCenter(e, center);
      if (center(0) > 0.5 && center(1) > 0.5) { refine.Append(e); }
   }
   mesh.GeneralRefinement(refine);
   return mesh;
}

// Three quadrants around a common vertical edge. The positive quadrant is
// refined, leaving two coarse neighbors which meet only along that edge.
static Mesh MakeThreeQuadrantHexMesh(bool rotated)
{
   Mesh mesh(3, 18, 3, 0, 3);
   for (int z = 0; z < 2; z++)
   {
      for (int y = 0; y < 3; y++)
      {
         for (int x = 0; x < 3; x++)
         {
            mesh.AddVertex(real_t(x - 1), real_t(y - 1), real_t(z));
         }
      }
   }
   for (int y = 0; y < 2; y++)
   {
      for (int x = 0; x < 2; x++)
      {
         if (x == 0 && y == 0) { continue; }
         const int v = x + 3*y;
         int hex[8] = {v, v+1, v+4, v+3, v+9, v+10, v+13, v+12};
         if (rotated)
         {
            const int perm[8] = {0, 4, 5, 1, 3, 7, 6, 2};
            for (int r = 0; r < (x + 2*y)%3; r++)
            {
               int copy[8];
               std::copy(hex, hex + 8, copy);
               for (int i = 0; i < 8; i++) { hex[i] = copy[perm[i]]; }
            }
         }
         mesh.AddHex(hex);
      }
   }
   mesh.FinalizeHexMesh(1, 1, true);
   mesh.RemoveUnusedVertices();
   mesh.EnsureNCMesh();
   Array<int> refine;
   Vector center(3);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementCenter(e, center);
      if (center(0) > 0 && center(1) > 0) { refine.Append(e); }
   }
   mesh.GeneralRefinement(refine);
   return mesh;
}

// A low-order grandchild can restrict an edge inside another master face.
static Mesh MakeInternalEdgeMesh(bool rotated)
{
   Mesh mesh(3, 12, 2, 0, 3);
   for (int z = 0; z < 2; z++)
   {
      for (int y = 0; y < 2; y++)
      {
         for (int x = 0; x < 3; x++)
         {
            mesh.AddVertex(real_t(x), real_t(y), real_t(z));
         }
      }
   }
   for (int x = 0; x < 2; x++)
   {
      int hex[8] = {x, x+1, x+4, x+3, x+6, x+7, x+10, x+9};
      if (rotated && x == 1)
      {
         const int perm[8] = {0, 4, 5, 1, 3, 7, 6, 2};
         int copy[8];
         std::copy(hex, hex+8, copy);
         for (int i = 0; i < 8; i++) { hex[i] = copy[perm[i]]; }
      }
      mesh.AddHex(hex);
   }
   mesh.FinalizeHexMesh(1, 1, true);
   mesh.EnsureNCMesh();
   Vector center(3);
   for (int level = 0; level < 2; level++)
   {
      Array<int> marked;
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         mesh.GetElementCenter(e, center);
         if (level == 0 ? center(0) > 1 :
             (center(0) > 1 && center(0) < 1.5 &&
              center(1) > 0.5 && center(2) < 0.5))
         {
            marked.Append(e);
         }
      }
      REQUIRE(marked.Size() == 1);
      mesh.GeneralRefinement(marked, 1, 0);
   }
   REQUIRE(mesh.GetNE() == 16);
   return mesh;
}

static void SetInternalEdgeOrders(FiniteElementSpace &fes, int high)
{
   Vector center(3);
   for (int e = 0; e < fes.GetNE(); e++)
   {
      fes.GetMesh()->GetElementCenter(e, center);
      const bool low = center(0) == 1.375 && center(1) == 0.625 &&
                       center(2) == 0.125;
      fes.SetElementOrder(e, low ? high - 1 : high);
   }
   fes.Update(false);
}

// Maximum jump of H1 values, ND tangential or RT normal components across faces
// shared by two elements of an axis-aligned hex mesh (conforming or
// nonconforming), sampled inside the overlap of the two faces.
static real_t MaxTraceJump(GridFunction &x, int &npoints)
{
   Mesh &mesh = *x.FESpace()->GetMesh();
   const bool scalar = x.FESpace()->GetFE(0)->GetRangeType() ==
                       FiniteElement::SCALAR;
   const bool normal = x.FESpace()->FEColl()->GetContType() ==
                       FiniteElementCollection::NORMAL;
   const int ne = mesh.GetNE();
   const real_t eps = 1e-12;
   std::vector<Vector> lo(ne), hi(ne);
   for (int e = 0; e < ne; e++)
   {
      Array<int> v;
      mesh.GetElementVertices(e, v);
      lo[e].SetSize(3); lo[e] = infinity();
      hi[e].SetSize(3); hi[e] = -infinity();
      for (int i : v)
      {
         const real_t *c = mesh.GetVertex(i);
         for (int d = 0; d < 3; d++)
         {
            lo[e](d) = std::min(lo[e](d), c[d]);
            hi[e](d) = std::max(hi[e](d), c[d]);
         }
      }
   }
   real_t jump = 0.0;
   npoints = 0;
   // enough points per direction to detect any nonzero polynomial jump
   const int n = x.FESpace()->GetMaxElementOrder() + 2;
   Vector pt(3), va(3), vb(3);
   IntegrationPoint ipa, ipb;
   for (int a = 0; a < ne; a++)
   {
      for (int b = a + 1; b < ne; b++)
      {
         for (int d = 0; d < 3; d++)
         {
            real_t plane;
            if (std::abs(hi[a](d) - lo[b](d)) < eps) { plane = hi[a](d); }
            else if (std::abs(hi[b](d) - lo[a](d)) < eps) { plane = lo[a](d); }
            else { continue; }
            const int t1 = (d + 1) % 3, t2 = (d + 2) % 3;
            const real_t l1 = std::max(lo[a](t1), lo[b](t1));
            const real_t h1 = std::min(hi[a](t1), hi[b](t1));
            const real_t l2 = std::max(lo[a](t2), lo[b](t2));
            const real_t h2 = std::min(hi[a](t2), hi[b](t2));
            if (h1 - l1 < eps || h2 - l2 < eps) { continue; }
            for (int i = 1; i < n; i++)
            {
               for (int j = 1; j < n; j++)
               {
                  pt(d) = plane;
                  pt(t1) = l1 + (h1 - l1) * i / n;
                  pt(t2) = l2 + (h2 - l2) * j / n;
                  InverseElementTransformation inv_a(
                     mesh.GetElementTransformation(a));
                  REQUIRE(inv_a.Transform(pt, ipa) ==
                          InverseElementTransformation::Inside);
                  x.GetVectorValue(a, ipa, va);
                  InverseElementTransformation inv_b(
                     mesh.GetElementTransformation(b));
                  REQUIRE(inv_b.Transform(pt, ipb) ==
                          InverseElementTransformation::Inside);
                  x.GetVectorValue(b, ipb, vb);
                  REQUIRE(va.CheckFinite() == 0);
                  REQUIRE(vb.CheckFinite() == 0);
                  if (scalar) { jump = std::max(jump, std::abs(va(0) - vb(0))); }
                  else if (normal) { jump = std::max(jump, std::abs(va(d) - vb(d))); }
                  else
                  {
                     jump = std::max({jump, std::abs(va(t1) - vb(t1)),
                                      std::abs(va(t2) - vb(t2))});
                  }
                  npoints++;
               }
            }
         }
      }
   }
   return jump;
}

} // namespace var_order_test

TEST_CASE("Low boundary-edge order preserves higher-order face modes",
          "[FiniteElementSpace][NCMesh][HpBoundaryEdge]")
{
   using namespace var_order_test;
   const int variant = GENERATE(0, 1, 2);
   const bool rotated = GENERATE(false, true);
   CAPTURE(variant, rotated);
   Mesh mesh = MakeThreeQuadrantHexMesh(rotated);
   REQUIRE(mesh.GetNE() == 10);
   std::unique_ptr<FiniteElementCollection> fec;
   if (variant == 0) { fec.reset(new H1_FECollection(1, 3)); }
   else
   {
      fec.reset(new ND_FECollection(1, 3, BasisType::GaussLobatto,
                                    variant == 1 ? BasisType::GaussLegendre :
                                    BasisType::IntegratedGLL));
   }
   FiniteElementSpace fespace(&mesh, fec.get());
   Vector center(3);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementCenter(e, center);
      fespace.SetElementOrder(e, center(0) < 0 && center(1) > 0 ? 1 : 2);
   }
   fespace.Update(false);
   REQUIRE(fespace.GetTrueVSize() == (variant == 0 ? 112 : 288));
   Vector tdofs(fespace.GetTrueVSize());
   tdofs.Randomize(1);
   GridFunction field(&fespace);
   field.SetFromTrueDofs(tdofs);
   int npoints;
   REQUIRE(MaxTraceJump(field, npoints) < 1e-12);
   REQUIRE(npoints > 0);
}

TEST_CASE("Internal edge constraints retain compatible high-order face modes",
          "[FiniteElementSpace][NCMesh][HpInternalEdge]")
{
   using namespace var_order_test;
   const int kind = GENERATE(0, 1, 2); // H1, nodal ND, integrated ND
   const bool rotated = GENERATE(false, true);
   CAPTURE(kind, rotated);
   Mesh mesh = MakeInternalEdgeMesh(rotated);
   std::unique_ptr<FiniteElementCollection> fec;
   if (kind == 0) { fec.reset(new H1_FECollection(2, 3)); }
   else
   {
      fec.reset(new ND_FECollection(2, 3, BasisType::GaussLobatto,
                                    kind == 1 ? BasisType::GaussLegendre :
                                    BasisType::IntegratedGLL));
   }
   FiniteElementSpace fes(&mesh, fec.get());
   SetInternalEdgeOrders(fes, 2);
   REQUIRE(fes.GetTrueVSize() == (kind == 0 ? 152 : 416));
   const SparseMatrix *P = fes.GetConformingProlongation();
   const SparseMatrix *R = fes.GetConformingRestriction();
   const SparseMatrix *Rhp = fes.GetHpConformingRestriction();
   REQUIRE(P != nullptr);
   REQUIRE(R != nullptr);
   REQUIRE(Rhp != nullptr);
   for (const auto *restriction : {R, Rhp})
   {
      std::unique_ptr<SparseMatrix> identity(Mult(*restriction, *P));
      for (int i = 0; i < identity->Height(); i++)
      {
         REQUIRE((*identity)(i, i) == MFEM_Approx(1.0));
         const int *cols = identity->GetRowColumns(i);
         const real_t *vals = identity->GetRowEntries(i);
         for (int j = 0; j < identity->RowSize(i); j++)
         {
            if (cols[j] != i) { REQUIRE(std::abs(vals[j]) < 1e-12); }
         }
      }
   }

   Vector truth(fes.GetTrueVSize());
   truth.Randomize(1);
   GridFunction reconstructed(&fes);
   reconstructed.SetFromTrueDofs(truth);
   int npoints;
   REQUIRE(MaxTraceJump(reconstructed, npoints) < 1e-11);
   REQUIRE(npoints > 0);

   auto factor = [](const Vector &x)
   {
      if (x[0] <= 1) { return real_t(1); }
      if (x[0] >= 1.5) { return real_t(0); }
      if (x[1] > 0.5 && x[2] < 0.5)
      {
         return std::max(real_t(0), 1-4*(x[0]-1));
      }
      return 1-2*(x[0]-1);
   };
   FunctionCoefficient h1([&](const Vector &x)
   { return (x[1]-0.5)*x[2]*(x[2]-0.5)*factor(x); });
   VectorFunctionCoefficient nd(3, [&](const Vector &x, Vector &v)
   {
      v = 0.0;
      v[2] = (x[1]-0.5)*x[2]*factor(x);
   });
   GridFunction projected(&fes);
   projected = 0.0;
   if (kind == 0) { projected.ProjectCoefficient(h1); }
   else { projected.ProjectCoefficient(nd); }
   REQUIRE(MaxTraceJump(projected, npoints) < 1e-11);
   Rhp->Mult(projected, truth);
   reconstructed.SetFromTrueDofs(truth);
   Array<int> dofs;
   Vector actual, expected;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      fes.GetElementVDofs(e, dofs);
      projected.GetSubVector(dofs, expected);
      reconstructed.GetSubVector(dofs, actual);
      actual -= expected;
      REQUIRE(actual.Normlinf() < 1e-11);
   }
}

TEST_CASE("RT face orders are independent of interior edge orders",
          "[FiniteElementSpace][NCMesh][RTFaceOrder]")
{
   using namespace var_order_test;
   const int high = GENERATE(2, 3);
   const int ob = GENERATE(BasisType::GaussLegendre, BasisType::IntegratedGLL);
   const bool rotated = GENERATE(false, true);
   CAPTURE(high, ob, rotated);
   Mesh mesh = MakeInternalEdgeMesh(rotated);
   RT_FECollection fec(high-1, 3, BasisType::GaussLobatto, ob);
   RT_FECollection lower(high-2, 3, BasisType::GaussLobatto, ob);
   FiniteElementSpace fixed(&mesh, &fec), variable(&mesh, &fec);
   SetInternalEdgeOrders(variable, high);
   // Normal traces on distinct faces are independent. Lowering one element
   // removes its interior modes and the degree difference on its six faces;
   // it does not constrain a second face merely meeting one of those edges.
   const int removed = fec.FiniteElementForGeometry(Geometry::CUBE)->GetDof() -
                       lower.FiniteElementForGeometry(Geometry::CUBE)->GetDof();
   REQUIRE(fixed.GetTrueVSize() == (high == 2 ? 416 : 1368));
   REQUIRE(variable.GetTrueVSize() == fixed.GetTrueVSize() - removed);
   Vector tdofs(variable.GetTrueVSize());
   tdofs.Randomize(1);
   GridFunction field(&variable);
   field.SetFromTrueDofs(tdofs);
   int npoints;
   REQUIRE(MaxTraceJump(field, npoints) < 1e-11);
   REQUIRE(npoints > 0);
}

#ifdef MFEM_USE_MPI
TEST_CASE("Shared hp DOFs communicate exactly their participating ranks",
          "[Parallel][VariableOrder][HpCutGroups]")
{
   int size, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (size != 1 && size != 2 && size != 4) { return; }
   const int nx = size == 1 ? 1 : 2, ny = size == 4 ? 2 : 1;
   Mesh serial = Mesh::MakeCartesian3D(nx, ny, 1, Element::HEXAHEDRON,
                                       nx, ny, 1);
   serial.EnsureNCMesh();
   Array<int> partition(serial.GetNE());
   Vector x(3);
   for (int i = 0; i < serial.GetNE(); ++i)
   {
      serial.GetElementCenter(i, x);
      partition[i] = int(x[0]) + nx * int(x[1]);
   }
   ParMesh mesh(MPI_COMM_WORLD, serial, partition.GetData());
   H1_FECollection collection(2, 3);

   // For p1 and p2 the physical nodal coordinates uniquely identify the
   // public scalar DOFs. Compute their rank sets from the serial elements.
   auto point = [](const Vector &v)
   {
      return std::array<int, 3> {{
            int(std::lround(2*v[0])),
            int(std::lround(2*v[1])),
            int(std::lround(2*v[2]))
         }
      };
   };
   for (auto ordering : {Ordering::byNODES, Ordering::byVDIM})
   {
      for (int copies : {1, 2})
      {
         ParFiniteElementSpace fes(&mesh, &collection, copies, ordering);
         for (int mixed : {0, 1, 0})
         {
            CAPTURE(size, rank, mixed, ordering, copies);
            auto order = [&](int r)
            {
               return mixed && (r == 0 || (size == 4 && r == 3)) ? 2 : 1;
            };
            std::map<std::array<int, 3>, int> masks;
            for (int i = 0; i < serial.GetNE(); ++i)
            {
               const int r = partition[i];
               const auto &nodes = collection.GetFE(Geometry::CUBE, order(r))->GetNodes();
               for (int j = 0; j < nodes.GetNPoints(); ++j)
               {
                  serial.GetElementTransformation(i)->Transform(nodes.IntPoint(j), x);
                  masks[point(x)] |= 1 << r;
               }
            }
            if (mixed && size == 4) { CHECK(masks.at({{2, 2, 1}}) == 9); }
            for (int i = 0; i < mesh.GetNE(); ++i) { fes.SetElementOrder(i, order(rank)); }
            fes.Update(false);

            Array<int> expected(fes.GetNDofs()), location(fes.GetNDofs()), local;
            expected = 0;
            for (int i = 0; i < mesh.GetNE(); ++i)
            {
               fes.GetElementDofs(i, local);
               const auto &nodes = fes.GetFE(i)->GetNodes();
               for (int j = 0; j < local.Size(); ++j)
               {
                  mesh.GetElementTransformation(i)->Transform(nodes.IntPoint(j), x);
                  const auto key = point(x);
                  expected[local[j]] = masks.at(key);
                  location[local[j]] = 1 + key[0] + 10*key[1] + 100*key[2];
               }
            }
            std::unique_ptr<GroupCommunicator> sgc(fes.ScalarGroupComm());
            Array<int> scalar_marker(fes.GetNDofs());
            scalar_marker = 1 << rank;
            sgc->Reduce<int>(scalar_marker, GroupCommunicator::BitOR);
            sgc->Bcast(scalar_marker);
            for (int i = 0; i < scalar_marker.Size(); ++i)
            {
               CHECK(scalar_marker[i] == expected[i]);
            }

            Array<int> marker(fes.GetVSize());
            for (int d = 0; d < fes.GetNDofs(); ++d)
            {
               for (int c = 0; c < copies; ++c)
               { marker[fes.DofToVDof(d, c)] = 1 << (rank + c*size); }
            }
            fes.Synchronize(marker);
            Array<int> values(fes.GetVSize());
            for (int d = 0; d < fes.GetNDofs(); ++d)
            {
               for (int c = 0; c < copies; ++c)
               {
                  values[fes.DofToVDof(d, c)] = (rank+1)*(location[d]+1000*c);
               }
            }
            fes.GroupComm().Reduce<int>(values, GroupCommunicator::Sum);
            fes.GroupComm().Bcast(values);
            for (int d = 0; d < fes.GetNDofs(); ++d)
            {
               CHECK(expected[d] != 0);
               int weight = 0;
               for (int r = 0; r < size; ++r)
               {
                  if (expected[d] & (1 << r)) { weight += r+1; }
               }
               for (int c = 0; c < copies; ++c)
               {
                  const int vdof = fes.DofToVDof(d, c);
                  CHECK(marker[vdof] == (expected[d] << (c*size)));
                  CHECK(values[vdof] == weight*(location[d]+1000*c));
               }
            }
         }
      }
   }
}

TEST_CASE("Shared hp ND slots retain physical orientation and component",
          "[Parallel][VariableOrder][HpCutGroups]")
{
   int size;
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   if (size != 2 && size != 4) { return; }
   Mesh serial = var_order_test::MakeInternalEdgeMesh(true);
   // Cut through the refined region to share several rotated quadrilaterals.
   Array<int> partition(serial.GetNE());
   for (int e = 0; e < serial.GetNE(); ++e) { partition[e] = e % size; }
   ParMesh mesh(MPI_COMM_WORLD, serial, partition.GetData());
   ND_FECollection collection(1, 3);
   for (auto ordering : {Ordering::byNODES, Ordering::byVDIM})
   {
      ParFiniteElementSpace fes(&mesh, &collection, 2, ordering);
      for (int e = 0; e < mesh.GetNE(); ++e) { fes.SetElementOrder(e, 3); }
      fes.Update(false);
      Vector expected(fes.GetVSize());
      expected = 0.0;
      for (int c = 0; c < 2; ++c)
      {
         VectorFunctionCoefficient field(3, [c](const Vector &x, Vector &a)
         {
            for (int j = 0; j < 3; ++j)
            {
               const int k = 1 + j + 3*c;
               a[j] = 13*k + x[0]*(1+k) + x[1]*(4+2*k) + x[2]*(9+3*k);
            }
         });
         for (int e = 0; e < mesh.GetNE(); ++e)
         {
            const auto *fe = fes.GetFE(e);
            Vector local(fe->GetDof());
            fe->Project(field, *mesh.GetElementTransformation(e), local);
            Array<int> dofs;
            fes.GetElementDofs(e, dofs);
            for (int j = 0; j < dofs.Size(); ++j)
            {
               const int d = dofs[j];
               expected[fes.DofToVDof(d >= 0 ? d : -1-d, c)] =
                  d >= 0 ? local[j] : -local[j];
            }
         }
      }
      // Bcast must preserve the physical edge and face trace in the common
      // signed orientation. Distinct nonconstant fields distinguish slots.
      fes.ApplyDofSigns(expected.HostReadWrite());
      Vector shared(expected);
      auto &gc = fes.GroupComm();
      gc.Bcast<real_t>(shared.HostReadWrite());
      shared -= expected;
      CHECK(shared.Normlinf() < 1e-10);
      Vector count(fes.GetVSize());
      count = 1.0;
      gc.Reduce<real_t>(count.HostReadWrite(), GroupCommunicator::Sum);
      gc.Bcast<real_t>(count.HostReadWrite());
      CHECK(count.Max() > 1);
      shared = expected;
      gc.Reduce<real_t>(shared.HostReadWrite(), GroupCommunicator::Sum);
      gc.Bcast<real_t>(shared.HostReadWrite());
      for (int j = 0; j < shared.Size(); ++j)
      {
         shared[j] -= count[j]*expected[j];
      }
      CHECK(shared.Normlinf() < 1e-10);
   }
}

TEST_CASE("Parallel internal edge constraints retain the maximal trace space",
          "[FiniteElementSpace][NCMesh][HpInternalEdge][Parallel]")
{
   using namespace var_order_test;
   const int kind = GENERATE(0, 1, 2); // H1, nodal ND, integrated ND
   const int high = GENERATE(2, 3);
   const bool rotated = GENERATE(false, true);
   CAPTURE(kind, high, rotated);
   Mesh serial = MakeInternalEdgeMesh(rotated);
   int size;
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   if (size > serial.GetNE()) { return; }

   Array<int> partition(serial.GetNE());
   for (int e = 0; e < serial.GetNE(); e++) { partition[e] = e % size; }
   ParMesh mesh(MPI_COMM_WORLD, serial, partition.GetData());
   std::unique_ptr<FiniteElementCollection> fec;
   if (kind == 0) { fec.reset(new H1_FECollection(high, 3)); }
   else
   {
      fec.reset(new ND_FECollection(high, 3, BasisType::GaussLobatto,
                                    kind == 1 ? BasisType::GaussLegendre :
                                    BasisType::IntegratedGLL));
   }
   ParFiniteElementSpace fes(&mesh, fec.get());
   SetInternalEdgeOrders(fes, high);
   // Broken-space dimensions minus the independently computed trace ranks.
   const int expected = kind == 0 ? (high == 2 ? 152 : 483) :
                        (high == 2 ? 416 : 1365);
   REQUIRE(fes.GlobalTrueVSize() == expected);

   const auto *restriction = fes.GetRestrictionMatrix();
   int failures = (!restriction || restriction->Height() != fes.GetTrueVSize() ||
                   restriction->Width() != fes.GetVSize());
   MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   REQUIRE(failures == 0);
   Vector truth(fes.GetTrueVSize()), restricted(fes.GetTrueVSize());
   ParGridFunction field(&fes);
   const HYPRE_BigInt offset = fes.GetMyTDofOffset();
   for (int col = 0; col < expected; ++col)
   {
      truth = 0.0;
      if (col >= offset && col < offset + truth.Size())
      {
         truth[col - offset] = 1.0;
      }
      field.SetFromTrueDofs(truth);
      restriction->Mult(field, restricted);
      restricted -= truth;
      failures += (restricted.CheckFinite() != 0 ||
                   restricted.Normlinf() >= 1e-12);
   }
   MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   REQUIRE(failures == 0);
}

TEST_CASE("Parallel hp vector copies use the scalar internal edge space",
          "[FiniteElementSpace][NCMesh][HpInternalEdgeVector][Parallel]")
{
   using namespace var_order_test;
   const int kind = GENERATE(0, 1, 2); // H1, nodal ND, integrated ND
   const int high = GENERATE(2, 3);
   const int copies = GENERATE(2, 3);
   const auto ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);
   CAPTURE(kind, high, copies, ordering);
   Mesh serial = MakeInternalEdgeMesh(high == 3);
   int size, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (size != 1 && size != 2 && size != 4) { return; }

   Array<int> partition(serial.GetNE());
   Vector center(3);
   for (int e = 0; e < serial.GetNE(); e++)
   {
      serial.GetElementCenter(e, center);
      int owner = 0;
      if (size == 2 && 1 < center[0] && center[0] < 1.5) { owner = 1; }
      if (size == 4 && center[0] > 1)
      {
         owner = center[0] < 1.25 ? 1 : (center[0] > 1.5 ? 2 : 3);
      }
      partition[e] = high == 3 ? size - owner - 1 : owner;
   }
   ParMesh mesh(MPI_COMM_WORLD, serial, partition.GetData());
   std::unique_ptr<FiniteElementCollection> fec;
   if (kind == 0) { fec.reset(new H1_FECollection(1, 3)); }
   else
   {
      fec.reset(new ND_FECollection(1, 3, BasisType::GaussLobatto,
                                    kind == 1 ? BasisType::GaussLegendre :
                                    BasisType::IntegratedGLL));
   }
   ParFiniteElementSpace scalar(&mesh, fec.get());
   SetInternalEdgeOrders(scalar, high);
   const int expected = kind == 0 ? (high == 2 ? 152 : 483) :
                        (high == 2 ? 416 : 1365);
   REQUIRE(scalar.GlobalTrueVSize() == expected);
   ParFiniteElementSpace vector(&mesh, fec.get(), copies, ordering);
   SetInternalEdgeOrders(vector, high);
   REQUIRE(vector.GlobalTrueVSize() == copies * expected);

   const auto component_dof = [&](int n, int i, int c)
   {
      return ordering == Ordering::byNODES ? i + c*n : i*copies + c;
   };
   const auto true_dof = [&](int i, int c)
   {
      return component_dof(scalar.GetTrueVSize(), i, c);
   };
   const auto local_dof = [&](int i, int c)
   {
      return component_dof(scalar.GetVSize(), i, c);
   };

   std::vector<int> starts(size+1);
   const int local_count = scalar.GetTrueVSize();
   MPI_Allgather(&local_count, 1, MPI_INT, starts.data()+1, 1, MPI_INT,
                 MPI_COMM_WORLD);
   for (int r = 0; r < size; r++) { starts[r+1] += starts[r]; }
   const auto global_dof = [&](int i, int c)
   {
      const int r = std::upper_bound(starts.begin(), starts.end(), i) -
                    starts.begin() - 1;
      return copies*starts[r] + component_dof(starts[r+1]-starts[r],
                                              i-starts[r], c);
   };
   int invalid = (vector.GetTrueVSize() != copies * local_count ||
                  vector.GetVSize() != copies * scalar.GetVSize() ||
                  vector.GetMyTDofOffset() != copies * starts[rank]);
   MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   REQUIRE(invalid == 0);

   SparseMatrix Pscalar, Pvector;
   scalar.Dof_TrueDof_Matrix()->MergeDiagAndOffd(Pscalar);
   vector.Dof_TrueDof_Matrix()->MergeDiagAndOffd(Pvector);
   const auto max_difference = [&](const SparseMatrix &a, const SparseMatrix &b,
                                   auto row_index, auto column_index)
   {
      real_t maximum = 0.0;
      for (int r = 0; r < a.Height(); r++)
      {
         for (int c = 0; c < copies; c++)
         {
            std::map<int, real_t> row;
            const int rr = row_index(r, c);
            for (int j = 0; j < a.RowSize(r); j++)
            {
               row[column_index(a.GetRowColumns(r)[j], c)] +=
                  a.GetRowEntries(r)[j];
            }
            for (int j = 0; j < b.RowSize(rr); j++)
            {
               row[b.GetRowColumns(rr)[j]] -= b.GetRowEntries(rr)[j];
            }
            for (const auto &entry : row)
            {
               if (!std::isfinite(entry.second)) { return infinity(); }
               maximum = std::max(maximum, std::abs(entry.second));
            }
         }
      }
      return maximum;
   };
   const auto &Rscalar = *scalar.GetRestrictionMatrix();
   const auto &Rvector = *vector.GetRestrictionMatrix();
   real_t errors[2] = {max_difference(Pscalar, Pvector, local_dof, global_dof),
                       max_difference(Rscalar, Rvector, true_dof, local_dof)
                      };
   for (int i = 0; i < scalar.GetVSize(); i++)
   {
      const int t = scalar.GetLocalTDofNumber(i);
      for (int c = 0; c < copies; c++)
      {
         invalid += (vector.GetLocalTDofNumber(local_dof(i, c)) !=
                     (t < 0 ? -1 : true_dof(t, c)));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   REQUIRE(errors[0] < 1e-12);
   REQUIRE(errors[1] < 1e-12);
   REQUIRE(invalid == 0);
}

TEST_CASE("Parallel variable-order H1 essential markers select vector components",
          "[FiniteElementSpace][NCMesh][HpH1VectorMarker][Parallel]")
{
   const int copies = GENERATE(2, 3);
   const auto ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);
   const int size = Mpi::WorldSize();
   if (size > 2) { return; }

   Mesh serial = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
   // Attribute 1 covers the x=0 end face. Attribute 2 covers the other faces.
   for (int b = 0; b < serial.GetNBE(); ++b)
   {
      Array<int> vertices;
      serial.GetBdrElementVertices(b, vertices);
      bool end_face = true;
      for (int v : vertices) { end_face &= (serial.GetVertex(v)[0] == 0.0); }
      serial.GetBdrElement(b)->SetAttribute(end_face ? 1 : 2);
   }
   serial.SetAttributes();
   serial.EnsureNCMesh();
   const int partition[] = {0, size - 1};
   ParMesh mesh(MPI_COMM_WORLD, serial, partition);
   H1_FECollection fec(1, 3);
   ParFiniteElementSpace scalar(&mesh, &fec), vector(&mesh, &fec, copies,
                                                     ordering);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      scalar.SetElementOrder(e, 2);
      vector.SetElementOrder(e, 2);
   }
   scalar.Update(false);
   vector.Update(false);
   REQUIRE(scalar.GlobalTrueVSize() == 45);
   REQUIRE(vector.GlobalTrueVSize() == 45 * copies);

   Array<int> attrs(2);
   for (int selection = 0; selection < 3; ++selection)
   {
      attrs = 0;
      if (selection == 1) { attrs[0] = 1; }
      if (selection == 2) { attrs = 1; }
      Array<int> scalar_list, all_vdofs;
      scalar.GetEssentialTrueDofs(attrs, scalar_list);
      vector.GetEssentialVDofs(attrs, all_vdofs);
      // 5x3x3 nodes, including 3 interior nodes. The end face has 3x3 nodes.
      const int expected_scalar = selection == 0 ? 0 : (selection == 1 ? 9 : 42);
      int global = scalar_list.Size();
      MPI_Allreduce(MPI_IN_PLACE, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      REQUIRE(global == expected_scalar);

      for (int component = -1; component < copies; ++component)
      {
         CAPTURE(copies, ordering, selection, component);
         Array<int> actual, helper, expected(vector.GetTrueVSize());
         expected = 0;
         for (int t : scalar_list)
         {
            for (int c = 0; c < copies; ++c)
            {
               if (component >= 0 && c != component) { continue; }
               const int vt = ordering == Ordering::byNODES ?
                              t + c * scalar.GetTrueVSize() : t * copies + c;
               expected[vt] = 1;
            }
         }
         vector.GetEssentialTrueDofs(attrs, actual, component);
         // The public helper must also honor component when the caller supplies
         // markers for every component.
         vector.GetEssentialTrueDofsVar(attrs, all_vdofs, helper, component);
         Array<int> expected_list;
         FiniteElementSpace::MarkerToList(expected, expected_list);
         int wrong = (actual != expected_list || helper != expected);
         MPI_Allreduce(MPI_IN_PLACE, &wrong, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
         REQUIRE(wrong == 0);
         global = actual.Size();
         MPI_Allreduce(MPI_IN_PLACE, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
         REQUIRE(global == expected_scalar * (component < 0 ? copies : 1));
      }
   }
}

TEST_CASE("Parallel RT face orders are independent of interior edge orders",
          "[FiniteElementSpace][NCMesh][RTFaceOrder][Parallel]")
{
   using namespace var_order_test;
   const int high = GENERATE(2, 3);
   const int ob = GENERATE(BasisType::GaussLegendre, BasisType::IntegratedGLL);
   const bool rotated = GENERATE(false, true);
   CAPTURE(high, ob, rotated);
   Mesh serial = MakeInternalEdgeMesh(rotated);
   Array<int> partition(serial.GetNE());
   int size, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   for (int e = 0; e < serial.GetNE(); e++) { partition[e] = e % size; }
   ParMesh mesh(MPI_COMM_WORLD, serial, partition.GetData());
   RT_FECollection fec(high-1, 3, BasisType::GaussLobatto, ob);
   RT_FECollection lower(high-2, 3, BasisType::GaussLobatto, ob);
   ParFiniteElementSpace fixed(&mesh, &fec), variable(&mesh, &fec);
   SetInternalEdgeOrders(variable, high);
   const int removed = fec.FiniteElementForGeometry(Geometry::CUBE)->GetDof() -
                       lower.FiniteElementForGeometry(Geometry::CUBE)->GetDof();
   REQUIRE(fixed.GlobalTrueVSize() == (high == 2 ? 416 : 1368));
   REQUIRE(variable.GlobalTrueVSize() == fixed.GlobalTrueVSize() - removed);
   Vector tdofs(variable.GetTrueVSize());
   tdofs.Randomize(rank+1);
   ParGridFunction field(&variable);
   field.SetFromTrueDofs(tdofs);
   int npoints;
   REQUIRE(MaxTraceJump(field, npoints) < 1e-11);
}
#endif
TEST_CASE("Integrated ND p-coarsening preserves the mean",
          "[FiniteElement][ND][L2Transfer]")
{
   ND_SegmentElement source(3, BasisType::IntegratedGLL);
   ND_SegmentElement target(1, BasisType::IntegratedGLL);
   IsoparametricTransformation identity;
   identity.SetIdentityTransformation(Geometry::SEGMENT);
   VectorFunctionCoefficient square(1, [](const Vector &x, Vector &v)
   {
      v(0) = x(0) * x(0);
   });
   Vector source_dofs(source.GetDof()), target_dofs(target.GetDof());
   source.Project(square, identity, source_dofs);
   DenseMatrix transfer;
   target.GetTransferMatrix(source, identity, transfer);
   transfer.Mult(source_dofs, target_dofs);
   // The target basis is the constant one on the unit segment.
   REQUIRE(target_dofs.CheckFinite() == 0);
   REQUIRE(target_dofs(0) == MFEM_Approx(real_t(1.0 / 3.0)));
}

TEST_CASE("Integrated vector p-transfer is L2 orthogonal",
          "[FiniteElement][ND][RT][L2Transfer]")
{
   const int dim = GENERATE(2, 3);
   const bool nd = GENERATE(true, false);
   const int low = GENERATE(1, 2);
   const bool coarsen = GENERATE(true, false);
   const bool scaled = GENERATE(false, true);
   CAPTURE(dim, nd, low, coarsen, scaled);
   const int source_order = coarsen ? low + 2 : low;
   const int target_order = coarsen ? low : low + 2;
   std::unique_ptr<FiniteElementCollection> source_fec, target_fec;
   if (nd)
   {
      source_fec.reset(new ND_FECollection(source_order, dim,
                                           BasisType::GaussLobatto,
                                           BasisType::IntegratedGLL));
      target_fec.reset(new ND_FECollection(target_order, dim,
                                           BasisType::GaussLobatto,
                                           BasisType::IntegratedGLL));
   }
   else
   {
      source_fec.reset(new RT_FECollection(source_order - 1, dim,
                                           BasisType::GaussLobatto,
                                           BasisType::IntegratedGLL));
      target_fec.reset(new RT_FECollection(target_order - 1, dim,
                                           BasisType::GaussLobatto,
                                           BasisType::IntegratedGLL));
   }
   const auto geom = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   const FiniteElement &source = *source_fec->FiniteElementForGeometry(geom);
   const FiniteElement &target = *target_fec->FiniteElementForGeometry(geom);
   IsoparametricTransformation trans;
   trans.SetIdentityTransformation(geom);
   const real_t scale[3] = {0.5, 0.75, 0.25};
   real_t det = 1.0;
   if (scaled)
   {
      for (int d = 0; d < dim; d++)
      {
         det *= scale[d];
         for (int j = 0; j < trans.GetPointMat().Width(); j++)
         {
            trans.GetPointMat()(d, j) *= scale[d];
         }
      }
   }
   DenseMatrix transfer;
   target.GetTransferMatrix(source, trans, transfer);
   REQUIRE(transfer.CheckFinite() == 0);

   // Check every source basis column against every target test function.
   // This separate, deliberately overintegrated rule checks the defining
   // orthogonality rather than comparing two transfer implementations.
   DenseMatrix residual(target.GetDof(), source.GetDof());
   DenseMatrix ts(target.GetDof(), dim), ss(source.GetDof(), dim);
   DenseMatrix transferred(source.GetDof(), dim);
   Vector point(dim);
   const IntegrationRule &ir = IntRules.Get(geom, 16);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      IntegrationPoint source_ip;
      trans.Transform(ip, point);
      source_ip.Set(point.GetData(), dim);
      target.CalcVShape(ip, ts);
      source.CalcVShape(source_ip, ss);
      MultAtB(transfer, ts, transferred);
      for (int d = 0; d < dim; d++)
      {
         // Analytic covariant/contravariant pullback for a diagonal map.
         const real_t pullback = !scaled ? 1.0 :
                                 (nd ? scale[d] : det / scale[d]);
         for (int j = 0; j < source.GetDof(); j++)
         {
            transferred(j, d) -= pullback * ss(j, d);
         }
      }
      AddMult_a_ABt(ip.weight, ts, transferred, residual);
   }
   REQUIRE(residual.CheckFinite() == 0);
   REQUIRE(residual.MaxMaxNorm() < 1e-11);
}

TEST_CASE("ND Transfer Matrix Pulls Back Covariantly",
          "[FiniteElement][ND]")
{
   // The transfer matrix from a coarse ND element to a fine one must map the
   // coarse reference field u_c to J^T u_c(F(x)), where F is the map from the
   // fine reference element into the coarse one and J its Jacobian, also when
   // J is not symmetric (a face whose axes are swapped) and when the fine
   // element is an edge inside a face (J of size 2x1).
   const int p = GENERATE(1, 2, 3);
   const int ob_type = GENERATE(BasisType::GaussLegendre,
                                BasisType::IntegratedGLL);
   CAPTURE(p, ob_type);
   ND_FECollection fec(p, 3, BasisType::GaussLobatto, ob_type);
   const FiniteElement &quad = *fec.FiniteElementForGeometry(Geometry::SQUARE);
   const FiniteElement &seg = *fec.FiniteElementForGeometry(Geometry::SEGMENT);

   IsoparametricTransformation id;
   id.SetFE(&QuadrilateralFE);
   id.GetPointMat().SetSize(2, 4);
   const real_t unit[2][4] = {{0, 1, 1, 0}, {0, 0, 1, 1}};
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 4; j++) { id.GetPointMat()(i, j) = unit[i][j]; }
   }

   const int comp = GENERATE(0, 1);
   CAPTURE(comp);
   {
      Vector u(2);
      u = 0.0;
      u(comp) = 1.0;
      VectorConstantCoefficient uc(u);
      Vector cdofs(quad.GetDof());
      quad.Project(uc, id, cdofs);

      SECTION("Face into face, axes swapped and halved")
      {
         // F(s,t) = (t/2, s): J = [0 1/2; 1 0]
         IsoparametricTransformation T;
         T.SetFE(&QuadrilateralFE);
         T.GetPointMat().SetSize(2, 4);
         const real_t pm[2][4] = {{0, 0, 0.5, 0.5}, {0, 1, 1, 0}};
         for (int i = 0; i < 2; i++)
         {
            for (int j = 0; j < 4; j++) { T.GetPointMat()(i, j) = pm[i][j]; }
         }
         DenseMatrix I;
         quad.GetTransferMatrix(quad, T, I);
         Vector fdofs(quad.GetDof());
         I.Mult(cdofs, fdofs);
         // J^T u
         const real_t expect[2] = {comp == 1 ? 1.0 : 0.0,
                                   comp == 0 ? 0.5 : 0.0
                                  };
         DenseMatrix vs(quad.GetDof(), 2);
         IntegrationPoint ip;
         for (int i = 1; i <= p + 1; i++)
         {
            for (int j = 1; j <= p + 1; j++)
            {
               ip.Set2(real_t(i) / (p + 2), real_t(j) / (p + 2));
               quad.CalcVShape(ip, vs);
               for (int d = 0; d < 2; d++)
               {
                  real_t v = 0.0;
                  for (int k = 0; k < quad.GetDof(); k++)
                  {
                     v += fdofs(k) * vs(k, d);
                  }
                  REQUIRE(std::abs(v - expect[d]) < 1e-12);
               }
            }
         }
      }

      SECTION("Edge inside a face")
      {
         // horizontal, vertical and reversed vertical edges inside the face
         const real_t ends[3][2][2] = {{{0.25, 0.5}, {0.75, 0.5}},
            {{0.5, 0.25}, {0.5, 0.75}},
            {{0.5, 0.75}, {0.5, 0.25}}
         };
         for (const auto &e : ends)
         {
            IsoparametricTransformation T;
            T.SetFE(&SegmentFE);
            T.GetPointMat().SetSize(2, 2);
            for (int i = 0; i < 2; i++)
            {
               for (int j = 0; j < 2; j++) { T.GetPointMat()(i, j) = e[j][i]; }
            }
            DenseMatrix I;
            seg.GetTransferMatrix(quad, T, I);
            Vector fdofs(seg.GetDof());
            I.Mult(cdofs, fdofs);
            // u . (J t) with J t = b - a
            const real_t expect = e[1][comp] - e[0][comp];
            DenseMatrix vs(seg.GetDof(), 1);
            IntegrationPoint ip;
            for (int i = 1; i <= p + 1; i++)
            {
               ip.x = real_t(i) / (p + 2);
               seg.CalcVShape(ip, vs);
               real_t v = 0.0;
               for (int k = 0; k < seg.GetDof(); k++) { v += fdofs(k) * vs(k, 0); }
               REQUIRE(std::abs(v - expect) < 1e-12);
            }
         }
      }
   }
}

TEST_CASE("Variable Order ND on Nonconforming Hex Mesh",
          "[FiniteElementSpace][NCMesh]")
{
   using namespace var_order_test;
   const int p = GENERATE(1, 2, 3);
   // IntegratedGLL (used by LOR) makes the elements non-nodal, so that the
   // transfer matrices come from the local L2 projection instead.
   const int ob_type = GENERATE(BasisType::GaussLegendre,
                                BasisType::IntegratedGLL);
   CAPTURE(p, ob_type);

   Mesh mesh = MakeLocallyRefinedHexMesh();
   ND_FECollection fec(p, 3, BasisType::GaussLobatto, ob_type);

   SECTION("Equal orders give the fixed-order space")
   {
      FiniteElementSpace fixed(&mesh, &fec);
      FiniteElementSpace var(&mesh, &fec);
      for (int e = 0; e < mesh.GetNE(); e++) { var.SetElementOrder(e, p); }
      var.Update(false);
      REQUIRE(var.IsVariableOrder());
      REQUIRE(var.GetNDofs() == fixed.GetNDofs());
      REQUIRE(var.GetNConformingDofs() == fixed.GetNConformingDofs());
   }

   SECTION("Mixed orders: tangential continuity")
   {
      FiniteElementSpace fespace(&mesh, &fec);
      Vector center(3);
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         mesh.GetElementCenter(e, center);
         fespace.SetElementOrder(e, center(0) > 0.5 && center(1) > 0.5 ?
                                 p + 1 : p);
      }
      fespace.Update(false);

      Vector tdofs(fespace.GetTrueVSize());
      tdofs.Randomize(1);
      GridFunction x(&fespace);
      fespace.GetProlongationMatrix()->Mult(tdofs, x);
      int npoints;
      REQUIRE(MaxTraceJump(x, npoints) < 1e-12);
      REQUIRE(npoints > 0);
   }
}

#ifdef MFEM_USE_MPI
TEST_CASE("Parallel Variable Order FiniteElementSpace",
          "[FiniteElementCollection]"
          "[FiniteElementSpace]"
          "[NCMesh][Parallel]")
{
   SECTION("Quad mesh")
   {
      // 2-by-2 element quad mesh
      Mesh mesh = MakeCartesianMesh(2, 2);
      mesh.EnsureNCMesh();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      // Standard H1 space with order 1 elements
      H1_FECollection fe_coll(1, pmesh.Dimension());
      ParFiniteElementSpace pfes(&pmesh, &fe_coll);

      REQUIRE(pfes.GlobalTrueVSize() == 9);

      // Convert to variable order space by p-refinement
      // Increase order on all elements
      for (int i = 0; i < pmesh.GetNE(); i++)
      {
         pfes.SetElementOrder(i, pfes.GetElementOrder(i) + 1);
      }
      pfes.Update(false);

      // DOFs for vertices + edges + elements = 9 + 12 + 4 = 25
      REQUIRE(pfes.GlobalTrueVSize() == 25);

      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank == 0) { pfes.SetElementOrder(0, 4); }
      pfes.Update(false);

      Array<Refinement> refs;
      if (rank == 0) { refs.Append(Refinement(0)); }
      pmesh.GeneralRefinement(refs);
      pfes.Update(false);

      TestSolvePar(pfes);
   }

   SECTION("Hex mesh")
   {
      // 2^3 element hex mesh
      Mesh mesh = MakeCartesianMesh(2, 3);
      mesh.EnsureNCMesh();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      // Standard H1 space with order 1 elements
      H1_FECollection fe_coll(1, pmesh.Dimension());
      ParFiniteElementSpace pfes(&pmesh, &fe_coll);

      REQUIRE(pfes.GlobalTrueVSize() == 27);  // 3^3

      // Convert to variable order space by p-refinement
      for (int i = 0; i < pmesh.GetNE(); i++)
      {
         pfes.SetElementOrder(i, pfes.GetElementOrder(i) + 1);
      }
      pfes.Update(false);

      // DOFs for vertices + edges + faces + elements = 27 + 54 + 36 + 8 = 125
      REQUIRE(pfes.GlobalTrueVSize() == 125);  // 5^3

      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank == 0) { pfes.SetElementOrder(0, 4); }
      pfes.Update(false);

      Array<Refinement> refs;
      if (rank == 0) { refs.Append(Refinement(0)); }
      pmesh.GeneralRefinement(refs);
      pfes.Update(false);

      TestSolvePar(pfes);
   }

   SECTION("Hex mesh with intermediate orders")
   {
      // Test ParFiniteElementSpace::MarkIntermediateEntityDofs
      // This test is designed for 2 MPI ranks. If more than 2 ranks are used,
      // the test is run on only the first 2 ranks via a split communicator.
      int numprocs, rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

      MPI_Comm comm2;
      MPI_Comm_split(MPI_COMM_WORLD, rank < 2 ? 0 : 1, rank, &comm2);

      if (rank < 2)
      {
         // 2x1x1 element hex mesh
         Mesh mesh = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
         mesh.EnsureNCMesh();

         Array<int> partition(2);
         partition = 0;
         if (numprocs > 1) { partition[1] = 1; }

         ParMesh pmesh(comm2, mesh, partition.GetData());
         mesh.Clear();

         // Standard H1 space with order 1 elements
         H1_FECollection fe_coll(1, pmesh.Dimension());
         ParFiniteElementSpace fespace(&pmesh, &fe_coll);

         {
            Array<Refinement> refs;
            if (rank == 1) { refs.Append(Refinement(0)); }
            pmesh.GeneralRefinement(refs);
            fespace.Update(false);
         }

         {
            Array<Refinement> refs;
            if (rank == 1) { refs.Append(Refinement(4)); }
            pmesh.GeneralRefinement(refs);
            fespace.Update(false);
         }

         if (rank == 1)
         {
            for (int elem=0; elem<pmesh.GetNE(); ++elem)
            {
               const int p_elem = fespace.GetElementOrder(elem);
               fespace.SetElementOrder(elem, p_elem + 1);
            }
         }
         fespace.Update(false);

         {
            Array<Refinement> refs;
            if (rank == 1) { refs.Append(Refinement(6)); }
            if (rank == 0) { refs.Append(Refinement(0)); }
            pmesh.GeneralRefinement(refs);
            fespace.Update(false);
         }

         {
            Array<Refinement> refs;
            if (rank == 1) { refs.Append(Refinement(10)); }
            pmesh.GeneralRefinement(refs);
            fespace.Update(false);
         }

         if (rank == 1) { fespace.SetElementOrder(3, 3); }
         fespace.Update(false);

         // Set at least order 2 everywhere
         for (int elem=0; elem<pmesh.GetNE(); ++elem)
         {
            const int p_elem = fespace.GetElementOrder(elem);
            if (p_elem < 2) { fespace.SetElementOrder(elem, 2); }
         }
         fespace.Update(false);

         TestSolvePar(fespace);
      }
   }

   SECTION("Quad/hex mesh ND/RT")
   {
      using namespace var_order_test;

      const auto space_type = GENERATE(SpaceType::RT, SpaceType::ND);
      const int dim = GENERATE(2, 3);

      Mesh mesh = MakeCartesianMesh(dim == 2 ? 4 : 2, dim);

      mesh.EnsureNCMesh();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      int ndof0, ncdof2;

      std::unique_ptr<FiniteElementCollection> fec;
      if (space_type == SpaceType::RT)
      {
         // Standard RT space with order 0 elements
         fec.reset(new RT_FECollection(0, dim));
         ndof0 = dim == 2 ? 40 : 36;
         // Compare to fixed RT_1 rather than encoding the erroneous RT_2
         // counts produced by Clone before the collection-order fix.
         RT_FECollection reference(1, dim);
         ParFiniteElementSpace reference_space(&pmesh, &reference);
         ncdof2 = reference_space.GlobalTrueVSize();
      }
      else
      {
         // Standard ND space with order 1 elements
         fec.reset(new ND_FECollection(1, dim));

         if (dim == 2)
         {
            ndof0 = 40;
            ncdof2 = 144;
         }
         else
         {
            ndof0 = 54;
            ncdof2 = 300;
         }
      }

      ParFiniteElementSpace fespace(&pmesh, fec.get());

      REQUIRE(fespace.GlobalTrueVSize() == ndof0);

      // Convert to variable order space by p-refinement
      // Increase order on all elements
      for (int i = 0; i < pmesh.GetNE(); i++)
      {
         fespace.SetElementOrder(i, fespace.GetElementOrder(i) + 1);
      }
      fespace.Update(false);

      REQUIRE(fespace.GlobalTrueVSize() == ncdof2);
      for (int e = 0; e < fespace.GetNE(); e++)
      {
         REQUIRE(fespace.GetFE(e)->GetOrder() == fespace.GetElementOrder(e));
      }

      TestSolveParVec(fespace);
   }
}

TEST_CASE("Serial-parallel Comparison for Variable Order FiniteElementSpace",
          "[FiniteElementCollection]"
          "[FiniteElementSpace]"
          "[NCMesh][Parallel]")
{

   int dimension = GENERATE(2, 3);
   Mesh mesh = MakeCartesianMesh(4, dimension);
   TestRandomPRefinement(mesh);
}

// Variable-order H1 spaces with base order > 1 have shared edge and face DOFs,
// whose variants can differ between the ranks sharing them.
TEST_CASE("Parallel Variable Order Group Communication",
          "[FiniteElementSpace]"
          "[NCMesh][Parallel]")
{
   const int dimension = GENERATE(2, 3);
   const int base_order = GENERATE(2, 3);
   enum class Orders { ByRank, Alternating, Lowered };
   const auto orders = GENERATE(Orders::ByRank, Orders::Alternating,
                                Orders::Lowered);
   // Parallel variable-order vector spaces ordered byVDIM are not supported.
   const int vdim = GENERATE(1, 2);
   CAPTURE(dimension, base_order, int(orders), vdim);

   Mesh mesh = MakeCartesianMesh(dimension == 2 ? 4 : 3, dimension);
   mesh.EnsureNCMesh();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   H1_FECollection fec(base_order, dimension);
   ParFiniteElementSpace pfes(&pmesh, &fec, vdim);
   const int rank = pmesh.GetMyRank();
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      int p = 1;
      if (orders == Orders::ByRank) { p = base_order + rank%2; }
      else if (orders == Orders::Alternating) { p = base_order + (e+rank)%2; }
      pfes.SetElementOrder(e, p);
   }
   pfes.Update(false);
   GroupCommunicator &gcomm = pfes.GroupComm();

   // All communicated DOFs must be valid local DOFs.
   const Table &group_ldof = gcomm.GroupLDofTable();
   for (int j = 0; j < group_ldof.Size_of_connections(); j++)
   {
      const int ldof = group_ldof.GetJ()[j];
      REQUIRE((ldof >= 0 && ldof < pfes.GetVSize()));
   }

   // Communicating a conforming vector must not change it, nor restricting it
   // to the cut-space true DOFs and prolongating it back.
   real_t comm_diff = 0.0;
   {
      Vector X(pfes.GetTrueVSize());
      for (int i = 0; i < X.Size(); i++)
      {
         X(i) = std::sin(1.0 + pfes.GetMyTDofOffset() + i);
      }
      ParGridFunction x(&pfes), y(&pfes);
      x.SetFromTrueDofs(X);
      y = x;
      gcomm.Reduce<real_t>(y.HostReadWrite(), GroupCommunicator::Max);
      gcomm.Bcast<real_t>(y.HostReadWrite());
      y -= x;
      comm_diff = y.Normlinf();

      int ncut = pfes.GetVSize();
      for (int g = 1; g < group_ldof.Size(); g++)
      {
         if (!gcomm.GetGroupTopology().IAmMaster(g))
         {
            ncut -= group_ldof.RowSize(g);
         }
      }
      Array<real_t> x_ldof(x.HostReadWrite(), x.Size()), y_ldof(x.Size());
      Array<real_t> x_cut(ncut);
      const DeviceGroupCommunicator &dgc = gcomm.GetDeviceComm();
      dgc.Restrict(x_ldof, x_cut);
      dgc.Prolongate(x_cut, y_ldof);
      for (int i = 0; i < x.Size(); i++)
      {
         comm_diff = std::max(comm_diff, std::abs(y_ldof[i] - x_ldof[i]));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &comm_diff, 1, MFEM_MPI_REAL_T, MPI_MAX,
                 MPI_COMM_WORLD);
   REQUIRE(comm_diff < 1e-12);

   if (vdim > 1) { return; }

   // Every DOF must be communicated with all ranks holding it. The ranks
   // holding a DOF are found from its node position and variant order.
   int count_diff = 0;
   {
      using Key = std::array<long long, 4>;
      std::vector<Key> keys(pfes.GetVSize());
      Array<int> dofs;
      Vector pt(3), ptd;
      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         pfes.GetElementDofs(e, dofs);
         const IntegrationRule &nodes = pfes.GetFE(e)->GetNodes();
         ElementTransformation *T = pmesh.GetElementTransformation(e);
         for (int j = 0; j < dofs.Size(); j++)
         {
            pt = 0.0;
            ptd.SetDataAndSize(pt.GetData(), dimension);
            T->Transform(nodes.IntPoint(j), ptd);
            const int d = dofs[j] >= 0 ? dofs[j] : -1 - dofs[j];
            const int p = (d < pmesh.GetNV()) ? 0 : pfes.GetElementOrder(e);
            keys[d] = {std::llround(1e8*pt(0)), std::llround(1e8*pt(1)),
                       std::llround(1e8*pt(2)), p
                      };
         }
      }

      const int nranks = pmesh.GetNRanks();
      Array<int> sizes(nranks), offsets(nranks);
      int size = 4*static_cast<int>(keys.size());
      MPI_Allgather(&size, 1, MPI_INT, sizes.GetData(), 1, MPI_INT,
                    MPI_COMM_WORLD);
      offsets[0] = 0;
      for (int r = 1; r < nranks; r++)
      {
         offsets[r] = offsets[r-1] + sizes[r-1];
      }
      std::vector<long long> all_keys(offsets.Last() + sizes.Last());
      MPI_Allgatherv(keys.data()->data(), size, MPI_LONG_LONG, all_keys.data(),
                     sizes.GetData(), offsets.GetData(), MPI_LONG_LONG,
                     MPI_COMM_WORLD);
      std::map<Key, int> num_ranks;
      for (int r = 0; r < nranks; r++)
      {
         std::set<Key> rank_keys;
         for (int i = offsets[r]; i < offsets[r] + sizes[r]; i += 4)
         {
            rank_keys.insert({all_keys[i], all_keys[i+1], all_keys[i+2],
                              all_keys[i+3]});
         }
         for (const Key &key : rank_keys) { num_ranks[key]++; }
      }

      Array<int> count(pfes.GetVSize());
      count = 1;
      gcomm.Reduce<int>(count, GroupCommunicator::Sum);
      gcomm.Bcast(count);
      for (int i = 0; i < count.Size(); i++)
      {
         count_diff += (count[i] != num_ranks[keys[i]]);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &count_diff, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   REQUIRE(count_diff == 0);

   // Boundary and full projections must agree on the essential true DOFs.
   FunctionCoefficient coeff([](const Vector &x)
   {
      real_t r = 0.0;
      for (int i = 0; i < x.Size(); i++) { r += (i + 1)*x(i)*x(i)*x(i); }
      return r;
   });
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_tdofs;
   pfes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

   ParGridFunction x_bdr(&pfes), x_full(&pfes);
   x_bdr = 0.0;
   x_bdr.ProjectBdrCoefficient(coeff, ess_bdr);
   x_full.ProjectCoefficient(coeff);

   Vector X_bdr(pfes.GetTrueVSize()), X_full(pfes.GetTrueVSize());
   x_bdr.GetTrueDofs(X_bdr);
   x_full.GetTrueDofs(X_full);

   real_t max_diff = 0.0;
   for (int i : ess_tdofs)
   {
      max_diff = std::max(max_diff, std::abs(X_bdr(i) - X_full(i)));
   }
   MPI_Allreduce(MPI_IN_PLACE, &max_diff, 1, MFEM_MPI_REAL_T, MPI_MAX,
                 MPI_COMM_WORLD);
   REQUIRE(max_diff < 1e-12);
}

// A variant of an edge shared by three ranks, present on only two of them.
TEST_CASE("Parallel Variable Order Rank Subset Communication",
          "[FiniteElementSpace]"
          "[NCMesh][Parallel]")
{
   const int nranks = Mpi::WorldSize(), rank = Mpi::WorldRank();
   if (nranks < 3 || nranks > 7) { return; }

   // The bottom hexes, on ranks 0, 1, 2, 2, share the edge x = y = 0.5,
   // 0 < z < 0.5. The top ones are on the other ranks, if any.
   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   mesh.EnsureNCMesh();
   Array<int> partition(mesh.GetNE());
   Vector c;
   for (int e = 0, top = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementCenter(e, c);
      if (c(2) < 0.5)
      {
         partition[e] = std::min(int(c(0) > 0.5) + 2*int(c(1) > 0.5), 2);
      }
      else { partition[e] = (nranks > 3) ? 3 + top++ % (nranks - 3) : 2; }
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh, partition.GetData());
   mesh.Clear();

   // Order 3 on ranks 0 and 1, order 2 elsewhere.
   H1_FECollection fec(2, 3);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      pfes.SetElementOrder(e, rank < 2 ? 3 : 2);
   }
   pfes.Update(false);

   // The DOFs inside the shared edge, with the z coordinates of their nodes.
   std::map<int, real_t> edge_dofs;
   Array<int> dofs;
   Vector pt(3);
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      pfes.GetElementDofs(e, dofs);
      const IntegrationRule &nodes = pfes.GetFE(e)->GetNodes();
      for (int j = 0; j < dofs.Size(); j++)
      {
         pmesh.GetElementTransformation(e)->Transform(nodes.IntPoint(j), pt);
         if (std::abs(pt(0) - 0.5) < 1e-12 && std::abs(pt(1) - 0.5) < 1e-12 &&
             pt(2) > 1e-12 && pt(2) < 0.5 - 1e-12)
         {
            edge_dofs[dofs[j] >= 0 ? dofs[j] : -1 - dofs[j]] = pt(2);
         }
      }
   }
   REQUIRE(edge_dofs.size() == (rank < 2 ? 2u : (rank == 2 ? 1u : 0u)));

   // On ranks 0 and 1, the order 3 DOFs are communicated with each other only,
   // node by node.
   GroupCommunicator &gcomm = pfes.GroupComm();
   Array<int> count(pfes.GetVSize());
   Vector z(pfes.GetVSize());
   count = 1;
   z = 0.0;
   for (const auto &dof : edge_dofs) { z(dof.first) = dof.second; }
   gcomm.Reduce<int>(count, GroupCommunicator::Sum);
   gcomm.Bcast(count);
   gcomm.Reduce<real_t>(z.HostReadWrite(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(z.HostReadWrite());
   for (const auto &dof : edge_dofs)
   {
      CAPTURE(rank, dof.first, dof.second);
      REQUIRE(count[dof.first] == (rank < 2 ? 2 : 1));
      REQUIRE(z(dof.first) == MFEM_Approx(count[dof.first]*dof.second));
   }
}
#endif  // MFEM_USE_MPI

// Exact solution: x^2 + y^2 + z^2
static real_t exact_sln(const Vector &p)
{
   real_t x = p(0), y = p(1);
   if (p.Size() == 3)
   {
      real_t z = p(2);
      return x*x + y*y + z*z;
   }
   else
   {
      return x*x + y*y;
   }
}

static real_t exact_rhs(const Vector &p)
{
   return (p.Size() == 3) ? -6.0 : -4.0;
}

static void TestSolve(FiniteElementSpace &fespace)
{
   Mesh *mesh = fespace.GetMesh();

   // exact solution and RHS for the problem -\Delta u = 1
   FunctionCoefficient exsol(exact_sln);
   FunctionCoefficient rhs(exact_rhs);

   // set up Dirichlet BC on the boundary
   Array<int> ess_attr(mesh->bdr_attributes.Max());
   ess_attr = 1;

   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   GridFunction x(&fespace);
   x = 0.0;
   x.ProjectBdrCoefficient(exsol, ess_attr);

   // assemble the linear form
   LinearForm lf(&fespace);
   lf.AddDomainIntegrator(new DomainLFIntegrator(rhs));
   lf.Assemble();

   // assemble the bilinear form.
   BilinearForm bf(&fespace);
   bf.AddDomainIntegrator(new DiffusionIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   // solve
   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 0, 500, 1e-30, 0.0);

   bf.RecoverFEMSolution(X, lf, x);

   // compute L2 error from the exact solution
   const real_t error = x.ComputeL2Error(exsol);
   REQUIRE(error == MFEM_Approx(0.0));

   // visualize
#ifdef MFEM_UNIT_DEBUG_VISUALIZE
   const char vishost[] = "localhost";
   const int  visport   = 19916;
   std::unique_ptr<GridFunction> vis_x = x.ProlongToMaxOrder();
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "solution\n" << *mesh << *vis_x;
#endif
}

// Quadratic exact solution for vector-valued spaces
void exact_sln_vec(const Vector &x, Vector &f)
{
   if (f.Size() == 3)
   {
      f(0) = x(1)*x(2);
      f(1) = x(0)*x(2);
      f(2) = x(0)*x(1);
   }
   else
   {
      f(0) = x(0)*x(1);
      f(1) = x(0)*x(1);
   }
}

static void TestSolveVec(FiniteElementSpace &fespace)
{
   Mesh *mesh = fespace.GetMesh();
   const int sdim = mesh->SpaceDimension();

   // Exact solution and RHS for the mass-matrix problem E = f
   VectorFunctionCoefficient exsol(sdim, exact_sln_vec);

   // No boundary conditions
   Array<int> ess_attr(mesh->bdr_attributes.Max());
   ess_attr = 0;

   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   GridFunction x(&fespace);
   x = 0.0;
   if (x.FESpace()->GetTypicalBE()->GetRangeDim() == 0)
   {
      x.ProjectBdrCoefficientNormal(exsol, ess_attr);
   }
   else
   {
      x.ProjectBdrCoefficientTangent(exsol, ess_attr);
   }

   // Assemble the linear form
   LinearForm lf(&fespace);
   lf.AddDomainIntegrator(new VectorFEDomainLFIntegrator(exsol));
   lf.Assemble();

   // Assemble the bilinear form
   BilinearForm bf(&fespace);
   bf.AddDomainIntegrator(new VectorFEMassIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   // Solve
   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 0, 500, 1e-30, 0.0);

   bf.RecoverFEMSolution(X, lf, x);

   // Compute L2 error from the exact solution
   const real_t error = x.ComputeL2Error(exsol);

   REQUIRE(error == MFEM_Approx(0.0));
}

#ifdef MFEM_USE_MPI
static void TestSolvePar(ParFiniteElementSpace &pfes)
{
   ParMesh *pmesh = pfes.GetParMesh();

   // exact solution and RHS for the problem -\Delta u = 1
   FunctionCoefficient exsol(exact_sln);
   FunctionCoefficient rhs(exact_rhs);

   // set up Dirichlet BC on the boundary
   Array<int> ess_attr(pmesh->bdr_attributes.Max());
   ess_attr = 1;

   Array<int> ess_tdof_list;
   pfes.GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   ParGridFunction x(&pfes);
   x = 0.0;
   x.ProjectBdrCoefficient(exsol, ess_attr);

   // assemble the linear form
   ParLinearForm lf(&pfes);
   lf.AddDomainIntegrator(new DomainLFIntegrator(rhs));
   lf.Assemble();

   // assemble the bilinear form.
   ParBilinearForm bf(&pfes);
   bf.AddDomainIntegrator(new DiffusionIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   // solve
   HypreBoomerAMG prec;
   CGSolver cg(pfes.GetComm());
   cg.SetRelTol(1e-30);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(prec);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   bf.RecoverFEMSolution(X, lf, x);

   // compute L2 error from the exact solution
   const real_t error = x.ComputeL2Error(exsol);
   REQUIRE(error == MFEM_Approx(0.0));
}

void TestSolveSerial1(const Mesh & mesh, GridFunction & x)
{
   FiniteElementSpace *fespace = x.FESpace();

   Array<int> ess_attr(mesh.bdr_attributes.Max());
   ess_attr = 1;  // Dirichlet BC everywhere

   Array<int> ess_tdof_list;
   fespace->GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   // assemble the linear form
   LinearForm lf(fespace);
   ConstantCoefficient one(1.0);
   lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   lf.Assemble();

   // assemble the bilinear form.
   BilinearForm bf(fespace);
   bf.SetDiagonalPolicy(Operator::DIAG_ONE);

   bf.AddDomainIntegrator(new DiffusionIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 10, 500, 1e-30, 0.0);
   std::cout << std::flush;

   bf.RecoverFEMSolution(X, lf, x);
}

void TestSolveParallel1(ParMesh &pmesh, ParGridFunction &x)
{
   ParFiniteElementSpace *pfes = x.ParFESpace();

   Array<int> ess_attr(pmesh.bdr_attributes.Max());
   ess_attr = 1;  // Dirichlet BC

   Array<int> ess_tdof_list;
   pfes->GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   // assemble the linear form
   ParLinearForm lf(pfes);
   ConstantCoefficient one(1.0);
   lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   lf.Assemble();

   // assemble the bilinear form.
   ParBilinearForm bf(pfes);
   bf.AddDomainIntegrator(new DiffusionIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   HypreBoomerAMG prec;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-30);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(10);
   cg.SetPreconditioner(prec);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   bf.RecoverFEMSolution(X, lf, x);
}

GridFunction *TestRandomPRefinement_serial(Mesh & mesh)
{
   // standard H1 space with order 1 elements
   auto *fec = new H1_FECollection(1, mesh.Dimension());
   auto *fespace = new FiniteElementSpace(&mesh, fec);

   for (int i=0; i<mesh.GetNE(); ++i)
   {
      const int p = mesh.GetAttribute(i);
      if (p > 1) { fespace->SetElementOrder(i, p); }
   }

   fespace->Update(false);

   auto *sol = new GridFunction(fespace);
   sol->MakeOwner(fec);
   *sol = 0.0;  // Essential DOF value
   TestSolveSerial1(mesh, *sol);
   return sol;
}

ParGridFunction *TestRandomPRefinement_parallel(Mesh & mesh)
{
   // standard H1 space with order 1 elements

   auto *pmsh = new ParMesh(MPI_COMM_WORLD, mesh);
   auto *pfec = new H1_FECollection(1, mesh.Dimension());
   auto *pfes = new ParFiniteElementSpace(pmsh, pfec);

   for (int i=0; i<pmsh->GetNE(); ++i)
   {
      const int p = pmsh->GetAttribute(i);
      if (p > 1) { pfes->SetElementOrder(i, p); }
   }

   pfes->Update(false);

   auto *sol = new ParGridFunction(pfes);
   sol->MakeOwner(pfec);
   *sol = 0.0;  // Essential DOF value
   TestSolveParallel1(*pmsh, *sol);
   return sol;
}

// This function is based on the assumption that each element has attribute
// equal to its index in the serial mesh. This assumption enables easily
// identifying serial and parallel elements, for element-wise comparisons.
real_t ErrorSerialParallel(const GridFunction & xser,
                           const ParGridFunction & xpar)
{
   const FiniteElementSpace *fespace = xser.FESpace();
   const ParFiniteElementSpace *pfespace = xpar.ParFESpace();

   Mesh *mesh = fespace->GetMesh();
   ParMesh *pmesh = pfespace->GetParMesh();

   // keep non-const as a workaround for MPI warning in MPI_Allgather below
   int npe = pmesh->GetNE();

   int numprocs, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   Array<int> allnpe(numprocs);
   MPI_Allgather(&npe, 1, MPI_INT, allnpe.GetData(), 1, MPI_INT, MPI_COMM_WORLD);

   int eos = 0;
   for (int i=0; i<rank; ++i)
   {
      eos += allnpe[i];
   }

   bool elemsMatch = true;
   real_t error = 0.0;

   xser.HostRead();
   xpar.HostRead();

   // Loop over only the local elements in the parallel mesh.
   for (int e=0; e<pmesh->GetNE(); ++e)
   {
      if (pmesh->GetAttribute(e) != mesh->GetAttribute(eos + e))
      {
         elemsMatch = false;
      }

      Array<int> sdofs, pdofs;

      fespace->GetElementDofs(eos + e, sdofs);
      pfespace->GetElementDofs(e, pdofs);

      if (sdofs.Size() != pdofs.Size())
      {
         elemsMatch = false;
      }

      for (int i=0; i<sdofs.Size(); ++i)
      {
         const real_t d = xser[sdofs[i]] - xpar[pdofs[i]];
         error += d * d;
      }
   }

   REQUIRE(elemsMatch);

   MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, MPI_COMM_WORLD);
   return error;
}

real_t CheckH1Continuity(ParGridFunction & x)
{
   x.ExchangeFaceNbrData();

   const ParFiniteElementSpace *fes = x.ParFESpace();
   ParMesh *mesh = fes->GetParMesh();

   const int dim = mesh->Dimension();

   // Following the example of KellyErrorEstimator::ComputeEstimates(),
   // we loop over interior faces and then shared faces.

   // Compute error contribution from local interior faces
   real_t errorMax = 0.0;
   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      if (mesh->FaceIsInterior(f))
      {
         int Inf1, Inf2, NCFace;
         mesh->GetFaceInfos(f, &Inf1, &Inf2, &NCFace);

         auto FT = mesh->GetFaceElementTransformations(f);

         const int faceOrder = dim == 3 ? fes->GetFaceOrder(f) :
                               fes->GetEdgeOrder(f);
         auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * faceOrder);
         const auto nip = int_rule.GetNPoints();

         // Convention
         // * Conforming face: Face side with smaller element id handles
         // the integration
         // * Non-conforming face: The slave handles the integration.
         // See FaceInfo documentation for details.
         bool isNCSlave    = FT->Elem2No >= 0 && NCFace >= 0;
         bool isConforming = FT->Elem2No >= 0 && NCFace == -1;
         if ((FT->Elem1No < FT->Elem2No && isConforming) || isNCSlave)
         {
            for (int i = 0; i < nip; i++)
            {
               const auto &fip = int_rule.IntPoint(i);
               IntegrationPoint ip;

               FT->Loc1.Transform(fip, ip);
               const real_t v1 = x.GetValue(FT->Elem1No, ip);

               FT->Loc2.Transform(fip, ip);
               const real_t v2 = x.GetValue(FT->Elem2No, ip);

               const real_t err_i = std::abs(v1 - v2);
               errorMax = std::max(errorMax, err_i);
            }
         }
      }
   }

   // Compute error contribution from shared interior faces
   for (int sf = 0; sf < mesh->GetNSharedFaces(); sf++)
   {
      const int f = mesh->GetSharedFace(sf);
      const bool trueInterior = mesh->FaceIsTrueInterior(f);
      if (!trueInterior) { continue; }

      auto FT = mesh->GetSharedFaceTransformations(sf, true);
      const int faceOrder = dim == 3 ? fes->GetFaceOrder(f) : fes->GetEdgeOrder(f);
      const auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * faceOrder);
      const auto nip = int_rule.GetNPoints();

      for (int i = 0; i < nip; i++)
      {
         const auto &fip = int_rule.IntPoint(i);
         IntegrationPoint ip;

         FT->Loc1.Transform(fip, ip);
         const real_t v1 = x.GetValue(FT->Elem1No, ip);

         FT->Loc2.Transform(fip, ip);
         const real_t v2 = x.GetValue(FT->Elem2No, ip);

         const real_t err_i = std::abs(v1 - v2);
         errorMax = std::max(errorMax, err_i);
      }
   }

   return errorMax;
}

static void TestRandomPRefinement(Mesh & mesh)
{
   for (int i=0; i<mesh.GetNE(); ++i)
   {
      mesh.SetAttribute(i, 1 + (i % 3));   // Order is 1, 2, or 3
   }
   mesh.EnsureNCMesh();

   GridFunction *solSerial = TestRandomPRefinement_serial(mesh);
   ParGridFunction *solParallel = TestRandomPRefinement_parallel(mesh);
   const real_t error = ErrorSerialParallel(*solSerial, *solParallel);
   REQUIRE(error == MFEM_Approx(0.0));

   // Check H1 continuity for the parallel solution.
   const real_t discontinuity = CheckH1Continuity(*solParallel);
   REQUIRE(discontinuity == MFEM_Approx(0.0));

   delete solParallel->ParFESpace()->GetParMesh();
   delete solSerial;
   delete solParallel;
}

static void TestSolveParVec(ParFiniteElementSpace &fespace)
{
   ParMesh *pmesh = fespace.GetParMesh();
   const int sdim = pmesh->SpaceDimension();

   // Exact solution and RHS for the mass-matrix problem E = f
   VectorFunctionCoefficient exsol(sdim, exact_sln_vec);

   // No boundary conditions
   Array<int> ess_attr(pmesh->bdr_attributes.Max());
   ess_attr = 0;

   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   ParGridFunction x(&fespace);
   x = 0.0;
   if (x.FESpace()->GetTypicalBE()->GetRangeDim() == 0)
   {
      x.ProjectBdrCoefficientNormal(exsol, ess_attr);
   }
   else
   {
      x.ProjectBdrCoefficientTangent(exsol, ess_attr);
   }

   // Assemble the linear form
   ParLinearForm lf(&fespace);
   lf.AddDomainIntegrator(new VectorFEDomainLFIntegrator(exsol));
   lf.Assemble();

   // Assemble the bilinear form
   ParBilinearForm bf(&fespace);
   bf.AddDomainIntegrator(new VectorFEMassIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   // Solve
   HypreBoomerAMG prec;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-30);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(prec);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   bf.RecoverFEMSolution(X, lf, x);

   // Compute L2 error from the exact solution
   const real_t error = x.ComputeL2Error(exsol);
   REQUIRE(error == MFEM_Approx(0.0));
}

TEST_CASE("Parallel multi-segment derefinement and order propagation",
          "[FiniteElementSpace][NCMesh][Parallel]")
{
   int nranks = 1;
   MPI_Comm_size(MPI_COMM_WORLD, &nranks);

   SECTION("Multi-segment derefinement send/recv for vector H1")
   {
      for (int ordering : {Ordering::byNODES, Ordering::byVDIM})
      {
         Mesh mesh = Mesh::MakeCartesian3D(6, 1, 1, Element::HEXAHEDRON);
         mesh.EnsureNCMesh();
         ParMesh pmesh(MPI_COMM_WORLD, mesh);
         pmesh.UniformRefinement();

         Array<int> part(pmesh.GetNE());
         Vector c(3);
         for (int e = 0; e < pmesh.GetNE(); e++)
         {
            pmesh.GetElementCenter(e, c);
            const int b = std::min(5, std::max(0, (int)std::floor(6.0 * c(0))));
            const int sub =
               (std::min(1, std::max(0, ((int)std::floor(12.0 * c(0))) % 2))) +
               (c(1) > 0.5 ? 2 : 0) + (c(2) > 0.5 ? 4 : 0);
            const int owner = b % nranks;
            part[e] = owner + (sub % (nranks - owner));
         }
         pmesh.Rebalance(part);

         H1_FECollection fec(1, 3);
         ParFiniteElementSpace fes(&pmesh, &fec, 2, ordering);
         VectorFunctionCoefficient vf(2, [](const Vector &p, Vector &v)
         {
            v(0) = 1.0 + 2.0 * p(0) - 0.5 * p(1) + 0.25 * p(2);
            v(1) = -0.3 + 0.7 * p(0) + 1.5 * p(1) - 0.8 * p(2);
         });
         ParGridFunction x(&fes);
         x.ProjectCoefficient(vf);
         REQUIRE(x.ComputeL2Error(vf) == MFEM_Approx(0.0));

         Vector zero(pmesh.GetNE());
         zero = 0.0;
         REQUIRE(pmesh.DerefineByError(zero, 1.0));
         fes.Update();
         x.Update();
         REQUIRE(pmesh.GetGlobalNE() == 6);
         REQUIRE(x.ComputeL2Error(vf) == MFEM_Approx(0.0));
      }
   }

   SECTION("Repeated variable-order propagation and ghost variant unpacking")
   {
      for (int iter = 0; iter < 3; ++iter)
      {
         Mesh serial = Mesh::MakeCartesian3D(3, 3, 2, Element::HEXAHEDRON);
         serial.EnsureNCMesh();
         Array<int> ref;
         Vector c(3);
         for (int e = 0; e < serial.GetNE(); ++e)
         {
            serial.GetElementCenter(e, c);
            if (c(0) < 0.4 && c(1) < 0.4) { ref.Append(e); }
         }
         serial.GeneralRefinement(ref);

         ParMesh pmesh(MPI_COMM_WORLD, serial);
         H1_FECollection fec(1, 3);
         ParFiniteElementSpace fes(&pmesh, &fec);
         for (int e = 0; e < pmesh.GetNE(); ++e)
         {
            pmesh.GetElementCenter(e, c);
            const int ord = (c(0) > 0.6) ? 3 : ((c(1) > 0.5) ? 2 : 1);
            fes.SetElementOrder(e, ord);
         }
         fes.Update(false);

         FunctionCoefficient lin([](const Vector &p)
         {
            return 0.5 + 1.3 * p(0) - 0.7 * p(1) + 0.4 * p(2);
         });
         ParGridFunction x(&fes);
         x.ProjectCoefficient(lin);
         Vector tx(fes.GetTrueVSize());
         x.ParallelProject(tx);
         x.Distribute(tx);
         REQUIRE(x.ComputeL2Error(lin) < 1e-12);
      }
   }
}

#endif  // MFEM_USE_MPI

}
