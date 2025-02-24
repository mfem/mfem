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

using namespace mfem;

namespace pos_bases
{

Element * GetElement(Geometry::Type type)
{
   Element *el = NULL;
   switch (type)
   {
      case Geometry::POINT:
         el = new Point;
         break;
      case Geometry::SEGMENT:
         el = new Segment;
         break;
      case Geometry::TRIANGLE:
         el = new Triangle;
         break;
      case Geometry::SQUARE:
         el = new Quadrilateral;
         break;
      case Geometry::TETRAHEDRON:
         el = new Tetrahedron;
         break;
      case Geometry::CUBE:
         el = new Hexahedron;
         break;
      case Geometry::PRISM:
         el = new Wedge;
         break;
      case Geometry::PYRAMID:
         el = new Pyramid;
         break;
      default:
         break;
   }
   return el;
}

// Build a mesh containing a single element
Mesh MakeElementMesh(Geometry::Type type, real_t * vertices)
{
   Element *elem = GetElement(type);

   int nvert = elem->GetNVertices();

   Array<int> el_inds(nvert), el_attr(1);
   for (int i=0; i<nvert; i++) { el_inds[i] = i; }
   el_attr[0] = 1;

   int dim = 0, sdim = -1;
   Geometry::Type bdr_type = Geometry::INVALID;
   if (type == Geometry::SEGMENT)
   {
      dim = 1;
      sdim = 1;
   }
   else if (type >= Geometry::TRIANGLE && type <= Geometry::SQUARE)
   {
      dim = 2;
      sdim = 2;
   }
   else if (type >= Geometry::TETRAHEDRON)
   {
      dim = 3;
      sdim = 3;
   }

   Mesh mesh(vertices, nvert, &el_inds[0], type, &el_attr[0], 1, NULL,
             bdr_type, NULL, 0, dim, sdim);

   mesh.Finalize();

   delete elem;

   return mesh;
}

static real_t ref_seg_vert[] = {0., 1.};
static real_t *equ_seg_vert = ref_seg_vert;

static real_t ref_tri_vert[] = {0.,0., 1.,0., 0.,1.};
static real_t equ_tri_vert[] = {0.,0., 1.,0., 0.5,0.8660254037844386};

static real_t ref_sqr_vert[] = {0.,0., 1.,0., 1.,1., 0.,1.};
static real_t *equ_sqr_vert = ref_sqr_vert;

static real_t ref_tet_vert[] = {0.,0.,0., 1.,0.,0., 0.,1.,0., 0.,0.,1.};
static real_t equ_tet_vert[] = {0.,0.,0., 1.,0.,0., 0.5,0.8660254037844386,0.,
                                0.5,0.2886751345948129,0.816496580927726
                               };

static real_t ref_cub_vert[] = {0.,0.,0., 1.,0.,0., 1.,1.,0., 0.,1.,0.,
                                0.,0.,1., 1.,0.,1., 1.,1.,1., 0.,1.,1.
                               };
static real_t *equ_cub_vert = ref_cub_vert;

static real_t ref_pri_vert[] = {0.,0.,0., 1.,0.,0., 0.,1.,0.,
                                0.,0.,1., 1.,0.,1., 0.,1.,1.
                               };
static real_t equ_pri_vert[] = {0.,0.,0., 1.,0.,0., 0.5,0.8660254037844386,0.,
                                0.,0.,1., 1.,0.,1., 0.5,0.8660254037844386,1.
                               };

static real_t ref_pyr_vert[] = {0.,0.,0., 1.,0.,0., 1.,1.,0., 0.,1.,0., 0.,0.,1.};
static real_t equ_pyr_vert[] = {0.,0.,0., 1.,0.,0., 1.,1.,0., 0.,1.,0.,
                                0.5,0.5,0.7071067811865475
                               };

TEST_CASE("Positive Bases",
          "[H1Pos_Segment]"
          "[H1Pos_Triangle]"
          "[H1Pos_Quadrilateral]"
          "[H1Pos_TetrahedronElement]"
          "[H1Pos_HexahedronElement]"
          "[H1Pos_WedgeElement]"
          "[H1Pos_PyramidElement]"
          "[L2Pos_Segment]"
          "[L2Pos_Triangle]"
          "[L2Pos_Quadrilateral]"
          "[L2Pos_TetrahedronElement]"
          "[L2Pos_HexahedronElement]"
          "[L2Pos_WedgeElement]"
          "[L2Pos_PyramidElement]")
{
   auto geom = GENERATE(Geometry::SEGMENT,
                        Geometry::TRIANGLE, Geometry::SQUARE,
                        Geometry::TETRAHEDRON, Geometry::CUBE,
                        Geometry::PRISM, Geometry::PYRAMID);
   auto ref = GENERATE(true, false);
   auto p = GENERATE(1, 2, 3, 4);

   CAPTURE(geom);
   CAPTURE(ref);
   CAPTURE(p);

   real_t *geom_vert = NULL;
   switch (geom)
   {
      case Geometry::SEGMENT:
         geom_vert = ref ? ref_seg_vert : equ_seg_vert;
         break;
      case Geometry::TRIANGLE:
         geom_vert = ref ? ref_tri_vert : equ_tri_vert;
         break;
      case Geometry::SQUARE:
         geom_vert = ref ? ref_sqr_vert : equ_sqr_vert;
         break;
      case Geometry::TETRAHEDRON:
         geom_vert = ref ? ref_tet_vert : equ_tet_vert;
         break;
      case Geometry::CUBE:
         geom_vert = ref ? ref_cub_vert : equ_cub_vert;
         break;
      case Geometry::PRISM:
         geom_vert = ref ? ref_pri_vert : equ_pri_vert;
         break;
      case Geometry::PYRAMID:
         geom_vert = ref ? ref_pyr_vert : equ_pyr_vert;
         break;
      default:
         break;
   };

   Mesh mesh = MakeElementMesh(geom, geom_vert);
   int dim = mesh.Dimension();

   H1Pos_FECollection h1_fec(p, dim);
   L2_FECollection    l2_fec(p-1, dim, BasisType::Positive);

   FiniteElementSpace h1_fes(&mesh, &h1_fec);
   FiniteElementSpace l2_fes(&mesh, &l2_fec);

   int h1_dof = h1_fes.GetFE(0)->GetDof();
   int l2_dof = l2_fes.GetFE(0)->GetDof();

   Vector h1_ones(h1_dof); h1_ones = 1.0;
   Vector l2_ones(l2_dof); l2_ones = 1.0;
   Vector h1_shape(h1_dof);
   Vector l2_shape(l2_dof);
   Vector h1_ds(dim);
   Vector l2_ds(dim);
   DenseMatrix h1_dshape(h1_dof, dim);
   DenseMatrix l2_dshape(l2_dof, dim);

   // Get a uniform grid of integration points independent of the basis
   // function order.
   RefinedGeometry* gref = GlobGeometryRefiner.Refine( geom, 2);
   const IntegrationRule& intRule = gref->RefPts;

   int npoints = intRule.GetNPoints();
   for (int i=0; i < npoints; ++i)
   {
      // Get the current integration point from intRule
      IntegrationPoint pt = intRule.IntPoint(i);

      // Evaluate the basis functions
      h1_fes.GetFE(0)->CalcShape(pt, h1_shape);
      l2_fes.GetFE(0)->CalcShape(pt, l2_shape);

      // Verify that the basis functions sum to one
      REQUIRE(h1_shape.Sum() == MFEM_Approx(1.));
      REQUIRE(l2_shape.Sum() == MFEM_Approx(1.));

      // Verify that the basis functions are non-negative and sum to one
      REQUIRE(h1_shape.Norml1() == MFEM_Approx(1.));
      REQUIRE(l2_shape.Norml1() == MFEM_Approx(1.));

      // Evaluate the gradients of the basis functions
      h1_fes.GetFE(0)->CalcDShape(pt, h1_dshape);
      l2_fes.GetFE(0)->CalcDShape(pt, l2_dshape);

      // Sum the columns of h1_dshape
      h1_dshape.MultTranspose(h1_ones, h1_ds);
      l2_dshape.MultTranspose(l2_ones, l2_ds);

      // Verify that the basis function derivatives sum to zero
      REQUIRE(h1_ds.Norml2() == MFEM_Approx(0.));
      REQUIRE(l2_ds.Norml2() == MFEM_Approx(0.));
   }
}

} // namespace pos_basis
