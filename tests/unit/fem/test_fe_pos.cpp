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

FiniteElement * GetH1PosFiniteElement(Geometry::Type type, int order)
{
   FiniteElement *fe = NULL;
   switch (type)
   {
      case Geometry::SEGMENT:
         fe = new H1Pos_SegmentElement(order);
         break;
      case Geometry::TRIANGLE:
         fe = new H1Pos_TriangleElement(order);
         break;
      case Geometry::SQUARE:
         fe = new H1Pos_QuadrilateralElement(order);
         break;
      case Geometry::TETRAHEDRON:
         fe = new H1Pos_TetrahedronElement(order);
         break;
      case Geometry::CUBE:
         fe = new H1Pos_HexahedronElement(order);
         break;
      case Geometry::PRISM:
         fe = new H1Pos_WedgeElement(order);
         break;
      case Geometry::PYRAMID:
         fe = new H1Pos_PyramidElement(order);
         break;
      default:
         break;
   }
   return fe;
}

FiniteElement * GetL2PosFiniteElement(Geometry::Type type, int order)
{
   FiniteElement *fe = NULL;
   switch (type)
   {
      case Geometry::SEGMENT:
         fe = new L2Pos_SegmentElement(order);
         break;
      case Geometry::TRIANGLE:
         fe = new L2Pos_TriangleElement(order);
         break;
      case Geometry::SQUARE:
         fe = new L2Pos_QuadrilateralElement(order);
         break;
      case Geometry::TETRAHEDRON:
         fe = new L2Pos_TetrahedronElement(order);
         break;
      case Geometry::CUBE:
         fe = new L2Pos_HexahedronElement(order);
         break;
      case Geometry::PRISM:
         fe = new L2Pos_WedgeElement(order);
         break;
      case Geometry::PYRAMID:
         fe = new L2Pos_PyramidElement(order);
         break;
      default:
         break;
   }
   return fe;
}

TEST_CASE("Positive H1 Bases",
          "[H1Pos_SegmentElement]"
          "[H1Pos_TriangleElement]"
          "[H1Pos_QuadrilateralElement]"
          "[H1Pos_TetrahedronElement]"
          "[H1Pos_HexahedronElement]"
          "[H1Pos_WedgeElement]"
          "[H1Pos_PyramidElement]")
{
   const int maxOrder = 5;
   const int resolution = 10;

   auto geom = GENERATE(Geometry::SEGMENT,
                        Geometry::TRIANGLE, Geometry::SQUARE,
                        Geometry::TETRAHEDRON, Geometry::CUBE,
                        Geometry::PRISM, Geometry::PYRAMID);
   auto p = GENERATE_COPY(range(1, maxOrder + 1));

   CAPTURE(geom);
   CAPTURE(p);

   SECTION("H1 Basis Summation")
   {
      FiniteElement *fe = GetH1PosFiniteElement(geom, p);

      int dim = fe->GetDim();
      int ndof = fe->GetDof();
      Vector ones(ndof); ones = 1.0;
      Vector zeros(dim);
      Vector shape(ndof);
      DenseMatrix dshape(ndof, dim);

      // Get a uniform grid of integration points
      RefinedGeometry* ref = GlobGeometryRefiner.Refine( fe->GetGeomType(),
                                                         resolution);
      const IntegrationRule& intRule = ref->RefPts;

      int npoints = intRule.GetNPoints();
      for (int i=0; i < npoints; ++i)
      {
         // Get the current integration point from intRule
         IntegrationPoint pt = intRule.IntPoint(i);

         fe->CalcShape(pt, shape);

         // Verify that the basis functions are non-negative
         REQUIRE(shape.Min() >= -2*std::numeric_limits<real_t>::epsilon());

         // Verify that the basis functions sum to one
         REQUIRE(shape * ones == MFEM_Approx(1.0));

         // Verify that the basis functions are non-negative
         REQUIRE(shape.Norml1() == MFEM_Approx(1.0));

         fe->CalcDShape(pt, dshape);

         dshape.MultTranspose(ones, zeros);

         // Verify that the gradients sum to zero
         REQUIRE(zeros.Norml2() == MFEM_Approx(0.0));
      }

      delete fe;
   }
}

TEST_CASE("Positive L2 Bases",
          "[L2Pos_SegmentElement]"
          "[L2Pos_TriangleElement]"
          "[L2Pos_QuadrilateralElement]"
          "[L2Pos_TetrahedronElement]"
          "[L2Pos_HexahedronElement]"
          "[L2Pos_WedgeElement]"
          "[L2Pos_PyramidElement]")
{
   const int maxOrder = 5;
   const int resolution = 10;

   auto geom = GENERATE(Geometry::SEGMENT,
                        Geometry::TRIANGLE, Geometry::SQUARE,
                        Geometry::TETRAHEDRON, Geometry::CUBE,
                        Geometry::PRISM, Geometry::PYRAMID);
   auto p = GENERATE_COPY(range(0, maxOrder + 1));

   CAPTURE(geom);
   CAPTURE(p);

   SECTION("L2 Basis Summation")
   {
      FiniteElement *fe = GetL2PosFiniteElement(geom, p);

      int dim = fe->GetDim();
      int ndof = fe->GetDof();
      Vector ones(ndof); ones = 1.0;
      Vector zeros(dim);
      Vector shape(ndof);
      DenseMatrix dshape(ndof, dim);

      // Get a uniform grid of integration points
      RefinedGeometry* ref = GlobGeometryRefiner.Refine( fe->GetGeomType(),
                                                         resolution);
      const IntegrationRule& intRule = ref->RefPts;

      int npoints = intRule.GetNPoints();
      for (int i=0; i < npoints; ++i)
      {
         // Get the current integration point from intRule
         IntegrationPoint pt = intRule.IntPoint(i);

         fe->CalcShape(pt, shape);

         // Verify that the basis functions are non-negative
         REQUIRE(shape.Min() >= -2*std::numeric_limits<real_t>::epsilon());

         // Verify that the basis functions sum to one
         REQUIRE(shape * ones == MFEM_Approx(1.0));

         // Verify that the basis functions are non-negative
         REQUIRE(shape.Norml1() == MFEM_Approx(1.0));

         fe->CalcDShape(pt, dshape);

         dshape.MultTranspose(ones, zeros);

         // Verify that the gradients sum to zero
         REQUIRE(zeros.Norml2() == MFEM_Approx(0.0));
      }

      delete fe;
   }
}
