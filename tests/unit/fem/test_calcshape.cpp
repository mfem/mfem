// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

#include <iostream>
#include <cmath>

using namespace mfem;

/**
 * Utility function to generate IntegerationPoints, based on param ip
 * that are outside the unit interval.  Results are placed in output
 * parameter arr.
 */
void GetRelatedIntegrationPoints(const IntegrationPoint& ip, int dim,
                                 Array<IntegrationPoint>& arr)
{
   IntegrationPoint pt = ip;
   int idx = 0;

   switch (dim)
   {
      case 1:
         arr.SetSize(3);

         pt.x =   ip.x;    arr[idx++] = pt;
         pt.x =  -ip.x;    arr[idx++] = pt;
         pt.x = 1+ip.x;    arr[idx++] = pt;
         break;
      case 2:
         arr.SetSize(7);

         pt.Set2(  ip.x,   ip.y); arr[idx++] = pt;
         pt.Set2( -ip.x,   ip.y); arr[idx++] = pt;
         pt.Set2(  ip.x,  -ip.y); arr[idx++] = pt;
         pt.Set2( -ip.x,  -ip.y); arr[idx++] = pt;
         pt.Set2(1+ip.x,   ip.y); arr[idx++] = pt;
         pt.Set2(  ip.x, 1+ip.y); arr[idx++] = pt;
         pt.Set2(1+ip.x, 1+ip.y); arr[idx++] = pt;
         break;
      case 3:
         arr.SetSize(15);

         pt.Set3(  ip.x,   ip.y,   ip.z );  arr[idx++] = pt;
         pt.Set3( -ip.x,   ip.y,   ip.z );  arr[idx++] = pt;
         pt.Set3(  ip.x,  -ip.y,   ip.z );  arr[idx++] = pt;
         pt.Set3( -ip.x,  -ip.y,   ip.z );  arr[idx++] = pt;
         pt.Set3(  ip.x,   ip.y,  -ip.z );  arr[idx++] = pt;
         pt.Set3( -ip.x,   ip.y,  -ip.z );  arr[idx++] = pt;
         pt.Set3(  ip.x,  -ip.y,  -ip.z );  arr[idx++] = pt;
         pt.Set3( -ip.x,  -ip.y,  -ip.z );  arr[idx++] = pt;
         pt.Set3(1+ip.x,   ip.y,   ip.z );  arr[idx++] = pt;
         pt.Set3(  ip.x, 1+ip.y,   ip.z );  arr[idx++] = pt;
         pt.Set3(1+ip.x, 1+ip.y,   ip.z );  arr[idx++] = pt;
         pt.Set3(  ip.x,   ip.y, 1+ip.z );  arr[idx++] = pt;
         pt.Set3(1+ip.x,   ip.y, 1+ip.z );  arr[idx++] = pt;
         pt.Set3(  ip.x, 1+ip.y, 1+ip.z );  arr[idx++] = pt;
         pt.Set3(1+ip.x, 1+ip.y, 1+ip.z );  arr[idx++] = pt;
         break;
   }
}

/**
 * Tests fe->CalcShape() over a grid of IntegrationPoints
 * of resolution res. Also tests at integration points
 * that are outside the element.
 */
void TestCalcShape(FiniteElement* fe, int res, double tol=1e-12)
{
   int dim = fe->GetDim();

   Vector weights( fe->GetDof() );

   // Get a uniform grid or integration points
   RefinedGeometry* ref = GlobGeometryRefiner.Refine( fe->GetGeomType(), res);
   const IntegrationRule& intRule = ref->RefPts;

   int npoints = intRule.GetNPoints();
   for (int i=0; i < npoints; ++i)
   {
      // Get the current integration point from intRule
      IntegrationPoint pt = intRule.IntPoint(i);

      // Get several variants of this integration point
      // some of which are inside the element and some are outside
      Array<IntegrationPoint> ipArr;
      GetRelatedIntegrationPoints( pt, dim, ipArr );

      // For each such integration point check that the weights
      // from CalcShape() sum to one
      for (int j=0; j < ipArr.Size(); ++j)
      {
         IntegrationPoint& ip = ipArr[j];
         fe->CalcShape(ip, weights);
         REQUIRE(weights.Sum() == MFEM_Approx(1., tol, tol));
      }
   }
}


TEST_CASE("CalcShape Lagrange",
          "[Lagrange1DFiniteElement]"
          "[BiLinear2DFiniteElement]"
          "[BiQuad2DFiniteElement]"
          "[LagrangeHexFiniteElement]")
{
   const int maxOrder = 5;
   const int resolution = 10;


   SECTION("Lagrange1DFiniteElement")
   {
      auto order = GENERATE_COPY(range(1, maxOrder + 1));
      CAPTURE(order);
      Lagrange1DFiniteElement fe(order);
      TestCalcShape(&fe, resolution);
   }

   SECTION("BiLinear2DFiniteElement")
   {
      BiLinear2DFiniteElement fe;
      TestCalcShape(&fe, resolution);
   }

   SECTION("BiQuad2DFiniteElement")
   {
      BiQuad2DFiniteElement fe;
      TestCalcShape(&fe, resolution);
   }


   SECTION("LagrangeHexFiniteElement")
   {
      // Comments for LagrangeHexFiniteElement state
      // that only degree 2 is functional for this class
      LagrangeHexFiniteElement fe(2);
      TestCalcShape(&fe, resolution);
   }
}

TEST_CASE("CalcShape H1",
          "[H1_SegmentElement]"
          "[H1_TriangleElement]"
          "[H1_QuadrilateralElement]"
          "[H1_TetrahedronElement]"
          "[H1_HexahedronElement]"
          "[H1_WedgeElement]")
{
   const int maxOrder = 5;
   const int resolution = 10;
   auto order = GENERATE_COPY(range(1, maxOrder + 1));

   CAPTURE(order);

   SECTION("H1_SegmentElement")
   {
      H1_SegmentElement fe(order);
      TestCalcShape(&fe, resolution, 2e-11*std::pow(10, order));
   }

   SECTION("H1_TriangleElement")
   {
      H1_TriangleElement fe(order);
      TestCalcShape(&fe, resolution, 2e-11*std::pow(10, order));
   }

   SECTION("H1_QuadrilateralElement")
   {
      H1_QuadrilateralElement fe(order);
      TestCalcShape(&fe, resolution, 2e-11*std::pow(10, order));
   }

   SECTION("H1_TetrahedronElement")
   {
      H1_TetrahedronElement fe(order);
      TestCalcShape(&fe, resolution, 2e-11*std::pow(10, order));
   }

   SECTION("H1_HexahedronElement")
   {
      H1_HexahedronElement fe(order);
      TestCalcShape(&fe, resolution, 2e-11*std::pow(10, order));
   }

   SECTION("H1_WedgeElement")
   {
      H1_WedgeElement fe(order);
      TestCalcShape(&fe, resolution, 2e-11*std::pow(10, order));
   }

}
