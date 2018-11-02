// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mfem.hpp"
#include "catch.hpp"

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
 * of resolution res. Also tests at integration poins
 * that are outside the element.
 */
void TestCalcShape(FiniteElement* fe, int res)
{
   IntegrationPoint ip;
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
         REQUIRE( weights.Sum() == Approx(1.) );
      }
   }
}


TEST_CASE("CalcShape for several Lagrange FiniteElement instances",
          "[Lagrange1DFiniteElement]"
          "[BiLinear2DFiniteElement]"
          "[BiQuad2DFiniteElement]"
          "[LagrangeHexFiniteElement]")
{
   int maxOrder = 5;
   int resolution = 10;

   SECTION("Lagrange1DFiniteElement")
   {
      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing Lagrange1DFiniteElement::CalcShape() "
                   << "for order " << order << std::endl;
         Lagrange1DFiniteElement fe(order);
         TestCalcShape(&fe, resolution);
      }
   }

   SECTION("BiLinear2DFiniteElement")
   {
      std::cout << "Testing BiLinear2DFiniteElement::CalcShape()" << std::endl;
      BiLinear2DFiniteElement fe;
      TestCalcShape(&fe, resolution);
   }

   SECTION("BiQuad2DFiniteElement")
   {
      std::cout << "Testing BiQuad2DFiniteElement::CalcShape()" << std::endl;
      BiQuad2DFiniteElement fe;
      TestCalcShape(&fe, resolution);
   }


   SECTION("LagrangeHexFiniteElement")
   {
      std::cout << "Testing LagrangeHexFiniteElement::CalcShape() "
                << "for order 2" << std::endl;

      // Comments for LagrangeHexFiniteElement state
      // that only degree 2 is functional for this class
      LagrangeHexFiniteElement fe(2);
      TestCalcShape(&fe, resolution);
   }
}

TEST_CASE("CalcShape for several H1 FiniteElement instances",
          "[H1_SegmentElement]"
          "[H1_QuadrilateralElement]"
          "[H1_HexahedronElement]")
{
   int maxOrder = 5;
   int resolution = 10;

   SECTION("H1_SegmentElement")
   {
      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing H1_SegmentElement::CalcShape() "
                   << "for order " << order << std::endl;
         H1_SegmentElement fe(order);
         TestCalcShape(&fe, resolution);
      }
   }

   SECTION("H1_QuadrilateralElement")
   {
      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing H1_QuadrilateralElement::CalcShape() "
                   << "for order " << order << std::endl;
         H1_QuadrilateralElement fe(order);
         TestCalcShape(&fe, resolution);
      }
   }

   SECTION("H1_HexahedronElement")
   {
      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing H1_HexahedronElement::CalcShape() "
                   << "for order " << order << std::endl;
         H1_HexahedronElement fe(order);
         TestCalcShape(&fe, resolution);
      }
   }

}
