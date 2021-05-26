// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced at
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
 *
 * Note: this is defined in test_calcshape.cpp
 */
void GetRelatedIntegrationPoints(const IntegrationPoint& ip, int dim,
                                 Array<IntegrationPoint>& arr);

/**
 * Utility function to setup IsoparametricTransformations for reference
 * elements of various types.
 *
 * Note: this is defined in test_calcvshape.cpp
 */
void GetReferenceTransformation(const Element::Type ElemType,
                                IsoparametricTransformation & T);

/**
 * Linear test function whose curl is equal to 1 in 2D and (1,1,1) in 3D.
*/
void test_curl_func(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v.SetSize(dim);
   v[0] = 4.0 * x[1];
   v[1] = 5.0 * x[0];
   if (dim == 3)
   {
      v[0] += 3.0 * x[2];
      v[1] += x[2];
      v[2] = 2.0 * (x[0] + x[1]);
   }
}

/**
 * Tests fe->CalcCurlShape() over a grid of IntegrationPoints
 * of resolution res. Also tests at integration points
 * that are outside the element.
 */
void TestCalcCurlShape(FiniteElement* fe, ElementTransformation * T, int res)
{
   int  dof = fe->GetDof();
   int  dim = fe->GetDim();
   int cdim = 2 * dim - 3;

   Vector dofs(dof);
   Vector v(cdim);
   DenseMatrix weights( dof, cdim );

   VectorFunctionCoefficient vCoef(dim, test_curl_func);

   fe->Project(vCoef, *T, dofs);

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
      // from CalcCurlShape() sum to one
      for (int j=0; j < ipArr.Size(); ++j)
      {
         IntegrationPoint& ip = ipArr[j];
         fe->CalcCurlShape(ip, weights);

         weights.MultTranspose(dofs, v);
         REQUIRE( v[0] == Approx(1.) );
         if (dim == 3)
         {
            REQUIRE( v[1] == Approx(1.) );
            REQUIRE( v[2] == Approx(1.) );
         }
      }
   }
}

TEST_CASE("CalcCurlShape for several ND FiniteElement instances",
          "[ND_TriangleElement]"
          "[ND_QuadrilateralElement]"
          "[ND_TetrahedronElement]"
          "[ND_WedgeElement]"
          "[ND_HexahedronElement]")
{
   int maxOrder = 5;
   int resolution = 10;

   SECTION("ND_TriangleElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TRIANGLE, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing ND_TriangleElement::CalcCurlShape() "
                   << "for order " << order << std::endl;
         ND_TriangleElement fe(order);
         TestCalcCurlShape(&fe, &T, resolution);
      }
   }

   SECTION("ND_QuadrilateralElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::QUADRILATERAL, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing ND_QuadrilateralElement::CalcCurlShape() "
                   << "for order " << order << std::endl;
         ND_QuadrilateralElement fe(order);
         TestCalcCurlShape(&fe, &T, resolution);
      }
   }

   SECTION("ND_TetrahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TETRAHEDRON, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing ND_TetrahedronElement::CalcCurlShape() "
                   << "for order " << order << std::endl;
         ND_TetrahedronElement fe(order);
         TestCalcCurlShape(&fe, &T, resolution);
      }
   }

   SECTION("ND_WedgeElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::WEDGE, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing ND_WedgeElement::CalcCurlShape() "
                   << "for order " << order << std::endl;
         ND_WedgeElement fe(order);
         TestCalcCurlShape(&fe, &T, resolution);
      }
   }

   SECTION("ND_HexahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::HEXAHEDRON, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing ND_HexahedronElement::CalcCurlShape() "
                   << "for order " << order << std::endl;
         ND_HexahedronElement fe(order);
         TestCalcCurlShape(&fe, &T, resolution);
      }
   }
}
