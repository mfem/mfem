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

   // Get a uniform grid of integration points
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

TEST_CASE("CalcCurlShape ND",
          "[ND_TriangleElement]"
          "[ND_QuadrilateralElement]"
          "[ND_TetrahedronElement]"
          "[ND_WedgeElement]"
          "[ND_HexahedronElement]")
{
   const int maxOrder = 5;
   const int resolution = 10;
   auto order = GENERATE_COPY(range(1, maxOrder + 1));

   CAPTURE(order);

   SECTION("ND_TriangleElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TRIANGLE, T);

      ND_TriangleElement fe(order);
      TestCalcCurlShape(&fe, &T, resolution);
   }

   SECTION("ND_QuadrilateralElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::QUADRILATERAL, T);

      ND_QuadrilateralElement fe(order);
      TestCalcCurlShape(&fe, &T, resolution);
   }

   SECTION("ND_TetrahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TETRAHEDRON, T);

      ND_TetrahedronElement fe(order);
      TestCalcCurlShape(&fe, &T, resolution);
   }

   SECTION("ND_WedgeElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::WEDGE, T);

      ND_WedgeElement fe(order);
      TestCalcCurlShape(&fe, &T, resolution);
   }

   SECTION("ND_HexahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::HEXAHEDRON, T);

      ND_HexahedronElement fe(order);
      TestCalcCurlShape(&fe, &T, resolution);
   }
}
