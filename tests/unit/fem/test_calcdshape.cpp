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
 * Linear test function whose gradient is equal to 1 in 1D, (1,1) in 2D.
 * and (1,1,1) in 3D.
*/
double test_grad_func(const Vector &x)
{
   int dim = x.Size();
   double v = x[0];
   if (dim > 1)
   {
      v += x[1];
   }
   if (dim > 2)
   {
      v += x[2];
   }
   return v;
}

/**
 * Tests fe->CalcDShape() over a grid of IntegrationPoints
 * of resolution res. Also tests at integration points
 * that are outside the element.
 */
void TestCalcDShape(FiniteElement* fe, ElementTransformation * T, int res)
{
   int  dof = fe->GetDof();
   int  dim = fe->GetDim();

   Vector dofs(dof);
   Vector v(dim);
   DenseMatrix weights( dof, dim );

   FunctionCoefficient vCoef(test_grad_func);

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
      // from CalcDShape() sum to one
      for (int j=0; j < ipArr.Size(); ++j)
      {
         IntegrationPoint& ip = ipArr[j];
         fe->CalcDShape(ip, weights);

         weights.MultTranspose(dofs, v);
         REQUIRE( v[0] == Approx(1.) );
         if (dim > 1)
         {
            REQUIRE( v[1] == Approx(1.) );
         }
         if (dim > 2)
         {
            REQUIRE( v[2] == Approx(1.) );
         }
      }
   }
}

TEST_CASE("CalcDShape H1",
          "[H1_SegmentElement]"
          "[H1_TriangleElement]"
          "[H1_QuadrilateralElement]"
          "[H1_TetrahedronElement]"
          "[H1_WedgeElement]"
          "[H1_HexahedronElement]")
{
   const int maxOrder = 5;
   const int resolution = 10;
   auto order = GENERATE_COPY(range(1, maxOrder + 1));

   CAPTURE(order);

   SECTION("H1_SegmentElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::SEGMENT, T);

      H1_SegmentElement fe(order);
      TestCalcDShape(&fe, &T, resolution);
   }

   SECTION("H1_TriangleElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TRIANGLE, T);

      H1_TriangleElement fe(order);
      TestCalcDShape(&fe, &T, resolution);
   }

   SECTION("H1_QuadrilateralElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::QUADRILATERAL, T);

      H1_QuadrilateralElement fe(order);
      TestCalcDShape(&fe, &T, resolution);
   }

   SECTION("H1_TetrahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TETRAHEDRON, T);

      H1_TetrahedronElement fe(order);
      TestCalcDShape(&fe, &T, resolution);
   }

   SECTION("H1_WedgeElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::WEDGE, T);

      H1_WedgeElement fe(order);
      TestCalcDShape(&fe, &T, resolution);
   }

   SECTION("H1_HexahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::HEXAHEDRON, T);

      H1_HexahedronElement fe(order);
      TestCalcDShape(&fe, &T, resolution);
   }
}
