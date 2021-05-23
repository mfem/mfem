// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-806117. All Rights
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

TEST_CASE("CalcDShape for several H1 FiniteElement instances",
          "[H1_SegmentElement]"
          "[H1_TriangleElement]"
          "[H1_QuadrilateralElement]"
          "[H1_TetrahedronElement]"
          "[H1_WedgeElement]"
          "[H1_HexahedronElement]")
{
   int maxOrder = 5;
   int resolution = 10;

   SECTION("H1_SegmentElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::SEGMENT, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing H1_SegmentElement::CalcDShape() "
                   << "for order " << order << std::endl;
         H1_SegmentElement fe(order);
         TestCalcDShape(&fe, &T, resolution);
      }
   }

   SECTION("H1_TriangleElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TRIANGLE, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing H1_TriangleElement::CalcDShape() "
                   << "for order " << order << std::endl;
         H1_TriangleElement fe(order);
         TestCalcDShape(&fe, &T, resolution);
      }
   }

   SECTION("H1_QuadrilateralElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::QUADRILATERAL, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing H1_QuadrilateralElement::CalcDShape() "
                   << "for order " << order << std::endl;
         H1_QuadrilateralElement fe(order);
         TestCalcDShape(&fe, &T, resolution);
      }
   }

   SECTION("H1_TetrahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TETRAHEDRON, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing H1_TetrahedronElement::CalcDShape() "
                   << "for order " << order << std::endl;
         H1_TetrahedronElement fe(order);
         TestCalcDShape(&fe, &T, resolution);
      }
   }

   SECTION("H1_WedgeElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::WEDGE, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing H1_WedgeElement::CalcDShape() "
                   << "for order " << order << std::endl;
         H1_WedgeElement fe(order);
         TestCalcDShape(&fe, &T, resolution);
      }
   }

   SECTION("H1_HexahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::HEXAHEDRON, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing H1_HexahedronElement::CalcDShape() "
                   << "for order " << order << std::endl;
         H1_HexahedronElement fe(order);
         TestCalcDShape(&fe, &T, resolution);
      }
   }
}
