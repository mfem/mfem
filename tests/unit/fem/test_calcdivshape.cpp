// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
 * Linear test function whose divergence is equal to 1.
*/
void test_div_func(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v.SetSize(dim);
   v[0] = (double)(dim + 1) * x[0];
   v[1] = -2.0 * x[1];
   if (dim == 3)
   {
      v[2] = -x[2];
   }
}

/**
 * Tests fe->CalcDivShape() over a grid of IntegrationPoints
 * of resolution res. Also tests at integration points
 * that are outside the element.
 */
void TestCalcDivShape(FiniteElement* fe, ElementTransformation * T, int res)
{
   int  dof = fe->GetDof();
   int  dim = fe->GetDim();

   Vector dofs(dof);
   Vector weights(dof);

   VectorFunctionCoefficient vCoef(dim, test_div_func);

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
      // from CalcDivShape() sum to one
      for (int j=0; j < ipArr.Size(); ++j)
      {
         IntegrationPoint& ip = ipArr[j];
         fe->CalcDivShape(ip, weights);

         REQUIRE( weights * dofs == Approx(1.) );
      }
   }
}

TEST_CASE("CalcDivShape for several RT FiniteElement instances",
          "[RT_TriangleElement]"
          "[RT_QuadrilateralElement]"
          "[RT_TetrahedronElement]"
          "[RT_WedgeElement]"
          "[RT_HexahedronElement]")
{
   int maxOrder = 5;
   int resolution = 10;

   SECTION("RT_TriangleElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TRIANGLE, T);

      for (int order = 1; order <= maxOrder; ++order)
      {
         std::cout << "Testing RT_TriangleElement::CalcDivShape() "
                   << "for order " << order << std::endl;
         RT_TriangleElement fe(order - 1);
         TestCalcDivShape(&fe, &T, resolution);
      }
   }

   SECTION("RT_QuadrilateralElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::QUADRILATERAL, T);

      for (int order = 1; order <= maxOrder; ++order)
      {
         std::cout << "Testing RT_QuadrilateralElement::CalcDivShape() "
                   << "for order " << order << std::endl;
         RT_QuadrilateralElement fe(order - 1);
         TestCalcDivShape(&fe, &T, resolution);
      }
   }

   SECTION("RT_TetrahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TETRAHEDRON, T);

      for (int order = 1; order <= maxOrder; ++order)
      {
         std::cout << "Testing RT_TetrahedronElement::CalcDivShape() "
                   << "for order " << order << std::endl;
         RT_TetrahedronElement fe(order - 1);
         TestCalcDivShape(&fe, &T, resolution);
      }
   }

   SECTION("RT_WedgeElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::WEDGE, T);

      for (int order = 1; order <= maxOrder; ++order)
      {
         std::cout << "Testing RT_WedgeElement::CalcDivShape() "
                   << "for order " << order << std::endl;
         RT_WedgeElement fe(order - 1);
         TestCalcDivShape(&fe, &T, resolution);
      }
   }

   SECTION("RT_HexahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::HEXAHEDRON, T);

      for (int order = 1; order <= maxOrder; ++order)
      {
         std::cout << "Testing RT_HexahedronElement::CalcDivShape() "
                   << "for order " << order << std::endl;
         RT_HexahedronElement fe(order - 1);
         TestCalcDivShape(&fe, &T, resolution);
      }
   }
}
