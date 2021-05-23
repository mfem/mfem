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
 */
void GetReferenceTransformation(const Element::Type ElemType,
                                IsoparametricTransformation & T)
{
   T.Attribute = 1;
   T.ElementNo = 0;

   switch (ElemType)
   {
      case Element::POINT :
         T.GetPointMat().SetSize(1, 1);
         T.GetPointMat()(0, 0) = 0.0;
         T.SetFE(&PointFE);
         break;
      case Element::SEGMENT :
         T.GetPointMat().SetSize(1, 2);
         T.GetPointMat()(0, 0) = 0.0;
         T.GetPointMat()(0, 1) = 1.0;
         T.SetFE(&SegmentFE);
         break;
      case Element::TRIANGLE :
         T.GetPointMat().SetSize(2, 3);
         T.GetPointMat()(0, 0) = 0.0;
         T.GetPointMat()(1, 0) = 0.0;
         T.GetPointMat()(0, 1) = 1.0;
         T.GetPointMat()(1, 1) = 0.0;
         T.GetPointMat()(0, 2) = 0.0;
         T.GetPointMat()(1, 2) = 1.0;
         T.SetFE(&TriangleFE);
         break;
      case Element::QUADRILATERAL :
         T.GetPointMat().SetSize(2, 4);
         T.GetPointMat()(0, 0) = 0.0;
         T.GetPointMat()(1, 0) = 0.0;
         T.GetPointMat()(0, 1) = 1.0;
         T.GetPointMat()(1, 1) = 0.0;
         T.GetPointMat()(0, 2) = 1.0;
         T.GetPointMat()(1, 2) = 1.0;
         T.GetPointMat()(0, 3) = 0.0;
         T.GetPointMat()(1, 3) = 1.0;
         T.SetFE(&QuadrilateralFE);
         break;
      case Element::TETRAHEDRON :
         T.GetPointMat().SetSize(3, 4);
         T.GetPointMat()(0, 0) = 0.0;
         T.GetPointMat()(1, 0) = 0.0;
         T.GetPointMat()(2, 0) = 0.0;
         T.GetPointMat()(0, 1) = 1.0;
         T.GetPointMat()(1, 1) = 0.0;
         T.GetPointMat()(2, 1) = 0.0;
         T.GetPointMat()(0, 2) = 0.0;
         T.GetPointMat()(1, 2) = 1.0;
         T.GetPointMat()(2, 2) = 0.0;
         T.GetPointMat()(0, 3) = 0.0;
         T.GetPointMat()(1, 3) = 0.0;
         T.GetPointMat()(2, 3) = 1.0;
         T.SetFE(&TetrahedronFE);
         break;
      case Element::HEXAHEDRON :
         T.GetPointMat().SetSize(3, 8);
         T.GetPointMat()(0, 0) = 0.0;
         T.GetPointMat()(1, 0) = 0.0;
         T.GetPointMat()(2, 0) = 0.0;
         T.GetPointMat()(0, 1) = 1.0;
         T.GetPointMat()(1, 1) = 0.0;
         T.GetPointMat()(2, 1) = 0.0;
         T.GetPointMat()(0, 2) = 1.0;
         T.GetPointMat()(1, 2) = 1.0;
         T.GetPointMat()(2, 2) = 0.0;
         T.GetPointMat()(0, 3) = 0.0;
         T.GetPointMat()(1, 3) = 1.0;
         T.GetPointMat()(2, 3) = 0.0;
         T.GetPointMat()(0, 4) = 0.0;
         T.GetPointMat()(1, 4) = 0.0;
         T.GetPointMat()(2, 4) = 1.0;
         T.GetPointMat()(0, 5) = 1.0;
         T.GetPointMat()(1, 5) = 0.0;
         T.GetPointMat()(2, 5) = 1.0;
         T.GetPointMat()(0, 6) = 1.0;
         T.GetPointMat()(1, 6) = 1.0;
         T.GetPointMat()(2, 6) = 1.0;
         T.GetPointMat()(0, 7) = 0.0;
         T.GetPointMat()(1, 7) = 1.0;
         T.GetPointMat()(2, 7) = 1.0;
         T.SetFE(&HexahedronFE);
         break;
      case Element::WEDGE :
         T.GetPointMat().SetSize(3, 6);
         T.GetPointMat()(0, 0) = 0.0;
         T.GetPointMat()(1, 0) = 0.0;
         T.GetPointMat()(2, 0) = 0.0;
         T.GetPointMat()(0, 1) = 1.0;
         T.GetPointMat()(1, 1) = 0.0;
         T.GetPointMat()(2, 1) = 0.0;
         T.GetPointMat()(0, 2) = 0.0;
         T.GetPointMat()(1, 2) = 1.0;
         T.GetPointMat()(2, 2) = 0.0;
         T.GetPointMat()(0, 3) = 0.0;
         T.GetPointMat()(1, 3) = 0.0;
         T.GetPointMat()(2, 3) = 1.0;
         T.GetPointMat()(0, 4) = 1.0;
         T.GetPointMat()(1, 4) = 0.0;
         T.GetPointMat()(2, 4) = 1.0;
         T.GetPointMat()(0, 5) = 0.0;
         T.GetPointMat()(1, 5) = 1.0;
         T.GetPointMat()(2, 5) = 1.0;
         T.SetFE(&WedgeFE);
         break;
      default:
         MFEM_ABORT("Unknown element type \"" << ElemType << "\"");
         break;
   }
}

/**
 * Tests fe->CalcVShape() over a grid of IntegrationPoints
 * of resolution res. Also tests at integration points
 * that are outside the element.
 */
void TestCalcVShape(FiniteElement* fe, ElementTransformation * T, int res)
{
   int dim = fe->GetDim();
   int dof = fe->GetDof();

   Vector dofsx(dof);
   Vector dofsy(dof);
   Vector dofsz(dof);
   Vector v(dim);
   Vector vx(dim); vx = 0.0; vx[0] = 1.0;
   Vector vy(dim); vy = 0.0;
   if (dim > 1) { vy[1] = 1.0; }
   Vector vz(dim); vz = 0.0;
   if (dim > 2) { vz[2] = 1.0; }
   DenseMatrix weights( dof, dim );

   VectorConstantCoefficient vxCoef(vx);
   VectorConstantCoefficient vyCoef(vy);
   VectorConstantCoefficient vzCoef(vz);

   fe->Project(vxCoef, *T, dofsx);
   if (dim> 1) { fe->Project(vyCoef, *T, dofsy); }
   if (dim> 2) { fe->Project(vzCoef, *T, dofsz); }

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
         fe->CalcVShape(ip, weights);

         weights.MultTranspose(dofsx, v);
         REQUIRE( v[0] == Approx(1.) );
         if (dim > 1)
         {
            weights.MultTranspose(dofsy, v);
            REQUIRE( v[1] == Approx(1.) );
         }
         if (dim > 2)
         {
            weights.MultTranspose(dofsz, v);
            REQUIRE( v[2] == Approx(1.) );
         }
      }
   }
}

/*
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
*/
TEST_CASE("CalcVShape for several ND FiniteElement instances",
          "[ND_SegmentElement]"
          "[ND_TriangleElement]"
          "[ND_QuadrilateralElement]"
          "[ND_TetrahedronElement]"
          "[ND_WedgeElement]"
          "[ND_HexahedronElement]")
{
   int maxOrder = 5;
   int resolution = 10;

   SECTION("ND_SegmentElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::SEGMENT, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing ND_SegmentElement::CalcVShape() "
                   << "for order " << order << std::endl;
         ND_SegmentElement fe(order);
         TestCalcVShape(&fe, &T, resolution);
      }
   }

   SECTION("ND_TriangleElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TRIANGLE, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing ND_TriangleElement::CalcVShape() "
                   << "for order " << order << std::endl;
         ND_TriangleElement fe(order);
         TestCalcVShape(&fe, &T, resolution);
      }
   }

   SECTION("ND_QuadrilateralElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::QUADRILATERAL, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing ND_QuadrilateralElement::CalcVShape() "
                   << "for order " << order << std::endl;
         ND_QuadrilateralElement fe(order);
         TestCalcVShape(&fe, &T, resolution);
      }
   }

   SECTION("ND_TetrahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TETRAHEDRON, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing ND_TetrahedronElement::CalcVShape() "
                   << "for order " << order << std::endl;
         ND_TetrahedronElement fe(order);
         TestCalcVShape(&fe, &T, resolution);
      }
   }

   SECTION("ND_WedgeElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::WEDGE, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing ND_WedgeElement::CalcVShape() "
                   << "for order " << order << std::endl;
         ND_WedgeElement fe(order);
         TestCalcVShape(&fe, &T, resolution);
      }
   }

   SECTION("ND_HexahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::HEXAHEDRON, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing ND_HexahedronElement::CalcVShape() "
                   << "for order " << order << std::endl;
         ND_HexahedronElement fe(order);
         TestCalcVShape(&fe, &T, resolution);
      }
   }
}

TEST_CASE("CalcVShape for several RT FiniteElement instances",
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

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing RT_TriangleElement::CalcVShape() "
                   << "for order " << order << std::endl;
         RT_TriangleElement fe(order);
         TestCalcVShape(&fe, &T, resolution);
      }
   }

   SECTION("RT_QuadrilateralElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::QUADRILATERAL, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing RT_QuadrilateralElement::CalcVShape() "
                   << "for order " << order << std::endl;
         RT_QuadrilateralElement fe(order);
         TestCalcVShape(&fe, &T, resolution);
      }
   }

   SECTION("RT_TetrahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TETRAHEDRON, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing RT_TetrahedronElement::CalcVShape() "
                   << "for order " << order << std::endl;
         RT_TetrahedronElement fe(order);
         TestCalcVShape(&fe, &T, resolution);
      }
   }

   SECTION("RT_WedgeElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::WEDGE, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing RT_WedgeElement::CalcVShape() "
                   << "for order " << order << std::endl;
         RT_WedgeElement fe(order);
         TestCalcVShape(&fe, &T, resolution);
      }
   }

   SECTION("RT_HexahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::HEXAHEDRON, T);

      for (int order =1; order <= maxOrder; ++order)
      {
         std::cout << "Testing RT_HexahedronElement::CalcVShape() "
                   << "for order " << order << std::endl;
         RT_HexahedronElement fe(order);
         TestCalcVShape(&fe, &T, resolution);
      }
   }
}
