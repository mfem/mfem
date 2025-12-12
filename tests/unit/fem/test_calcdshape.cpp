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

         // Pyramid basis functions are poorly behaved outside the
         // reference pyramid
         if (fe->GetGeomType() == Geometry::PYRAMID &&
             (ip.z >= 1.0 || ip.y > 1.0 - ip.z || ip.x > 1.0 - ip.z)) { continue; }

         CAPTURE(ip.x, ip.y, ip.z);

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
          "[H1_FuentesPyramidElement]"
          "[H1_BergotPyramidElement]"
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

   SECTION("H1_FuentesPyramidElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::PYRAMID, T);

      H1_FuentesPyramidElement fe(order);
      TestCalcDShape(&fe, &T, resolution);
   }

   SECTION("H1_BergotPyramidElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::PYRAMID, T);

      H1_BergotPyramidElement fe(order);
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

/**
 * Tests fe->CalcDShape() over a set of IntegrationPoints
 * chosen based on the order. Compares the computed derivatives against
 * approximate derivatives computed using the secant method.
 */
void TestFDCalcDShape(FiniteElement* fe, ElementTransformation * T, int order)
{
   int  dof = fe->GetDof();
   int  dim = fe->GetDim();

   Vector pshape(dof);
   Vector mshape(dof);
   Vector fd;
   DenseMatrix dshape( dof, dim );
   DenseMatrix fdshape( dof, dim );

   // Optimal step size for central difference
   real_t h = std::cbrt(std::numeric_limits<real_t>::epsilon());
   real_t inv2h = 0.5 / h;

   // Error in the finite difference approximation of the derivative of a
   // Legendre polynomial: P_n'''(1) h^2 / 6. Because we use shifted and scaled
   // Legendre polynomials we need to increase these estimates by 2^3. We also
   // make use of the fact that the third derivatives of Legendre polynomials
   // are bounded by +/- (n+1)(n+2)(n+3)(n+4)(n+5)(n+6)/48.
   real_t err_est = (order + 1) * (order + 2) * (order + 3) *
                    (order + 4) * (order + 5) * (order + 6) * h * h / 36.0;

   bool pyr = fe->GetGeomType() == Geometry::PYRAMID;

   const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), 2*order+dim-1);

   IntegrationPoint ptp;
   IntegrationPoint ptm;

   int npoints = ir->GetNPoints();
   for (int i=0; i < npoints; ++i)
   {
      // Get the current integration point from the integration rule
      IntegrationPoint pt = ir->IntPoint(i);
      fe->CalcDShape(pt, dshape);
      for (int d=0; d<dim; d++)
      {
         // Compute shifted integration points
         switch (d)
         {
            case 0:
               ptm.x = pt.x - h; ptm.y = pt.y; ptm.z = pt.z;
               ptp.x = pt.x + h; ptp.y = pt.y; ptp.z = pt.z;
               break;
            case 1:
               ptm.x = pt.x; ptm.y = pt.y - h; ptm.z = pt.z;
               ptp.x = pt.x; ptp.y = pt.y + h; ptp.z = pt.z;
               break;
            case 2:
               ptm.x = pt.x; ptm.y = pt.y; ptm.z = pt.z - h;
               ptp.x = pt.x; ptp.y = pt.y; ptp.z = pt.z + h;
               break;
            default:
               ptm = pt;
               ptp = pt;
         }

         // Compute shape functions at the shifted points
         fe->CalcShape(ptm, mshape);
         fe->CalcShape(ptp, pshape);

         // Compute approximate derivatives using the secant method
         fdshape.GetColumnReference(d, fd);
         add(inv2h, pshape, -inv2h, mshape, fd);
      }

      // Compute the difference between the computed derivative and its
      // finite difference approximation
      fdshape -= dshape;

      // Due to the scaling of the Legendre polynomials, as the integration
      // points approach the apex of a pyramid the derivatives in the x and y
      // directions become infinite. Therefore, we need to scale the finite
      // difference error estimate by the following z-dependent factor.
      real_t pyr_fac = pyr ? std::pow(1.0/(1.0-pt.z), 3) : 1.0;

      // Determine the maximum difference between the two derivative
      // calculations
      real_t max_err = fdshape.MaxMaxNorm();

      // The additional factor of dim is added to account for the product
      // rule used in computing derivatives of our basis functions which are
      // products of Legendre polynomials in the different coordinates.
      REQUIRE( max_err < dim * pyr_fac * err_est );
   }
}

TEST_CASE("CalcDShape vs FD H1",
          "[H1_SegmentElement]"
          "[H1_TriangleElement]"
          "[H1_QuadrilateralElement]"
          "[H1_TetrahedronElement]"
          "[H1_WedgeElement]"
          "[H1_FuentesPyramidElement]"
          "[H1_BergotPyramidElement]"
          "[H1_HexahedronElement]")
{
   const int maxOrder = 5;
   auto order = GENERATE_COPY(range(1, maxOrder + 1));

   CAPTURE(order);

   SECTION("H1_SegmentElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::SEGMENT, T);

      H1_SegmentElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("H1_TriangleElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TRIANGLE, T);

      H1_TriangleElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("H1_QuadrilateralElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::QUADRILATERAL, T);

      H1_QuadrilateralElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("H1_TetrahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TETRAHEDRON, T);

      H1_TetrahedronElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("H1_WedgeElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::WEDGE, T);

      H1_WedgeElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("H1_FuentesPyramidElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::PYRAMID, T);

      H1_FuentesPyramidElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("H1_BergotPyramidElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::PYRAMID, T);

      H1_BergotPyramidElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("H1_HexahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::HEXAHEDRON, T);

      H1_HexahedronElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

}

TEST_CASE("CalcDShape vs FD L2",
          "[L2_SegmentElement]"
          "[L2_TriangleElement]"
          "[L2_QuadrilateralElement]"
          "[L2_TetrahedronElement]"
          "[L2_WedgeElement]"
          "[L2_FuentesPyramidElement]"
          "[L2_BergotPyramidElement]"
          "[L2_HexahedronElement]")
{
   const int maxOrder = 5;
   auto order = GENERATE_COPY(range(0, maxOrder));

   CAPTURE(order);

   SECTION("L2_SegmentElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::SEGMENT, T);

      L2_SegmentElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("L2_TriangleElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TRIANGLE, T);

      L2_TriangleElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("L2_QuadrilateralElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::QUADRILATERAL, T);

      L2_QuadrilateralElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("L2_TetrahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TETRAHEDRON, T);

      L2_TetrahedronElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("L2_WedgeElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::WEDGE, T);

      L2_WedgeElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("L2_FuentesPyramidElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::PYRAMID, T);

      L2_FuentesPyramidElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("L2_BergotPyramidElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::PYRAMID, T);

      L2_BergotPyramidElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

   SECTION("L2_HexahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::HEXAHEDRON, T);

      L2_HexahedronElement fe(order);
      TestFDCalcDShape(&fe, &T, order);
   }

}
