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

         // Pyramid basis functions are poorly behaved outside the
         // reference pyramid
         if (fe->GetGeomType() == Geometry::PYRAMID &&
             (ip.z < 0.0 || ip.z >= 1.0 ||
              ip.y < 0.0 || ip.y > 1.0 - ip.z ||
              ip.x < 0.0 || ip.x > 1.0 - ip.z)) { continue; }

         CAPTURE(ip.x, ip.y, ip.z);

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
          "[ND_FuentesPyramidElement]"
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

   SECTION("ND_FuentesPyramidElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::PYRAMID, T);

      ND_FuentesPyramidElement fe(order);
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

/**
 * Tests fe->CalcCurlShape() over a set of IntegrationPoints
 * chosen based on the order. Compares the computed derivatives against
 * approximate derivatives computed using the secant method.
 */
void TestFDCalcCurlShape(FiniteElement* fe, ElementTransformation * T,
                         int order)
{
   int  dof = fe->GetDof();
   int  dim = fe->GetDim();
   int cdim = fe->GetCurlDim();

   DenseMatrix pshape(dof, dim);
   DenseMatrix mshape(dof, dim);
   Vector pcomp;
   Vector mcomp;
   Vector fdcomp(dof);
   Vector fdshapecol;
   DenseMatrix dshape(dof, cdim);
   DenseMatrix fdshape(dof, cdim);

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
      fe->CalcCurlShape(pt, dshape);

      CAPTURE(pt.x, pt.y, dim == 3 ? pt.z : 0_r);

      fdshape = 0.0;
      for (int d=0; d<dim; d++)
      {
         const int d1 = (d + 1) % 3;
         const int d2 = (d + 2) % 3;

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
         fe->CalcVShape(ptm, mshape);
         fe->CalcVShape(ptp, pshape);

         if (dim == 2 && d1 < 2)
         {
            // Extract the component to be differentiated
            mshape.GetColumnReference(d1, mcomp);
            pshape.GetColumnReference(d1, pcomp);

            // Compute approximate derivatives using the secant method
            add(inv2h, pcomp, -inv2h, mcomp, fdcomp);

            fdshape.GetColumnReference(0, fdshapecol);
            fdshapecol += fdcomp;
         }
         if (dim == 2 && d2 < 2)
         {
            // Extract the component to be differentiated
            mshape.GetColumnReference(d2, mcomp);
            pshape.GetColumnReference(d2, pcomp);

            // Compute approximate derivatives using the secant method
            add(inv2h, pcomp, -inv2h, mcomp, fdcomp);

            fdshape.GetColumnReference(0, fdshapecol);
            fdshapecol -= fdcomp;
         }
         if (dim == 3)
         {
            // Extract the component to be differentiated
            mshape.GetColumnReference(d1, mcomp);
            pshape.GetColumnReference(d1, pcomp);

            // Compute approximate derivatives using the secant method
            add(inv2h, pcomp, -inv2h, mcomp, fdcomp);

            fdshape.GetColumnReference(d2, fdshapecol);
            fdshapecol += fdcomp;

            // Extract the component to be differentiated
            mshape.GetColumnReference(d2, mcomp);
            pshape.GetColumnReference(d2, pcomp);

            // Compute approximate derivatives using the secant method
            add(inv2h, pcomp, -inv2h, mcomp, fdcomp);

            fdshape.GetColumnReference(d1, fdshapecol);
            fdshapecol -= fdcomp;
         }
      }

      // Compute the difference between the computed derivative and its
      // finite difference approximation
      fdshape -= dshape;

      // Due to the scaling of the Legendre polynomials, as the integration
      // points approach the apex of a pyramid the derivatives in the x and y
      // directions become infinite. Therefore, we need to scale the finite
      // difference error estimate by the following z-dependent factor. The
      // truncation error involves the third derivative of the Legendre
      // polynomial which adds three factors of 1/(1-z). Some of the basis
      // functions are constructed using first derivatives of Legendre
      // polynomials which adds one additional factor of 1/(1-z).
      real_t pyr_fac = pyr ? std::pow(1.0/(1.0-pt.z), 4) : 1.0;

      // Determine the maximum difference between the two derivative
      // calculations
      real_t max_err = fdshape.MaxMaxNorm();

      // The factor of two is added to account for the sum of derivatives in
      // each direction needed to form the curl. The factor of dim is added to
      // account for the product rule used in computing derivatives of our
      // basis functions which are products of Legendre polynomials in the
      // different coordinates.
      REQUIRE( max_err < 2 * dim * pyr_fac * err_est );
   }
}

TEST_CASE("CalcCurlShape vs FD ND",
          "[ND_TriangleElement]"
          "[ND_QuadrilateralElement]"
          "[ND_TetrahedronElement]"
          "[ND_WedgeElement]"
          "[ND_FuentesPyramidElement]"
          "[ND_HexahedronElement]")
{
   const int maxOrder = 5;
   auto order = GENERATE_COPY(range(1, maxOrder + 1));

   CAPTURE(order);

   SECTION("ND_TriangleElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TRIANGLE, T);

      ND_TriangleElement fe(order);
      TestFDCalcCurlShape(&fe, &T, order);
   }
   SECTION("ND_QuadrilateralElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::QUADRILATERAL, T);

      ND_QuadrilateralElement fe(order);
      TestFDCalcCurlShape(&fe, &T, order);
   }
   SECTION("ND_TetrahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TETRAHEDRON, T);

      ND_TetrahedronElement fe(order);
      TestFDCalcCurlShape(&fe, &T, order);
   }
   SECTION("ND_WedgeElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::WEDGE, T);

      ND_WedgeElement fe(order);
      TestFDCalcCurlShape(&fe, &T, order);
   }

   SECTION("ND_FuentesPyramidElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::PYRAMID, T);

      ND_FuentesPyramidElement fe(order);
      TestFDCalcCurlShape(&fe, &T, order);
   }

   SECTION("ND_HexahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::HEXAHEDRON, T);

      ND_HexahedronElement fe(order);
      TestFDCalcCurlShape(&fe, &T, order);
   }
}
