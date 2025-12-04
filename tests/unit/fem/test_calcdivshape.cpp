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

         // Pyramid basis functions are poorly behaved outside the
         // reference pyramid
         if (fe->GetGeomType() == Geometry::PYRAMID &&
             (ip.z >= 1.0 || ip.y > 1.0 - ip.z || ip.x > 1.0 - ip.z)) { continue; }

         CAPTURE(ip.x, ip.y, dim == 3 ? ip.z : 0_r);

         fe->CalcDivShape(ip, weights);

         REQUIRE( weights * dofs == Approx(1.) );
      }
   }
}

TEST_CASE("CalcDivShape RT",
          "[RT_TriangleElement]"
          "[RT_QuadrilateralElement]"
          "[RT_TetrahedronElement]"
          "[RT_WedgeElement]"
          "[RT_FuentesPyramidElement]"
          "[RT_HexahedronElement]")
{
   const int maxOrder = 5;
   const int resolution = 10;
   auto order = GENERATE_COPY(range(1, maxOrder + 1));

   CAPTURE(order);

   SECTION("RT_TriangleElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TRIANGLE, T);

      RT_TriangleElement fe(order - 1);
      TestCalcDivShape(&fe, &T, resolution);
   }

   SECTION("RT_QuadrilateralElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::QUADRILATERAL, T);

      RT_QuadrilateralElement fe(order - 1);
      TestCalcDivShape(&fe, &T, resolution);
   }

   SECTION("RT_TetrahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TETRAHEDRON, T);

      RT_TetrahedronElement fe(order - 1);
      TestCalcDivShape(&fe, &T, resolution);
   }

   SECTION("RT_WedgeElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::WEDGE, T);

      RT_WedgeElement fe(order - 1);
      TestCalcDivShape(&fe, &T, resolution);
   }

   SECTION("RT_FuentesPyramidElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::PYRAMID, T);

      RT_FuentesPyramidElement fe(order - 1);
      TestCalcDivShape(&fe, &T, resolution);
   }

   SECTION("RT_HexahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::HEXAHEDRON, T);

      RT_HexahedronElement fe(order - 1);
      TestCalcDivShape(&fe, &T, resolution);
   }
}

/**
 * Tests fe->CalcDivShape() over a set of IntegrationPoints
 * chosen based on the order. Compares the computed derivatives against
 * approximate derivatives computed using the secant method.
 */
void TestFDCalcDivShape(FiniteElement* fe, ElementTransformation * T, int order)
{
   int  dof = fe->GetDof();
   int  dim = fe->GetDim();

   DenseMatrix pshape(dof, dim);
   DenseMatrix mshape(dof, dim);
   Vector pcomp;
   Vector mcomp;
   Vector dshape(dof);
   Vector fdcomp(dof), fdshape(dof);

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
      fe->CalcDivShape(pt, dshape);

      CAPTURE(pt.x, pt.y, dim == 3 ? pt.z : 0_r);

      fdshape = 0.0;
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
         fe->CalcVShape(ptm, mshape);
         fe->CalcVShape(ptp, pshape);

         // Extract the component to be differentiated
         mshape.GetColumnReference(d, mcomp);
         pshape.GetColumnReference(d, pcomp);

         // Compute approximate derivatives using the secant method
         add(inv2h, pcomp, -inv2h, mcomp, fdcomp);

         fdshape += fdcomp;
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
      real_t max_err = fdshape.Normlinf();

      // The first factor of dim is added to account for the product
      // rule used in computing derivatives of our basis functions which are
      // products of Legendre polynomials in the different coordinates. The
      // second factor of dim is added to account for the sum of derivatives
      // in each direction needed to form the divergence.
      REQUIRE( max_err < dim * dim * pyr_fac * err_est );
   }
}

TEST_CASE("CalcDivShape vs FD RT",
          "[RT_TriangleElement]"
          "[RT_QuadrilateralElement]"
          "[RT_TetrahedronElement]"
          "[RT_WedgeElement]"
          "[RT_FuentesPyramidElement]"
          "[RT_HexahedronElement]")
{
   const int maxOrder = 5;
   auto order = GENERATE_COPY(range(1, maxOrder + 1));

   CAPTURE(order);

   SECTION("RT_TriangleElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TRIANGLE, T);

      RT_TriangleElement fe(order - 1);
      TestFDCalcDivShape(&fe, &T, order);
   }

   SECTION("RT_QuadrilateralElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::QUADRILATERAL, T);

      RT_QuadrilateralElement fe(order - 1);
      TestFDCalcDivShape(&fe, &T, order);
   }

   SECTION("RT_TetrahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::TETRAHEDRON, T);

      RT_TetrahedronElement fe(order - 1);
      TestFDCalcDivShape(&fe, &T, order);
   }

   SECTION("RT_WedgeElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::WEDGE, T);

      RT_WedgeElement fe(order - 1);
      TestFDCalcDivShape(&fe, &T, order);
   }

   SECTION("RT_FuentesPyramidElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::PYRAMID, T);

      RT_FuentesPyramidElement fe(order - 1);
      TestFDCalcDivShape(&fe, &T, order);
   }

   SECTION("RT_HexahedronElement")
   {
      IsoparametricTransformation T;
      GetReferenceTransformation(Element::HEXAHEDRON, T);

      RT_HexahedronElement fe(order - 1);
      TestFDCalcDivShape(&fe, &T, order);
   }
}
