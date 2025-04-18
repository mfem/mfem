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

enum Plane {XY_PLANE = 0, YZ_PLANE = 1, ZX_PLANE = 2};

static int plane_ = XY_PLANE;

void test_r2d_curl_func(const Vector &x, Vector &v)
{
   v.SetSize(3);

   int i0 = (0 + plane_) % 3;
   int i1 = (1 + plane_) % 3;
   int i2 = (2 + plane_) % 3;

   v[i0] = 4.0 * x[i1];
   v[i1] = 5.0 * x[i0];
   v[i2] = 1.0 * (x[i1] - x[i0]);
}

enum TraceDir {X_DIR = 1, Y_DIR = 2, Z_DIR = 4};

/**
 * Tests fe->CalcCurlShape() over a grid of IntegrationPoints
 * of resolution res. Also tests at integration points
 * that are outside the element.
 */
void TestCalcCurlShape(FiniteElement* fe, ElementTransformation * T, int res,
                       int mask = 7)
{
   CAPTURE(plane_);
   CAPTURE(mask);

   int  dof = fe->GetDof();
   int  dim = fe->GetDim();
   int rdim = fe->GetRangeDim();
   int cdim = fe->GetCurlDim();

   Vector dofs(dof);
   Vector v(cdim);
   DenseMatrix weights( dof, cdim );

   VectorFunctionCoefficient vCoef(dim, test_curl_func);
   VectorFunctionCoefficient vR2DCoef(dim, test_r2d_curl_func);

   if (rdim == dim)
   {
      fe->Project(vCoef, *T, dofs);
   }
   else
   {
      fe->Project(vR2DCoef, *T, dofs);
   }

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

         T->SetIntPoint(&ip);
         fe->CalcPhysCurlShape(*T, weights);

         weights.MultTranspose(dofs, v);
         if (mask & X_DIR)
         {
            REQUIRE( v[0] == Approx(1.) );
         }
         if (rdim == 3)
         {
            if (mask & Y_DIR)
            {
               REQUIRE( v[1] == Approx(1.) );
            }
            if (mask & Z_DIR)
            {
               REQUIRE( v[2] == Approx(1.) );
            }
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
          "[ND_HexahedronElement]"
          "[ND_R2D_SegmentElement]"
          "[ND_R2D_TriangleElement]"
          "[ND_R2D_QuadrilateralElement]")
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

   SECTION("ND_R2D_SegmentElement")
   {
      ND_R2D_SegmentElement fe(order);

      real_t v0[3];
      real_t v1[3];
      real_t v2[3];
      real_t v3[3];

      // xy-plane
      plane_ = XY_PLANE;
      Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
      for (int f=0; f<4; f++)
      {
         CAPTURE(f);
         FaceElementTransformations * TS = mesh.GetFaceElementTransformations(f);
         TestCalcCurlShape(&fe, TS, resolution, (f % 2) ? X_DIR : Y_DIR);
      }

      // yz-plane
      plane_ = YZ_PLANE;
      v0[0] = 0.0; v0[1] = 0.0; v0[2] = 0.0;
      v1[0] = 0.0; v1[1] = 1.0; v1[2] = 0.0;
      v2[0] = 0.0; v2[1] = 0.0; v2[2] = 1.0;
      v3[0] = 0.0; v3[1] = 1.0; v3[2] = 1.0;

      mesh.SetCurvature(1, false, 3); // Set Space Dimension to 3
      mesh.SetCurvature(-1); // Remove Nodes GridFunction
      mesh.SetNode(0, v0);
      mesh.SetNode(1, v1);
      mesh.SetNode(2, v2);
      mesh.SetNode(3, v3);

      for (int f=0; f<4; f++)
      {
         CAPTURE(f);
         FaceElementTransformations * TS = mesh.GetFaceElementTransformations(f);
         TestCalcCurlShape(&fe, TS, resolution, (f % 2) ? Y_DIR : Z_DIR);
      }

      // zx-plane
      plane_ = ZX_PLANE;
      v0[0] = 0.0; v0[1] = 0.0; v0[2] = 0.0;
      v1[0] = 0.0; v1[1] = 0.0; v1[2] = 1.0;
      v2[0] = 1.0; v2[1] = 0.0; v2[2] = 0.0;
      v3[0] = 1.0; v3[1] = 0.0; v3[2] = 1.0;

      mesh.SetNode(0, v0);
      mesh.SetNode(1, v1);
      mesh.SetNode(2, v2);
      mesh.SetNode(3, v3);

      for (int f=0; f<4; f++)
      {
         CAPTURE(f);
         FaceElementTransformations * TS = mesh.GetFaceElementTransformations(f);
         TestCalcCurlShape(&fe, TS, resolution, (f % 2) ? Z_DIR : X_DIR);
      }

      plane_ = XY_PLANE;
   }

   SECTION("ND_R2D_TriangleElement")
   {
      ND_R2D_TriangleElement fe(order);
      IsoparametricTransformation T;

      // xy-plane
      plane_ = XY_PLANE;
      GetReferenceTransformation(Element::TRIANGLE, T);
      TestCalcCurlShape(&fe, &T, resolution);

      // yz-plane
      plane_ = YZ_PLANE;
      T.GetPointMat().SetSize(3, 3);
      T.GetPointMat()(0, 0) = 0.0;
      T.GetPointMat()(1, 0) = 0.0;
      T.GetPointMat()(2, 0) = 0.0;
      T.GetPointMat()(0, 1) = 0.0;
      T.GetPointMat()(1, 1) = 1.0;
      T.GetPointMat()(2, 1) = 0.0;
      T.GetPointMat()(0, 2) = 0.0;
      T.GetPointMat()(1, 2) = 0.0;
      T.GetPointMat()(2, 2) = 1.0;
      T.SetFE(&TriangleFE);
      TestCalcCurlShape(&fe, &T, resolution);

      // zx-plane
      plane_ = ZX_PLANE;
      T.GetPointMat().SetSize(3, 3);
      T.GetPointMat()(0, 0) = 0.0;
      T.GetPointMat()(1, 0) = 0.0;
      T.GetPointMat()(2, 0) = 0.0;
      T.GetPointMat()(0, 1) = 0.0;
      T.GetPointMat()(1, 1) = 0.0;
      T.GetPointMat()(2, 1) = 1.0;
      T.GetPointMat()(0, 2) = 1.0;
      T.GetPointMat()(1, 2) = 0.0;
      T.GetPointMat()(2, 2) = 0.0;
      T.SetFE(&TriangleFE);
      TestCalcCurlShape(&fe, &T, resolution);

      plane_ = XY_PLANE;
   }

   SECTION("ND_R2D_QuadrilateralElement")
   {
      ND_R2D_QuadrilateralElement fe(order);
      IsoparametricTransformation T;

      // xy-plane
      plane_ = XY_PLANE;
      GetReferenceTransformation(Element::QUADRILATERAL, T);
      TestCalcCurlShape(&fe, &T, resolution);

      // yz-plane
      plane_ = YZ_PLANE;
      T.GetPointMat().SetSize(3, 4);
      T.GetPointMat()(0, 0) = 0.0;
      T.GetPointMat()(1, 0) = 0.0;
      T.GetPointMat()(2, 0) = 0.0;
      T.GetPointMat()(0, 1) = 0.0;
      T.GetPointMat()(1, 1) = 1.0;
      T.GetPointMat()(2, 1) = 0.0;
      T.GetPointMat()(0, 2) = 0.0;
      T.GetPointMat()(1, 2) = 1.0;
      T.GetPointMat()(2, 2) = 1.0;
      T.GetPointMat()(0, 3) = 0.0;
      T.GetPointMat()(1, 3) = 0.0;
      T.GetPointMat()(2, 3) = 1.0;
      T.SetFE(&QuadrilateralFE);
      TestCalcCurlShape(&fe, &T, resolution);

      // zx-plane
      plane_ = ZX_PLANE;
      T.GetPointMat().SetSize(3, 4);
      T.GetPointMat()(0, 0) = 0.0;
      T.GetPointMat()(1, 0) = 0.0;
      T.GetPointMat()(2, 0) = 0.0;
      T.GetPointMat()(0, 1) = 0.0;
      T.GetPointMat()(1, 1) = 0.0;
      T.GetPointMat()(2, 1) = 1.0;
      T.GetPointMat()(0, 2) = 1.0;
      T.GetPointMat()(1, 2) = 0.0;
      T.GetPointMat()(2, 2) = 1.0;
      T.GetPointMat()(0, 3) = 1.0;
      T.GetPointMat()(1, 3) = 0.0;
      T.GetPointMat()(2, 3) = 0.0;
      T.SetFE(&QuadrilateralFE);
      TestCalcCurlShape(&fe, &T, resolution);

      plane_ = XY_PLANE;
   }
}

/**
 * Tests fe->CalcCurlShape() over a set of IntegrationPoints
 * chosen based on the order. Compares the computed derivatives against
 * approximate derivatives computed using the secant method.
 */
void TestFDCalcCurlShape(FiniteElement* fe, ElementTransformation * T,
                         int order, int mask = 7)
{
   int  dof = fe->GetDof();
   int  dim = fe->GetDim();
   int rdim = fe->GetRangeDim();
   int cdim = fe->GetCurlDim();

   DenseMatrix pshape(dof, rdim);
   DenseMatrix mshape(dof, rdim);
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

      CAPTURE(pt.x, pt.y, pt.z);

      fdshape = 0.0;
      for (int d=0; d<rdim; d++)
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

         if (rdim == 2 && d1 < 2)
         {
            // Extract the component to be differentiated
            mshape.GetColumnReference(d1, mcomp);
            pshape.GetColumnReference(d1, pcomp);

            // Compute approximate derivatives using the secant method
            add(inv2h, pcomp, -inv2h, mcomp, fdcomp);

            fdshape.GetColumnReference(0, fdshapecol);
            fdshapecol += fdcomp;
         }
         if (rdim == 2 && d2 < 2)
         {
            // Extract the component to be differentiated
            mshape.GetColumnReference(d2, mcomp);
            pshape.GetColumnReference(d2, pcomp);

            // Compute approximate derivatives using the secant method
            add(inv2h, pcomp, -inv2h, mcomp, fdcomp);

            fdshape.GetColumnReference(0, fdshapecol);
            fdshapecol -= fdcomp;
         }
         if (rdim == 3)
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
          "[ND_HexahedronElement]"
          "[ND_R2D_SegmentElement]"
          "[ND_R2D_TriangleElement]"
          "[ND_R2D_QuadrilateralElement]")
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

   SECTION("ND_R2D_SegmentElement")
   {
      ND_R2D_SegmentElement fe(order);

      real_t v0[3];
      real_t v1[3];
      real_t v2[3];
      real_t v3[3];

      // xy-plane
      Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
      for (int f=0; f<4; f++)
      {
         FaceElementTransformations * TS =
            mesh.GetFaceElementTransformations(f);
         TestFDCalcCurlShape(&fe, TS, order, ((f % 2) ? Y_DIR : X_DIR) | Z_DIR);
      }

      // yz-plane
      v0[0] = 0.0; v0[1] = 0.0; v0[2] = 0.0;
      v1[0] = 0.0; v1[1] = 1.0; v1[2] = 0.0;
      v2[0] = 0.0; v2[1] = 0.0; v2[2] = 1.0;
      v3[0] = 0.0; v3[1] = 1.0; v3[2] = 1.0;

      mesh.SetCurvature(1, false, 3); // Set Space Dimension to 3
      mesh.SetCurvature(-1); // Remove Nodes GridFunction
      mesh.SetNode(0, v0);
      mesh.SetNode(1, v1);
      mesh.SetNode(2, v2);
      mesh.SetNode(3, v3);

      for (int f=0; f<4; f++)
      {
         FaceElementTransformations * TS =
            mesh.GetFaceElementTransformations(f);
         TestFDCalcCurlShape(&fe, TS, order, ((f % 2) ? Z_DIR : Y_DIR) | X_DIR);
      }

      // zx-plane
      v0[0] = 0.0; v0[1] = 0.0; v0[2] = 0.0;
      v1[0] = 0.0; v1[1] = 0.0; v1[2] = 1.0;
      v2[0] = 1.0; v2[1] = 0.0; v2[2] = 0.0;
      v3[0] = 1.0; v3[1] = 0.0; v3[2] = 1.0;

      mesh.SetNode(0, v0);
      mesh.SetNode(1, v1);
      mesh.SetNode(2, v2);
      mesh.SetNode(3, v3);

      for (int f=0; f<4; f++)
      {
         FaceElementTransformations * TS =
            mesh.GetFaceElementTransformations(f);
         TestFDCalcCurlShape(&fe, TS, order, ((f % 2) ? X_DIR : Z_DIR) | Y_DIR);
      }
   }

   SECTION("ND_R2D_TriangleElement")
   {
      ND_R2D_TriangleElement fe(order);
      IsoparametricTransformation T;

      // xy-plane
      GetReferenceTransformation(Element::TRIANGLE, T);
      TestFDCalcCurlShape(&fe, &T, order);

      // yz-plane
      T.GetPointMat().SetSize(3, 3);
      T.GetPointMat()(0, 0) = 0.0;
      T.GetPointMat()(1, 0) = 0.0;
      T.GetPointMat()(2, 0) = 0.0;
      T.GetPointMat()(0, 1) = 0.0;
      T.GetPointMat()(1, 1) = 1.0;
      T.GetPointMat()(2, 1) = 0.0;
      T.GetPointMat()(0, 2) = 0.0;
      T.GetPointMat()(1, 2) = 0.0;
      T.GetPointMat()(2, 2) = 1.0;
      T.SetFE(&TriangleFE);
      TestFDCalcCurlShape(&fe, &T, order);

      // zx-plane
      T.GetPointMat().SetSize(3, 3);
      T.GetPointMat()(0, 0) = 0.0;
      T.GetPointMat()(1, 0) = 0.0;
      T.GetPointMat()(2, 0) = 0.0;
      T.GetPointMat()(0, 1) = 0.0;
      T.GetPointMat()(1, 1) = 0.0;
      T.GetPointMat()(2, 1) = 1.0;
      T.GetPointMat()(0, 2) = 1.0;
      T.GetPointMat()(1, 2) = 0.0;
      T.GetPointMat()(2, 2) = 0.0;
      T.SetFE(&TriangleFE);
      TestFDCalcCurlShape(&fe, &T, order);
   }

   SECTION("ND_R2D_QuadrilateralElement")
   {
      ND_R2D_QuadrilateralElement fe(order);
      IsoparametricTransformation T;

      // xy-plane
      GetReferenceTransformation(Element::QUADRILATERAL, T);
      TestFDCalcCurlShape(&fe, &T, order);

      // yz-plane
      T.GetPointMat().SetSize(3, 4);
      T.GetPointMat()(0, 0) = 0.0;
      T.GetPointMat()(1, 0) = 0.0;
      T.GetPointMat()(2, 0) = 0.0;
      T.GetPointMat()(0, 1) = 0.0;
      T.GetPointMat()(1, 1) = 1.0;
      T.GetPointMat()(2, 1) = 0.0;
      T.GetPointMat()(0, 2) = 0.0;
      T.GetPointMat()(1, 2) = 1.0;
      T.GetPointMat()(2, 2) = 1.0;
      T.GetPointMat()(0, 3) = 0.0;
      T.GetPointMat()(1, 3) = 0.0;
      T.GetPointMat()(2, 3) = 1.0;
      T.SetFE(&QuadrilateralFE);
      TestFDCalcCurlShape(&fe, &T, order);

      // zx-plane
      T.GetPointMat().SetSize(3, 4);
      T.GetPointMat()(0, 0) = 0.0;
      T.GetPointMat()(1, 0) = 0.0;
      T.GetPointMat()(2, 0) = 0.0;
      T.GetPointMat()(0, 1) = 0.0;
      T.GetPointMat()(1, 1) = 0.0;
      T.GetPointMat()(2, 1) = 1.0;
      T.GetPointMat()(0, 2) = 1.0;
      T.GetPointMat()(1, 2) = 0.0;
      T.GetPointMat()(2, 2) = 1.0;
      T.GetPointMat()(0, 3) = 1.0;
      T.GetPointMat()(1, 3) = 0.0;
      T.GetPointMat()(2, 3) = 0.0;
      T.SetFE(&QuadrilateralFE);
      TestFDCalcCurlShape(&fe, &T, order);
   }
}
