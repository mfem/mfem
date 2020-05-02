// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include <string>
#include <sstream>
#include <fstream>

using namespace mfem;

// Prefix string for a single element 2D mfem quad mesh
std::string meshPrefixStr =
   "MFEM mesh v1.0"  "\n\n"
   "dimension"         "\n"
   "2"               "\n\n"
   "elements"          "\n"
   "1"                 "\n"
   "1 3 0 1 2 3"     "\n\n"
   "boundary"          "\n"
   "0"               "\n\n";

// Nodal grid function for a C-shaped quadratic quadrilateral
std::string CShapedNodesStr =
   "vertices"                            "\n"
   "4"                                 "\n\n"
   "nodes"                               "\n"
   "FiniteElementSpace"                  "\n"
   "FiniteElementCollection: Quadratic"  "\n"
   "VDim: 2"                             "\n"
   "Ordering: 1"                         "\n"
   "0 0"                                 "\n"
   "0 2"                                 "\n"
   "0 6"                                 "\n"
   "0 8"                                 "\n"
   "0 1"                                 "\n"
   "-6 4"                                "\n"
   "0 7"                                 "\n"
   "-8 4"                                "\n"
   "-7 4"                                "\n";

TEST_CASE("InverseElementTransformation",
          "[InverseElementTransformation]")
{
   typedef InverseElementTransformation InvTransform;

   // Create quadratic with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << meshPrefixStr << CShapedNodesStr;
   Mesh mesh( meshStr );

   REQUIRE( mesh.GetNE() == 1 );
   REQUIRE( mesh.GetNodes() != NULL );

   // Optionally, dump mesh to disk
   bool dumpMesh = false;
   if (dumpMesh)
   {
      std::string filename = "c_shaped_quadratic_mesh";
      VisItDataCollection dataCol(filename, &mesh);
      dataCol.Save();
   }

   const int res = 100;
   const int dim = 2;
   const double tol = 2e-14;

   SECTION("{ C-shaped Q2 Quad }")
   {
      // Create a uniform grid of integration points over the element
      const int geom = mesh.GetElementBaseGeometry(0);
      RefinedGeometry* ref =
         GlobGeometryRefiner.Refine(Geometry::Type(geom), res);
      const IntegrationRule& intRule = ref->RefPts;

      // Create a transformation
      IsoparametricTransformation tr;
      mesh.GetElementTransformation(0, &tr);
      Vector v(dim);

      const int npts = intRule.GetNPoints();
      int pts_found = 0;
      double max_err = 0.0;
      for (int i=0; i<npts; ++i)
      {
         // Transform the integration point into space
         const IntegrationPoint& ip = intRule.IntPoint(i);
         tr.Transform(ip, v);

         // Now reverse the transformation
         IntegrationPoint ipRev;

         int res = tr.TransformBack(v, ipRev);

         // Check that the reverse transform was successful
         if ( res == InvTransform::Inside )
         {
            pts_found++;

            // Accumulate the maximal error
            max_err = std::max(max_err, std::abs(ipRev.x - ip.x));
            max_err = std::max(max_err, std::abs(ipRev.y - ip.y));
         }
      }
      std::cout << "Points found: " << pts_found << '/' << npts << '\n'
                << "Maximum error: " << max_err << '\n';
      REQUIRE( pts_found == npts );
      REQUIRE( max_err <= tol );
   }

   SECTION("{ Spiral Q20 Quad }")
   {
      // Load the spiral mesh from file:
      std::ifstream mesh_file("./data/quad-spiral-q20.mesh");
      REQUIRE( mesh_file.good() );

      const int npts = 100; // number of random points to test
      const int min_found_pts = 93;
      const int rand_seed = 189548;
      srand(rand_seed);

      Mesh mesh(mesh_file);
      REQUIRE( mesh.Dimension() == 2 );
      REQUIRE( mesh.SpaceDimension() == 2 );
      REQUIRE( mesh.GetNE() == 1 );

      ElementTransformation &T = *mesh.GetElementTransformation(0);
      InvTransform inv_T(&T);
      // inv_T.SetInitialGuessType(InvTransform::ClosestPhysNode);
      inv_T.SetInitialGuessType(InvTransform::ClosestRefNode);
      // inv_T.SetSolverType(InvTransform::Newton);
      // inv_T.SetSolverType(InvTransform::NewtonSegmentProject);
      inv_T.SetSolverType(InvTransform::NewtonElementProject);
      inv_T.SetPrintLevel(0); // 0 - print errors
      IntegrationPoint ip, ipRev;
      Vector pt;

      int pts_found = 0;
      double max_err = 0.0;
      for (int i = 0; i < npts; i++)
      {
         Geometry::GetRandomPoint(T.GetGeometryType(), ip);
         T.Transform(ip, pt);

         const int res = inv_T.Transform(pt, ipRev);
         if (res == InvTransform::Inside)
         {
            pts_found++;

            // Accumulate the maximal error
            max_err = std::max(max_err, std::abs(ipRev.x - ip.x));
            max_err = std::max(max_err, std::abs(ipRev.y - ip.y));
         }
      }
      std::cout << "Points found: " << pts_found << '/' << npts << '\n'
                << "Maximum error: " << max_err << '\n';
      REQUIRE( pts_found >= min_found_pts );
      REQUIRE( max_err <= tol );
   }
}
