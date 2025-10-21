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
#include "unit_tests.hpp"

#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

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

// Nodal grid function for a C-shaped quadratic quadrilateral embedded in 3D
std::string EmbCShapedNodesStr =
   "vertices"                            "\n"
   "4"                                 "\n\n"
   "nodes"                               "\n"
   "FiniteElementSpace"                  "\n"
   "FiniteElementCollection: Quadratic"  "\n"
   "VDim: 3"                             "\n"
   "Ordering: 1"                         "\n"
   "0 0 0"                               "\n"
   "0 2 2"                               "\n"
   "0 6 6"                                "\n"
   "0 8 8"                                "\n"
   "0 1 1"                                "\n"
   "-6 4 4"                               "\n"
   "0 7 7"                                "\n"
   "-8 4 4"                               "\n"
   "-7 4 4"                               "\n";

TEST_CASE("InverseElementTransformation",
          "[InverseElementTransformation]")
{
   typedef InverseElementTransformation InvTransform;

   const real_t tol = 2e-14;

   SECTION("{ C-shaped Q2 Quad }")
   {
      // Create quadratic with single C-shaped quadrilateral
      std::stringstream meshStr;
      meshStr << meshPrefixStr << CShapedNodesStr;
      Mesh mesh( meshStr );

      REQUIRE( mesh.GetNE() == 1 );
      REQUIRE( mesh.GetNodes() != nullptr );

      // Optionally, dump mesh to disk
      bool dumpMesh = false;
      if (dumpMesh)
      {
         std::string filename = "c_shaped_quadratic_mesh";
         VisItDataCollection dataCol(filename, &mesh);
         dataCol.Save();
      }

      const int times = 100;
      const int dim = 2;

      // Create a uniform grid of integration points over the element
      const int geom = mesh.GetElementBaseGeometry(0);
      RefinedGeometry* ref =
         GlobGeometryRefiner.Refine(Geometry::Type(geom), times);
      const IntegrationRule& intRule = ref->RefPts;

      // Create a transformation
      IsoparametricTransformation tr;
      mesh.GetElementTransformation(0, &tr);
      Vector v(dim);

      const int npts = intRule.GetNPoints();
      int pts_found = 0;
      real_t max_err = 0.0;
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
      CAPTURE(pts_found, npts, max_err);
      REQUIRE( pts_found == npts );
      REQUIRE( max_err <= tol );
   }

   SECTION("{ Spiral Q20 Quad }")
   {
      // Load the spiral mesh from file:
      std::ifstream mesh_file("./data/quad-spiral-q20.mesh");
      REQUIRE( mesh_file.good() );

      const int npts = 100; // number of random points to test
      const int rand_seed = 189548;
      srand(rand_seed);

      Mesh mesh(mesh_file);
      REQUIRE( mesh.Dimension() == 2 );
      REQUIRE( mesh.SpaceDimension() == 2 );
      REQUIRE( mesh.GetNE() == 1 );

      ElementTransformation &T = *mesh.GetElementTransformation(0);
      InvTransform inv_T(&T);
      inv_T.SetInitialGuessType(InvTransform::EdgeScan);
      // inv_T.SetSolverType(InvTransform::Newton);
      // inv_T.SetSolverType(InvTransform::NewtonSegmentProject);
      int desired_order = 4;
      inv_T.SetSolverType(InvTransform::NewtonElementProject);
      inv_T.SetInitGuessRelOrder(desired_order - T.Order());
      inv_T.SetInitGuessPointsType(Quadrature1D::ClosedUniform);
      inv_T.SetPrintLevel(-1); // 0 - print errors
      IntegrationPoint ip, ipRev;
      Vector pt;

      int pts_found = 0;
      real_t max_err = 0.0;
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
      CAPTURE(pts_found, npts, max_err);
      REQUIRE( pts_found == npts );
      REQUIRE( max_err <= tol );
   }
}

TEST_CASE("BatchInverseElementTransformation",
          "[InverseElementTransformation], [GPU]")
{
   const real_t tol = 4e-13;

   SECTION("{ Segment Q2 1D }")
   {
      // basic 1D segment
      Mesh mesh = Mesh::MakeCartesian1D(1);
      mesh.SetCurvature(2);

      REQUIRE( mesh.GetNE() == 1 );
      REQUIRE( mesh.GetNodes() != nullptr );

      const int times = 100;
      const int dim = mesh.Dimension();
      const int sdim = mesh.SpaceDimension();

      REQUIRE( dim == 1 );
      REQUIRE( sdim == 1 );

      // Create a uniform grid of integration points over the element
      const int geom = mesh.GetElementBaseGeometry(0);
      RefinedGeometry* ref =
         GlobGeometryRefiner.Refine(Geometry::Type(geom), times);
      const IntegrationRule& intRule = ref->RefPts;

      // Create a transformation
      IsoparametricTransformation tr;
      mesh.GetElementTransformation(0, &tr);
      Vector v(dim);

      const int npts = intRule.GetNPoints();
      Vector orig_ref_space;
      Vector phys_space;
      Array<int> elems;
      Array<int> res_type;
      Vector res_ref_space;

      BatchInverseElementTransformation itransform(mesh);
      itransform.SetInitialGuessType(InverseElementTransformation::Center);

      orig_ref_space.SetSize(npts * dim);
      phys_space.SetSize(npts * sdim);
      elems.SetSize(npts);
      res_type.SetSize(npts);
      res_ref_space.SetSize(npts * dim);

      orig_ref_space.HostWrite();
      phys_space.HostWrite();
      elems.HostWrite();

      for (int i=0; i<npts; ++i)
      {
         elems[i] = 0;
         // Transform the integration point into space
         const IntegrationPoint& ip = intRule.IntPoint(i);
         tr.Transform(ip, v);

         real_t tmp[3];
         ip.Get(tmp, dim);
         for (int d = 0; d < dim; ++d)
         {
            orig_ref_space(i + d * npts) = tmp[d];
         }
         for (int d = 0; d < sdim; ++d)
         {
            phys_space(i + d * npts) = v(d);
         }
      }

      // now batch reverse transform
      itransform.Transform(phys_space, elems, res_type, res_ref_space);
      res_type.HostRead();
      res_ref_space.HostRead();
      int pts_found = 0;
      real_t max_err = 0;
      for (int i = 0; i < npts; ++i)
      {
         if (AsConst(res_type)[i] == InverseElementTransformation::Inside)
         {
            ++pts_found;
            for (int d = 0; d < dim; ++d)
            {
               max_err = fmax(max_err,
                              fabs(AsConst(res_ref_space)[i + d * npts] -
                                   orig_ref_space[i + d * npts]));
            }
         }
      }

      CAPTURE(pts_found, npts, max_err);
      REQUIRE( pts_found == npts );
      REQUIRE( max_err <= tol );
   }

   SECTION("{ Segment Q2 2D }")
   {
      // basic 1D segment embedded in 2D space
      Mesh mesh = Mesh::MakeCartesian1D(1);
      mesh.SetCurvature(2, false, 2);
      // apply some simple transform
      {
         VectorFunctionCoefficient coeff(
            2, std::function<void(const Vector &, Vector &)>(
               [](const Vector &xi, Vector &c)
         {
            c[0] = xi[0];
            c[1] = xi[0] * xi[0];
         }));
         mesh.Transform(coeff);
      }

      REQUIRE( mesh.GetNE() == 1 );
      REQUIRE( mesh.GetNodes() != nullptr );

      const int times = 100;
      const int dim = mesh.Dimension();
      const int sdim = mesh.SpaceDimension();

      REQUIRE( dim == 1 );
      REQUIRE( sdim == 2 );

      // Create a uniform grid of integration points over the element
      const int geom = mesh.GetElementBaseGeometry(0);
      RefinedGeometry* ref =
         GlobGeometryRefiner.Refine(Geometry::Type(geom), times);
      const IntegrationRule& intRule = ref->RefPts;

      // Create a transformation
      IsoparametricTransformation tr;
      mesh.GetElementTransformation(0, &tr);
      Vector v(dim);

      const int npts = intRule.GetNPoints();
      Vector orig_ref_space;
      Vector phys_space;
      Array<int> elems;
      Array<int> res_type;
      Vector res_ref_space;

      BatchInverseElementTransformation itransform(mesh);
      itransform.SetInitialGuessType(InverseElementTransformation::Center);

      orig_ref_space.SetSize(npts * dim);
      phys_space.SetSize(npts * sdim);
      elems.SetSize(npts);
      res_type.SetSize(npts);
      res_ref_space.SetSize(npts * dim);

      orig_ref_space.HostWrite();
      phys_space.HostWrite();
      elems.HostWrite();

      for (int i=0; i<npts; ++i)
      {
         elems[i] = 0;
         // Transform the integration point into space
         const IntegrationPoint& ip = intRule.IntPoint(i);
         tr.Transform(ip, v);

         real_t tmp[3];
         ip.Get(tmp, dim);
         for (int d = 0; d < dim; ++d)
         {
            orig_ref_space(i + d * npts) = tmp[d];
         }
         for (int d = 0; d < sdim; ++d)
         {
            phys_space(i + d * npts) = v(d);
         }
      }

      // now batch reverse transform
      itransform.Transform(phys_space, elems, res_type, res_ref_space);
      res_type.HostRead();
      res_ref_space.HostRead();
      int pts_found = 0;
      real_t max_err = 0;
      for (int i = 0; i < npts; ++i)
      {
         if (AsConst(res_type)[i] == InverseElementTransformation::Inside)
         {
            ++pts_found;
            for (int d = 0; d < dim; ++d)
            {
               max_err = fmax(max_err,
                              fabs(AsConst(res_ref_space)[i + d * npts] -
                                   orig_ref_space[i + d * npts]));
            }
         }
      }

      CAPTURE(pts_found, npts, max_err);
      REQUIRE( pts_found == npts );
      REQUIRE( max_err <= tol );
   }

   SECTION("{ Segment Q2 3D }")
   {
      // basic 1D segment embedded in 2D space
      Mesh mesh = Mesh::MakeCartesian1D(1);
      mesh.SetCurvature(2, false, 3);
      // apply some simple transform
      {
         VectorFunctionCoefficient coeff(
            3, std::function<void(const Vector &, Vector &)>(
               [](const Vector &xi, Vector &c)
         {
            c[0] = xi[0];
            c[1] = xi[0] * xi[0];
            c[2] = 1 - xi[0] * xi[0];
         }));
         mesh.Transform(coeff);
      }

      REQUIRE( mesh.GetNE() == 1 );
      REQUIRE( mesh.GetNodes() != nullptr );

      const int times = 100;
      const int dim = mesh.Dimension();
      const int sdim = mesh.SpaceDimension();

      REQUIRE( dim == 1 );
      REQUIRE( sdim == 3 );

      // Create a uniform grid of integration points over the element
      const int geom = mesh.GetElementBaseGeometry(0);
      RefinedGeometry* ref =
         GlobGeometryRefiner.Refine(Geometry::Type(geom), times);
      const IntegrationRule& intRule = ref->RefPts;

      // Create a transformation
      IsoparametricTransformation tr;
      mesh.GetElementTransformation(0, &tr);
      Vector v(dim);

      const int npts = intRule.GetNPoints();
      Vector orig_ref_space;
      Vector phys_space;
      Array<int> elems;
      Array<int> res_type;
      Vector res_ref_space;

      BatchInverseElementTransformation itransform(mesh);
      itransform.SetInitialGuessType(InverseElementTransformation::Center);

      orig_ref_space.SetSize(npts * dim);
      phys_space.SetSize(npts * sdim);
      elems.SetSize(npts);
      res_type.SetSize(npts);
      res_ref_space.SetSize(npts * dim);

      orig_ref_space.HostWrite();
      phys_space.HostWrite();
      elems.HostWrite();

      for (int i=0; i<npts; ++i)
      {
         elems[i] = 0;
         // Transform the integration point into space
         const IntegrationPoint& ip = intRule.IntPoint(i);
         tr.Transform(ip, v);

         real_t tmp[3];
         ip.Get(tmp, dim);
         for (int d = 0; d < dim; ++d)
         {
            orig_ref_space(i + d * npts) = tmp[d];
         }
         for (int d = 0; d < sdim; ++d)
         {
            phys_space(i + d * npts) = v(d);
         }
      }

      // now batch reverse transform
      itransform.Transform(phys_space, elems, res_type, res_ref_space);
      res_type.HostRead();
      res_ref_space.HostRead();
      int pts_found = 0;
      real_t max_err = 0;
      for (int i = 0; i < npts; ++i)
      {
         if (AsConst(res_type)[i] == InverseElementTransformation::Inside)
         {
            ++pts_found;
            for (int d = 0; d < dim; ++d)
            {
               max_err = fmax(max_err,
                              fabs(AsConst(res_ref_space)[i + d * npts] -
                                   orig_ref_space[i + d * npts]));
            }
         }
      }

      CAPTURE(pts_found, npts, max_err);
      REQUIRE( pts_found == npts );
      REQUIRE( max_err <= tol );
   }

   SECTION("{ C-shaped Q2 Quad }")
   {
      // Create quadratic with single C-shaped quadrilateral
      std::stringstream meshStr;
      meshStr << meshPrefixStr << CShapedNodesStr;
      Mesh mesh( meshStr );

      REQUIRE( mesh.GetNE() == 1 );
      REQUIRE( mesh.GetNodes() != nullptr );

      // Optionally, dump mesh to disk
      bool dumpMesh = false;
      if (dumpMesh)
      {
         std::string filename = "c_shaped_quadratic_mesh";
         VisItDataCollection dataCol(filename, &mesh);
         dataCol.Save();
      }

      const int times = 100;
      const int dim = mesh.Dimension();
      const int sdim = mesh.SpaceDimension();

      REQUIRE( dim == 2 );
      REQUIRE( sdim == 2 );

      // Create a uniform grid of integration points over the element
      const int geom = mesh.GetElementBaseGeometry(0);
      RefinedGeometry* ref =
         GlobGeometryRefiner.Refine(Geometry::Type(geom), times);
      const IntegrationRule& intRule = ref->RefPts;

      // Create a transformation
      IsoparametricTransformation tr;
      mesh.GetElementTransformation(0, &tr);
      Vector v(dim);

      const int npts = intRule.GetNPoints();
      Vector orig_ref_space;
      Vector phys_space;
      Array<int> elems;
      Array<int> res_type;
      Vector res_ref_space;

      BatchInverseElementTransformation itransform(mesh);
      // itransform.SetInitialGuessType(InverseElementTransformation::EdgeScan);
      // itransform.SetInitGuessRelOrder(3);
      // itransform.SetInitGuessPointsType(Quadrature1D::ClosedUniform);

      orig_ref_space.SetSize(npts * dim);
      phys_space.SetSize(npts * sdim);
      elems.SetSize(npts);
      res_type.SetSize(npts);
      res_ref_space.SetSize(npts * dim);

      orig_ref_space.HostWrite();
      phys_space.HostWrite();
      elems.HostWrite();

      for (int i=0; i<npts; ++i)
      {
         elems[i] = 0;
         // Transform the integration point into space
         const IntegrationPoint& ip = intRule.IntPoint(i);
         tr.Transform(ip, v);

         real_t tmp[3];
         ip.Get(tmp, dim);
         for (int d = 0; d < dim; ++d)
         {
            orig_ref_space(i + d * npts) = tmp[d];
            phys_space(i + d * npts) = v(d);
         }

         for (int d = 0; d < sdim; ++d)
         {
            phys_space(i + d * npts) = v(d);
         }
      }

      // now batch reverse transform
      itransform.Transform(phys_space, elems, res_type, res_ref_space);
      res_type.HostRead();
      res_ref_space.HostRead();
      int pts_found = 0;
      real_t max_err = 0;
      for (int i = 0; i < npts; ++i)
      {
         if (AsConst(res_type)[i] == InverseElementTransformation::Inside)
         {
            ++pts_found;
            for (int d = 0; d < dim; ++d)
            {
               max_err = fmax(max_err,
                              fabs(AsConst(res_ref_space)[i + d * npts] -
                                   orig_ref_space[i + d * npts]));
            }
         }
      }

      CAPTURE(pts_found, npts, max_err);
      REQUIRE( pts_found == npts );
      REQUIRE( max_err <= tol );
   }

   SECTION("{ C-shaped Q2 Quad 3D }")
   {
      // Create quadratic with single C-shaped quadrilateral
      std::stringstream meshStr;
      meshStr << meshPrefixStr << EmbCShapedNodesStr;
      Mesh mesh( meshStr );

      REQUIRE( mesh.GetNE() == 1 );
      REQUIRE( mesh.GetNodes() != nullptr );

      // Optionally, dump mesh to disk
      bool dumpMesh = false;
      if (dumpMesh)
      {
         std::string filename = "emb_c_shaped_quadratic_mesh";
         VisItDataCollection dataCol(filename, &mesh);
         dataCol.Save();
      }

      const int times = 100;
      const int dim = mesh.Dimension();
      const int sdim = mesh.SpaceDimension();

      REQUIRE( dim == 2 );
      REQUIRE( sdim == 3 );

      // Create a uniform grid of integration points over the element
      const int geom = mesh.GetElementBaseGeometry(0);
      RefinedGeometry* ref =
         GlobGeometryRefiner.Refine(Geometry::Type(geom), times);
      const IntegrationRule& intRule = ref->RefPts;

      // Create a transformation
      IsoparametricTransformation tr;
      mesh.GetElementTransformation(0, &tr);
      Vector v(dim);

      const int npts = intRule.GetNPoints();
      Vector orig_ref_space;
      Vector phys_space;
      Array<int> elems;
      Array<int> res_type;
      Vector res_ref_space;

      BatchInverseElementTransformation itransform(mesh);
      itransform.SetInitialGuessType(InverseElementTransformation::Center);

      orig_ref_space.SetSize(npts * dim);
      phys_space.SetSize(npts * sdim);
      elems.SetSize(npts);
      res_type.SetSize(npts);
      res_ref_space.SetSize(npts * dim);

      orig_ref_space.HostWrite();
      phys_space.HostWrite();
      elems.HostWrite();

      for (int i=0; i<npts; ++i)
      {
         elems[i] = 0;
         // Transform the integration point into space
         const IntegrationPoint& ip = intRule.IntPoint(i);
         tr.Transform(ip, v);

         real_t tmp[3];
         ip.Get(tmp, dim);
         for (int d = 0; d < dim; ++d)
         {
            orig_ref_space(i + d * npts) = tmp[d];
         }
         for (int d = 0; d < sdim; ++d)
         {
            phys_space(i + d * npts) = v(d);
         }
      }

      // now batch reverse transform
      itransform.Transform(phys_space, elems, res_type, res_ref_space);
      res_type.HostRead();
      res_ref_space.HostRead();
      int pts_found = 0;
      real_t max_err = 0;
      for (int i = 0; i < npts; ++i)
      {
         if (AsConst(res_type)[i] == InverseElementTransformation::Inside)
         {
            ++pts_found;
            for (int d = 0; d < dim; ++d)
            {
               max_err = fmax(max_err,
                              fabs(AsConst(res_ref_space)[i + d * npts] -
                                   orig_ref_space[i + d * npts]));
            }
         }
      }

      CAPTURE(pts_found, npts, max_err);
      REQUIRE( pts_found == npts );
      REQUIRE( max_err <= tol );
   }

   SECTION("{ Spiral Q20 Quad }")
   {
      // Load the spiral mesh from file:
      std::ifstream mesh_file("./data/quad-spiral-q20.mesh");
      REQUIRE( mesh_file.good() );

      const int npts = 100; // number of random points to test
      const int rand_seed = 189548;
      srand(rand_seed);

      Mesh mesh(mesh_file);
      REQUIRE( mesh.Dimension() == 2 );
      REQUIRE( mesh.SpaceDimension() == 2 );
      REQUIRE( mesh.GetNE() == 1 );

      int dim = mesh.Dimension();
      Vector orig_ref_space;
      Vector phys_space;
      Array<int> elems;
      Array<int> res_type;
      Vector res_ref_space;

      BatchInverseElementTransformation itransform(mesh);
      itransform.SetInitialGuessType(InverseElementTransformation::EdgeScan);
      itransform.SetInitGuessOrder(3);
      itransform.SetInitGuessPointsType(Quadrature1D::ClosedUniform);

      orig_ref_space.SetSize(npts * dim);
      phys_space.SetSize(npts * dim);
      elems.SetSize(npts);
      res_type.SetSize(npts);
      res_ref_space.SetSize(npts * dim);

      orig_ref_space.HostWrite();
      phys_space.HostWrite();
      elems.HostWrite();

      ElementTransformation &T = *mesh.GetElementTransformation(0);
      IntegrationPoint ip;
      Vector pt;
      pt.SetSize(dim);

      for (int i = 0; i < npts; i++)
      {
         elems[i] = 0;
         Geometry::GetRandomPoint(T.GetGeometryType(), ip);
         T.Transform(ip, pt);

         real_t tmp[3];
         ip.Get(tmp, dim);
         for (int d = 0; d < dim; ++d)
         {
            orig_ref_space(i + d * npts) = tmp[d];
            phys_space(i + d * npts) = pt(d);
         }
      }

      // now batch reverse transform
      itransform.Transform(phys_space, elems, res_type, res_ref_space);
      res_type.HostRead();
      res_ref_space.HostReadWrite();
      res_ref_space -= orig_ref_space;
      res_ref_space.HostRead();
      int pts_found = 0;
      real_t max_err = 0;
      for (int i = 0; i < npts; ++i)
      {
         if (AsConst(res_type)[i] == InverseElementTransformation::Inside)
         {
            ++pts_found;
            for (int d = 0; d < dim; ++d)
            {
               max_err = fmax(max_err,
                              fabs(AsConst(res_ref_space)[i + d * npts]));
            }
         }
      }

      CAPTURE(pts_found, npts, max_err);
      REQUIRE( pts_found == npts );
      REQUIRE( max_err <= tol );
   }

   SECTION("{ Multi Spiral Q20 Quad }")
   {
      // Load the spiral mesh from file:
      std::ifstream mesh_file("./data/quad-spiral-q20.mesh");
      REQUIRE( mesh_file.good() );

      const int npts = 100; // number of random points to test
      const int rand_seed = 189548;
      srand(rand_seed);

      Mesh mesh(mesh_file);
      mesh.UniformRefinement();
      mesh.UniformRefinement();
      mesh.UniformRefinement();
      REQUIRE( mesh.Dimension() == 2 );
      REQUIRE( mesh.SpaceDimension() == 2 );
      REQUIRE(mesh.GetNE() == 4 * 4 * 4);

      std::mt19937 gen(rand_seed);
      std::uniform_int_distribution<int> distr(0, mesh.GetNE() - 1);

      int dim = mesh.Dimension();
      Vector orig_ref_space;
      Vector phys_space;
      Array<int> elems;
      Array<int> res_type;
      Vector res_ref_space;

      BatchInverseElementTransformation itransform(mesh);
      itransform.SetInitialGuessType(InverseElementTransformation::EdgeScan);
      itransform.SetInitGuessRelOrder(3 - 20);
      itransform.SetInitGuessPointsType(Quadrature1D::ClosedUniform);

      orig_ref_space.SetSize(npts * dim);
      phys_space.SetSize(npts * dim);
      elems.SetSize(npts);
      res_type.SetSize(npts);
      res_ref_space.SetSize(npts * dim);

      orig_ref_space.HostWrite();
      phys_space.HostWrite();
      elems.HostWrite();

      IntegrationPoint ip;
      Vector pt;
      pt.SetSize(dim);

      for (int i = 0; i < npts; i++)
      {
         elems[i] = distr(gen);
         ElementTransformation &T = *mesh.GetElementTransformation(elems[i]);
         Geometry::GetRandomPoint(T.GetGeometryType(), ip);
         T.Transform(ip, pt);

         real_t tmp[3];
         ip.Get(tmp, dim);
         for (int d = 0; d < dim; ++d)
         {
            orig_ref_space(i + d * npts) = tmp[d];
            phys_space(i + d * npts) = pt(d);
         }
      }

      // now batch reverse transform
      itransform.Transform(phys_space, elems, res_type, res_ref_space);
      res_type.HostRead();
      res_ref_space.HostReadWrite();
      res_ref_space -= orig_ref_space;
      res_ref_space.HostRead();
      int pts_found = 0;
      real_t max_err = 0;
      for (int i = 0; i < npts; ++i)
      {
         if (AsConst(res_type)[i] == InverseElementTransformation::Inside)
         {
            ++pts_found;
            for (int d = 0; d < dim; ++d)
            {
               max_err = fmax(max_err,
                              fabs(AsConst(res_ref_space)[i + d * npts]));
            }
         }
      }

      CAPTURE(pts_found, npts, max_err);
      REQUIRE( pts_found == npts );
      REQUIRE( max_err <= tol );
   }

   SECTION("{ 3D Spiral }")
   {
      // Load the spiral mesh from file:
      std::ifstream mesh_file("./data/spiral_3D_p9.mesh");
      REQUIRE( mesh_file.good() );

      const int npts = 100; // number of random points to test
      const int rand_seed = 189548;
      srand(rand_seed);

      Mesh mesh(mesh_file);
      REQUIRE( mesh.Dimension() == 3 );
      REQUIRE( mesh.SpaceDimension() == 3 );
      REQUIRE( mesh.GetNE() == 1 );

      int dim = mesh.Dimension();
      Vector orig_ref_space;
      Vector phys_space;
      Array<int> elems;
      Array<int> res_type;
      Vector res_ref_space;

      BatchInverseElementTransformation itransform(mesh);

      orig_ref_space.SetSize(npts * dim);
      phys_space.SetSize(npts * dim);
      elems.SetSize(npts);
      res_type.SetSize(npts);
      res_ref_space.SetSize(npts * dim);

      orig_ref_space.HostWrite();
      phys_space.HostWrite();
      elems.HostWrite();

      ElementTransformation &T = *mesh.GetElementTransformation(0);
      IntegrationPoint ip;
      Vector pt;
      pt.SetSize(dim);

      for (int i = 0; i < npts; i++)
      {
         elems[i] = 0;
         Geometry::GetRandomPoint(T.GetGeometryType(), ip);
         T.Transform(ip, pt);

         real_t tmp[3];
         ip.Get(tmp, dim);
         for (int d = 0; d < dim; ++d)
         {
            orig_ref_space(i + d * npts) = tmp[d];
            phys_space(i + d * npts) = pt(d);
         }
      }

      // now batch reverse transform
      itransform.Transform(phys_space, elems, res_type, res_ref_space);
      res_type.HostRead();
      res_ref_space.HostReadWrite();
      res_ref_space -= orig_ref_space;
      res_ref_space.HostRead();
      int pts_found = 0;
      real_t max_err = 0;
      for (int i = 0; i < npts; ++i)
      {
         if (AsConst(res_type)[i] == InverseElementTransformation::Inside)
         {
            ++pts_found;
            for (int d = 0; d < dim; ++d)
            {
               max_err = fmax(max_err,
                              fabs(AsConst(res_ref_space)[i + d * npts]));
            }
         }
      }

      CAPTURE(pts_found, npts, max_err);
      REQUIRE( pts_found == npts );
      REQUIRE( max_err <= tol );
   }

   SECTION("{ Multi 3D Spiral }")
   {
      // Load the spiral mesh from file:
      std::ifstream mesh_file("./data/spiral_3D_p9.mesh");
      REQUIRE( mesh_file.good() );

      const int npts = 100; // number of random points to test
      const int rand_seed = 189548;
      srand(rand_seed);

      Mesh mesh(mesh_file);
      mesh.UniformRefinement();
      mesh.UniformRefinement();
      mesh.UniformRefinement();
      REQUIRE( mesh.Dimension() == 3 );
      REQUIRE( mesh.SpaceDimension() == 3 );
      REQUIRE( mesh.GetNE() == 8*8*8 );

      std::mt19937 gen(rand_seed);
      std::uniform_int_distribution<int> distr(0, mesh.GetNE() - 1);

      int dim = mesh.Dimension();
      Vector orig_ref_space;
      Vector phys_space;
      Array<int> elems;
      Array<int> res_type;
      Vector res_ref_space;

      BatchInverseElementTransformation itransform(mesh);

      orig_ref_space.SetSize(npts * dim);
      phys_space.SetSize(npts * dim);
      elems.SetSize(npts);
      res_type.SetSize(npts);
      res_ref_space.SetSize(npts * dim);

      orig_ref_space.HostWrite();
      phys_space.HostWrite();
      elems.HostWrite();

      IntegrationPoint ip;
      Vector pt;
      pt.SetSize(dim);

      for (int i = 0; i < npts; i++)
      {
         elems[i] = distr(gen);
         ElementTransformation &T = *mesh.GetElementTransformation(elems[i]);
         Geometry::GetRandomPoint(T.GetGeometryType(), ip);
         T.Transform(ip, pt);

         real_t tmp[3];
         ip.Get(tmp, dim);
         for (int d = 0; d < dim; ++d)
         {
            orig_ref_space(i + d * npts) = tmp[d];
            phys_space(i + d * npts) = pt(d);
         }
      }

      // now batch reverse transform
      itransform.Transform(phys_space, elems, res_type, res_ref_space);
      res_type.HostRead();
      res_ref_space.HostReadWrite();
      res_ref_space -= orig_ref_space;
      res_ref_space.HostRead();
      int pts_found = 0;
      real_t max_err = 0;
      for (int i = 0; i < npts; ++i)
      {
         if (AsConst(res_type)[i] == InverseElementTransformation::Inside)
         {
            ++pts_found;
            for (int d = 0; d < dim; ++d)
            {
               max_err = fmax(max_err,
                              fabs(AsConst(res_ref_space)[i + d * npts]));
            }
         }
      }

      CAPTURE(pts_found, npts, max_err);
      REQUIRE( pts_found == npts );
      REQUIRE( max_err <= tol );
   }
}
