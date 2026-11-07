// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
#include "linalg/dtensor.hpp"
#include "unit_tests.hpp"

using namespace mfem;

TEST_CASE("Geometric factor Jacobians", "[Mesh]")
{
   const auto mesh_fname = GENERATE(
                              "../../data/inline-segment.mesh",
                              "../../data/star.mesh",
                              "../../data/star-q3.mesh",
                              "../../data/fichera.mesh",
                              "../../data/fichera-q3.mesh",
                              "../../data/star-surf.mesh", // surface mesh
                              "../../data/square-disc-surf.mesh" // surface tri mesh
                           );
   CAPTURE(mesh_fname);

   Mesh mesh = Mesh::LoadFromFile(mesh_fname);
   const int order = 3;
   const auto &ir = IntRules.Get(mesh.GetTypicalElementGeometry(), order);
   auto *geom = mesh.GetGeometricFactors(ir, GeometricFactors::DETERMINANTS);
   geom->detJ.HostRead();

   const int nq = ir.Size();
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      auto &T = *mesh.GetElementTransformation(i);
      for (int iq = 0; iq < nq; ++iq)
      {
         T.SetIntPoint(&ir[iq]);
         REQUIRE(geom->detJ(iq + i*nq) == MFEM_Approx(T.Weight()));
      }
   }
}

TEST_CASE("Face geometric factor Jacobians", "[Mesh]")
{
   const auto mesh_fname = GENERATE("../../data/star.mesh",
                                    "../../data/star-q3.mesh",
                                    "../../data/fichera.mesh",
                                    "../../data/fichera-q3.mesh");

   Mesh mesh = Mesh::LoadFromFile(mesh_fname);
   const int dim = mesh.Dimension();
   const int order = 3;
   const auto &ir = IntRules.Get(mesh.GetFaceGeometry(0), order);

   const int nq = ir.Size();
   const int nq1d = (int)std::floor(std::pow(ir.GetNPoints(), 1.0/(dim-1))+0.5);
   const int nf = mesh.GetNFbyType(FaceType::Boundary);

   auto *geom = mesh.GetFaceGeometricFactors(
                   ir,
                   FaceGeometricFactors::JACOBIANS | FaceGeometricFactors::DETERMINANTS,
                   FaceType::Boundary);
   const auto J = Reshape(geom->J.HostRead(), nq, dim*(dim - 1), nf);

   int idx = 0;
   for (int f = 0; f < mesh.GetNumFaces(); ++f)
   {
      const auto finfo = mesh.GetFaceInformation(f);
      if (!finfo.IsBoundary()) { continue; }
      auto &T = *mesh.GetFaceElementTransformations(f);
      for (int iq = 0; iq < nq; ++iq)
      {
         const int iq_lex = ToLexOrdering(
                               dim, finfo.element[0].local_face_id, nq1d, iq);
         T.SetAllIntPoints(&ir[iq]);
         auto &Jac1 = T.Jacobian();
         DenseMatrix Jac2(dim, dim-1);
         for (int i = 0; i < dim*(dim - 1); ++i)
         {
            Jac2.GetData()[i] = J(iq_lex,i,idx);
         }
         REQUIRE(Jac1.Weight() == MFEM_Approx(Jac2.Weight()));
      }
      ++idx;
   }
}

TEST_CASE("Face geometric factors on tensor-product meshes", "[Mesh]")
{
   const auto dim = GENERATE(2, 3);
   const auto type = GENERATE(FaceType::Interior, FaceType::Boundary);
   CAPTURE(dim, static_cast<int>(type));
   Mesh mesh = dim == 2 ? Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL) :
               Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   const Geometry::Type face_geom = dim == 2 ? Geometry::SEGMENT :
                                    Geometry::SQUARE;
   const IntegrationRule &ir = IntRules.Get(face_geom, 2);
   const auto *factors = mesh.GetFaceGeometricFactors(
                            ir, FaceGeometricFactors::DETERMINANTS, type);
   REQUIRE(factors->detJ.Size() == ir.Size() * mesh.GetNFbyType(type));

   // A single hex has no interior faces; this remains a valid empty result.
   if (dim == 3 && type == FaceType::Interior)
   {
      Mesh single = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
      const auto *empty = single.GetFaceGeometricFactors(
                             ir, FaceGeometricFactors::DETERMINANTS, type);
      REQUIRE(single.GetNFbyType(type) == 0);
      REQUIRE(empty->detJ.Size() == 0);
   }
}

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("Face geometric factors reject unsupported nodal spaces", "[Mesh]")
{
   const auto mesh_type = GENERATE(0, 1, 2, 3, 4, 5);
   const auto type = GENERATE(FaceType::Interior, FaceType::Boundary);
   CAPTURE(mesh_type, static_cast<int>(type));
   Mesh mesh;
   Geometry::Type face_geom;
   if (mesh_type == 0)
   {
      mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE);
      face_geom = Geometry::SEGMENT;
   }
   else if (mesh_type == 1)
   {
      mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
      face_geom = Geometry::TRIANGLE;
   }
   else if (mesh_type == 2)
   {
      mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::WEDGE);
      face_geom = Geometry::SQUARE;
   }
   else if (mesh_type == 3)
   {
      mesh = Mesh::LoadFromFile("../../data/square-mixed.mesh");
      face_geom = Geometry::SEGMENT;
   }
   else if (mesh_type == 4)
   {
      mesh = Mesh(2, 3, 1, 3);
      mesh.AddVertex(0., 0.);
      mesh.AddVertex(1., 0.);
      mesh.AddVertex(0., 1.);
      mesh.AddTriangle(0, 1, 2);
      mesh.AddBdrSegment(0, 1);
      mesh.AddBdrSegment(1, 2);
      mesh.AddBdrSegment(2, 0);
      mesh.FinalizeTriMesh(1, 0, true);
      face_geom = Geometry::SEGMENT;
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::WEDGE);
      face_geom = Geometry::SQUARE;
   }
   const IntegrationRule &ir = IntRules.Get(face_geom, 0);
   REQUIRE_THROWS_WITH(
      mesh.GetFaceGeometricFactors(ir, FaceGeometricFactors::DETERMINANTS, type),
      Catch::Matchers::Contains("FaceGeometricFactors requires a homogeneous "
                                "tensor-product nodal finite element space"));
}
#endif

#ifdef MFEM_USE_MPI
TEST_CASE("Face geometric factors on empty partitions", "[Parallel][Mesh]")
{
   if (Mpi::WorldSize() < 3) { return; }

   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
   int partition[] = {0, 1};
   ParMesh pmesh(MPI_COMM_WORLD, mesh, partition);
   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, 2);
   const int flags = FaceGeometricFactors::COORDINATES |
                     FaceGeometricFactors::JACOBIANS |
                     FaceGeometricFactors::DETERMINANTS |
                     FaceGeometricFactors::NORMALS;
   for (auto type : {FaceType::Interior, FaceType::Boundary})
   {
      const auto *factors = pmesh.GetFaceGeometricFactors(ir, flags, type);
      const int count = ir.Size() * pmesh.GetNFbyType(type);
      REQUIRE(factors->X.Size() == 2 * count);
      REQUIRE(factors->J.Size() == 2 * count);
      REQUIRE(factors->detJ.Size() == count);
      REQUIRE(factors->normal.Size() == 2 * count);
      if (Mpi::WorldRank() >= 2)
      {
         REQUIRE(pmesh.GetNE() == 0);
         REQUIRE(count == 0);
      }
   }
}

#ifdef MFEM_USE_EXCEPTIONS
TEST_CASE("Face geometric factors on mixed parallel partitions",
          "[Parallel][Mesh]")
{
   if (Mpi::WorldSize() < 3) { return; }

   Mesh mesh = Mesh::LoadFromFile("../../data/square-mixed.mesh");
   Array<int> partition(mesh.GetNE());
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      partition[i] = mesh.GetElementGeometry(i) == Geometry::SQUARE ? 0 : 1;
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh, partition.GetData());
   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, 0);
   for (auto type : {FaceType::Interior, FaceType::Boundary})
   {
      if (Mpi::WorldRank() == 1)
      {
         REQUIRE_THROWS_WITH(
            pmesh.GetFaceGeometricFactors(ir, FaceGeometricFactors::DETERMINANTS, type),
            Catch::Matchers::Contains("FaceGeometricFactors requires a homogeneous "
                                      "tensor-product nodal finite element space"));
      }
      else
      {
         const auto *factors = pmesh.GetFaceGeometricFactors(
                                  ir, FaceGeometricFactors::DETERMINANTS, type);
         REQUIRE(factors->detJ.Size() == ir.Size() * pmesh.GetNFbyType(type));
         if (Mpi::WorldRank() >= 2) { REQUIRE(pmesh.GetNE() == 0); }
      }
   }
}
#endif
#endif
