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
