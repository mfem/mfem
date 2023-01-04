// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

using namespace mfem;

Mesh MakeCartesianMesh(int nx, int dim)
{
   if (dim == 2)
   {
      return Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL, true);
   }
   else
   {
      return Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON);
   }
}

TEST_CASE("RT Face Restriction", "[FaceRestriction]")
{
   const int dim = GENERATE(2, 3);
   const int nx = 3;
   const int order = 4;

   CAPTURE(dim);

   Mesh mesh = MakeCartesianMesh(nx, dim);

   RT_FECollection fec(order-1, dim);
   FiniteElementSpace fes(&mesh, &fec);

   auto ordering = ElementDofOrdering::LEXICOGRAPHIC;
   auto ftype = FaceType::Boundary;
   const int nfaces = fes.GetNFbyType(FaceType::Boundary);
   const int ndof_per_face = int(pow(order, dim-1));
   const FaceRestriction *face_restr =
      fes.GetFaceRestriction(ordering, ftype);

   REQUIRE(face_restr != nullptr);

   Array<int> bdr_dofs;
   fes.GetBoundaryTrueDofs(bdr_dofs);

   // Set gf to have random values on the boundary, zero on the interior
   GridFunction gf(&fes);
   gf.Randomize(0);
   gf.SetSubVectorComplement(bdr_dofs, 0.0);

   // Mapping to face E-vector and back to L-vector should give back the
   // original grid function.
   Vector face_vec(face_restr->Height());
   REQUIRE(face_vec.Size() == fes.GetNFbyType(ftype)*ndof_per_face);
   face_restr->Mult(gf, face_vec);
   GridFunction gf2(&fes);
   face_restr->MultTranspose(face_vec, gf2);

   gf2 -= gf;
   REQUIRE(gf2.Normlinf() == MFEM_Approx(0.0));
}
