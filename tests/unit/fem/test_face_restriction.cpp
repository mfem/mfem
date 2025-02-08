// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

namespace face_restriction_test { enum class SpaceType {RT, ND}; }

TEST_CASE("Vector FE Face Restriction", "[FaceRestriction]")
{
   using namespace face_restriction_test;

   const auto space_type = GENERATE(SpaceType::RT, SpaceType::ND);
   const int dim = GENERATE(2, 3);
   const int nx = 3;
   const int order = 4;

   CAPTURE(dim);

   Mesh mesh = MakeCartesianMesh(nx, dim);

   int ndof_per_face;
   std::unique_ptr<FiniteElementCollection> fec;
   if (space_type == SpaceType::RT)
   {
      fec.reset(new RT_FECollection(order-1, dim));
      ndof_per_face = int(pow(order, dim-1));
   }
   else
   {
      fec.reset(new ND_FECollection(order, dim));
      ndof_per_face = (dim - 1)*order*int(pow(order + 1, dim - 2));
   }

   FiniteElementSpace fes(&mesh, fec.get());

   auto ordering = ElementDofOrdering::LEXICOGRAPHIC;
   auto ftype = FaceType::Boundary;
   const int nfaces = fes.GetNFbyType(FaceType::Boundary);
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
   REQUIRE(face_vec.Size() == nfaces*ndof_per_face);
   face_restr->Mult(gf, face_vec);

   if (space_type == SpaceType::ND && dim == 3)
   {
      // Adjust for multiplicity. In all other cases, each boundary DOF is
      // unique (not shared between faces). In the case of 3D ND elements, some
      // boundary DOFs are shared between two faces (i.e. those that lie on
      // element edges).
      //
      // This adjustment will ensure that the original vector is recovered after
      // multiplying by the transpose of the face restriction operator.
      //
      // Note that this assumes the mesh contains only hexahedral elements.
      const int n = order*(order+1);
      for (int f = 0; f < fes.GetNFbyType(ftype); ++f)
      {
         for (int d = 0; d < 2; ++d)
         {
            const int mx = (d == 0) ? order : order + 1;
            const int my = (d == 0) ? order + 1 : order;
            for (int i = 0; i < n; ++i)
            {
               const int ix = i % mx;
               const int iy = i / mx;
               if ((d == 0 && (iy == 0 || iy == my - 1)) ||
                   (d == 1 && (ix == 0 || ix == mx - 1)))
               {
                  face_vec[f*ndof_per_face + d*n + i] *= 0.5;
               }
            }
         }
      }
   }

   GridFunction gf2(&fes);
   face_restr->MultTranspose(face_vec, gf2);

   gf2 -= gf;
   REQUIRE(gf2.Normlinf() == MFEM_Approx(0.0));
}
