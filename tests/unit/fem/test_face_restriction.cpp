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


TEST_CASE("2D RT Face Restriction", "[FaceRestriction]")
{
   const int nx = 3;
   const int order = 4;

   Mesh mesh = Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL, true);
   const int dim = mesh.Dimension();

   RT_FECollection fec(order-1, dim);
   FiniteElementSpace fes(&mesh, &fec);

   auto ordering = ElementDofOrdering::LEXICOGRAPHIC;
   auto ftype = FaceType::Boundary;
   const FaceRestriction *face_restr =
      fes.GetFaceRestriction(ordering, ftype);

   REQUIRE(face_restr != nullptr);

   GridFunction gf(&fes);
   VectorFunctionCoefficient coeff(dim, [](const Vector &xvec, Vector &v)
   {
      v[0] = 2*(xvec[0] - 0.5);
      v[1] = 20*(xvec[1] - 0.5);
      // v[0] = 1.0;
      // v[1] = 2.0;
   });

   Array<int> attr;
   if (mesh.bdr_attributes.Size())
   {
      attr.SetSize(mesh.bdr_attributes.Max());
      attr = 1;
   }

   gf = 0.0;
   gf.ProjectBdrCoefficient(coeff, attr);

   Vector face_vec(face_restr->Height());
   face_restr->Mult(gf, face_vec);

   const int nfaces = fes.GetNFbyType(ftype);
   REQUIRE(face_vec.Size() == nfaces*order);

   // for (int f = 0; f < nfaces; ++f)
   // {
   //    std::cout << "Face " << f << ":\n";
   //    for (int i = 0; i < order; ++i)
   //    {
   //       std::cout << "    " << face_vec[f*order + i] << '\n';
   //    }
   // }
   // std::cout << '\n';

   GridFunction gf2(&fes);
   gf2 = 0.0;
   face_restr->MultTranspose(face_vec, gf2);

   // for (int i = 0; i < gf.Size(); ++i)
   // {
   //    printf("%8.2f   %8.2f    %8.2f\n", gf[i], gf2[i], gf[i]-gf2[i]);
   // }

   gf2 -= gf;
   REQUIRE(gf2.Normlinf() == MFEM_Approx(0.0));
}
