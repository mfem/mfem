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
#include "unit_tests.hpp"

namespace mfem
{

// Check complex orientation constraints for Nedelec elements
// of order >= 2 on triangular faces with orientations 1-4.

TEST_CASE("Nedelec arbitrary orientation tri faces",
          "[FiniteElementSpace]"
          "[NCMesh]")
{
   const int dim = 3;
   const int order = GENERATE(1, 2, 3);

   Mesh mesh(2, 2, 2, Element::TETRAHEDRON);
   mesh.EnsureNCMesh(true);
   // NOTE: no mesh.ReorientTetMesh()

   int ncomplex = 0;
   for (int i = 0; i < mesh.GetNumFaces(); i++)
   {
      int elem1, elem2, inf1, inf2;
      mesh.GetFaceElements(i, &elem1, &elem2);
      mesh.GetFaceInfos(i, &inf1, &inf2);

      int ori = inf2 % 64;
      if (elem2 >= 0 && ori >= 1 && ori <= 4)
      {
         //mfem::out << "Face orientation " << ori << " found\n";
         ncomplex++;
      }
   }

   // check that there are triangular faces with orientations 1-4 in the mesh
   // -- if there aren't any it's not an error per se, but the following
   // test won't work...
   REQUIRE(ncomplex > 0);

   ND_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   if (order < 2)
   {
      // no need for double faces at order 1
      REQUIRE(fespace.GetVSize() == fespace.GetTrueVSize());
   }
   else
   {
      // there are extra DOFs in the space due to the double faces
      REQUIRE(fespace.GetVSize() > fespace.GetTrueVSize());
   }
}

} // namespace mfem
