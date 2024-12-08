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
