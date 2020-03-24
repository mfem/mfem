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

using namespace mfem;

namespace tet_reorder
{

TEST_CASE("Tetrahedron Reordering")
{
   typedef Geometry::Constants<Geometry::TETRAHEDRON> g_const;

   int p = 7;
   double tol = 1e-6;

   SECTION("Geometry order " + std::to_string(p))
   {
      for (int nd=0; nd<=1; nd++)
         SECTION("Discontinuous flag set to " + std::to_string(nd))
      {
         for (int no=0; no<=1; no++)
            SECTION("VDim ordering type set to " + std::to_string(no))
         {
            for (int o=0; o<g_const::NumOrient; o++)
               SECTION("Initial orientation set to " + std::to_string(o))
            {
               Mesh mesh(3, 4, 1);

               double c[3];
               c[0] = 0.0; c[1] = 0.0; c[2] = 3.0;
               mesh.AddVertex(c);
               c[0] = 0.0; c[1] = 2.0; c[2] = 0.0;
               mesh.AddVertex(c);
               c[0] = 1.0; c[1] = 0.0; c[2] = 0.0;
               mesh.AddVertex(c);
               c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
               mesh.AddVertex(c);

               const int * vo = g_const::Orient[o];
               mesh.AddTet(vo);
               mesh.FinalizeMesh(0, false);

               mesh.SetCurvature(p, nd, 3, no);
               double vol0 = mesh.GetElementVolume(0);
               REQUIRE(fabs(vol0 - 1.0 + 2.0 * (o % 2)) < tol);

               mesh.Finalize(true, true);
               double vol1 = mesh.GetElementVolume(0);
               REQUIRE(fabs(vol1 - 1.0) < tol);
            }
         }
      }
   }
}

} // namespace tet_reorder
