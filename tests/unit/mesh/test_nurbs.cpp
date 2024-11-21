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
using namespace mfem;

#include "unit_tests.hpp"

TEST_CASE("NURBS knot insertion and removal", "[NURBS]")
{
   auto mesh_fname = "../../data/pipe-nurbs.mesh";
   Mesh mesh1(mesh_fname, 1, 1);
   Mesh mesh2(mesh_fname, 1, 1);

   Vector k0(1);
   Vector k1(1);
   Vector k2(1);

   k0[0] = 0.5;
   k1[0] = 0.5;
   k2[0] = 0.5;

   Array<Vector*> knots(3);
   knots[0] = &k0;
   knots[1] = &k1;
   knots[2] = &k2;

   mesh1.KnotInsert(knots);

   REQUIRE(mesh1.GetNodes()->Size() > mesh2.GetNodes()->Size());

   mesh1.KnotRemove(knots);

   // At this point, mesh1 and mesh2 should coincide. Verify this by comparing
   // their Nodes GridFunctions.
   REQUIRE(mesh1.GetNodes()->Size() == mesh2.GetNodes()->Size());

   Vector d(*mesh1.GetNodes());
   d -= *mesh2.GetNodes();
   const real_t error = d.Norml2();
   REQUIRE(error == MFEM_Approx(0.0));
}

TEST_CASE("NURBS refinement and coarsening by spacing formulas", "[NURBS]")
{
   auto mesh_fname = GENERATE("../../data/beam-quad-nurbs-sf.mesh",
                              "../../data/square-nurbs-pw.mesh");

   Mesh mesh1(mesh_fname, 1, 1);
   Mesh mesh2(mesh_fname, 1, 1);

   const bool beam = mesh1.GetNE() > 1;

   Array<int> rf(2);
   // [24, 12] works for beam mesh
   rf[0] = 24;
   rf[1] = beam ? 12 : 24;

   mesh1.NURBSUniformRefinement(rf);

   rf[0] = 12;
   rf[1] = beam ? 6 : 12;

   mesh2.NURBSUniformRefinement(rf);

   REQUIRE(mesh1.GetNodes()->Size() > mesh2.GetNodes()->Size());

   mesh1.NURBSCoarsening(2);

   // At this point, mesh1 and mesh2 should coincide. Verify this by comparing
   // their Nodes GridFunctions.
   REQUIRE(mesh1.GetNodes()->Size() == mesh2.GetNodes()->Size());

   Vector d(*mesh1.GetNodes());
   d -= *mesh2.GetNodes();
   const real_t error = d.Norml2();
   REQUIRE(error == MFEM_Approx(0.0));
}
