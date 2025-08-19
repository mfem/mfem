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

TEST_CASE("NURBS mesh reconstruction", "[NURBS]")
{
   auto mesh_fname =
      GENERATE("../../data/segment-nurbs.mesh",
               "../../data/square-nurbs.mesh",
               "../../data/beam-quad-nurbs.mesh",
               "../../data/pipe-nurbs.mesh",
               "../../miniapps/nurbs/meshes/two-squares-nurbs.mesh",
               "../../miniapps/nurbs/meshes/two-squares-nurbs-rot.mesh",
               "../../miniapps/nurbs/meshes/two-squares-nurbs-autoedge.mesh",
               "../../miniapps/nurbs/meshes/plus-nurbs.mesh",
               "../../miniapps/nurbs/meshes/plus-nurbs-permuted.mesh",
               "../../miniapps/nurbs/meshes/ijk-hex-nurbs.mesh");

   Mesh mesh1(mesh_fname, 1, 1);

   // Reconstruct mesh using patches + topology
   Array<NURBSPatch*> patches;
   mesh1.GetNURBSPatches(patches);
   const Mesh patchtopo = mesh1.NURBSext->GetPatchTopology();

   NURBSExtension ne(&patchtopo, patches);
   Mesh mesh2(ne);

   // Meshes should be identical
   REQUIRE(mesh1.GetNodes()->Size() > 0);
   REQUIRE(mesh1.GetNodes()->Size() == mesh2.GetNodes()->Size());

   Vector diff(*mesh1.GetNodes());
   diff -= *mesh2.GetNodes();
   const real_t error = diff.Norml2();
   REQUIRE(error == MFEM_Approx(0.0));

   // Compare weights (these are stored separately from nodes)
   REQUIRE(mesh1.NURBSext->GetWeights().Size() > 0);
   REQUIRE(mesh1.NURBSext->GetWeights().Size() ==
           mesh2.NURBSext->GetWeights().Size());

   Vector wdiff = mesh1.NURBSext->GetWeights();
   wdiff -= mesh2.NURBSext->GetWeights();
   const real_t werror = wdiff.Norml2();
   REQUIRE(werror == MFEM_Approx(0.0));

   // Cleanup
   for (auto *p : patches) { delete p; }
}

TEST_CASE("NURBS NC-patch mesh loading", "[NURBS]")
{
   auto mesh_fname = GENERATE("../../data/nc3-nurbs.mesh",
                              "../../data/nc-nurbs3d.mesh");

   Mesh mesh(mesh_fname, 1, 1);
   const int dim = mesh.Dimension();
   const int ne = dim == 2 ? 6 : 24;
   REQUIRE(mesh.GetNE() == ne);

   mesh.NURBSUniformRefinement();
   REQUIRE(mesh.GetNE() == ne * std::pow(2, dim));
}

TEST_CASE("NURBS NC-patch large meshes", "[MFEMData][NURBS]")
{
   auto mesh_fname = GENERATE("bricks2D.mesh",
                              "schwarz2D.mesh",
                              "schwarz3D.mesh");

   const std::string & fpath = (mfem_data_dir + "/nurbs/nc_patch/");

   Mesh mesh(fpath + mesh_fname, 1, 1);
   const int dim = mesh.Dimension();
   const int ne = mesh.GetNE();

   mesh.NURBSUniformRefinement();
   REQUIRE(mesh.GetNE() == ne * std::pow(2, dim));
}
