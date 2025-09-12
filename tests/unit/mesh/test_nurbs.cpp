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

TEST_CASE("Location conversion check", "[NURBS]")
{

   KnotVector kv(3, Vector({0.0,
                            0.2,0.2,0.2,
                            0.5,0.5,0.5,
                            0.8,0.8,0.8,
                            1.0}));

   mfem::out<<"knotvector : ";
   kv.Print(mfem::out);

   constexpr int samples = 31;
   for (int i = 0; i < samples; i++)
   {
      const real_t u = i/real_t(samples-1);
      const int ks = kv.GetSpan (u);
      REQUIRE( ((kv[ks] <= u) && (u <= kv[ks+1])) );
      const real_t xi = kv.GetRefPoint(u, ks);
      REQUIRE( ((0.0 <= xi) && (xi <= 1.0)) );
      const real_t un = kv.GetKnotLocation(xi,ks);
      REQUIRE((un - u) == MFEM_Approx(0.0));

      mfem::out<<i<<" : "<<ks<<" ";
      mfem::out<<kv[ks] <<" "<<u<<" "<<kv[ks+1]<<" : ";
      mfem::out<<u<<" "<<un<<" = "<<un -u<<std::endl;
   }

   for (int i = 0; i < kv.Size(); i++)
   {
      const real_t u = kv[i];
      const int ks = kv.GetSpan (u);
      REQUIRE( ((kv[ks] <= u) && (u <= kv[ks+1])) );
      const real_t xi = kv.GetRefPoint(u, ks);
      REQUIRE( ((0.0 <= xi) && (xi <= 1.0)) );
      const real_t un = kv.GetKnotLocation(xi,ks);
      REQUIRE((un - u) == MFEM_Approx(0.0));

      mfem::out<<i<<" : "<<ks<<" ";
      mfem::out<<kv[ks] <<" "<<u<<" "<<kv[ks+1]<<" : ";
      mfem::out<<u<<" "<<un<<" = "<<un -u<<std::endl;
   }

   KnotVector kv2(1, Vector({0.0, 1.0/3.0, 2.0/3.0, 1.0}));
   mfem::out<<"knotvector2 : ";
   kv2.Print(mfem::out);

   for (int i = 0; i < samples; i++)
   {
      const real_t u = i/real_t(samples-1);
      const int ks = kv2.GetSpan (u);
      REQUIRE( ((kv2[ks] <= u) && (u <= kv2[ks+1])) );
      const real_t xi = kv2.GetRefPoint(u, ks);
      REQUIRE( ((0.0 <= xi) && (xi <= 1.0)) );
      const real_t un = kv2.GetKnotLocation(xi,ks);
      REQUIRE((un - u) == MFEM_Approx(0.0));

      mfem::out<<i<<" : "<<ks<<" ";
      mfem::out<<kv2[ks] <<" "<<u<<" "<<kv2[ks+1]<<" : ";
      mfem::out<<u<<" "<<un<<" = "<<un -u<<std::endl;
   }

   for (int i = 0; i < kv2.Size(); i++)
   {
      const real_t u = kv2[i];
      const int ks = kv2.GetSpan (u);
      REQUIRE( ((kv2[ks] <= u) && (u <= kv2[ks+1])) );
      const real_t xi = kv2.GetRefPoint(u, ks);
      REQUIRE( ((0.0 <= xi) && (xi <= 1.0)) );
      const real_t un = kv2.GetKnotLocation(xi,ks);
      REQUIRE((un - u) == MFEM_Approx(0.0));

      mfem::out<<i<<" : "<<ks<<" ";
      mfem::out<<kv2[ks] <<" "<<u<<" "<<kv2[ks+1]<<" : ";
      mfem::out<<u<<" "<<un<<" = "<<un -u<<std::endl;
   }

}

TEST_CASE("Greville, Botella and Demko points", "[NURBS]")
{

   Vector xi;
   for ( int p = 1; p <= 9; p++)
   {
      mfem::out<<"Order : "<<p<<std::endl;
      KnotVector kvp(p, Vector({0., 1.}));
      mfem::out<<"Knotvector : "; kvp.Print(mfem::out);

      kvp.GetGreville(xi);
      mfem::out<<"Greville points : "; xi.Print(std::cout,999);

      kvp.GetBotella(xi);
      mfem::out<<"Botella  points : "; xi.Print(std::cout,999);

      kvp.GetDemko(xi);
      mfem::out<<"Demko    points : "; xi.Print(std::cout,999);
   }

   KnotVector kv(3, Vector({0.0, 0.3, 0.3, 0.3, 0.6, 1.0}));

   mfem::out<<"Knotvector : "; kv.Print(mfem::out);

   // Greville
   Vector greville(kv.GetNCP());
   for (int i = 0; i < kv.GetNCP(); i++)
   {
      greville[i] = kv.GetGreville(i);
   }
   mfem::out<<"Greville points : "; greville.Print(mfem::out, 32);

   Vector gref({0.0,0.1,0.2,0.3,0.4,19./30,26./30, 1.0});
   for (int i = 0; i < kv.GetNCP(); i++)
   {
      REQUIRE((greville[i] - gref[i]) == MFEM_Approx(0.0));
   }

   // Botella
   Vector botella(kv.GetNCP());
   for (int i = 0; i < kv.GetNCP(); i++)
   {
      botella[i] = kv.GetBotella(i);
   }
   mfem::out<<"Botella  points : "; botella.Print(mfem::out, 32);

   Vector bref({0.0,0.1,0.2,0.3,
                0.444007481526490333,
                0.626666666666666594,
                0.828131261741523739, 1.0});
   for (int i = 0; i < kv.GetNCP(); i++)
   {
      REQUIRE((botella[i] - bref[i]) == MFEM_Approx(0.0));
   }

   // Demko
   Vector demko(kv.GetNCP());
   for (int i = 0; i < kv.GetNCP(); i++)
   {
      demko[i] = kv.GetDemko(i);
   }
   mfem::out<<"Demko    points : "; demko.Print(mfem::out, 32);

   Vector dref({0.0,0.075,0.225,0.3,
                0.406122105546614987,
                0.621569465634039919,
                0.87385648854468001,1.0});
   for (int i = 0; i < kv.GetNCP(); i++)
   {
      REQUIRE((demko[i] - dref[i]) == MFEM_Approx(0.0,1e-9,1e-9));
   }

   // Chebyshev spline
   Vector a(kv.GetNCP());
   Vector x(kv.GetNCP());
   for ( int i = 0; i <x.Size(); i++)
   {
      x[i] = std::pow(-1.0, i);
   }
   kv.GetInterpolant(x, demko, a);
   mfem::out<<"Chebyshev spline coeff : "; a.Print(mfem::out, 32);

   Vector aref({1.0, -5.0, 5.0, -1.0,
                3.24079982256718635,
                -5.51623733136825933,
                3.75648721902370486, -1.0});
   for (int i = 0; i < kv.GetNCP(); i++)
   {
      REQUIRE((a[i] - aref[i]) == MFEM_Approx(0.0));
   }

   mfem::out<<"Chebyshev spline \n";
   kv.PrintFunction(mfem::out, a, 21);

}
