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
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h> // rmdir
#else
#include <direct.h> // _rmdir
#define rmdir(dir) _rmdir(dir)
#endif

using namespace mfem;

TEST_CASE("Save and load from collections", "[DataCollection]")
{
   SECTION("Visit data files")
   {
      //Set up a small mesh and a couple of grid function on that mesh
      Mesh *mesh = new Mesh(2, 3, Element::QUADRILATERAL, 0, 2.0, 3.0);
      FiniteElementCollection *fec = new LinearFECollection;
      FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
      GridFunction *u = new GridFunction(fespace);
      GridFunction *v = new GridFunction(fespace);

      int N = u->Size();
      for (int i = 0; i < N; ++i)
      {
         (*u)(i) = double(i);
         (*v)(i) = double(N - i - 1);
      }

      SECTION("Uncompressed MFEM format")
      {
         //Collect the mesh and grid functions into a DataCollection and test that they got in there
         VisItDataCollection dc("base", mesh);
         dc.RegisterField("u", u);
         dc.RegisterField("v", v);
         dc.SetCycle(5);
         dc.SetTime(8.0);
         REQUIRE(dc.GetMesh() == mesh );
         REQUIRE(dc.HasField("u"));
         REQUIRE(dc.HasField("v"));
         REQUIRE(dc.GetCycle() == 5);
         REQUIRE(dc.GetTime() == 8.0);

         //Save the DataCollection and load it into a new DataCollection for comparison
         dc.SetPadDigits(5);
         dc.Save();

         VisItDataCollection dc_new("base");
         dc_new.SetPadDigits(5);
         dc_new.Load(dc.GetCycle());
         Mesh* mesh_new = dc_new.GetMesh();
         GridFunction *u_new = dc_new.GetField("u");
         GridFunction *v_new = dc_new.GetField("v");
         REQUIRE(mesh_new);
         REQUIRE(u_new);
         REQUIRE(v_new);

         //Compare some collection parameters for old and new
         std::string name, name_new;
         name = dc.GetCollectionName();
         name_new = dc_new.GetCollectionName();
         REQUIRE(name == name_new);
         REQUIRE(dc.GetCycle() == dc_new.GetCycle());
         REQUIRE(dc.GetTime() == dc_new.GetTime());

         //Compare the new new mesh with the old mesh
         //(Just a basic comparison here, a full comparison should be done in Mesh unit testing)
         REQUIRE(mesh->Dimension() == mesh_new->Dimension());
         REQUIRE(mesh->SpaceDimension() == mesh_new->SpaceDimension());

         Vector vert, vert_diff;
         mesh->GetVertices(vert);
         mesh_new->GetVertices(vert_diff);
         vert_diff -= vert;
         REQUIRE(vert_diff.Normlinf() < 1e-10);

         //Compare the old and new grid functions
         //(Just a basic comparison here, a full comparison should be done in GridFunction unit testing)
         Vector u_diff(*u_new), v_diff(*v_new);
         u_diff -= *u;
         v_diff -= *v;
         REQUIRE(u_diff.Normlinf() < 1e-10);
         REQUIRE(v_diff.Normlinf() < 1e-10);

         //Cleanup all the files
         REQUIRE(remove("base_00005.mfem_root") == 0);
         REQUIRE(remove("base_00005/mesh.00000") == 0);
         REQUIRE(remove("base_00005/u.00000") == 0);
         REQUIRE(remove("base_00005/v.00000") == 0);
         REQUIRE(rmdir("base_00005") == 0);
      }

#ifdef MFEM_USE_ZLIB
      SECTION("Compressed MFEM format")
      {
         //Collect the mesh and grid functions into a DataCollection and test that they got in there
         VisItDataCollection dc("base", mesh);
         dc.RegisterField("u", u);
         dc.RegisterField("v", v);
         dc.SetCycle(5);
         dc.SetTime(8.0);
         REQUIRE(dc.GetMesh() == mesh );
         REQUIRE(dc.HasField("u"));
         REQUIRE(dc.HasField("v"));
         REQUIRE(dc.GetCycle() == 5);
         REQUIRE(dc.GetTime() == 8.0);

         //Save the DataCollection and load it into a new DataCollection for comparison
         dc.SetPadDigits(5);
         dc.SetCompression(true);
         dc.Save();

         VisItDataCollection dc_new("base");
         dc_new.SetPadDigits(5);
         dc_new.Load(dc.GetCycle());
         Mesh* mesh_new = dc_new.GetMesh();
         GridFunction *u_new = dc_new.GetField("u");
         GridFunction *v_new = dc_new.GetField("v");
         REQUIRE(mesh_new);
         REQUIRE(u_new);
         REQUIRE(v_new);

         //Compare some collection parameters for old and new
         std::string name, name_new;
         name = dc.GetCollectionName();
         name_new = dc_new.GetCollectionName();
         REQUIRE(name == name_new);
         REQUIRE(dc.GetCycle() == dc_new.GetCycle());
         REQUIRE(dc.GetTime() == dc_new.GetTime());

         //Compare the new new mesh with the old mesh
         //(Just a basic comparison here, a full comparison should be done in Mesh unit testing)
         REQUIRE(mesh->Dimension() == mesh_new->Dimension());
         REQUIRE(mesh->SpaceDimension() == mesh_new->SpaceDimension());

         Vector vert, vert_diff;
         mesh->GetVertices(vert);
         mesh_new->GetVertices(vert_diff);
         vert_diff -= vert;
         REQUIRE(vert_diff.Normlinf() < 1e-10);

         //Compare the old and new grid functions
         //(Just a basic comparison here, a full comparison should be done in GridFunction unit testing)
         Vector u_diff(*u_new), v_diff(*v_new);
         u_diff -= *u;
         v_diff -= *v;
         REQUIRE(u_diff.Normlinf() < 1e-10);
         REQUIRE(v_diff.Normlinf() < 1e-10);

         //Cleanup all the files
         REQUIRE(remove("base_00005.mfem_root") == 0);
         REQUIRE(remove("base_00005/mesh.00000") == 0);
         REQUIRE(remove("base_00005/u.00000") == 0);
         REQUIRE(remove("base_00005/v.00000") == 0);
         REQUIRE(rmdir("base_00005") == 0);
      }
#endif
   }
}
