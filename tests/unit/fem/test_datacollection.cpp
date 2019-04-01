// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mfem.hpp"
#include "catch.hpp"
#include <stdio.h>
#include <unistd.h>  // rmdir

using namespace mfem;

TEST_CASE("Visit data collection for input and output", "[VisItDataCollection]")
{
   SECTION("Save and load visit files from mesh and fields")
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

      //Compare some collection parameter for old and new
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
}
