// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
#include "general/tinyxml2.h"
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
   SECTION("VisIt data files")
   {
      std::cout<<"Testing VisIt data files"<<std::endl;
      // Set up a small mesh and a couple of grid function on that mesh
      Mesh mesh = Mesh::MakeCartesian2D(2, 3, Element::QUADRILATERAL, 0, 2.0, 3.0);
      FiniteElementCollection *fec = new LinearFECollection;
      FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, fec);
      GridFunction *u = new GridFunction(fespace);
      GridFunction *v = new GridFunction(fespace);

      int N = u->Size();
      for (int i = 0; i < N; ++i)
      {
         (*u)(i) = double(i);
         (*v)(i) = double(N - i - 1);
      }

      int intOrder = 3;

      QuadratureSpace *qspace = new QuadratureSpace(&mesh, intOrder);
      QuadratureFunction *qs = new QuadratureFunction(qspace, 1);
      QuadratureFunction *qv = new QuadratureFunction(qspace, 2);

      int Nq = qs->Size();
      for (int i = 0; i < Nq; ++i)
      {
         (*qs)(i) = double(i);
         (*qv)(2*i+0) = double(i);
         (*qv)(2*i+1) = double(Nq - i - 1);
      }


      SECTION("Uncompressed MFEM format")
      {
         std::cout<<"Testing uncompressed MFEM format"<<std::endl;

         // Collect the mesh and grid functions into a DataCollection and test that they got in there
         VisItDataCollection dc("base", &mesh);
         dc.RegisterField("u", u);
         dc.RegisterField("v", v);
         dc.RegisterQField("qs",qs);
         dc.RegisterQField("qv",qv);
         dc.SetCycle(5);
         dc.SetTime(8.0);
         REQUIRE(dc.GetMesh() == &mesh);
         bool has_u = dc.HasField("u");
         REQUIRE(has_u);
         bool has_v = dc.HasField("v");
         REQUIRE(has_v);
         bool has_qs = dc.HasQField("qs");
         REQUIRE(has_qs);
         bool has_qv = dc.HasQField("qv");
         REQUIRE(has_qv);
         REQUIRE(dc.GetCycle() == 5);
         REQUIRE(dc.GetTime() == 8.0);

         // Save the DataCollection and load it into a new DataCollection for comparison
         dc.SetPadDigits(5);
         dc.Save();

         VisItDataCollection dc_new("base");
         dc_new.SetPadDigits(5);
         dc_new.Load(dc.GetCycle());
         Mesh* mesh_new = dc_new.GetMesh();
         GridFunction *u_new = dc_new.GetField("u");
         GridFunction *v_new = dc_new.GetField("v");
         QuadratureFunction *qs_new = dc_new.GetQField("qs");
         QuadratureFunction *qv_new = dc_new.GetQField("qv");
         REQUIRE(mesh_new);
         REQUIRE(u_new);
         REQUIRE(v_new);
         REQUIRE(qs_new);
         REQUIRE(qv_new);

         // Compare some collection parameters for old and new
         std::string name, name_new;
         name = dc.GetCollectionName();
         name_new = dc_new.GetCollectionName();
         REQUIRE(name == name_new);
         REQUIRE(dc.GetCycle() == dc_new.GetCycle());
         REQUIRE(dc.GetTime() == dc_new.GetTime());

         // Compare the new mesh with the old mesh
         // (Just a basic comparison here, a full comparison should be done in Mesh unit testing)
         REQUIRE(mesh.Dimension() == mesh_new->Dimension());
         REQUIRE(mesh.SpaceDimension() == mesh_new->SpaceDimension());

         Vector vert, vert_diff;
         mesh.GetVertices(vert);
         mesh_new->GetVertices(vert_diff);
         vert_diff -= vert;
         REQUIRE(vert_diff.Normlinf() < 1e-10);

         // Compare the old and new grid functions
         // (Just a basic comparison here, a full comparison should be done in GridFunction unit testing)
         Vector u_diff(*u_new), v_diff(*v_new);
         u_diff -= *u;
         v_diff -= *v;
         REQUIRE(u_diff.Normlinf() < 1e-10);
         REQUIRE(v_diff.Normlinf() < 1e-10);

         // Compare the old and new quadrature functions
         // (Just a basic comparison here, a full comparison should be done in GridFunction unit testing)
         Vector qs_diff(*qs_new), qv_diff(*qv_new);
         qs_diff -= *qs;
         qv_diff -= *qv;
         REQUIRE(qs_diff.Normlinf() < 1e-10);
         REQUIRE(qv_diff.Normlinf() < 1e-10);

         // Cleanup all the files
         REQUIRE(remove("base_00005.mfem_root") == 0);
         REQUIRE(remove("base_00005/mesh.00000") == 0);
         REQUIRE(remove("base_00005/u.00000") == 0);
         REQUIRE(remove("base_00005/v.00000") == 0);
         REQUIRE(remove("base_00005/qs.00000") == 0);
         REQUIRE(remove("base_00005/qv.00000") == 0);
         REQUIRE(rmdir("base_00005") == 0);
      }

#ifdef MFEM_USE_ZLIB
      SECTION("Compressed MFEM format")
      {
         std::cout<<"Testing compressed MFEM format"<<std::endl;

         // Collect the mesh and grid functions into a DataCollection and test that they got in there
         VisItDataCollection dc("base", &mesh);
         dc.RegisterField("u", u);
         dc.RegisterField("v", v);
         dc.RegisterQField("qs",qs);
         dc.RegisterQField("qv",qv);
         dc.SetCycle(5);
         dc.SetTime(8.0);
         REQUIRE(dc.GetMesh() == &mesh);
         bool has_u = dc.HasField("u");
         REQUIRE(has_u);
         bool has_v = dc.HasField("v");
         REQUIRE(has_v);
         bool has_qs = dc.HasQField("qs");
         REQUIRE(has_qs);
         bool has_qv = dc.HasQField("qv");
         REQUIRE(has_qv);
         REQUIRE(dc.GetCycle() == 5);
         REQUIRE(dc.GetTime() == 8.0);

         // Save the DataCollection and load it into a new DataCollection for comparison
         dc.SetPadDigits(5);
         dc.SetCompression(true);
         dc.Save();

         VisItDataCollection dc_new("base");
         dc_new.SetPadDigits(5);
         dc_new.Load(dc.GetCycle());
         Mesh *mesh_new = dc_new.GetMesh();
         GridFunction *u_new = dc_new.GetField("u");
         GridFunction *v_new = dc_new.GetField("v");
         QuadratureFunction *qs_new = dc_new.GetQField("qs");
         QuadratureFunction *qv_new = dc_new.GetQField("qv");
         REQUIRE(mesh_new);
         REQUIRE(u_new);
         REQUIRE(v_new);
         REQUIRE(qs_new);
         REQUIRE(qv_new);

         // Compare some collection parameters for old and new
         std::string name, name_new;
         name = dc.GetCollectionName();
         name_new = dc_new.GetCollectionName();
         REQUIRE(name == name_new);
         REQUIRE(dc.GetCycle() == dc_new.GetCycle());
         REQUIRE(dc.GetTime() == dc_new.GetTime());

         // Compare the new mesh with the old mesh
         // (Just a basic comparison here, a full comparison should be done in Mesh unit testing)
         REQUIRE(mesh.Dimension() == mesh_new->Dimension());
         REQUIRE(mesh.SpaceDimension() == mesh_new->SpaceDimension());

         Vector vert, vert_diff;
         mesh.GetVertices(vert);
         mesh_new->GetVertices(vert_diff);
         vert_diff -= vert;
         REQUIRE(vert_diff.Normlinf() < 1e-10);

         // Compare the old and new grid functions
         // (Just a basic comparison here, a full comparison should be done in GridFunction unit testing)
         Vector u_diff(*u_new), v_diff(*v_new);
         u_diff -= *u;
         v_diff -= *v;
         REQUIRE(u_diff.Normlinf() < 1e-10);
         REQUIRE(v_diff.Normlinf() < 1e-10);

         // Compare the old and new quadrature functions
         // (Just a basic comparison here, a full comparison should be done in GridFunction unit testing)
         Vector qs_diff(*qs_new), qv_diff(*qv_new);
         qs_diff -= *qs;
         qv_diff -= *qv;
         REQUIRE(qs_diff.Normlinf() < 1e-10);
         REQUIRE(qv_diff.Normlinf() < 1e-10);

         // Cleanup all the files
         REQUIRE(remove("base_00005.mfem_root") == 0);
         REQUIRE(remove("base_00005/mesh.00000") == 0);
         REQUIRE(remove("base_00005/u.00000") == 0);
         REQUIRE(remove("base_00005/v.00000") == 0);
         REQUIRE(remove("base_00005/qs.00000") == 0);
         REQUIRE(remove("base_00005/qv.00000") == 0);
         REQUIRE(rmdir("base_00005") == 0);
      }
#endif
   }

}

void SaveDataCollection(DataCollection &dc, int cycle, double t)
{
   dc.SetCycle(cycle);
   dc.SetTime(t);
   dc.Save();
}

TEST_CASE("ParaView restart mode", "[ParaView]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 3, Element::QUADRILATERAL);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction u(&fes);
   u = 0.0;

   // Write initial dataset with three timesteps: 0, 1, 2.
   {
      ParaViewDataCollection dc("ParaView", &mesh);
      dc.RegisterField("u", &u);
      SaveDataCollection(dc, 0, 0);
      SaveDataCollection(dc, 1, 1);
      SaveDataCollection(dc, 2, 2);
   }

   // Using restart mode, append to the existing dataset, overwriting timesteps
   // 1 and 2 with 1 and 1.5.
   {
      ParaViewDataCollection dc("ParaView", &mesh);
      dc.UseRestartMode(true);
      dc.RegisterField("u", &u);
      SaveDataCollection(dc, 1, 1.0);
      SaveDataCollection(dc, 2, 1.5);
   }

   // Parse the resulting PVD file, and verify that the structure is correct,
   // and that it contains three timesteps: 0, 1, and 1.5.
   using namespace tinyxml2;
   auto StringCompare = [](const char *s1, const char *s2)
   {
      if (s1 == NULL || s2 == NULL) { return false; }
      return strcmp(s1, s2) == 0;
   };
   auto VerifyDataset = [StringCompare](const XMLElement *ds, double t_ref)
   {
      REQUIRE(ds);
      REQUIRE(StringCompare(ds->Name(), "DataSet"));
      const char *timestep = ds->Attribute("timestep");
      REQUIRE(timestep);
      double t = std::stod(timestep);
      REQUIRE(t == MFEM_Approx(t_ref));
   };

   XMLDocument xml;
   xml.LoadFile("ParaView/ParaView.pvd");
   REQUIRE(xml.ErrorID() == XML_SUCCESS);

   const XMLElement *vtkfile = xml.FirstChildElement();
   REQUIRE(vtkfile);
   REQUIRE(StringCompare(vtkfile->Name(), "VTKFile"));
   const XMLElement *collection = vtkfile->FirstChildElement();
   REQUIRE(collection);
   REQUIRE(StringCompare(collection->Name(), "Collection"));

   const XMLElement *dataset = collection->FirstChildElement();
   VerifyDataset(dataset, 0.0);
   dataset = dataset->NextSiblingElement();
   VerifyDataset(dataset, 1.0);
   dataset = dataset->NextSiblingElement();
   VerifyDataset(dataset, 1.5);
   REQUIRE(dataset->NextSiblingElement() == NULL);

   // Clean up
   for (int c=0; c<=2; ++c)
   {
      std::string prefix = "ParaView/Cycle00000" + std::to_string(c);
      REQUIRE(remove((prefix + "/data.pvtu").c_str()) == 0);
      REQUIRE(remove((prefix + "/proc000000.vtu").c_str()) == 0);
      REQUIRE(rmdir(prefix.c_str()) == 0);
   }
   REQUIRE(remove("ParaView/ParaView.pvd") == 0);
   REQUIRE(rmdir("ParaView") == 0);
}
