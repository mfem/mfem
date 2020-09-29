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

using namespace mfem;

#ifdef MFEM_USE_ZLIB

TEST_CASE("Save and load file", "[zlib]")
{
   std::string mesh_name = "zlib_test.mesh";
   Mesh mesh(2, 3, Element::QUADRILATERAL, 0, 2.0, 3.0);

   SECTION("Save compressed mesh with bool argument")
   {
      // Write compressed mesh to disk.
      ofgzstream mesh_file(mesh_name, true);
      mesh.Print(mesh_file);

      REQUIRE(mesh_file.fail() == false);
   }

   SECTION("Save compressed mesh saved with bool argument")
   {
      // Load compressed mesh and create new mesh object.
      Mesh loaded_mesh(mesh_name.c_str());

      REQUIRE(mesh.Dimension() == loaded_mesh.Dimension());
      REQUIRE(std::remove(mesh_name.c_str()) == 0);
   }

   SECTION("Save compressed mesh with string argument")
   {
      ofgzstream mesh_file(mesh_name, "zwb6");
      mesh.Print(mesh_file);

      REQUIRE(mesh_file.fail() == false);
   }

   SECTION("Load compressed mesh saved with string argument")
   {
      // Load compressed mesh and create new mesh object.
      Mesh loaded_mesh(mesh_name.c_str());

      REQUIRE(mesh.Dimension() == loaded_mesh.Dimension());
      REQUIRE(std::remove(mesh_name.c_str()) == 0);
   }
}

#endif
