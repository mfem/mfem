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

using namespace mfem;

TEST_CASE("VTU XML Reader", "[Mesh][VTU][XML]")
{
   std::vector<std::string> mesh_filenames = {
      "quad_append_b64.vtu", "quad_append_raw.vtu",
      "quad_ascii.vtu", "quad_binary.vtu"
   };
#ifdef MFEM_USE_ZLIB
   // Test VTU meshes with binary compression only if compiled with zlib enabled
   mesh_filenames.insert(mesh_filenames.end(), {
      "quad_append_b64_compress.vtu", "quad_append_raw_compress.vtu",
      "quad_binary_compress.vtu"
   });
#endif
   for (const std::string &fname : mesh_filenames)
   {
      Mesh mesh = Mesh::LoadFromFile(("data/" + fname).c_str());
      REQUIRE(mesh.Dimension() == 2);
      REQUIRE(mesh.GetNE() == 9);
      REQUIRE(mesh.GetNV() == 16);
      REQUIRE(mesh.HasGeometry(Geometry::POINT));
      REQUIRE(mesh.HasGeometry(Geometry::SEGMENT));
      REQUIRE(mesh.HasGeometry(Geometry::SQUARE));
      REQUIRE(mesh.GetNumGeometries(2) == 1);
   }
}
