// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
   std::vector<std::string> mesh_filenames =
   {
      "quad_append_b64.vtu", "quad_append_raw.vtu",
      "quad_ascii.vtu", "quad_binary.vtu"
   };
#ifdef MFEM_USE_ZLIB
   // Test VTU meshes with binary compression only if compiled with zlib enabled
   mesh_filenames.insert(mesh_filenames.end(),
   {
      "quad_append_b64_compress.vtu", "quad_append_raw_compress.vtu",
      "quad_binary_compress.vtu"
   });
#endif

   const auto fname = GENERATE_COPY(from_range(mesh_filenames));

   Mesh mesh = Mesh::LoadFromFile("data/" + fname);
   REQUIRE(mesh.Dimension() == 2);
   REQUIRE(mesh.GetNE() == 9);
   REQUIRE(mesh.GetNV() == 16);
   REQUIRE(mesh.HasGeometry(Geometry::POINT));
   REQUIRE(mesh.HasGeometry(Geometry::SEGMENT));
   REQUIRE(mesh.HasGeometry(Geometry::SQUARE));
   REQUIRE(mesh.GetNumGeometries(2) == 1);
}

TEST_CASE("VTU Attributes", "[VTU][XML]")
{
   // quad_attribute.vtu contains the attributes in a cell data array named
   // "attribute"
   Mesh mesh_1 = Mesh::LoadFromFile("data/quad_attribute.vtu");
   // quad_material_attribute.vtu has cell data arrays named "material" and
   // "attribute". The one named "material" should take precedence.
   Mesh mesh_2 = Mesh::LoadFromFile("data/quad_material_attribute.vtu");

   REQUIRE(mesh_1.GetNE() == 9);
   REQUIRE(mesh_2.GetNE() == 9);

   for (int i = 0; i < mesh_1.GetNE(); ++i)
   {
      REQUIRE(mesh_1.GetAttribute(i) == i+1);
      REQUIRE(mesh_2.GetAttribute(i) == i+1);
   }
}

TEST_CASE("VTU Attribute Arrays", "[VTU][XML]")
{
   // The attributes are read from the cell data array named "material" or
   // "attribute" (with "material" taking precedence), independent of the order
   // of the arrays and of the active "Scalars" array.
   std::vector<std::string> cell_data_blocks =
   {
      R"(<CellData Scalars="attribute">
        <DataArray type="Int32" Name="quality" format="ascii">7</DataArray>
        <DataArray type="Int32" Name="attribute" format="ascii">9</DataArray>
      </CellData>)",
      R"(<CellData>
        <DataArray type="Int32" Name="attribute" format="ascii">9</DataArray>
      </CellData>)",
      R"(<CellData Scalars="attribute">
        <DataArray type="Int32" Name="attribute" format="ascii">7</DataArray>
        <DataArray type="Int32" Name="material" format="ascii">9</DataArray>
      </CellData>)",
      R"(<CellData Scalars="material">
        <DataArray type="Int32" Name="material" format="ascii">9</DataArray>
        <DataArray type="Int32" Name="attribute" format="ascii">7</DataArray>
      </CellData>)"
   };

   const auto cell_data = GENERATE_COPY(from_range(cell_data_blocks));

   std::istringstream vtu(R"(<VTKFile type="UnstructuredGrid" version="0.1">
  <UnstructuredGrid>
    <Piece NumberOfPoints="3" NumberOfCells="1">
      )" + cell_data + R"(
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 0  1 0 0  0 1 0
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          0 1 2
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">3</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">5</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>)");

   Mesh mesh(vtu);
   REQUIRE(mesh.GetNE() == 1);
   REQUIRE(mesh.GetAttribute(0) == 9);
}

TEST_CASE("VTU XML Compressed Blocks", "[VTU][XML][MFEMData]")
{
#ifdef MFEM_USE_ZLIB
   // VTU meshes with binary compression require zlib enabled
   auto filename = GENERATE(
                      "bracket_appended_compressed.vtu",
                      "bracket_appended_encoded_compressed.vtu",
                      "bracket_inline_compressed.vtu"
                   );

   Mesh mesh = Mesh::LoadFromFile(mfem_data_dir + "/vtk/" + filename);

   REQUIRE(mesh.Dimension() == 3);
   REQUIRE(mesh.GetNE() == 206208);
   REQUIRE(mesh.GetNV() == 50000);
   REQUIRE(mesh.HasGeometry(Geometry::TETRAHEDRON));
   REQUIRE(mesh.GetNumGeometries(3) == 1);
#endif
}
