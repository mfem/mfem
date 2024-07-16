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
#include "unit_tests.hpp"

using namespace mfem;

TEST_CASE("Exodus Hex8", "[MFEMData][Mesh][ExodusII]")
{
#ifdef MFEM_USE_NETCDF
   const std::string fname = "simple-cube-hex8.e";

   const std::string & fpath = (mfem_data_dir + "/exodusii/" + fname);

   Mesh mesh(fpath);

   REQUIRE(mesh.Dimension() == 3);
   REQUIRE(mesh.GetNE() == 27);
   REQUIRE(mesh.GetNV() == 64);
   REQUIRE(mesh.GetNodalFESpace() == nullptr);
   REQUIRE(mesh.HasGeometry(Geometry::CUBE));
#endif
}

TEST_CASE("Exodus Hex27", "[MFEMData][Mesh][ExodusII]")
{
#ifdef MFEM_USE_NETCDF
   const std::string fname = "simple-cube-hex27.e";

   const std::string & fpath = (mfem_data_dir + "/exodusii/" + fname);

   Mesh mesh(fpath);

   REQUIRE(mesh.Dimension() == 3);
   REQUIRE(mesh.GetNE() == 27);
   REQUIRE(mesh.GetNV() == 64);
   REQUIRE(mesh.GetNodalFESpace() != nullptr);
   REQUIRE(mesh.HasGeometry(Geometry::CUBE));
#endif
}

TEST_CASE("Exodus Tet4", "[MFEMData][Mesh][ExodusII]")
{
#ifdef MFEM_USE_NETCDF
   const std::string fname = "simple-cube-tet4.e";

   const std::string & fpath = (mfem_data_dir + "/exodusii/" + fname);

   Mesh mesh(fpath);

   REQUIRE(mesh.Dimension() == 3);
   REQUIRE(mesh.GetNE() == 295);
   REQUIRE(mesh.GetNV() == 98);
   REQUIRE(mesh.GetNodalFESpace() == nullptr);
   REQUIRE(mesh.HasGeometry(Geometry::TETRAHEDRON));
#endif
}

TEST_CASE("Exodus Tet10", "[MFEMData][Mesh][ExodusII]")
{
#ifdef MFEM_USE_NETCDF
   const std::string fname = "simple-cube-tet10.e";

   const std::string & fpath = (mfem_data_dir + "/exodusii/" + fname);

   Mesh mesh(fpath);

   REQUIRE(mesh.Dimension() == 3);
   REQUIRE(mesh.GetNE() == 295);
   REQUIRE(mesh.GetNV() == 98);
   REQUIRE(mesh.GetNodalFESpace() != nullptr);
   REQUIRE(mesh.HasGeometry(Geometry::TETRAHEDRON));
#endif
}

TEST_CASE("Exodus Pyramid5", "[MFEMData][Mesh][ExodusII]")
{
#ifdef MFEM_USE_NETCDF
   const std::string fname = "simple-cube-pyramid5.e";

   const std::string & fpath = (mfem_data_dir + "/exodusii/" + fname);

   Mesh mesh(fpath);

   REQUIRE(mesh.Dimension() == 3);
   REQUIRE(mesh.GetNE() == 750);
   REQUIRE(mesh.GetNV() == 341);
   REQUIRE(mesh.GetNodalFESpace() == nullptr);
   REQUIRE(mesh.HasGeometry(Geometry::PYRAMID));
#endif
}

TEST_CASE("Exodus Wedge6", "[MFEMData][Mesh][ExodusII]")
{
#ifdef MFEM_USE_NETCDF
   const std::string fname = "simple-cube-wedge6.e";

   const std::string & fpath = (mfem_data_dir + "/exodusii/" + fname);

   Mesh mesh(fpath);

   REQUIRE(mesh.Dimension() == 3);
   REQUIRE(mesh.GetNE() == 250);
   REQUIRE(mesh.GetNV() == 216);
   REQUIRE(mesh.GetNodalFESpace() == nullptr);
   REQUIRE(mesh.HasGeometry(Geometry::PRISM));
#endif
}

TEST_CASE("Exodus Wedge18", "[MFEMData][Mesh][ExodusII]")
{
#ifdef MFEM_USE_NETCDF
   const std::string fname = "simple-cube-wedge18.e";

   const std::string & fpath = (mfem_data_dir + "/exodusii/" + fname);

   Mesh mesh(fpath);

   REQUIRE(mesh.Dimension() == 3);
   REQUIRE(mesh.GetNE() == 250);
   REQUIRE(mesh.GetNV() == 216);
   REQUIRE(mesh.GetNodalFESpace() != nullptr);
   REQUIRE(mesh.HasGeometry(Geometry::PRISM));
#endif
}

TEST_CASE("Exodus Mixed Mesh Order 1", "[MFEMData][Mesh][ExodusII]")
{
#ifdef MFEM_USE_NETCDF
   const std::string fname = "simple-cube-multi-element-order1.e";

   const std::string & fpath = (mfem_data_dir + "/exodusii/" + fname);

   Mesh mesh(fpath);

   REQUIRE(mesh.Dimension() == 3);
   REQUIRE(mesh.GetNodalFESpace() == nullptr);

   // Pyramid, Wedge, Hex, Tet.
   REQUIRE(mesh.HasGeometry(Geometry::PYRAMID));
   REQUIRE(mesh.HasGeometry(Geometry::PRISM));
   REQUIRE(mesh.HasGeometry(Geometry::CUBE));
   REQUIRE(mesh.HasGeometry(Geometry::TETRAHEDRON));
#endif
}

TEST_CASE("Exodus Mixed Mesh Order 2", "[MFEMData][Mesh][ExodusII]")
{
#ifdef MFEM_USE_NETCDF
   const std::string fname = "simple-cube-multi-element-order2.e";

   const std::string & fpath = (mfem_data_dir + "/exodusii/" + fname);

   Mesh mesh(fpath);

   REQUIRE(mesh.Dimension() == 3);
   REQUIRE(mesh.GetNodalFESpace() != nullptr);

   // Hex, Tet only.
   REQUIRE(mesh.HasGeometry(Geometry::CUBE));
   REQUIRE(mesh.HasGeometry(Geometry::TETRAHEDRON));
#endif
}
