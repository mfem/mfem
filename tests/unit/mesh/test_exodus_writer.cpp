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

#ifdef MFEM_USE_NETCDF
static void CompareMeshes(Mesh &mesh1, Mesh &mesh2)
{
   REQUIRE(mesh1.GetNE() == mesh2.GetNE());
   REQUIRE(mesh1.GetNV() == mesh2.GetNV());
   REQUIRE(mesh1.GetNBE() == mesh2.GetNBE());
   REQUIRE(mesh1.GetNFaces() == mesh2.GetNFaces());

   const FiniteElementSpace *fespace1 = mesh1.GetNodalFESpace();
   const FiniteElementSpace *fespace2 = mesh2.GetNodalFESpace();

   // Check elements.
   Array<int> element_faces1, element_faces2;
   Array<int> element_orient1, element_orient2;
   Array<int> dofs1, dofs2;

   for (int ielement = 0; ielement < mesh1.GetNE(); ielement++)
   {
      int attr1 = mesh1.GetAttribute(ielement);
      int attr2 = mesh2.GetAttribute(ielement);

      REQUIRE(attr1 == attr2);

      Element::Type type1 = mesh1.GetElementType(ielement);
      Element::Type type2 = mesh2.GetElementType(ielement);

      REQUIRE(type1 == type2);

      mesh1.GetElementFaces(ielement, element_faces1, element_orient1);
      mesh2.GetElementFaces(ielement, element_faces2, element_orient2);

      REQUIRE(element_faces1 == element_faces2);
      REQUIRE(element_orient1 == element_orient2);

      if (fespace1 && fespace2)
      {
         fespace1->GetElementDofs(ielement, dofs1);
         fespace2->GetElementDofs(ielement, dofs2);
      }
      else
      {
         mesh1.GetElementVertices(ielement, dofs1);
         mesh2.GetElementVertices(ielement, dofs2);
      }

      REQUIRE(dofs1 == dofs2);
   }

   // Check bdr elements.
   for (int ibdr_element = 0; ibdr_element < mesh1.GetNBE(); ibdr_element++)
   {
      int attr1 = mesh1.GetBdrAttribute(ibdr_element);
      int attr2 = mesh2.GetBdrAttribute(ibdr_element);

      REQUIRE(attr1 == attr2);

      Element::Type type1 = mesh1.GetBdrElementType(ibdr_element);
      Element::Type type2 = mesh2.GetBdrElementType(ibdr_element);

      REQUIRE(type1 == type2);

      int face_index1 = mesh1.GetBdrElementFaceIndex(ibdr_element);
      int face_index2 = mesh2.GetBdrElementFaceIndex(ibdr_element);

      REQUIRE(face_index1 == face_index2);
   }

   // Check face vertices.
   Array<int> face_vertices1, face_vertices2;
   for (int iface_index = 0; iface_index < mesh1.GetNFaces(); iface_index++)
   {
      mesh1.GetFaceVertices(iface_index, face_vertices1);
      mesh2.GetFaceVertices(iface_index, face_vertices2);

      REQUIRE(face_vertices1 == face_vertices2);
   }
}
#endif

TEST_CASE("ExodusII Write Hex8", "[Mesh][MFEMData]")
{
#ifdef MFEM_USE_NETCDF
   // Load Exodus II mesh from file.
   std::string filename = "simple-cube-hex8.e";
   Mesh original_mesh = Mesh::LoadFromFile(mfem_data_dir + "/exodusii/" + filename,
                                           0, 0, true);

   // Write generated Exodus II mesh to file.
   std::string fpath_generated = "simple-cube-hex8-out.e";
   original_mesh.PrintExodusII(fpath_generated);

   // Load generated Exodus II mesh.
   Mesh generated_mesh = Mesh::LoadFromFile(fpath_generated, 0, 0, true);

   CompareMeshes(original_mesh, generated_mesh);

   // Remove temporary file.
   REQUIRE(remove(fpath_generated.c_str()) == 0);
#endif
}

TEST_CASE("ExodusII Write Hex27", "[Mesh][MFEMData]")
{
#ifdef MFEM_USE_NETCDF
   // Load Exodus II mesh from file.
   std::string filename = "simple-cube-hex27.e";
   Mesh original_mesh = Mesh::LoadFromFile(mfem_data_dir + "/exodusii/" + filename,
                                           0, 0, true);

   // Write generated Exodus II mesh to file.
   std::string fpath_generated = "simple-cube-hex27-out.e";
   original_mesh.PrintExodusII(fpath_generated);

   // Load generated Exodus II mesh.
   Mesh generated_mesh = Mesh::LoadFromFile(fpath_generated, 0, 0, true);

   CompareMeshes(original_mesh, generated_mesh);

   REQUIRE(remove(fpath_generated.c_str()) == 0);
#endif
}

TEST_CASE("ExodusII Write Tet4", "[Mesh][MFEMData]")
{
#ifdef MFEM_USE_NETCDF
   // Load Exodus II mesh from file. NB: - Do NOT refine as this changes vertex ordering!
   std::string filename = "simple-cube-tet4.e";
   Mesh original_mesh = Mesh::LoadFromFile(mfem_data_dir + "/exodusii/" + filename,
                                           0, 0, true);

   // Write generated Exodus II mesh to file.
   std::string fpath_generated = "simple-cube-tet4-out.e";
   original_mesh.PrintExodusII(fpath_generated);

   // Load generated Exodus II mesh.
   Mesh generated_mesh = Mesh::LoadFromFile(fpath_generated, 0, 0, true);

   CompareMeshes(original_mesh, generated_mesh);

   REQUIRE(remove(fpath_generated.c_str()) == 0);
#endif
}

TEST_CASE("ExodusII Write Tet10", "[Mesh][MFEMData]")
{
#ifdef MFEM_USE_NETCDF
   // Load Exodus II mesh from file.
   std::string filename = "simple-cube-tet10.e";
   Mesh original_mesh = Mesh::LoadFromFile(mfem_data_dir + "/exodusii/" + filename,
                                           0, 0, true);

   // Write generated Exodus II mesh to file.
   std::string fpath_generated = "simple-cube-tet10-out.e";
   original_mesh.PrintExodusII(fpath_generated);

   // Load generated Exodus II mesh.
   Mesh generated_mesh = Mesh::LoadFromFile(fpath_generated, 0, 0, true);

   CompareMeshes(original_mesh, generated_mesh);

   REQUIRE(remove(fpath_generated.c_str()) == 0);
#endif
}

// TEST_CASE("ExodusII Write Wedge6", "[Mesh][MFEMData]")
// {
// #ifdef MFEM_USE_NETCDF
//    std::string filename = "simple-cube-wedge6.e";
//    Mesh original_mesh = Mesh::LoadFromFile(mfem_data_dir + "/exodusii/" + filename, 0, 0, true);

//    std::string fpath_generated = "simple-cube-wedge6-out.e";
//    original_mesh.PrintExodusII(fpath_generated);

//    Mesh generated_mesh = Mesh::LoadFromFile(fpath_generated, 0, 0, true);
//    CompareMeshes(original_mesh, generated_mesh);
//
//    REQUIRE(remove(fpath_generated.c_str()) == 0);
// #endif
// }

// TEST_CASE("ExodusII Write Wedge18", "[Mesh][MFEMData]")
// {
// #ifdef MFEM_USE_NETCDF
//    std::string filename = "simple-cube-wedge18.e";
//    Mesh original_mesh = Mesh::LoadFromFile(mfem_data_dir + "/exodusii/" + filename, 0, 0, true);

//    std::string fpath_generated = "simple-cube-wedge18-out.e";
//    original_mesh.PrintExodusII(fpath_generated);

//    Mesh generated_mesh = Mesh::LoadFromFile(fpath_generated, 0, 0, true);
//    CompareMeshes(original_mesh, generated_mesh);

//    REQUIRE(remove(fpath_generated.c_str()) == 0);
// #endif
// }

// TEST_CASE("ExodusII Write Pyramid5", "[Mesh][MFEMData]")
// {
// #ifdef MFEM_USE_NETCDF
//    std::string filename = "simple-cube-pyramid5.e";
//    Mesh original_mesh = Mesh::LoadFromFile(mfem_data_dir + "/exodusii/" + filename, 0, 0, true);

//    std::string fpath_generated = "simple-cube-pyramid5-out.e";
//    original_mesh.PrintExodusII(fpath_generated);

//    Mesh generated_mesh = Mesh::LoadFromFile(fpath_generated, 0, 0, true);
//    CompareMeshes(original_mesh, generated_mesh);

//    REQUIRE(remove(fpath_generated.c_str()) == 0);
// #endif
// }

// TEST_CASE("ExodusII Write Pyramid14", "[Mesh][MFEMData]")
// {
// #ifdef MFEM_USE_NETCDF
//    std::string filename = "simple-cube-pyramid14.e";
//    Mesh original_mesh = Mesh::LoadFromFile(mfem_data_dir + "/exodusii/" + filename, 0, 0, true);

//    std::string fpath_generated = "simple-cube-pyramid14-out.e";
//    original_mesh.PrintExodusII(fpath_generated);

//    Mesh generated_mesh = Mesh::LoadFromFile(fpath_generated, 0, 0, true);
//    CompareMeshes(original_mesh, generated_mesh);

//    REQUIRE(remove(fpath_generated.c_str()) == 0);
// #endif
// }

// TEST_CASE("ExodusII Write Mixed First-Order", "[Mesh][MFEMData]")
// {
// #ifdef MFEM_USE_NETCDF
//    // Contains Hex8, Tet4, Wedge6, Pyramid5 elements.
//    std::string filename = "simple-cube-multi-element-order1.e";
//    Mesh original_mesh = Mesh::LoadFromFile(mfem_data_dir + "/exodusii/" + filename, 0, 0, true);

//    std::string fpath_generated = "simple-cube-multi-element-order1-out.e";
//    original_mesh.PrintExodusII(fpath_generated);

//    Mesh generated_mesh = Mesh::LoadFromFile(fpath_generated, 0, 0, true);
//    CompareMeshes(original_mesh, generated_mesh);

//    REQUIRE(remove(fpath_generated.c_str()) == 0);
// #endif
// }

// TEST_CASE("ExodusII Write Mixed Second-Order", "[Mesh][MFEMData]")
// {
// #ifdef MFEM_USE_NETCDF
//    // Contains Hex27 and Tet10 elements.
//    std::string filename = "simple-cube-multi-element-order2.e";
//    Mesh original_mesh = Mesh::LoadFromFile(mfem_data_dir + "/exodusii/" + filename, 0, 0, true);

//    std::string fpath_generated = "simple-cube-multi-element-order2-out.e";
//    original_mesh.PrintExodusII(fpath_generated);

//    Mesh generated_mesh = Mesh::LoadFromFile(fpath_generated, 0, 0, true);
//    CompareMeshes(original_mesh, generated_mesh);

//    REQUIRE(remove(fpath_generated.c_str()) == 0);
// #endif
// }