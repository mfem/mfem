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

TEST_CASE("ExodusII Write Hex8", "[Mesh]")
{
#ifdef MFEM_USE_NETCDF
   // Load Exodus II mesh from file.
   std::string fpath_original = "data/simple-cube-hex8.e";
   Mesh original_mesh(fpath_original);

   // Write generated Exodus II mesh to file.
   std::string fpath_generated = "data/simple-cube-hex8-out.e";
   original_mesh.WriteExodusII(fpath_generated);

   // Load generated Exodus II mesh.
   Mesh generated_mesh(fpath_generated);

   // Compare.
   REQUIRE(original_mesh.GetNE() == generated_mesh.GetNE());
   REQUIRE(original_mesh.GetNBE() == generated_mesh.GetNBE());
   REQUIRE(original_mesh.GetNFaces() == generated_mesh.GetNFaces());

   for (int ielement = 0; ielement < original_mesh.GetNE(); ielement++)
   {
      Element * orig_element = original_mesh.GetElement(ielement);
      Element * gen_element = generated_mesh.GetElement(ielement);

      REQUIRE(orig_element->GetType() == gen_element->GetType());
      REQUIRE(orig_element->GetAttribute() == gen_element->GetAttribute());

      Array<int> orig_vertices, gen_vertices;

      orig_element->GetVertices(orig_vertices);
      gen_element->GetVertices(gen_vertices);

      for (int i = 0; i < orig_vertices.Size(); i++)
      {
         REQUIRE(orig_vertices[i] == gen_vertices[i]);
      }

      // TODO: - add checks that face vertices line-up.

      // TODO: - add checks that coordinates line up.
   }

   for (int ibdr_element = 0; ibdr_element < original_mesh.GetNBE();
        ibdr_element++)
   {
      Element * orig_bdr_element = original_mesh.GetBdrElement(ibdr_element);
      Element * gen_bdr_element = generated_mesh.GetBdrElement(ibdr_element);

      REQUIRE(orig_bdr_element->GetType() == gen_bdr_element->GetType());
      REQUIRE(gen_bdr_element->GetAttribute() == gen_bdr_element->GetAttribute());

      // TODO: - check vertices are correct.
   }
#endif
}