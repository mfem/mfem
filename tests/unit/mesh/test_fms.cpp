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
using namespace mfem;

#include "unit_tests.hpp"

#ifdef MFEM_USE_FMS
TEST_CASE("Load FMS datacollection", "[FMS]")
{
   FMSDataCollection fms_dc("../../data/star-q3");
   fms_dc.SetPadDigits(0);
   fms_dc.SetPadDigitsCycle(0);
   fms_dc.Load();
}

TEST_CASE("Compare FMS mesh from file")
{
   auto mfem_mesh = new Mesh("../../data/star-q3.mesh");

   FMSDataCollection fms_dc("../../data/star-q3");
   fms_dc.SetPadDigits(0);
   fms_dc.SetPadDigitsCycle(0);
   fms_dc.Load();
   auto fms_mesh = fms_dc.GetMesh();

   SECTION("Geometric entities")
   {
      REQUIRE(mfem_mesh->GetNBE() == fms_mesh->GetNBE());
      REQUIRE(mfem_mesh->GetNE() == fms_mesh->GetNE());
      REQUIRE(mfem_mesh->GetNFaces() == fms_mesh->GetNFaces());
   }

   SECTION("Nodes")
   {
      mfem_mesh->EnsureNodes();
      fms_mesh->EnsureNodes();

      auto mfem_nodes = mfem_mesh->GetNodes();
      auto fms_nodes = fms_mesh->GetNodes();

      for (int i = 0; i < mfem_nodes->GetTrueVector().Size(); i++)
      {
         REQUIRE(mfem_nodes->GetTrueVector()(i)
                 == fms_nodes->GetTrueVector()(i));
      }
   }

   delete mfem_mesh;
}
#endif
