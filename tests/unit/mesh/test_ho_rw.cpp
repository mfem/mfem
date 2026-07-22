// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

TEST_CASE("MFEM High-Order Read/Write", "[Mesh]")
{
   std::string mesh_fname = GENERATE("../../data/ref-segment.mesh",
                                     "../../data/ref-triangle.mesh",
                                     "../../data/ref-square.mesh",
                                     "../../data/ref-tetrahedron.mesh",
                                     "../../data/ref-cube.mesh",
                                     "../../data/ref-prism.mesh",
                                     "../../data/ref-pyramid.mesh",
                                     "../../data/ref-pyramid.mesh-pyr0",
                                     "../../data/tinyzoo-2d.mesh",
                                     "../../data/tinyzoo-3d.mesh",
                                     "../../data/tinyzoo-3d.mesh-pyr0"
                                    );

   auto order = GENERATE(1, 2, 3, 4);
   auto discont = GENERATE(0, 1);
   int pyrtype = 1;

   std::size_t d = mesh_fname.rfind("-");
   if (d == mesh_fname.length() - 5)
   {
      mesh_fname.resize(mesh_fname.length() - 5);
      pyrtype = 0;
   }

   CAPTURE(mesh_fname);
   CAPTURE(order);
   CAPTURE(discont);
   CAPTURE(pyrtype);

   int ne = mesh_fname == "../../data/tinyzoo-2d.mesh" ? 2 :
            (mesh_fname == "../../data/tinyzoo-3d.mesh" ? 4 : 1);

   Mesh mesh(mesh_fname, 1, 1);
   REQUIRE(mesh.GetNE() == ne);

   const int dim = mesh.Dimension();

   const char * fe_coll_name = NULL;
   if (order > 1 )
   {
      mesh.SetCurvature(order, discont == 1, dim, 1, pyrtype);
      fe_coll_name = mesh.GetNodes()->FESpace()->FEColl()->Name();
   }

   CAPTURE(fe_coll_name);

   std::ostringstream oss;
   mesh.Print(oss);

   std::istringstream iss(oss.str());
   Mesh imesh(iss);

   REQUIRE(imesh.GetNE() == ne);
}

} // namespace mfem
