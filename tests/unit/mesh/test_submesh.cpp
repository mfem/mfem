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

TEST_CASE("Simple SubMesh construction", "[SubMesh]")
{
   using namespace std;

   // Create a cartesian mesh in 2D with two attributes that have the following
   // topology
   //
   // +--------------------------+
   // |                          |
   // |     +-------------+      |
   // |     |             |      |
   // |     |     2       |  1   |
   // |     |             |      |
   // |     +-------------+      |
   // |                          |
   // +--------------------------+
   //

   Mesh mesh = Mesh::MakeCartesian2D(5, 5, Element::QUADRILATERAL, true, 1.0, 1.0,
                                     false);

   for (int i = 0; i < mesh.GetNE(); i++)
   {
      Element *el = mesh.GetElement(i);
      el->SetAttribute(1);

      Array<int> vertices;
      el->GetVertices(vertices);

      for (int j = 0; j < vertices.Size(); j++)
      {
         double *coords = mesh.GetVertex(vertices[j]);

         if (coords[0] >= 0.25 &&
             coords[0] <= 0.75 &&
             coords[1] >= 0.25 &&
             coords[1] <= 0.75)
         {
            el->SetAttribute(2);
         }
      }
   }
   mesh.SetAttributes();

   L2_FECollection fec(0, 2);
   FiniteElementSpace l2fes(&mesh, &fec);

   Vector tmp(mesh.attributes.Size());
   for (int i = 0; i < tmp.Size(); i++)
   {
      tmp(i) = mesh.attributes[i];
   }

   PWConstCoefficient attributes_coeff(tmp);

   GridFunction attributes_gf(&l2fes);
   attributes_gf.ProjectCoefficient(attributes_coeff);

   ParaViewDataCollection pvdc("test_submesh_mesh_output", &mesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(false);
   pvdc.SetCycle(0);
   pvdc.SetTime(0.0);
   pvdc.RegisterField("attributes", &attributes_gf);
   pvdc.Save();

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;

   SubMesh submesh = SubMesh::CreateFromDomain(mesh, subdomain_attributes);

   out << "SubMesh statistics:\n"
     << "NE: " << submesh.GetNE() << "\n"
     << "NVTX: " << submesh.GetNV() << "\n"
     << std::endl;

   L2_FECollection fec_sub(0, 2);
   FiniteElementSpace l2fes_sub(&submesh, &fec_sub);
   GridFunction attributes_sub_gf(&l2fes_sub);

   ParaViewDataCollection submesh_pvdc("test_submesh_submesh_output", &submesh);
   submesh_pvdc.SetDataFormat(VTKFormat::BINARY32);
   submesh_pvdc.SetHighOrderOutput(false);
   submesh_pvdc.SetCycle(0);
   submesh_pvdc.SetTime(0.0);
   submesh_pvdc.RegisterField("attributes", &attributes_sub_gf);
   submesh_pvdc.Save();
}
