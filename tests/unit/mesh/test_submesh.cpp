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

TEST_CASE("Domain SubMesh construction", "[SubMesh]")
{
   using namespace std;

   // Create a cartesian mesh in 2D with two attributes that have the following
   // topology.
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

   // Deform original mesh
   mesh.EnsureNodes();
   mesh.SetCurvature(2);
   GridFunction *nodes = mesh.GetNodes();

   auto node_movement_coeff = VectorFunctionCoefficient(mesh.Dimension(),
                                                        [](const Vector &coords, Vector &u)
   {
      double x = coords(0);
      double y = coords(1);

      u(0) = x;
      u(1) = y + 0.05 * sin(x*2.0*M_PI);
   });

   mesh.Transform(node_movement_coeff);

   H1_FECollection h1_fec(1, 2);
   FiniteElementSpace parent_h1_fes(&mesh, &h1_fec);

   GridFunction parent_gf(&parent_h1_fes);
   parent_gf = 0.0;

   auto parent_coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      return y + 0.05 * sin(x*2.0*M_PI);
   });

   parent_gf.ProjectCoefficient(parent_coeff);

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;

   SubMesh submesh = SubMesh::CreateFromDomain(mesh, subdomain_attributes);
   FiniteElementSpace sub_h1_fes(&submesh, &h1_fec);

   GridFunction sub_gf(&sub_h1_fes);
   sub_gf = 0.0;

   SubMesh::Transfer(parent_gf, sub_gf);

   out << "Domain SubMesh statistics:\n"
       << "NE: " << submesh.GetNE() << "\n"
       << "NVTX: " << submesh.GetNV() << "\n"
       << std::endl;

   // char vishost[] = "orchid-wired";
   // int  visport   = 19916;

   // socketstream meshsock(vishost, visport);
   // meshsock.precision(8);
   // meshsock << "solution\n" << mesh << parent_gf << flush;
   // meshsock << "keys mrRjn" << flush;

   // socketstream submeshsock(vishost, visport);
   // submeshsock.precision(8);
   // submeshsock << "solution\n" << submesh << sub_gf << flush;
   // submeshsock << "keys mrRjn" << flush;
}

TEST_CASE("Surface SubMesh construction", "[SubMesh]")
{
   using namespace std;

   double Hy = 1.0;

   Mesh mesh = Mesh::MakeCartesian3D(5, 5, 5, Element::HEXAHEDRON, 1.0, Hy, 1.0, false);

   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      Element *el = mesh.GetBdrElement(i);
      el->SetAttribute(1);

      Array<int> vertices;
      el->GetVertices(vertices);

      bool all_vtx_inside = true;
      for (int j = 0; j < vertices.Size(); j++)
      {
         if (mesh.GetVertex(vertices[j])[1] < Hy)
         {
	   all_vtx_inside = false;
         }
      }
      if (all_vtx_inside)
      {
	el->SetAttribute(2);
      }
   }
   mesh.SetAttributes();

   // Deform original mesh
   mesh.EnsureNodes();
   mesh.SetCurvature(2);
   GridFunction *nodes = mesh.GetNodes();

   auto node_movement_coeff = VectorFunctionCoefficient(mesh.Dimension(),
                                                        [](const Vector &coords, Vector &u)
   {
      double x = coords(0);
      double y = coords(1);
      double z = coords(2);

      u(0) = x;
      u(1) = y + 0.05 * sin(x*2.0*M_PI);
      u(2) = z;
   });

   mesh.Transform(node_movement_coeff);

   H1_FECollection h1_fec(1, 3);
   FiniteElementSpace parent_h1_fes(&mesh, &h1_fec);

   GridFunction parent_gf(&parent_h1_fes);
   parent_gf = 0.0;

   auto parent_coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      return y + 0.05 * sin(x*2.0*M_PI);
   });

   parent_gf.ProjectCoefficient(parent_coeff);

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;

   SubMesh submesh = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);
   FiniteElementSpace sub_h1_fes(&submesh, &h1_fec);

   GridFunction sub_gf(&sub_h1_fes);
   sub_gf = 0.0;

   SubMesh::Transfer(parent_gf, sub_gf);

   out << "Boundary SubMesh statistics:\n"
     << "NE: " << submesh.GetNE() << "\n"
     << "NVTX: " << submesh.GetNV() << "\n"
     << std::endl;

   char vishost[] = "orchid-wired";
   int  visport   = 19916;

   socketstream meshsock(vishost, visport);
   meshsock.precision(8);
   meshsock << "solution\n" << mesh << parent_gf << flush;
   meshsock << "keys mrRjn" << flush;

   socketstream submeshsock(vishost, visport);
   submeshsock.precision(8);
   submeshsock << "solution\n" << submesh << sub_gf << flush;
   submeshsock << "keys mrRjn" << flush;
}
