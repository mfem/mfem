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
#include <iomanip>

using namespace mfem;

#include "unit_tests.hpp"

enum FECType { H1, L2 };
static const char *fectype_str[] = { "H1", "L2" };
enum TransferType { ParentToSub, SubToParent };
static const char *transfer_str[] = { "ParentToSub", "SubToParent" };
static const char *element_str[] = { "POINT", "SEGMENT", "TRIANGLE", "QUADRILATERAL", "TETRAHEDRON", "HEXAHEDRON", "WEDGE", "PYRAMID" };

FiniteElementCollection* create_fec(FECType fectype, int p, int dim)
{
  switch(fectype)
  {
    case H1:
      return new H1_FECollection(p, dim);
      break;
    case L2:
      return new L2_FECollection(p, dim);
      break;
  }

  return nullptr;
}

void test_2d(int dim, Element::Type element_type, FECType fectype, int polynomial_order, int mesh_polynomial_order, TransferType transfer_type)
{
   Mesh mesh = Mesh::MakeCartesian2D(5, 5, element_type, true, 1.0, 1.0,
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
   mesh.SetCurvature(mesh_polynomial_order);
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

   FiniteElementCollection *fec = create_fec(fectype, polynomial_order, dim);
   FiniteElementSpace parent_fes(&mesh, fec);

   GridFunction parent_gf(&parent_fes);
   parent_gf = 0.0;

   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      return y + 0.05 * sin(x*2.0*M_PI);
   });

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;

   SubMesh submesh = SubMesh::CreateFromDomain(mesh, subdomain_attributes);
   FiniteElementSpace sub_fes(&submesh, fec);

   GridFunction sub_gf(&sub_fes);
   sub_gf = 0.0;

   double l2err = 1e12;
   if (transfer_type == ParentToSub)
   {
     parent_gf.ProjectCoefficient(coeff);
     SubMesh::Transfer(parent_gf, sub_gf);
     l2err = sub_gf.ComputeL2Error(coeff);
     REQUIRE(l2err < 1e-2);
   } else if (transfer_type == SubToParent)
   {
     sub_gf.ProjectCoefficient(coeff);
     SubMesh::Transfer(sub_gf, parent_gf);
     l2err = parent_gf.ComputeL2Error(coeff);
     REQUIRE(l2err < 0.6);
   }

}

void test_3d(int dim, Element::Type element_type, FECType fectype, int polynomial_order, int mesh_polynomial_order, TransferType transfer_type)
{
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
   mesh.SetCurvature(mesh_polynomial_order);
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

   FiniteElementCollection *fec = create_fec(fectype, polynomial_order, dim);
   FiniteElementSpace parent_fes(&mesh, fec);

   GridFunction parent_gf(&parent_fes);
   parent_gf = 0.0;

   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      double z = coords(2);
      return y + 0.05 * sin(x*2.0*M_PI) + z;
   });

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;

   SubMesh submesh = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);
   FiniteElementSpace sub_fes(&submesh, fec);

   GridFunction sub_gf(&sub_fes);
   sub_gf = 0.0;

   double l2err = 1e12;
   if (transfer_type == ParentToSub)
   {
     parent_gf.ProjectCoefficient(coeff);
     SubMesh::Transfer(parent_gf, sub_gf);
     l2err = sub_gf.ComputeL2Error(coeff);
     REQUIRE(l2err < 1e-2);
   } else if (transfer_type == SubToParent)
   {
     sub_gf.ProjectCoefficient(coeff);
     SubMesh::Transfer(sub_gf, parent_gf);
     l2err = parent_gf.ComputeL2Error(coeff);
     REQUIRE(l2err < 0.6);
   }

   // char vishost[] = "orchid";
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

TEST_CASE("GENERATE TEST", "[SubMesh]")
{
  auto fectype = GENERATE(FECType::H1, FECType::L2);
  auto polynomial_order = GENERATE(1, 4);
  auto mesh_polynomial_order = GENERATE(1, 2);
  auto transfer_type = GENERATE(TransferType::ParentToSub, TransferType::SubToParent);
  const char separator = ' ';

  SECTION("2D") {
    int dim = 2;
    auto element = GENERATE(Element::QUADRILATERAL, Element::TRIANGLE);
    out << std::setw(2) << std::setfill(separator) << dim;
    out << std::setw(2) << std::setfill(separator) << mesh_polynomial_order;
    out << std::setw(14) << std::setfill(separator) << element_str[element];
    out << std::setw(3) << std::setfill(separator) << fectype_str[fectype];
    out << std::setw(2) << std::setfill(separator) << polynomial_order;
    out << std::setw(12) << std::setfill(separator) << transfer_str[transfer_type];
    out << std::endl;
    test_2d(dim, element, fectype, polynomial_order, mesh_polynomial_order, transfer_type);
  }

  SECTION("3D") {
    int dim = 3;
    auto element = GENERATE(Element::HEXAHEDRON, Element::TETRAHEDRON);
    out << std::setw(2) << std::setfill(separator) << dim;
    out << std::setw(2) << std::setfill(separator) << mesh_polynomial_order;
    out << std::setw(14) << std::setfill(separator) << element_str[element];
    out << std::setw(3) << std::setfill(separator) << fectype_str[fectype];
    out << std::setw(2) << std::setfill(separator) << polynomial_order;
    out << std::setw(12) << std::setfill(separator) << transfer_str[transfer_type];
    out << std::endl;
    test_3d(dim, element, fectype, polynomial_order, mesh_polynomial_order, transfer_type);
  }
}

