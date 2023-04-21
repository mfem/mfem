// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include <memory>
#include "unit_tests.hpp"

using namespace mfem;

enum FECType
{
   H1,
   L2
};
enum FieldType
{
   SCALAR,
   VECTOR
};
enum TransferType
{
   ParentToSub,
   SubToParent
};

FiniteElementCollection *create_fec(FECType fec_type, int p, int dim)
{
   switch (fec_type)
   {
      case H1:
         return new H1_FECollection(p, dim);
         break;
      case L2:
         return new L2_FECollection(p, dim, BasisType::GaussLobatto);
         break;
   }

   return nullptr;
}

void test_2d(Element::Type element_type,
             FECType fec_type,
             FieldType field_type,
             int polynomial_order,
             int mesh_polynomial_order,
             TransferType transfer_type,
             SubMesh::From from)
{
   constexpr int dim = 2;
   const int vdim = (field_type == FieldType::SCALAR) ? 1 : dim;
   double Hy = 1.0;
   Mesh mesh = Mesh::MakeCartesian2D(5, 5, element_type, true, 1.0, Hy, false);

   if (from == SubMesh::From::Boundary)
   {
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         Element *el = mesh.GetBdrElement(i);
         el->SetAttribute(1);

         Array<int> vertices;
         el->GetVertices(vertices);

         bool all_vtx_inside = true;
         for (int j = 0; j < vertices.Size(); j++)
         {
            if (mesh.GetVertex(vertices[j])[1] < 1.0)
            {
               all_vtx_inside = false;
            }
         }
         if (all_vtx_inside)
         {
            el->SetAttribute(2);
         }
      }
   }
   else if (from == SubMesh::From::Domain)
   {
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
   }
   mesh.SetAttributes();

   // Deform original mesh
   mesh.EnsureNodes();
   mesh.SetCurvature(mesh_polynomial_order);

   auto node_movement_coeff = VectorFunctionCoefficient(mesh.Dimension(),
                                                        [](const Vector &coords, Vector &u)
   {
      double x = coords(0);
      double y = coords(1);

      u(0) = x;
      u(1) = y + 0.05 * sin(x * 2.0 * M_PI);
   });

   mesh.Transform(node_movement_coeff);

   FiniteElementCollection *fec = create_fec(fec_type, polynomial_order, dim);
   FiniteElementSpace parent_fes(&mesh, fec, vdim);

   GridFunction parent_gf(&parent_fes);
   parent_gf = 0.0;

   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      return y + 0.05 * sin(x * 2.0 * M_PI);
   });

   auto vcoeff = VectorFunctionCoefficient(dim, [](const Vector &coords,
                                                   Vector &V)
   {
      V.SetSize(2);
      double x = coords(0);
      double y = coords(1);

      V(0) = y + 0.05 * sin(x * 2.0 * M_PI);
      V(1) = x + 0.05 * sin(y * 2.0 * M_PI);
   });

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;

   SubMesh* submesh = nullptr;
   if (from == SubMesh::From::Domain)
   {
      submesh = new SubMesh(SubMesh::CreateFromDomain(mesh, subdomain_attributes));
   }
   else
   {
      submesh = new SubMesh(SubMesh::CreateFromBoundary(mesh, subdomain_attributes));
   }

   REQUIRE(submesh->GetNE() != 0);

   FiniteElementCollection *sub_fec = create_fec(fec_type, polynomial_order,
                                                 submesh->Dimension());
   FiniteElementSpace sub_fes(submesh, sub_fec, vdim);

   GridFunction sub_gf(&sub_fes);
   sub_gf = 0.0;

   if (transfer_type == ParentToSub)
   {
      GridFunction sub_ex_gf(&sub_fes);

      if (vdim == 1)
      {
         parent_gf.ProjectCoefficient(coeff);
         sub_ex_gf.ProjectCoefficient(coeff);
      }
      else
      {
         parent_gf.ProjectCoefficient(vcoeff);
         sub_ex_gf.ProjectCoefficient(vcoeff);
      }
      SubMesh::Transfer(parent_gf, sub_gf);

      sub_gf -= sub_ex_gf;
      REQUIRE(sub_gf.Norml2() < 1e-10);
   }
   else if (transfer_type == SubToParent)
   {
      GridFunction parent_ex_gf(&parent_fes);

      if (vdim == 1)
      {
         parent_gf.ProjectCoefficient(coeff);
         sub_gf.ProjectCoefficient(coeff);
         parent_ex_gf.ProjectCoefficient(coeff);
      }
      else
      {
         parent_gf.ProjectCoefficient(vcoeff);
         sub_gf.ProjectCoefficient(vcoeff);
         parent_ex_gf.ProjectCoefficient(vcoeff);
      }

      SubMesh::Transfer(sub_gf, parent_gf);

      parent_gf -= parent_ex_gf;
      REQUIRE(parent_gf.Norml2() < 1e-10);
   }
   delete submesh;
   delete sub_fec;
   delete fec;
}

void test_3d(Element::Type element_type,
             FECType fec_type,
             FieldType field_type,
             int polynomial_order,
             int mesh_polynomial_order,
             TransferType transfer_type,
             SubMesh::From from)
{
   constexpr int dim = 3;
   const int vdim = (field_type == FieldType::SCALAR) ? 1 : dim;
   double Hy = 1.0;
   Mesh mesh = Mesh::MakeCartesian3D(5, 5, 5, element_type, 1.0, Hy, 1.0, false);

   if (from == SubMesh::From::Boundary)
   {
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
   }
   else if (from == SubMesh::From::Domain)
   {
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         Element *el = mesh.GetElement(i);
         el->SetAttribute(1);

         Array<int> vertices;
         el->GetVertices(vertices);

         bool all_vtx_inside = true;
         for (int j = 0; j < vertices.Size(); j++)
         {
            if (mesh.GetVertex(vertices[j])[1] > 0.5 * Hy)
            {
               all_vtx_inside = false;
            }
         }
         if (all_vtx_inside)
         {
            el->SetAttribute(2);
         }
      }
   }
   mesh.SetAttributes();

   // Deform original mesh
   mesh.EnsureNodes();
   mesh.SetCurvature(mesh_polynomial_order);

   auto node_movement_coeff = VectorFunctionCoefficient(mesh.Dimension(),
                                                        [](const Vector &coords, Vector &u)
   {
      double x = coords(0);
      double y = coords(1);
      double z = coords(2);

      u(0) = x;
      u(1) = y + 0.05 * sin(x * 2.0 * M_PI);
      u(2) = z;
   });

   mesh.Transform(node_movement_coeff);

   FiniteElementCollection *fec = create_fec(fec_type, polynomial_order, dim);
   FiniteElementSpace parent_fes(&mesh, fec, vdim);

   GridFunction parent_gf(&parent_fes);

   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      double z = coords(2);
      return y + 0.05 * sin(x * 2.0 * M_PI) + z;
   });

   auto vcoeff = VectorFunctionCoefficient(dim, [](const Vector &coords,
                                                   Vector &V)
   {
      V.SetSize(3);
      double x = coords(0);
      double y = coords(1);
      double z = coords(2);

      V(0) = y + 0.05 * sin(x * 2.0 * M_PI) + z;
      V(1) = z + 0.05 * sin(y * 2.0 * M_PI) + x;
      V(2) = x + 0.05 * sin(z * 2.0 * M_PI) + y;
   });

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;

   SubMesh* submesh = nullptr;
   if (from == SubMesh::From::Domain)
   {
      submesh = new SubMesh(SubMesh::CreateFromDomain(mesh, subdomain_attributes));
   }
   else
   {
      submesh = new SubMesh(SubMesh::CreateFromBoundary(mesh, subdomain_attributes));
   }

   REQUIRE(submesh->GetNE() != 0);

   FiniteElementCollection *sub_fec = create_fec(fec_type, polynomial_order,
                                                 submesh->Dimension());
   FiniteElementSpace sub_fes(submesh, sub_fec, vdim);

   GridFunction sub_gf(&sub_fes);
   sub_gf = 0.0;

   if (transfer_type == ParentToSub)
   {
      GridFunction sub_ex_gf(&sub_fes);

      if (vdim == 1)
      {
         parent_gf.ProjectCoefficient(coeff);
         sub_ex_gf.ProjectCoefficient(coeff);
      }
      else
      {
         parent_gf.ProjectCoefficient(vcoeff);
         sub_ex_gf.ProjectCoefficient(vcoeff);
      }
      SubMesh::Transfer(parent_gf, sub_gf);

      REQUIRE(sub_gf.Norml2() != 0.0);

      sub_gf -= sub_ex_gf;
      REQUIRE(sub_gf.Norml2() < 1e-10);
   }
   else if (transfer_type == SubToParent)
   {
      GridFunction parent_ex_gf(&parent_fes);

      if (vdim == 1)
      {
         parent_gf.ProjectCoefficient(coeff);
         sub_gf.ProjectCoefficient(coeff);
         parent_ex_gf.ProjectCoefficient(coeff);
      }
      else
      {
         parent_gf.ProjectCoefficient(vcoeff);
         sub_gf.ProjectCoefficient(vcoeff);
         parent_ex_gf.ProjectCoefficient(vcoeff);
      }

      SubMesh::Transfer(sub_gf, parent_gf);

      REQUIRE(parent_gf.Norml2() != 0.0);

      parent_gf -= parent_ex_gf;
      REQUIRE(parent_gf.Norml2() < 1e-10);
   }
   delete submesh;
   delete sub_fec;
   delete fec;
}

TEST_CASE("SubMesh", "[SubMesh]")
{
   int polynomial_order = 4;
   int mesh_polynomial_order = 2;
   auto fec_type = GENERATE(FECType::H1, FECType::L2);
   auto field_type = GENERATE(FieldType::SCALAR, FieldType::VECTOR);
   auto transfer_type = GENERATE(TransferType::ParentToSub,
                                 TransferType::SubToParent);
   auto from = GENERATE(SubMesh::From::Domain,
                        SubMesh::From::Boundary);

   SECTION("2D")
   {
      auto element = GENERATE(Element::QUADRILATERAL, Element::TRIANGLE);
      if (fec_type == FECType::L2 && from == SubMesh::From::Boundary && false)
      {
         return;
      }
      test_2d(element, fec_type, field_type, polynomial_order,
              mesh_polynomial_order, transfer_type, from);
   }

   SECTION("3D")
   {
      auto element = GENERATE(Element::HEXAHEDRON, Element::TETRAHEDRON);
      if (fec_type == FECType::L2 &&
          from == SubMesh::From::Boundary && false)
      {
         return;
      }
      test_3d(element, fec_type, field_type, polynomial_order,
              mesh_polynomial_order, transfer_type, from);
   }
}
