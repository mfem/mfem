// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

#ifdef MFEM_USE_MPI

namespace ParSubMeshTests
{
enum FECType
{
   H1,
   L2
};

enum TransferType
{
   ParentToSub,
   SubToParent
};

FiniteElementCollection *create_fec(FECType fectype, int p, int dim)
{
   switch (fectype)
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
             int polynomial_order,
             int mesh_polynomial_order,
             TransferType transfer_type,
             SubMesh::From from)
{
   constexpr int dim = 2;
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

   // Create parallel mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   FiniteElementCollection *fec = create_fec(fec_type, polynomial_order, dim);
   ParFiniteElementSpace parent_fes(&pmesh, fec);

   ParGridFunction parent_gf(&parent_fes);

   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      return y + 0.05 * sin(x * 2.0 * M_PI);
   });

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;

   ParSubMesh* submesh = nullptr;
   if (from == SubMesh::From::Domain)
   {
      submesh = new ParSubMesh(ParSubMesh::CreateFromDomain(pmesh,
                                                            subdomain_attributes));
   }
   else
   {
      submesh = new ParSubMesh(ParSubMesh::CreateFromBoundary(pmesh,
                                                              subdomain_attributes));
   }

   int ne_local = submesh->GetNE();
   int ne_global = 0;
   MPI_Allreduce(&ne_local, &ne_global, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   REQUIRE(ne_global != 0);

   FiniteElementCollection *sub_fec = create_fec(fec_type, polynomial_order,
                                                 submesh->Dimension());
   ParFiniteElementSpace sub_fes(submesh, sub_fec);

   ParGridFunction sub_gf(&sub_fes);
   sub_gf = 0.0;

   if (transfer_type == ParentToSub)
   {
      parent_gf.ProjectCoefficient(coeff);
      ParSubMesh::Transfer(parent_gf, sub_gf);

      ParGridFunction sub_ex_gf(&sub_fes);
      sub_ex_gf.ProjectCoefficient(coeff);

      double norm_local = sub_gf.Norml2();
      double norm_global = 0.0;
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(norm_global != 0.0);

      sub_gf -= sub_ex_gf;
      norm_local = sub_gf.Norml2();
      norm_global = 0.0;
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(norm_global < 1e-10);
   }
   else if (transfer_type == SubToParent)
   {
      parent_gf.ProjectCoefficient(coeff);

      sub_gf.ProjectCoefficient(coeff);
      ParSubMesh::Transfer(sub_gf, parent_gf);

      ParGridFunction parent_ex_gf(&parent_fes);
      parent_ex_gf.ProjectCoefficient(coeff);

      double norm_local = parent_gf.Norml2();
      double norm_global = 0.0;
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(norm_global != 0.0);

      parent_gf -= parent_ex_gf;
      norm_local = parent_gf.Norml2();
      norm_global = 0.0;
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(norm_global < 1e-10);
   }
}

void test_3d(Element::Type element_type,
             FECType fec_type,
             int polynomial_order,
             int mesh_polynomial_order,
             TransferType transfer_type,
             SubMesh::From from)
{
   constexpr int dim = 3;
   double Hy = 1.0;
   Mesh mesh = Mesh::MakeCartesian3D(5, 5, 5, element_type, 1.0, Hy, 1.0, false);

   if (from == SubMesh::From::Boundary)
   {

      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         Element *el = mesh.GetBdrElement(i);

         Array<int> vtx;
         el->GetVertices(vtx);

         for (int v = 0; v < vtx.Size(); v++)
         {
            double* c = mesh.GetVertex(vtx[v]);
            if (c[1] == 0.0)
            {
               el->SetAttribute(2);
            }
            else
            {
               el->SetAttribute(1);
            }
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

   // Create parallel mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   FiniteElementCollection *fec = create_fec(fec_type, polynomial_order, dim);
   ParFiniteElementSpace parent_fes(&pmesh, fec);

   ParGridFunction parent_gf(&parent_fes);

   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      double z = coords(2);
      return y + 0.05 * sin(x * 2.0 * M_PI) + z;
   });

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;

   ParSubMesh* submesh = nullptr;
   if (from == SubMesh::From::Domain)
   {
      submesh = new ParSubMesh(ParSubMesh::CreateFromDomain(pmesh,
                                                            subdomain_attributes));
   }
   else
   {
      submesh = new ParSubMesh(ParSubMesh::CreateFromBoundary(pmesh,
                                                              subdomain_attributes));
   }

   int ne_local = submesh->GetNE();
   int ne_global = 0;
   MPI_Allreduce(&ne_local, &ne_global, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   REQUIRE(ne_global != 0);

   FiniteElementCollection *sub_fec = create_fec(fec_type, polynomial_order,
                                                 submesh->Dimension());
   ParFiniteElementSpace sub_fes(submesh, sub_fec);

   ParGridFunction sub_gf(&sub_fes);
   sub_gf = 0.0;

   if (transfer_type == ParentToSub)
   {
      parent_gf.ProjectCoefficient(coeff);
      ParSubMesh::Transfer(parent_gf, sub_gf);

      ParGridFunction sub_ex_gf(&sub_fes);
      sub_ex_gf.ProjectCoefficient(coeff);

      double norm_local = sub_gf.Norml2();
      double norm_global = 0.0;
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(norm_global != 0.0);

      sub_gf -= sub_ex_gf;
      norm_local = sub_gf.Norml2();
      norm_global = 0.0;
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(norm_global < 1e-10);
   }
   else if (transfer_type == SubToParent)
   {
      parent_gf.ProjectCoefficient(coeff);

      sub_gf.ProjectCoefficient(coeff);
      ParSubMesh::Transfer(sub_gf, parent_gf);

      ParGridFunction parent_ex_gf(&parent_fes);
      parent_ex_gf.ProjectCoefficient(coeff);

      double norm_local = parent_gf.Norml2();
      double norm_global = 0.0;
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(norm_global != 0.0);

      parent_gf -= parent_ex_gf;
      norm_local = parent_gf.Norml2();
      norm_global = 0.0;
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(norm_global < 1e-10);
   }
}

TEST_CASE("ParSubMesh", "[Parallel],[ParSubMesh]")
{
   int polynomial_order = 4;
   int mesh_polynomial_order = 2;
   auto fec_type = GENERATE(FECType::H1, FECType::L2);
   auto transfer_type = GENERATE(TransferType::ParentToSub,
                                 TransferType::SubToParent);
   auto from = GENERATE(SubMesh::From::Domain,
                        SubMesh::From::Boundary);

   SECTION("2D")
   {
      auto element = GENERATE(Element::QUADRILATERAL, Element::TRIANGLE);
      if (fec_type == FECType::L2 &&
          transfer_type == TransferType::SubToParent &&
          from == SubMesh::From::Boundary)
      {
         return;
      }
      test_2d(element, fec_type, polynomial_order, mesh_polynomial_order,
              transfer_type, from);
   }

   SECTION("3D")
   {
      auto element = GENERATE(Element::HEXAHEDRON, Element::TETRAHEDRON);
      if (fec_type == FECType::L2 &&
          transfer_type == TransferType::SubToParent &&
          from == SubMesh::From::Boundary)
      {
         return;
      }
      test_3d(element, fec_type, polynomial_order, mesh_polynomial_order,
              transfer_type, from);
   }
}

TEST_CASE("ParSubMeshDirectTransfer", "[Parallel],[ParSubMeshDirectTransfer]")
{
   // Circle: sideset 1
   // Domain boundary: sideset 2
   Mesh *serial_parent_mesh = new Mesh("test.e");
   ParMesh parent_mesh(MPI_COMM_WORLD, *serial_parent_mesh);
   delete serial_parent_mesh;

   Array<int> cylinder_domain_attributes(1);
   cylinder_domain_attributes[0] = 1;

   Array<int> outer_domain_attributes(1);
   outer_domain_attributes[0] = 2;

   Array<int> cylinder_surface_attributes(1);
   cylinder_surface_attributes[0] = 9;

   auto cylinder_submesh = ParSubMesh::CreateFromDomain(parent_mesh,
                                                        cylinder_domain_attributes);

   auto outer_submesh = ParSubMesh::CreateFromDomain(parent_mesh,
                                                     outer_domain_attributes);

   auto cylinder_surface_submesh = ParSubMesh::CreateFromBoundary(parent_mesh,
                                                                  cylinder_surface_attributes);

   FiniteElementCollection *fec = create_fec(FECType::H1, 2, 3);

   ParFiniteElementSpace parent_fes(&parent_mesh, fec);
   ParGridFunction parent_gf(&parent_fes);

   ParFiniteElementSpace cylinder_fes(&cylinder_submesh, fec);
   ParGridFunction cylinder_gf(&cylinder_fes);

   ParFiniteElementSpace outer_fes(&outer_submesh, fec);
   ParGridFunction outer_gf(&outer_fes);

   ParFiniteElementSpace cylinder_surface_fes(&cylinder_surface_submesh, fec);
   ParGridFunction cylinder_surface_gf(&cylinder_surface_fes);

   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      double z = coords(2);
      return y + 0.05 * sin(x * 2.0 * M_PI) + z;
   });

   auto flipped_coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      double z = coords(2);
      return -(y + 0.05 * sin(x * 2.0 * M_PI) + z);
   });

   parent_gf.ProjectCoefficient(coeff);
   outer_gf = 0.0;

   ParSubMesh::Transfer(parent_gf, cylinder_gf);
   ParSubMesh::Transfer(cylinder_gf, cylinder_surface_gf);
   ParSubMesh::Transfer(cylinder_surface_gf, outer_gf);

   char vishost[] = "128.15.198.77";
   int  visport   = 19916;

   socketstream parent_mesh_socket(vishost, visport);
   parent_mesh_socket.precision(8);
   parent_mesh_socket << "solution\n" << parent_mesh << parent_gf;
   parent_mesh_socket << "keys ml" << std::flush;

   socketstream cylinder_domain_socket(vishost, visport);
   cylinder_domain_socket.precision(8);
   cylinder_domain_socket << "solution\n" << cylinder_submesh << cylinder_gf;
   cylinder_domain_socket << "keys ml" << std::flush;

   socketstream outer_domain_socket(vishost, visport);
   outer_domain_socket.precision(8);
   outer_domain_socket << "solution\n" << outer_submesh << outer_gf;
   outer_domain_socket << "keys ml" << std::flush;

   socketstream cylinder_surface_socket(vishost, visport);
   cylinder_surface_socket.precision(8);
   cylinder_surface_socket << "solution\n" << cylinder_surface_submesh <<
                           cylinder_surface_gf;
   cylinder_surface_socket << "keys ml" << std::flush;
}

} // namespace ParSubMeshTests

#endif // MFEM_USE_MPI
