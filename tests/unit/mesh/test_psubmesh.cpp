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

void multidomain_test_2d(FECType fec_type)
{
   const int p = 2;
   double Hy = 1.0;
   Mesh serial_parent_mesh = Mesh::MakeCartesian2D(5, 5,
                                                   Element::QUADRILATERAL, true, 1.0, Hy,
                                                   false);

   for (int i = 0; i < serial_parent_mesh.GetNBE(); i++)
   {
      Element *el = serial_parent_mesh.GetBdrElement(i);
      el->SetAttribute(1);

      Array<int> vertices;
      el->GetVertices(vertices);

      bool all_vtx_inside = true;
      for (int j = 0; j < vertices.Size(); j++)
      {
         if (serial_parent_mesh.GetVertex(vertices[j])[1] < 1.0)
         {
            all_vtx_inside = false;
         }
      }
      if (all_vtx_inside)
      {
         el->SetAttribute(2);
      }
   }
   for (int i = 0; i < serial_parent_mesh.GetNE(); i++)
   {
      Element *el = serial_parent_mesh.GetElement(i);
      el->SetAttribute(1);

      Array<int> vertices;
      el->GetVertices(vertices);

      for (int j = 0; j < vertices.Size(); j++)
      {
         double *coords = serial_parent_mesh.GetVertex(vertices[j]);

         if (coords[0] >= 0.25 &&
             coords[0] <= 0.75 &&
             coords[1] >= 0.25 &&
             coords[1] <= 0.75)
         {
            el->SetAttribute(2);
         }
      }
   }
   serial_parent_mesh.SetAttributes();
   serial_parent_mesh.EnsureNodes();
   serial_parent_mesh.SetCurvature(p);

   auto node_movement_coeff = VectorFunctionCoefficient(
                                 serial_parent_mesh.Dimension(),
                                 [](const Vector &coords, Vector &u)
   {
      double x = coords(0);
      double y = coords(1);

      u(0) = x;
      u(1) = y + 0.05 * sin(x * 2.0 * M_PI);
   });

   serial_parent_mesh.Transform(node_movement_coeff);

   ParMesh parent_mesh(MPI_COMM_WORLD, serial_parent_mesh);

   Array<int> domain1(1);
   domain1[0] = 1;

   Array<int> boundary1(1);
   boundary1[0] = 2;

   auto domain_submesh = ParSubMesh::CreateFromDomain(parent_mesh,
                                                      domain1);

   auto boundary_submesh = ParSubMesh::CreateFromDomain(parent_mesh,
                                                        boundary1);

   FiniteElementCollection *fec = create_fec(fec_type, p, parent_mesh.Dimension());

   ParFiniteElementSpace parent_fes(&parent_mesh, fec);
   ParGridFunction parent_gf(&parent_fes);

   ParFiniteElementSpace domain1_fes(&domain_submesh, fec);
   ParGridFunction domain1_gf(&domain1_fes);

   FiniteElementCollection *surface_fec = create_fec(fec_type, p,
                                                     domain_submesh.Dimension());
   ParFiniteElementSpace boundary1_fes(&boundary_submesh, surface_fec);
   ParGridFunction boundary1_gf(&boundary1_fes);

   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      return y + 0.05 * sin(x * 2.0 * M_PI);
   });

   parent_gf.ProjectCoefficient(coeff);

   Vector tmp;

   ParGridFunction parent_gf_ex(&parent_fes);
   parent_gf_ex.ProjectCoefficient(coeff);

   ParGridFunction domain1_gf_ex(&domain1_fes);
   domain1_gf_ex.ProjectCoefficient(coeff);

   ParGridFunction boundary1_gf_ex(&boundary1_fes);
   boundary1_gf_ex.ProjectCoefficient(coeff);

   auto CHECK_GLOBAL_NORM = [](Vector &v)
   {
      double norm_local = v.Norml2(), norm_global = 0.0;
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(norm_global < 1e-8);
   };

   SECTION("ParentToSubMesh")
   {
      SECTION("Volume to matching volume")
      {
         ParSubMesh::Transfer(parent_gf, domain1_gf);
         tmp = domain1_gf_ex;
         tmp -= domain1_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Volume to matching surface")
      {
         ParSubMesh::Transfer(parent_gf, boundary1_gf);
         tmp = boundary1_gf_ex;
         tmp -= boundary1_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
   }
   SECTION("SubMeshToParent")
   {
      SECTION("Volume to matching volume")
      {
         parent_gf.ProjectCoefficient(coeff);
         domain1_gf.ProjectCoefficient(coeff);
         ParSubMesh::Transfer(domain1_gf, parent_gf);
         tmp = parent_gf_ex;
         tmp -= parent_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Surface to matching surface in volume")
      {
         boundary1_gf.ProjectCoefficient(coeff);
         ParSubMesh::Transfer(boundary1_gf, parent_gf);
         tmp = parent_gf_ex;
         tmp -= parent_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
   }
   delete surface_fec;
   delete fec;
}

void multidomain_test_3d(FECType fec_type)
{
   const int p = 2;
   // Circle: sideset 1
   // Domain boundary: sideset 2
   Mesh *serial_parent_mesh = new
   Mesh("../../miniapps/multidomain/multidomain-hex.mesh");
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

   int num_local_be = cylinder_surface_submesh.GetNBE();
   int num_global_be = 0;
   MPI_Allreduce(&num_local_be, &num_global_be, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   REQUIRE(num_global_be == 16);
   REQUIRE(cylinder_surface_submesh.bdr_attributes[0] == 900);

   FiniteElementCollection *fec = create_fec(fec_type, p, parent_mesh.Dimension());

   ParFiniteElementSpace parent_fes(&parent_mesh, fec);
   ParGridFunction parent_gf(&parent_fes);

   ParFiniteElementSpace cylinder_fes(&cylinder_submesh, fec);
   ParGridFunction cylinder_gf(&cylinder_fes);

   ParFiniteElementSpace outer_fes(&outer_submesh, fec);
   ParGridFunction outer_gf(&outer_fes);

   FiniteElementCollection *surface_fec = create_fec(fec_type, p,
                                                     cylinder_surface_submesh.Dimension());
   ParFiniteElementSpace cylinder_surface_fes(&cylinder_surface_submesh,
                                              surface_fec);
   ParGridFunction cylinder_surface_gf(&cylinder_surface_fes);

   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      double x = coords(0);
      double y = coords(1);
      double z = coords(2);
      return y + 0.05 * sin(x * 2.0 * M_PI) + z;
   });

   parent_gf.ProjectCoefficient(coeff);

   Vector tmp;

   ParGridFunction parent_gf_ex(&parent_fes);
   parent_gf_ex.ProjectCoefficient(coeff);

   ParGridFunction cylinder_gf_ex(&cylinder_fes);
   cylinder_gf_ex.ProjectCoefficient(coeff);

   ParGridFunction cylinder_surface_gf_ex(&cylinder_surface_fes);
   cylinder_surface_gf_ex.ProjectCoefficient(coeff);

   ParGridFunction outer_gf_ex(&outer_fes);
   outer_gf_ex.ProjectCoefficient(coeff);

   auto CHECK_GLOBAL_NORM = [](Vector &v)
   {
      double norm_local = v.Norml2(), norm_global = 0.0;
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      REQUIRE(norm_global < 1e-8);
   };

   SECTION("ParentToSubMesh")
   {
      SECTION("Volume to matching volume")
      {
         ParSubMesh::Transfer(parent_gf, cylinder_gf);
         tmp = cylinder_gf_ex;
         tmp -= cylinder_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Volume to matching surface")
      {
         ParSubMesh::Transfer(parent_gf, cylinder_surface_gf);
         tmp = cylinder_surface_gf_ex;
         tmp -= cylinder_surface_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
   }
   SECTION("SubMeshToParent")
   {
      SECTION("Volume to matching volume")
      {
         parent_gf.ProjectCoefficient(coeff);
         cylinder_gf.ProjectCoefficient(coeff);
         ParSubMesh::Transfer(cylinder_gf, parent_gf);
         tmp = parent_gf_ex;
         tmp -= parent_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Volume to matching volume")
      {
         outer_gf.ProjectCoefficient(coeff);
         ParSubMesh::Transfer(outer_gf, parent_gf);
         tmp = parent_gf_ex;
         tmp -= parent_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Surface to matching surface in volume")
      {
         cylinder_surface_gf.ProjectCoefficient(coeff);
         ParSubMesh::Transfer(cylinder_surface_gf, parent_gf);
         tmp = parent_gf_ex;
         tmp -= parent_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
   }
   SECTION("SubMeshToSubMesh")
   {
      SECTION("Volume to matching volume")
      {
         cylinder_gf.ProjectCoefficient(coeff);
         outer_gf.ProjectCoefficient(coeff);
         outer_gf_ex.ProjectCoefficient(coeff);
         ParSubMesh::Transfer(cylinder_gf, outer_gf);
         tmp = outer_gf_ex;
         tmp -= outer_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Volume to matching surface on volume")
      {
         cylinder_gf.ProjectCoefficient(coeff);
         outer_gf.ProjectCoefficient(coeff);
         cylinder_gf_ex.ProjectCoefficient(coeff);
         ParSubMesh::Transfer(outer_gf, cylinder_gf);
         tmp = cylinder_gf_ex;
         tmp -= cylinder_gf;
         CHECK_GLOBAL_NORM(tmp);
      }

      SECTION("Volume to matching surface")
      {
         cylinder_gf.ProjectCoefficient(coeff);
         cylinder_surface_gf_ex.ProjectCoefficient(coeff);
         ParSubMesh::Transfer(cylinder_gf, cylinder_surface_gf);
         tmp = cylinder_surface_gf_ex;
         tmp -= cylinder_surface_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
   }
   delete surface_fec;
   delete fec;
}

TEST_CASE("ParSubMesh", "[Parallel],[ParSubMesh]")
{
   auto fec_type = GENERATE(FECType::H1, FECType::L2);
   multidomain_test_2d(fec_type);
   multidomain_test_3d(fec_type);
}

} // namespace ParSubMeshTests

#endif // MFEM_USE_MPI
