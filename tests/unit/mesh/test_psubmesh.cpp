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
#include "mesh_test_utils.hpp"

using namespace mfem;

#ifdef MFEM_USE_MPI

namespace ParSubMeshTests
{

void CHECK_GLOBAL_NORM(Vector &v, bool small = true)
{
   real_t norm_local = v.Norml2(), norm_global = 0.0;
   MPI_Allreduce(&norm_local, &norm_global, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, MPI_COMM_WORLD);
   if (small)
   {
      REQUIRE(norm_global < 1e-8);
   }
   else
   {
      REQUIRE(norm_global > 1e-8);
   }
};


FiniteElementCollection *create_surf_fec(FECType fectype, int p, int dim)
{
   switch (fectype)
   {
      case FECType::H1:
         return new H1_FECollection(p, dim);
      case FECType::ND:
         return new ND_FECollection(p, dim);
      case FECType::RT:
         return new L2_FECollection(p - 1, dim, BasisType::GaussLegendre,
                                    FiniteElement::INTEGRAL);
      case FECType::L2:
         return new L2_FECollection(p, dim, BasisType::GaussLobatto);
   }

   return nullptr;
}
class SurfaceNormalCoef : public VectorCoefficient
{
public:
   SurfaceNormalCoef(int dim) : VectorCoefficient(dim) {}

   using VectorCoefficient::Eval;

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      V.SetSize(vdim);
      CalcOrtho(T.Jacobian(), V);
      V /= V.Norml2();
   }

};

void multidomain_test_2d(FECType fec_type)
{
   constexpr int dim = 2;
   const int p = 2;
   real_t Hy = 1.0;
   Mesh serial_parent_mesh = Mesh::MakeCartesian2D(5, 5,
                                                   Element::QUADRILATERAL,
                                                   true, 1.0, Hy,
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
         real_t *coords = serial_parent_mesh.GetVertex(vertices[j]);

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
      real_t x = coords(0);
      real_t y = coords(1);

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

   auto boundary_submesh = ParSubMesh::CreateFromBoundary(parent_mesh,
                                                          boundary1);

   FiniteElementCollection *fec = create_fec(fec_type, p,
                                             parent_mesh.Dimension());

   ParFiniteElementSpace parent_fes(&parent_mesh, fec);
   ParGridFunction parent_gf(&parent_fes);
   ParGridFunction parent_gf_ex(&parent_fes);

   ParFiniteElementSpace domain1_fes(&domain_submesh, fec);
   ParGridFunction domain1_gf(&domain1_fes);
   ParGridFunction domain1_gf_ex(&domain1_fes);

   FiniteElementCollection *surface_fec =
      create_surf_fec(fec_type, p, boundary_submesh.Dimension());
   ParFiniteElementSpace boundary1_fes(&boundary_submesh, surface_fec);
   ParGridFunction boundary1_gf(&boundary1_fes);
   ParGridFunction boundary1_gf_ex(&boundary1_fes);

   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      real_t x = coords(0);
      real_t y = coords(1);
      return y + 0.05 * sin(x * 2.0 * M_PI);
   });

   auto vcoeff = VectorFunctionCoefficient(dim, [](const Vector &coords,
                                                   Vector &V)
   {
      V.SetSize(2);
      real_t x = coords(0);
      real_t y = coords(1);

      V(0) = y + 0.05 * sin(x * 2.0 * M_PI);
      V(1) = x + 0.05 * sin(y * 2.0 * M_PI);
   });

   SurfaceNormalCoef normalcoeff(dim);
   InnerProductCoefficient nvcoeff(normalcoeff, vcoeff);

   if (fec_type == FECType::H1 || fec_type == FECType::L2)
   {
      parent_gf.ProjectCoefficient(coeff);
      parent_gf_ex.ProjectCoefficient(coeff);
      domain1_gf_ex.ProjectCoefficient(coeff);
      boundary1_gf_ex.ProjectCoefficient(coeff);
   }
   else if (fec_type == FECType::ND)
   {
      parent_gf.ProjectCoefficient(vcoeff);
      parent_gf_ex.ProjectCoefficient(vcoeff);
      domain1_gf_ex.ProjectCoefficient(vcoeff);
      boundary1_gf_ex.ProjectCoefficient(vcoeff);
   }
   else
   {
      parent_gf.ProjectCoefficient(vcoeff);
      parent_gf_ex.ProjectCoefficient(vcoeff);
      domain1_gf_ex.ProjectCoefficient(vcoeff);
      boundary1_gf_ex.ProjectCoefficient(nvcoeff);
   }

   Vector tmp;


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
         if (fec_type == FECType::H1 || fec_type == FECType::L2)
         {
            parent_gf.ProjectCoefficient(coeff);
            domain1_gf.ProjectCoefficient(coeff);
         }
         else
         {
            parent_gf.ProjectCoefficient(vcoeff);
            domain1_gf.ProjectCoefficient(vcoeff);
         }
         ParSubMesh::Transfer(domain1_gf, parent_gf);
         tmp = parent_gf_ex;
         tmp -= parent_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Surface to matching surface in volume")
      {
         if (fec_type == FECType::H1 || fec_type == FECType::L2)
         {
            boundary1_gf.ProjectCoefficient(coeff);
         }
         else if (fec_type == FECType::ND)
         {
            boundary1_gf.ProjectCoefficient(vcoeff);
         }
         else
         {
            boundary1_gf.ProjectCoefficient(nvcoeff);
         }
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
   constexpr int dim = 3;
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

   auto cylinder_submesh =
      ParSubMesh::CreateFromDomain(parent_mesh, cylinder_domain_attributes);

   auto outer_submesh =
      ParSubMesh::CreateFromDomain(parent_mesh, outer_domain_attributes);

   auto cylinder_surface_submesh =
      ParSubMesh::CreateFromBoundary(parent_mesh, cylinder_surface_attributes);

   Array<int> cylinder_cyl_surf_marker(cylinder_submesh.bdr_attributes.Max());
   cylinder_cyl_surf_marker = 0;
   cylinder_cyl_surf_marker[8] = 1;

   Array<int> outer_cyl_surf_marker(outer_submesh.bdr_attributes.Max());
   outer_cyl_surf_marker = 0;
   outer_cyl_surf_marker[8] = 1;

   int num_local_be = cylinder_surface_submesh.GetNBE();
   int num_global_be = 0;
   MPI_Allreduce(&num_local_be, &num_global_be, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   REQUIRE(num_global_be == 16);
   REQUIRE(cylinder_surface_submesh.bdr_attributes[0] ==
           parent_mesh.bdr_attributes.Max() + 1);

   FiniteElementCollection *fec = create_fec(fec_type, p,
                                             parent_mesh.Dimension());

   ParFiniteElementSpace parent_fes(&parent_mesh, fec);
   ParGridFunction parent_gf(&parent_fes);
   ParGridFunction parent_gf_ex(&parent_fes);

   ParFiniteElementSpace cylinder_fes(&cylinder_submesh, fec);
   ParGridFunction cylinder_gf(&cylinder_fes);
   ParGridFunction cylinder_gf_ex(&cylinder_fes);

   ParFiniteElementSpace outer_fes(&outer_submesh, fec);
   ParGridFunction outer_gf(&outer_fes);
   ParGridFunction outer_gf_ex(&outer_fes);

   FiniteElementCollection *surface_fec =
      create_surf_fec(fec_type, p, cylinder_surface_submesh.Dimension());
   ParFiniteElementSpace cylinder_surface_fes(&cylinder_surface_submesh,
                                              surface_fec);
   ParGridFunction cylinder_surface_gf(&cylinder_surface_fes);
   ParGridFunction cylinder_surface_gf_ex(&cylinder_surface_fes);

   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      real_t x = coords(0);
      real_t y = coords(1);
      real_t z = coords(2);
      return y + 0.05 * sin(x * 2.0 * M_PI) + z;
   });

   auto vcoeff = VectorFunctionCoefficient(dim, [](const Vector &coords,
                                                   Vector &V)
   {
      V.SetSize(3);
      real_t x = coords(0);
      real_t y = coords(1);
      real_t z = coords(2);

      V(0) = y + 0.05 * sin(x * 2.0 * M_PI) + z;
      V(1) = z + 0.05 * sin(y * 2.0 * M_PI) + x;
      V(2) = x + 0.05 * sin(z * 2.0 * M_PI) + y;
   });

   auto vzerocoeff = VectorFunctionCoefficient(dim, [](const Vector &,
                                                       Vector &V)
   {
      V.SetSize(3);
      V = 0.0;
   });

   SurfaceNormalCoef normalcoeff(dim);
   InnerProductCoefficient nvcoeff(normalcoeff, vcoeff);

   if (fec_type == FECType::H1 || fec_type == FECType::L2)
   {
      parent_gf.ProjectCoefficient(coeff);
      parent_gf_ex.ProjectCoefficient(coeff);
      cylinder_gf_ex.ProjectCoefficient(coeff);
      cylinder_surface_gf_ex.ProjectCoefficient(coeff);
      outer_gf_ex.ProjectCoefficient(coeff);
   }
   else if (fec_type == FECType::ND)
   {
      parent_gf.ProjectCoefficient(vcoeff);
      parent_gf_ex.ProjectCoefficient(vcoeff);
      cylinder_gf_ex.ProjectCoefficient(vcoeff);
      cylinder_surface_gf_ex.ProjectCoefficient(vcoeff);
      outer_gf_ex.ProjectCoefficient(vcoeff);
   }
   else
   {
      parent_gf.ProjectCoefficient(vcoeff);
      parent_gf_ex.ProjectCoefficient(vcoeff);
      cylinder_gf_ex.ProjectCoefficient(vcoeff);
      cylinder_surface_gf_ex.ProjectCoefficient(nvcoeff);
      outer_gf_ex.ProjectCoefficient(vcoeff);
   }

   Vector tmp;

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
         if (fec_type == FECType::H1 || fec_type == FECType::L2)
         {
            parent_gf.ProjectCoefficient(coeff);
            cylinder_gf.ProjectCoefficient(coeff);
         }
         else
         {
            parent_gf.ProjectCoefficient(vcoeff);
            cylinder_gf.ProjectCoefficient(vcoeff);
         }
         ParSubMesh::Transfer(cylinder_gf, parent_gf);
         tmp = parent_gf_ex;
         tmp -= parent_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Volume to matching volume")
      {
         if (fec_type == FECType::H1 || fec_type == FECType::L2)
         {
            outer_gf.ProjectCoefficient(coeff);
         }
         else
         {
            outer_gf.ProjectCoefficient(vcoeff);
         }
         ParSubMesh::Transfer(outer_gf, parent_gf);
         tmp = parent_gf_ex;
         tmp -= parent_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Surface to matching surface in volume")
      {
         if (fec_type == FECType::H1 || fec_type == FECType::L2)
         {
            cylinder_surface_gf.ProjectCoefficient(coeff);
         }
         else if (fec_type == FECType::ND)
         {
            cylinder_surface_gf.ProjectCoefficient(vcoeff);
         }
         else
         {
            cylinder_surface_gf.ProjectCoefficient(nvcoeff);
         }
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
         if (fec_type == FECType::H1 || fec_type == FECType::L2)
         {
            cylinder_gf.ProjectCoefficient(coeff);
            outer_gf.ProjectCoefficient(coeff);
            outer_gf_ex.ProjectCoefficient(coeff);
         }
         else
         {
            cylinder_gf.ProjectCoefficient(vcoeff);
            outer_gf.ProjectCoefficient(vcoeff);
            outer_gf.ProjectBdrCoefficient(vzerocoeff,
                                           outer_cyl_surf_marker);
            outer_gf_ex.ProjectCoefficient(vcoeff);
         }
         ParSubMesh::Transfer(cylinder_gf, outer_gf);
         tmp = outer_gf_ex;
         tmp -= outer_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Volume to matching volume (reversed)")
      {
         if (fec_type == FECType::H1 || fec_type == FECType::L2)
         {
            cylinder_gf.ProjectCoefficient(coeff);
            outer_gf.ProjectCoefficient(coeff);
            outer_gf_ex.ProjectCoefficient(coeff);
         }
         else
         {
            outer_gf.ProjectCoefficient(vcoeff);
            cylinder_gf.ProjectCoefficient(vcoeff);
            cylinder_gf.ProjectBdrCoefficient(vzerocoeff,
                                              cylinder_cyl_surf_marker);
            cylinder_gf_ex.ProjectCoefficient(vcoeff);
         }
         ParSubMesh::Transfer(outer_gf, cylinder_gf);
         tmp = cylinder_gf_ex;
         tmp -= cylinder_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
      SECTION("Volume to matching surface on volume")
      {
         if (fec_type == FECType::H1 || fec_type == FECType::L2)
         {
            cylinder_gf.ProjectCoefficient(coeff);
            outer_gf.ProjectCoefficient(coeff);
            cylinder_gf_ex.ProjectCoefficient(coeff);
         }
         else
         {
            cylinder_gf.ProjectCoefficient(vcoeff);
            outer_gf.ProjectCoefficient(vcoeff);
            cylinder_gf_ex.ProjectCoefficient(vcoeff);
         }
         ParSubMesh::Transfer(outer_gf, cylinder_gf);
         tmp = cylinder_gf_ex;
         tmp -= cylinder_gf;
         CHECK_GLOBAL_NORM(tmp);
      }

      SECTION("Volume to matching surface")
      {
         if (fec_type == FECType::H1 || fec_type == FECType::L2)
         {
            cylinder_gf.ProjectCoefficient(coeff);
            cylinder_surface_gf_ex.ProjectCoefficient(coeff);
         }
         else if (fec_type == FECType::ND)
         {
            cylinder_gf.ProjectCoefficient(vcoeff);
            cylinder_surface_gf_ex.ProjectCoefficient(vcoeff);
         }
         else
         {
            cylinder_gf.ProjectCoefficient(vcoeff);
            cylinder_surface_gf_ex.ProjectCoefficient(nvcoeff);
         }
         ParSubMesh::Transfer(cylinder_gf, cylinder_surface_gf);
         tmp = cylinder_surface_gf_ex;
         tmp -= cylinder_surface_gf;
         CHECK_GLOBAL_NORM(tmp);
      }
   }
   delete surface_fec;
   delete fec;
}

TEST_CASE("ParSubMesh", "[Parallel],[SubMesh]")
{
   auto fec_type = GENERATE(FECType::H1, FECType::ND, FECType::RT, FECType::L2);
   multidomain_test_2d(fec_type);
   multidomain_test_3d(fec_type);
}

Array<int> count_be(ParMesh &mesh)
{
   const int bdr_max = mesh.bdr_attributes.Size() > 0 ?
                       mesh.bdr_attributes.Max() : 6;

   Array<int> counts(bdr_max + 1);
   counts = 0;

   for (int i=0; i<mesh.GetNBE(); i++)
   {
      counts[mesh.GetBdrAttribute(i)]++;
   }

   Array<int> glb_counts(bdr_max + 1);
   glb_counts = 0;
   MPI_Reduce(counts, glb_counts, bdr_max + 1,
              MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

   return glb_counts;
}

TEST_CASE("ParSubMesh Interior Boundaries", "[Parallel],[SubMesh]")
{
   // whether to NC refine the attribute 1 elements
   auto make_nc = GENERATE(false, true);
   int num_procs = Mpi::WorldSize();
   Mesh serial_mesh = Mesh::MakeCartesian3D(num_procs, num_procs, 1,
                                            Element::HEXAHEDRON,
                                            1.0, 1.0, 0.1, false);

   // Assign alternating element attributes to each element to create a
   // checkerboard pattern
   for (int i=0; i < serial_mesh.GetNE(); i++)
   {
      int attr = (i + (1 + num_procs % 2) * (i / num_procs)) % 2 + 1;
      serial_mesh.SetAttribute(i, attr);
   }
   int bdr_max = serial_mesh.bdr_attributes.Max();

   // Label all interior faces as boundary elements
   Array<int> v(4);
   for (int i=0; i < serial_mesh.GetNumFaces(); i++)
   {
      if (serial_mesh.FaceIsInterior(i))
      {
         serial_mesh.GetFaceVertices(i, v);
         serial_mesh.AddBdrQuad(v, bdr_max + i + 1);
      }
   }
   serial_mesh.FinalizeMesh();
   serial_mesh.SetAttributes();

   // Create an intentionally bad partitioning
   Array<int> partitioning(num_procs * num_procs);
   for (int i = 0; i < num_procs * num_procs; i++)
   {
      // The following creates a shifting pattern where neighboring elements are
      // never owned by the same processor
      partitioning[i] = (2 * num_procs - 1 - (i % num_procs) -
                         i / num_procs) % num_procs;
   }
   if (make_nc)
   {
      serial_mesh.EnsureNCMesh(true);
   }
   ParMesh parent_mesh(MPI_COMM_WORLD, serial_mesh, partitioning);

   if (make_nc)
   {
      // Refine after partitioning so that the checkerboard pattern persists.
      Array<int> el_to_refine;
      for (int i = 0; i < parent_mesh.GetNE(); i++)
      {
         if (parent_mesh.GetAttribute(i) == 1)
         {
            el_to_refine.Append(i);
         }
      }
      parent_mesh.GeneralRefinement(el_to_refine);
   }

   // Create a pair of domain-based sub meshes
   Array<int> domain1(1);
   domain1[0] = 1;
   Array<int> domain2(1);
   domain2[0] = 2;

   auto domain1_submesh = ParSubMesh::CreateFromDomain(parent_mesh,
                                                       domain1);
   auto domain2_submesh = ParSubMesh::CreateFromDomain(parent_mesh,
                                                       domain2);

   // Create histograms of boundary attributes in each sub-domain
   auto be1 = count_be(domain1_submesh);
   auto be2 = count_be(domain2_submesh);
   REQUIRE(((be1.Size() >= 7) && (be2.Size() >= 7)));

   // Only the root process has valid histograms
   if (Mpi::Root())
   {
      // Verify that all exterior boundary elements were accounted for. If an NC
      // refine has occurred, there will be extra faces on half the checkerboard
      const int num_top_refined = make_nc ? (num_procs/2)*(num_procs/2)
                                  + ((num_procs+1)/2)*((num_procs+1)/2) : 0;
      const int num_side_refined = make_nc ? (num_procs+1)/2 : 0;
      CHECK(be1[1] + be2[1] == num_procs * num_procs + 3 * num_top_refined);
      CHECK(be1[2] + be2[2] == num_procs + 3 * num_side_refined);
      CHECK(be1[3] + be2[3] == num_procs + 3 * num_side_refined);
      CHECK(be1[4] + be2[4] == num_procs + 3 * num_side_refined);
      CHECK(be1[5] + be2[5] == num_procs + 3 * num_side_refined);
      CHECK(be1[6] + be2[6] == num_procs * num_procs + 3 * num_top_refined);

      // Verify that all interior boundary elements of serial mesh appear
      // correct number of times in each submesh
      for (int i=0; i < serial_mesh.GetNumFaces(); i++)
      {
         if (serial_mesh.FaceIsInterior(i))
         {
            const int attr = bdr_max + i + 1;
            REQUIRE(attr < be1.Size());
            REQUIRE(attr < be2.Size());
            CAPTURE(make_nc, i, attr, bdr_max, be1[attr], be2[attr]);
            CHECK(be1[attr] == (make_nc ? 4 : 1));
            CHECK(be2[attr] == 1);
         }
      }
   }
}
/**
 * @brief Helper class for testing a ParNCMesh
 *
 */
struct ParNCMeshExposed : public ParNCMesh
{
   ParNCMeshExposed(const ParNCMesh &ncmesh) : ParNCMesh(ncmesh) {}
   using ParNCMesh::elements;
   using ParNCMesh::leaf_elements;
   int CountUniqueLeafElements() const
   {
      int local = 0;
      for (auto i : leaf_elements)
      {
         if (elements[i].rank == MyRank)
         {
            local++;
         }
      }
      int global = 0;
      MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, GetGlobalMPI_Comm());
      return global;
   }
};

void CheckProjectMatch(ParMesh &mesh, ParSubMesh &submesh, FECType fec_type,
                       bool check_pr = true)
{
   int p = 3;
   CAPTURE(fec_type);
   auto fec = std::unique_ptr<FiniteElementCollection>(create_fec(fec_type, p,
                                                                  mesh.Dimension()));
   auto sub_fec = std::unique_ptr<FiniteElementCollection>(create_fec(fec_type, p,
                                                                      submesh.Dimension()));

   ParFiniteElementSpace fes(&mesh, fec.get());
   ParFiniteElementSpace sub_fes(&submesh, sub_fec.get());
   ParGridFunction gf(&fes), gf_ext(&fes);
   ParGridFunction sub_gf(&sub_fes), sub_gf_ext(&sub_fes);
   auto coeff = FunctionCoefficient([](const Vector &coords)
   {
      real_t x = coords(0);
      real_t y = coords(1);
      real_t z = coords(2);
      return 0.02 * sin(y * 5.0 * M_PI)
             + 0.03 * sin(x * 5.0 * M_PI)
             + 0.05 * sin(z * 5.0 * M_PI);
   });

   auto vcoeff = VectorFunctionCoefficient(mesh.SpaceDimension(),
                                           [](const Vector &coords, Vector &V)
   {
      V.SetSize(3);
      real_t x = coords(0);
      real_t y = coords(1);
      real_t z = coords(2);

      V(0) = 0.02 * sin(y * 3.0 * M_PI)
             + 0.03 * sin(x * 2.0 * M_PI)
             + 0.05 * sin(z * 4.0 * M_PI);
      V(1) = 0.02 * sin(z * 3.0 * M_PI)
             + 0.03 * sin(y * 2.0 * M_PI)
             + 0.05 * sin(x * 4.0 * M_PI);
      V(2) = 0.02 * sin(x * 3.0 * M_PI)
             + 0.03 * sin(y * 2.0 * M_PI)
             + 0.05 * sin(z * 4.0 * M_PI);
   });

   if (fec_type == FECType::H1 || fec_type == FECType::L2)
   {
      gf.ProjectCoefficient(coeff);
      sub_gf.ProjectCoefficient(coeff);
   }
   else
   {
      gf.ProjectCoefficient(vcoeff);
      sub_gf.ProjectCoefficient(vcoeff);
   }
   gf_ext = gf;
   sub_gf_ext = sub_gf;

   SECTION("ParentToSubMesh")
   {
      // Direct transfer should be identical
      ParSubMesh::Transfer(gf, sub_gf);
      auto tmp = sub_gf_ext;
      tmp -= sub_gf;
      CHECK_GLOBAL_NORM(tmp);
   }
   SECTION("PRConstraint")
   {
      // Application of PR should be identical in mesh and submesh for an
      // external boundary.
      if (mesh.Nonconforming())
      {
         Vector tmp;
         if (const auto *P = fes.GetProlongationMatrix())
         {
            const auto *R = fes.GetRestrictionMatrix();
            tmp.SetSize(R->Height());
            R->Mult(gf, tmp);
            P->Mult(tmp, gf);
         }
         if (const auto *P = sub_fes.GetProlongationMatrix())
         {
            const auto *R = sub_fes.GetRestrictionMatrix();
            tmp.SetSize(R->Height());
            R->Mult(sub_gf_ext, tmp);
            P->Mult(tmp, sub_gf_ext);
         }
         ParSubMesh::Transfer(gf, sub_gf);
         tmp = sub_gf_ext;
         tmp -= sub_gf;
         CHECK_GLOBAL_NORM(tmp, check_pr);
      }
   }
}

TEST_CASE("VolumeParNCSubMesh", "[Parallel],[SubMesh]")
{
   bool use_tet = GENERATE(false,true);

   auto mesh = use_tet ? OrientedTriFaceMesh(1, true) : DividingPlaneMesh(false,
                                                                          true);
   mesh.EnsureNCMesh(true);
   SECTION("UniformRefinement2")
   {
      mesh.UniformRefinement();
      mesh.UniformRefinement();
      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      SECTION("SingleAttribute")
      {
         Array<int> subdomain_attributes(1);
         subdomain_attributes[0] = GENERATE(range(1,2));
         auto submesh = ParSubMesh::CreateFromDomain(pmesh, subdomain_attributes);

         // Cast to an exposed variant to explore the internals.
         auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
         CHECK(pncmesh_exposed.GetNumRootElements() == 1);
         CHECK(pncmesh_exposed.CountUniqueLeafElements() == 8*8);
         for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
         {
            CheckProjectMatch(pmesh, submesh, fec_type);
         }
      }

      SECTION("UniformRefineTwoAttribute")
      {
         Array<int> subdomain_attributes(2);
         subdomain_attributes[0] = 1;
         subdomain_attributes[1] = 2;
         auto submesh = ParSubMesh::CreateFromDomain(pmesh, subdomain_attributes);

         // Cast to an exposed variant to explore the internals.
         auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
         CHECK(pncmesh_exposed.GetNumRootElements() ==
               pmesh.ncmesh->GetNumRootElements());
         CHECK(pncmesh_exposed.CountUniqueLeafElements() == 2*8*8);
         for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
         {
            CheckProjectMatch(pmesh, submesh, fec_type);
         }
      }
   }

   SECTION("Nonconformal")
   {
      mesh.UniformRefinement();
      Array<int> subdomain_attributes{GENERATE(1,2)};
      auto backwards = GENERATE(false, true);
      SECTION("ConsistentWithParent")
      {
         RefineSingleUnattachedElement(mesh, subdomain_attributes[0],
                                       mesh.bdr_attributes.Max(), backwards);
         {
            ParMesh pmesh(MPI_COMM_WORLD, mesh);
            auto submesh = ParSubMesh::CreateFromDomain(pmesh, subdomain_attributes);

            // Cast to an exposed variant to explore the internals.
            auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
            CHECK(pncmesh_exposed.GetNumRootElements() == 1);
            CHECK(pncmesh_exposed.CountUniqueLeafElements() == 8 - 1 + 8);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(pmesh, submesh, fec_type, true);
            }
         }
         RefineSingleUnattachedElement(mesh, subdomain_attributes[0],
                                       mesh.bdr_attributes.Max(), backwards);
         {
            ParMesh pmesh(MPI_COMM_WORLD, mesh);
            auto submesh = ParSubMesh::CreateFromDomain(pmesh, subdomain_attributes);

            // Cast to an exposed variant to explore the internals.
            auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
            CHECK(pncmesh_exposed.GetNumRootElements() == 1);
            CHECK(pncmesh_exposed.CountUniqueLeafElements() == 8 - 1 + 8 - 1 + 8);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(pmesh, submesh, fec_type, true);
            }
         }
      }

      SECTION("InconsistentWithParent")
      {
         RefineSingleAttachedElement(mesh, subdomain_attributes[0],
                                     mesh.bdr_attributes.Max(), backwards);
         {
            ParMesh pmesh(MPI_COMM_WORLD, mesh);
            auto submesh = ParSubMesh::CreateFromDomain(pmesh, subdomain_attributes);

            // Cast to an exposed variant to explore the internals.
            auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
            CHECK(pncmesh_exposed.GetNumRootElements() == 1);
            CHECK(pncmesh_exposed.CountUniqueLeafElements() == 8 - 1 + 8);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(pmesh, submesh, fec_type, false);
            }
         }
         RefineSingleAttachedElement(mesh, subdomain_attributes[0],
                                     mesh.bdr_attributes.Max(), backwards);
         {
            ParMesh pmesh(MPI_COMM_WORLD, mesh);
            auto submesh = ParSubMesh::CreateFromDomain(pmesh, subdomain_attributes);

            // Cast to an exposed variant to explore the internals.
            auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
            CHECK(pncmesh_exposed.GetNumRootElements() == 1);
            CHECK(pncmesh_exposed.CountUniqueLeafElements() == 8 - 1 + 8 - 1 + 8);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(pmesh, submesh, fec_type, false);
            }
         }
      }
   }
}

TEST_CASE("ExteriorSurfaceParNCSubMesh", "[Parallel],[SubMesh]")
{
   SECTION("Hex")
   {
      auto mesh = Mesh("../../data/ref-cube.mesh", 1, 1);
      mesh.EnsureNCMesh(true);
      SECTION("UniformRefinement2")
      {
         mesh.UniformRefinement();
         mesh.UniformRefinement();
         ParMesh pmesh(MPI_COMM_WORLD, mesh);
         SECTION("SingleAttribute")
         {
            Array<int> subdomain_attributes(1);
            subdomain_attributes[0] = GENERATE(range(1,6));
            auto submesh = ParSubMesh::CreateFromBoundary(pmesh, subdomain_attributes);

            // Cast to an exposed variant to explore the internals.
            auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
            CHECK(pncmesh_exposed.GetNumRootElements() == 1);
            CHECK(pncmesh_exposed.CountUniqueLeafElements() == 4*4);
            CHECK(submesh.bdr_attributes.Size() == 1);
            CHECK(submesh.bdr_attributes[0] == mesh.bdr_attributes.Max() + 1);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(pmesh, submesh, fec_type);
            }
         }

         SECTION("UniformRefineTwoAttribute")
         {
            Array<int> subdomain_attributes(2);
            subdomain_attributes[0] = GENERATE(range(1,6));
            subdomain_attributes[1] = 1 + (subdomain_attributes[0] % 6);
            auto submesh = ParSubMesh::CreateFromBoundary(pmesh, subdomain_attributes);

            // Cast to an exposed variant to explore the internals.
            auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
            CHECK(pncmesh_exposed.GetNumRootElements() == 2);
            CHECK(pncmesh_exposed.CountUniqueLeafElements() == 2*4*4);
            CHECK(submesh.bdr_attributes.Size() == 1);
            CHECK(submesh.bdr_attributes[0] == mesh.bdr_attributes.Max() + 1);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(pmesh, submesh, fec_type);
            }
         }
      }

      SECTION("NonconformalRefine")
      {
         Array<int> subdomain_attributes(1);
         subdomain_attributes[0] = GENERATE(range(1,6));
         mesh.UniformRefinement();
         RefineSingleAttachedElement(mesh, 1, subdomain_attributes[0], true);
         SECTION("Single")
         {
            ParMesh pmesh(MPI_COMM_WORLD, mesh);
            auto submesh = ParSubMesh::CreateFromBoundary(pmesh, subdomain_attributes);

            // Cast to an exposed variant to explore the internals.
            auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
            CHECK(pncmesh_exposed.GetNumRootElements() == 1);
            CHECK(pncmesh_exposed.CountUniqueLeafElements() == 4 - 1 + 4);
            CHECK(submesh.bdr_attributes.Size() == 1);
            CHECK(submesh.bdr_attributes[0] == mesh.bdr_attributes.Max() + 1);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(pmesh, submesh, fec_type);
            }

         }
         SECTION("Double")
         {
            RefineSingleAttachedElement(mesh, 1, subdomain_attributes[0], false);
            ParMesh pmesh(MPI_COMM_WORLD, mesh);
            auto submesh = ParSubMesh::CreateFromBoundary(pmesh, subdomain_attributes);

            // Cast to an exposed variant to explore the internals.
            auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
            CHECK(pncmesh_exposed.GetNumRootElements() == 1);
            CHECK(pncmesh_exposed.CountUniqueLeafElements() == 4 - 1 + 4 - 1 + 4);
            CHECK(submesh.bdr_attributes.Size() == 1);
            CHECK(submesh.bdr_attributes[0] == mesh.bdr_attributes.Max() + 1);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(pmesh, submesh, fec_type);
            }
         }
      }
   }

   SECTION("Tet")
   {
      auto mesh = Mesh("../../data/ref-tetrahedron.mesh");
      mesh.EnsureNCMesh(true);
      SECTION("UniformRefinement2")
      {
         mesh.UniformRefinement();
         mesh.UniformRefinement();
         ParMesh pmesh(MPI_COMM_WORLD, mesh);
         SECTION("SingleAttribute")
         {
            Array<int> subdomain_attributes(1);
            subdomain_attributes[0] = GENERATE(range(1,4));
            auto submesh = ParSubMesh::CreateFromBoundary(pmesh, subdomain_attributes);

            // Cast to an exposed variant to explore the internals.
            auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
            CHECK(pncmesh_exposed.GetNumRootElements() == 1);
            CHECK(pncmesh_exposed.CountUniqueLeafElements() == 4*4);
            CHECK(submesh.bdr_attributes.Size() == 1);
            CHECK(submesh.bdr_attributes[0] == mesh.bdr_attributes.Max() + 1);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(pmesh, submesh, fec_type);
            }
         }

         SECTION("UniformRefineTwoAttribute")
         {
            Array<int> subdomain_attributes(2);
            subdomain_attributes[0] = GENERATE(range(1,4));
            subdomain_attributes[1] = 1 + (subdomain_attributes[0] % 4);
            auto submesh = ParSubMesh::CreateFromBoundary(pmesh, subdomain_attributes);

            // Cast to an exposed variant to explore the internals.
            auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
            CHECK(pncmesh_exposed.GetNumRootElements() == 2);
            CHECK(pncmesh_exposed.CountUniqueLeafElements() == 2*4*4);
            CHECK(submesh.bdr_attributes.Size() == 1);
            CHECK(submesh.bdr_attributes[0] == mesh.bdr_attributes.Max() + 1);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(pmesh, submesh, fec_type);
            }
         }
      }

      SECTION("NonconformalRefine")
      {
         Array<int> subdomain_attributes(1);
         subdomain_attributes[0] = GENERATE(range(1,4));
         mesh.UniformRefinement();
         RefineSingleAttachedElement(mesh, 1, subdomain_attributes[0], true);
         SECTION("Single")
         {
            ParMesh pmesh(MPI_COMM_WORLD, mesh);
            auto submesh = ParSubMesh::CreateFromBoundary(pmesh, subdomain_attributes);

            // Cast to an exposed variant to explore the internals.
            auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
            CHECK(pncmesh_exposed.GetNumRootElements() == 1);
            CHECK(pncmesh_exposed.CountUniqueLeafElements() == 4 - 1 + 4);
            CHECK(submesh.bdr_attributes.Size() == 1);
            CHECK(submesh.bdr_attributes[0] == mesh.bdr_attributes.Max() + 1);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(pmesh, submesh, fec_type);
            }
         }
         SECTION("Double")
         {
            RefineSingleAttachedElement(mesh, 1, subdomain_attributes[0], false);
            ParMesh pmesh(MPI_COMM_WORLD, mesh);
            auto submesh = ParSubMesh::CreateFromBoundary(pmesh, subdomain_attributes);

            // Cast to an exposed variant to explore the internals.
            auto pncmesh_exposed = ParNCMeshExposed(*submesh.pncmesh);
            CHECK(pncmesh_exposed.GetNumRootElements() == 1);
            CHECK(pncmesh_exposed.CountUniqueLeafElements() == 4 - 1 + 4 - 1 + 4);
            CHECK(submesh.bdr_attributes.Size() == 1);
            CHECK(submesh.bdr_attributes[0] == mesh.bdr_attributes.Max() + 1);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(pmesh, submesh, fec_type);
            }
         }
      }
   }


}


} // namespace ParSubMeshTests

#endif // MFEM_USE_MPI
