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
#include <iomanip>
#include <memory>
#include "unit_tests.hpp"
#include "mesh_test_utils.hpp"

using namespace mfem;

enum class FieldType
{
   SCALAR,
   VECTOR
};
enum class TransferType
{
   ParentToSub,
   SubToParent
};

void test_2d(Element::Type element_type,
             FECType fec_type,
             FieldType field_type,
             int polynomial_order,
             int mesh_polynomial_order,
             TransferType transfer_type,
             SubMesh::From from)
{
   constexpr int dim = 2;
   const int vdim = (field_type == FieldType::SCALAR ||
                     fec_type == FECType::ND) ? 1 : dim;
   real_t Hy = 1.0;
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
            real_t *coords = mesh.GetVertex(vertices[j]);

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
      real_t x = coords(0);
      real_t y = coords(1);

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

   if (transfer_type == TransferType::ParentToSub)
   {
      GridFunction sub_ex_gf(&sub_fes);

      if (vdim == 1 && (fec_type == FECType::H1 || fec_type == FECType::L2))
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
   else if (transfer_type == TransferType::SubToParent)
   {
      GridFunction parent_ex_gf(&parent_fes);

      if (vdim == 1 && (fec_type == FECType::H1 || fec_type == FECType::L2))
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

void test_3d(Element::Type element_type,
             FECType fec_type,
             FieldType field_type,
             int polynomial_order,
             int mesh_polynomial_order,
             TransferType transfer_type,
             SubMesh::From from)
{
   constexpr int dim = 3;
   const int vdim = (field_type == FieldType::SCALAR ||
                     fec_type == FECType::ND) ? 1 : dim;
   real_t Hy = 1.0;
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
      real_t x = coords(0);
      real_t y = coords(1);
      real_t z = coords(2);

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

   if (transfer_type == TransferType::ParentToSub)
   {
      GridFunction sub_ex_gf(&sub_fes);

      if (vdim == 1 && (fec_type == FECType::H1 || fec_type == FECType::L2))
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
   else if (transfer_type == TransferType::SubToParent)
   {
      GridFunction parent_ex_gf(&parent_fes);

      if (vdim == 1 && (fec_type == FECType::H1 || fec_type == FECType::L2))
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
   auto fec_type = GENERATE(FECType::H1, FECType::ND, FECType::L2);
   auto field_type = GENERATE(FieldType::SCALAR, FieldType::VECTOR);
   auto transfer_type = GENERATE(TransferType::ParentToSub,
                                 TransferType::SubToParent);
   auto from = GENERATE(SubMesh::From::Domain,
                        SubMesh::From::Boundary);

   if (fec_type == FECType::ND && field_type == FieldType::VECTOR)
   {
      return;
   }
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
      auto element = GENERATE(Element::HEXAHEDRON, Element::TETRAHEDRON,
                              Element::WEDGE);
      if (fec_type == FECType::L2 &&
          from == SubMesh::From::Boundary && false)
      {
         return;
      }
      test_3d(element, fec_type, field_type, polynomial_order,
              mesh_polynomial_order, transfer_type, from);
   }
}

TEST_CASE("InterfaceTransferSolve", "[SubMesh]")
{
   // Solve Poisson on a pair of cubes fully coupled, transfer to the interface
   // then solve on subdomains using the 2D solution as the boundary condition.
   int polynomial_order = 4;
   auto fec_type = FECType::H1;

   // 1. Define meshes
   auto mesh = DividingPlaneMesh(false, true);
   mesh.UniformRefinement();
   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 1;
   auto left_vol = SubMesh::CreateFromDomain(mesh, subdomain_attributes);
   subdomain_attributes[0] = 2;
   auto right_vol = SubMesh::CreateFromDomain(mesh, subdomain_attributes);

   subdomain_attributes[0] = mesh.bdr_attributes.Max();
   auto interface = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);

   // 2. Define fespaces
   auto vol_fec = std::unique_ptr<FiniteElementCollection>(create_fec(fec_type,
                                                                      polynomial_order, 3));
   auto surf_fec = std::unique_ptr<FiniteElementCollection>(create_fec(fec_type,
                                                                       polynomial_order, 2));

   auto fespace = FiniteElementSpace(&mesh, vol_fec.get());
   auto left_fespace = FiniteElementSpace(&left_vol, vol_fec.get());
   auto right_fespace = FiniteElementSpace(&right_vol, vol_fec.get());
   auto interface_fespace = FiniteElementSpace(&interface, surf_fec.get());

   // 3. Solve full problem with homogeneous boundary conditions and transfer to interface space.
   ConstantCoefficient one(1.0);

   // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
   OperatorPtr A;
   Vector B, X;
   // Manufactured solution u = sin(pi x) sin(pi y) sin(pi z).
   // f = - (u_xx + u_yy + u_zz) = d * pi^2 sin(pi x) sin(pi y) sin(pi z)
   auto f = FunctionCoefficient([](const Vector &x)
   {
      double c = M_PI * M_PI * 3;
      for (int i = 0; i < 3; ++i)
      {
         c *= std::sin(M_PI * x(i));
      }
      return c;
   });

   auto SolveHomogeneous = [&](FiniteElementSpace &fespace)
   {
      Array<int> ess_tdof_list, ess_bdr;
      ess_bdr.SetSize(fespace.GetMesh()->bdr_attributes.Max());
      ess_bdr = 1;
      if (fespace.GetMesh()->Dimension() > 2)
      {
         // The interior of the volume has an extra bc
         ess_bdr.Last() = 0;
      }

      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      LinearForm b(&fespace);
      b.AddDomainIntegrator(new DomainLFIntegrator(f));
      b.Assemble();
      GridFunction x(&fespace);
      x = 0.0;
      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 1e3, 1e-16, 0.0);
      a.RecoverFEMSolution(X, b, x);
      return x;
   };

   auto x_vol = SolveHomogeneous(fespace);
   GridFunction x_int(&interface_fespace);
   SubMesh::Transfer(x_vol, x_int);

   // 4. Transfer solution to left and right subproblems and solve
   auto SolveOnSubVolume = [&](FiniteElementSpace &fespace)
   {
      Array<int> ess_tdof_list, ess_bdr;
      ess_bdr.SetSize(fespace.GetMesh()->bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      LinearForm b(&fespace);
      b.AddDomainIntegrator(new DomainLFIntegrator(f));
      b.Assemble();
      GridFunction x(&fespace);
      x = 0.0;
      SubMesh::Transfer(x_int, x);
      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 1e3, 1e-16, 0.0);
      a.RecoverFEMSolution(X, b, x);
      return x;
   };

   auto x_right = SolveOnSubVolume(right_fespace);
   auto x_left = SolveOnSubVolume(left_fespace);

   // 5. Transfer the left and right solutions onto a duplicate of the full solve
   // and compare. Given the choice of boundary conditions, should match exactly.
   auto x_sub = x_vol;
   x_sub = 0.0;
   SubMesh::Transfer(x_left, x_sub);
   SubMesh::Transfer(x_right, x_sub);
   x_sub -= x_vol;

   CHECK((x_sub.Norml2() / x_sub.Size()) == MFEM_Approx(0.0, 1e-7, 1e-7));
}

/**
 * @brief Helper class for testing a NCMesh
 *
 */
struct NCMeshExposed : public NCMesh
{
   NCMeshExposed(const NCMesh &ncmesh) : NCMesh(ncmesh) {}
   using NCMesh::elements;
   using NCMesh::leaf_elements;
   int CountUniqueLeafElements() const
   {
      int local = 0;
      for (auto i : leaf_elements)
      {
         if (elements[i].rank == MyRank)
         {
            ++local;
         }
      }
      return local;
   }
};

void CHECK_NORM(Vector &v, bool small = true)
{
   if (small)
   {
      REQUIRE(v.Norml2() < 1e-8);
   }
   else
   {
      REQUIRE(v.Norml2() > 1e-8);
   }
};

void CheckProjectMatch(Mesh &mesh, SubMesh &submesh, FECType fec_type,
                       bool check_pr = true)
{
   int p = 3;
   auto fec = std::unique_ptr<FiniteElementCollection>(create_fec(fec_type, p,
                                                                  mesh.Dimension()));
   auto sub_fec = std::unique_ptr<FiniteElementCollection>(create_fec(fec_type, p,
                                                                      submesh.Dimension()));

   FiniteElementSpace fes(&mesh, fec.get());
   FiniteElementSpace sub_fes(&submesh, sub_fec.get());
   GridFunction gf(&fes), gf_ext(&fes);
   GridFunction sub_gf(&sub_fes), sub_gf_ext(&sub_fes);
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
      SubMesh::Transfer(gf, sub_gf);
      auto tmp = sub_gf_ext;
      tmp -= sub_gf;
      CHECK_NORM(tmp);
   }
   SECTION("PRConstraint")
   {
      // Application of PR should be identical in mesh and submesh for an external boundary.
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
         SubMesh::Transfer(gf, sub_gf);
         tmp = sub_gf_ext;
         tmp -= sub_gf;
         CHECK_NORM(tmp, check_pr);
      }
   }
}

TEST_CASE("VolumeNCSubMesh", "[SubMesh]")
{
   bool use_tet = GENERATE(false,true);

   auto mesh = use_tet ? OrientedTriFaceMesh(1, true) : DividingPlaneMesh(false,
                                                                          true);
   mesh.EnsureNCMesh(true);
   SECTION("UniformRefinement2")
   {
      mesh.UniformRefinement();
      mesh.UniformRefinement();
      SECTION("SingleAttribute")
      {
         Array<int> subdomain_attributes(1);
         subdomain_attributes[0] = GENERATE(range(1,2));
         auto submesh = SubMesh::CreateFromDomain(mesh, subdomain_attributes);

         // Cast to an exposed variant to explore the internals.
         auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
         CHECK(ncmesh_exposed.GetNumRootElements() == 1);
         CHECK(ncmesh_exposed.CountUniqueLeafElements() == 8*8);
         for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
         {
            CheckProjectMatch(mesh, submesh, fec_type);
         }
      }
      SECTION("UniformRefineTwoAttribute")
      {
         Array<int> subdomain_attributes(2);
         subdomain_attributes[0] = 1;
         subdomain_attributes[1] = 2;
         auto submesh = SubMesh::CreateFromDomain(mesh, subdomain_attributes);

         // Cast to an exposed variant to explore the internals.
         auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
         CHECK(ncmesh_exposed.GetNumRootElements() == mesh.ncmesh->GetNumRootElements());
         CHECK(ncmesh_exposed.CountUniqueLeafElements() == 2*8*8);
         for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
         {
            CheckProjectMatch(mesh, submesh, fec_type);
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
            auto submesh = SubMesh::CreateFromDomain(mesh, subdomain_attributes);
            auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
            CHECK(ncmesh_exposed.GetNumRootElements() == 1);
            CHECK(ncmesh_exposed.CountUniqueLeafElements() == 8 - 1 + 8);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(mesh, submesh, fec_type, true);
            }
         }
         RefineSingleUnattachedElement(mesh, subdomain_attributes[0],
                                       mesh.bdr_attributes.Max(), backwards);
         {
            auto submesh = SubMesh::CreateFromDomain(mesh, subdomain_attributes);
            auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
            CHECK(ncmesh_exposed.GetNumRootElements() == 1);
            CHECK(ncmesh_exposed.CountUniqueLeafElements() == 8 - 1 + 8 - 1 + 8);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(mesh, submesh, fec_type, true);
            }
         }
      }

      SECTION("InconsistentWithParent")
      {
         RefineSingleAttachedElement(mesh, subdomain_attributes[0],
                                     mesh.bdr_attributes.Max(), backwards);
         {
            auto submesh = SubMesh::CreateFromDomain(mesh, subdomain_attributes);
            auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
            CHECK(ncmesh_exposed.GetNumRootElements() == 1);
            CHECK(ncmesh_exposed.CountUniqueLeafElements() == 8 - 1 + 8);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(mesh, submesh, fec_type, false);
            }
         }
         RefineSingleAttachedElement(mesh, subdomain_attributes[0],
                                     mesh.bdr_attributes.Max(), backwards);
         {
            auto submesh = SubMesh::CreateFromDomain(mesh, subdomain_attributes);
            auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
            CHECK(ncmesh_exposed.GetNumRootElements() == 1);
            CHECK(ncmesh_exposed.CountUniqueLeafElements() == 8 - 1 + 8 - 1 + 8);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(mesh, submesh, fec_type, false);
            }
         }
      }
   }
}

TEST_CASE("ExteriorSurfaceNCSubMesh", "[SubMesh]")
{
   SECTION("Hex")
   {
      auto mesh = Mesh("../../data/ref-cube.mesh", 1, 1);
      mesh.EnsureNCMesh(true);
      SECTION("UniformRefinement2")
      {
         mesh.UniformRefinement();
         mesh.UniformRefinement();
         SECTION("SingleAttribute")
         {
            Array<int> subdomain_attributes{GENERATE(range(1,6))};
            auto submesh = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);
            auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
            CHECK(ncmesh_exposed.GetNumRootElements() == 1);
            CHECK(ncmesh_exposed.CountUniqueLeafElements() == 4*4);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(mesh, submesh, fec_type);
            }
         }
         SECTION("UniformRefineTwoAttribute")
         {
            Array<int> subdomain_attributes(2);
            subdomain_attributes[0] = GENERATE(range(1,6));
            subdomain_attributes[1] = 1 + (subdomain_attributes[0] % 6);
            auto submesh = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);
            auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
            CHECK(ncmesh_exposed.GetNumRootElements() == 2);
            CHECK(ncmesh_exposed.CountUniqueLeafElements() == 2*4*4);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(mesh, submesh, fec_type);
            }
         }
      }

      SECTION("NonconformalRefine")
      {
         Array<int> subdomain_attributes{GENERATE(range(1,6))};
         mesh.UniformRefinement();
         RefineSingleAttachedElement(mesh, 1, subdomain_attributes[0], true);
         SECTION("Single")
         {
            auto submesh = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);
            auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
            CHECK(ncmesh_exposed.GetNumRootElements() == 1);
            CHECK(ncmesh_exposed.CountUniqueLeafElements() == 4 - 1 + 4);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(mesh, submesh, fec_type);
            }
         }
         SECTION("Double")
         {
            RefineSingleAttachedElement(mesh, 1, subdomain_attributes[0], false);
            auto submesh = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);
            auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
            CHECK(ncmesh_exposed.GetNumRootElements() == 1);
            CHECK(ncmesh_exposed.CountUniqueLeafElements() == 4 - 1 + 4 - 1 + 4);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(mesh, submesh, fec_type);
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
         SECTION("SingleAttribute")
         {
            Array<int> subdomain_attributes{GENERATE(range(1,4))};
            auto submesh = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);
            auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
            CHECK(ncmesh_exposed.GetNumRootElements() == 1);
            CHECK(ncmesh_exposed.CountUniqueLeafElements() == 4*4);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(mesh, submesh, fec_type);
            }
         }

         SECTION("UniformRefineTwoAttribute")
         {
            Array<int> subdomain_attributes(2);
            subdomain_attributes[0] = GENERATE(range(1,4));
            subdomain_attributes[1] = 1 + (subdomain_attributes[0] % 4);
            auto submesh = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);
            auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
            CHECK(ncmesh_exposed.GetNumRootElements() == 2);
            CHECK(ncmesh_exposed.CountUniqueLeafElements() == 2*4*4);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(mesh, submesh, fec_type);
            }
         }
      }

      SECTION("NonconformalRefine")
      {
         Array<int> subdomain_attributes{GENERATE(range(1,4))};
         mesh.UniformRefinement();
         RefineSingleAttachedElement(mesh, 1, subdomain_attributes[0], true);
         SECTION("Single")
         {
            auto submesh = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);
            auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
            CHECK(ncmesh_exposed.GetNumRootElements() == 1);
            CHECK(ncmesh_exposed.CountUniqueLeafElements() == 4 - 1 + 4);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(mesh, submesh, fec_type);
            }
         }
         SECTION("Double")
         {
            RefineSingleAttachedElement(mesh, 1, subdomain_attributes[0], false);
            auto submesh = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);
            auto ncmesh_exposed = NCMeshExposed(*submesh.ncmesh);
            CHECK(ncmesh_exposed.GetNumRootElements() == 1);
            CHECK(ncmesh_exposed.CountUniqueLeafElements() == 4 - 1 + 4 - 1 + 4);
            for (auto fec_type : {FECType::H1, FECType::L2, FECType::ND, FECType::RT})
            {
               CheckProjectMatch(mesh, submesh, fec_type);
            }
         }
      }
   }
}

/**
 * @brief Collect element ids from a mesh whose attribute equals @a attr.
 */
static Array<int> ElementsWithAttribute(const Mesh &mesh, int attr)
{
   Array<int> ids;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      if (mesh.GetAttribute(i) == attr) { ids.Append(i); }
   }
   return ids;
}

/**
 * @brief Collect boundary element ids from a mesh whose attribute equals @a attr.
 */
static Array<int> BdrElementsWithAttribute(const Mesh &mesh, int attr)
{
   Array<int> ids;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      if (mesh.GetBdrAttribute(i) == attr) { ids.Append(i); }
   }
   return ids;
}

/// Return true for FEC types whose GridFunction requires a VectorCoefficient.
static bool IsVectorFEC(FECType fec_type)
{
   return fec_type == FECType::ND || fec_type == FECType::RT;
}

/// Project the appropriate scalar or vector coefficient onto @a gf.
static void ProjectSmoothCoeff(GridFunction &gf, int sdim)
{
   if (gf.VectorDim() == 1)
   {
      auto coeff = FunctionCoefficient([](const Vector &c)
      {
         return 1.0 + 0.3*sin(c(0)*M_PI) + 0.5*cos(c(1)*M_PI);
      });
      gf.ProjectCoefficient(coeff);
   }
   else
   {
      auto vcoeff = VectorFunctionCoefficient(sdim,
                                              [](const Vector &c, Vector &v)
      {
         v.SetSize(c.Size());
         v(0) = 1.0 + 0.3*sin(c(0)*M_PI) + 0.5*cos(c(1)*M_PI);
         v(1) = 0.7 + 0.4*cos(c(0)*M_PI) + 0.2*sin(c(1)*M_PI);
         if (c.Size() > 2)
         {
            v(2) = 0.5 + 0.3*sin(c(2)*M_PI);
         }
      });
      gf.ProjectCoefficient(vcoeff);
   }
}

/**
 * @brief Verify that SubMesh::CreateFromElements gives the same topology and
 * field-transfer result as SubMesh::CreateFromDomain for the same set of
 * elements (selected by attribute).
 */
static void CheckFromElementsMatchesDomain(Mesh &mesh, int attr,
                                           FECType fec_type)
{
   Array<int> attrs{attr};
   auto sm_attr = SubMesh::CreateFromDomain(mesh, attrs);
   auto ids = ElementsWithAttribute(mesh, attr);
   auto sm_ids  = SubMesh::CreateFromElements(mesh, ids);

   // Same element count
   REQUIRE(sm_ids.GetNE() == sm_attr.GetNE());
   // Same vertex count
   REQUIRE(sm_ids.GetNV() == sm_attr.GetNV());

   int p = 2;
   auto fec = std::unique_ptr<FiniteElementCollection>(
                 create_fec(fec_type, p, mesh.Dimension()));
   auto sub_fec = std::unique_ptr<FiniteElementCollection>(
                     create_fec(fec_type, p, sm_attr.Dimension()));

   FiniteElementSpace fes_parent(&mesh, fec.get());
   GridFunction gf_parent(&fes_parent);
   ProjectSmoothCoeff(gf_parent, mesh.SpaceDimension());

   FiniteElementSpace sub_fes_attr(&sm_attr, sub_fec.get());
   GridFunction gf_attr(&sub_fes_attr);
   gf_attr = 0.0;
   SubMesh::Transfer(gf_parent, gf_attr);

   FiniteElementSpace sub_fes_ids(&sm_ids, sub_fec.get());
   GridFunction gf_ids(&sub_fes_ids);
   gf_ids = 0.0;
   SubMesh::Transfer(gf_parent, gf_ids);

   // Both submeshes should have the same L2 norm (up to round-off)
   REQUIRE(gf_attr.Norml2() == Approx(gf_ids.Norml2()).epsilon(1e-10));
}

/**
 * @brief Verify that SubMesh::CreateFromBdrElements gives the same topology
 * and field-transfer result as SubMesh::CreateFromBoundary for elements
 * selected by attribute.
 */
static void CheckFromBdrElementsMatchesBoundary(Mesh &mesh, int attr,
                                                FECType fec_type)
{
   Array<int> attrs{attr};
   auto sm_attr = SubMesh::CreateFromBoundary(mesh, attrs);
   auto ids = BdrElementsWithAttribute(mesh, attr);
   auto sm_ids  = SubMesh::CreateFromBdrElements(mesh, ids);

   REQUIRE(sm_ids.GetNE() == sm_attr.GetNE());
   REQUIRE(sm_ids.GetNV() == sm_attr.GetNV());

   int p = 2;
   auto fec = std::unique_ptr<FiniteElementCollection>(
                 create_fec(fec_type, p, mesh.Dimension()));
   auto sub_fec = std::unique_ptr<FiniteElementCollection>(
                     create_fec(fec_type, p, sm_attr.Dimension()));

   FiniteElementSpace fes_parent(&mesh, fec.get());
   GridFunction gf_parent(&fes_parent);
   ProjectSmoothCoeff(gf_parent, mesh.SpaceDimension());

   FiniteElementSpace sub_fes_attr(&sm_attr, sub_fec.get());
   GridFunction gf_attr(&sub_fes_attr);
   gf_attr = 0.0;
   SubMesh::Transfer(gf_parent, gf_attr);

   FiniteElementSpace sub_fes_ids(&sm_ids, sub_fec.get());
   GridFunction gf_ids(&sub_fes_ids);
   gf_ids = 0.0;
   SubMesh::Transfer(gf_parent, gf_ids);

   REQUIRE(gf_attr.Norml2() == Approx(gf_ids.Norml2()).epsilon(1e-10));
}

TEST_CASE("SubMesh from elements matches domain", "[SubMesh]")
{
   // Build a 2-attribute 3D mesh (same as the DividingPlaneMesh helper)
   auto mesh = DividingPlaneMesh(false, true);

   SECTION("H1") { CheckFromElementsMatchesDomain(mesh, 1, FECType::H1); }
   SECTION("L2") { CheckFromElementsMatchesDomain(mesh, 1, FECType::L2); }
   SECTION("ND") { CheckFromElementsMatchesDomain(mesh, 2, FECType::ND); }
   SECTION("RT") { CheckFromElementsMatchesDomain(mesh, 2, FECType::RT); }
}

TEST_CASE("SubMesh from bdr eleemnts matches boundary", "[SubMesh]")
{
   auto mesh = DividingPlaneMesh(false, true);
   int max_battr = mesh.bdr_attributes.Max();
   int battr = GENERATE_COPY(range(1, max_battr + 1));

   SECTION("H1") { CheckFromBdrElementsMatchesBoundary(mesh, battr, FECType::H1); }
   SECTION("L2") { CheckFromBdrElementsMatchesBoundary(mesh, battr, FECType::L2); }
}

/// Check that every vertex in @a sm sits at the same coordinates as its
/// corresponding parent vertex.  Works for both domain and boundary submeshes.
static void CheckVertexCoordinates(const SubMesh &sm, const Mesh &parent)
{
   const auto &vtx_map = sm.GetParentVertexIDMap();
   const int sdim = parent.SpaceDimension();

   bool ok = true;
   for (int sv = 0; sv < sm.GetNV(); sv++)
   {
      const real_t *sub_coords    = sm.GetVertex(sv);
      const real_t *parent_coords = parent.GetVertex(vtx_map[sv]);
      for (int d = 0; d < sdim; d++)
      {
         if (sub_coords[d] != Approx(parent_coords[d]).epsilon(1e-14))
         {
            ok = false;
            break;
         }
      }
   }
   REQUIRE(ok);
}

/**
 * @brief Build a 4x4x4 hex mesh and extract two spatially separated
 * horizontal slabs (z-centroid < 0.25 and z-centroid > 0.75) as a single
 * element list — two disconnected regions.
 */
TEST_CASE("SubMesh from elements disconnected", "[SubMesh]")
{
   Mesh mesh = Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON,
                                     1.0, 1.0, 1.0, false);

   Array<int> bottom_ids, top_ids, combined_ids;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      Array<int> verts;
      mesh.GetElementVertices(i, verts);
      real_t z_avg = 0.0;
      for (int v : verts)
      {
         z_avg += mesh.GetVertex(v)[2];
      }
      z_avg /= verts.Size();

      if (z_avg < 0.25)
      {
         bottom_ids.Append(i);
         combined_ids.Append(i);
      }
      else if (z_avg > 0.75)
      {
         top_ids.Append(i);
         combined_ids.Append(i);
      }
   }

   REQUIRE(bottom_ids.Size() > 0);
   REQUIRE(top_ids.Size() > 0);

   auto sm = SubMesh::CreateFromElements(mesh, combined_ids);

   // Element count must match the requested list exactly
   REQUIRE(sm.GetNE() == combined_ids.Size());

   // Every parent element id in the map must come from the requested set
   const auto &pids = sm.GetParentElementIDMap();
   REQUIRE(pids.Size() == combined_ids.Size());
   bool all_found = true;
   for (int i = 0; i < pids.Size(); i++)
   {
      bool found = bottom_ids.Find(pids[i]) >= 0 || top_ids.Find(pids[i]) >= 0;
      if (!found)
      {
         all_found = false;
         break;
      }
   }
   REQUIRE(all_found);

   // Every submesh vertex must be geometrically coincident with its parent vertex
   CheckVertexCoordinates(sm, mesh);

   auto coeff = FunctionCoefficient([](const Vector &c)
   {
      return c(2) + 0.1*sin(c(0)*M_PI);
   });

   SECTION("H1 parent-to-sub transfer")
   {
      H1_FECollection fec(2, mesh.Dimension());
      H1_FECollection sub_fec(2, sm.Dimension());
      FiniteElementSpace fes(&mesh, &fec);
      FiniteElementSpace sub_fes(&sm, &sub_fec);

      GridFunction gf(&fes), sub_gf(&sub_fes), sub_ex(&sub_fes);
      gf.ProjectCoefficient(coeff);
      sub_ex.ProjectCoefficient(coeff);

      sub_gf = 0.0;
      SubMesh::Transfer(gf, sub_gf);

      sub_gf -= sub_ex;
      REQUIRE(sub_gf.Norml2() < 1e-10);
   }

   SECTION("H1 sub-to-parent transfer")
   {
      H1_FECollection fec(2, mesh.Dimension());
      H1_FECollection sub_fec(2, sm.Dimension());
      FiniteElementSpace fes(&mesh, &fec);
      FiniteElementSpace sub_fes(&sm, &sub_fec);

      // Initialize parent with the exact projection so that DOFs outside the
      // submesh already hold the correct value.  Transfer overwrites only the
      // submesh DOFs, so the full parent should still match the exact projection.
      GridFunction gf(&fes), gf_ex(&fes), sub_gf(&sub_fes);
      gf.ProjectCoefficient(coeff);
      gf_ex = gf;
      sub_gf.ProjectCoefficient(coeff);

      SubMesh::Transfer(sub_gf, gf);

      gf -= gf_ex;
      REQUIRE(gf.Norml2() < 1e-10);
   }
}

/**
 * @brief Two disconnected patches of boundary elements (z=0 and z=1 faces)
 * extracted via CreateFromBdrElements.
 */
TEST_CASE("SubMesh from bdr elements disconnected", "[SubMesh]")
{
   Mesh mesh = Mesh::MakeCartesian3D(3, 3, 3, Element::HEXAHEDRON,
                                     1.0, 1.0, 1.0, false);

   // Boundary elements on z=0 (bottom) and z=1 (top) — disconnected faces
   Array<int> combined_bdr;
   int n_bottom = 0, n_top = 0;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      Array<int> verts;
      mesh.GetBdrElementVertices(i, verts);
      real_t z_avg = 0.0;
      for (int v : verts) { z_avg += mesh.GetVertex(v)[2]; }
      z_avg /= verts.Size();

      if (z_avg < 1e-10)
      {
         combined_bdr.Append(i);
         n_bottom++;
      }
      else if (z_avg > 1.0 - 1e-10)
      {
         combined_bdr.Append(i);
         n_top++;
      }
   }

   REQUIRE(n_bottom > 0);
   REQUIRE(n_top > 0);

   auto sm = SubMesh::CreateFromBdrElements(mesh, combined_bdr);

   REQUIRE(sm.GetNE() == combined_bdr.Size());
   REQUIRE(sm.Dimension() == mesh.Dimension() - 1);

   // Parent element ID map must cover exactly the requested boundary elements
   const auto &pids = sm.GetParentElementIDMap();
   REQUIRE(pids.Size() == combined_bdr.Size());

   // Every submesh vertex must be geometrically coincident with its parent vertex
   CheckVertexCoordinates(sm, mesh);

   auto coeff = FunctionCoefficient([](const Vector &c)
   {
      // Nontrivial on the z=0 and z=1 faces
      return 1.0 + 0.5*c(0) + 0.3*c(1);
   });

   SECTION("H1 transfer to disconnected surface submesh")
   {
      H1_FECollection fec(2, mesh.Dimension());
      H1_FECollection sub_fec(2, sm.Dimension());
      FiniteElementSpace fes(&mesh, &fec);
      FiniteElementSpace sub_fes(&sm, &sub_fec);

      GridFunction gf(&fes), sub_gf(&sub_fes), sub_ex(&sub_fes);
      gf.ProjectCoefficient(coeff);
      sub_ex.ProjectCoefficient(coeff);

      sub_gf = 0.0;
      SubMesh::Transfer(gf, sub_gf);

      sub_gf -= sub_ex;
      REQUIRE(sub_gf.Norml2() < 1e-10);
   }
}

TEST_CASE("SubMesh from elements, single element", "[SubMesh]")
{
   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL,
                                     false, 1.0, 1.0, false);
   int target = mesh.GetNE() / 2;
   Array<int> ids{target};

   auto sm = SubMesh::CreateFromElements(mesh, ids);

   REQUIRE(sm.GetNE() == 1);
   REQUIRE(sm.GetParentElementIDMap()[0] == target);

   // Vertex count must match the parent element
   Array<int> sub_verts, par_verts;
   sm.GetElementVertices(0, sub_verts);
   mesh.GetElementVertices(target, par_verts);
   REQUIRE(sub_verts.Size() == par_verts.Size());

   // Each submesh vertex must sit at the same coordinates as the parent vertex
   // it was mapped from.
   CheckVertexCoordinates(sm, mesh);
}

TEST_CASE("SubMesh from bdr elements, single element", "[SubMesh]")
{
   Mesh mesh = Mesh::MakeCartesian3D(3, 3, 3, Element::HEXAHEDRON,
                                     1.0, 1.0, 1.0, false);
   int target = 0;
   Array<int> ids{target};

   auto sm = SubMesh::CreateFromBdrElements(mesh, ids);

   REQUIRE(sm.GetNE() == 1);
   REQUIRE(sm.GetParentElementIDMap()[0] == target);
   REQUIRE(sm.Dimension() == 2);

   // Vertex count must match the parent boundary element
   Array<int> sub_verts, par_verts;
   sm.GetElementVertices(0, sub_verts);
   mesh.GetBdrElementVertices(target, par_verts);
   REQUIRE(sub_verts.Size() == par_verts.Size());

   CheckVertexCoordinates(sm, mesh);
}
