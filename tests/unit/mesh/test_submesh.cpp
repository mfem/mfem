// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

TEST_CASE("Non-Contiguous SubMesh", "[SubMesh]")
{
   std::string mesh_file = "../../data/checker-3d.mesh";

   std::ifstream ifs(mesh_file);
   Mesh mesh(ifs);

   REQUIRE(mesh.GetNV() == 64);
   REQUIRE(mesh.GetNE() == 27);
   REQUIRE(mesh.GetNBE() == 54);
   REQUIRE(mesh.EulerNumber() == 1); // 64 - 144 + 108 - 27

   Array<int> attr1(1); attr1 = 1;
   Mesh submesh1 = SubMesh::CreateFromDomain(mesh, attr1);

   // Corner elements (non-contiguous)
   REQUIRE(submesh1.GetNV() == 64);
   REQUIRE(submesh1.GetNE() == 8);
   REQUIRE(submesh1.GetNBE() == 48);
   REQUIRE(submesh1.EulerNumber() == 8); // 64 - 96 + 48 - 8

   // Edge elements (shared vertices and edges)
   Array<int> attr2(1); attr2 = 2;
   Mesh submesh2 = SubMesh::CreateFromDomain(mesh, attr2);

   REQUIRE(submesh2.GetNV() == 56);
   REQUIRE(submesh2.GetNE() == 12);
   REQUIRE(submesh2.GetNBE() == 72);
   REQUIRE(submesh2.EulerNumber() == -4); // 56 - 120  + 72 - 12

   // Corner and zone-center elements (shared vertices)
   Array<int> attr14(2); attr14[0] = 1; attr14[1] = 4;
   Mesh submesh14 = SubMesh::CreateFromDomain(mesh, attr14);

   REQUIRE(submesh14.GetNV() == 64);
   REQUIRE(submesh14.GetNE() == 9);
   REQUIRE(submesh14.GetNBE() == 54);
   REQUIRE(submesh14.EulerNumber() == 1); // 64 - 108 + 54 - 9

   // 3D checkerboard pattern including corners
   Array<int> attr13(2); attr13[0] = 1; attr13[1] = 3;
   SubMesh submesh13 = SubMesh::CreateFromDomain(mesh, attr13);

   REQUIRE(submesh13.GetNV() == 64);
   REQUIRE(submesh13.GetNE() == 14);
   REQUIRE(submesh13.GetNBE() == 84);
   REQUIRE(submesh13.EulerNumber() == 2); // 64 - 132 + 84 - 14

   // 3D checkerboard pattern excluding corners
   Array<int> attr24(2); attr24[0] = 2; attr24[1] = 4;
   SubMesh submesh24 = SubMesh::CreateFromDomain(mesh, attr24);

   REQUIRE(submesh24.GetNV() == 56);
   REQUIRE(submesh24.GetNE() == 13);
   REQUIRE(submesh24.GetNBE() == 78);
   REQUIRE(submesh24.EulerNumber() == 1); // 56 - 120 + 78 - 13
}

TEST_CASE("Non-Contiguous SubMesh Transfers", "[SubMesh]")
{
   int order = 3; // Produce multiple DoFs per edge and face
   int dim = GENERATE(2, 3);
   auto fec_type = GENERATE(FECType::H1, FECType::ND, FECType::RT, FECType::L2);

   std::string mesh_file = (dim == 2) ?
                           "../../data/checker-2d.mesh" : "../../data/checker-3d.mesh";

   CAPTURE(dim);
   CAPTURE(mesh_file);
   CAPTURE(fec_type);

   std::ifstream ifs(mesh_file);
   Mesh mesh(ifs);

   // 3D checkerboard pattern including corners
   Array<int> attr13(2); attr13[0] = 1; attr13[1] = 3;
   SubMesh submesh13 = SubMesh::CreateFromDomain(mesh, attr13);

   // 3D checkerboard pattern excluding corners
   Array<int> attr24(dim-1); attr24[0] = 2;
   if (dim == 3) { attr24[1] = 4; }
   SubMesh submesh24 = SubMesh::CreateFromDomain(mesh, attr24);

   // Project random field onto full domain
   FiniteElementCollection *fec = create_fec(fec_type, order, dim);
   FiniteElementSpace fes(&mesh, fec);
   GridFunction gf0(&fes); gf0.Randomize();
   real_t nrm = gf0.Norml2();
   CAPTURE(gf0.Size());
   REQUIRE(nrm != 0.0);

   // Transfer portion of random field to submesh
   FiniteElementSpace fes13(&submesh13, fec);
   GridFunction gf13(&fes13);
   SubMesh::Transfer(gf0, gf13);

   // Transfer complimentary portion of random field to submesh
   FiniteElementSpace fes24(&submesh24, fec);
   GridFunction gf24(&fes24);
   SubMesh::Transfer(gf0, gf24);

   // Create a target field on the full domain
   GridFunction gf1(&fes);

   // Transfer both portions and compare to original field
   SubMesh::Transfer(gf13, gf1);
   SubMesh::Transfer(gf24, gf1);
   gf1 -= gf0;
   REQUIRE(gf1.Norml2() < 1e-10);

   // Transfer both portions in the opposite order and compare to original field
   SubMesh::Transfer(gf24, gf1);
   SubMesh::Transfer(gf13, gf1);
   gf1 -= gf0;
   REQUIRE(gf1.Norml2() < 1e-10 * nrm);

   delete fec;
}
