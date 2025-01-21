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
#include "mesh_test_utils.hpp"
#include "unit_tests.hpp"

#include <array>
namespace mfem
{

constexpr real_t EPS = 1e-10;

// Test case: Verify that a conforming mesh yields the same norm for the
//            assembled diagonal with PA when using the standard (conforming)
//            Mesh vs. the corresponding (non-conforming) NCMesh. (note:
//            permutations of the values in the diagonal are expected)
TEST_CASE("NCMesh PA diagonal", "[NCMesh]")
{
   SECTION("Quad mesh")
   {
      int ne = 2;
      Mesh mesh = Mesh::MakeCartesian2D(
                     ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      Mesh nc_mesh = Mesh::MakeCartesian2D(
                        ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      nc_mesh.EnsureNCMesh();

      mesh.UniformRefinement();
      nc_mesh.UniformRefinement();

      int dim = 2;
      for (int order = 1; order <= 3; ++order)
      {
         ND_FECollection fec(order, dim);

         FiniteElementSpace fes(&mesh, &fec);
         FiniteElementSpace nc_fes(&nc_mesh, &fec);

         BilinearForm a(&fes);
         BilinearForm nc_a(&nc_fes);

         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         nc_a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

         ConstantCoefficient coef(1.0);
         a.AddDomainIntegrator(new CurlCurlIntegrator(coef));
         nc_a.AddDomainIntegrator(new CurlCurlIntegrator(coef));

         a.Assemble();
         nc_a.Assemble();

         Vector diag(fes.GetTrueVSize());
         Vector nc_diag(nc_fes.GetTrueVSize());
         a.AssembleDiagonal(diag);
         nc_a.AssembleDiagonal(nc_diag);

         real_t error = fabs(diag.Norml2() - nc_diag.Norml2());
         CAPTURE(order, error);
         REQUIRE(error == MFEM_Approx(0.0, EPS));
      }
   }

   SECTION("Hexa mesh")
   {
      int ne = 2;
      Mesh mesh = Mesh::MakeCartesian3D(
                     ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      Mesh nc_mesh = Mesh::MakeCartesian3D(
                        ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      nc_mesh.EnsureNCMesh();

      mesh.UniformRefinement();
      nc_mesh.UniformRefinement();

      int dim = 3;
      for (int order = 1; order <= 3; ++order)
      {
         ND_FECollection fec(order, dim);

         FiniteElementSpace fes(&mesh, &fec);
         FiniteElementSpace nc_fes(&nc_mesh, &fec);

         BilinearForm a(&fes);
         BilinearForm nc_a(&nc_fes);

         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         nc_a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

         ConstantCoefficient coef(1.0);
         a.AddDomainIntegrator(new CurlCurlIntegrator(coef));
         nc_a.AddDomainIntegrator(new CurlCurlIntegrator(coef));

         a.Assemble();
         nc_a.Assemble();

         Vector diag(fes.GetTrueVSize());
         Vector nc_diag(nc_fes.GetTrueVSize());
         a.AssembleDiagonal(diag);
         nc_a.AssembleDiagonal(nc_diag);

         real_t error = fabs(diag.Sum() - nc_diag.Sum());
         CAPTURE(order, error);
         REQUIRE(error == MFEM_Approx(0.0, EPS));
      }
   }

} // test case

TEST_CASE("NCMesh 3D Refined Volume", "[NCMesh]")
{
   auto mesh_fname = GENERATE("../../data/ref-tetrahedron.mesh",
                              "../../data/ref-cube.mesh",
                              "../../data/ref-prism.mesh",
                              "../../data/ref-pyramid.mesh"
                             );

   auto ref_type = GENERATE(Refinement::X,
                            Refinement::Y,
                            Refinement::Z,
                            Refinement::XY,
                            Refinement::XZ,
                            Refinement::YZ,
                            Refinement::XYZ);

   const real_t scale = GENERATE(0.5, 0.25);  // Only affects hex mesh so far

   if (scale != 0.5 && std::strcmp(mesh_fname, "../../data/ref-cube.mesh") != 0)
   {
      return;
   }

   Mesh mesh(mesh_fname, 1, 1);
   mesh.EnsureNCMesh(true);
   real_t original_volume = mesh.GetElementVolume(0);
   Array<Refinement> ref(1);
   ref[0].Set(0, ref_type, scale);

   mesh.GeneralRefinement(ref, 1);
   real_t summed_volume = 0.0;
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      summed_volume += mesh.GetElementVolume(i);
   }
   REQUIRE(summed_volume == MFEM_Approx(original_volume));
} // test case

TEST_CASE("NCMesh 3D Derefined Volume", "[NCMesh]")
{
   auto mesh_fname = GENERATE("../../data/ref-tetrahedron.mesh",
                              "../../data/ref-cube.mesh",
                              "../../data/ref-prism.mesh",
                              "../../data/ref-pyramid.mesh"
                             );

   auto ref_type = GENERATE(Refinement::XYZ);

   Mesh mesh(mesh_fname, 1, 1);
   mesh.EnsureNCMesh(true);
   real_t original_volume = mesh.GetElementVolume(0);
   Array<Refinement> ref(1);
   ref[0].Set(0, ref_type);

   mesh.GeneralRefinement(ref, 1);

   Array<real_t> elem_error(mesh.GetNE());
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      elem_error[i] = 0.0;
   }
   mesh.DerefineByError(elem_error, 1.0);

   real_t derefined_volume = mesh.GetElementVolume(0);
   REQUIRE(derefined_volume == MFEM_Approx(original_volume));
} // test case


#ifdef MFEM_USE_MPI

// Test case: Verify that a conforming mesh yields the same norm for the
//            assembled diagonal with PA when using the standard (conforming)
//            Mesh vs. the corresponding (non-conforming) NCMesh. (note:
//            permutations of the values in the diagonal are expected)
TEST_CASE("pNCMesh PA diagonal",  "[Parallel], [NCMesh]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   SECTION("Quad pmesh")
   {
      int ne = 2;
      Mesh mesh = Mesh::MakeCartesian2D(
                     ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      Mesh nc_mesh = Mesh::MakeCartesian2D(
                        ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      nc_mesh.EnsureNCMesh();

      mesh.UniformRefinement();
      nc_mesh.UniformRefinement();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      ParMesh nc_pmesh(MPI_COMM_WORLD, nc_mesh);

      int dim = 2;
      for (int order = 1; order <= 3; ++order)
      {
         ND_FECollection fec(order, dim);

         ParFiniteElementSpace pfes(&pmesh, &fec);
         ParFiniteElementSpace nc_pfes(&nc_pmesh, &fec);

         ParBilinearForm a(&pfes);
         ParBilinearForm nc_a(&nc_pfes);

         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         nc_a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

         ConstantCoefficient coef(1.0);
         a.AddDomainIntegrator(new CurlCurlIntegrator(coef));
         nc_a.AddDomainIntegrator(new CurlCurlIntegrator(coef));

         a.Assemble();
         nc_a.Assemble();

         Vector diag(pfes.GetTrueVSize());
         Vector nc_diag(nc_pfes.GetTrueVSize());
         a.AssembleDiagonal(diag);
         nc_a.AssembleDiagonal(nc_diag);

         real_t diag_lsum = diag.Sum(), nc_diag_lsum = nc_diag.Sum();
         real_t diag_gsum = 0.0, nc_diag_gsum = 0.0;
         MPI_Allreduce(&diag_lsum, &diag_gsum, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
         MPI_Allreduce(&nc_diag_lsum, &nc_diag_gsum, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);

         real_t error = fabs(diag_gsum - nc_diag_gsum);
         CAPTURE(order, error);
         REQUIRE(error == MFEM_Approx(0.0, EPS));
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }

   SECTION("Hexa pmesh")
   {
      int ne = 2;
      Mesh mesh = Mesh::MakeCartesian3D(
                     ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      Mesh nc_mesh = Mesh::MakeCartesian3D(
                        ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      nc_mesh.EnsureNCMesh();

      mesh.UniformRefinement();
      nc_mesh.UniformRefinement();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      ParMesh nc_pmesh(MPI_COMM_WORLD, nc_mesh);

      int dim = 3;
      for (int order = 1; order <= 3; ++order)
      {
         ND_FECollection fec(order, dim);

         ParFiniteElementSpace pfes(&pmesh, &fec);
         ParFiniteElementSpace nc_pfes(&nc_pmesh, &fec);

         ParBilinearForm a(&pfes);
         ParBilinearForm nc_a(&nc_pfes);

         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         nc_a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

         ConstantCoefficient coef(1.0);
         a.AddDomainIntegrator(new CurlCurlIntegrator(coef));
         nc_a.AddDomainIntegrator(new CurlCurlIntegrator(coef));

         a.Assemble();
         nc_a.Assemble();

         Vector diag(pfes.GetTrueVSize());
         Vector nc_diag(nc_pfes.GetTrueVSize());
         a.AssembleDiagonal(diag);
         nc_a.AssembleDiagonal(nc_diag);

         real_t diag_lsum = diag.Sum(), nc_diag_lsum = nc_diag.Sum();
         real_t diag_gsum = 0.0, nc_diag_gsum = 0.0;
         MPI_Allreduce(&diag_lsum, &diag_gsum, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_SUM, MPI_COMM_WORLD);
         MPI_Allreduce(&nc_diag_lsum, &nc_diag_gsum, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);

         CAPTURE(order, diag_gsum, nc_diag_gsum);
         REQUIRE(nc_diag_gsum == MFEM_Approx(diag_gsum));
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }
} // test case

TEST_CASE("EdgeFaceConstraint", "[Parallel], [NCMesh]")
{
   auto exact_soln = [](const Vector& x)
   {
      // sin(|| x - d ||^2) -> non polynomial but very smooth.
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      d -= x;
      return std::sin(d * d);
   };

   SECTION("ReferenceTet")
   {
      constexpr int refining_rank = 0;
      auto smesh = Mesh("../../data/ref-tetrahedron.mesh");

      REQUIRE(smesh.GetNE() == 1);
      {
         // Start the test with two tetrahedra attached by triangle.
         auto single_edge_refine = Array<Refinement>(1);
         single_edge_refine[0].Set(0, Refinement::X);
         smesh.GeneralRefinement(single_edge_refine, 0); // conformal
      }

      REQUIRE(smesh.GetNE() == 2);
      smesh.EnsureNCMesh(true);
      smesh.Finalize();

      auto partition = std::unique_ptr<int[]>(new int[smesh.GetNE()]);
      partition[0] = 0;
      partition[1] = Mpi::WorldSize() > 1 ? 1 : 0;

      auto pmesh = ParMesh(MPI_COMM_WORLD, smesh, partition.get());

      // Construct the NC refined mesh in parallel and serial. Once constructed
      // a global L2 projected solution should match exactly on each.
      Array<int> refines, serial_refines(1);
      if (Mpi::WorldRank() == refining_rank)
      {
         refines.Append(0);
      }

      // Must be called on all ranks as it uses MPI calls internally. All ranks
      // will use the global element number dictated by rank 0 though.
      serial_refines[0] = pmesh.GetGlobalElementNum(0);
      MPI_Bcast(&serial_refines[0], 1, MPI_INT, refining_rank, MPI_COMM_WORLD);

      // Rank 0 refines the parallel mesh, all ranks refine the serial mesh
      smesh.GeneralRefinement(serial_refines, 1); // nonconformal
      pmesh.GeneralRefinement(refines, 1); // nonconformal

      REQUIRE(pmesh.GetGlobalNE() == 8 + 1);
      REQUIRE(smesh.GetNE() == 8 + 1);

      // Each pair of indices here represents sequential element indices to
      // refine. First the i element is refined, then in the resulting mesh the
      // j element is refined. These pairs were arrived at by looping over all
      // possible i,j pairs and checking for the addition of a face-edge
      // constraint.
      std::vector<std::pair<int,int>> indices{{2,13}, {3,13}, {6,2}, {6,3}};

      // Rank 0 has all but one element in the parallel mesh. The remaining
      // element is owned by another processor if the number of ranks is greater
      // than one.
      for (const auto &ij : indices)
      {
         int i = ij.first;
         int j = ij.second;
         if (Mpi::WorldRank() == refining_rank)
         {
            refines[0] = i;
         }
         // Inform all ranks of the serial mesh
         serial_refines[0] = pmesh.GetGlobalElementNum(i);
         MPI_Bcast(&serial_refines[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

         ParMesh tmp(pmesh);
         tmp.GeneralRefinement(refines);

         REQUIRE(tmp.GetGlobalNE() == 1 + 8 - 1 + 8); // 16 elements

         Mesh stmp(smesh);
         stmp.GeneralRefinement(serial_refines);
         REQUIRE(stmp.GetNE() == 1 + 8 - 1 + 8); // 16 elements

         if (Mpi::WorldRank() == refining_rank)
         {
            refines[0] = j;
         }
         // Inform all ranks of the serial mesh
         serial_refines[0] = tmp.GetGlobalElementNum(j);
         MPI_Bcast(&serial_refines[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

         ParMesh ttmp(tmp);
         ttmp.GeneralRefinement(refines);

         REQUIRE(ttmp.GetGlobalNE() == 1 + 8 - 1 + 8 - 1 + 8); // 23 elements

         Mesh sttmp(stmp);
         sttmp.GeneralRefinement(serial_refines);
         REQUIRE(sttmp.GetNE() == 1 + 8 - 1 + 8 - 1 + 8); // 23 elements

         // Loop over interior faces, fill and check face transform on the
         // serial.
         for (int iface = 0; iface < sttmp.GetNumFaces(); ++iface)
         {
            const auto face_transform = sttmp.GetFaceElementTransformations(iface);
            CHECK(face_transform->CheckConsistency(0) < 1e-12);
         }

         for (int iface = 0; iface < ttmp.GetNumFacesWithGhost(); ++iface)
         {
            const auto face_transform = ttmp.GetFaceElementTransformations(iface);
            CHECK(face_transform->CheckConsistency(0) < 1e-12);
         }

         // Use P4 to ensure there's a few fully interior DOF.
         {
            auto error = CheckL2Projection(ttmp, sttmp, 4, exact_soln);
            real_t constexpr tol = 1e-9;
            CHECK(std::abs(error[1] - error[0]) < tol);
         }
         ttmp.ExchangeFaceNbrData();
         ttmp.Rebalance();
         {
            auto error = CheckL2Projection(ttmp, sttmp, 4, exact_soln);
            real_t constexpr tol = 1e-9;
            CHECK(std::abs(error[1] - error[0]) < tol);
         }
      }
   }

   auto check_serial_parallel_h1_equivalance = [](Mesh &smesh)
   {
      constexpr int dim = 3;
      constexpr int order = 2;
      H1_FECollection nd_fec(order, dim);
      FiniteElementSpace fes(&smesh, &nd_fec);
      const auto serial_ntdof = fes.GetTrueVSize();

      ParMesh mesh(MPI_COMM_WORLD, smesh);
      ParFiniteElementSpace pfes(&mesh, &nd_fec);
      const auto parallel_ntdof = pfes.GlobalTrueVSize();

      // If nc constraints have been observed correctly, the number of true dof
      // in parallel should match the number of true dof in serial. If the
      // number of parallel dofs is greater, then a slave constraint has not
      // been fully labeled.
      CHECK(serial_ntdof == parallel_ntdof);
   };

   auto check_serial_parallel_nd_equivalence = [](Mesh &smesh)
   {
      constexpr int dim = 3;
      constexpr int order = 1;
      ND_FECollection nd_fec(order, dim);
      FiniteElementSpace fes(&smesh, &nd_fec);
      const auto serial_ntdof = fes.GetTrueVSize();

      ParMesh mesh(MPI_COMM_WORLD, smesh);
      ParFiniteElementSpace pfes(&mesh, &nd_fec);
      const auto parallel_ntdof = pfes.GlobalTrueVSize();

      // If nc constraints have been observed correctly, the number of true dof
      // in parallel should match the number of true dof in serial. If the
      // number of parallel dofs is greater, then a slave constraint has not
      // been fully labeled.
      CHECK(serial_ntdof == parallel_ntdof);
   };

   SECTION("LevelTwoRefinement")
   {
      Mesh smesh("../../data/ref-tetrahedron.mesh");
      Array<Refinement> aniso_ref(1);
      aniso_ref[0].Set(0, Refinement::X);
      smesh.GeneralRefinement(aniso_ref);
      smesh.UniformRefinement();
      smesh.EnsureNCMesh(true);
      Array<int> el_to_refine(1);

      for (int n = 0; n < smesh.GetNE(); n++)
      {
         Mesh smesh2(smesh);
         el_to_refine[0] = n;
         smesh2.GeneralRefinement(el_to_refine);
         for (int m = 0; m < smesh2.GetNE(); m++)
         {
            Mesh smesh3(smesh2);
            el_to_refine[0] = m;
            smesh3.GeneralRefinement(el_to_refine);
            CAPTURE(n,m);
            check_serial_parallel_nd_equivalence(smesh3);
            check_serial_parallel_h1_equivalance(smesh3);
         }
      }
   }

   SECTION("EdgeCasePartition")
   {
      Mesh smesh("../../data/ref-tetrahedron.mesh");
      smesh.UniformRefinement();
      smesh.EnsureNCMesh(true);
      Array<int> el_to_refine(1);

      el_to_refine[0] = 0;
      smesh.GeneralRefinement(el_to_refine);

      // This particular partition was found by brute force search. The default
      // rebalancing can in rare cases produce similar local patterns,
      // particularly for highly adapted meshes.
      auto partition = std::unique_ptr<int[]>(new int[smesh.GetNE()]);
      if (Mpi::WorldSize() > 1)
      {
         auto bad_partition = std::vector<int> {0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
         std::copy(bad_partition.begin(), bad_partition.end(), partition.get());
      }
      else
      {
         for (int i = 0; i < smesh.GetNE(); i++)
         {
            partition[i] = 0;
         }
      }
      ParMesh pmesh(MPI_COMM_WORLD, smesh, partition.get());
      {
         constexpr int dim = 3;
         constexpr int order = 1;
         ND_FECollection nd_fec(order, dim);
         FiniteElementSpace fes(&smesh, &nd_fec);
         const auto serial_ntdof = fes.GetTrueVSize();
         ParFiniteElementSpace pfes(&pmesh, &nd_fec);
         pfes.ExchangeFaceNbrData();
         const auto parallel_ntdof = pfes.GlobalTrueVSize();
         CHECK(serial_ntdof == parallel_ntdof);
      }
      for (int order = 1; order <= 4; order++)
      {
         CAPTURE(order);
         auto error = CheckL2Projection(pmesh, smesh, order, exact_soln);
         real_t constexpr tol = 1e-9;
         CHECK(std::abs(error[1] - error[0]) < tol);
      }
   }

} // test case

TEST_CASE("P2Q1PureTetHexPri",  "[Parallel], [NCMesh]")
{
   auto exact_soln = [](const Vector& x)
   {
      // sin(|| x - d ||^2) -> non polynomial but very smooth.
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      d -= x;
      return std::sin(d * d);
   };

   auto el_type = GENERATE(Geometry::TETRAHEDRON,
                           Geometry::CUBE,
                           Geometry::PRISM);
   int variant = GENERATE(0,1,2);

   if (variant > 0 && el_type != Geometry::PRISM)
   {
      return;
   }

   CAPTURE(el_type, variant);

   auto smesh = CylinderMesh(el_type, false, variant);

   for (auto ref : {0,1,2})
   {
      if (ref == 1) { smesh.UniformRefinement(); }

      smesh.EnsureNCMesh(true);

      if (ref == 2) { smesh.UniformRefinement(); }

      smesh.Finalize();

      auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);

      // P2 ensures there are triangles without dofs
      auto error = CheckL2Projection(pmesh, smesh, 2, exact_soln);
      CHECK(std::abs(error[1] - error[0]) < 1e-9);
   }
} // test case

TEST_CASE("PNQ2PureTetHexPri",  "[Parallel], [NCMesh]")
{
   auto exact_soln = [](const Vector& x)
   {
      // sin(|| x - d ||^2) -> non polynomial but very smooth.
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      d -= x;
      return std::sin(d * d);
   };

   auto el_type = GENERATE(Geometry::TETRAHEDRON,
                           Geometry::CUBE,
                           Geometry::PRISM);
   int variant = GENERATE(0,1,2);

   if (variant > 0 && el_type != Geometry::PRISM)
   {
      return;
   }

   CAPTURE(el_type, variant);

   auto smesh = CylinderMesh(el_type, true);

   for (auto ref : {0,1,2})
   {
      if (ref == 1) { smesh.UniformRefinement(); }

      smesh.EnsureNCMesh(true);

      if (ref == 2) { smesh.UniformRefinement(); }

      smesh.Finalize();

      auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);

      for (int p = 1; p < 3; ++p)
      {
         auto error = CheckL2Projection(pmesh, smesh, p, exact_soln);
         CHECK(std::abs(error[1] - error[0]) < 1e-9);
      }
   }
} // test case

TEST_CASE("GetVectorValueInFaceNeighborElement", "[Parallel], [NCMesh]")
{
   // The aim of this test is to verify the correct behaviour of the
   // GetVectorValue method when called on face neighbor elements in a non
   // conforming mesh.
   auto smesh = Mesh("../../data/beam-tet.mesh");

   for (int nc_level : {0,1,2,3})
   {
      for (int skip : {1,2})
      {
         for (bool use_ND : {false, true})
         {
            TestVectorValueInVolume(smesh, nc_level, skip, use_ND);
         }
      }
   }
}

TEST_CASE("TetCornerRefines", "[Parallel], [NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = OrientedTriFaceMesh(1, true);
   smesh.EnsureNCMesh(true);

   auto pmesh = CheckParMeshNBE(smesh);

   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   REQUIRE(pmesh->Nonconforming());

   std::map<int, int> local_to_shared;
   for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
   {
      local_to_shared[pmesh->GetSharedFace(i)] = i;
   }

   // Count the number of internal faces via the boundary elements
   int num_internal = 0;
   for (int n = 0; n < pmesh->GetNBE(); ++n)
   {
      int f, o;
      pmesh->GetBdrElementFace(n, &f, &o);
      if (CheckFaceInternal(*pmesh, f, local_to_shared))
      {
         ++num_internal;
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK(num_internal == 1);

   CheckPoisson(*pmesh, p, pmesh->bdr_attributes.Max());
   CheckPoisson(*pmesh, p);
}

TEST_CASE("InteriorBoundaryReferenceTets", "[Parallel], [NCMesh]")
{
   constexpr auto seed = 314159;
   srand(seed);
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = OrientedTriFaceMesh(1, true);
   smesh.EnsureNCMesh(true);

   auto pmesh = CheckParMeshNBE(smesh);

   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   REQUIRE(pmesh->Nonconforming());

   std::map<int, int> local_to_shared;
   for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
   {
      local_to_shared[pmesh->GetSharedFace(i)] = i;
   }

   // Count the number of internal faces via the boundary elements
   int num_internal = 0;
   for (int n = 0; n < pmesh->GetNBE(); ++n)
   {
      int f, o;
      pmesh->GetBdrElementFace(n, &f, &o);
      if (CheckFaceInternal(*pmesh, f, local_to_shared))
      {
         ++num_internal;
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK(num_internal == 1);

   CheckPoisson(*pmesh, p, pmesh->bdr_attributes.Max());
   CheckPoisson(*pmesh, p);

   int num_initial_ess_tdof = CountEssentialDof<H1_FECollection>(*pmesh, p,
                                                                 smesh.bdr_attributes.Max());
   if (Mpi::Root())
   {
      REQUIRE(num_initial_ess_tdof > 0);
   }
   // Level of refinement difference across the processor boundary from root
   // zero to the others
   auto ref_level = GENERATE(1,2,3);
   auto refined_attribute = GENERATE(1,2);
   CAPTURE(ref_level);
   CAPTURE(refined_attribute);

   Mesh modified_smesh(smesh);
   for (int r = 0; r < ref_level; r++)
   {
      Array<int> el_to_refine;
      for (int n = 0; n < modified_smesh.GetNE(); n++)
      {
         if (modified_smesh.GetAttribute(n) == refined_attribute)
         {
            el_to_refine.Append(n);
         }
      }
      modified_smesh.GeneralRefinement(el_to_refine);
   }

   // There should now be some internal boundary elements, where there was one
   // before.
   CHECK(modified_smesh.GetNBE() == 3 /* external boundaries of unrefined  */
         + std::pow(4, ref_level) /* internal boundaries */
         + (3 * std::pow(4, ref_level)) /* external boundaries of refined */);

   // Force the partition to have the edge case of a parent and child being
   // divided across the processor boundary.
   auto partition = std::unique_ptr<int[]>(new int[modified_smesh.GetNE()]);
   for (int i = 0; i < modified_smesh.GetNE(); i++)
   {
      // Randomly assign to any processor but zero.
      partition[i] = Mpi::WorldSize() > 1 ? 1 + rand() % (Mpi::WorldSize() - 1) : 0;
   }
   if (Mpi::WorldSize() > 0)
   {
      // Make sure rank 0 has the non-refined attribute. This ensures it will
      // have a parent face with only ghost children.
      const int unrefined_attribute = refined_attribute == 1 ? 2 : 1;
      Array<int> root_element;
      for (int n = 0; n < modified_smesh.GetNE(); n++)
      {
         if (modified_smesh.GetAttribute(n) == unrefined_attribute)
         {
            root_element.Append(n);
         }
      }
      REQUIRE(root_element.Size() == 1);
      partition[root_element[0]] = 0;
   }

   pmesh = CheckParMeshNBE(modified_smesh, partition);
   pmesh->Finalize();
   pmesh->FinalizeTopology();
   pmesh->ExchangeFaceNbrData();

   auto check_faces = [&]()
   {
      // repopulate the local to shared map.
      local_to_shared.clear();
      for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
      {
         local_to_shared[pmesh->GetSharedFace(i)] = i;
      }

      // Count the number of internal faces via the boundary elements
      num_internal = 0;
      for (int n = 0; n < pmesh->GetNBE(); ++n)
      {
         int f, o;
         pmesh->GetBdrElementFace(n, &f, &o);
         if (CheckFaceInternal(*pmesh, f, local_to_shared))
         {
            ++num_internal;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      CHECK(num_internal == std::pow(4, ref_level));
      CheckPoisson(*pmesh, p, smesh.bdr_attributes.Max());
      CheckPoisson(*pmesh, p);
   };

   check_faces();
   pmesh->Rebalance();
   pmesh->ExchangeFaceNbrData();
   check_faces();
}

TEST_CASE("InteriorBoundaryInlineRefines", "[Parallel], [NCMesh]")
{
   const auto use_tet = GENERATE(false, true);
   const int p = use_tet ? GENERATE(1,2,3) : GENERATE(1,2);
   CAPTURE(p);

   const auto fname = use_tet ? "../../data/inline-tet.mesh" :
                      "../../data/inline-hex.mesh";
   auto smesh = Mesh(fname);
   smesh.FinalizeTopology();
   smesh.Finalize();

   // Mark even and odd elements with different attributes
   const auto num_attributes = 3;
   for (int i = 0; i < smesh.GetNE(); ++i)
   {
      smesh.SetAttribute(i, (i % num_attributes) + 1);
   }

   smesh.SetAttributes();
   const int initial_nbe = smesh.GetNBE();

   // Introduce internal boundary elements
   const int new_attribute = smesh.bdr_attributes.Max() + 1;
   for (int f = 0; f < smesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = smesh.GetFace(f)->Duplicate(&smesh);
         new_elem->SetAttribute(new_attribute);
         smesh.AddBdrElement(new_elem);
      }
   }

   smesh.FinalizeTopology(); // Finalize to build relevant tables
   smesh.Finalize();

   smesh.EnsureNCMesh(true);

   // Boundary elements must've been added to make the test valid
   int num_internal_serial = smesh.GetNBE() - initial_nbe;
   REQUIRE(num_internal_serial > 0);

   auto partition = std::unique_ptr<int[]>(new int[smesh.GetNE()]);

   for (int i = 0; i < smesh.GetNE(); ++i)
   {
      partition[i] = i % Mpi::WorldSize(); // checkerboard partition
   }

   auto pmesh = CheckParMeshNBE(smesh, partition);

   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   std::map<int, int> local_to_shared;
   for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
   {
      local_to_shared[pmesh->GetSharedFace(i)] = i;
   }

   // Count the number of internal faces via the boundary elements
   int num_internal = 0;
   for (int n = 0; n < pmesh->GetNBE(); ++n)
   {
      int f, o;
      pmesh->GetBdrElementFace(n, &f, &o);
      if (CheckFaceInternal(*pmesh, f, local_to_shared))
      {
         ++num_internal;
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   CHECK(num_internal == num_internal_serial);

   CheckPoisson(*pmesh, p, new_attribute);
   CheckPoisson(*pmesh, p);

   // Mark all elements of a given attribute for refinement to a given depth.
   const auto ref_level = GENERATE(1,2);
   const auto marked_attribute = GENERATE(1,2,3);
   REQUIRE(marked_attribute <= num_attributes);
   CAPTURE(ref_level);
   CAPTURE(marked_attribute);
   for (int r = 0; r < ref_level; r++)
   {
      Array<int> elem_to_refine;
      for (int i = 0; i < smesh.GetNE(); ++i)
      {
         if (smesh.GetAttribute(i) == marked_attribute)
         {
            elem_to_refine.Append(i);
         }
      }
      smesh.GeneralRefinement(elem_to_refine);
   }

   pmesh = CheckParMeshNBE(smesh);
   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   // Count the number of internal boundary elements
   num_internal_serial = 0;
   for (int n = 0; n < smesh.GetNBE(); ++n)
   {
      int f, o;
      smesh.GetBdrElementFace(n, &f, &o);
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         ++num_internal_serial;
      }
   }

   auto check_faces = [&]()
   {
      // repopulate the local to shared map.
      local_to_shared.clear();
      for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
      {
         local_to_shared[pmesh->GetSharedFace(i)] = i;
      }

      num_internal = 0;
      for (int n = 0; n < pmesh->GetNBE(); ++n)
      {
         int f, o;
         pmesh->GetBdrElementFace(n, &f, &o);
         if (CheckFaceInternal(*pmesh, f, local_to_shared))
         {
            ++num_internal;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      CHECK(num_internal == num_internal_serial);

      CheckPoisson(*pmesh, p, pmesh->bdr_attributes.Max());
      CheckPoisson(*pmesh, p);
   };

   check_faces();
   pmesh->Rebalance();
   pmesh->ExchangeFaceNbrData();
   check_faces();
}

TEST_CASE("InteriorBoundaryReferenceCubes", "[Parallel], [NCMesh]")
{
   auto p = GENERATE(1,2);
   CAPTURE(p);

   auto smesh = DividingPlaneMesh(false, true);

   auto pmesh = CheckParMeshNBE(smesh);

   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   REQUIRE(pmesh->Conforming());

   std::map<int, int> local_to_shared;
   for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
   {
      local_to_shared[pmesh->GetSharedFace(i)] = i;
   }

   // Count the number of internal faces via the boundary elements
   int num_internal = 0;
   for (int n = 0; n < pmesh->GetNBE(); ++n)
   {
      int f, o;
      pmesh->GetBdrElementFace(n, &f, &o);
      if (CheckFaceInternal(*pmesh, f, local_to_shared))
      {
         ++num_internal;
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK(num_internal == 1);

   CheckPoisson(*pmesh, p, pmesh->bdr_attributes.Max());
   CheckPoisson(*pmesh, p);

   for (int refined_elem : {0, 1})
   {
      // Now NC refine one of the attached elements, this should result in 4
      // internal boundary elements.
      Array<int> el_to_refine;
      el_to_refine.Append(refined_elem);

      Mesh modified_smesh(smesh);
      modified_smesh.GeneralRefinement(el_to_refine);

      // There should now be four internal boundary elements, where there was
      // one before.
      CHECK(modified_smesh.GetNBE() == 5 /* external boundaries of unrefined  */
            + 4 /* internal boundaries */
            + (5 * 4) /* external boundaries of refined */);

      // Force the partition to have the edge case of a parent and child being
      // divided across the processor boundary. This necessitates the
      // GhostBoundaryElement treatment.
      auto partition = std::unique_ptr<int[]>(new int[modified_smesh.GetNE()]);
      srand(314159);
      for (int i = 0; i < modified_smesh.GetNE(); ++i)
      {
         // Randomly assign to any processor but zero.
         partition[i] = Mpi::WorldSize() > 1 ? 1 + rand() % (Mpi::WorldSize() - 1) : 0;
      }
      if (Mpi::WorldSize() > 0)
      {
         // Make sure on rank 1 there is a parent face with only ghost child
         // faces. This can cause issues with higher order dofs being
         // uncontrolled.
         partition[refined_elem == 0 ? modified_smesh.GetNE() - 1 : 0] = 0;
      }

      pmesh = CheckParMeshNBE(modified_smesh, partition);
      pmesh->Finalize();
      pmesh->FinalizeTopology();
      pmesh->ExchangeFaceNbrData();

      auto check_faces = [&]()
      {
         // repopulate the local to shared map.
         local_to_shared.clear();
         for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
         {
            local_to_shared[pmesh->GetSharedFace(i)] = i;
         }

         // Count the number of internal faces via the boundary elements
         num_internal = 0;
         for (int n = 0; n < pmesh->GetNBE(); ++n)
         {
            int f, o;
            pmesh->GetBdrElementFace(n, &f, &o);
            if (CheckFaceInternal(*pmesh, f, local_to_shared))
            {
               ++num_internal;
            }
         }
         MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
         CHECK(num_internal == 4);

         CAPTURE(refined_elem);
         CheckPoisson(*pmesh, p, smesh.bdr_attributes.Max());
         CheckPoisson(*pmesh, p);
      };

      check_faces();
      pmesh->Rebalance();
      pmesh->ExchangeFaceNbrData();
      check_faces();
   }
}

TEST_CASE("ParMeshInternalBoundaryTetStarMesh", "[Parallel], [NCMesh]")
{
   auto smesh = TetStarMesh();
   smesh.EnsureNCMesh(true);

   if (Mpi::WorldSize() < 5) { return; }

   auto partition = std::unique_ptr<int[]>(new int[5]);
   for (int i = 0; i < 5; i++)
   {
      partition[i] = i;
   }
   auto pmesh = CheckParMeshNBE(smesh, partition);
   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   REQUIRE(pmesh->Nonconforming());

   std::map<int, int> local_to_shared;
   for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
   {
      local_to_shared[pmesh->GetSharedFace(i)] = i;
   }

   // Count the number of internal faces via the boundary elements
   int num_internal = 0;
   for (int n = 0; n < pmesh->GetNBE(); ++n)
   {
      int f, o;
      pmesh->GetBdrElementFace(n, &f, &o);
      if (CheckFaceInternal(*pmesh, f, local_to_shared))
      {
         ++num_internal;
      }
   }

   const int rank = Mpi::WorldRank();
   SECTION("Unrefined")
   {
      MPI_Allreduce(MPI_IN_PLACE, &num_internal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      CHECK(num_internal == 4);

      CHECK(CountEssentialDof<H1_FECollection>(*pmesh, 1,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 4 : 0));
      CHECK(CountEssentialDof<H1_FECollection>(*pmesh, 2,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 4 + 6 : 0));
      CHECK(CountEssentialDof<H1_FECollection>(*pmesh, 3,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 4 + 6*2 + 4*1 : 0));
      CHECK(CountEssentialDof<H1_FECollection>(*pmesh, 4,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 4 + 6*3 + 4*3 : 0));

      CHECK(CountEssentialDof<ND_FECollection>(*pmesh, 1,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 6 : 0));
      CHECK(CountEssentialDof<ND_FECollection>(*pmesh, 2,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 20 : 0));
      CHECK(CountEssentialDof<ND_FECollection>(*pmesh, 3,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 42 : 0));
      CHECK(CountEssentialDof<ND_FECollection>(*pmesh, 4,
                                               smesh.bdr_attributes.Max()) == (rank == 0 ? 72 : 0));
      CHECK(pmesh->GetNBE() == (rank == 0 ? 4 : (rank < 5 ? 3 : 0)));
   }

   SECTION("Refinement")
   {
      // Refining an element attached to the core should not change the number
      // of essential DOF, or the owner of them.

      const int refined_attribute = GENERATE(1,2,3,4,5); // equal to rank of owner + 1
      int ref_level = GENERATE(0, 1, 2, 3);
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < pmesh->GetNE(); n++)
         {
            if (pmesh->GetAttribute(n) == refined_attribute)
            {
               el_to_refine.Append(n);
            }
         }
         pmesh->GeneralRefinement(el_to_refine);
      }
      pmesh->ExchangeFaceNbrData();

      CAPTURE(rank);
      CAPTURE(refined_attribute);
      CAPTURE(ref_level);
      CHECK(pmesh->GetNE() == (rank == refined_attribute - 1 ? std::pow(8,
                                                                        ref_level) : 1));
      CHECK(pmesh->GetNBE() == (rank == refined_attribute - 1
                                ? std::pow(4, ref_level + 1)
                                : (ref_level == 0 && rank == 0 ? 4 : 3)));

      // Refining on only one side of the boundary face should not change the
      // number of essential true dofs, which should match the number within the
      // original face.
      CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 1,
                                                  smesh.bdr_attributes.Max()) == 4);
      CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 2,
                                                  smesh.bdr_attributes.Max()) == 4 + 6);
      CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 3,
                                                  smesh.bdr_attributes.Max()) == 4 + 6*2 + 4*1);
      CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 4,
                                                  smesh.bdr_attributes.Max()) == 4 + 6*3 + 4*3);

      CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 1,
                                                  smesh.bdr_attributes.Max()) == (rank == 0 ? 6 : 0));
      CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 2,
                                                  smesh.bdr_attributes.Max()) == (rank == 0 ? 20 : 0));
      CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 3,
                                                  smesh.bdr_attributes.Max()) == (rank == 0 ? 42 : 0));
      CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 4,
                                                  smesh.bdr_attributes.Max()) == (rank == 0 ? 72 : 0));
   }
}

TEST_CASE("ParDividingPlaneMesh", "[Parallel], [NCMesh]")
{
   auto refine_attribute = [](Mesh& mesh, int attr, int ref_level)
   {
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < mesh.GetNE(); n++)
         {
            if (mesh.GetAttribute(n) == attr)
            {
               el_to_refine.Append(n);
            }
         }
         mesh.GeneralRefinement(el_to_refine);
      }
   };

   SECTION("Hex")
   {
      auto mesh = DividingPlaneMesh(false);
      mesh.EnsureNCMesh(true);

      CHECK(mesh.GetNBE() == 2 * 5 + 1);
      CHECK(mesh.GetNE() == 2);

      SECTION("H1Hex")
      {
         mesh.UniformRefinement();
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == 3*3);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == 5*5);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == 7*7);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 4,
                                                  mesh.bdr_attributes.Max()) == 9*9);

         auto attr = GENERATE(1,2);
         auto ref_level = GENERATE(1,2);
         refine_attribute(mesh, attr, ref_level);

         CHECK(CountEssentialDof<H1_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == 3*3);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == 5*5);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == 7*7);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 4,
                                                  mesh.bdr_attributes.Max()) == 9*9);
      }
   }

   SECTION("Tet")
   {
      auto mesh = DividingPlaneMesh(true, true);
      mesh.EnsureNCMesh(true);
      auto pmesh = CheckParMeshNBE(mesh);
      pmesh->FinalizeTopology();
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      CHECK(pmesh->bdr_attributes.Max() == mesh.bdr_attributes.Max());

      auto attr = GENERATE(1,2);
      auto ref_level = GENERATE(1,2);
      CAPTURE(attr);
      CAPTURE(ref_level);

      const int initial_num_vert = 4;
      const int initial_num_edge = 5;
      const int initial_num_face = 2;
      SECTION("H1Tet")
      {
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 1,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert);
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 2,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert + initial_num_edge);
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 3,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert + 2*initial_num_edge +
               initial_num_face);
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 4,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert + 3*initial_num_edge +
               3*initial_num_face);

         refine_attribute(*pmesh, attr, ref_level);

         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 1,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert);
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 2,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert + initial_num_edge);
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 3,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert + 2*initial_num_edge +
               initial_num_face);
         CHECK(ParCountEssentialDof<H1_FECollection>(*pmesh, 4,
                                                     mesh.bdr_attributes.Max()) == initial_num_vert + 3*initial_num_edge +
               3*initial_num_face);
      }

      SECTION("NDTet")
      {
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 1,
                                                     mesh.bdr_attributes.Max()) == 5);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 2,
                                                     mesh.bdr_attributes.Max()) == 14);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 3,
                                                     mesh.bdr_attributes.Max()) == 27);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 4,
                                                     mesh.bdr_attributes.Max()) == 44);

         refine_attribute(*pmesh, attr, ref_level);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 1,
                                                     mesh.bdr_attributes.Max()) == 5);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 2,
                                                     mesh.bdr_attributes.Max()) == 14);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 3,
                                                     mesh.bdr_attributes.Max()) == 27);
         CHECK(ParCountEssentialDof<ND_FECollection>(*pmesh, 4,
                                                     mesh.bdr_attributes.Max()) == 44);
      }
   }
}

TEST_CASE("ParTetFaceFlips", "[Parallel], [NCMesh]")
{
   /*
      1. Define an ND space, and project a smooth non polynomial function onto
         the space.
      2. Compute y-z components in the face, and check that they are equal when
         evaluated from either side of the face. Tangential continuity of the ND
         space should ensure they are identical, if orientations are correctly
         accounted for.
      3. Mark the mesh as NC, build a new FESpace, and repeat. There should be
         no change as the faces are "conformal" though they are within the NC
         structure.
      3. Partition the mesh, create a ParFESpace and repeat the above. There
         should be no difference in conformal parallel.
      4. Construct the ParMesh from the NCMesh and repeat. As above, there
         should be no change.
      5. Perform NC refinement on one side of the internal face, the number of
         conformal dof in the face will not change, so there should also be no
         difference. This will be complicated by ensuring the slave evaluations
         are at the same points.
   */

   auto orientation = GENERATE(1,3,5);
   auto smesh = OrientedTriFaceMesh(orientation);
   smesh.EnsureNodes();

   CHECK(smesh.GetNBE() == 1);

   // A smooth function in each vector component
   constexpr int order = 3, dim = 3, quadrature_order = 4;
   constexpr real_t kappa = 2 * M_PI;
   auto E_exact = [=](const Vector &x, Vector &E)
   {
      E(0) = cos(kappa * x(1));
      E(1) = cos(kappa * x(2));
      E(2) = cos(kappa * x(0));
   };
   VectorFunctionCoefficient E_coeff(dim, E_exact);

   // Helper for evaluating the ND grid function on either side of the first
   // conformal shared face. Specific to the pair of tet mesh described above,
   // but can be generalized.
   auto check_parallel_nc_conformal = [&](ParMesh &mesh)
   {
      ND_FECollection fe_collection(order, dim);
      ParFiniteElementSpace fe_space(&mesh, &fe_collection);
      ParGridFunction E(&fe_space);

      E.ProjectCoefficient(E_coeff);
      E.ExchangeFaceNbrData();

      auto *P = fe_space.GetProlongationMatrix();
      if (P != nullptr)
      {
         // Projection does not respect the non-conformal constraints. Extract
         // the true (conformal) and prolongate to get the NC respecting
         // projection.
         auto E_true = E.GetTrueVector();
         P->Mult(E_true, E);
         E.ExchangeFaceNbrData();
      }
      ParGridFunction * const coords = dynamic_cast<ParGridFunction*>
                                       (mesh.GetNodes());

      const auto &ir = IntRules.Get(Geometry::Type::TRIANGLE, quadrature_order);
      IntegrationRule left_eir(ir.GetNPoints()),
                      right_eir(ir.GetNPoints()); // element integration rules

      bool y_valid = true, z_valid = true;
      for (int n = 0; n < mesh.GetNBE(); n++)
      {
         auto f = mesh.GetBdrElementFaceIndex(n);

         auto finfo = mesh.GetFaceInformation(f);
         auto &face_element_transform = finfo.IsShared()
                                        ? *mesh.GetSharedFaceTransformationsByLocalIndex(f, true)
                                        : *mesh.GetFaceElementTransformations(f);

         face_element_transform.Loc1.Transform(ir, left_eir);
         face_element_transform.Loc2.Transform(ir, right_eir);

         constexpr real_t tol = 1e-14;
         REQUIRE(left_eir.GetNPoints() == ir.GetNPoints());
         REQUIRE(right_eir.GetNPoints() == ir.GetNPoints());
         Vector left_val, right_val;
         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            face_element_transform.Elem1->SetIntPoint(&left_eir[i]);
            coords->GetVectorValue(*face_element_transform.Elem1, left_eir[i], left_val);
            coords->GetVectorValue(*face_element_transform.Elem1, left_eir[i], right_val);
            REQUIRE(std::abs(left_val(0) - right_val(0)) < tol);
            REQUIRE(std::abs(left_val(1) - right_val(1)) < tol);
            REQUIRE(std::abs(left_val(2) - right_val(2)) < tol);
            E.GetVectorValue(*face_element_transform.Elem1, left_eir[i], left_val);

            face_element_transform.Elem2->SetIntPoint(&right_eir[i]);
            E.GetVectorValue(*face_element_transform.Elem2, right_eir[i], right_val);

            // Check that the second and third rows agree. The y and z should
            // agree as the normal is in the x direction.
            y_valid &= (std::abs(left_val(1) - right_val(1)) < tol);
            z_valid &= (std::abs(left_val(2) - right_val(2)) < tol);
         }
      }
      CHECK(y_valid);
      CHECK(z_valid);

      return fe_space.GlobalTrueVSize();
   };

   SECTION("Conformal")
   {
      auto partition_flag = GENERATE(false, true);
      CAPTURE(partition_flag);
      auto partition = std::unique_ptr<int[]>(new int[2]);
      if (Mpi::WorldSize() > 1)
      {
         partition[0] = partition_flag ? 0 : 1; partition[1] = partition_flag  ? 1 : 0;
      }
      else
      {
         partition[0] = 0; partition[1] = 0;
      }
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("ConformalSerialUniformRefined")
   {
      smesh.UniformRefinement();
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("ConformalParallelUniformRefined")
   {
      auto partition_flag = GENERATE(false, true);
      CAPTURE(partition_flag);
      auto partition = std::unique_ptr<int[]>(new int[2]);
      if (Mpi::WorldSize() > 1)
      {
         partition[0] = partition_flag ? 0 : 1; partition[1] = partition_flag  ? 1 : 0;
      }
      else
      {
         partition[0] = 0; partition[1] = 0;
      }
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->UniformRefinement();
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();
      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("Nonconformal")
   {
      auto partition_flag = GENERATE(false, true);
      CAPTURE(partition_flag);
      auto partition = std::unique_ptr<int[]>(new int[2]);
      if (Mpi::WorldSize() > 1)
      {
         partition[0] = partition_flag ? 0 : 1; partition[1] = partition_flag  ? 1 : 0;
      }
      else
      {
         partition[0] = 0; partition[1] = 0;
      }
      smesh.EnsureNCMesh(true);
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("NonconformalSerialUniformRefined")
   {
      smesh.UniformRefinement();
      smesh.EnsureNCMesh(true);
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("NonconformalSerialRefined")
   {
      smesh.EnsureNCMesh(true);
      int ref_level = GENERATE(1, 2);
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < smesh.GetNE(); n++)
         {
            if (smesh.GetAttribute(n) == 2)
            {
               el_to_refine.Append(n);
            }
         }
         smesh.GeneralRefinement(el_to_refine);
      }
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("NonconformalParallelUniformRefined")
   {
      auto partition_flag = GENERATE(false, true);
      CAPTURE(partition_flag);
      auto partition = std::unique_ptr<int[]>(new int[2]);
      if (Mpi::WorldSize() > 1)
      {
         partition[0] = partition_flag ? 0 : 1; partition[1] = partition_flag  ? 1 : 0;
      }
      else
      {
         partition[0] = 0; partition[1] = 0;
      }
      smesh.EnsureNCMesh(true);
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->UniformRefinement();
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("NonconformalParallelRefined")
   {
      auto partition_flag = GENERATE(false, true);
      CAPTURE(partition_flag);
      auto partition = std::unique_ptr<int[]>(new int[2]);
      if (Mpi::WorldSize() > 1)
      {
         partition[0] = partition_flag ? 0 : 1; partition[1] = partition_flag  ? 1 : 0;
      }
      else
      {
         partition[0] = 0; partition[1] = 0;
      }
      smesh.EnsureNCMesh(true);
      auto pmesh = CheckParMeshNBE(smesh);
      int ref_level = GENERATE(1, 2);
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < pmesh->GetNE(); n++)
         {
            if (pmesh->GetAttribute(n) == 2)
            {
               el_to_refine.Append(n);
            }
         }
         pmesh->GeneralRefinement(el_to_refine);
      }
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      check_parallel_nc_conformal(*pmesh);
   }

   SECTION("NonconformalLevelTwoRefined")
   {
      smesh.EnsureNCMesh(true);
      smesh.UniformRefinement();
      Array<int> el_to_refine(1);
      for (int n = 0; n < smesh.GetNE(); n++)
      {
         if (smesh.GetAttribute(n) == 2)
         {
            CAPTURE(n);
            Mesh smesh2(smesh);
            el_to_refine[0] = n;
            smesh2.GeneralRefinement(el_to_refine);
            for (int m = 0; m < smesh2.GetNE(); m++)
            {
               if (smesh2.GetAttribute(m) == 2)
               {
                  CAPTURE(m);
                  Mesh smesh3(smesh2);
                  el_to_refine[0] = m;
                  smesh3.GeneralRefinement(el_to_refine);
                  check_parallel_nc_conformal(*CheckParMeshNBE(smesh3));
               }
            }
         }
      }
   }

}

TEST_CASE("Parallel RP=I", "[Parallel], [NCMesh]")
{
   const int order = GENERATE(1, 2, 3);
   CAPTURE(order);
   const int dim = 3;

   SECTION("Hex")
   {
      auto smesh = DividingPlaneMesh(false, true);
      Array<int> refinements(1);
      refinements[0] = 0;
      smesh.GeneralRefinement(refinements);
      ParMesh mesh(MPI_COMM_WORLD, smesh);
      SECTION("ND")
      {
         ND_FECollection fec(order, dim);
         ParFiniteElementSpace fespace(&mesh, &fec);
         CHECK(CheckRPIdentity(fespace));
      }
      SECTION("H1")
      {
         H1_FECollection fec(order, dim);
         ParFiniteElementSpace fespace(&mesh, &fec);
         CHECK(CheckRPIdentity(fespace));
      }
   }

   SECTION("Tet")
   {
      auto orientation = GENERATE(1,3,5);
      Array<int> refinements(1);
      refinements[0] = 0;
      auto smesh = OrientedTriFaceMesh(orientation, true);
      smesh.EnsureNCMesh(true); // Always checking NC
      ParMesh mesh(MPI_COMM_WORLD, smesh);
      SECTION("ND")
      {
         ND_FECollection fec(order, dim);
         ParFiniteElementSpace fespace(&mesh, &fec);
         CHECK(CheckRPIdentity(fespace));
      }
      SECTION("H1")
      {
         H1_FECollection fec(order, dim);
         ParFiniteElementSpace fespace(&mesh, &fec);
         CHECK(CheckRPIdentity(fespace));
      }
   }
}

#endif // MFEM_USE_MPI

TEST_CASE("ReferenceCubeInternalBoundaries", "[NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = DividingPlaneMesh(false, true);
   smesh.EnsureNCMesh();
   CHECK(smesh.GetNBE() == 2 * 5 + 1);

   auto with_internal = CheckPoisson(smesh, p); // Include the internal boundary
   auto without_internal = CheckPoisson(smesh, p,
                                        smesh.bdr_attributes.Max()); // Exclude the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal); break;
      case 2:
         CHECK(with_internal == without_internal + 1); break;
      case 3:
         CHECK(with_internal == without_internal + 4); break;
   }

   auto ref_type = char(GENERATE(Refinement::XYZ));
   Array<Refinement> refs(1);
   for (auto ref : {0,1})
   {
      auto ssmesh = Mesh(smesh);

      CAPTURE(ref_type);

      // Now NC refine one of the attached elements, this should result in 2
      // internal boundary elements.
      refs[0].index = ref;
      refs[0].SetType(ref_type);

      ssmesh.GeneralRefinement(refs);

      // There should now be four internal boundary elements, where there was
      // one before.
      if (ref_type == 2 /* Y */ || ref_type == 4 /* Z */)
      {
         CHECK(ssmesh.GetNBE() == 5 /* external boundaries of unrefined element  */
               + 2 /* internal boundaries */
               + (2 * 4) /* external boundaries of refined elements */);
      }
      else if (ref_type == 6)
      {
         CHECK(ssmesh.GetNBE() == 5 /* external boundaries of unrefined element  */
               + 4 /* internal boundaries */
               + (4 * 3) /* external boundaries of refined elements */);
      }
      else if (ref_type == 7)
      {
         CHECK(ssmesh.GetNBE() == 5 /* external boundaries of unrefined element  */
               + 4 /* internal boundaries */
               + (4 * 3 + 4 * 2) /* external boundaries of refined elements */);
      }
      else
      {
         MFEM_ABORT("!");
      }

      // Count the number of internal boundary elements
      int num_internal = 0;
      for (int n = 0; n < ssmesh.GetNBE(); ++n)
      {
         int f, o;
         ssmesh.GetBdrElementFace(n, &f, &o);
         int e1, e2;
         ssmesh.GetFaceElements(f, &e1, &e2);
         if (e1 >= 0 && e2 >= 0 && ssmesh.GetAttribute(e1) != ssmesh.GetAttribute(e2))
         {
            ++num_internal;
         }
      }
      CHECK(num_internal == (ref_type <= 4 ? 2 : 4));

      ssmesh.FinalizeTopology();
      ssmesh.Finalize();

      without_internal = CheckPoisson(ssmesh, p,
                                      ssmesh.bdr_attributes.Max()); // Exclude the internal boundary
      with_internal = CheckPoisson(ssmesh, p); // Include the internal boundary

      // All slaves dofs that are introduced on the face are constrained by the
      // master dofs, thus the additional constraints on the internal boundary
      // are purely on the master face, which matches the initial unrefined
      // case.
      switch (p)
      {
         case 1:
            CHECK(with_internal == without_internal); break;
         case 2:
            CHECK(with_internal == without_internal + 1); break;
         case 3:
            CHECK(with_internal == without_internal + 4); break;
      }
   }
}

TEST_CASE("RefinedCubesInternalBoundaries", "[NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = DividingPlaneMesh(false, true);

   smesh.UniformRefinement();

   smesh.FinalizeTopology();
   smesh.Finalize();

   // Exactly four boundary elements must be added
   CHECK(smesh.GetNBE() == 2 * 5 * 4 + 4);

   smesh.EnsureNCMesh();
   CHECK(smesh.GetNBE() == 2 * 5 * 4 + 4);

   int without_internal = CheckPoisson(smesh, p,
                                       7); // Exclude the internal boundary
   int with_internal = CheckPoisson(smesh, p); // Include the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal + 1); break;
      case 2:
         CHECK(with_internal == without_internal + 3 * 3); break;
      case 3:
         CHECK(with_internal == without_internal + 5 * 5); break;
   }

   // Mark all elements on one side of the attribute boundary to refine
   Array<Refinement> refs;
   for (int n = 0; n < smesh.GetNE(); ++n)
   {
      if (smesh.GetAttribute(n) == 2)
      {
         refs.Append(Refinement{n});
      }
   }

   smesh.GeneralRefinement(refs);
   smesh.FinalizeTopology();
   smesh.Finalize();

   // There should now be 16 internal boundary elements, where there were 4
   // before.
   CHECK(smesh.GetNBE() == 5 * 4 /* external boundaries of unrefined domain  */
         + 4 * 4 /* internal boundaries */
         + 5 * 16 /* external boundaries of refined elements */);

   // Count the number of internal boundary elements
   int num_internal = 0;
   for (int n = 0; n < smesh.GetNBE(); ++n)
   {
      int f, o;
      smesh.GetBdrElementFace(n, &f, &o);
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         ++num_internal;
      }
   }
   CHECK(num_internal == 16);


   without_internal = CheckPoisson(smesh, p,
                                   smesh.bdr_attributes.Max()); // Exclude the internal boundary
   with_internal = CheckPoisson(smesh, p); // Include the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal + 1); break;
      case 2:
         CHECK(with_internal == without_internal + 3 * 3); break;
      case 3:
         CHECK(with_internal == without_internal + 5 * 5); break;
   }
}

TEST_CASE("ReferenceTetInternalBoundaries", "[NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");
   Array<Refinement> refs;
   refs.Append(Refinement(0, Refinement::X));
   smesh.GeneralRefinement(refs);

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(1, 2);

   REQUIRE(smesh.GetNE() == 2);
   REQUIRE(smesh.GetNBE() == 2 * 3);

   // Introduce an internal boundary element
   for (int f = 0; f < smesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = smesh.GetFace(f)->Duplicate(&smesh);
         new_elem->SetAttribute(5);
         smesh.AddBdrElement(new_elem);
      }
   }

   // Exactly one boundary element must be added
   CHECK(smesh.GetNBE() == 2 * 3 + 1);

   smesh.EnsureNCMesh(true);

   // Still exactly one boundary element must be added
   CHECK(smesh.GetNBE() == 2 * 3 + 1);

   smesh.FinalizeTopology();
   smesh.Finalize();

   auto without_internal = CheckPoisson(smesh, p,
                                        5); // Exclude the internal boundary
   auto with_internal = CheckPoisson(smesh, p); // Include the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal); break;
      case 2:
         CHECK(with_internal == without_internal); break;
      case 3:
         CHECK(with_internal == without_internal + 1); break;
   }

   // Now NC refine one of the attached elements, this should result in 2
   // internal boundary elements.
   for (int ref : {0, 1})
   {
      refs[0].Set(ref, Refinement::XYZ);
      auto ssmesh = Mesh(smesh);
      ssmesh.GeneralRefinement(refs);

      // There should now be four internal boundary elements, where there was
      // one before.
      CHECK(ssmesh.GetNBE() == 3 /* external boundaries of unrefined element  */
            + 4 /* internal boundaries */
            + (3 * 4) /* external boundaries of refined element */);

      // Count the number of internal boundary elements
      int num_internal = 0;
      for (int n = 0; n < ssmesh.GetNBE(); ++n)
      {
         int f, o;
         ssmesh.GetBdrElementFace(n, &f, &o);
         int e1, e2;
         ssmesh.GetFaceElements(f, &e1, &e2);
         if (e1 >= 0 && e2 >= 0 && ssmesh.GetAttribute(e1) != ssmesh.GetAttribute(e2))
         {
            ++num_internal;
         }
      }
      CHECK(num_internal == 4);

      without_internal = CheckPoisson(ssmesh, p, 5); // Exclude the internal boundary
      with_internal = CheckPoisson(ssmesh, p); // Include the internal boundary

      switch (p)
      {
         case 1:
            CHECK(with_internal == without_internal); break;
         case 2:
            CHECK(with_internal == without_internal); break;
         case 3:
            CHECK(with_internal == without_internal + 1); break;
      }
   }
}

TEST_CASE("RefinedTetsInternalBoundaries", "[NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");
   Array<Refinement> refs;
   refs.Append(Refinement(0, Refinement::X));
   smesh.GeneralRefinement(refs);

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(1, 2);

   REQUIRE(smesh.GetNE() == 2);
   REQUIRE(smesh.GetNBE() == 2 * 3);

   smesh.UniformRefinement();

   CHECK(smesh.GetNBE() == 2 * 3 * 4);

   // Introduce internal boundary elements
   for (int f = 0; f < smesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = smesh.GetFace(f)->Duplicate(&smesh);
         new_elem->SetAttribute(5);
         smesh.AddBdrElement(new_elem);
      }
   }

   // Exactly four boundary elements must be added
   CHECK(smesh.GetNBE() == 2 * 3 * 4 + 4);

   smesh.EnsureNCMesh(true);

   // Still exactly one boundary element must be added
   CHECK(smesh.GetNBE() == 2 * 3 * 4 + 4);

   smesh.FinalizeTopology();
   smesh.Finalize();

   auto without_internal = CheckPoisson(smesh, p,
                                        5); // Exclude the internal boundary
   auto with_internal = CheckPoisson(smesh, p); // Include the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal); break;
      case 2:
         CHECK(with_internal == without_internal + 3); break;
      case 3:
         CHECK(with_internal == without_internal + 10); break;
   }

   // Now NC refine all elements with the 2 attribute.

   // Mark all elements on one side of the attribute boundary to refine
   refs.DeleteAll();
   for (int n = 0; n < smesh.GetNE(); ++n)
   {
      if (smesh.GetAttribute(n) == 2)
      {
         refs.Append(Refinement{n});
      }
   }

   smesh.GeneralRefinement(refs);

   // There should now be four internal boundary elements, where there was one
   // before.
   CHECK(smesh.GetNBE() == 3 * 4 /* external boundaries of unrefined elements  */
         + 4 * 4 /* internal boundaries */
         + (3 * 4 * 4) /* external boundaries of refined elements */);

   // Count the number of internal boundary elements
   int num_internal = 0;
   for (int n = 0; n < smesh.GetNBE(); ++n)
   {
      int f, o;
      smesh.GetBdrElementFace(n, &f, &o);
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         ++num_internal;
      }
   }
   CHECK(num_internal == 4 * 4);

   without_internal = CheckPoisson(smesh, p, 5); // Exclude the internal boundary
   with_internal = CheckPoisson(smesh, p); // Include the internal boundary

   switch (p)
   {
      case 1:
         CHECK(with_internal == without_internal); break;
      case 2:
         CHECK(with_internal == without_internal + 3); break;
      case 3:
         CHECK(with_internal == without_internal + 10); break;
   }
}

TEST_CASE("PoissonOnReferenceCubeNC", "[NCMesh]")
{
   auto smesh = DividingPlaneMesh(false, true);
   auto p = GENERATE(1, 2, 3);
   CAPTURE(p);

   // Check that Poisson can be solved on the domain
   CheckPoisson(smesh, p);
   auto ref_type = char(GENERATE(Refinement::X, Refinement::Y, Refinement::Z,
                                 Refinement::XY, Refinement::XZ, Refinement::YZ,
                                 Refinement::XYZ));
   CAPTURE(ref_type);

   const real_t scale = GENERATE(0.5, 0.25);
   CAPTURE(scale);

   Array<Refinement> refs(1);
   for (auto refined_elem : {0}) // The left or the right element
   {
      auto ssmesh = Mesh(smesh);
      refs[0].Set(refined_elem, ref_type, scale);

      ssmesh.GeneralRefinement(refs);
      ssmesh.FinalizeTopology();
      ssmesh.Finalize();

      CAPTURE(refined_elem);
      CheckPoisson(ssmesh, p);
   }
}

TEST_CASE("PoissonOnReferenceTetNC", "[NCMesh]")
{
   auto p = GENERATE(1, 2, 3);
   CAPTURE(p);
   auto orientation = GENERATE(1,3,5);
   auto smesh = OrientedTriFaceMesh(orientation, true);
   smesh.EnsureNCMesh(true);

   CheckPoisson(smesh, p);
   Array<int> refs(1);
   for (auto refined_elem : {0, 1})
   {
      auto ssmesh = Mesh(smesh);
      refs[0] = refined_elem;
      ssmesh.GeneralRefinement(refs);
      ssmesh.FinalizeTopology();
      ssmesh.Finalize();

      CAPTURE(refined_elem);
      CheckPoisson(ssmesh, p);
   }
}

TEST_CASE("TetBoundaryRefinement", "[NCMesh]")
{
   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");

   smesh.FinalizeTopology();
   smesh.Finalize(true);
   smesh.UniformRefinement();

   smesh.EnsureNCMesh(true);

   CHECK(smesh.GetNBE() == 4 * 4);

   // Loop over elements and mark for refinement if any vertices match the
   // original
   auto refine_corners = [&]()
   {
      Array<int> vertices, elements;
      // reference vertices of (0,0,0), (1,0,0), (0,1,0), (0,0,1) are [0,3]
      auto original_vert = [](int i) { return i >= 0 && i <= 3; };
      for (int n = 0; n < smesh.GetNE(); ++n)
      {
         smesh.GetElementVertices(n, vertices);
         if (std::any_of(vertices.begin(), vertices.end(), original_vert))
         {
            elements.Append(n);
         }
      }

      smesh.GeneralRefinement(elements);
      smesh.FinalizeTopology();
      smesh.Finalize();
   };

   constexpr int max_ref_levels = 4;
   for (int r = 0; r < max_ref_levels; r++)
   {
      refine_corners();
      CHECK(smesh.GetNBE() == 4 * (4 + 3 * 3 * (r + 1)));
   }
}

TEST_CASE("TetInternalBoundaryRefinement", "[NCMesh]")
{
   auto orientation = GENERATE(1,3,5);
   auto smesh = OrientedTriFaceMesh(orientation, true);

   smesh.FinalizeTopology();
   smesh.Finalize(true);
   smesh.UniformRefinement();
   smesh.EnsureNCMesh(true);
   CHECK(smesh.GetNBE() == (2*3 + 1) * 4);

   CHECK(CountEssentialDof<H1_FECollection>(smesh, 1,
                                            smesh.bdr_attributes.Max()) == 6);
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 2,
                                            smesh.bdr_attributes.Max()) == 6 + 3 * 3);
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 3,
                                            smesh.bdr_attributes.Max()) == 10 + 3 * 6);

   int refined_attribute = GENERATE(1,2);
   int ref_level = GENERATE(1,2,3);
   for (int r = 0; r < ref_level; r++)
   {
      Array<int> el_to_refine;
      for (int n = 0; n < smesh.GetNE(); n++)
      {
         if (smesh.GetAttribute(n) == refined_attribute)
         {
            el_to_refine.Append(n);
         }
      }
      smesh.GeneralRefinement(el_to_refine);
   }

   // Refining on only one side of the boundary face should not change the
   // number of essential true dofs
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 1,
                                            smesh.bdr_attributes.Max()) == 6);
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 2,
                                            smesh.bdr_attributes.Max()) == 6 + 3 * 3);
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 3,
                                            smesh.bdr_attributes.Max()) == 10 + 3 * 6);

   // The number of boundary faces should have increased.
   CHECK(smesh.GetNBE() == 3 * 4 + (3 + 1) * std::pow(4, 1+ref_level));
}

TEST_CASE("TetInternalBoundaryTetStarMesh", "[NCMesh]")
{
   auto smesh = TetStarMesh();
   smesh.EnsureNCMesh(true);

   SECTION("Unrefined")
   {
      CHECK(smesh.GetNBE() == 4 * 3 + 4);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 1,
                                               smesh.bdr_attributes.Max()) == 4);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 2,
                                               smesh.bdr_attributes.Max()) == 4 + 6);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 3,
                                               smesh.bdr_attributes.Max()) == 4 + 6*2 + 4*1);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 4,
                                               smesh.bdr_attributes.Max()) == 4 + 6*3 + 4*3);

      CHECK(CountEssentialDof<ND_FECollection>(smesh, 1,
                                               smesh.bdr_attributes.Max()) == 6);
      CHECK(CountEssentialDof<ND_FECollection>(smesh, 2,
                                               smesh.bdr_attributes.Max()) == 20);
      CHECK(CountEssentialDof<ND_FECollection>(smesh, 3,
                                               smesh.bdr_attributes.Max()) == 42);
      CHECK(CountEssentialDof<ND_FECollection>(smesh, 4,
                                               smesh.bdr_attributes.Max()) == 72);
   }

   SECTION("Refined")
   {
      int refined_attribute = GENERATE(1,2,3,4,5);
      int ref_level = GENERATE(1,2,3);
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < smesh.GetNE(); n++)
         {
            if (smesh.GetAttribute(n) == refined_attribute)
            {
               el_to_refine.Append(n);
            }
         }
         smesh.GeneralRefinement(el_to_refine);
      }

      // Refining on only one side of the boundary face should not change the
      // number of essential true dofs
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 1,
                                               smesh.bdr_attributes.Max()) == 4);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 2,
                                               smesh.bdr_attributes.Max()) == 4 + 6);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 3,
                                               smesh.bdr_attributes.Max()) == 4 + 6*2 + 4*1);
      CHECK(CountEssentialDof<H1_FECollection>(smesh, 4,
                                               smesh.bdr_attributes.Max()) == 4 + 6*3 + 4*3);

      CHECK(CountEssentialDof<ND_FECollection>(smesh, 1,
                                               smesh.bdr_attributes.Max()) == 6);
      CHECK(CountEssentialDof<ND_FECollection>(smesh, 2,
                                               smesh.bdr_attributes.Max()) == 6 * 2 + 4 * 2); // 2 per edge, 2 per face
      CHECK(CountEssentialDof<ND_FECollection>(smesh, 3,
                                               smesh.bdr_attributes.Max()) == 42);
      CHECK(CountEssentialDof<ND_FECollection>(smesh, 4,
                                               smesh.bdr_attributes.Max()) == 72);

      // The number of boundary faces should have increased.
      CHECK(smesh.GetNBE() == 3 * 4 + 4 * std::pow(4,ref_level));
   }
}

TEST_CASE("DividingPlaneMesh", "[NCMesh]")
{
   auto refine_attribute = [](Mesh& mesh, int attr, int ref_level)
   {
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < mesh.GetNE(); n++)
         {
            if (mesh.GetAttribute(n) == attr)
            {
               el_to_refine.Append(n);
            }
         }
         mesh.GeneralRefinement(el_to_refine);
      }
   };

   SECTION("Hex")
   {
      auto mesh = DividingPlaneMesh(false);
      mesh.EnsureNCMesh(true);

      CHECK(mesh.GetNBE() == 2 * 5 + 1);
      CHECK(mesh.GetNE() == 2);

      auto attr = GENERATE(1,2);
      auto ref_level = GENERATE(1,2);

      const int num_vert = ref_level == 1 ? 5*5 : 9*9;
      const int num_edge = ref_level == 1 ? 2*4*5 : 2*8*9;
      const int num_face = ref_level == 1 ? 4*4 : 8*8;

      SECTION("H1Hex")
      {
         mesh.UniformRefinement();
         CHECK(CountEssentialDof<H1_FECollection, true>(mesh, 1,
                                                        mesh.bdr_attributes.Max()) == 3*3);
         CHECK(CountEssentialDof<H1_FECollection, true>(mesh, 2,
                                                        mesh.bdr_attributes.Max()) == 5*5);
         CHECK(CountEssentialDof<H1_FECollection, true>(mesh, 3,
                                                        mesh.bdr_attributes.Max()) == 7*7);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == 3*3);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == 5*5);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == 7*7);

         refine_attribute(mesh, attr, ref_level);

         CHECK(CountEssentialDof<H1_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == 3*3);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == 5*5);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == 7*7);

         // Add the slave face dofs, then subtract off the vertex dofs which are
         // double counted due to being shared.
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == 3*3 + num_vert - 3*3);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == 5*5 + num_vert + num_edge + num_face - 3*3);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == 7*7 + num_vert + 2*num_edge + 4*num_face - 3*3);

      }

      SECTION("NDHex")
      {
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 1,
                                                        mesh.bdr_attributes.Max()) == 4);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 2,
                                                        mesh.bdr_attributes.Max()) == 4*2 + 2*2);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 3,
                                                        mesh.bdr_attributes.Max()) == 4*3 + 2*2*3);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == 4);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == 4*2 + 2*2);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == 4*3 + 2*2*3);

         mesh.UniformRefinement();
         const int initial_num_edge = 12;
         const int initial_num_face = 4;
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 1,
                                                        mesh.bdr_attributes.Max()) == initial_num_edge);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 2,
                                                        mesh.bdr_attributes.Max()) == initial_num_edge*2 + initial_num_face*2*2);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 3,
                                                        mesh.bdr_attributes.Max()) == initial_num_edge*3 + initial_num_face*2*2*3);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == initial_num_edge);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == initial_num_edge*2 + initial_num_face*2*2);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == initial_num_edge*3 + initial_num_face*2*2*3);

         refine_attribute(mesh, attr, ref_level);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 1,
                                                        mesh.bdr_attributes.Max()) == initial_num_edge);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 2,
                                                        mesh.bdr_attributes.Max()) == initial_num_edge*2 + initial_num_face*2*2);
         CHECK(CountEssentialDof<ND_FECollection, true>(mesh, 3,
                                                        mesh.bdr_attributes.Max()) == initial_num_edge*3 + initial_num_face*2*2*3);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == (num_edge+initial_num_edge));
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == (num_edge+initial_num_edge)*2 +
               (num_face+initial_num_face)*2*2);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == (num_edge+initial_num_edge)*3 +
               (num_face+initial_num_face)*2*2*3);
      }
   }

   SECTION("Tet")
   {
      auto mesh = DividingPlaneMesh(true);
      mesh.EnsureNCMesh(true);

      CHECK(mesh.GetNBE() == 2 * 5 * 2 + 2);
      CHECK(mesh.GetNE() == 2 * 6);

      auto attr = GENERATE(1,2);
      auto ref_level = GENERATE(1,2);
      CAPTURE(attr);
      CAPTURE(ref_level);

      const int initial_num_vert = 4;
      const int initial_num_edge = 5;
      const int initial_num_face = 2;

      const int num_vert = ref_level == 1 ? 9 : 25;
      const int num_edge = ref_level == 1 ? 16 : 56;
      const int num_face = ref_level == 1 ? 8 : 32;

      SECTION("H1Tet")
      {
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert + initial_num_edge);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert + 2*initial_num_edge +
               initial_num_face);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 4,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert + 3*initial_num_edge +
               3*initial_num_face);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == initial_num_vert);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == initial_num_vert + initial_num_edge);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == initial_num_vert + 2*initial_num_edge +
               initial_num_face);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 4,
                                                         mesh.bdr_attributes.Max()) == initial_num_vert + 3*initial_num_edge +
               3*initial_num_face);

         refine_attribute(mesh, attr, ref_level);

         CHECK(CountEssentialDof<H1_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert + initial_num_edge);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert + 2*initial_num_edge +
               initial_num_face);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 4,
                                                  mesh.bdr_attributes.Max()) == initial_num_vert + 3*initial_num_edge +
               3*initial_num_face);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == num_vert);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == num_vert + num_edge + initial_num_edge);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == num_vert + 2*num_edge + num_face +
               2*initial_num_edge + initial_num_face);
         CHECK(CountEssentialDof<H1_FECollection, false>(mesh, 4,
                                                         mesh.bdr_attributes.Max()) == num_vert + 3*num_edge + 3*num_face +
               3*initial_num_edge + 3*initial_num_face);
      }

      SECTION("NDTet")
      {
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == 5);
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == 14);
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == 27);
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 4,
                                                  mesh.bdr_attributes.Max()) == 44);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == 5);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == 14);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == 27);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 4,
                                                         mesh.bdr_attributes.Max()) == 44);

         refine_attribute(mesh, attr, ref_level);

         CHECK(CountEssentialDof<ND_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == 5);
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == 14);
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == 27);
         CHECK(CountEssentialDof<ND_FECollection>(mesh, 4,
                                                  mesh.bdr_attributes.Max()) == 44);

         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 1,
                                                         mesh.bdr_attributes.Max()) == 5 + num_edge);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 2,
                                                         mesh.bdr_attributes.Max()) == 14 + 2 * num_edge + 2*num_face);
         CHECK(CountEssentialDof<ND_FECollection, false>(mesh, 3,
                                                         mesh.bdr_attributes.Max()) == 27 + 3 * num_edge + 2*3*num_face);
      }
   }
}

TEST_CASE("TetFaceFlips", "[NCMesh]")
{
   auto orientation = GENERATE(1,3,5);
   CAPTURE(orientation);
   auto smesh = OrientedTriFaceMesh(orientation, true);

   // A smooth function in each vector component
   constexpr int order = 3, dim = 3, quadrature_order = 4;
   constexpr real_t kappa = 2 * M_PI;
   auto E_exact = [=](const Vector &x, Vector &E)
   {
      E(0) = cos(kappa * x(1));
      E(1) = cos(kappa * x(2));
      E(2) = cos(kappa * x(0));
   };
   VectorFunctionCoefficient E_coeff(dim, E_exact);

   auto check_serial_nd_conformal = [&](Mesh &mesh, int num_essential_tdof,
                                        int num_essential_vdof)
   {
      ND_FECollection fe_collection(order, dim);
      FiniteElementSpace fe_space(&mesh, &fe_collection);
      GridFunction E(&fe_space);

      E.ProjectCoefficient(E_coeff);

      auto *P = fe_space.GetProlongationMatrix();
      if (P != nullptr)
      {
         // Projection does not respect the non-conformal constraints. Extract
         // the true (conformal) and prolongate to get the NC respecting
         // projection.
         auto E_true = E.GetTrueVector();
         P->Mult(E_true, E);
      }
      mesh.EnsureNodes();
      GridFunction * const coords = mesh.GetNodes();

      const auto &ir = IntRules.Get(Geometry::Type::TRIANGLE, quadrature_order);
      IntegrationRule left_eir(ir.GetNPoints()),
                      right_eir(ir.GetNPoints()); // element integration rules

      Array<int> bdr_attr_is_ess = mesh.bdr_attributes, tdof_list;
      bdr_attr_is_ess = 0;
      bdr_attr_is_ess.Last() = 1;
      fe_space.GetEssentialTrueDofs(bdr_attr_is_ess, tdof_list);

      Array<int> ess_vdof_marker, vdof_list;
      fe_space.GetEssentialVDofs(bdr_attr_is_ess, ess_vdof_marker);
      fe_space.MarkerToList(ess_vdof_marker, vdof_list);

      CHECK(num_essential_tdof == tdof_list.Size());
      if (num_essential_vdof != -1)
      {
         CHECK(num_essential_vdof == vdof_list.Size());
      }

      bool y_valid = true, z_valid = true;
      for (int n = 0; n < mesh.GetNBE(); n++)
      {
         // NOTE: only works for internal boundaries
         if (bdr_attr_is_ess[mesh.GetBdrAttribute(n) - 1])
         {
            auto f = mesh.GetBdrElementFaceIndex(n);
            auto &face_element_transform = *mesh.GetFaceElementTransformations(f);

            if (face_element_transform.Elem2 == nullptr)
            {
               // not internal, nothing to check.
               continue;
            }

            face_element_transform.Loc1.Transform(ir, left_eir);
            face_element_transform.Loc2.Transform(ir, right_eir);

            constexpr real_t tol = 1e-14;
            REQUIRE(left_eir.GetNPoints() == ir.GetNPoints());
            REQUIRE(right_eir.GetNPoints() == ir.GetNPoints());
            Vector left_val, right_val;
            for (int i = 0; i < ir.GetNPoints(); i++)
            {
               face_element_transform.Elem1->SetIntPoint(&left_eir[i]);
               coords->GetVectorValue(*face_element_transform.Elem1, left_eir[i], left_val);
               coords->GetVectorValue(*face_element_transform.Elem1, left_eir[i], right_val);
               REQUIRE(std::abs(left_val(0) - right_val(0)) < tol);
               REQUIRE(std::abs(left_val(1) - right_val(1)) < tol);
               REQUIRE(std::abs(left_val(2) - right_val(2)) < tol);
               E.GetVectorValue(*face_element_transform.Elem1, left_eir[i], left_val);

               face_element_transform.Elem2->SetIntPoint(&right_eir[i]);
               E.GetVectorValue(*face_element_transform.Elem2, right_eir[i], right_val);

               // Check that the second and third rows agree. The y and z should
               // agree as the normal is in the x direction
               y_valid &= (std::abs(left_val(1) - right_val(1)) < tol);
               z_valid &= (std::abs(left_val(2) - right_val(2)) < tol);
            }
         }
      }
      CHECK(y_valid);
      CHECK(z_valid);
   };

   SECTION("Conformal")
   {
      const int ntdof = 3*3 + 3*2;
      const int nvdof = ntdof;
      check_serial_nd_conformal(smesh, ntdof, nvdof);
   }

   SECTION("Nonconformal")
   {
      smesh.EnsureNCMesh(true);
      const int ntdof = 3*3 + 3*2;
      const int nvdof = ntdof;
      check_serial_nd_conformal(smesh, ntdof, nvdof);
   }

   SECTION("ConformalUniformRefined")
   {
      smesh.UniformRefinement();
      const int ntdof = 9*3 + 4*3*2;
      const int nvdof = ntdof;
      check_serial_nd_conformal(smesh, ntdof, nvdof);
   }

   SECTION("NonconformalUniformRefined")
   {
      smesh.EnsureNCMesh(true);
      smesh.UniformRefinement();
      const int ntdof = 9*3 + 4*3*2;
      const int nvdof = ntdof;
      check_serial_nd_conformal(smesh, ntdof, nvdof);
   }

   SECTION("NonconformalRefined")
   {
      smesh.EnsureNCMesh(true);
      int ref_level = GENERATE(1, 2);
      CAPTURE(ref_level);
      for (int r = 0; r < ref_level; r++)
      {
         Array<int> el_to_refine;
         for (int n = 0; n < smesh.GetNE(); n++)
         {
            if (smesh.GetAttribute(n) == 2)
            {
               el_to_refine.Append(n);
            }
         }
         smesh.GeneralRefinement(el_to_refine);
      }
      const int ntdof = 3*3 + 3*2;
      const int nvdof = ntdof + (ref_level == 1 ? 9*3 + 4*3*2 : 30*3 + 16*3*2);
      check_serial_nd_conformal(smesh, ntdof, nvdof);
   }

   SECTION("NonconformalLevelTwoRefined")
   {
      smesh.EnsureNCMesh(true);
      Array<int> el_to_refine;

      smesh.UniformRefinement();

      const int ntdof = 9*3 + 4*3*2;
      el_to_refine.SetSize(1);

      auto n = GENERATE(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
      auto m = GENERATE(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22);

      if (n < smesh.GetNE() && smesh.GetAttribute(n) == 2)
      {
         el_to_refine[0] = n;
         CAPTURE(n);
         smesh.GeneralRefinement(el_to_refine);
         check_serial_nd_conformal(smesh, ntdof, -1);

         if (smesh.GetAttribute(m) == 2)
         {
            el_to_refine[0] = m;
            CAPTURE(m);
            smesh.GeneralRefinement(el_to_refine);
            check_serial_nd_conformal(smesh, ntdof, -1);
         }
      }

   }
}

TEST_CASE("RP=I", "[NCMesh]")
{
   auto check_fespace = [](const FiniteElementSpace& fespace)
   {
      auto * const R = fespace.GetConformingRestriction();
      auto * const P = fespace.GetConformingProlongation();

      REQUIRE(R != nullptr);
      REQUIRE(P != nullptr);

      // Vector notation
      Vector e_i(R->Height()), e_j(P->Width());
      Vector Rrow(R->Width()), Pcol(P->Height());
      bool valid = true;
      for (int i = 0; i < R->Height(); i++)
      {
         e_i = 0.0;
         e_i(i) = 1.0;
         R->MultTranspose(e_i, Rrow);
         for (int j = 0; j < P->Width(); j++)
         {
            e_j = 0.0;
            e_j(j) = 1.0;
            P->Mult(e_j, Pcol);

            valid &= (Rrow * Pcol == (i == j ? 1.0 : 0.0));
         }
      }
      CHECK(valid);

      // Index notation
      CHECK(R->Height() == P->Width());
      CHECK(R->Width() == P->Height());
      valid = true;
      for (int i = 0; i < R->Height(); i++)
         for (int j = 0; j < P->Width(); j++)
         {
            real_t dot = 0.0;
            for (int k = 0; k < R->Width(); k++)
            {
               dot += (*R)(i,k)*(*P)(k,j);
            }
            CHECK(dot == (i == j ? 1.0 : 0.0));
            valid &= (dot == (i == j ? 1.0 : 0.0));
         }
      CHECK(valid);
   };

   SECTION("Hex")
   {
      const int dim = 3;
      const int order = GENERATE(1, 2);
      // Split the hex into a pair, then isotropically refine one of them.
      Mesh mesh("../../data/ref-cube.mesh");
      Array<Refinement> refinements(1);
      refinements[0].Set(0, Refinement::X);

      mesh.GeneralRefinement(refinements);
      refinements[0].SetType(Refinement::XYZ);

      mesh.GeneralRefinement(refinements);
      SECTION("ND")
      {
         ND_FECollection fec(order, dim);
         FiniteElementSpace fespace(&mesh, &fec);
         check_fespace(fespace);
      }
      SECTION("H1")
      {
         H1_FECollection fec(order, dim);
         FiniteElementSpace fespace(&mesh, &fec);
         check_fespace(fespace);
      }
   }

   SECTION("Tet")
   {
      const int dim = 3;
      const int order = GENERATE(1, 2);
      // Split the hex into a pair, then isotropically refine one of them.
      Mesh mesh("../../data/ref-tetrahedron.mesh");
      Array<Refinement> refinements(1);
      refinements[0].Set(0, Refinement::X);
      mesh.GeneralRefinement(refinements);
      mesh.EnsureNCMesh(true);
      refinements[0].SetType(Refinement::XYZ);
      mesh.GeneralRefinement(refinements);
      SECTION("ND")
      {
         ND_FECollection fec(order, dim);
         FiniteElementSpace fespace(&mesh, &fec);
         check_fespace(fespace);
      }
      SECTION("H1")
      {
         H1_FECollection fec(order, dim);
         FiniteElementSpace fespace(&mesh, &fec);
         check_fespace(fespace);
      }
   }
}

TEST_CASE("InternalBoundaryProjectBdrCoefficient", "[NCMesh]")
{
   auto test_project_H1 = [](Mesh &mesh, int order, double coef)
   {
      MFEM_ASSERT(std::abs(coef) > 0,
                  "Non zero coef value required for meaningful test.");
      H1_FECollection fe_collection(order, mesh.SpaceDimension());
      FiniteElementSpace fe_space(&mesh, &fe_collection);
      GridFunction x(&fe_space);
      x = -coef;
      ConstantCoefficient c(coef);

      // Check projecting on the internal face sets essential dof.
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr.Last() = 1; // internal boundary
      x.ProjectBdrCoefficient(c, ess_bdr);

      Array<int> ess_vdofs_list, ess_vdofs_marker;
      fe_space.GetEssentialVDofs(ess_bdr, ess_vdofs_marker);
      fe_space.MarkerToList(ess_vdofs_marker, ess_vdofs_list);
      for (auto ess_dof : ess_vdofs_list)
      {
         CHECK(x[ess_dof] == Approx(coef).epsilon(1e-8));
      }

      int iess = 0;
      for (int i = 0; i < x.Size(); i++)
      {
         if (iess < ess_vdofs_list.Size() && i == ess_vdofs_list[iess])
         {
            iess++;
            continue;
         }
         CHECK(x[i] == Approx(-coef).epsilon(1e-8));
      }

   };

   auto OneSidedNCRefine = [](Mesh &mesh)
   {
      // Pick one element attached to the new boundary attribute and refine.
      const auto interface_attr = mesh.bdr_attributes.Max();
      Array<int> el_to_ref;
      for (int nbe = 0; nbe < mesh.GetNBE(); nbe++)
      {
         if (mesh.GetBdrAttribute(nbe) == interface_attr)
         {
            int f, o, e1, e2;
            mesh.GetBdrElementFace(nbe, &f, &o);
            mesh.GetFaceElements(f, &e1, &e2);
            el_to_ref.Append(e1);
         }
      }
      mesh.GeneralRefinement(el_to_ref);
      return;
   };

   SECTION("Hex")
   {
      auto smesh = DividingPlaneMesh(false, true);
      smesh.EnsureNCMesh(true);
      OneSidedNCRefine(smesh);
      test_project_H1(smesh, 2, 0.25);
   }

   SECTION("Tet")
   {
      auto smesh = DividingPlaneMesh(true, true);
      smesh.EnsureNCMesh(true);
      OneSidedNCRefine(smesh);
      test_project_H1(smesh, 3, 0.25);
   }
}

} // namespace mfem
