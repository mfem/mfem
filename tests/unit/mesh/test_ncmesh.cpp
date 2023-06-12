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

namespace mfem
{

constexpr double EPS = 1e-10;

// Test case: Verify that a conforming mesh yields the same norm for the
//            assembled diagonal with PA when using the standard (conforming)
//            Mesh vs. the corresponding (non-conforming) NCMesh.
//            (note: permutations of the values in the diagonal are expected)
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

         double error = fabs(diag.Norml2() - nc_diag.Norml2());
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

         double error = fabs(diag.Sum() - nc_diag.Sum());
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

   Mesh mesh(mesh_fname, 1, 1);
   mesh.EnsureNCMesh(true);
   double original_volume = mesh.GetElementVolume(0);
   Array<Refinement> ref(1);
   ref[0].ref_type = ref_type; ref[0].index = 0;

   mesh.GeneralRefinement(ref, 1);
   double summed_volume = 0.0;
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
   double original_volume = mesh.GetElementVolume(0);
   Array<Refinement> ref(1);
   ref[0].ref_type = ref_type; ref[0].index = 0;

   mesh.GeneralRefinement(ref, 1);

   Array<double> elem_error(mesh.GetNE());
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      elem_error[i] = 0.0;
   }
   mesh.DerefineByError(elem_error, 1.0);

   double derefined_volume = mesh.GetElementVolume(0);
   REQUIRE(derefined_volume == MFEM_Approx(original_volume));
} // test case


#ifdef MFEM_USE_MPI

// Test case: Verify that a conforming mesh yields the same norm for the
//            assembled diagonal with PA when using the standard (conforming)
//            Mesh vs. the corresponding (non-conforming) NCMesh.
//            (note: permutations of the values in the diagonal are expected)
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

         double diag_lsum = diag.Sum(), nc_diag_lsum = nc_diag.Sum();
         double diag_gsum = 0.0, nc_diag_gsum = 0.0;
         MPI_Allreduce(&diag_lsum, &diag_gsum, 1, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD);
         MPI_Allreduce(&nc_diag_lsum, &nc_diag_gsum, 1, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD);
         double error = fabs(diag_gsum - nc_diag_gsum);
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

         double diag_lsum = diag.Sum(), nc_diag_lsum = nc_diag.Sum();
         double diag_gsum = 0.0, nc_diag_gsum = 0.0;
         MPI_Allreduce(&diag_lsum, &diag_gsum, 1, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD);
         MPI_Allreduce(&nc_diag_lsum, &nc_diag_gsum, 1, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD);
         double error = fabs(diag_gsum - nc_diag_gsum);
         CAPTURE(order, error);
         REQUIRE(error == MFEM_Approx(0.0, EPS));
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }

} // test case

TEST_CASE("FaceEdgeConstraint",  "[Parallel], [NCMesh]")
{
   constexpr int refining_rank = 0;
   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");

   REQUIRE(smesh.GetNE() == 1);
   {
      // Start the test with two tetrahedra attached by triangle.
      auto single_edge_refine = Array<Refinement>(1);
      single_edge_refine[0].index = 0;
      single_edge_refine[0].ref_type = Refinement::X;

      smesh.GeneralRefinement(single_edge_refine, 0); // conformal
   }

   auto exact_soln = [](const Vector& x)
   {
      // sin(|| x - d ||^2) -> non polynomial but very smooth.
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      d -= x;
      return std::sin(d * d);
   };

   // Given a parallel and a serial mesh, perform an L2 projection and check the
   // solutions match exactly.
   auto check_l2_projection = [&exact_soln](ParMesh& pmesh, Mesh& smesh, int order)
   {

      REQUIRE(pmesh.GetGlobalNE() == smesh.GetNE());
      REQUIRE(pmesh.Dimension() == smesh.Dimension());
      REQUIRE(pmesh.SpaceDimension() == smesh.SpaceDimension());

      // Make an H1 space, then a mass matrix operator and invert it.
      // If all non-conformal constraints have been conveyed correctly, the
      // resulting DOF should match exactly on the serial and the parallel
      // solution.

      H1_FECollection fec(order, smesh.Dimension());
      ConstantCoefficient one(1.0);
      FunctionCoefficient rhs_coef(exact_soln);

      constexpr double linear_tol = 1e-16;

      // serial solve
      auto serror = [&]
      {
         FiniteElementSpace fes(&smesh, &fec);
         // solution vectors
         GridFunction x(&fes);
         x = 0.0;

         LinearForm b(&fes);
         b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
         b.Assemble();

         BilinearForm a(&fes);
         a.AddDomainIntegrator(new MassIntegrator(one));
         a.Assemble();

         SparseMatrix A;
         Vector B, X;

         Array<int> empty_tdof_list;
         a.FormLinearSystem(empty_tdof_list, x, b, A, X, B);

#ifndef MFEM_USE_SUITESPARSE
         // 9. Define a simple symmetric Gauss-Seidel preconditioner and use it to
         //    solve the system AX=B with PCG.
         GSSmoother M(A);
         PCG(A, M, B, X, -1, 500, linear_tol, 0.0);
#else
         // 9. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(A);
         umf_solver.Mult(B, X);
#endif

         a.RecoverFEMSolution(X, b, x);
         return x.ComputeL2Error(rhs_coef);
      }();

      auto perror = [&]
      {
         // parallel solve
         ParFiniteElementSpace fes(&pmesh, &fec);
         ParLinearForm b(&fes);

         ParGridFunction x(&fes);
         x = 0.0;

         b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
         b.Assemble();

         ParBilinearForm a(&fes);
         a.AddDomainIntegrator(new MassIntegrator(one));
         a.Assemble();

         HypreParMatrix A;
         Vector B, X;
         Array<int> empty_tdof_list;
         a.FormLinearSystem(empty_tdof_list, x, b, A, X, B);

         HypreBoomerAMG amg(A);
         HyprePCG pcg(A);
         amg.SetPrintLevel(-1);
         pcg.SetTol(linear_tol);
         pcg.SetMaxIter(500);
         pcg.SetPrintLevel(-1);
         pcg.SetPreconditioner(amg);
         pcg.Mult(B, X);
         a.RecoverFEMSolution(X, b, x);
         return x.ComputeL2Error(rhs_coef);
      }();

      constexpr double test_tol = 1e-9;
      CHECK(std::abs(serror - perror) < test_tol);

   };

   REQUIRE(smesh.GetNE() == 2);
   smesh.EnsureNCMesh(true);
   smesh.Finalize();

   auto partition = std::unique_ptr<int[]>(new int[smesh.GetNE()]);
   partition[0] = 0;
   partition[1] = Mpi::WorldSize() > 1 ? 1 : 0;

   auto pmesh = ParMesh(MPI_COMM_WORLD, smesh, partition.get());

   // Construct the NC refined mesh in parallel and serial. Once constructed a
   // global L2 projected solution should match exactly on each.
   Array<int> refines, serial_refines(1);
   if (Mpi::WorldRank() == refining_rank)
   {
      refines.Append(0);
   }

   // Must be called on all ranks as it uses MPI calls internally.
   // All ranks will use the global element number dictated by rank 0 though.
   serial_refines[0] = pmesh.GetGlobalElementNum(0);
   MPI_Bcast(&serial_refines[0], 1, MPI_INT, refining_rank, MPI_COMM_WORLD);

   // Rank 0 refines the parallel mesh, all ranks refine the serial mesh
   smesh.GeneralRefinement(serial_refines, 1); // nonconformal
   pmesh.GeneralRefinement(refines, 1); // nonconformal

   REQUIRE(pmesh.GetGlobalNE() == 8 + 1);
   REQUIRE(smesh.GetNE() == 8 + 1);

   // Each pair of indices here represents sequential element indices to refine.
   // First the i element is refined, then in the resulting mesh the j element is
   // refined. These pairs were arrived at by looping over all possible i,j pairs and
   // checking for the addition of a face-edge constraint.
   std::vector<std::pair<int,int>> indices{{2,13}, {3,13}, {6,2}, {6,3}};

   // Rank 0 has all but one element in the parallel mesh. The remaining element
   // is owned by another processor if the number of ranks is greater than one.
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

      // Loop over interior faces, fill and check face transform on the serial.
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
      check_l2_projection(ttmp, sttmp, 4);

      ttmp.ExchangeFaceNbrData();
      ttmp.Rebalance();

      check_l2_projection(ttmp, sttmp, 4);
   }
} // test case


#endif // MFEM_USE_MPI

} // namespace mfem
