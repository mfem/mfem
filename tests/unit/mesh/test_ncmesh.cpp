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
#include "mesh_test_utils.hpp"
#include "unit_tests.hpp"

#include <array>
namespace mfem
{

constexpr double EPS = 1e-10;

// Helper to count H1 essential dofs for a given order with a given attribute
template <typename FECollection, bool TDOF = true>
int CountEssentialDof(Mesh &mesh, int order, int attribute)
{
   constexpr int dim = 3;
   FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
   bdr_attr_is_ess = 0;
   bdr_attr_is_ess[mesh.bdr_attributes.Find(attribute)] = 1;

   if (TDOF)
   {
      Array<int> ess_tdof_list;
      fes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
      return ess_tdof_list.Size();
   }
   else
   {
      // VDOF
      Array<int> ess_vdof_marker, vdof_list;
      fes.GetEssentialVDofs(bdr_attr_is_ess, ess_vdof_marker);
      fes.MarkerToList(ess_vdof_marker, vdof_list);
      return vdof_list.Size();
   }
};

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

// Helper to create a mesh of a tet with four face neighbor tets and internal boundary between
Mesh StarMesh()
{
   const int nnode = 4 + 4;
   const int nelem = 5;

   Mesh mesh(3, nnode, nelem);

   // central tet
   mesh.AddVertex(0.0, 0.0, 0.0);
   mesh.AddVertex(1.0, 0.0, 0.0);
   mesh.AddVertex(0.0, 1.0, 0.0);
   mesh.AddVertex(0.0, 0.0, 1.0);

   mesh.AddVertex( 1.0,  1.0,  1.0); // opposite 0
   mesh.AddVertex(-1.0,  0.0,  0.0); // opposite 1
   mesh.AddVertex( 0.0, -1.0,  0.0); // opposite 2
   mesh.AddVertex( 0.0,  0.0, -1.0); // opposite 3

   mesh.AddTet(0, 1, 2, 3, 1); // central
   mesh.AddTet(4, 1, 2, 3, 2); // opposite 0
   mesh.AddTet(0, 5, 2, 3, 3); // opposite 1
   mesh.AddTet(0, 1, 6, 3, 4); // opposite 2
   mesh.AddTet(0, 1, 2, 7, 5); // opposite 3

   mesh.FinalizeTopology();
   mesh.Finalize(true, true);

   // Introduce internal boundary elements
   const int new_attribute = mesh.bdr_attributes.Max() + 1;
   Array<int> original_boundary_vertices;
   for (int f = 0; f < mesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      mesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && mesh.GetAttribute(e1) != mesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = mesh.GetFace(f)->Duplicate(&mesh);
         new_elem->SetAttribute(new_attribute);
         new_elem->GetVertices(original_boundary_vertices);
         mesh.AddBdrElement(new_elem);
      }
   }
   mesh.SetAttributes();
   mesh.FinalizeTopology();
   mesh.Finalize(true, true);

   return mesh;
}

Mesh DividingPlaneMesh(bool tet_mesh = true, bool split = true)
{
   auto mesh = Mesh("../../data/ref-cube.mesh");
   {
      Array<Refinement> refs;
      refs.Append(Refinement(0, Refinement::X));
      mesh.GeneralRefinement(refs);
   }
   delete mesh.ncmesh;
   mesh.ncmesh = nullptr;
   mesh.FinalizeTopology();
   mesh.Finalize(true, true);

   mesh.SetAttribute(0, 1);
   mesh.SetAttribute(1, split ? 2 : 1);

   // Introduce internal boundary elements
   const int new_attribute = mesh.bdr_attributes.Max() + 1;
   for (int f = 0; f < mesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      mesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && mesh.GetAttribute(e1) != mesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = mesh.GetFace(f)->Duplicate(&mesh);
         new_elem->SetAttribute(new_attribute);
         mesh.AddBdrElement(new_elem);
      }
   }
   if (tet_mesh)
   {
      mesh = Mesh::MakeSimplicial(mesh);
   }
   mesh.FinalizeTopology();
   mesh.Finalize(true, true);
   return mesh;
}

// Define a pair of tet with a shared triangle in the y-z plane.
// Vary the vertex ordering to achieve the 3 possible odd orientations
Mesh OrientedTriFaceMesh(int orientation, bool add_extbdr = false)
{
   REQUIRE((orientation == 1 || orientation == 3 || orientation == 5));

   Mesh mesh(3, 4, 2);
   mesh.AddVertex(-1.0, 0.0, 0.0);
   mesh.AddVertex(0.0, 0.0, 0.0);
   mesh.AddVertex(0.0, 1.0, 0.0);
   mesh.AddVertex(0.0, 0.0, 1.0);

   // opposing vertex
   mesh.AddVertex(1.0, 0.0, 0.0);

   mesh.AddTet(0, 1, 2, 3, 1);

   switch (orientation)
   {
      case 1:
         mesh.AddTet(4,2,1,3,2); break;
      case 3:
         mesh.AddTet(4,3,2,1,2); break;
      case 5:
         mesh.AddTet(4,1,3,2,2); break;
   }

   mesh.FinalizeTopology(add_extbdr);
   mesh.SetAttributes();

   auto *bdr = new Triangle(1,2,3,
                            mesh.bdr_attributes.Size() == 0 ? 1 : mesh.bdr_attributes.Max() + 1);
   mesh.AddBdrElement(bdr);

   mesh.FinalizeTopology(false);
   mesh.Finalize();
   return mesh;
};

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

// Given a parallel and a serial mesh, perform an L2 projection and check the
// solutions match exactly.
std::array<double, 2> CheckL2Projection(ParMesh& pmesh, Mesh& smesh, int order,
                                        std::function<double(Vector const&)> exact_soln)
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

      double snorm = x.ComputeL2Error(rhs_coef);

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
      return x.ComputeL2Error(rhs_coef) / snorm;
   }();

   auto perror = [&]
   {
      // parallel solve
      ParFiniteElementSpace fes(&pmesh, &fec);
      ParLinearForm b(&fes);

      ParGridFunction x(&fes);
      x = 0.0;

      double pnorm = x.ComputeL2Error(rhs_coef);
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
      return x.ComputeL2Error(rhs_coef) / pnorm;
   }();

   return {serror, perror};
}

TEST_CASE("EdgeFaceConstraint",  "[Parallel], [NCMesh]")
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
         single_edge_refine[0].index = 0;
         single_edge_refine[0].ref_type = Refinement::X;

         smesh.GeneralRefinement(single_edge_refine, 0); // conformal
      }


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
         {
            auto error = CheckL2Projection(ttmp, sttmp, 4, exact_soln);
            double constexpr tol = 1e-9;
            CHECK(std::abs(error[1] - error[0]) < tol);
         }
         ttmp.ExchangeFaceNbrData();
         ttmp.Rebalance();
         {
            auto error = CheckL2Projection(ttmp, sttmp, 4, exact_soln);
            double constexpr tol = 1e-9;
            CHECK(std::abs(error[1] - error[0]) < tol);
         }
      }
   }

   auto CheckSerialParallelH1Equivalence = [](Mesh &smesh)
   {
      constexpr int dim = 3;
      constexpr int order = 2;
      H1_FECollection nd_fec(order, dim);
      FiniteElementSpace fes(&smesh, &nd_fec);
      const auto serial_ntdof = fes.GetTrueVSize();

      ParMesh mesh(MPI_COMM_WORLD, smesh);
      ParFiniteElementSpace pfes(&mesh, &nd_fec);
      const auto parallel_ntdof = pfes.GlobalTrueVSize();

      // If nc constraints have been observed correctly, the number of true dof in
      // parallel should match the number of true dof in serial. If the number of
      // parallel dofs is greater, then a slave constraint has not been fully labeled.
      CHECK(serial_ntdof == parallel_ntdof);
   };

   auto CheckSerialParallelNDEquivalence = [](Mesh &smesh)
   {
      constexpr int dim = 3;
      constexpr int order = 1;
      ND_FECollection nd_fec(order, dim);
      FiniteElementSpace fes(&smesh, &nd_fec);
      const auto serial_ntdof = fes.GetTrueVSize();

      ParMesh mesh(MPI_COMM_WORLD, smesh);
      ParFiniteElementSpace pfes(&mesh, &nd_fec);
      const auto parallel_ntdof = pfes.GlobalTrueVSize();

      // If nc constraints have been observed correctly, the number of true dof in
      // parallel should match the number of true dof in serial. If the number of
      // parallel dofs is greater, then a slave constraint has not been fully labeled.
      CHECK(serial_ntdof == parallel_ntdof);
   };

   SECTION("LevelTwoRefinement")
   {
      Mesh smesh("../../data/ref-tetrahedron.mesh");
      Array<Refinement> aniso_ref(1);
      aniso_ref[0].index = 0;
      aniso_ref[0].ref_type = Refinement::X;
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
            CheckSerialParallelNDEquivalence(smesh3);
            CheckSerialParallelH1Equivalence(smesh3);
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

      // This particular partition was found by brute force search. The default rebalancing
      // can in rare cases produce similar local patterns, particularly for highly adapted meshes.
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
         double constexpr tol = 1e-9;
         CHECK(std::abs(error[1] - error[0]) < tol);
      }
   }

} // test case

Mesh CylinderMesh(Geometry::Type el_type, bool quadratic, int variant = 0)
{
   double c[3];

   int nnodes = (el_type == Geometry::CUBE) ? 24 : 15;
   int nelems = 8; // Geometry::PRISM
   if (el_type == Geometry::CUBE)        { nelems = 10; }
   if (el_type == Geometry::TETRAHEDRON) { nelems = 24; }

   Mesh mesh(3, nnodes, nelems);

   for (int i=0; i<3; i++)
   {
      if (el_type != Geometry::CUBE)
      {
         c[0] = 0.0;  c[1] = 0.0;  c[2] = 2.74 * i;
         mesh.AddVertex(c);
      }

      for (int j=0; j<4; j++)
      {
         if (el_type == Geometry::CUBE)
         {
            c[0] = 1.14 * ((j + 1) % 2) * (1 - j);
            c[1] = 1.14 * (j % 2) * (2 - j);
            c[2] = 2.74 * i;
            mesh.AddVertex(c);
         }

         c[0] = 2.74 * ((j + 1) % 2) * (1 - j);
         c[1] = 2.74 * (j % 2) * (2 - j);
         c[2] = 2.74 * i;
         mesh.AddVertex(c);
      }
   }

   for (int i=0; i<2; i++)
   {
      if (el_type == Geometry::CUBE)
      {
         mesh.AddHex(8*i, 8*i+2, 8*i+4, 8*i+6,
                     8*(i+1), 8*(i+1)+2, 8*(i+1)+4, 8*(i+1)+6);
      }

      for (int j=0; j<4; j++)
      {
         if (el_type == Geometry::PRISM)
         {
            switch (variant)
            {
               case 0:
                  mesh.AddWedge(5*i, 5*i+j+1, 5*i+(j+1)%4+1,
                                5*(i+1), 5*(i+1)+j+1, 5*(i+1)+(j+1)%4+1);
                  break;
               case 1:
                  mesh.AddWedge(5*i, 5*i+j+1, 5*i+(j+1)%4+1,
                                5*(i+1), 5*(i+1)+j+1, 5*(i+1)+(j+1)%4+1);
                  break;
               case 2:
                  mesh.AddWedge(5*i+(j+1)%4+1, 5*i, 5*i+j+1,
                                5*(i+1)+(j+1)%4+1, 5*(i+1), 5*(i+1)+j+1);
                  break;
            }
         }
         else if (el_type == Geometry::CUBE)
         {
            mesh.AddHex(8*i+2*j, 8*i+2*j+1, 8*i+(2*j+3)%8, 8*i+(2*j+2)%8,
                        8*(i+1)+2*j, 8*(i+1)+2*j+1, 8*(i+1)+(2*j+3)%8,
                        8*(i+1)+(2*j+2)%8);
         }
         else if (el_type == Geometry::TETRAHEDRON)
         {
            mesh.AddTet(5*i, 5*i+j+1, 5*i+(j+1)%4+1, 5*(i+1));
            mesh.AddTet(5*i+j+1, 5*i+(j+1)%4+1, 5*(i+1), 5*(i+1)+j+1);
            mesh.AddTet(5*i+(j+1)%4+1, 5*(i+1), 5*(i+1)+j+1, 5*(i+1)+(j+1)%4+1);
         }
      }
   }

   mesh.FinalizeTopology();

   if (quadratic)
   {
      mesh.SetCurvature(2);

      if (el_type == Geometry::CUBE)
      {
         auto quad_cyl_hex = [](const Vector& x, Vector& d)
         {
            d.SetSize(3);
            d = x;
            const double Rmax = 2.74;
            const double Rmin = 1.14;
            double ax = std::abs(x[0]);
            if (ax <= 1e-6) { return; }
            double ay = std::abs(x[1]);
            if (ay <= 1e-6) { return; }
            double r = ax + ay;
            if (r <= Rmin + 1e-6) { return; }

            double sx = std::copysign(1.0, x[0]);
            double sy = std::copysign(1.0, x[1]);

            double R = (Rmax - Rmin) * Rmax / (r - Rmin);
            double r2 = r * r;
            double R2 = R * R;

            double acosarg = 0.5 * (r + std::sqrt(2.0 * R2 - r2)) / R;
            double tR = std::acos(std::min(acosarg, 1.0));
            double tQ = (1.0 + sx * sy * (ay - ax) / r);
            double tP = 0.25 * M_PI * (3.0 - (2.0 + sx) * sy);

            double t = tR + (0.25 * M_PI - tR) * tQ + tP;

            double s0 = std::sqrt(2.0 * R2 - r2);
            double s1 = 0.25 * std::pow(r + s0, 2);
            double s = std::sqrt(R2 - s1);

            d[0] = R * std::cos(t) - sx * s;
            d[1] = R * std::sin(t) - sy * s;

            return;
         };

         mesh.Transform(quad_cyl_hex);
      }
      else
      {
         auto quad_cyl = [](const Vector& x, Vector& d)
         {
            d.SetSize(3);
            d = x;
            double ax = std::abs(x[0]);
            double ay = std::abs(x[1]);
            double r = ax + ay;
            if (r < 1e-6) { return; }

            double sx = std::copysign(1.0, x[0]);
            double sy = std::copysign(1.0, x[1]);

            double t = ((2.0 - (1.0 + sx) * sy) * ax +
                        (2.0 - sy) * ay) * 0.5 * M_PI / r;
            d[0] = r * std::cos(t);
            d[1] = r * std::sin(t);

            return;
         };

         mesh.Transform(quad_cyl);
      }
   }

   mesh.Finalize(true);

   return mesh;
}

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

/**
 * @brief Test GetVectorValue on face neighbor elements for nonconformal meshes
 *
 * @param smesh The serial mesh to start from
 * @param nc_level Depth of refinement on processor boundaries
 * @param skip Refine every "skip" processor boundary element
 * @param use_ND Whether to use Nedelec elements (which are sensitive to orientation)
 */
void TestVectorValueInVolume(Mesh &smesh, int nc_level, int skip, bool use_ND)
{
   auto vector_exact_soln = [](const Vector& x, Vector& v)
   {
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      v = (d -= x);
   };

   smesh.Finalize();
   smesh.EnsureNCMesh(true);

   auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);

   // Apply refinement on face neighbors to achieve a given nc level mismatch.
   for (int i = 0; i < nc_level; ++i)
   {
      // To refine the face neighbors, need to know where they are.
      pmesh.ExchangeFaceNbrData();
      Array<int> elem_to_refine;
      // Refine only on odd ranks.
      if ((Mpi::WorldRank() + 1) % 2 == 0)
      {
         // Refine a subset of all shared faces. Using a subset helps to
         // mix in conformal faces with nonconformal faces.
         for (int n = 0; n < pmesh.GetNSharedFaces(); ++n)
         {
            if (n % skip != 0) { continue; }
            const int local_face = pmesh.GetSharedFace(n);
            const auto &face_info = pmesh.GetFaceInformation(local_face);
            REQUIRE(face_info.IsShared());
            REQUIRE(face_info.element[1].location == Mesh::ElementLocation::FaceNbr);
            elem_to_refine.Append(face_info.element[0].index);
         }
      }
      pmesh.GeneralRefinement(elem_to_refine);
   }

   // Do not rebalance again! The test is also checking for nc refinements
   // along the processor boundary.

   // Create a grid function of the mesh coordinates
   pmesh.ExchangeFaceNbrData();
   pmesh.EnsureNodes();
   REQUIRE(pmesh.OwnsNodes());
   GridFunction * const coords = pmesh.GetNodes();
   dynamic_cast<ParGridFunction *>(pmesh.GetNodes())->ExchangeFaceNbrData();

   // Project the linear function onto the mesh. Quadratic ND tetrahedral
   // elements are the first to require face orientations.
   const int order = 2, dim = 3;
   std::unique_ptr<FiniteElementCollection> fec;
   if (use_ND)
   {
      fec = std::unique_ptr<ND_FECollection>(new ND_FECollection(order, dim));
   }
   else
   {
      fec = std::unique_ptr<RT_FECollection>(new RT_FECollection(order, dim));
   }
   ParFiniteElementSpace pnd_fes(&pmesh, fec.get());

   ParGridFunction psol(&pnd_fes);

   VectorFunctionCoefficient func(3, vector_exact_soln);
   psol.ProjectCoefficient(func);
   psol.ExchangeFaceNbrData();

   mfem::Vector value(3), exact(3), position(3);
   const IntegrationRule &ir = mfem::IntRules.Get(Geometry::Type::TETRAHEDRON,
                                                  order + 1);

   // Check that non-ghost elements match up on the serial and parallel spaces.
   for (int n = 0; n < pmesh.GetNE(); ++n)
   {
      constexpr double tol = 1e-12;
      for (const auto &ip : ir)
      {
         coords->GetVectorValue(n, ip, position);
         psol.GetVectorValue(n, ip, value);

         vector_exact_soln(position, exact);

         REQUIRE(value.Size() == exact.Size());
         CHECK((value -= exact).Normlinf() < tol);
      }
   }

   // Loop over face neighbor elements and check the vector values match in the
   // face neighbor elements.
   for (int n = 0; n < pmesh.GetNSharedFaces(); ++n)
   {
      const int local_face = pmesh.GetSharedFace(n);
      const auto &face_info = pmesh.GetFaceInformation(local_face);
      REQUIRE(face_info.IsShared());
      REQUIRE(face_info.element[1].location == Mesh::ElementLocation::FaceNbr);

      auto &T = *pmesh.GetFaceNbrElementTransformation(face_info.element[1].index);

      constexpr double tol = 1e-12;
      for (const auto &ip : ir)
      {
         T.SetIntPoint(&ip);
         coords->GetVectorValue(T, ip, position);
         psol.GetVectorValue(T, ip, value);

         vector_exact_soln(position, exact);

         REQUIRE(value.Size() == exact.Size());
         CHECK((value -= exact).Normlinf() < tol);
      }
   }
}

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

   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");

   REQUIRE(smesh.GetNBE() == 4);

   Array<Refinement> refs;
   refs.Append(Refinement(0, Refinement::X));
   smesh.GeneralRefinement(refs);

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(0, 1);
   smesh.SetAttribute(1, 2);

   REQUIRE(smesh.GetNBE() == 2 * 3);

   smesh.FinalizeTopology();
   smesh.Finalize();

   // Introduce an internal boundary element
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
         break;
      }
   }
   smesh.FinalizeTopology(); // Finalize to build relevant tables
   smesh.Finalize();

   // Exactly one boundary element must be added
   REQUIRE(smesh.GetNBE() == 2 * 3 + 1);

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

// Count the number of essential dofs on a ParMesh.
template <typename FECollection, bool TDOF = true>
int CountEssentialDof(ParMesh &mesh, int order, int attribute)
{
   constexpr int dim = 3;
   FECollection fec(order, dim);
   ParFiniteElementSpace pfes(&mesh, &fec);

   Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
   bdr_attr_is_ess = 0;
   bdr_attr_is_ess[mesh.bdr_attributes.Find(attribute)] = 1;

   Array<int> ess_tdof_list;
   pfes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
   if (TDOF)
   {
      pfes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
      return ess_tdof_list.Size();
   }
   else
   {
      // VDOF
      Array<int> ess_vdof_marker, vdof_list;
      pfes.GetEssentialVDofs(bdr_attr_is_ess, ess_vdof_marker);
      pfes.MarkerToList(ess_vdof_marker, vdof_list);
      return vdof_list.Size();
   }
};

template <typename FECollection, bool TDOF = true>
int ParCountEssentialDof(ParMesh &mesh, int order, int attribute)
{
   auto num_essential_dof = CountEssentialDof<FECollection, TDOF>(mesh, order,
                                                                  attribute);
   MPI_Allreduce(MPI_IN_PLACE, &num_essential_dof, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   return num_essential_dof;
};

TEST_CASE("InteriorBoundaryReferenceTets", "[Parallel], [NCMesh]")
{
   constexpr auto seed = 314159;
   srand(seed);
   auto p = 1;//GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");

   REQUIRE(smesh.GetNBE() == 4);

   Array<Refinement> refs;
   refs.Append(Refinement(0, Refinement::X));
   smesh.GeneralRefinement(refs);

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(0, 1);
   smesh.SetAttribute(1, 2);

   REQUIRE(smesh.GetNBE() == 2 * 3);

   smesh.FinalizeTopology();
   smesh.Finalize();

   // Introduce an internal boundary element
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
         break;
      }
   }
   smesh.FinalizeTopology(); // Finalize to build relevant tables
   smesh.Finalize();

   // Exactly one boundary element must be added
   REQUIRE(smesh.GetNBE() == 2 * 3 + 1);

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
   // Level of refinement difference across the processor boundary from root zero to the
   // others
   auto ref_level = 1;//GENERATE(1,2,3);
   auto refined_attribute = 2;//GENERATE(1,2);
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
   // divided across the processor boundary. This necessitates the
   // GhostBoundaryElement treatment.
   auto partition = std::unique_ptr<int[]>(new int[modified_smesh.GetNE()]);
   for (int i = 0; i < modified_smesh.GetNE(); i++)
   {
      // Randomly assign to any processor but zero.
      partition[i] = Mpi::WorldSize() > 1 ? 1 + rand() % (Mpi::WorldSize() - 1) : 0;
   }
   if (Mpi::WorldSize() > 0)
   {
      // Make sure rank 0 has the non-refined attribute. This ensures it will have
      // a parent face with only ghost children.
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

   // return;
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

TEST_CASE("InteriorBoundaryInlineTetRefines", "[Parallel], [NCMesh]")
{
   int p = GENERATE(1,2);
   CAPTURE(p);

   auto smesh = Mesh("../../data/inline-tet.mesh");
   smesh.FinalizeTopology();
   smesh.Finalize();

   // Mark even and odd elements with different attributes
   auto num_attributes = 3;
   for (int i = 0; i < smesh.GetNE(); ++i)
   {
      smesh.SetAttribute(i, (i % num_attributes) + 1);
   }

   smesh.SetAttributes();
   int initial_nbe = smesh.GetNBE();

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
   auto ref_level = GENERATE(1,2);
   auto marked_attribute = GENERATE(1,2,3);
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
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = Mesh("../../data/ref-cube.mesh");
   smesh.EnsureNCMesh();

   REQUIRE(smesh.GetNBE() == 6);

   Array<Refinement> refs;
   refs.Append(Refinement(0, Refinement::X));
   smesh.GeneralRefinement(refs);

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(0, 1);
   smesh.SetAttribute(1, 2);

   REQUIRE(smesh.GetNBE() == 2 * 5);

   // Throw away the NCMesh, will restart NC later.
   delete smesh.ncmesh;
   smesh.ncmesh = nullptr;

   smesh.FinalizeTopology();
   smesh.Finalize();

   // Introduce an internal boundary element
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
         break;
      }
   }
   smesh.FinalizeTopology(); // Finalize to build relevant tables
   smesh.Finalize();

   // Exactly one boundary element must be added
   REQUIRE(smesh.GetNBE() == 2 * 5 + 1);

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

      // There should now be four internal boundary elements, where there was one
      // before.
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
         // Make sure on rank 1 there is a parent face with only ghost child
         // faces. This can cause issues with higher order dofs being uncontrolled.
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

TEST_CASE("InteriorBoundaryInlineHexRefines", "[Parallel], [NCMesh]")
{
   int p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = Mesh("../../data/inline-hex.mesh");
   smesh.FinalizeTopology();
   smesh.Finalize();

   // Mark even and odd elements with different attributes
   for (int i = 0; i < smesh.GetNE(); ++i)
   {
      smesh.SetAttribute(i, (i % 2) + 1);
   }

   smesh.SetAttributes();
   int initial_nbe = smesh.GetNBE();

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

   // Mark every third element for refinement
   Array<int> elem_to_refine;
   const int factor = 3;
   for (int i = 0; i < smesh.GetNE()/factor; ++i)
   {
      elem_to_refine.Append(factor * i);
   }
   smesh.GeneralRefinement(elem_to_refine);

   pmesh = CheckParMeshNBE(smesh);
   pmesh->FinalizeTopology();
   pmesh->Finalize();
   pmesh->ExchangeFaceNbrData();

   // repopulate the local to shared map.
   local_to_shared.clear();
   for (int i = 0; i < pmesh->GetNSharedFaces(); ++i)
   {
      local_to_shared[pmesh->GetSharedFace(i)] = i;
   }

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
}

TEST_CASE("ParMeshInternalBoundaryStarMesh", "[Parallel], [NCMesh]")
{
   auto smesh = StarMesh();
   smesh.EnsureNCMesh(true);

   if (Mpi::WorldSize() < 5) { return;}

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
      // Refining an element attached to the core should not change the number of essential
      // DOF, or the owner of them.

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

      // Refining on only one side of the boundary face should not change the number of
      // essential true dofs, which should match the number within the original face.
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
   auto RefineAttribute = [](Mesh& mesh, int attr, int ref_level)
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
         RefineAttribute(mesh, attr, ref_level);

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

         RefineAttribute(*pmesh, attr, ref_level);

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

         RefineAttribute(*pmesh, attr, ref_level);
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
      1. Define an ND space, and project a smooth non polynomial function onto the space.
      2. Compute y-z components in the face, and check that they are equal when evaluated
         from either side of the face. Tangential continuity of the ND space should ensure
         they are identical, if orientations are correctly accounted for.
      3. Mark the mesh as NC, build a new FESpace, and repeat. There should be no change as
         the faces are "conformal" though they are within the NC structure.
      3. Partition the mesh, create a ParFESpace and repeat the above. There should be no
         difference in conformal parallel.
      4. Construct the ParMesh from the NCMesh and repeat. As above, there should be no
         change.
      5. Perform NC refinement on one side of the internal face, the number of conformal dof
         in the face will not change, so there should also be no difference. This will be
         complicated by ensuring the slave evaluations are at the same points.
   */

   auto orientation = GENERATE(1,3,5);
   auto smesh = OrientedTriFaceMesh(orientation);
   smesh.EnsureNodes();

   CHECK(smesh.GetNBE() == 1);

   // A smooth function in each vector component
   constexpr int order = 3, dim = 3, quadrature_order = 4;
   constexpr double kappa = 2 * M_PI;
   auto E_exact = [=](const Vector &x, Vector &E)
   {
      E(0) = cos(kappa * x(1));
      E(1) = cos(kappa * x(2));
      E(2) = cos(kappa * x(0));
   };
   VectorFunctionCoefficient E_coeff(dim, E_exact);

   // Helper for evaluating the ND grid function on either side of the first conformal shared face.
   // Specific to the pair of tet mesh described above, but can be generalized.
   auto CheckParallelNDConformal = [&](ParMesh &mesh)
   {
      ND_FECollection fe_collection(order, dim);
      ParFiniteElementSpace fe_space(&mesh, &fe_collection);
      ParGridFunction E(&fe_space);

      E.ProjectCoefficient(E_coeff);
      E.ExchangeFaceNbrData();

      auto *P = fe_space.GetProlongationMatrix();
      if (P != nullptr)
      {
         // Projection does not respect the non-conformal constraints.
         // Extract the true (conformal) and prolongate to get the NC respecting projection.
         auto E_true = E.GetTrueVector();
         P->Mult(E_true, E);
         E.ExchangeFaceNbrData();
      }
      ParGridFunction * const coords = dynamic_cast<ParGridFunction*>
                                       (mesh.GetNodes());

      const auto &ir = IntRules.Get(Geometry::Type::TRIANGLE, quadrature_order);
      IntegrationRule left_eir(ir.GetNPoints()),
                      right_eir(ir.GetNPoints()); // element integration rules

      for (int n = 0; n < mesh.GetNBE(); n++)
      {
         auto f = mesh.GetBdrElementFaceIndex(n);

         auto finfo = mesh.GetFaceInformation(f);
         auto &face_element_transform = finfo.IsShared()
                                        ? *mesh.GetSharedFaceTransformationsByLocalIndex(f, true)
                                        : *mesh.GetFaceElementTransformations(f);

         face_element_transform.Loc1.Transform(ir, left_eir);
         face_element_transform.Loc2.Transform(ir, right_eir);

         constexpr double tol = 1e-14;
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

            // Check that the second and third rows agree.
            // The y and z should agree as the normal is in the x direction
            CHECK(std::abs(left_val(1) - right_val(1)) < tol);
            CHECK(std::abs(left_val(2) - right_val(2)) < tol);
         }
      }

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

      CheckParallelNDConformal(*pmesh);
   }

   SECTION("ConformalSerialUniformRefined")
   {
      smesh.UniformRefinement();
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      CheckParallelNDConformal(*pmesh);
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
      CheckParallelNDConformal(*pmesh);
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

      CheckParallelNDConformal(*pmesh);
   }

   SECTION("NonconformalSerialUniformRefined")
   {
      smesh.UniformRefinement();
      smesh.EnsureNCMesh(true);
      auto pmesh = CheckParMeshNBE(smesh);
      pmesh->Finalize();
      pmesh->ExchangeFaceNbrData();

      CheckParallelNDConformal(*pmesh);
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

      CheckParallelNDConformal(*pmesh);
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

      CheckParallelNDConformal(*pmesh);
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

      CheckParallelNDConformal(*pmesh);
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
                  CheckParallelNDConformal(*CheckParMeshNBE(smesh3));
               }
            }
         }
      }
   }

}

// Helper to check the identity PR = I on a ParFiniteElementSpace.
void CheckRPIdentity(const ParFiniteElementSpace& pfespace)
{
   const SparseMatrix *R = pfespace.GetRestrictionMatrix();
   HypreParMatrix *P = pfespace.Dof_TrueDof_Matrix();

   REQUIRE(R != nullptr);
   REQUIRE(P != nullptr);

   HypreParMatrix *hR = new HypreParMatrix(
      pfespace.GetComm(), pfespace.GlobalTrueVSize(),
      pfespace.GlobalVSize(), pfespace.GetTrueDofOffsets(),
      pfespace.GetDofOffsets(),
      const_cast<SparseMatrix*>(R)); // Non owning so cast is ok

   REQUIRE(hR->Height() == P->Width());
   REQUIRE(hR->Width() == P->Height());

   REQUIRE(hR != nullptr);
   HypreParMatrix *I = ParMult(hR, P);

   // Square matrix so the "diag" is the only bit we need.
   SparseMatrix diag;
   I->GetDiag(diag);
   for (int i = 0; i < diag.Height(); i++)
      for (int j = 0; j < diag.Width(); j++)
      {
         // cast to const to force a zero return rather than an abort.
         CHECK(const_cast<const SparseMatrix&>(diag)(i, j)  == (i == j ? 1.0 : 0.0));
      }

   delete hR;
   delete I;
}

TEST_CASE("Parallel RP=I", "[Parallel], [NCMesh]")
{
   const int order = GENERATE(1, 2, 3);
   CAPTURE(order);
   const int dim = 3;

   SECTION("Hex")
   {
      // Split the hex into a pair, then isotropically refine one of them.
      Mesh smesh("../../data/ref-cube.mesh");
      Array<Refinement> refinements(1);
      refinements[0].index = 0;
      refinements[0].ref_type = Refinement::X;
      smesh.GeneralRefinement(refinements);
      refinements[0].ref_type = Refinement::XYZ;
      smesh.GeneralRefinement(refinements);
      ParMesh mesh(MPI_COMM_WORLD, smesh);
      SECTION("ND")
      {
         ND_FECollection fec(order, dim);
         ParFiniteElementSpace fespace(&mesh, &fec);
         CheckRPIdentity(fespace);
      }
      SECTION("H1")
      {
         H1_FECollection fec(order, dim);
         ParFiniteElementSpace fespace(&mesh, &fec);
         CheckRPIdentity(fespace);
      }
   }

   SECTION("Tet")
   {
      // Split the hex into a pair, then isotropically refine one of them.
      Mesh smesh("../../data/ref-tetrahedron.mesh");
      Array<Refinement> refinements(1);
      refinements[0].index = 0;
      refinements[0].ref_type = Refinement::X;
      smesh.GeneralRefinement(refinements);
      bool use_nc = GENERATE(false, true);
      smesh.EnsureNCMesh(use_nc);
      refinements[0].ref_type = Refinement::XYZ;
      smesh.GeneralRefinement(refinements);
      smesh.EnsureNCMesh(true); // Always checking NC
      ParMesh mesh(MPI_COMM_WORLD, smesh);
      SECTION("ND")
      {
         ND_FECollection fec(order, dim);
         ParFiniteElementSpace fespace(&mesh, &fec);
         CheckRPIdentity(fespace);
      }
      SECTION("H1")
      {
         H1_FECollection fec(order, dim);
         ParFiniteElementSpace fespace(&mesh, &fec);
         CheckRPIdentity(fespace);
      }
   }
}

#endif // MFEM_USE_MPI

TEST_CASE("ReferenceCubeInternalBoundaries", "[NCMesh]")
{
   auto p = GENERATE(1,2,3);
   CAPTURE(p);

   auto smesh = Mesh("../../data/ref-cube.mesh");

   CheckPoisson(smesh, p);

   smesh.EnsureNCMesh();
   Array<Refinement> refs;
   refs.Append(Refinement(0, Refinement::X));
   smesh.GeneralRefinement(refs);

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(1, 2);

   REQUIRE(smesh.GetNBE() == 2 * 5);

   delete smesh.ncmesh;
   smesh.ncmesh = nullptr;

   // Introduce an internal boundary element
   for (int f = 0; f < smesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = smesh.GetFace(f)->Duplicate(&smesh);
         new_elem->SetAttribute(7);
         smesh.AddBdrElement(new_elem);
      }
   }

   smesh.FinalizeTopology();
   smesh.Finalize();

   // Exactly one boundary element must be added
   CHECK(smesh.GetNBE() == 2 * 5 + 1);

   smesh.EnsureNCMesh();
   CHECK(smesh.GetNBE() == 2 * 5 + 1);

   int without_internal, with_internal;
   with_internal = CheckPoisson(smesh, p); // Include the internal boundary
   without_internal = CheckPoisson(smesh, p,
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

   auto ref_type = char(GENERATE(//Refinement::Y, Refinement::Z, Refinement::YZ,
                           Refinement::XYZ));

   for (auto ref : {0,1})
   {
      refs[0].index = ref;

      auto ssmesh = Mesh(smesh);

      CAPTURE(ref_type);

      // Now NC refine one of the attached elements, this should result in 2
      // internal boundary elements.
      refs[0].ref_type = ref_type;

      ssmesh.GeneralRefinement(refs);

      // There should now be four internal boundary elements, where there was one
      // before.
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

      // All slaves dofs that are introduced on the face are constrained by
      // the master dofs, thus the additional constraints on the internal
      // boundary are purely on the master face, which matches the initial
      // unrefined case.
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

   auto smesh = Mesh("../../data/ref-cube.mesh");
   smesh.EnsureNCMesh();
   Array<Refinement> refs;
   refs.Append(Refinement(0, Refinement::X));
   smesh.GeneralRefinement(refs);

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(1, 2);

   REQUIRE(smesh.GetNBE() == 2 * 5);

   delete smesh.ncmesh;
   smesh.ncmesh = nullptr;

   smesh.UniformRefinement();

   // Introduce four internal boundary elements
   for (int f = 0; f < smesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = smesh.GetFace(f)->Duplicate(&smesh);
         new_elem->SetAttribute(7);
         smesh.AddBdrElement(new_elem);
      }
   }

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
   refs.DeleteAll();
   for (int n = 0; n < smesh.GetNE(); ++n)
   {
      if (smesh.GetAttribute(n) == 2)
      {
         refs.Append(Refinement{n, Refinement::XYZ});
      }
   }

   smesh.GeneralRefinement(refs);

   smesh.FinalizeTopology();
   smesh.Finalize();

   // There should now be 16 internal boundary elements, where there were 4 before

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
      refs[0].index = ref;
      refs[0].ref_type = Refinement::XYZ;
      auto ssmesh = Mesh(smesh);
      ssmesh.GeneralRefinement(refs);

      // There should now be four internal boundary elements, where there was one
      // before.
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
         refs.Append(Refinement{n, Refinement::XYZ});
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
   auto smesh = Mesh("../../data/ref-cube.mesh");
   smesh.EnsureNCMesh();
   Array<Refinement> refs(1);
   refs[0].index = 0;
   refs[0].ref_type = Refinement::X;
   smesh.GeneralRefinement(refs);

   // Now have two elements.
   smesh.FinalizeTopology();
   smesh.Finalize();

   auto p = GENERATE(1, 2, 3);
   CAPTURE(p);

   // Check that Poisson can be solved on the domain
   CheckPoisson(smesh, p);

   auto ref_type = char(GENERATE(Refinement::X, Refinement::Y, Refinement::Z,
                                 Refinement::XY, Refinement::XZ, Refinement::YZ,
                                 Refinement::XYZ));
   CAPTURE(ref_type);
   for (auto refined_elem : {0}) // The left or the right element
   {
      refs[0].index = refined_elem;
      auto ssmesh = Mesh(smesh);

      // Now NC refine one of the attached elements
      refs[0].ref_type = ref_type;

      ssmesh.GeneralRefinement(refs);
      ssmesh.FinalizeTopology();
      ssmesh.Finalize();

      CAPTURE(refined_elem);
      CheckPoisson(ssmesh, p);
   }
}

TEST_CASE("PoissonOnReferenceTetNC", "[NCMesh]")
{
   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");

   auto p = GENERATE(1, 2, 3);
   CAPTURE(p);

   CheckPoisson(smesh, p);

   Array<Refinement> refs(1);
   refs[0].index = 0;
   refs[0].ref_type = Refinement::X;

   smesh.GeneralRefinement(refs);

   // Now have two elements.
   smesh.FinalizeTopology();
   smesh.Finalize();

   // Check that Poisson can be solved on the pair of tets
   CheckPoisson(smesh, p);

   auto nc = GENERATE(false, true);
   CAPTURE(nc);

   smesh.EnsureNCMesh(GENERATE(false, true));

   for (auto refined_elem : {0, 1})
   {
      auto ssmesh = Mesh(smesh);

      refs[0].index = refined_elem;
      refs[0].ref_type = Refinement::XYZ;

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
   auto smesh = Mesh("../../data/ref-tetrahedron.mesh");

   REQUIRE(smesh.GetNBE() == 4);

   {
      Array<Refinement> refs;
      refs.Append(Refinement(0, Refinement::X));
      smesh.GeneralRefinement(refs);
   }

   // Now have a pair of elements, make the second element a different
   // attribute.
   smesh.SetAttribute(0, 1);
   smesh.SetAttribute(1, 2);

   // Introduce an internal boundary element
   const int new_attribute = smesh.bdr_attributes.Max() + 1;
   Array<int> original_boundary_vertices;
   for (int f = 0; f < smesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      smesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && smesh.GetAttribute(e1) != smesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = smesh.GetFace(f)->Duplicate(&smesh);
         new_elem->SetAttribute(new_attribute);
         new_elem->GetVertices(original_boundary_vertices);
         smesh.AddBdrElement(new_elem);
         break;
      }
   }

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

   // Refining on only one side of the boundary face should not change the number of
   // essential true dofs
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 1,
                                            smesh.bdr_attributes.Max()) == 6);
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 2,
                                            smesh.bdr_attributes.Max()) == 6 + 3 * 3);
   CHECK(CountEssentialDof<H1_FECollection>(smesh, 3,
                                            smesh.bdr_attributes.Max()) == 10 + 3 * 6);

   // The number of boundary faces should have increased.
   CHECK(smesh.GetNBE() == 3 * 4 + (3 + 1) * std::pow(4, 1+ref_level));
}

TEST_CASE("TetInternalBoundaryStarMesh", "[NCMesh]")
{
   auto smesh = StarMesh();
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

      // Refining on only one side of the boundary face should not change the number of
      // essential true dofs
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
   auto RefineAttribute = [](Mesh& mesh, int attr, int ref_level)
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

         RefineAttribute(mesh, attr, ref_level);

         CHECK(CountEssentialDof<H1_FECollection>(mesh, 1,
                                                  mesh.bdr_attributes.Max()) == 3*3);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 2,
                                                  mesh.bdr_attributes.Max()) == 5*5);
         CHECK(CountEssentialDof<H1_FECollection>(mesh, 3,
                                                  mesh.bdr_attributes.Max()) == 7*7);

         // Add the slave face dofs, then subtract off the vertex dofs which are double
         // counted due to being shared.
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

         RefineAttribute(mesh, attr, ref_level);
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

         RefineAttribute(mesh, attr, ref_level);

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

         RefineAttribute(mesh, attr, ref_level);

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
   constexpr double kappa = 2 * M_PI;
   auto E_exact = [=](const Vector &x, Vector &E)
   {
      E(0) = cos(kappa * x(1));
      E(1) = cos(kappa * x(2));
      E(2) = cos(kappa * x(0));
   };
   VectorFunctionCoefficient E_coeff(dim, E_exact);

   auto CheckSerialNDConformal = [&](Mesh &mesh, int num_essential_tdof,
                                     int num_essential_vdof)
   {
      ND_FECollection fe_collection(order, dim);
      FiniteElementSpace fe_space(&mesh, &fe_collection);
      GridFunction E(&fe_space);

      E.ProjectCoefficient(E_coeff);

      auto *P = fe_space.GetProlongationMatrix();
      if (P != nullptr)
      {
         // Projection does not respect the non-conformal constraints.
         // Extract the true (conformal) and prolongate to get the NC respecting projection.
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

            constexpr double tol = 1e-14;
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

               // Check that the second and third rows agree.
               // The y and z should agree as the normal is in the x direction
               CHECK(std::abs(left_val(1) - right_val(1)) < tol);
               CHECK(std::abs(left_val(2) - right_val(2)) < tol);
            }
         }
      }
   };

   SECTION("Conformal")
   {
      const int ntdof = 3*3 + 3*2;
      const int nvdof = ntdof;
      CheckSerialNDConformal(smesh, ntdof, nvdof);
   }

   SECTION("Nonconformal")
   {
      smesh.EnsureNCMesh(true);
      const int ntdof = 3*3 + 3*2;
      const int nvdof = ntdof;
      CheckSerialNDConformal(smesh, ntdof, nvdof);
   }

   SECTION("ConformalUniformRefined")
   {
      smesh.UniformRefinement();
      const int ntdof = 9*3 + 4*3*2;
      const int nvdof = ntdof;
      CheckSerialNDConformal(smesh, ntdof, nvdof);
   }

   SECTION("NonconformalUniformRefined")
   {
      smesh.EnsureNCMesh(true);
      smesh.UniformRefinement();
      const int ntdof = 9*3 + 4*3*2;
      const int nvdof = ntdof;
      CheckSerialNDConformal(smesh, ntdof, nvdof);
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
      CheckSerialNDConformal(smesh, ntdof, nvdof);
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
         CheckSerialNDConformal(smesh, ntdof, -1);

         if (smesh.GetAttribute(m) == 2)
         {
            el_to_refine[0] = m;
            CAPTURE(m);
            smesh.GeneralRefinement(el_to_refine);
            CheckSerialNDConformal(smesh, ntdof, -1);
         }
      }

   }
}

TEST_CASE("RP=I", "[NCMesh]")
{
   auto CheckFESpace = [](const FiniteElementSpace& fespace)
   {
      auto * const R = fespace.GetConformingRestriction();
      auto * const P = fespace.GetConformingProlongation();

      REQUIRE(R != nullptr);
      REQUIRE(P != nullptr);

      // Vector notation
      Vector e_i(R->Height()), e_j(P->Width());
      Vector Rrow(R->Width()), Pcol(P->Height());
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

            CHECK(Rrow * Pcol == (i == j ? 1.0 : 0.0));
         }
      }

      // Index notation
      CHECK(R->Height() == P->Width());
      CHECK(R->Width() == P->Height());
      for (int i = 0; i < R->Height(); i++)
         for (int j = 0; j < P->Width(); j++)
         {
            double dot = 0.0;
            for (int k = 0; k < R->Width(); k++)
            {
               dot += (*R)(i,k)*(*P)(k,j);
            }
            CHECK(dot == (i == j ? 1.0 : 0.0));
         }
   };

   SECTION("Hex")
   {
      const int dim = 3;
      const int order = GENERATE(1, 2);
      // Split the hex into a pair, then isotropically refine one of them.
      Mesh mesh("../../data/ref-cube.mesh");
      Array<Refinement> refinements(1);
      refinements[0].index = 0;
      refinements[0].ref_type = Refinement::X;
      mesh.GeneralRefinement(refinements);
      refinements[0].ref_type = Refinement::XYZ;
      mesh.GeneralRefinement(refinements);
      SECTION("ND")
      {
         ND_FECollection fec(order, dim);
         FiniteElementSpace fespace(&mesh, &fec);
         CheckFESpace(fespace);
      }
      SECTION("H1")
      {
         H1_FECollection fec(order, dim);
         FiniteElementSpace fespace(&mesh, &fec);
         CheckFESpace(fespace);
      }
   }

   SECTION("Tet")
   {
      const int dim = 3;
      const int order = GENERATE(1, 2);
      // Split the hex into a pair, then isotropically refine one of them.
      Mesh mesh("../../data/ref-tetrahedron.mesh");
      Array<Refinement> refinements(1);
      refinements[0].index = 0;
      refinements[0].ref_type = Refinement::X;
      mesh.GeneralRefinement(refinements);
      mesh.EnsureNCMesh(true);
      refinements[0].ref_type = Refinement::XYZ;
      mesh.GeneralRefinement(refinements);
      SECTION("ND")
      {
         ND_FECollection fec(order, dim);
         FiniteElementSpace fespace(&mesh, &fec);
         CheckFESpace(fespace);
      }
      SECTION("H1")
      {
         H1_FECollection fec(order, dim);
         FiniteElementSpace fespace(&mesh, &fec);
         CheckFESpace(fespace);
      }
   }
}

} // namespace mfem
