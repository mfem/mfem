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
#include "unit_tests.hpp"
#include "navier_solver.hpp"

using namespace mfem;
using namespace mfem::navier;

namespace
{

Array<int> Marker(const ParMesh &pmesh, std::initializer_list<int> attrs)
{
   Array<int> marker(pmesh.bdr_attributes.Max());
   marker = 0;
   for (int attr : attrs)
   {
      marker[attr - 1] = 1;
   }
   return marker;
}

class TestNavierSolver : public NavierSolver
{
public:
   using NavierSolver::NavierSolver;
   using NavierSolver::vel_ess_tdof;
};

} // namespace

TEST_CASE("Navier component Dirichlet BCs", "[Parallel][Navier]")
{
   constexpr real_t dt = 0.01;
   const bool pa = GENERATE(false, true);
   CAPTURE(pa);

   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                     1.0, 1.0, 1.0);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   TestNavierSolver solver(&pmesh, 2, 0.1);
   solver.EnablePA(pa);

   auto x_faces = Marker(pmesh, {3, 5});
   auto y_faces = Marker(pmesh, {2, 4});
   auto z_faces = Marker(pmesh, {1, 6});

   Array<int> domain(pmesh.attributes.Max());
   domain = 1;

   ConstantCoefficient zero(0.0);

   SECTION("Essential DOFs")
   {
      int expected_count = 0;

      SECTION("Single component")
      {
         solver.AddVelDirichletBC(new ConstantCoefficient(0.0), y_faces, 1);
         expected_count = 50;
      }
      SECTION("Multiple components on same attribute")
      {
         solver.AddVelDirichletBC(new ConstantCoefficient(0.0), y_faces, 1);
         solver.AddVelDirichletBC(new ConstantCoefficient(0.0), y_faces, 2);
         expected_count = 100;
      }
      SECTION("Component and vector BCs")
      {
         solver.AddVelDirichletBC(new ConstantCoefficient(0.0), y_faces, 1);
         solver.AddVelDirichletBC(
            new VectorConstantCoefficient(Vector({0.0, 0.0, 0.0})), x_faces);
         expected_count = 180;
      }

      solver.Setup(dt);

      const Array<int> &dofs = solver.vel_ess_tdof;
      int loc_count = dofs.Size(), glob_count = 0;
      int loc_unsorted = 0, glob_unsorted = 0;
      for (int i = 1; i < dofs.Size(); ++i)
      {
         if (dofs[i] <= dofs[i - 1]) { loc_unsorted = 1; }
      }

      MPI_Allreduce(&loc_count, &glob_count, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(&loc_unsorted, &glob_unsorted, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);

      // Q2 hexes on a 2x2x2 mesh have 5x5 = 25 scalar DOFs per face:
      // 2 y-faces -> 50 DOFs/component; adding full-vector x-faces (150 DOFs)
      // shares 4 edges x 5 nodes = 20 y-DOFs -> 180 distinct DOFs.
      REQUIRE(glob_count == expected_count);
      REQUIRE(glob_unsorted == 0);
   }

   SECTION("Tangential slip wall")
   {
      solver.AddVelDirichletBC(new ConstantCoefficient(0.0), y_faces, 1);
      solver.AddAccelTerm(
         new VectorConstantCoefficient(Vector({1.0, 0.0, 0.0})), domain);
      solver.Setup(dt);

      real_t t = 0.0;
      solver.Step(t, dt, 0);

      VectorConstantCoefficient u_ex(Vector({t, 0_r, 0_r}));
      REQUIRE(t == MFEM_Approx(dt));
      REQUIRE(solver.GetCurrentVelocity()->ComputeL2Error(u_ex) < 1e-7);
      REQUIRE(solver.GetCurrentPressure()->ComputeL2Error(zero) < 1e-5);
   }

   SECTION("Inhomogeneous normal component")
   {
      VectorConstantCoefficient u_ex(Vector({1.0, 0.0, 0.0}));
      solver.GetCurrentVelocity()->ProjectCoefficient(u_ex);

      solver.AddVelDirichletBC(new ConstantCoefficient(1.0), x_faces, 0);
      solver.AddVelDirichletBC(new ConstantCoefficient(0.0), y_faces, 1);
      solver.AddVelDirichletBC(new ConstantCoefficient(0.0), z_faces, 2);
      solver.Setup(dt);

      real_t t = 0.0;
      solver.Step(t, dt, 0);

      REQUIRE(solver.GetCurrentVelocity()->ComputeL2Error(u_ex) < 1e-7);
      REQUIRE(solver.GetCurrentPressure()->ComputeL2Error(zero) < 1e-5);
   }

   SECTION("Time-dependent normal component")
   {
      auto ramp = [](const Vector &, real_t t) { return t; };
      solver.AddVelDirichletBC(new FunctionCoefficient(ramp), x_faces, 0);
      solver.AddVelDirichletBC(new ConstantCoefficient(0.0), y_faces, 1);
      solver.AddVelDirichletBC(new ConstantCoefficient(0.0), z_faces, 2);
      solver.AddAccelTerm(
         new VectorConstantCoefficient(Vector({1.0, 0.0, 0.0})), domain);
      solver.Setup(dt);

      real_t t = 0.0;
      for (int step = 0; step < 3; ++step)
      {
         solver.Step(t, dt, step);

         VectorConstantCoefficient u_ex(Vector({t, 0_r, 0_r}));
         REQUIRE(t == MFEM_Approx((step + 1) * dt));
         REQUIRE(solver.GetCurrentVelocity()->ComputeL2Error(u_ex) < 1e-7);
         REQUIRE(solver.GetCurrentPressure()->ComputeL2Error(zero) < 1e-5);
      }
   }
}

TEST_CASE("Component boundary projection", "[Parallel][Navier]")
{
   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                     1.0, 1.0, 1.0);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   H1_FECollection fec(2, 3);
   ParFiniteElementSpace fes(&pmesh, &fec, 3);

   ParGridFunction x_gf(&fes), u_gf(&fes);
   pmesh.GetNodes(x_gf);

   VectorConstantCoefficient u_init(Vector({2.0, 3.0, 4.0}));
   u_gf.ProjectCoefficient(u_init);

   auto y_faces = Marker(pmesh, {2, 4});
   auto top = Marker(pmesh, {6});
   ConstantCoefficient c_y(7.0), c_z(9.0);

   Coefficient *comps[3] = {nullptr, &c_y, nullptr};
   u_gf.ProjectBdrCoefficient(comps, y_faces);

   comps[1] = nullptr;
   comps[2] = &c_z;
   u_gf.ProjectBdrCoefficient(comps, top);

   real_t loc_err = 0.0;
   for (int i = 0; i < fes.GetNDofs(); ++i)
   {
      const real_t y = x_gf(fes.DofToVDof(i, 1));
      const real_t z = x_gf(fes.DofToVDof(i, 2));
      const bool on_y = (std::abs(y) < 1e-12 || std::abs(y - 1.0) < 1e-12);
      const bool on_top = (std::abs(z - 1.0) < 1e-12);

      const real_t ux = u_gf(fes.DofToVDof(i, 0));
      const real_t uy = u_gf(fes.DofToVDof(i, 1));
      const real_t uz = u_gf(fes.DofToVDof(i, 2));

      loc_err = std::max(loc_err, std::abs(ux - 2_r));
      loc_err = std::max(loc_err, std::abs(uy - (on_y ? 7_r : 3_r)));
      loc_err = std::max(loc_err, std::abs(uz - (on_top ? 9_r : 4_r)));
   }

   real_t glob_err = 0.0;
   MPI_Allreduce(&loc_err, &glob_err, 1, MFEM_MPI_REAL_T, MPI_MAX,
                 MPI_COMM_WORLD);
   REQUIRE(glob_err < 1e-12);
}
