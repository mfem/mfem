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

using namespace mfem;

#ifdef MFEM_USE_MPI

namespace
{

// Map each element to its geometric row in the serial mesh.
int GetElementRow(const Mesh &mesh, int ntasks, int e)
{
   Array<int> vertices;
   mesh.GetElementVertices(e, vertices);

   real_t y = 0.0;
   for (int i = 0; i < vertices.Size(); i++)
   {
      y += mesh.GetVertex(vertices[i])[1];
   }
   y /= vertices.Size();

   return std::min(ntasks - 1, int(y * ntasks));
}

// Assign one geometric mesh row to each MPI rank.
void MakeRowPartitioning(const Mesh &mesh, int ntasks, Array<int> &partitioning)
{
   partitioning.SetSize(mesh.GetNE());
   for (int i = 0; i < partitioning.Size(); i++)
   {
      partitioning[i] = GetElementRow(mesh, ntasks, i);
   }
}

// Shared values differ only on the lower and upper local vertex rows.
real_t ExpectedReductionValue(int rank, int ntasks, bool top,
                              DeviceSharedDofCommunicator::Op op)
{
   const bool first_rank = (rank == 0);
   const bool last_rank = (rank == ntasks - 1);

   if (op == DeviceSharedDofCommunicator::Op::Min)
   {
      if (top) { return rank; }
      return first_rank ? rank : rank - 1;
   }
   if (op == DeviceSharedDofCommunicator::Op::Max)
   {
      if (!top) { return rank; }
      return last_rank ? rank : rank + 1;
   }

   if (top) { return last_rank ? rank : 2 * rank + 1; }
   return first_rank ? rank : 2 * rank - 1;
}

// Max attribute wins in ProjectDiscCoefficient(coeff).
real_t ExpectedProjectedValue(int rank, int ntasks, bool top)
{
   if (!top) { return rank; }
   return (rank == ntasks - 1) ? rank : rank + 1;
}

}

TEST_CASE("DeviceSharedDofCommunicator", "[Parallel]")
{
   const int ntasks = Mpi::WorldSize();
   const int rank = Mpi::WorldRank();

   Mesh serial_mesh = Mesh::MakeCartesian2D(ntasks, ntasks,
                                            Element::QUADRILATERAL,
                                            1.0, 1.0, false);

   Array<int> partitioning;
   MakeRowPartitioning(serial_mesh, ntasks, partitioning);

   ParMesh par_mesh(MPI_COMM_WORLD, serial_mesh, partitioning);
   H1_FECollection fec(1, par_mesh.Dimension());
   ParFiniteElementSpace pfes(&par_mesh, &fec);

   REQUIRE(pfes.Conforming());

   const real_t y_bottom = real_t(rank) / ntasks;
   const real_t y_top = real_t(rank + 1) / ntasks;
   const real_t tol = 1.0e-12;

   auto check_reduction = [&](DeviceSharedDofCommunicator::Op op)
   {
      // Start with the row number on every local dof copy.
      Vector x(pfes.GetVSize());
      x = real_t(rank);

      const auto *dof_comm = pfes.GetDeviceSharedDofCommunicator();
      dof_comm->ReduceAndBcast(x, op);

      // Order 1 H1 scalar dofs coincide with mesh vertices.
      Array<int> vdofs;
      for (int i = 0; i < par_mesh.GetNV(); i++)
      {
         const real_t *v = par_mesh.GetVertex(i);
         const bool top = std::abs(v[1] - y_top) < tol;
         const bool bottom = std::abs(v[1] - y_bottom) < tol;

         REQUIRE((top || bottom));
         pfes.GetVertexVDofs(i, vdofs);
         REQUIRE(x(vdofs[0]) == MFEM_Approx(
                    ExpectedReductionValue(rank, ntasks, top, op)));
      }
   };

   SECTION("Min")
   {
      check_reduction(DeviceSharedDofCommunicator::Op::Min);
   }

   SECTION("Max")
   {
      check_reduction(DeviceSharedDofCommunicator::Op::Max);
   }

   SECTION("Sum")
   {
      check_reduction(DeviceSharedDofCommunicator::Op::Sum);
   }
}

TEST_CASE("ProjectDiscCoefficient Row Partition", "[Parallel]")
{
   const int ntasks = Mpi::WorldSize();
   const int rank = Mpi::WorldRank();

   Mesh serial_mesh = Mesh::MakeCartesian2D(ntasks, ntasks,
                                            Element::QUADRILATERAL,
                                            1.0, 1.0, false);

   Array<int> partitioning;
   MakeRowPartitioning(serial_mesh, ntasks, partitioning);

   ParMesh par_mesh(MPI_COMM_WORLD, serial_mesh, partitioning);
   H1_FECollection fec(1, par_mesh.Dimension());
   ParFiniteElementSpace pfes(&par_mesh, &fec);
   ParGridFunction gf(&pfes);

   ConstantCoefficient coeff(rank);
   gf.ProjectDiscCoefficient(coeff);

   const real_t y_bottom = real_t(rank) / ntasks;
   const real_t y_top = real_t(rank + 1) / ntasks;
   const real_t tol = 1.0e-12;

   // The upper row wins on shared top vertices because it has larger attribute.
   Array<int> vdofs;
   for (int i = 0; i < par_mesh.GetNV(); i++)
   {
      const real_t *v = par_mesh.GetVertex(i);
      const bool top = std::abs(v[1] - y_top) < tol;
      const bool bottom = std::abs(v[1] - y_bottom) < tol;

      REQUIRE((top || bottom));
      pfes.GetVertexVDofs(i, vdofs);
      REQUIRE(gf(vdofs[0]) == MFEM_Approx(
                 ExpectedProjectedValue(rank, ntasks, top)));
   }
}

#endif
