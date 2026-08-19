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

#include "unit_tests.hpp"
#include "mfem.hpp"
#include "../mesh/mesh_test_utils.hpp"

#include <set>
#include <unordered_set>
#include <vector>

using namespace mfem;

#ifdef MFEM_USE_MPI

TEST_CASE("BoundaryEdgeDOFsPartitionInvariant",
          "[Parallel][ParMesh][BoundaryEdgeDOFs]")
{
   constexpr int orientation = 3;
   constexpr int order = 1;

   // Use all available MPI processes for partitioning
   const int test_num_procs = Mpi::WorldSize();

   // Create base mesh
   Mesh base_mesh = OrientedTriFaceMesh(orientation, true);
   base_mesh.UniformRefinement();
   const int n_elements = base_mesh.GetNE();

   // Use a small set of representative partitionings
   std::vector<std::vector<int>> all_partitionings;
   // 1. All elements on rank 0
   all_partitionings.push_back(std::vector<int>(n_elements, 0));
   if (test_num_procs > 1)
   {
      // 2. Block partition: first half on rank 0, second half on last rank
      std::vector<int> &block = all_partitionings.emplace_back(n_elements);
      for (int i = 0; i < n_elements; i++)
      {
         block[i] = (i < n_elements/2) ? 0 : test_num_procs-1;
      }

      // 3. Round-robin partition: elements assigned cyclically to all ranks
      std::vector<int> &round_robin = all_partitionings.emplace_back(n_elements);
      for (int i = 0; i < n_elements; i++)
      {
         round_robin[i] = i % test_num_procs;
      }
   }

   // Create reusable FEC
   ND_FECollection fec(order, 3);

   std::vector<int> all_results;
   all_results.reserve(all_partitionings.size());

   // Test each partitioning
   for (const auto& partition : all_partitionings)
   {
      // Create parallel mesh with current partitioning
      Mesh test_mesh = OrientedTriFaceMesh(orientation, true);
      test_mesh.UniformRefinement();
      // For single process, use default partitioning; for multiple, use custom partition
      ParMesh pmesh = (test_num_procs == 1) ?
                      ParMesh(MPI_COMM_WORLD, test_mesh) :
                      ParMesh(MPI_COMM_WORLD, test_mesh, partition.data());

      // Create finite element space
      ParFiniteElementSpace fespace(&pmesh, &fec);

      // Extract boundary edge DOFs
      Array<int> ess_tdof_list;
      Array<int> boundary_edge_ldofs;
      std::vector<Array<int>> attr_to_elements;

      // Select the shared face to be the tested boundary
      int bdr_attr = pmesh.bdr_attributes.Max();
      Array<int> bdr_attrs(1);
      bdr_attrs[0] = bdr_attr;

      fespace.GetBoundaryElementsByAttribute(bdr_attrs, attr_to_elements);
      Array<int> boundary_elements = attr_to_elements[0];

      Array<int> dof_edges, dof_boundary_elements, ess_edge_list;

      fespace.GetBoundaryLoopEdgeDofs(boundary_elements, ess_tdof_list,
                                      boundary_edge_ldofs, nullptr, &dof_edges,
                                      &dof_boundary_elements, &ess_edge_list);

      // Collect total boundary edge DOFs
      int local_dofs = boundary_edge_ldofs.Size();
      int total_dofs;
      MPI_Allreduce(&local_dofs, &total_dofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      all_results.push_back(total_dofs);
   }

   // The set of boundary edge DOFs is a property of the mesh geometry and must
   // not depend on how the elements are distributed across ranks. Each result
   // is the global count of selected boundary edge DOFs for one partitioning, so
   // if the method correctly removes the artificial edges introduced at
   // processor boundaries, every partitioning yields the same total. A mismatch
   // means some partition kept or dropped a DOF that another did not.
   REQUIRE(!all_results.empty());
   // One refinement splits the triangular face into four sub-triangles. Its
   // perimeter has six loop edges (order-1 ND: one DOF per edge); the three
   // interior edges of the middle sub-triangle are shared and correctly dropped.
   constexpr int expected = 6;
   for (int result : all_results)
   {
      REQUIRE(result == expected);
   }
}

TEST_CASE("BoundaryEdgeDOFsBasicFunctionality",
          "[Parallel][ParMesh][BoundaryEdgeDOFs]")
{
   const int orientation = GENERATE(1, 3, 5);
   const int order = GENERATE(1, 2);

   CAPTURE(orientation, order);

   // Create test mesh
   Mesh mesh = OrientedTriFaceMesh(orientation, true);
   mesh.UniformRefinement();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   // Create finite element space
   ND_FECollection fec(order, 3);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   // Test boundary edge DOF extraction
   Array<int> ess_tdof_list;
   Array<int> boundary_edge_ldofs;
   Array<int> ldof_marker;
   std::vector<Array<int>> attr_to_elements;

   // Get boundary elements for the shared face
   int bdr_attr = pmesh.bdr_attributes.Max();
   Array<int> bdr_attrs(1);
   bdr_attrs[0] = bdr_attr;

   fespace.GetBoundaryElementsByAttribute(bdr_attrs, attr_to_elements);
   Array<int> boundary_elements = attr_to_elements[0];

   Array<int> dof_edges, dof_boundary_elements, ess_edge_list;

   fespace.GetBoundaryLoopEdgeDofs(boundary_elements, ess_tdof_list,
                                   boundary_edge_ldofs, &ldof_marker, &dof_edges,
                                   &dof_boundary_elements, &ess_edge_list);

   // Basic validation
   REQUIRE(ldof_marker.Size() == fespace.GetVSize());
   REQUIRE(ess_tdof_list.Size() >= 0);

   // The output arrays share a single indexing, so they must have equal size.
   REQUIRE(boundary_edge_ldofs.Size() == dof_edges.Size());
   REQUIRE(dof_edges.Size() == dof_boundary_elements.Size());

   // Verify all boundary edge DOFs are marked in ldof_marker
   for (int dof : boundary_edge_ldofs)
   {
      REQUIRE(ldof_marker[dof] == 1);
   }
}

// Helper function to compute boundary loop length
real_t ComputeBoundaryLoopLength(ParMesh* pmesh, const Array<int>& dof_edges)
{
   real_t local_length = 0.0;
   std::unordered_set<int> processed_edges;

   for (int i = 0; i < dof_edges.Size(); i++)
   {
      int edge_id = dof_edges[i];
      if (!processed_edges.insert(edge_id).second) { continue; }

      Array<int> edge_verts;
      pmesh->GetEdgeVertices(edge_id, edge_verts);
      const real_t* v0 = pmesh->GetVertex(edge_verts[0]);
      const real_t* v1 = pmesh->GetVertex(edge_verts[1]);

      real_t edge_length = 0.0;
      for (int d = 0; d < pmesh->SpaceDimension(); d++)
      {
         real_t diff = v1[d] - v0[d];
         edge_length += diff * diff;
      }
      local_length += sqrt(edge_length);
   }
   return local_length;
}

TEST_CASE("BoundaryEdgeDOFsNestedCubes",
          "[Parallel][ParMesh][BoundaryEdgeDOFs]")
{
   const int order = GENERATE(1, 2);

   // Expected processor-invariant results for nested cubes mesh (1 refinement)
   // order=1: 16 tdofs, sum=16.0, length=2.0
   // order=2: 32 tdofs, sum=32.0, length=2.0
   int exp_tdofs = (order == 1) ? 16 : 32;
   real_t exp_sum = (order == 1) ? real_t(16.0) : real_t(32.0);
   real_t exp_length = real_t(2.0);

   struct BoundaryTest
   {
      int attr_value;
      Vector normal;
      std::string name;
   };

   std::vector<BoundaryTest> boundary_tests =
   {
      {7, Vector({0, 0, -1}), "-z"},
      {8, Vector({0, 0, 1}), "+z"},
      {9, Vector({0, -1, 0}), "-y"},
      {10, Vector({1, 0, 0}), "+x"},
      {11, Vector({0, 1, 0}), "+y"},
      {12, Vector({-1, 0, 0}), "-x"}
   };

   const char* mesh_file = "../../data/nested_cubes.msh";
   Mesh mesh(mesh_file, 1, 1);
   mesh.UniformRefinement();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   ND_FECollection fec(order, 3);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   for (const auto& test : boundary_tests)
   {
      CAPTURE(test.name, test.attr_value, order, num_procs);

      std::vector<Array<int>> attr_to_elements;
      Array<int> bdr_attrs(1);
      bdr_attrs[0] = test.attr_value;

      fespace.GetBoundaryElementsByAttribute(bdr_attrs, attr_to_elements);
      Array<int> boundary_elements = attr_to_elements[0];

      Array<int> ess_tdof_list;
      Array<int> ldof_marker;
      Array<int> boundary_edge_ldofs;
      Array<int> dof_edges, dof_boundary_elements, ess_edge_list;

      fespace.GetBoundaryLoopEdgeDofs(boundary_elements, ess_tdof_list,
                                      boundary_edge_ldofs, &ldof_marker, &dof_edges,
                                      &dof_boundary_elements, &ess_edge_list);

      Array<int> dof_orientations;
      fespace.ComputeLoopEdgeOrientations(dof_edges, dof_boundary_elements,
                                          test.normal, dof_orientations);

      ParGridFunction x(&fespace);
      x = real_t(0.0);
      for (int i = 0; i < boundary_edge_ldofs.Size(); i++)
      {
         x(boundary_edge_ldofs[i]) = real_t(1.0) * dof_orientations[i];
      }

      GroupCommunicator *gc = fespace.ScalarGroupComm();
      Array<int> global_marker(ldof_marker);
      gc->Reduce<int>(global_marker.GetData(), GroupCommunicator::BitOR<int>);
      gc->Bcast(global_marker);

      Array<real_t> values(x.GetData(), x.Size());
      gc->ReduceBegin(values.GetData());
      gc->ReduceMarked<real_t>(values.GetData(), global_marker, 0,
                               GroupCommunicator::MaxAbs<real_t>);
      gc->Bcast(values.GetData());
      delete gc;

      Vector x_true;
      x.GetTrueDofs(x_true);

      int local_nonzero_tdofs = 0;
      real_t local_tdof_sum = 0.0;
      for (int tdof = 0; tdof < x_true.Size(); tdof++)
      {
         real_t tdof_value = x_true(tdof);
         if (abs(tdof_value) > 1e-12)
         {
            local_nonzero_tdofs++;
            local_tdof_sum += abs(tdof_value);
         }
      }

      real_t local_length = ComputeBoundaryLoopLength(&pmesh, dof_edges);

      int global_nonzero_tdofs;
      real_t global_tdof_sum, total_length;
      MPI_Allreduce(&local_nonzero_tdofs, &global_nonzero_tdofs, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(&local_tdof_sum, &global_tdof_sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(&local_length, &total_length, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      // Verify processor-invariant results match expected values
      REQUIRE(global_nonzero_tdofs == exp_tdofs);
      REQUIRE(abs(global_tdof_sum - exp_sum) < real_t(1e-12));
      REQUIRE(abs(total_length - exp_length) < real_t(1e-12));
   }
}

TEST_CASE("BoundaryEdgeDOFs2DSquareInSquare",
          "[Parallel][ParMesh][BoundaryEdgeDOFs]")
{
   // Test 2D boundary edge DOF extraction using square-in-square mesh
   constexpr int order = 2;

   // Test multiple inner boundary attributes
   std::vector<int> inner_attrs_to_test = {5, 6, 7, 8};

   // Load 2D square-in-square mesh from file
   const char* mesh_file = "../../data/square_in_square.msh";
   Mesh serial_mesh(mesh_file, 1, 1);
   serial_mesh.UniformRefinement();

   int num_procs = Mpi::WorldSize();

   // Test each boundary attribute
   for (int inner_attr : inner_attrs_to_test)
   {
      CAPTURE(inner_attr); // Capture the attribute being tested for better test output

      // Test that results are consistent across different mesh partitionings
      const int n_elements = serial_mesh.GetNE();

      // Generate multiple different partitionings
      std::vector<std::vector<int>> all_partitionings;

      // 1. All elements on rank 0
      all_partitionings.push_back(std::vector<int>(n_elements, 0));

      if (num_procs > 1)
      {
         // 2. Block partition: first half on rank 0, second half on last rank
         std::vector<int> block(n_elements);
         for (int i = 0; i < n_elements; i++)
         {
            block[i] = (i < n_elements/2) ? 0 : num_procs-1;
         }
         all_partitionings.push_back(block);

         // 3. Round-robin partition: elements assigned cyclically to all ranks
         std::vector<int> round_robin(n_elements);
         for (int i = 0; i < n_elements; i++)
         {
            round_robin[i] = i % num_procs;
         }
         all_partitionings.push_back(round_robin);
      }

      ND_FECollection fec(order, 2);
      std::vector<int> all_dof_results;
      all_dof_results.reserve(all_partitionings.size());

      // Test each partitioning
      for (const auto& partition : all_partitionings)
      {
         // Create parallel mesh with current partitioning
         Mesh test_mesh(mesh_file, 1, 1);
         test_mesh.UniformRefinement();
         ParMesh pmesh = (num_procs == 1) ?
                         ParMesh(MPI_COMM_WORLD, test_mesh) :
                         ParMesh(MPI_COMM_WORLD, test_mesh, partition.data());

         ParFiniteElementSpace fespace(&pmesh, &fec);

         // Find boundary elements with the inner attribute
         std::vector<Array<int>> attr_to_elements;
         Array<int> inner_attrs(1);
         inner_attrs[0] = inner_attr;
         fespace.GetBoundaryElementsByAttribute(inner_attrs, attr_to_elements);

         Array<int> inner_boundary_elements = attr_to_elements[0];

         Array<int> ess_tdofs, ess_edges;
         Array<int> boundary_dofs;
         Array<int> dof_edges, dof_boundary_elements;

         fespace.GetBoundaryLoopEdgeDofs(inner_boundary_elements, ess_tdofs,
                                         boundary_dofs, nullptr, &dof_edges,
                                         &dof_boundary_elements, &ess_edges);

         // The output arrays share one indexing, so their sizes must match.
         REQUIRE(boundary_dofs.Size() == dof_edges.Size());
         REQUIRE(dof_edges.Size() == dof_boundary_elements.Size());

         // Gather global counts for this partitioning
         int local_dof_count = boundary_dofs.Size();
         int global_dof_count;
         MPI_Allreduce(&local_dof_count, &global_dof_count, 1, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD);

         all_dof_results.push_back(global_dof_count);
      }

      // Verify all partitionings give identical results
      REQUIRE(!all_dof_results.empty());

      int expected_dofs = all_dof_results[0];
      for (int result : all_dof_results)
      {
         REQUIRE(result == expected_dofs);
      }
   } // End of inner_attr loop
}

TEST_CASE("BoundaryEdgeDOFsSharedDOFsAreOwnedBySomeRank",
          "[Parallel][ParMesh][BoundaryEdgeDOFs]")
{
   // Every selected shared DOF must appear in exactly one rank's ess_tdof_list.
   // Only the group master owns the corresponding true DOF and returns a
   // non-negative value from GetLocalTDofNumber(), so if the master holds none
   // of the selected boundary elements the DOF would be emitted by no rank at
   // all unless the local marker is synchronized across the sharing group.
   const int nranks = Mpi::WorldSize();
   if (nranks < 2) { return; }

   constexpr int order = 1;
   ND_FECollection fec(order, 3);

   for (int orientation : {1, 3, 5})
   {
      Mesh probe = OrientedTriFaceMesh(orientation, true);
      probe.UniformRefinement();
      const int ne = probe.GetNE();

      // Several partitionings, to vary which rank masters each shared group
      std::vector<std::vector<int>> partitionings;
      {
         std::vector<int> round_robin(ne), block(ne), strided(ne);
         for (int i = 0; i < ne; i++)
         {
            round_robin[i] = i % nranks;
            block[i] = (i < ne/2) ? 0 : nranks-1;
            strided[i] = (i * 7 + 3) % nranks;
         }
         partitionings = {round_robin, block, strided};
      }

      for (const auto &partition : partitionings)
      {
         Mesh mesh = OrientedTriFaceMesh(orientation, true);
         mesh.UniformRefinement();
         ParMesh pmesh(MPI_COMM_WORLD, mesh, partition.data());
         ParFiniteElementSpace fes(&pmesh, &fec);

         const int bdr_attr = pmesh.bdr_attributes.Max();
         Array<int> bdr_attrs(1);
         bdr_attrs[0] = bdr_attr;
         std::vector<Array<int>> attr_to_elements;
         fes.GetBoundaryElementsByAttribute(bdr_attrs, attr_to_elements);
         Array<int> bdr_elements = attr_to_elements[0];

         Array<int> ess_tdofs;
         Array<int> boundary_dofs;
         fes.GetBoundaryLoopEdgeDofs(bdr_elements, ess_tdofs, boundary_dofs);

         // Identify DOFs by global true DOF number, which is agreed upon by all
         // ranks sharing the DOF, then compare the set selected anywhere with
         // the set actually emitted in ess_tdof_list.
         std::set<HYPRE_BigInt> selected, emitted;
         for (int dof : boundary_dofs)
         {
            selected.insert(fes.GetGlobalTDofNumber(dof));
         }
         for (int i = 0; i < ess_tdofs.Size(); i++)
         {
            emitted.insert(fes.GetMyTDofOffset() + ess_tdofs[i]);
         }

         auto all_gather = [nranks](const std::set<HYPRE_BigInt> &s)
         {
            std::vector<HYPRE_BigInt> local(s.begin(), s.end());
            int n = static_cast<int>(local.size()), total = 0;
            std::vector<int> counts(nranks), bytes(nranks), displs(nranks);
            MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT,
                          MPI_COMM_WORLD);
            constexpr int sz = sizeof(HYPRE_BigInt);
            for (int r = 0; r < nranks; r++)
            {
               displs[r] = total * sz;
               total += counts[r];
               bytes[r] = counts[r] * sz;
            }
            std::vector<HYPRE_BigInt> all(total);
            MPI_Allgatherv(local.data(), n * sz, MPI_BYTE, all.data(),
                           bytes.data(), displs.data(), MPI_BYTE,
                           MPI_COMM_WORLD);
            return std::set<HYPRE_BigInt>(all.begin(), all.end());
         };

         // Gather both sets across all ranks. global_selected is every shared
         // boundary DOF chosen on any rank; global_emitted is every true DOF
         // actually placed in some rank's ess_tdof_list. A selected DOF missing
         // from global_emitted is one that no rank owns and outputs, which is
         // exactly the synchronization bug this test guards against.
         const std::set<HYPRE_BigInt> global_selected = all_gather(selected);
         const std::set<HYPRE_BigInt> global_emitted = all_gather(emitted);

         int num_missing = 0;
         for (auto gtdof : global_selected)
         {
            if (!global_emitted.count(gtdof)) { num_missing++; }
         }

         CAPTURE(orientation, nranks, global_selected.size(),
                 global_emitted.size(), num_missing);
         REQUIRE(num_missing == 0);
      }
   }
}

TEST_CASE("BoundaryEdgeDOFs2DLoopVertexDOFsPartitionInvariant",
          "[Parallel][ParMesh][BoundaryEdgeDOFs]")
{
   // A closed boundary loop split between ranks must give the same result as
   // the serial code. With a collection carrying vertex DOFs (ND_R2D), a vertex
   // shared by two boundary segments is interior to the loop and must be
   // dropped. When the two segments live on different ranks, each rank sees the
   // vertex only once locally, so the occurrence parity has to be reconciled
   // across the sharing group.
   if (Mpi::WorldSize() < 2) { return; }

   constexpr int order = 1;
   ND_R2D_FECollection fec(order, 2);

   // Serial reference result
   Mesh serial_mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                            1.0, 1.0);
   FiniteElementSpace serial_fes(&serial_mesh, &fec);

   Array<int> serial_bdr_elements(serial_mesh.GetNBE());
   for (int i = 0; i < serial_bdr_elements.Size(); i++)
   {
      serial_bdr_elements[i] = i;
   }

   Array<int> serial_boundary_dofs;

   serial_fes.GetBoundaryLoopEdgeDofs(serial_bdr_elements, serial_boundary_dofs);

   const int serial_count = serial_boundary_dofs.Size();

   // Compare against several partitionings of the same mesh
   const int num_procs = Mpi::WorldSize();
   std::vector<std::vector<int>> partitionings;
   {
      Mesh probe = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                         1.0, 1.0);
      const int ne = probe.GetNE();

      std::vector<int> block(ne), round_robin(ne);
      for (int i = 0; i < ne; i++)
      {
         block[i] = (i < ne/2) ? 0 : num_procs-1;
         round_robin[i] = i % num_procs;
      }
      partitionings.push_back(block);
      partitionings.push_back(round_robin);
   }

   for (const auto &partition : partitionings)
   {
      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                        1.0, 1.0);
      ParMesh pmesh(MPI_COMM_WORLD, mesh, partition.data());
      ParFiniteElementSpace pfes(&pmesh, &fec);

      Array<int> local_bdr_elements(pmesh.GetNBE());
      for (int i = 0; i < local_bdr_elements.Size(); i++)
      {
         local_bdr_elements[i] = i;
      }

      Array<int> ess_tdofs;
      Array<int> local_boundary_dofs;
      pfes.GetBoundaryLoopEdgeDofs(local_bdr_elements, ess_tdofs,
                                   local_boundary_dofs);

      // The true DOFs are owned by exactly one rank each, so summing the local
      // counts gives a partition-independent global count.
      int local_tdofs = ess_tdofs.Size();
      int global_tdofs = 0;
      MPI_Allreduce(&local_tdofs, &global_tdofs, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);

      CAPTURE(num_procs, serial_count, global_tdofs);
      REQUIRE(global_tdofs == serial_count);
   }
}

TEST_CASE("GroupCommunicatorReduceMarkedByGroupStride",
          "[Parallel][GroupCommunicator]")
{
   // Regression test for the neighbor-major stride of the byGroup receive
   // buffer: with more than one DOF in a group, the contributions to DOF i are
   // at buf[j*nldofs + i], so reducing a single marked DOF must gather the
   // strided values rather than reading a contiguous run.
   const int rank = Mpi::WorldRank();
   const int nranks = Mpi::WorldSize();
   if (nranks < 3) { return; }

   ListOfIntegerSets groups;

   IntegerSet local_group(1);
   local_group[0] = rank;
   groups.Insert(local_group);

   IntegerSet shared_group(nranks);
   for (int r = 0; r < nranks; r++)
   {
      shared_group[r] = r;
   }
   groups.Insert(shared_group);

   GroupTopology topology(MPI_COMM_WORLD);
   topology.Create(groups, 4983);

   GroupCommunicator comm(topology, GroupCommunicator::byGroup);
   // Two DOFs in the same shared group, so the buffer stride is 2.
   Array<int> ldof_group(2);
   ldof_group = 1;
   comm.Create(ldof_group);

   Array<real_t> values(2);
   values[0] = real_t(10.0) * rank + real_t(1.0);
   values[1] = real_t(100.0) * rank + real_t(2.0);

   Array<int> marker(2);
   marker = 1;

   comm.ReduceBegin(values.GetData());
   comm.ReduceMarked<real_t>(values.GetData(), marker, 0,
                             GroupCommunicator::Sum<real_t>);
   comm.Bcast(values);

   const real_t rank_sum = real_t(nranks) * real_t(nranks - 1) / real_t(2.0);
   REQUIRE(values[0] == MFEM_Approx(real_t(10.0) * rank_sum + real_t(nranks)));
   REQUIRE(values[1] == MFEM_Approx(real_t(100.0) * rank_sum +
                                    real_t(2.0) * real_t(nranks)));
}

#endif // MFEM_USE_MPI
