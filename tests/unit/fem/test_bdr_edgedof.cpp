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

using namespace mfem;

#ifdef MFEM_USE_MPI

namespace boundary_edge_dof_test
{

// Generate all possible partitionings of n_elements into num_procs
void GeneratePartitionings(int n_elements, int num_procs,
                           std::vector<std::vector<int>>& all_partitionings)
{
   std::vector<int> partition(n_elements);

   auto generate = [&](auto&& self, int elem) -> void
   {
      if (elem == n_elements)
      {
         all_partitionings.push_back(partition);
         return;
      }

      for (int proc = 0; proc < num_procs; proc++)
      {
         partition[elem] = proc;
         self(self, elem + 1);
      }
   };

   generate(generate, 0);
}

} // namespace boundary_edge_dof_test

TEST_CASE("BoundaryEdgeDoFsPartitionInvariant",
          "[Parallel][ParMesh][BoundaryEdgeDoFs]")
{
   const int orientation = 3;
   const int order = 1;

   // Adapt to actual number of MPI processes (1 or 2)
   const int available_procs = Mpi::WorldSize();
   const int test_num_procs = std::min(available_procs, 2);

   // Create base mesh
   Mesh base_mesh = OrientedTriFaceMesh(orientation, true);
   base_mesh.UniformRefinement();
   const int n_elements = base_mesh.GetNE();

   // Generate all partitionings for available processes
   std::vector<std::vector<int>> all_partitionings;
   boundary_edge_dof_test::GeneratePartitionings(n_elements, test_num_procs,
                                                 all_partitionings);

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

      // Extract boundary edge DoFs
      Array<int> ess_tdof_list;
      std::unordered_set<int> boundary_edge_ldofs;
      Array<int> ldof_marker;
      std::unordered_map<int, Array<int>> attr_to_elements;

      // Select the shared face to be the tested boundary
      int bdr_attr = pmesh.bdr_attributes.Max();
      Array<int> bdr_attrs(1);
      bdr_attrs[0] = bdr_attr;

      fespace.GetBoundaryElementsByAttribute(bdr_attrs, attr_to_elements);
      Array<int> boundary_elements = attr_to_elements[bdr_attr];

      std::unordered_map<int, int> dof_to_edge, dof_to_orientation;
      std::unordered_map<int, int> dof_to_boundary_element;
      Array<int> ess_edge_list;

      fespace.GetBoundaryEdgeDofs(boundary_elements, ess_tdof_list, ldof_marker,
                                  boundary_edge_ldofs, &dof_to_edge, &dof_to_orientation,
                                  &dof_to_boundary_element, &ess_edge_list);

      // Collect total boundary edge DoFs
      int local_dofs = boundary_edge_ldofs.size();
      int total_dofs;
      MPI_Allreduce(&local_dofs, &total_dofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      all_results.push_back(total_dofs);
   }

   // Verify all results are identical
   REQUIRE(!all_results.empty());
   int expected = all_results[0];
   for (int result : all_results)
   {
      REQUIRE(result == expected);
   }
}

TEST_CASE("BoundaryEdgeDoFsBasicFunctionality",
          "[Parallel][ParMesh][BoundaryEdgeDoFs]")
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
   std::unordered_set<int> boundary_edge_ldofs;
   Array<int> ldof_marker;
   std::unordered_map<int, Array<int>> attr_to_elements;

   // Get boundary elements for the shared face
   int bdr_attr = pmesh.bdr_attributes.Max();
   Array<int> bdr_attrs(1);
   bdr_attrs[0] = bdr_attr;

   fespace.GetBoundaryElementsByAttribute(bdr_attrs, attr_to_elements);
   Array<int> boundary_elements = attr_to_elements[bdr_attr];

   std::unordered_map<int, int> dof_to_edge, dof_to_orientation;
   std::unordered_map<int, int> dof_to_boundary_element;
   Array<int> ess_edge_list;

   fespace.GetBoundaryEdgeDofs(boundary_elements, ess_tdof_list, ldof_marker,
                               boundary_edge_ldofs, &dof_to_edge, &dof_to_orientation,
                               &dof_to_boundary_element, &ess_edge_list);

   // Basic validation
   REQUIRE(boundary_edge_ldofs.size() >= 0);
   REQUIRE(ldof_marker.Size() == fespace.GetVSize());
   REQUIRE(ess_tdof_list.Size() >= 0);

   // Verify consistency between different outputs
   REQUIRE(boundary_edge_ldofs.size() == dof_to_edge.size());
   REQUIRE(dof_to_edge.size() == dof_to_orientation.size());
   REQUIRE(dof_to_edge.size() == dof_to_boundary_element.size());

   // Verify all boundary edge DOFs are marked in ldof_marker
   for (int dof : boundary_edge_ldofs)
   {
      REQUIRE(ldof_marker[dof] == 1);
   }
}

// Helper function to compute boundary loop length
double ComputeBoundaryLoopLength(ParMesh* pmesh,
                                 const std::unordered_map<int, int>& dof_to_edge)
{
   double local_length = 0.0;
   std::unordered_set<int> processed_edges;

   for (const auto& pair : dof_to_edge)
   {
      int edge_id = pair.second;
      if (!processed_edges.insert(edge_id).second) { continue; }

      Array<int> edge_verts;
      pmesh->GetEdgeVertices(edge_id, edge_verts);
      const double* v0 = pmesh->GetVertex(edge_verts[0]);
      const double* v1 = pmesh->GetVertex(edge_verts[1]);

      double edge_length = 0.0;
      for (int i = 0; i < pmesh->SpaceDimension(); i++)
      {
         double diff = v1[i] - v0[i];
         edge_length += diff * diff;
      }
      local_length += sqrt(edge_length);
   }
   return local_length;
}

TEST_CASE("BoundaryEdgeDoFsNestedCubes",
          "[Parallel][ParMesh][BoundaryEdgeDoFs]")
{
   const int order = GENERATE(1, 2);

   // Expected processor-invariant results for nested cubes mesh (1 refinement)
   // order=1: 24 tdofs, sum=24.0, length=2.0
   // order=2: 48 tdofs, sum=48.0, length=2.0
   int exp_tdofs = (order == 1) ? 24 : 48;
   double exp_sum = (order == 1) ? 24.0 : 48.0;
   double exp_length = 2.0;

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

      std::unordered_map<int, Array<int>> attr_to_elements;
      Array<int> bdr_attrs(1);
      bdr_attrs[0] = test.attr_value;

      fespace.GetBoundaryElementsByAttribute(bdr_attrs, attr_to_elements);
      if (attr_to_elements.find(test.attr_value) == attr_to_elements.end())
      {
         continue;
      }

      Array<int> boundary_elements = attr_to_elements[test.attr_value];

      Array<int> ess_tdof_list;
      Array<int> ldof_marker;
      std::unordered_set<int> boundary_edge_ldofs;
      std::unordered_map<int, int> dof_to_edge, dof_to_orientation;
      std::unordered_map<int, int> dof_to_boundary_element;
      Array<int> ess_edge_list;

      fespace.GetBoundaryEdgeDofs(boundary_elements, ess_tdof_list, ldof_marker,
                                  boundary_edge_ldofs, &dof_to_edge, &dof_to_orientation,
                                  &dof_to_boundary_element, &ess_edge_list);

      std::unordered_map<int, int> edge_loop_orientation;
      fespace.ComputeLoopEdgeOrientations(dof_to_edge, dof_to_boundary_element,
                                          test.normal, edge_loop_orientation);

      ParGridFunction x(&fespace);
      x = 0.0;
      for (int dof : boundary_edge_ldofs)
      {
         int edge = dof_to_edge[dof];
         int orientation = edge_loop_orientation[edge];
         x(dof) = 1.0 * orientation;
      }

      GroupCommunicator *gc = fespace.ScalarGroupComm();
      Array<int> global_marker(ldof_marker);
      gc->Reduce<int>(global_marker.GetData(), GroupCommunicator::BitOR<int>);
      gc->Bcast(global_marker);

      Array<double> values(x.GetData(), x.Size());
      gc->ReduceBegin(values.GetData());
      gc->ReduceMarked<double>(values.GetData(), global_marker, 0,
                               GroupCommunicator::MaxAbs<double>);
      gc->Bcast(values.GetData());
      delete gc;

      Vector x_true;
      x.GetTrueDofs(x_true);

      int local_nonzero_tdofs = 0;
      double local_tdof_sum = 0.0;
      for (int tdof = 0; tdof < x_true.Size(); tdof++)
      {
         double tdof_value = x_true(tdof);
         if (abs(tdof_value) > 1e-12)
         {
            local_nonzero_tdofs++;
            local_tdof_sum += abs(tdof_value);
         }
      }

      double local_length = ComputeBoundaryLoopLength(&pmesh, dof_to_edge);

      int global_nonzero_tdofs;
      double global_tdof_sum, total_length;
      MPI_Allreduce(&local_nonzero_tdofs, &global_nonzero_tdofs, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(&local_tdof_sum, &global_tdof_sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(&local_length, &total_length, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      // Verify processor-invariant results match expected values
      REQUIRE(global_nonzero_tdofs == exp_tdofs);
      REQUIRE(abs(global_tdof_sum - exp_sum) < 1e-12);
      REQUIRE(abs(total_length - exp_length) < 1e-12);
   }
}

#endif // MFEM_USE_MPI